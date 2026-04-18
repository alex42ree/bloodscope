"""LLM-assisted biomarker ID resolution when dictionary matching fails.

Environment:

- ``BIOMARKER_LLM_FALLBACK``: set to ``0`` to disable (e.g. in tests). Default ``1``:
  fallback runs when ``ANTHROPIC_API_KEY`` is set.
- ``BIOMARKER_FALLBACK_TOP_K``: shortlist size per name (default 25).
- ``CLAUDE_MODEL`` / ``ANTHROPIC_API_KEY``: same as ``llm_parser``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

SYSTEM_PROMPT = """\
You assign laboratory test names to canonical biomarker identifiers.
For each item you receive an original name (as on the report), the document language code, \
and a closed list of candidate biomarkers (id + English LOINC-style name).
Pick exactly one candidate id that best matches the original name, or null if none are appropriate.
You must not invent ids: every non-null biomarker_id MUST appear in that item's candidates list.
Respond ONLY with valid JSON (no markdown), exactly this shape:
{"results":[{"original_name":"<string>","biomarker_id":"<id>"|null},...]}
The results array must have the same length and order as the input items array.
"""


def shortlist_candidates(
    original_name: str,
    biomarkers_index: dict[str, dict[str, Any]],
    k: int,
) -> list[dict[str, str]]:
    """Top-K biomarkers by rapidfuzz WRatio against ``id`` + ``en_name``."""
    items = list(biomarkers_index.items())
    if not items:
        return []
    choices = [f"{bid} {data['en_name']}" for bid, data in items]
    extracted = process.extract(
        original_name,
        choices,
        scorer=fuzz.WRatio,
        limit=min(k, len(choices)),
    )
    out: list[dict[str, str]] = []
    for _, _, idx in extracted:
        bid, data = items[idx]
        out.append({"id": bid, "en_name": str(data.get("en_name", ""))})
    return out


def batch_resolve_biomarker_ids(
    original_names: list[str],
    language: str,
    biomarkers_index: dict[str, dict[str, Any]],
    *,
    api_key: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
) -> dict[str, str | None]:
    """Resolve each distinct original_name to a biomarker_id or None.

    Parameters
    ----------
    original_names :
        Names that failed dictionary matching (may contain duplicates).
    language :
        Report language (es, de, en, fr).
    biomarkers_index :
        Same structure as ``biomarkers.json`` index: id -> row dict.

    Returns
    -------
    dict
        Maps each **distinct** ``original_name`` to a validated ``biomarker_id`` or ``None``.
        Callers can map duplicate markers using the same key.
    """
    if not original_names or not biomarkers_index:
        return {}

    unique = list(dict.fromkeys(original_names))
    k = top_k if top_k is not None else int(os.getenv("BIOMARKER_FALLBACK_TOP_K", "25"))

    items_payload: list[dict[str, Any]] = []
    shortlist_by_name: dict[str, list[dict[str, str]]] = {}
    for name in unique:
        candidates = shortlist_candidates(name, biomarkers_index, k)
        shortlist_by_name[name] = candidates
        items_payload.append({
            "original_name": name,
            "candidates": candidates,
        })

    resolved_model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
    if not key.strip():
        logger.warning("LLM biomarker fallback skipped: ANTHROPIC_API_KEY not set")
        return {n: None for n in unique}

    client = anthropic.Anthropic(api_key=key)
    user_content = json.dumps(
        {"language": language, "items": items_payload},
        ensure_ascii=False,
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        logger.info("LLM biomarker batch attempt %d/%d (%d names)", attempt, MAX_RETRIES + 1, len(unique))

        message = client.messages.create(
            model=resolved_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("LLM biomarker batch: invalid JSON: %s", exc)
            continue

        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(items_payload):
            last_error = ValueError("results length mismatch")
            logger.warning("LLM biomarker batch: expected %d results, got %s", len(items_payload), type(results))
            continue

        out: dict[str, str | None] = {}

        for i, name in enumerate(unique):
            row = results[i] if i < len(results) else {}
            if not isinstance(row, dict):
                out[name] = None
                continue
            bid = row.get("biomarker_id")
            candidates = shortlist_by_name[name]
            allowed = {c["id"] for c in candidates}

            if bid is None or (isinstance(bid, str) and bid.strip() == ""):
                out[name] = None
                logger.info("LLM biomarker fallback: '%s' -> None", name)
                continue

            if not isinstance(bid, str):
                out[name] = None
                continue

            if bid not in allowed:
                logger.warning(
                    "LLM biomarker fallback rejected invalid id '%s' for '%s' (not in shortlist)",
                    bid,
                    name,
                )
                out[name] = None
                continue

            out[name] = bid
            logger.info("LLM biomarker fallback: '%s' -> %s", name, bid)

        return out

    logger.warning("LLM biomarker batch failed after retries: %s", last_error)
    return {n: None for n in unique}


def llm_fallback_enabled() -> bool:
    """True if batch LLM resolution should run (feature flag + API key)."""
    if os.getenv("BIOMARKER_LLM_FALLBACK", "1").strip() == "0":
        return False
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        return False
    return True
