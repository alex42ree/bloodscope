"""Biomarker name matching against the translations database."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from core.data_paths import get_default_runtime_data_dir

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 85


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def _extract_term(variant) -> str:
    """Extract the term string from a variant (object or plain string)."""
    if isinstance(variant, dict):
        return variant["term"]
    return variant


def _build_lookup(translations: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Build ``{language: [(normalized_variant, biomarker_id), ...]}``."""
    lookup: dict[str, list[tuple[str, str]]] = {}
    for entry in translations:
        lang = entry["language"]
        bid = entry["biomarker_id"]
        if lang not in lookup:
            lookup[lang] = []
        for variant in entry["variants"]:
            term = _extract_term(variant)
            lookup[lang].append((normalize(term), bid))
    return lookup


class BiomarkerMatcher:
    """Stateful matcher that caches the translations lookup."""

    def __init__(self, translations_path: str | Path) -> None:
        path = Path(translations_path)
        if not path.exists():
            logger.warning("Translations file not found: %s — matcher will return None for all", path)
            self._lookup: dict[str, list[tuple[str, str]]] = {}
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._lookup = _build_lookup(data.get("translations", []))

    def match(
        self,
        original_name: str,
        language: str,
    ) -> Optional[str]:
        """Match an original biomarker name to a canonical biomarker_id.

        Tries exact normalized match first, then fuzzy match via rapidfuzz.

        Returns
        -------
        str or None
            The biomarker_id if matched, otherwise None.
        """
        normalized_name = normalize(original_name)
        lang_entries = self._lookup.get(language, [])

        for variant, bid in lang_entries:
            if variant == normalized_name:
                return bid

        all_entries = [
            entry for entries in self._lookup.values() for entry in entries
        ]
        for variant, bid in all_entries:
            if variant == normalized_name:
                return bid

        best_score = 0.0
        best_bid: Optional[str] = None
        for variant, bid in all_entries:
            score = fuzz.ratio(normalized_name, variant)
            if score > best_score:
                best_score = score
                best_bid = bid

        if best_score >= FUZZY_THRESHOLD and best_bid is not None:
            logger.info(
                "Fuzzy matched '%s' → '%s' (score=%.1f)",
                original_name, best_bid, best_score,
            )
            return best_bid

        logger.warning("No match for biomarker name: '%s' (lang=%s)", original_name, language)
        return None


def load_matcher(translations_path: str | Path | None = None) -> BiomarkerMatcher:
    """Convenience factory to create a BiomarkerMatcher."""
    path = (
        translations_path
        if translations_path is not None
        else (get_default_runtime_data_dir() / "translations.json")
    )
    return BiomarkerMatcher(path)
