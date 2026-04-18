"""Generate ranges.json (optimal ranges) from configured web sources via LLM extraction.

Each source URL is fetched, its content sent to Claude for structured extraction,
and the results are fuzzy-matched to biomarkers.json IDs.  Every entry records
provenance (source name, URL, extraction timestamp).

Usage
-----
::

    # Show which biomarkers have / lack optimal ranges
    python -m cli.init_ranges discover \\
        --biomarkers data/biomarkers.json \\
        --ranges data/ranges.json

    # Generate ranges.json from all configured sources
    python -m cli.init_ranges generate \\
        --biomarkers data/biomarkers.json \\
        --sources data_generation/range_sources.json \\
        --output data/ranges.json

    # Only process missing biomarkers
    python -m cli.init_ranges generate --merge ...

    # Process a single source
    python -m cli.init_ranges generate --source "FMU Blood Tracking Form" ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click
from rapidfuzz import fuzz, process

from core.data_paths import get_default_runtime_data_dir, get_generation_config_dir

_DEFAULT_RANGE_SOURCES = str(get_generation_config_dir() / "range_sources.json")

EXTRACTION_PROMPT = """\
You are a biomedical data extractor.  The following content comes from a health/medical \
website or document that lists optimal (functional medicine) reference ranges for blood \
biomarkers.

Extract every biomarker optimal range you can find.  Return ONLY valid JSON (no \
markdown, no commentary) matching this schema:

{
  "ranges": [
    {
      "biomarker_name": "human-readable name, e.g. Hemoglobin",
      "optimal_low": numeric_or_null,
      "optimal_high": numeric_or_null,
      "unit": "unit string, e.g. g/dL",
      "sex": "male" | "female" | "any",
      "notes": "optional context"
    }
  ]
}

Rules:
- If the source gives a single threshold (e.g. "<5.3%"), set optimal_low to null and \
optimal_high to 5.3.
- If the source gives ">X", set optimal_low to X and optimal_high to null.
- If sex-specific ranges are given, emit separate entries for male and female.
- If no sex distinction is mentioned, use "any".
- Use the exact unit as printed in the source.
- Include ALL biomarkers you can find, even if ranges are incomplete.
"""

MATCH_THRESHOLD = 70
API_DELAY = 0.5


def _log(msg: str) -> None:
    """Print a message and flush immediately so output is visible in real time."""
    click.echo(msg)
    sys.stdout.flush()


def _check_api_key() -> str:
    """Verify ANTHROPIC_API_KEY is available. Exits with error if not."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        _log("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        _log("       Set it in your .env file or export it in your shell.")
        sys.exit(1)
    _log(f"  API key found (ends with ...{key[-4:]})")
    return key


def _check_model() -> str:
    """Return the configured Claude model name."""
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    _log(f"  Model: {model}")
    return model


def _fetch_url(url: str, content_type: str = "html") -> str:
    """Fetch content from a URL and return as text."""
    headers = {"User-Agent": "BloodScope/1.0 (biomarker range extraction)"}
    req = Request(url, headers=headers)
    t0 = time.time()
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    elapsed = time.time() - t0
    _log(f"    Fetched {len(raw):,} bytes in {elapsed:.1f}s")

    if content_type == "pdf":
        _log("    Extracting text from PDF...")
        text = _extract_pdf_text(raw)
        _log(f"    Extracted {len(text):,} characters from PDF")
        return text

    text = raw.decode("utf-8", errors="replace")
    _log(f"    Decoded {len(text):,} characters")
    return text


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import io
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text)


def _llm_extract(content: str, source_name: str, api_key: str, model: str) -> list[dict]:
    """Send fetched content to Claude and return extracted ranges."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    max_content = 100_000
    if len(content) > max_content:
        _log(f"    Truncating content from {len(content):,} to {max_content:,} characters")
        content = content[:max_content]

    user_msg = (
        f"Source: {source_name}\n\n"
        f"--- BEGIN CONTENT ---\n{content}\n--- END CONTENT ---"
    )

    _log(f"    Sending {len(user_msg):,} chars to {model}...")
    t0 = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = time.time() - t0
    _log(f"    LLM response received in {elapsed:.1f}s "
         f"(input={message.usage.input_tokens} tokens, output={message.usage.output_tokens} tokens)")

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON: {exc}\nRaw: {raw[:500]}") from exc

    ranges = data.get("ranges", [])
    _log(f"    Parsed {len(ranges)} range entries from LLM output")
    return ranges


def _build_match_choices(biomarkers: list[dict]) -> dict[str, str]:
    """Build {en_name_lower: biomarker_id} for fuzzy matching."""
    choices: dict[str, str] = {}
    for bm in biomarkers:
        choices[bm["en_name"].lower()] = bm["id"]
        short = bm["id"].replace("_", " ")
        choices[short] = bm["id"]
    return choices


def _match_biomarker(
    extracted_name: str,
    choices: dict[str, str],
    biomarker_units: dict[str, str],
) -> Optional[str]:
    """Fuzzy-match an extracted biomarker name to a canonical ID."""
    name_lower = extracted_name.lower().strip()

    for choice_name, bid in choices.items():
        if name_lower == choice_name:
            return bid

    result = process.extractOne(
        name_lower,
        choices.keys(),
        scorer=fuzz.WRatio,
        score_cutoff=MATCH_THRESHOLD,
    )
    if result is not None:
        matched_name, score, _ = result
        return choices[matched_name]

    return None


def _validate_range(entry: dict) -> bool:
    """Validate that an extracted range is plausible."""
    opt_low = entry.get("optimal_low")
    opt_high = entry.get("optimal_high")

    if opt_low is not None and opt_high is not None:
        if opt_low >= opt_high:
            return False

    if opt_low is None and opt_high is None:
        return False

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Generate optimal range data from web sources via LLM extraction."""


@cli.command()
@click.option("--biomarkers", required=True, type=click.Path(exists=True),
              help="Path to biomarkers.json")
@click.option("--ranges", "ranges", default=None, type=click.Path(),
              help="Path to existing ranges.json (default: $BLOODSCOPE_DATA_DIR/ranges.json or data/ranges.json)")
def discover(biomarkers: str, ranges: str | None) -> None:
    """Show which biomarkers have optimal ranges and which are missing."""
    bm_data = json.loads(Path(biomarkers).read_text(encoding="utf-8"))
    bm_list = bm_data.get("biomarkers", [])
    all_ids = {bm["id"] for bm in bm_list}

    ranges_resolved = ranges if ranges is not None else str(get_default_runtime_data_dir() / "ranges.json")
    ranges_path = Path(ranges_resolved)
    covered: set[str] = set()
    if ranges_path.exists():
        rng_data = json.loads(ranges_path.read_text(encoding="utf-8"))
        covered = {r["biomarker_id"] for r in rng_data.get("ranges", [])}

    missing = sorted(all_ids - covered)
    has_ranges = sorted(covered & all_ids)

    _log(f"Total biomarkers: {len(all_ids)}")
    _log(f"  With optimal ranges: {len(has_ranges)}")
    _log(f"  Missing optimal ranges: {len(missing)}")

    if has_ranges:
        _log("\nBiomarkers with optimal ranges:")
        for bid in has_ranges:
            _log(f"  + {bid}")

    if missing:
        _log("\nBiomarkers without optimal ranges:")
        for bid in missing:
            _log(f"  - {bid}")


@cli.command()
@click.option("--biomarkers", required=True, type=click.Path(exists=True),
              help="Path to biomarkers.json")
@click.option("--sources", "sources", default=_DEFAULT_RANGE_SOURCES, type=click.Path(exists=True),
              help="Path to range_sources.json (default: data_generation/range_sources.json)")
@click.option("--output", required=True, type=click.Path(),
              help="Output path for ranges.json")
@click.option("--merge", is_flag=True, default=False,
              help="Keep existing ranges; only add missing biomarkers.")
@click.option("--source", "single_source", default=None,
              help="Process only the named source (must match a name in range_sources.json).")
def generate(
    biomarkers: str, sources: str, output: str,
    merge: bool, single_source: str | None,
) -> None:
    """Fetch sources, extract optimal ranges via LLM, match to biomarkers, validate, and write."""
    _log("=" * 60)
    _log("init_ranges: Generate optimal ranges from web sources")
    _log("=" * 60)

    # --- Pre-flight checks ---
    _log("\n[1/5] Pre-flight checks...")
    api_key = _check_api_key()
    model = _check_model()

    _log(f"  Biomarkers file: {biomarkers}")
    bm_data = json.loads(Path(biomarkers).read_text(encoding="utf-8"))
    bm_list = bm_data.get("biomarkers", [])
    _log(f"  Loaded {len(bm_list)} biomarkers")
    if not bm_list:
        raise click.ClickException("No biomarkers found — run init_biomarkers first")

    bm_units = {bm["id"]: bm["standard_unit"] for bm in bm_list}

    _log(f"  Sources file: {sources}")
    src_data = json.loads(Path(sources).read_text(encoding="utf-8"))
    source_list: list[dict] = src_data.get("sources", [])
    source_list.sort(key=lambda s: s.get("priority", 999))
    _log(f"  Loaded {len(source_list)} source(s)")
    for s in source_list:
        _log(f"    [{s.get('priority', '?')}] {s['name']}: {s['url']}")

    if single_source:
        source_list = [s for s in source_list if s["name"] == single_source]
        if not source_list:
            raise click.ClickException(f"Source '{single_source}' not found in {sources}")
        _log(f"  Filtering to single source: {single_source}")

    # --- Load existing ranges (merge mode) ---
    existing_ranges: list[dict] = []
    existing_keys: set[tuple[str, str]] = set()
    if merge:
        out_path = Path(output)
        if out_path.exists():
            rng_data = json.loads(out_path.read_text(encoding="utf-8"))
            existing_ranges = rng_data.get("ranges", [])
            existing_keys = {(r["biomarker_id"], r["sex"]) for r in existing_ranges}
            _log(f"\n  Merge mode: {len(existing_ranges)} existing entries loaded, will skip those.")
        else:
            _log(f"\n  Merge mode: output file {output} does not exist yet, starting fresh.")

    choices = _build_match_choices(bm_list)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_results: list[dict] = list(existing_ranges) if merge else []
    result_keys: set[tuple[str, str]] = set(existing_keys)
    total_new = 0
    total_skipped = 0
    total_unmatched = 0
    total_invalid = 0
    grand_t0 = time.time()

    # --- Process each source ---
    for src_idx, src in enumerate(source_list, 1):
        src_name = src["name"]
        src_url = src["url"]
        src_type = src.get("type", "html")

        step = f"[{1 + src_idx}/5]"
        _log(f"\n{step} Processing source {src_idx}/{len(source_list)}: {src_name}")
        _log(f"    URL: {src_url}")
        _log(f"    Type: {src_type}")

        # -- Fetch --
        _log(f"    Fetching content...")
        try:
            content = _fetch_url(src_url, content_type=src_type)
        except RuntimeError as exc:
            _log(f"    ERROR fetching: {exc}")
            _log(f"    Skipping this source.")
            continue

        if not content.strip():
            _log(f"    WARNING: Fetched content is empty — skipping this source.")
            continue

        # -- LLM extraction --
        _log(f"    Extracting ranges via LLM...")
        try:
            extracted = _llm_extract(content, src_name, api_key, model)
        except RuntimeError as exc:
            _log(f"    ERROR from LLM: {exc}")
            _log(f"    Skipping this source.")
            continue

        if not extracted:
            _log(f"    WARNING: LLM returned 0 range entries — skipping.")
            continue

        # -- Match & validate --
        _log(f"    Matching {len(extracted)} entries to biomarkers.json...")
        src_new = 0
        src_skipped = 0
        for entry in extracted:
            bm_name = entry.get("biomarker_name", "")
            if not bm_name:
                continue

            bid = _match_biomarker(bm_name, choices, bm_units)
            if bid is None:
                _log(f"      ? No match for '{bm_name}'")
                total_unmatched += 1
                continue

            if not _validate_range(entry):
                _log(f"      ! Invalid range for '{bm_name}' -> {bid}: "
                     f"low={entry.get('optimal_low')}, high={entry.get('optimal_high')}")
                total_invalid += 1
                continue

            sex = entry.get("sex", "any")
            if sex not in ("male", "female", "any"):
                sex = "any"

            key = (bid, sex)
            if key in result_keys:
                src_skipped += 1
                total_skipped += 1
                continue

            opt_low = entry.get("optimal_low")
            opt_high = entry.get("optimal_high")

            range_entry = {
                "biomarker_id": bid,
                "sex": sex,
                "age_min": 18,
                "age_max": 120,
                "unit": bm_units.get(bid, entry.get("unit", "")),
                "optimal_low": opt_low,
                "optimal_high": opt_high,
                "source": src_name,
                "source_url": src_url,
            }

            all_results.append(range_entry)
            result_keys.add(key)
            src_new += 1
            total_new += 1
            _log(f"      + {bid:30s} ({sex:6s}) [{opt_low} - {opt_high}] {bm_units.get(bid, '')}")

        _log(f"    Source done: {src_new} new, {src_skipped} skipped (higher-priority source already covered)")
        time.sleep(API_DELAY)

    # --- Write output ---
    _log(f"\n[5/5] Writing output...")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"ranges": all_results}
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    elapsed = time.time() - grand_t0
    _log(f"\nDone in {elapsed:.1f}s.")
    _log(f"Wrote {len(all_results)} total optimal range entries to {output}")
    _log(f"  {total_new} new, {total_skipped} skipped, "
         f"{total_unmatched} unmatched, {total_invalid} invalid")

    covered_ids = {r["biomarker_id"] for r in all_results}
    all_ids = {bm["id"] for bm in bm_list}
    uncovered = len(all_ids - covered_ids)
    if uncovered:
        _log(f"  {uncovered} biomarkers still without optimal ranges")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
