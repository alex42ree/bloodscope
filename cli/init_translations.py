"""Import translations from LOINC Linguistic Variant CSV files.

Instead of hardcoding filenames, this script reads the
``LinguisticVariants.csv`` index to discover all available variant files.
It then filters by the languages listed in ``data_generation/languages.json`` (default) and
merges all regional variants (e.g. esES + esAR + esMX) into a single
language entry so the matcher benefits from regional differences.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import click
import pandas as pd

_DEFAULT_LANGUAGES = str(Path(__file__).resolve().parent.parent / "data_generation" / "languages.json")

RELEVANT_COLUMNS = [
    "LOINC_NUM",
    "COMPONENT",
    "LinguisticVariantDisplayName",
    "SHORTNAME",
    "RELATEDNAMES2",
]


def _normalize_term(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFKD", text)
    return text.lower()


def _extract_variants_from_row(row: pd.Series) -> list[str]:
    """Extract all unique translation terms from a single LOINC linguistic row."""
    terms: list[str] = []

    for col in ("COMPONENT", "LinguisticVariantDisplayName", "SHORTNAME"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            terms.append(str(val).strip())

    related = row.get("RELATEDNAMES2")
    if pd.notna(related) and str(related).strip():
        for part in str(related).split(";"):
            part = part.strip()
            if part:
                terms.append(part)

    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        key = _normalize_term(t)
        if key and key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _load_biomarker_loinc_map(biomarkers_path: str) -> dict[str, str]:
    """Return ``{loinc_code: biomarker_id}``."""
    path = Path(biomarkers_path)
    if not path.exists():
        raise click.ClickException(f"Biomarkers file not found: {biomarkers_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {b["loinc_code"]: b["id"] for b in data.get("biomarkers", [])}


def _load_languages(languages_path: str) -> list[str]:
    path = Path(languages_path)
    if not path.exists():
        raise click.ClickException(f"Languages file not found: {languages_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("supported_languages", [])


def _discover_variant_files(
    loinc_dir: str,
    supported_languages: list[str],
) -> dict[str, list[Path]]:
    """Read LinguisticVariants.csv index and return ``{lang: [paths]}``."""
    index_path = Path(loinc_dir) / "LinguisticVariants.csv"
    if not index_path.exists():
        raise click.ClickException(
            f"LinguisticVariants.csv not found in {loinc_dir}. "
            "Make sure --loinc-dir points to the LinguisticVariants directory."
        )

    index_df = pd.read_csv(index_path, dtype=str)
    result: dict[str, list[Path]] = {}

    for _, row in index_df.iterrows():
        iso_lang = str(row["ISO_LANGUAGE"]).strip().lower()
        if iso_lang not in supported_languages:
            continue

        iso_country = str(row["ISO_COUNTRY"]).strip()
        variant_id = str(row["ID"]).strip()
        filename = f"{iso_lang}{iso_country}{variant_id}LinguisticVariant.csv"
        filepath = Path(loinc_dir) / filename

        if not filepath.exists():
            click.echo(f"  Warning: discovered {filename} in index but file not found — skipping")
            continue

        if iso_lang not in result:
            result[iso_lang] = []
        result[iso_lang].append(filepath)

    return result


def _load_existing(output_path: str) -> dict:
    p = Path(output_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"translations": []}


def _find_entry(translations: list[dict], biomarker_id: str, language: str) -> dict | None:
    for entry in translations:
        if entry["biomarker_id"] == biomarker_id and entry["language"] == language:
            return entry
    return None


def _existing_terms(entry: dict) -> set[str]:
    return {_normalize_term(v["term"]) if isinstance(v, dict) else _normalize_term(v)
            for v in entry.get("variants", [])}


@click.command()
@click.option("--loinc-dir", required=True, type=click.Path(exists=True),
              help="Path to LOINC AccessoryFiles/LinguisticVariants directory")
@click.option("--biomarkers", required=True, type=click.Path(exists=True),
              help="Path to biomarkers.json")
@click.option("--output", required=True, type=click.Path(),
              help="Path to translations.json (will merge if it exists)")
@click.option("--languages", default=_DEFAULT_LANGUAGES, type=click.Path(exists=True),
              help="Path to languages.json (defines supported languages; default: data_generation/languages.json)")
def main(loinc_dir: str, biomarkers: str, output: str, languages: str) -> None:
    """Import LOINC linguistic variant CSVs into translations.json."""
    supported = _load_languages(languages)
    click.echo(f"Supported languages: {', '.join(supported)}")

    loinc_map = _load_biomarker_loinc_map(biomarkers)
    click.echo(f"Loaded {len(loinc_map)} biomarker LOINC codes from {biomarkers}")

    variant_files = _discover_variant_files(loinc_dir, supported)
    click.echo(f"Discovered variant files for: {', '.join(sorted(variant_files.keys()))}")

    data = _load_existing(output)
    total_added = 0

    for language, filepaths in sorted(variant_files.items()):
        for filepath in filepaths:
            click.echo(f"  Reading {filepath.name} ({language}) ...")
            try:
                df = pd.read_csv(filepath, dtype=str, low_memory=False)
            except Exception as exc:
                click.echo(f"    Error reading {filepath.name}: {exc}")
                continue

            if "LOINC_NUM" not in df.columns:
                click.echo(f"    Warning: LOINC_NUM column missing in {filepath.name}")
                continue

            added = 0
            skipped_dup = 0
            matched_codes = 0
            for _, row in df.iterrows():
                loinc_code = str(row["LOINC_NUM"]).strip()
                if loinc_code not in loinc_map:
                    continue

                matched_codes += 1
                biomarker_id = loinc_map[loinc_code]
                terms = _extract_variants_from_row(row)
                if not terms:
                    continue

                entry = _find_entry(data["translations"], biomarker_id, language)
                if entry is None:
                    entry = {"biomarker_id": biomarker_id, "language": language, "variants": []}
                    data["translations"].append(entry)

                existing = _existing_terms(entry)
                for term in terms:
                    if _normalize_term(term) not in existing:
                        entry["variants"].append({
                            "term": term,
                            "source": "loinc",
                            "lab": None,
                        })
                        existing.add(_normalize_term(term))
                        added += 1
                    else:
                        skipped_dup += 1

            click.echo(
                f"    {matched_codes} LOINC matches, "
                f"{added} new variants added, "
                f"{skipped_dup} already present"
            )
            total_added += added

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    click.echo(f"\nTotal: added {total_added} LOINC variants. "
               f"Wrote {len(data['translations'])} entries to {output}")


if __name__ == "__main__":
    main()
