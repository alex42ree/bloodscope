"""CLI for managing translations.json (multilingual biomarker name variants).

Variant objects have the form ``{"term": "...", "source": "...", "lab": ...}``.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import click

DEFAULT_PATH = "data/translations.json"


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"translations": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _find_entry(translations: list[dict], biomarker_id: str, language: str) -> dict | None:
    for entry in translations:
        if entry["biomarker_id"] == biomarker_id and entry["language"] == language:
            return entry
    return None


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def _get_term(v) -> str:
    return v["term"] if isinstance(v, dict) else v


def _existing_terms_norm(entry: dict) -> set[str]:
    return {_norm(_get_term(v)) for v in entry.get("variants", [])}


# ── CLI group ────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """Manage multilingual biomarker translations."""


@cli.command()
@click.option("--biomarker-id", required=True, help="Canonical biomarker ID")
@click.option("--language", required=True, type=click.Choice(["es", "de", "en", "fr"]))
@click.option("--variant", required=True, help="Translation variant to add")
@click.option("--source", default="manual", type=click.Choice(["loinc", "lab_pdf", "llm", "manual"]))
@click.option("--lab", default=None, help="Lab name (for lab_pdf source)")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def add(biomarker_id: str, language: str, variant: str, source: str, lab: str | None, filepath: str) -> None:
    """Add a single translation variant for a biomarker."""
    data = _load(filepath)
    entry = _find_entry(data["translations"], biomarker_id, language)

    if entry is None:
        entry = {"biomarker_id": biomarker_id, "language": language, "variants": []}
        data["translations"].append(entry)

    if _norm(variant) not in _existing_terms_norm(entry):
        entry["variants"].append({"term": variant, "source": source, "lab": lab})
        _save(data, filepath)
        click.echo(f"Added '{variant}' for {biomarker_id} ({language}) [source={source}]")
    else:
        click.echo(f"Variant '{variant}' already exists for {biomarker_id} ({language})")


@cli.command()
@click.option("--biomarker-id", default=None, help="Filter by biomarker ID")
@click.option("--lab", default=None, help="Filter by lab name")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def show(biomarker_id: str | None, lab: str | None, filepath: str) -> None:
    """Show translations, optionally filtered by biomarker or lab."""
    data = _load(filepath)
    for entry in data["translations"]:
        if biomarker_id and entry["biomarker_id"] != biomarker_id:
            continue
        variants = entry["variants"]
        if lab:
            variants = [v for v in variants if isinstance(v, dict) and v.get("lab") == lab]
        if not variants:
            continue
        terms = [_get_term(v) for v in variants]
        click.echo(f"  {entry['biomarker_id']} [{entry['language']}]: {', '.join(terms)}")


@cli.command("import")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True),
              help="CSV with columns: biomarker_id, language, variant (optional: source, lab)")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def import_csv(csv_path: str, filepath: str) -> None:
    """Bulk-import translations from a CSV file."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"biomarker_id", "language", "variant"}
    if not required.issubset(set(df.columns)):
        raise click.ClickException(f"CSV must have columns: {required}")

    data = _load(filepath)
    count = 0
    for _, row in df.iterrows():
        bid = str(row["biomarker_id"])
        lang = str(row["language"])
        var = str(row["variant"])
        source = str(row.get("source", "manual")) if "source" in df.columns else "manual"
        lab_val = str(row["lab"]) if "lab" in df.columns and pd.notna(row.get("lab")) else None

        entry = _find_entry(data["translations"], bid, lang)
        if entry is None:
            entry = {"biomarker_id": bid, "language": lang, "variants": []}
            data["translations"].append(entry)
        if _norm(var) not in _existing_terms_norm(entry):
            entry["variants"].append({"term": var, "source": source, "lab": lab_val})
            count += 1

    _save(data, filepath)
    click.echo(f"Imported {count} new variants from {csv_path}")


@cli.command()
@click.option("--language", required=True, type=click.Choice(["es", "de", "en", "fr"]))
@click.option("--biomarkers-file", default="data/biomarkers.json")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def gaps(language: str, biomarkers_file: str, filepath: str) -> None:
    """Identify biomarkers that have no translations for a given language."""
    bio_path = Path(biomarkers_file)
    if not bio_path.exists():
        raise click.ClickException(f"Biomarkers file not found: {biomarkers_file}")

    bio_data = json.loads(bio_path.read_text(encoding="utf-8"))
    all_ids = {b["id"] for b in bio_data.get("biomarkers", [])}

    data = _load(filepath)
    covered = {e["biomarker_id"] for e in data["translations"] if e["language"] == language}

    missing = sorted(all_ids - covered)
    if missing:
        click.echo(f"Missing {language} translations for {len(missing)} biomarkers:")
        for bid in missing:
            click.echo(f"  - {bid}")
    else:
        click.echo(f"All biomarkers have {language} translations!")


@cli.command()
@click.option("--file", "filepath", default=DEFAULT_PATH)
def stats(filepath: str) -> None:
    """Show variant count per source type."""
    data = _load(filepath)
    counts: dict[str, int] = {}
    for entry in data["translations"]:
        for v in entry["variants"]:
            src = v["source"] if isinstance(v, dict) else "legacy"
            counts[src] = counts.get(src, 0) + 1
    if not counts:
        click.echo("No translations found.")
        return
    for src, cnt in sorted(counts.items()):
        click.echo(f"  {src}: {cnt} variants")
    click.echo(f"  Total: {sum(counts.values())} variants")


@cli.command("check-duplicates")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def check_duplicates(filepath: str) -> None:
    """Find terms that map to multiple different biomarker IDs."""
    data = _load(filepath)
    term_map: dict[str, list[str]] = {}
    for entry in data["translations"]:
        bid = entry["biomarker_id"]
        for v in entry["variants"]:
            key = _norm(_get_term(v))
            if key not in term_map:
                term_map[key] = []
            if bid not in term_map[key]:
                term_map[key].append(bid)

    dupes = {t: bids for t, bids in term_map.items() if len(bids) > 1}
    if dupes:
        click.echo(f"Found {len(dupes)} duplicate terms:")
        for term, bids in sorted(dupes.items()):
            click.echo(f"  '{term}' -> {', '.join(bids)}")
    else:
        click.echo("No duplicate terms found.")


@cli.command("export-csv")
@click.option("--output", required=True, type=click.Path())
@click.option("--file", "filepath", default=DEFAULT_PATH)
def export_csv(output: str, filepath: str) -> None:
    """Export translations to a flat CSV for review."""
    data = _load(filepath)
    rows = []
    for entry in data["translations"]:
        for v in entry["variants"]:
            rows.append({
                "biomarker_id": entry["biomarker_id"],
                "language": entry["language"],
                "term": _get_term(v),
                "source": v.get("source", "") if isinstance(v, dict) else "",
                "lab": v.get("lab", "") if isinstance(v, dict) else "",
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    click.echo(f"Exported {len(rows)} rows to {output}")


if __name__ == "__main__":
    cli()
