"""CLI for managing ranges.json (optimal ranges for biomarker classification)."""

from __future__ import annotations

import json
from pathlib import Path

import click

DEFAULT_PATH = "data/ranges.json"


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ranges": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _find_range(
    ranges: list[dict], biomarker_id: str, sex: str, age_min: int, age_max: int,
) -> tuple[int, dict] | tuple[None, None]:
    for i, r in enumerate(ranges):
        if (
            r["biomarker_id"] == biomarker_id
            and r["sex"] == sex
            and r["age_min"] == age_min
            and r["age_max"] == age_max
        ):
            return i, r
    return None, None


@click.group()
def cli() -> None:
    """Manage biomarker optimal ranges."""


@cli.command("set")
@click.option("--biomarker-id", required=True)
@click.option("--sex", required=True, type=click.Choice(["male", "female", "any"]))
@click.option("--age-min", required=True, type=int)
@click.option("--age-max", required=True, type=int)
@click.option("--optimal-low", required=True, type=float)
@click.option("--optimal-high", required=True, type=float)
@click.option("--unit", default="", help="Unit for this range (informational)")
@click.option("--source", default="", help="Source reference for this range")
@click.option("--source-url", default=None, help="URL of the source")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def set_range(
    biomarker_id: str, sex: str, age_min: int, age_max: int,
    optimal_low: float, optimal_high: float,
    unit: str, source: str, source_url: str | None, filepath: str,
) -> None:
    """Add or update an optimal range for a biomarker."""
    if optimal_low >= optimal_high:
        raise click.ClickException("optimal_low must be < optimal_high")

    data = _load(filepath)
    idx, existing = _find_range(data["ranges"], biomarker_id, sex, age_min, age_max)

    entry = {
        "biomarker_id": biomarker_id,
        "sex": sex,
        "age_min": age_min,
        "age_max": age_max,
        "unit": unit,
        "optimal_low": optimal_low,
        "optimal_high": optimal_high,
        "source": source,
        "source_url": source_url,
    }

    if idx is not None:
        data["ranges"][idx] = entry
        action = "Updated"
    else:
        data["ranges"].append(entry)
        action = "Added"

    _save(data, filepath)
    click.echo(f"{action} optimal range for {biomarker_id} ({sex}, age {age_min}-{age_max})")


@cli.command()
@click.option("--biomarker-id", required=True)
@click.option("--file", "filepath", default=DEFAULT_PATH)
def show(biomarker_id: str, filepath: str) -> None:
    """Show all optimal ranges for a biomarker."""
    data = _load(filepath)
    found = False
    for r in data["ranges"]:
        if r["biomarker_id"] == biomarker_id:
            found = True
            click.echo(
                f"  {r['sex']} age {r['age_min']}-{r['age_max']}: "
                f"optimal [{r['optimal_low']}-{r['optimal_high']}] "
                f"{r.get('unit', '')} (source: {r.get('source', 'N/A')})"
            )
    if not found:
        click.echo(f"No optimal ranges found for {biomarker_id}")


@cli.command("import")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True))
@click.option("--file", "filepath", default=DEFAULT_PATH)
def import_csv(csv_path: str, filepath: str) -> None:
    """Bulk-import optimal ranges from a CSV file.

    Expected columns: biomarker_id, sex, age_min, age_max,
    optimal_low, optimal_high, unit, source
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"biomarker_id", "sex", "age_min", "age_max",
                "optimal_low", "optimal_high"}
    if not required.issubset(set(df.columns)):
        raise click.ClickException(f"CSV must have columns: {required}")

    data = _load(filepath)
    count = 0
    for _, row in df.iterrows():
        entry = {
            "biomarker_id": str(row["biomarker_id"]),
            "sex": str(row["sex"]),
            "age_min": int(row["age_min"]),
            "age_max": int(row["age_max"]),
            "unit": str(row.get("unit", "")),
            "optimal_low": float(row["optimal_low"]),
            "optimal_high": float(row["optimal_high"]),
            "source": str(row.get("source", "")),
            "source_url": str(row.get("source_url", "")) or None,
        }
        if entry["optimal_low"] >= entry["optimal_high"]:
            click.echo(f"  Skipping {entry['biomarker_id']}: optimal_low >= optimal_high")
            continue

        idx, _ = _find_range(
            data["ranges"], entry["biomarker_id"],
            entry["sex"], entry["age_min"], entry["age_max"],
        )
        if idx is not None:
            data["ranges"][idx] = entry
        else:
            data["ranges"].append(entry)
        count += 1

    _save(data, filepath)
    click.echo(f"Imported/updated {count} optimal ranges from {csv_path}")


@cli.command()
@click.option("--biomarkers-file", default="data/biomarkers.json")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def validate(biomarkers_file: str, filepath: str) -> None:
    """Check that every biomarker has at least one optimal range entry."""
    bio_path = Path(biomarkers_file)
    if not bio_path.exists():
        raise click.ClickException(f"Biomarkers file not found: {biomarkers_file}")

    bio_data = json.loads(bio_path.read_text(encoding="utf-8"))
    all_ids = {b["id"] for b in bio_data.get("biomarkers", [])}

    data = _load(filepath)
    covered = {r["biomarker_id"] for r in data["ranges"]}

    missing = sorted(all_ids - covered)
    if missing:
        click.echo(f"INFO: {len(missing)} biomarkers have no optimal ranges (will classify as normal/out_of_range only):")
        for bid in missing:
            click.echo(f"  - {bid}")
    else:
        click.echo(f"All {len(all_ids)} biomarkers have at least one optimal range entry.")

    extra = sorted(covered - all_ids)
    if extra:
        click.echo(f"\nNote: {len(extra)} range entries reference unknown biomarker IDs:")
        for bid in extra:
            click.echo(f"  - {bid}")


if __name__ == "__main__":
    cli()
