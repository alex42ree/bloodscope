"""CLI for managing unit_conversions.json.

Each conversion entry now carries ``molecular_weight``, ``source``, and
``bidirectional`` metadata as described in the addendum spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

DEFAULT_PATH = "data/unit_conversions.json"


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"conversions": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _find_conversion(
    conversions: list[dict], biomarker_id: str, from_unit: str, to_unit: str,
) -> tuple[int, dict] | tuple[None, None]:
    for i, c in enumerate(conversions):
        if (
            c["biomarker_id"] == biomarker_id
            and c["from_unit"] == from_unit
            and c["to_unit"] == to_unit
        ):
            return i, c
    return None, None


@click.group()
def cli() -> None:
    """Manage biomarker-specific unit conversion factors."""


@cli.command()
@click.option("--biomarker-id", required=True)
@click.option("--from-unit", required=True)
@click.option("--to-unit", required=True)
@click.option("--factor", required=True, type=float,
              help="Multiplication factor: to_value = from_value * factor")
@click.option("--molecular-weight", default=None, type=float,
              help="Molecular weight in g/mol (for molar conversions)")
@click.option("--source", default="",
              help="Source reference (e.g. 'AMA SI Conversion Table', 'dimensional')")
@click.option("--bidirectional/--no-bidirectional", default=True,
              help="Whether reverse conversion = divide by factor")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def add(
    biomarker_id: str, from_unit: str, to_unit: str, factor: float,
    molecular_weight: float | None, source: str, bidirectional: bool,
    filepath: str,
) -> None:
    """Add or update a unit conversion factor."""
    if factor <= 0:
        raise click.ClickException("Factor must be positive")

    data = _load(filepath)
    idx, _ = _find_conversion(data["conversions"], biomarker_id, from_unit, to_unit)

    entry = {
        "biomarker_id": biomarker_id,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "factor": factor,
        "molecular_weight": molecular_weight,
        "source": source,
        "bidirectional": bidirectional,
    }

    if idx is not None:
        data["conversions"][idx] = entry
        action = "Updated"
    else:
        data["conversions"].append(entry)
        action = "Added"

    _save(data, filepath)
    click.echo(f"{action} conversion for {biomarker_id}: {from_unit} -> {to_unit} "
               f"(x{factor}, MW={molecular_weight}, bidir={bidirectional})")


@cli.command("list")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def list_conversions(filepath: str) -> None:
    """List all unit conversions."""
    data = _load(filepath)
    conversions = data.get("conversions", [])
    if not conversions:
        click.echo("No conversions defined.")
        return
    for c in conversions:
        mw = c.get("molecular_weight")
        mw_str = f", MW={mw}" if mw else ""
        src = c.get("source", "")
        src_str = f" [{src}]" if src else ""
        click.echo(
            f"  {c['biomarker_id']}: {c['from_unit']} -> {c['to_unit']} "
            f"(x{c['factor']}{mw_str}){src_str}"
        )
    click.echo(f"\nTotal: {len(conversions)} conversions")


@cli.command()
@click.option("--biomarker-id", required=True)
@click.option("--file", "filepath", default=DEFAULT_PATH)
def show(biomarker_id: str, filepath: str) -> None:
    """Show all conversions for a specific biomarker."""
    data = _load(filepath)
    found = False
    for c in data.get("conversions", []):
        if c["biomarker_id"] == biomarker_id:
            found = True
            mw = c.get("molecular_weight")
            bidir = c.get("bidirectional", True)
            click.echo(
                f"  {c['from_unit']} -> {c['to_unit']}: x{c['factor']} "
                f"(MW={mw}, bidir={bidir}, source={c.get('source', '')})"
            )
    if not found:
        click.echo(f"No conversions found for {biomarker_id}")


@cli.command()
@click.option("--against", "biomarkers_file", default="data/biomarkers.json",
              help="Path to biomarkers.json")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def validate(biomarkers_file: str, filepath: str) -> None:
    """Check that every biomarker with a non-trivial unit has conversions."""
    bio_path = Path(biomarkers_file)
    if not bio_path.exists():
        raise click.ClickException(f"Biomarkers file not found: {biomarkers_file}")

    bio_data = json.loads(bio_path.read_text(encoding="utf-8"))
    data = _load(filepath)

    covered_ids = {c["biomarker_id"] for c in data.get("conversions", [])}

    trivial_units = {"%", "ratio", "index"}
    needs_conversion = []
    for b in bio_data.get("biomarkers", []):
        if b["standard_unit"] not in trivial_units:
            needs_conversion.append(b["id"])

    missing = sorted(set(needs_conversion) - covered_ids)
    if missing:
        click.echo(f"WARNING: {len(missing)} biomarkers may need conversion entries:")
        for bid in missing:
            click.echo(f"  - {bid}")
    else:
        click.echo("All non-trivial biomarkers have conversion entries.")


@cli.command("check-consistency")
@click.option("--file", "filepath", default=DEFAULT_PATH)
def check_consistency(filepath: str) -> None:
    """Check that related biomarkers share consistent factors.

    For example, total_cholesterol, ldl_cholesterol, and hdl_cholesterol
    should all use the same mmol/L -> mg/dL factor.
    """
    data = _load(filepath)
    by_units: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for c in data.get("conversions", []):
        key = (c["from_unit"], c["to_unit"])
        if key not in by_units:
            by_units[key] = []
        by_units[key].append((c["biomarker_id"], c["factor"]))

    issues = 0
    for (from_u, to_u), entries in by_units.items():
        if len(entries) < 2:
            continue
        factors = {f for _, f in entries}
        if len(factors) > 1:
            click.echo(f"\n  {from_u} -> {to_u} has different factors:")
            for bid, factor in sorted(entries):
                click.echo(f"    {bid}: x{factor}")
            issues += 1

    if issues == 0:
        click.echo("All unit pairs have consistent factors across biomarkers "
                    "(where different factors are expected due to different molecular weights, "
                    "this is normal).")
    else:
        click.echo(f"\n{issues} unit pair(s) with varying factors (review if unexpected).")


if __name__ == "__main__":
    cli()
