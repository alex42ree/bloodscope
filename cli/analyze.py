"""Analyze a lab-report PDF from the command line.

Runs the full BloodScope pipeline (extract, match, convert, classify) and
prints a categorised results table.  Optionally writes the complete
PipelineResult as JSON.

Usage
-----
::

    python -m cli.analyze path/to/report.pdf

    python -m cli.analyze path/to/report.pdf --sex male --dob 1990-05-15

    python -m cli.analyze path/to/report.pdf --sex female --dob 1985-03-22 \
        --output result.json
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from core.data_paths import get_default_runtime_data_dir
from core.exceptions import BloodScopePdfError
from core.lab_reference import format_lab_reference_for_display
from core.pdf_validation import validate_pdf_path
from core.measurement_display import format_measurement_display
from core.pipeline import process_report
from core.schemas import Classification, ClassifiedMarker, PipelineResult

load_dotenv()

logger = logging.getLogger(__name__)

CATEGORY_LABELS: dict[str, str] = {
    "hematology": "Hematology",
    "lipid": "Lipid Panel",
    "metabolism": "Metabolism",
    "liver": "Liver Function",
    "kidney": "Kidney Function",
    "thyroid": "Thyroid",
    "electrolytes": "Electrolytes & Minerals",
    "vitamins": "Vitamins",
    "inflammation": "Inflammation",
    "hormones": "Hormones",
    "coagulation": "Coagulation",
    "cardiac": "Cardiac",
    "pancreas": "Pancreas",
    "immunology": "Immunology",
    "other": "Other",
}

STATUS_LABELS = {
    Classification.OPTIMAL: "OPTIMAL",
    Classification.NORMAL: "NORMAL",
    Classification.OUT_OF_RANGE: "OUT OF RANGE",
    Classification.UNKNOWN: "UNKNOWN",
}

REQUIRED_DATA_FILES = [
    "biomarkers.json",
    "translations.json",
    "unit_conversions.json",
    "ranges.json",
]


def _log(msg: str) -> None:
    click.echo(msg)
    sys.stdout.flush()


def _preflight(pdf_path: Path, data_dir: Path) -> bool:
    """Return True when all pre-conditions are met, else print errors."""
    ok = True

    if not pdf_path.is_file():
        _log(f"ERROR: PDF not found: {pdf_path}")
        ok = False
    else:
        try:
            validate_pdf_path(pdf_path)
        except BloodScopePdfError as exc:
            _log(f"ERROR: {exc}")
            ok = False

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        _log("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env or export it.")
        ok = False

    for name in REQUIRED_DATA_FILES:
        p = data_dir / name
        if not p.is_file():
            _log(f"ERROR: Required data file missing: {p}")
            ok = False

    return ok


def _print_table(markers: list[ClassifiedMarker], title: str) -> None:
    """Print a fixed-width table for a category group."""
    if not markers:
        return

    col_w = {"name": 30, "value": 12, "unit": 10, "ref": 18, "optimal": 18, "status": 14}

    _log(f"\n  {title}")
    _log(f"  {'─' * (sum(col_w.values()) + len(col_w) - 1)}")

    header = (
        f"  {'Biomarker':<{col_w['name']}} "
        f"{'Value':>{col_w['value']}} "
        f"{'Unit':<{col_w['unit']}} "
        f"{'Reference':<{col_w['ref']}} "
        f"{'Optimal':<{col_w['optimal']}} "
        f"{'Status':<{col_w['status']}}"
    )
    _log(header)
    _log(f"  {'─' * (sum(col_w.values()) + len(col_w) - 1)}")

    for m in markers:
        name = (m.en_name or m.original_name)[:col_w["name"]]
        val_str = format_measurement_display(
            m.value, m.converted_value, m.value_modifier, use_converted=True,
        )
        unit = (m.standard_unit or m.unit)[:col_w["unit"]]
        ref = format_lab_reference_for_display(m.lab_reference_low, m.lab_reference_high)
        opt = format_lab_reference_for_display(m.optimal_low, m.optimal_high)
        status = STATUS_LABELS.get(m.classification, "?")

        _log(
            f"  {name:<{col_w['name']}} "
            f"{val_str:>{col_w['value']}} "
            f"{unit:<{col_w['unit']}} "
            f"{ref:<{col_w['ref']}} "
            f"{opt:<{col_w['optimal']}} "
            f"{status:<{col_w['status']}}"
        )


def _print_unclassified(markers: list[ClassifiedMarker]) -> None:
    if not markers:
        return

    _log("\n  Unclassified Biomarkers (not matched to database)")
    _log(f"  {'─' * 60}")
    _log(f"  {'Original Name':<30} {'Value':>12} {'Unit':<10}")
    _log(f"  {'─' * 60}")

    for m in markers:
        val_str = format_measurement_display(
            m.value, m.converted_value, m.value_modifier, use_converted=True,
        )
        _log(f"  {m.original_name:<30} {val_str:>12} {m.unit:<10}")


def _print_summary(result: PipelineResult) -> None:
    total = len(result.classified)
    optimal = sum(1 for m in result.classified if m.classification == Classification.OPTIMAL)
    normal = sum(1 for m in result.classified if m.classification == Classification.NORMAL)
    oor = sum(1 for m in result.classified if m.classification == Classification.OUT_OF_RANGE)
    unknown = sum(1 for m in result.classified if m.classification == Classification.UNKNOWN)
    unmatched = len(result.unclassified)

    _log("")
    _log("  ══════════════════════════════════════════")
    _log(f"  Total classified:  {total}")
    _log(f"    Optimal:         {optimal}")
    _log(f"    Normal:          {normal}")
    _log(f"    Out of Range:    {oor}")
    _log(f"    Unknown:         {unknown}")
    if unmatched:
        _log(f"  Unmatched:         {unmatched}")
    _log("  ══════════════════════════════════════════")


def _print_patient_context(result: PipelineResult) -> None:
    """Print effective vs extracted sex/DOB/age used for classification."""
    p = result.patient
    ext_sex = p.sex or "(not in PDF)"
    ext_dob = p.date_of_birth.isoformat() if p.date_of_birth else "(not in PDF)"
    eff_dob = result.effective_date_of_birth.isoformat() if result.effective_date_of_birth else "—"
    _log("")
    _log("  Patient context (classification)")
    _log(f"  {'─' * 50}")
    _log(
        f"  Sex:   {result.effective_sex}  [{result.sex_source}]"
        f"   (PDF extraction: {ext_sex})",
    )
    _log(
        f"  DOB:   {eff_dob}  [{result.date_of_birth_source}]"
        f"   (PDF extraction: {ext_dob})",
    )
    _log(f"  Age:   {result.effective_age} (for optimal / range matching)")
    _log(f"  {'─' * 50}")


def _print_results(result: PipelineResult) -> None:
    """Group classified markers by category and print tables."""
    _print_patient_context(result)
    by_category: dict[str, list[ClassifiedMarker]] = {}
    for m in result.classified:
        cat = m.category or "other"
        by_category.setdefault(cat, []).append(m)

    for cat_key in CATEGORY_LABELS:
        if cat_key in by_category:
            _print_table(by_category[cat_key], CATEGORY_LABELS[cat_key])

    for cat_key, markers in by_category.items():
        if cat_key not in CATEGORY_LABELS:
            _print_table(markers, cat_key.title())

    _print_unclassified(result.unclassified)
    _print_summary(result)


@click.command("analyze")
@click.argument("pdf", type=click.Path(exists=False))
@click.option(
    "--sex",
    default=None,
    type=click.Choice(["male", "female"]),
    help="Patient sex; omit to use value extracted from the PDF.",
)
@click.option(
    "--dob",
    "dob",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Date of birth YYYY-MM-DD; omit to use PDF extraction or age from report.",
)
@click.option("-o", "--output", "output_path", default=None, type=click.Path(), help="Write full JSON result to this file.")
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(),
    help="Directory with biomarkers.json, translations.json, … "
    "(default: BLOODSCOPE_DATA_DIR or ./data).",
)
def analyze(pdf: str, sex: str | None, dob: datetime | None, output_path: str | None, data_dir: str | None) -> None:
    """Analyze a lab-report PDF and print classified biomarker results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdf_path = Path(pdf)
    data_path = Path(data_dir) if data_dir else get_default_runtime_data_dir()
    dob_date: date | None = dob.date() if dob is not None else None

    if not _preflight(pdf_path, data_path):
        raise SystemExit(1)

    hint = []
    if sex:
        hint.append(f"sex override={sex}")
    if dob_date:
        hint.append(f"dob override={dob_date}")
    extra = f" ({', '.join(hint)})" if hint else " (PDF extraction for sex/age when omitted)"
    _log(f"Analyzing {pdf_path.name}{extra} ...")

    try:
        result = process_report(str(pdf_path), sex, dob_date, data_dir=data_path)
    except BloodScopePdfError as exc:
        _log(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        _log(f"ERROR: Pipeline failed: {exc}")
        raise SystemExit(1) from exc

    _print_results(result)

    if output_path:
        out = Path(output_path)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        _log(f"\nFull result written to {out}")


if __name__ == "__main__":
    analyze()
