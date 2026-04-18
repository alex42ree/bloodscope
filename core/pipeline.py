"""End-to-end processing pipeline: PDF -> classified biomarkers."""

from __future__ import annotations

import io
import json
import logging
from datetime import date
from pathlib import Path
from typing import BinaryIO, Literal, Union

from core.classifier import classify, load_ranges, lookup_optimal
from core.lab_reference import lab_reference_interval_kind
from core.converter import load_converter
from core.data_paths import get_default_runtime_data_dir
from core.exceptions import ScannedPdfNotSupportedError
from core.extractor import extract_text_from_pdf
from core.pdf_validation import get_max_pdf_pages, validate_pdf_bytes, validate_pdf_path
from core.llm_biomarker_match import batch_resolve_biomarker_ids, llm_fallback_enabled
from core.llm_parser import parse_lab_report
from core.matcher import load_matcher
from core.schemas import (
    Classification,
    ClassifiedMarker,
    ExtractionResult,
    PipelineResult,
)

logger = logging.getLogger(__name__)


def _resolve_age(
    report_date: date,
    *,
    effective_dob: date | None,
    age_at_report: int | None,
) -> int:
    """Compute age for classification: prefer DOB + report date, else PDF age_at_report."""
    if effective_dob is not None:
        return max(0, (report_date - effective_dob).days // 365)
    if age_at_report is not None:
        return age_at_report
    raise ValueError(
        "Cannot determine patient age: no date of birth (PDF or override) and no age_at_report "
        "from extraction. Provide --dob or enter date of birth in the app, or ensure the lab "
        "report includes age or date of birth.",
    )


def _merge_patient_context(
    extraction: ExtractionResult,
    sex: str | None,
    date_of_birth: date | None,
    report_date: date,
) -> tuple[
    str,
    date | None,
    int,
    Literal["extracted", "override"],
    Literal["extracted", "override", "none"],
]:
    """Merge CLI/UI overrides with PDF extraction; return effective sex, DOB, age, and sources."""
    p = extraction.patient

    if sex is not None:
        effective_sex = sex
        sex_source: Literal["extracted", "override"] = "override"
    elif p.sex is not None:
        effective_sex = p.sex
        sex_source = "extracted"
    else:
        raise ValueError(
            "Patient sex is unknown: not in the lab report extraction and no --sex / UI override.",
        )

    if date_of_birth is not None:
        effective_dob = date_of_birth
        dob_source: Literal["extracted", "override", "none"] = "override"
    elif p.date_of_birth is not None:
        effective_dob = p.date_of_birth
        dob_source = "extracted"
    else:
        effective_dob = None
        dob_source = "none"

    age = _resolve_age(
        report_date,
        effective_dob=effective_dob,
        age_at_report=p.age_at_report,
    )

    return effective_sex, effective_dob, age, sex_source, dob_source


def _load_biomarkers_index(path: str | Path | None = None) -> dict[str, dict]:
    """Return ``{biomarker_id: {...}}`` from biomarkers.json."""
    p = Path(path) if path is not None else (get_default_runtime_data_dir() / "biomarkers.json")
    if not p.exists():
        logger.warning("biomarkers.json not found at %s", p)
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {b["id"]: b for b in data.get("biomarkers", [])}


def process_report(
    pdf_file: Union[str, BinaryIO],
    sex: str | None = None,
    date_of_birth: date | None = None,
    *,
    data_dir: str | Path | None = None,
) -> PipelineResult:
    """Run the full pipeline on a lab-report PDF.

    Parameters
    ----------
    pdf_file:
        Path or file-like object of the uploaded PDF.
    sex:
        ``"male"`` or ``"female"``. If omitted, uses sex from PDF extraction.
    date_of_birth:
        Patient date of birth. If omitted, uses DOB from extraction when present;
        age may also come from ``age_at_report`` in the PDF.
    data_dir:
        Directory containing the JSON data files. If omitted, uses
        ``BLOODSCOPE_DATA_DIR`` or ``data`` (see :func:`core.data_paths.get_default_runtime_data_dir`).

    Returns
    -------
    PipelineResult
        Contains ``patient`` (extracted), ``effective_*`` fields used for classification,
        and ``classified`` / ``unclassified`` marker lists.
    """
    data_dir = Path(data_dir) if data_dir is not None else get_default_runtime_data_dir()
    max_pages = get_max_pdf_pages()

    # 1. Validate size/pages and extract text
    logger.info("Extracting text from PDF ...")
    if isinstance(pdf_file, (str, Path)):
        validate_pdf_path(pdf_file)
        text = extract_text_from_pdf(pdf_file, max_pages=max_pages)
    else:
        raw = pdf_file.read()
        if hasattr(pdf_file, "seek"):
            pdf_file.seek(0)
        validate_pdf_bytes(raw)
        text = extract_text_from_pdf(io.BytesIO(raw), max_pages=max_pages)

    if not text.strip():
        raise ScannedPdfNotSupportedError(
            "No text could be extracted from this PDF. BloodScope only supports text-based "
            "lab reports (not scanned or image-only PDFs). Export or save your report as a "
            "searchable PDF and try again.",
        )

    # 2. LLM extraction
    logger.info("Sending text to LLM for structured extraction ...")
    extraction: ExtractionResult = parse_lab_report(text)

    # 3. Merge overrides with extraction (sex, DOB, age for classification)
    report_date = extraction.patient.report_date or date.today()
    effective_sex, effective_dob, age, sex_src, dob_src = _merge_patient_context(
        extraction, sex, date_of_birth, report_date,
    )

    language = extraction.patient.source_language

    # 4. Load reference data
    matcher = load_matcher(data_dir / "translations.json")
    converter = load_converter(data_dir / "unit_conversions.json")
    ranges_df = load_ranges(data_dir / "ranges.json")
    biomarkers_index = _load_biomarkers_index(data_dir / "biomarkers.json")

    classified: list[ClassifiedMarker] = []
    unclassified: list[ClassifiedMarker] = []

    resolved_ids: list[str | None] = []
    for marker in extraction.markers:
        bid = matcher.match(marker.original_name, language)
        resolved_ids.append(bid)

    if llm_fallback_enabled():
        failed_idx = [
            i for i, bid in enumerate(resolved_ids)
            if bid is None or bid not in biomarkers_index
        ]
        if failed_idx:
            names = [extraction.markers[i].original_name for i in failed_idx]
            name_map = batch_resolve_biomarker_ids(
                names, language, biomarkers_index,
            )
            for i in failed_idx:
                new_bid = name_map.get(extraction.markers[i].original_name)
                if new_bid and new_bid in biomarkers_index:
                    resolved_ids[i] = new_bid

    for i, marker in enumerate(extraction.markers):
        biomarker_id = resolved_ids[i]

        if biomarker_id is None or biomarker_id not in biomarkers_index:
            unclassified.append(ClassifiedMarker(
                value=marker.value,
                unit=marker.unit,
                original_name=marker.original_name,
                original_unit=marker.unit,
                value_modifier=marker.value_modifier,
                flagged=marker.flagged,
            ))
            continue

        bio_def = biomarkers_index[biomarker_id]
        standard_unit = bio_def["standard_unit"]

        # 6. Convert units
        conversion_method = None
        try:
            converted, conversion_method = converter.convert(
                biomarker_id, marker.value, marker.unit, standard_unit,
            )
        except ValueError:
            logger.warning(
                "Could not convert %s from %s to %s — using original value",
                biomarker_id, marker.unit, standard_unit,
            )
            converted = marker.value
            standard_unit = marker.unit

        decimal_places = bio_def.get("decimal_places", 2)
        converted = round(converted, decimal_places)

        # 7. Convert lab reference ranges to standard unit (same conversion as value)
        lab_low = marker.reference_low
        lab_high = marker.reference_high
        if lab_low is not None and marker.unit != standard_unit:
            try:
                lab_low, _ = converter.convert(biomarker_id, lab_low, marker.unit, standard_unit)
                lab_low = round(lab_low, decimal_places)
            except ValueError:
                lab_low = None
        if lab_high is not None and marker.unit != standard_unit:
            try:
                lab_high, _ = converter.convert(biomarker_id, lab_high, marker.unit, standard_unit)
                lab_high = round(lab_high, decimal_places)
            except ValueError:
                lab_high = None

        # 8. Classify (hybrid: lab ranges for normal, ranges.json for optimal)
        cls = classify(
            converted, biomarker_id, effective_sex, age, ranges_df,
            lab_low=lab_low, lab_high=lab_high,
        )

        opt_lo, opt_hi = lookup_optimal(biomarker_id, effective_sex, age, ranges_df)

        classified.append(ClassifiedMarker(
            biomarker_id=biomarker_id,
            en_name=bio_def["en_name"],
            category=bio_def["category"],
            value=marker.value,
            converted_value=converted,
            unit=marker.unit,
            standard_unit=standard_unit,
            conversion_method=conversion_method,
            classification=cls,
            lab_reference_low=lab_low,
            lab_reference_high=lab_high,
            lab_reference_interval_kind=lab_reference_interval_kind(lab_low, lab_high),
            optimal_low=opt_lo,
            optimal_high=opt_hi,
            original_name=marker.original_name,
            original_unit=marker.unit,
            value_modifier=marker.value_modifier,
            flagged=marker.flagged,
        ))

    logger.info(
        "Pipeline complete: %d classified, %d unclassified",
        len(classified), len(unclassified),
    )
    return PipelineResult(
        patient=extraction.patient,
        classified=classified,
        unclassified=unclassified,
        effective_sex=effective_sex,
        effective_date_of_birth=effective_dob,
        effective_age=age,
        sex_source=sex_src,
        date_of_birth_source=dob_src,
    )
