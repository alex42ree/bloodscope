"""BloodScope – Streamlit application."""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data_paths import get_default_runtime_data_dir
from core.exceptions import BloodScopePdfError
from core.pdf_validation import get_max_pdf_bytes, get_max_pdf_pages, validate_pdf_bytes
from core.lab_reference import format_lab_reference_for_display
from core.measurement_display import format_measurement_display
from core.pipeline import process_report
from core.schemas import Classification, PipelineResult

load_dotenv()

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

STATUS_DISPLAY = {
    Classification.OPTIMAL: ("Optimal", "🟢", "background-color: #d4edda"),
    Classification.NORMAL: ("Normal", "🟡", "background-color: #fff3cd"),
    Classification.OUT_OF_RANGE: ("Out of Range", "🔴", "background-color: #f8d7da"),
    Classification.UNKNOWN: ("Unknown", "⚪", ""),
}

_PIPELINE_LOGGERS = (
    "core.extractor",
    "core.pipeline",
    "core.matcher",
    "core.llm_biomarker_match",
    "core.llm_parser",
    "core.converter",
    "core.classifier",
)


def _dataframe_height_px(num_rows: int, row_height_px: int = 36, header_px: int = 52) -> int:
    """Tall enough to show all rows without a cramped inner scroll (default viewport is short)."""
    if num_rows <= 0:
        return 80
    return min(header_px + num_rows * row_height_px, 20000)


def _render_results_table(markers: list, title: str) -> None:
    """Render a category table of classified markers."""
    if not markers:
        return

    st.subheader(title)

    rows = []
    for m in markers:
        label, icon, _ = STATUS_DISPLAY[m.classification]
        rows.append({
            "Biomarker": m.en_name or m.original_name,
            "Value": format_measurement_display(
                m.value, m.converted_value, m.value_modifier, use_converted=True,
            ),
            "Unit": m.standard_unit or m.unit,
            "Reference": format_lab_reference_for_display(
                m.lab_reference_low, m.lab_reference_high,
            ),
            "Optimal": format_lab_reference_for_display(m.optimal_low, m.optimal_high),
            "Status": f"{icon} {label}",
        })

    df = pd.DataFrame(rows)

    def _color_status(val: str) -> str:
        if "Optimal" in str(val):
            return "background-color: #d4edda; color: #155724"
        if "Normal" in str(val):
            return "background-color: #fff3cd; color: #856404"
        if "Out of Range" in str(val):
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = df.style.map(_color_status, subset=["Status"])
    st.dataframe(
        styled,
        width="stretch",
        height=_dataframe_height_px(len(df)),
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="BloodScope", page_icon="🔬", layout="wide")
    st.title("🔬 BloodScope")
    st.markdown("Upload a lab report PDF to analyze your biomarkers.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Upload & Patient Info")
        pdf_file = st.file_uploader(
            "Lab Report (PDF)",
            type=["pdf"],
            accept_multiple_files=False,
        )
        st.caption(
            "When you choose “From PDF (auto)” for sex or date of birth, those values are taken from the lab report. "
            "Choose Male/Female or Enter manually for date of birth only when you need to override the PDF.",
        )
        st.caption(
            f"**Supported PDFs:** text-based (searchable) lab reports only — not scanned or image-only PDFs. "
            f"Max size {get_max_pdf_bytes() / (1024 * 1024):.0f} MB, max {get_max_pdf_pages()} pages.",
        )
        sex_choice = st.radio(
            "Sex",
            ["From PDF (auto)", "Male", "Female"],
            index=0,
        )
        sex_override = None
        if sex_choice == "Male":
            sex_override = "male"
        elif sex_choice == "Female":
            sex_override = "female"

        dob_choice = st.radio(
            "Date of Birth",
            ["From PDF (auto)", "Enter manually"],
            index=0,
        )
        dob_override = None
        if dob_choice == "Enter manually":
            dob_override = st.date_input(
                "Date of birth (required for classification)",
                value=date(1990, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today() - timedelta(days=365),
            )
            st.caption("A date of birth is required when entering manually so age-based ranges can be applied.")
        analyze = st.button("Analyze", type="primary", width="stretch")

    if not pdf_file:
        st.info("Upload a lab report PDF in the sidebar to get started.")
        return

    if not analyze:
        return

    pdf_bytes = pdf_file.getvalue()
    try:
        validate_pdf_bytes(pdf_bytes)
    except BloodScopePdfError as exc:
        st.error(str(exc))
        return

    # --- Processing ---
    log_buffer = io.StringIO()
    log_handler = logging.StreamHandler(log_buffer)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    pipeline_loggers = [logging.getLogger(name) for name in _PIPELINE_LOGGERS]
    result: PipelineResult | None = None
    pipeline_error: Exception | None = None
    with st.spinner("Extracting and analyzing your lab report..."):
        try:
            for lg in pipeline_loggers:
                lg.addHandler(log_handler)
            result = process_report(
                io.BytesIO(pdf_bytes), sex_override, dob_override, data_dir=get_default_runtime_data_dir(),
            )
        except Exception as e:
            pipeline_error = e
        finally:
            for lg in pipeline_loggers:
                lg.removeHandler(log_handler)

    analysis_log = log_buffer.getvalue().strip() or "(no log lines captured)"

    if pipeline_error is not None:
        with st.expander("Analysis log", expanded=True):
            st.code(analysis_log, language="text")
        st.error(f"Error processing report: {pipeline_error}")
        return

    assert result is not None

    # --- Summary ---
    col1, col2, col3, col4 = st.columns(4)
    total = len(result.classified)
    optimal = sum(1 for m in result.classified if m.classification == Classification.OPTIMAL)
    normal = sum(1 for m in result.classified if m.classification == Classification.NORMAL)
    oor = sum(1 for m in result.classified if m.classification == Classification.OUT_OF_RANGE)

    col1.metric("Total Biomarkers", total)
    col2.metric("Optimal", optimal)
    col3.metric("Normal", normal)
    col4.metric("Out of Range", oor)

    st.divider()

    with st.expander("Analysis log", expanded=False):
        st.code(analysis_log, language="text")

    with st.expander("Patient context (extracted vs used for analysis)", expanded=False):
        p = result.patient
        st.markdown(
            f"| Field | From PDF extraction | Used for classification |\n"
            f"|-------|---------------------|-------------------------|\n"
            f"| Sex | {p.sex or '—'} | **{result.effective_sex}** ({result.sex_source}) |\n"
            f"| Date of birth | {p.date_of_birth or '—'} | "
            f"**{result.effective_date_of_birth or '—'}** ({result.date_of_birth_source}) |\n"
            f"| Age (for ranges) | — | **{result.effective_age}** |",
        )

    st.divider()

    # --- Grouped results ---
    categories_seen: dict[str, list] = {}
    for m in result.classified:
        cat = m.category or "other"
        if cat not in categories_seen:
            categories_seen[cat] = []
        categories_seen[cat].append(m)

    category_order = list(CATEGORY_LABELS.keys())
    for cat in category_order:
        if cat in categories_seen:
            _render_results_table(categories_seen[cat], CATEGORY_LABELS.get(cat, cat.title()))

    for cat, markers in categories_seen.items():
        if cat not in category_order:
            _render_results_table(markers, CATEGORY_LABELS.get(cat, cat.title()))

    # --- Unclassified ---
    if result.unclassified:
        st.divider()
        st.subheader("Unclassified Biomarkers")
        st.caption("These biomarkers could not be matched to the database.")
        rows = []
        for m in result.unclassified:
            rows.append({
                "Original Name": m.original_name,
                "Value": format_measurement_display(
                    m.value, m.converted_value, m.value_modifier, use_converted=True,
                ),
                "Unit": m.unit,
            })
        udf = pd.DataFrame(rows)
        st.dataframe(
            udf,
            width="stretch",
            height=_dataframe_height_px(len(udf)),
            hide_index=True,
        )

    # --- Disclaimer (core only; optimal-range data sources are in README) ---
    st.divider()
    st.caption(
        "**Disclaimer:** BloodScope is not a medical device. This analysis does not replace "
        "professional medical advice. "
        "“Optimal” bands are narrower than typical lab reference intervals and are not a "
        "universally accepted clinical standard; how those ranges are sourced is described in "
        "the project README. "
        "Report content is sent to the Anthropic API for extraction and analysis."
    )


if __name__ == "__main__":
    main()
