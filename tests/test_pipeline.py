"""Integration tests for core.pipeline with mocked LLM and PDF extraction."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from core.exceptions import ScannedPdfNotSupportedError
from core.schemas import (
    Classification,
    ExtractionResult,
    ExtractedMarker,
    PatientInfo,
    PipelineResult,
)
from core.pipeline import process_report


def _mock_extraction() -> ExtractionResult:
    return ExtractionResult(
        patient=PatientInfo(
            sex="male",
            source_language="es",
            report_date="2024-06-01",
            age_at_report=34,
        ),
        markers=[
            ExtractedMarker(
                original_name="Hemoglobina",
                value=14.5,
                unit="g/dL",
                reference_low=13.0,
                reference_high=17.5,
                flagged=False,
            ),
            ExtractedMarker(
                original_name="Glucosa en ayunas",
                value=5.5,
                unit="mmol/L",
                reference_low=3.9,
                reference_high=5.8,
                flagged=False,
            ),
            ExtractedMarker(
                original_name="UnknownMarkerXYZ",
                value=42.0,
                unit="U/L",
                flagged=True,
            ),
        ],
    )


class TestPipeline:
    @patch("core.pipeline.validate_pdf_path", return_value=1)
    @patch("core.pipeline.parse_lab_report")
    @patch("core.pipeline.extract_text_from_pdf")
    def test_end_to_end(
        self,
        mock_extract: ...,
        mock_parse: ...,
        _mock_validate_pdf: ...,
        tmp_data_dir: Path,
        sample_biomarkers: list[dict],
        sample_translations: list[dict],
        sample_ranges: list[dict],
        sample_conversions: list[dict],
    ):
        mock_extract.return_value = "some pdf text"
        mock_parse.return_value = _mock_extraction()

        (tmp_data_dir / "biomarkers.json").write_text(
            json.dumps({"biomarkers": sample_biomarkers})
        )
        (tmp_data_dir / "translations.json").write_text(
            json.dumps({"translations": sample_translations}, ensure_ascii=False)
        )
        (tmp_data_dir / "ranges.json").write_text(
            json.dumps({"ranges": sample_ranges})
        )
        (tmp_data_dir / "unit_conversions.json").write_text(
            json.dumps({"conversions": sample_conversions})
        )

        result: PipelineResult = process_report(
            "dummy.pdf", "male", date(1990, 1, 1), data_dir=tmp_data_dir,
        )

        assert result.effective_sex == "male"
        assert result.sex_source == "override"
        assert result.date_of_birth_source == "override"
        assert result.effective_age == 34

        assert len(result.classified) == 2
        assert len(result.unclassified) == 1

        hb = next(m for m in result.classified if m.biomarker_id == "hemoglobin")
        assert hb.classification == Classification.OPTIMAL
        assert hb.converted_value == 14.5
        assert hb.en_name == "Hemoglobin"
        assert hb.conversion_method == "identity"
        assert hb.lab_reference_low == 13.0
        assert hb.lab_reference_high == 17.5

        glucose = next(m for m in result.classified if m.biomarker_id == "glucose_fasting")
        assert glucose.converted_value == round(5.5 * 18.0182, 0)
        assert glucose.conversion_method == "custom"
        assert glucose.classification == Classification.NORMAL
        assert glucose.lab_reference_low is not None
        assert glucose.lab_reference_high is not None

        unknown = result.unclassified[0]
        assert unknown.original_name == "UnknownMarkerXYZ"
        assert unknown.flagged is True

    @patch("core.pipeline.batch_resolve_biomarker_ids")
    @patch("core.pipeline.llm_fallback_enabled")
    @patch("core.pipeline.validate_pdf_path", return_value=1)
    @patch("core.pipeline.parse_lab_report")
    @patch("core.pipeline.extract_text_from_pdf")
    def test_llm_fallback_resolves_unmatched(
        self,
        mock_extract: ...,
        mock_parse: ...,
        _mock_validate_pdf: ...,
        mock_llm_enabled: ...,
        mock_batch_resolve: ...,
        tmp_data_dir: Path,
        sample_biomarkers: list[dict],
        sample_translations: list[dict],
        sample_ranges: list[dict],
        sample_conversions: list[dict],
    ):
        mock_extract.return_value = "some pdf text"
        mock_parse.return_value = _mock_extraction()
        mock_llm_enabled.return_value = True
        mock_batch_resolve.return_value = {"UnknownMarkerXYZ": "creatinine"}

        (tmp_data_dir / "biomarkers.json").write_text(
            json.dumps({"biomarkers": sample_biomarkers})
        )
        (tmp_data_dir / "translations.json").write_text(
            json.dumps({"translations": sample_translations}, ensure_ascii=False)
        )
        (tmp_data_dir / "ranges.json").write_text(
            json.dumps({"ranges": sample_ranges})
        )
        (tmp_data_dir / "unit_conversions.json").write_text(
            json.dumps({"conversions": sample_conversions})
        )

        result: PipelineResult = process_report(
            "dummy.pdf", "male", date(1990, 1, 1), data_dir=tmp_data_dir,
        )

        assert len(result.unclassified) == 0
        assert len(result.classified) == 3
        cr = next(m for m in result.classified if m.biomarker_id == "creatinine")
        assert cr.original_name == "UnknownMarkerXYZ"
        mock_batch_resolve.assert_called_once()

    @patch("core.pipeline.validate_pdf_path", return_value=1)
    @patch("core.pipeline.parse_lab_report")
    @patch("core.pipeline.extract_text_from_pdf")
    def test_empty_text_raises(self, mock_extract, mock_parse, _mock_validate_pdf):
        mock_extract.return_value = ""
        with __import__("pytest").raises(ScannedPdfNotSupportedError, match="No text"):
            process_report("dummy.pdf", "male", date(1990, 1, 1))

    @patch("core.pipeline.validate_pdf_path", return_value=1)
    @patch("core.pipeline.parse_lab_report")
    @patch("core.pipeline.extract_text_from_pdf")
    def test_no_override_uses_extraction(
        self,
        mock_extract: ...,
        mock_parse: ...,
        _mock_validate_pdf: ...,
        tmp_data_dir: Path,
        sample_biomarkers: list[dict],
        sample_translations: list[dict],
        sample_ranges: list[dict],
        sample_conversions: list[dict],
    ):
        mock_extract.return_value = "text"
        mock_parse.return_value = _mock_extraction()

        (tmp_data_dir / "biomarkers.json").write_text(
            json.dumps({"biomarkers": sample_biomarkers})
        )
        (tmp_data_dir / "translations.json").write_text(
            json.dumps({"translations": sample_translations}, ensure_ascii=False)
        )
        (tmp_data_dir / "ranges.json").write_text(
            json.dumps({"ranges": sample_ranges})
        )
        (tmp_data_dir / "unit_conversions.json").write_text(
            json.dumps({"conversions": sample_conversions})
        )

        result = process_report("dummy.pdf", None, None, data_dir=tmp_data_dir)
        assert result.effective_sex == "male"
        assert result.sex_source == "extracted"
        assert result.date_of_birth_source == "none"
        assert result.effective_age == 34

    @patch("core.pipeline.validate_pdf_path", return_value=1)
    @patch("core.pipeline.parse_lab_report")
    @patch("core.pipeline.extract_text_from_pdf")
    def test_sex_override_wins(
        self,
        mock_extract: ...,
        mock_parse: ...,
        _mock_validate_pdf: ...,
        tmp_data_dir: Path,
        sample_biomarkers: list[dict],
        sample_translations: list[dict],
        sample_ranges: list[dict],
        sample_conversions: list[dict],
    ):
        mock_extract.return_value = "text"
        mock_parse.return_value = _mock_extraction()

        (tmp_data_dir / "biomarkers.json").write_text(
            json.dumps({"biomarkers": sample_biomarkers})
        )
        (tmp_data_dir / "translations.json").write_text(
            json.dumps({"translations": sample_translations}, ensure_ascii=False)
        )
        (tmp_data_dir / "ranges.json").write_text(
            json.dumps({"ranges": sample_ranges})
        )
        (tmp_data_dir / "unit_conversions.json").write_text(
            json.dumps({"conversions": sample_conversions})
        )

        result = process_report("dummy.pdf", "female", None, data_dir=tmp_data_dir)
        assert result.effective_sex == "female"
        assert result.sex_source == "override"

    @patch("core.pipeline.validate_pdf_path", return_value=1)
    @patch("core.pipeline.parse_lab_report")
    @patch("core.pipeline.extract_text_from_pdf")
    def test_cannot_resolve_age_raises(
        self,
        mock_extract: ...,
        mock_parse: ...,
        _mock_validate_pdf: ...,
        tmp_data_dir: Path,
        sample_biomarkers: list[dict],
        sample_translations: list[dict],
        sample_ranges: list[dict],
        sample_conversions: list[dict],
    ):
        mock_extract.return_value = "text"
        bad = ExtractionResult(
            patient=PatientInfo(
                sex="male",
                source_language="es",
                report_date=date(2024, 6, 1),
                date_of_birth=None,
                age_at_report=None,
            ),
            markers=[],
        )
        mock_parse.return_value = bad

        (tmp_data_dir / "biomarkers.json").write_text(
            json.dumps({"biomarkers": sample_biomarkers})
        )
        (tmp_data_dir / "translations.json").write_text(
            json.dumps({"translations": sample_translations}, ensure_ascii=False)
        )
        (tmp_data_dir / "ranges.json").write_text(
            json.dumps({"ranges": sample_ranges})
        )
        (tmp_data_dir / "unit_conversions.json").write_text(
            json.dumps({"conversions": sample_conversions})
        )

        with __import__("pytest").raises(ValueError, match="Cannot determine patient age"):
            process_report("dummy.pdf", None, None, data_dir=tmp_data_dir)
