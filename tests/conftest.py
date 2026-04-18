"""Shared fixtures for BloodScope tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def disable_llm_biomarker_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid real Anthropic calls during tests when ANTHROPIC_API_KEY is set."""
    monkeypatch.setenv("BIOMARKER_LLM_FALLBACK", "0")


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp data directory with minimal JSON files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_biomarkers() -> list[dict]:
    return [
        {
            "id": "hemoglobin",
            "loinc_code": "718-7",
            "en_name": "Hemoglobin",
            "category": "hematology",
            "standard_unit": "g/dL",
            "description": "Hemoglobin mass concentration in blood",
            "decimal_places": 1,
        },
        {
            "id": "total_cholesterol",
            "loinc_code": "2093-3",
            "en_name": "Total Cholesterol",
            "category": "lipid",
            "standard_unit": "mg/dL",
            "description": "Total cholesterol in serum or plasma",
            "decimal_places": 0,
        },
        {
            "id": "glucose_fasting",
            "loinc_code": "1558-6",
            "en_name": "Fasting Glucose",
            "category": "metabolism",
            "standard_unit": "mg/dL",
            "description": "Fasting glucose in blood",
            "decimal_places": 0,
        },
        {
            "id": "creatinine",
            "loinc_code": "2160-0",
            "en_name": "Creatinine",
            "category": "kidney",
            "standard_unit": "mg/dL",
            "description": "Creatinine in serum or plasma",
            "decimal_places": 2,
        },
    ]


def _v(term: str, source: str = "loinc", lab: str | None = None) -> dict:
    """Shorthand to build a translation variant object."""
    return {"term": term, "source": source, "lab": lab}


@pytest.fixture
def sample_translations() -> list[dict]:
    return [
        {"biomarker_id": "hemoglobin", "language": "es", "variants": [_v("hemoglobina", "lab_pdf", "eurofins")]},
        {"biomarker_id": "hemoglobin", "language": "de", "variants": [_v("hämoglobin")]},
        {"biomarker_id": "hemoglobin", "language": "en", "variants": [_v("hemoglobin"), _v("hgb"), _v("hb")]},
        {"biomarker_id": "hemoglobin", "language": "fr", "variants": [_v("hémoglobine")]},
        {"biomarker_id": "total_cholesterol", "language": "es", "variants": [_v("colesterol total")]},
        {"biomarker_id": "total_cholesterol", "language": "de", "variants": [_v("gesamtcholesterin"), _v("cholesterin gesamt")]},
        {"biomarker_id": "total_cholesterol", "language": "en", "variants": [_v("total cholesterol")]},
        {"biomarker_id": "glucose_fasting", "language": "es", "variants": [_v("glucosa en ayunas"), _v("glucosa basal")]},
        {"biomarker_id": "glucose_fasting", "language": "de", "variants": [_v("nüchternglukose"), _v("glukose nüchtern")]},
        {"biomarker_id": "glucose_fasting", "language": "en", "variants": [_v("fasting glucose"), _v("glucose, fasting")]},
        {"biomarker_id": "creatinine", "language": "es", "variants": [_v("creatinina")]},
        {"biomarker_id": "creatinine", "language": "de", "variants": [_v("kreatinin")]},
        {"biomarker_id": "creatinine", "language": "en", "variants": [_v("creatinine")]},
    ]


@pytest.fixture
def sample_ranges() -> list[dict]:
    return [
        {
            "biomarker_id": "hemoglobin", "sex": "male", "age_min": 18, "age_max": 120,
            "unit": "g/dL", "optimal_low": 14.0, "optimal_high": 16.0,
            "source": "FMU Blood Tracking Form", "source_url": None,
        },
        {
            "biomarker_id": "hemoglobin", "sex": "female", "age_min": 18, "age_max": 120,
            "unit": "g/dL", "optimal_low": 12.5, "optimal_high": 15.0,
            "source": "FMU Blood Tracking Form", "source_url": None,
        },
        {
            "biomarker_id": "total_cholesterol", "sex": "any", "age_min": 18, "age_max": 120,
            "unit": "mg/dL", "optimal_low": 150, "optimal_high": 180,
            "source": "FMU Blood Tracking Form", "source_url": None,
        },
        {
            "biomarker_id": "glucose_fasting", "sex": "any", "age_min": 18, "age_max": 120,
            "unit": "mg/dL", "optimal_low": 80, "optimal_high": 95,
            "source": "FMU Blood Tracking Form", "source_url": None,
        },
        {
            "biomarker_id": "creatinine", "sex": "male", "age_min": 18, "age_max": 120,
            "unit": "mg/dL", "optimal_low": 0.8, "optimal_high": 1.1,
            "source": "FMU Blood Tracking Form", "source_url": None,
        },
    ]


@pytest.fixture
def sample_conversions() -> list[dict]:
    return [
        {
            "biomarker_id": "glucose_fasting", "from_unit": "mmol/L", "to_unit": "mg/dL",
            "factor": 18.0182, "molecular_weight": 180.16,
            "source": "AMA SI Conversion Table", "bidirectional": True,
        },
        {
            "biomarker_id": "total_cholesterol", "from_unit": "mmol/L", "to_unit": "mg/dL",
            "factor": 38.67, "molecular_weight": 386.65,
            "source": "AMA SI Conversion Table", "bidirectional": True,
        },
        {
            "biomarker_id": "creatinine", "from_unit": "µmol/L", "to_unit": "mg/dL",
            "factor": 0.01131, "molecular_weight": 113.12,
            "source": "AMA SI Conversion Table", "bidirectional": True,
        },
        {
            "biomarker_id": "hemoglobin", "from_unit": "g/L", "to_unit": "g/dL",
            "factor": 0.1, "molecular_weight": None,
            "source": "dimensional", "bidirectional": True,
        },
    ]


@pytest.fixture
def translations_file(tmp_data_dir: Path, sample_translations: list[dict]) -> Path:
    path = tmp_data_dir / "translations.json"
    path.write_text(json.dumps({"translations": sample_translations}, ensure_ascii=False))
    return path


@pytest.fixture
def conversions_file(tmp_data_dir: Path, sample_conversions: list[dict]) -> Path:
    path = tmp_data_dir / "unit_conversions.json"
    path.write_text(json.dumps({"conversions": sample_conversions}, ensure_ascii=False))
    return path


@pytest.fixture
def ranges_file(tmp_data_dir: Path, sample_ranges: list[dict]) -> Path:
    path = tmp_data_dir / "ranges.json"
    path.write_text(json.dumps({"ranges": sample_ranges}, ensure_ascii=False))
    return path


@pytest.fixture
def biomarkers_file(tmp_data_dir: Path, sample_biomarkers: list[dict]) -> Path:
    path = tmp_data_dir / "biomarkers.json"
    path.write_text(json.dumps({"biomarkers": sample_biomarkers}, ensure_ascii=False))
    return path


@pytest.fixture
def ranges_df(sample_ranges: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(sample_ranges)
