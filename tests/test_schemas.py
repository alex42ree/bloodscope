"""Tests for core.schemas – Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.schemas import (
    Biomarker,
    ClassifiedMarker,
    Classification,
    ExtractedMarker,
    ExtractionResult,
    OptimalRange,
    PatientInfo,
    PipelineResult,
    Translation,
    TranslationVariant,
    UnitConversion,
)


class TestPatientInfo:
    def test_valid_male(self):
        p = PatientInfo(sex="male", source_language="en")
        assert p.sex == "male"

    def test_sex_optional_omitted(self):
        p = PatientInfo(source_language="en")
        assert p.sex is None

    def test_valid_female_with_dates(self):
        p = PatientInfo(
            sex="female", source_language="de",
            date_of_birth="1990-01-15", report_date="2024-06-01",
            age_at_report=34, lab_name="Synlab",
        )
        assert p.age_at_report == 34

    def test_invalid_sex(self):
        with pytest.raises(ValidationError):
            PatientInfo(sex="other", source_language="en")

    def test_invalid_language(self):
        with pytest.raises(ValidationError):
            PatientInfo(sex="male", source_language="it")

    def test_negative_age(self):
        with pytest.raises(ValidationError):
            PatientInfo(sex="male", source_language="en", age_at_report=-1)


class TestExtractedMarker:
    def test_basic_marker(self):
        m = ExtractedMarker(
            original_name="Hemoglobina", value=14.5, unit="g/dL",
        )
        assert m.flagged is False
        assert m.value_modifier is None

    def test_with_modifier(self):
        m = ExtractedMarker(
            original_name="CRP", value=0.2, unit="mg/L",
            value_modifier="<",
        )
        assert m.value_modifier == "<"

    def test_invalid_modifier(self):
        with pytest.raises(ValidationError):
            ExtractedMarker(
                original_name="CRP", value=0.2, unit="mg/L",
                value_modifier="<=",
            )

    def test_with_reference_range(self):
        m = ExtractedMarker(
            original_name="Glucosa", value=99, unit="mg/dL",
            reference_low=70, reference_high=110, flagged=False,
        )
        assert m.reference_low == 70

    def test_flagged(self):
        m = ExtractedMarker(
            original_name="Colesterol total", value=250, unit="mg/dL",
            flagged=True,
        )
        assert m.flagged is True


class TestExtractionResult:
    def test_valid_result(self):
        r = ExtractionResult(
            patient=PatientInfo(sex="male", source_language="es"),
            markers=[
                ExtractedMarker(original_name="Hemoglobina", value=14.5, unit="g/dL"),
            ],
        )
        assert len(r.markers) == 1

    def test_empty_markers(self):
        r = ExtractionResult(
            patient=PatientInfo(sex="female", source_language="en"),
            markers=[],
        )
        assert len(r.markers) == 0


class TestBiomarker:
    def test_valid(self):
        b = Biomarker(
            id="rbc", loinc_code="26453-1", en_name="Red Blood Cells (RBC)",
            category="hematology", standard_unit="x10^6/µL",
        )
        assert b.decimal_places == 2
        assert b.loinc_ucum_unit is None
        assert b.loinc_example_units == []
        assert b.loinc_property is None

    def test_with_loinc_metadata(self):
        b = Biomarker(
            id="rbc", loinc_code="26453-1", en_name="Red Blood Cells (RBC)",
            category="hematology", standard_unit="x10^6/µL",
            loinc_ucum_unit="10*6/uL",
            loinc_example_units=["10*12/L"],
            loinc_property="NCnc",
        )
        assert b.loinc_ucum_unit == "10*6/uL"
        assert b.loinc_example_units == ["10*12/L"]
        assert b.loinc_property == "NCnc"

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            Biomarker(id="rbc", loinc_code="26453-1")


class TestTranslationVariant:
    def test_valid(self):
        v = TranslationVariant(term="hematíes", source="loinc")
        assert v.lab is None

    def test_with_lab(self):
        v = TranslationVariant(term="eritrocitos", source="lab_pdf", lab="eurofins")
        assert v.lab == "eurofins"

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            TranslationVariant(term="test", source="invalid_source")

    def test_default_source(self):
        v = TranslationVariant(term="test")
        assert v.source == "manual"


class TestTranslation:
    def test_valid(self):
        t = Translation(
            biomarker_id="rbc", language="es",
            variants=[
                TranslationVariant(term="hematíes", source="loinc"),
                TranslationVariant(term="eritrocitos", source="loinc"),
            ],
        )
        assert len(t.variants) == 2

    def test_empty_variants(self):
        with pytest.raises(ValidationError):
            Translation(biomarker_id="rbc", language="es", variants=[])


class TestOptimalRange:
    def test_valid(self):
        r = OptimalRange(
            biomarker_id="hemoglobin", sex="male", age_min=18, age_max=120,
            unit="g/dL", optimal_low=14.0, optimal_high=16.0,
        )
        assert r.source == ""
        assert r.source_url is None

    def test_with_source_url(self):
        r = OptimalRange(
            biomarker_id="glucose", sex="any", age_min=18, age_max=120,
            unit="mg/dL", optimal_low=80.0, optimal_high=95.0,
            source="FMU Blood Tracking Form",
            source_url="http://example.com/fmu.pdf",
        )
        assert r.source_url == "http://example.com/fmu.pdf"

    def test_negative_age(self):
        with pytest.raises(ValidationError):
            OptimalRange(
                biomarker_id="hemoglobin", sex="male", age_min=-1, age_max=120,
                unit="g/dL", optimal_low=14.0, optimal_high=16.0,
            )


class TestUnitConversion:
    def test_valid(self):
        c = UnitConversion(
            biomarker_id="glucose", from_unit="mmol/L", to_unit="mg/dL", factor=18.0182,
        )
        assert c.factor == 18.0182
        assert c.bidirectional is True
        assert c.molecular_weight is None

    def test_with_metadata(self):
        c = UnitConversion(
            biomarker_id="glucose", from_unit="mmol/L", to_unit="mg/dL",
            factor=18.016, molecular_weight=180.16,
            molecular_weight_source="PubChem CID 5793",
            source="NLM UCUM API (molar)",
            source_url="https://ucum.nlm.nih.gov/ucum-service/v1/ucumtransform/1.0/from/mmol%2FL/to/mg%2FdL/MOLWEIGHT/180.16",
            bidirectional=True,
            generated_at="2026-04-15T17:09:00Z",
        )
        assert c.molecular_weight == 180.16
        assert c.molecular_weight_source == "PubChem CID 5793"
        assert c.source == "NLM UCUM API (molar)"
        assert c.source_url is not None
        assert c.generated_at == "2026-04-15T17:09:00Z"

    def test_negative_factor(self):
        with pytest.raises(ValidationError):
            UnitConversion(
                biomarker_id="glucose", from_unit="mmol/L", to_unit="mg/dL", factor=-1.0,
            )

    def test_zero_factor(self):
        with pytest.raises(ValidationError):
            UnitConversion(
                biomarker_id="glucose", from_unit="mmol/L", to_unit="mg/dL", factor=0,
            )


class TestClassifiedMarker:
    def test_defaults(self):
        m = ClassifiedMarker(
            value=14.5, unit="g/dL",
            original_name="Hemoglobina", original_unit="g/dL",
        )
        assert m.classification == Classification.UNKNOWN
        assert m.biomarker_id is None
        assert m.lab_reference_low is None
        assert m.lab_reference_high is None

    def test_full(self):
        m = ClassifiedMarker(
            biomarker_id="hemoglobin", en_name="Hemoglobin",
            category="hematology", value=14.5, converted_value=14.5,
            unit="g/dL", standard_unit="g/dL",
            classification=Classification.OPTIMAL,
            lab_reference_low=13.0, lab_reference_high=17.5,
            optimal_low=14.0, optimal_high=16.0,
            original_name="Hemoglobina", original_unit="g/dL",
        )
        assert m.classification == Classification.OPTIMAL
        assert m.lab_reference_low == 13.0
        assert m.lab_reference_high == 17.5


class TestPipelineResult:
    def test_with_effective_fields(self):
        r = PipelineResult(
            patient=PatientInfo(sex="male", source_language="es"),
            effective_sex="male",
            effective_date_of_birth=None,
            effective_age=45,
            sex_source="extracted",
            date_of_birth_source="none",
        )
        assert r.effective_age == 45
        assert r.sex_source == "extracted"
