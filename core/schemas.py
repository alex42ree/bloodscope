from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.lab_reference import LabReferenceIntervalKind


# --- Enums ---

class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    ANY = "any"


class Classification(str, Enum):
    OPTIMAL = "optimal"
    NORMAL = "normal"
    OUT_OF_RANGE = "out_of_range"
    UNKNOWN = "unknown"


# --- LLM extraction output schemas ---

class PatientInfo(BaseModel):
    sex: Optional[str] = Field(None, pattern=r"^(male|female)$")
    date_of_birth: Optional[date] = None
    age_at_report: Optional[int] = Field(None, ge=0, le=150)
    report_date: Optional[date] = None
    lab_name: Optional[str] = None
    source_language: str = Field(..., pattern=r"^(es|de|en|fr)$")


class ExtractedMarker(BaseModel):
    original_name: str
    value: float
    value_modifier: Optional[str] = Field(None, pattern=r"^[<>]$")
    unit: str
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    flagged: bool = False


class ExtractionResult(BaseModel):
    patient: PatientInfo
    markers: list[ExtractedMarker]


# --- Static data schemas (JSON files) ---

class Biomarker(BaseModel):
    id: str
    loinc_code: str
    en_name: str
    category: str
    standard_unit: str
    description: str = ""
    decimal_places: int = 2
    loinc_ucum_unit: Optional[str] = None
    loinc_example_units: list[str] = Field(default_factory=list)
    loinc_property: Optional[str] = None


class BiomarkersFile(BaseModel):
    biomarkers: list[Biomarker]


class TranslationVariant(BaseModel):
    term: str
    source: str = Field("manual", pattern=r"^(loinc|lab_pdf|llm|manual)$")
    lab: Optional[str] = None


class Translation(BaseModel):
    biomarker_id: str
    language: str = Field(..., pattern=r"^(es|de|en|fr)$")
    variants: list[TranslationVariant] = Field(..., min_length=1)


class TranslationsFile(BaseModel):
    translations: list[Translation]


class OptimalRange(BaseModel):
    biomarker_id: str
    sex: str = Field(..., pattern=r"^(male|female|any)$")
    age_min: int = Field(..., ge=0)
    age_max: int = Field(..., ge=0)
    unit: str
    optimal_low: float
    optimal_high: float
    source: str = ""
    source_url: Optional[str] = None


class RangesFile(BaseModel):
    ranges: list[OptimalRange]


class UnitConversion(BaseModel):
    biomarker_id: str
    from_unit: str
    to_unit: str
    factor: float = Field(..., gt=0)
    molecular_weight: Optional[float] = None
    molecular_weight_source: Optional[str] = None
    source: str = ""
    source_url: Optional[str] = None
    bidirectional: bool = True
    generated_at: Optional[str] = None


class UnitConversionsFile(BaseModel):
    conversions: list[UnitConversion]


# --- Pipeline output schema ---

class ClassifiedMarker(BaseModel):
    biomarker_id: Optional[str] = None
    en_name: Optional[str] = None
    category: Optional[str] = None
    value: float
    converted_value: Optional[float] = None
    unit: str
    standard_unit: Optional[str] = None
    conversion_method: Optional[str] = None
    classification: Classification = Classification.UNKNOWN
    lab_reference_low: Optional[float] = None
    lab_reference_high: Optional[float] = None
    lab_reference_interval_kind: LabReferenceIntervalKind = "none"
    optimal_low: Optional[float] = None
    optimal_high: Optional[float] = None
    original_name: str
    original_unit: str
    value_modifier: Optional[str] = None
    flagged: bool = False


class PipelineResult(BaseModel):
    patient: PatientInfo
    classified: list[ClassifiedMarker] = []
    unclassified: list[ClassifiedMarker] = []
    effective_sex: str = Field(..., pattern=r"^(male|female)$")
    effective_date_of_birth: Optional[date] = None
    effective_age: int = Field(..., ge=0, le=150)
    sex_source: Literal["extracted", "override"]
    date_of_birth_source: Literal["extracted", "override", "none"]
