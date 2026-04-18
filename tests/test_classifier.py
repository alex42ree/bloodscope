"""Tests for core.classifier – hybrid classification logic."""

from __future__ import annotations

import pandas as pd
import pytest

from core.classifier import Classification, classify, lookup_optimal


def _make_ranges_df(rows: list[dict] | None = None) -> pd.DataFrame:
    cols = [
        "biomarker_id", "sex", "age_min", "age_max", "unit",
        "optimal_low", "optimal_high", "source", "source_url",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


HEMOGLOBIN_MALE = {
    "biomarker_id": "hemoglobin",
    "sex": "male",
    "age_min": 18,
    "age_max": 120,
    "unit": "g/dL",
    "optimal_low": 14.0,
    "optimal_high": 16.0,
    "source": "FMU",
    "source_url": None,
}

HEMOGLOBIN_ANY = {
    **HEMOGLOBIN_MALE,
    "sex": "any",
    "optimal_low": 13.5,
    "optimal_high": 15.5,
}


class TestClassifyWithLabAndOptimal:
    """Lab ranges + optimal ranges both present."""

    def setup_method(self):
        self.df = _make_ranges_df([HEMOGLOBIN_MALE])

    def test_optimal(self):
        result = classify(
            15.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.OPTIMAL

    def test_normal_above_optimal(self):
        result = classify(
            17.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL

    def test_normal_below_optimal(self):
        result = classify(
            13.5, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL

    def test_out_of_range_high(self):
        result = classify(
            18.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.OUT_OF_RANGE

    def test_out_of_range_low(self):
        result = classify(
            12.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.OUT_OF_RANGE

    def test_at_lab_boundary_low(self):
        result = classify(
            13.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL

    def test_at_optimal_boundary(self):
        result = classify(
            14.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.OPTIMAL


class TestClassifyLabOnlyNoOptimal:
    """Lab ranges present but no optimal ranges in ranges.json."""

    def setup_method(self):
        self.df = _make_ranges_df()

    def test_normal(self):
        result = classify(
            15.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL

    def test_out_of_range(self):
        result = classify(
            18.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.OUT_OF_RANGE


class TestClassifyNoLabRanges:
    """No lab ranges extracted from the PDF."""

    def setup_method(self):
        self.df = _make_ranges_df([HEMOGLOBIN_MALE])

    def test_unknown_without_lab_ranges(self):
        result = classify(15.0, "hemoglobin", "male", 35, self.df)
        assert result == Classification.UNKNOWN

    def test_unknown_with_none_lab(self):
        result = classify(
            15.0, "hemoglobin", "male", 35, self.df,
            lab_low=None, lab_high=None,
        )
        assert result == Classification.UNKNOWN

    def test_lower_only_lab_normal(self):
        """Ref like ``> 40`` (HDL): only lower bound from extraction."""
        result = classify(
            49.0, "hdl_cholesterol", "male", 35, self.df,
            lab_low=40.0, lab_high=None,
        )
        assert result == Classification.NORMAL

    def test_lower_only_out_of_range(self):
        result = classify(
            35.0, "hdl_cholesterol", "male", 35, self.df,
            lab_low=40.0, lab_high=None,
        )
        assert result == Classification.OUT_OF_RANGE

    def test_upper_only_lab_normal(self):
        """Ref like ``< 5.7`` (HbA1c %): only upper bound from extraction."""
        result = classify(
            4.7, "hba1c", "male", 35, self.df,
            lab_low=None, lab_high=5.7,
        )
        assert result == Classification.NORMAL

    def test_upper_only_out_of_range(self):
        result = classify(
            6.0, "hba1c", "male", 35, self.df,
            lab_low=None, lab_high=5.7,
        )
        assert result == Classification.OUT_OF_RANGE

    def test_upper_only_at_boundary_inclusive(self):
        result = classify(
            5.7, "hba1c", "male", 35, self.df,
            lab_low=None, lab_high=5.7,
        )
        assert result == Classification.NORMAL


class TestClassifyOneSidedWithOptimal:
    """One-sided lab refs combined with optimal band from ranges.json."""

    def test_upper_only_optimal_inside(self):
        df = _make_ranges_df([{
            "biomarker_id": "hba1c",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "%",
            "optimal_low": 4.0,
            "optimal_high": 5.0,
            "source": "test",
            "source_url": None,
        }])
        result = classify(
            4.5, "hba1c", "male", 35, df,
            lab_low=None, lab_high=5.7,
        )
        assert result == Classification.OPTIMAL

    def test_upper_only_optimal_discarded_when_spans_above_lab_cap(self):
        df = _make_ranges_df([{
            "biomarker_id": "hba1c",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "%",
            "optimal_low": 4.0,
            "optimal_high": 6.0,
            "source": "test",
            "source_url": None,
        }])
        result = classify(
            5.0, "hba1c", "male", 35, df,
            lab_low=None, lab_high=5.7,
        )
        assert result == Classification.NORMAL

    def test_lower_only_optimal_inside(self):
        df = _make_ranges_df([{
            "biomarker_id": "hdl_cholesterol",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "mg/dL",
            "optimal_low": 50.0,
            "optimal_high": 60.0,
            "source": "test",
            "source_url": None,
        }])
        result = classify(
            55.0, "hdl_cholesterol", "male", 35, df,
            lab_low=40.0, lab_high=None,
        )
        assert result == Classification.OPTIMAL

    def test_lower_only_optimal_discarded_when_extends_below_lab_floor(self):
        df = _make_ranges_df([{
            "biomarker_id": "hdl_cholesterol",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "mg/dL",
            "optimal_low": 35.0,
            "optimal_high": 55.0,
            "source": "test",
            "source_url": None,
        }])
        result = classify(
            50.0, "hdl_cholesterol", "male", 35, df,
            lab_low=40.0, lab_high=None,
        )
        assert result == Classification.NORMAL


class TestContainmentCheck:
    """Optimal range outside lab normal range should be discarded."""

    def setup_method(self):
        bad_optimal = {
            "biomarker_id": "hemoglobin",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "g/dL",
            "optimal_low": 12.0,
            "optimal_high": 18.0,
            "source": "bad",
            "source_url": None,
        }
        self.df = _make_ranges_df([bad_optimal])

    def test_discards_optimal_outside_normal(self):
        result = classify(
            15.0, "hemoglobin", "male", 35, self.df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL

    def test_partial_overlap_low(self):
        partial = {
            "biomarker_id": "hemoglobin",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "g/dL",
            "optimal_low": 12.0,
            "optimal_high": 16.0,
            "source": "bad",
            "source_url": None,
        }
        df = _make_ranges_df([partial])
        result = classify(
            15.0, "hemoglobin", "male", 35, df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL

    def test_partial_overlap_high(self):
        partial = {
            "biomarker_id": "hemoglobin",
            "sex": "male",
            "age_min": 18,
            "age_max": 120,
            "unit": "g/dL",
            "optimal_low": 14.0,
            "optimal_high": 18.0,
            "source": "bad",
            "source_url": None,
        }
        df = _make_ranges_df([partial])
        result = classify(
            15.0, "hemoglobin", "male", 35, df,
            lab_low=13.0, lab_high=17.5,
        )
        assert result == Classification.NORMAL


class TestPlausibilityChecks:
    """Implausible ranges (low >= high) are rejected."""

    def setup_method(self):
        self.df = _make_ranges_df([HEMOGLOBIN_MALE])

    def test_lab_low_equals_high(self):
        result = classify(
            15.0, "hemoglobin", "male", 35, self.df,
            lab_low=15.0, lab_high=15.0,
        )
        assert result == Classification.UNKNOWN

    def test_lab_low_greater_than_high(self):
        result = classify(
            15.0, "hemoglobin", "male", 35, self.df,
            lab_low=17.5, lab_high=13.0,
        )
        assert result == Classification.UNKNOWN


class TestSexAndAgeMatching:
    """Verify sex-specific and age-bucket matching."""

    def test_sex_specific_preferred(self):
        df = _make_ranges_df([HEMOGLOBIN_MALE, HEMOGLOBIN_ANY])
        opt_lo, opt_hi = lookup_optimal("hemoglobin", "male", 35, df)
        assert opt_lo == 14.0
        assert opt_hi == 16.0

    def test_falls_back_to_any(self):
        df = _make_ranges_df([HEMOGLOBIN_ANY])
        opt_lo, opt_hi = lookup_optimal("hemoglobin", "male", 35, df)
        assert opt_lo == 13.5
        assert opt_hi == 15.5

    def test_age_out_of_range(self):
        age_limited = {
            **HEMOGLOBIN_MALE,
            "age_min": 30,
            "age_max": 34,
        }
        df = _make_ranges_df([age_limited])
        opt_lo, opt_hi = lookup_optimal("hemoglobin", "male", 40, df)
        assert opt_lo is None
        assert opt_hi is None

    def test_no_match_for_biomarker(self):
        df = _make_ranges_df([HEMOGLOBIN_MALE])
        opt_lo, opt_hi = lookup_optimal("glucose", "male", 35, df)
        assert opt_lo is None
        assert opt_hi is None

    def test_empty_df(self):
        df = _make_ranges_df()
        opt_lo, opt_hi = lookup_optimal("hemoglobin", "male", 35, df)
        assert opt_lo is None
        assert opt_hi is None
