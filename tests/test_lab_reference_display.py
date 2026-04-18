"""Tests for lab reference shape and display formatting."""

from __future__ import annotations

import pandas as pd

from core.lab_reference import (
    format_lab_reference_for_display,
    lab_reference_interval_kind,
)
from core.measurement_display import format_measurement_display


class TestLabReferenceIntervalKind:
    def test_both(self):
        assert lab_reference_interval_kind(13.0, 17.5) == "both"

    def test_upper_only(self):
        assert lab_reference_interval_kind(None, 5.7) == "upper_only"

    def test_lower_only(self):
        assert lab_reference_interval_kind(40.0, None) == "lower_only"

    def test_none_empty(self):
        assert lab_reference_interval_kind(None, None) == "none"

    def test_none_invalid_both(self):
        assert lab_reference_interval_kind(17.5, 13.0) == "none"

    def test_none_equal_bounds(self):
        assert lab_reference_interval_kind(15.0, 15.0) == "none"


class TestFormatLabReferenceForDisplay:
    def test_empty(self):
        assert format_lab_reference_for_display(None, None) == "-"

    def test_invalid_interval(self):
        assert format_lab_reference_for_display(10.0, 5.0) == "—"

    def test_upper_only(self):
        assert format_lab_reference_for_display(None, 5.7) == "< 5.7"

    def test_lower_only(self):
        assert format_lab_reference_for_display(40.0, None) == "> 40.0"

    def test_both(self):
        assert format_lab_reference_for_display(13.0, 17.5) == "13.0 - 17.5"

    def test_zero_low_special_case(self):
        assert format_lab_reference_for_display(0.0, 200.0) == "< 200.0"


class TestFormatMeasurementDisplay:
    def test_plain_float(self):
        assert format_measurement_display(4.7, 4.7, None) == "4.7"

    def test_with_modifier_uses_converted(self):
        assert format_measurement_display(0.2, 0.2, "<") == "<0.2"

    def test_use_raw_value(self):
        assert format_measurement_display(0.2, None, "<", use_converted=False) == "<0.2"

    def test_converted_preferred(self):
        assert format_measurement_display(99.0, 5.5, None) == "5.5"


class TestDataframeValueColumnAllString:
    """Regression: Streamlit Arrow rejects mixed float/str in 'Value'."""

    def test_mixed_markers_produce_string_value_column(self):
        rows = [
            {
                "Value": format_measurement_display(1.0, 1.0, None),
                "Unit": "mg/dL",
            },
            {
                "Value": format_measurement_display(0.2, 0.2, "<"),
                "Unit": "mg/dL",
            },
        ]
        df = pd.DataFrame(rows)
        assert all(isinstance(x, str) for x in df["Value"])
