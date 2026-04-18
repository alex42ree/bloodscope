"""Tests for core.converter."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.converter import UnitConverter, normalize_unit


class TestNormalizeUnit:
    def test_alias_cell_count(self):
        assert normalize_unit("x10^6/mm³") == "x10^6/µL"
        assert normalize_unit("10*6/uL") == "x10^6/µL"

    def test_alias_casing(self):
        assert normalize_unit("g/dl") == "g/dL"
        assert normalize_unit("mg/DL") == "mg/dL"

    def test_alias_enzyme(self):
        assert normalize_unit("UI/L") == "U/L"
        assert normalize_unit("u/l") == "U/L"

    def test_alias_thyroid(self):
        assert normalize_unit("uIU/mL") == "µIU/mL"
        assert normalize_unit("mUI/L") == "mIU/L"

    def test_alias_micro(self):
        assert normalize_unit("ug/dL") == "µg/dL"

    def test_passthrough(self):
        assert normalize_unit("mg/dL") == "mg/dL"
        assert normalize_unit("g/dL") == "g/dL"


class TestUnitConverter:
    def test_identity_conversion(self, conversions_file: Path):
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("hemoglobin", 14.5, "g/dL", "g/dL")
        assert val == 14.5
        assert method == "identity"

    def test_custom_map_glucose(self, conversions_file: Path):
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("glucose_fasting", 5.5, "mmol/L", "mg/dL")
        assert pytest.approx(val, rel=0.01) == 5.5 * 18.0182
        assert method == "custom"

    def test_custom_map_cholesterol(self, conversions_file: Path):
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("total_cholesterol", 5.0, "mmol/L", "mg/dL")
        assert pytest.approx(val, rel=0.01) == 5.0 * 38.67
        assert method == "custom"

    def test_custom_map_creatinine(self, conversions_file: Path):
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("creatinine", 88.0, "µmol/L", "mg/dL")
        assert pytest.approx(val, rel=0.01) == 88.0 * 0.01131
        assert method == "custom"

    def test_custom_map_hemoglobin(self, conversions_file: Path):
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("hemoglobin", 140.0, "g/L", "g/dL")
        assert pytest.approx(val, rel=0.01) == 14.0
        assert method == "custom"

    def test_reverse_conversion(self, conversions_file: Path):
        """If only A->B is defined, B->A should work via bidirectional reverse."""
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("hemoglobin", 14.0, "g/dL", "g/L")
        assert pytest.approx(val, rel=0.01) == 140.0
        assert method == "custom_reverse"

    def test_pint_fallback(self, conversions_file: Path):
        """Pint can handle g/L -> g/dL natively."""
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("some_marker", 1.0, "g/L", "g/dL")
        assert pytest.approx(val, rel=0.01) == 0.1
        assert method == "pint"

    def test_no_conversion_raises(self, conversions_file: Path):
        conv = UnitConverter(conversions_file)
        with pytest.raises(ValueError, match="No conversion"):
            conv.convert("hemoglobin", 14.5, "foobar", "bazqux")

    def test_missing_file(self, tmp_path: Path):
        conv = UnitConverter(tmp_path / "nope.json")
        val, method = conv.convert("x", 5.0, "mg/dL", "mg/dL")
        assert val == 5.0
        assert method == "identity"

    def test_unit_alias_resolution(self, conversions_file: Path):
        """Lab uses 'g/dl' (lowercase) but conversion is stored as 'g/dL'."""
        conv = UnitConverter(conversions_file)
        val, method = conv.convert("hemoglobin", 140.0, "g/l", "g/dl")
        assert pytest.approx(val, rel=0.01) == 14.0

    def test_non_bidirectional_blocks_reverse(self, tmp_data_dir: Path):
        """A non-bidirectional entry should not allow reverse conversion."""
        import json
        conv_data = {
            "conversions": [{
                "biomarker_id": "test",
                "from_unit": "A",
                "to_unit": "B",
                "factor": 2.0,
                "bidirectional": False,
                "source": "test",
                "molecular_weight": None,
            }]
        }
        path = tmp_data_dir / "conv.json"
        path.write_text(json.dumps(conv_data))
        conv = UnitConverter(path)
        with pytest.raises(ValueError, match="No conversion"):
            conv.convert("test", 10.0, "B", "A")
