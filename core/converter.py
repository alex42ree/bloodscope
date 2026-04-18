"""Unit conversion: UNIT_ALIASES normalization + custom biomarker map + Pint fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pint

from core.data_paths import get_default_runtime_data_dir

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()

UNIT_ALIASES: dict[str, str] = {
    # Cell-count units
    "x10^6/mm³": "x10^6/µL",
    "x10^6/mm3": "x10^6/µL",
    "10^6/mm³":  "x10^6/µL",
    "10*6/uL":   "x10^6/µL",
    "mill/µl":   "x10^6/µL",
    "x10³/mm³":  "x10^3/µL",
    "x10^3/mm3": "x10^3/µL",
    "10^3/mm³":  "x10^3/µL",
    "10*3/uL":   "x10^3/µL",
    # Mass concentration – casing
    "g/dl":      "g/dL",
    "g/DL":      "g/dL",
    "mg/dl":     "mg/dL",
    "mg/DL":     "mg/dL",
    "µg/dl":     "µg/dL",
    "ug/dL":     "µg/dL",
    # Volume concentration
    "g/l":       "g/L",
    "mg/l":      "mg/L",
    # Enzymes
    "U/l":       "U/L",
    "u/l":       "U/L",
    "UI/L":      "U/L",
    # Thyroid
    "uIU/mL":    "µIU/mL",
    "uUI/mL":    "µIU/mL",
    "mUI/L":     "mIU/L",
    # Percent
    "percent":   "%",
}


def normalize_unit(unit: str) -> str:
    """Normalise a unit string via the UNIT_ALIASES table."""
    return UNIT_ALIASES.get(unit.strip(), unit.strip())


def _lookup_key(unit: str) -> str:
    """Lowercase key for internal lookup (after alias resolution)."""
    u = unit.replace("µ", "u").replace("μ", "u")
    return u.lower()


ConversionEntry = dict  # full entry from unit_conversions.json
ConversionKey = tuple[str, str, str]  # (biomarker_id, from_key, to_key)


class UnitConverter:
    """Two-tier unit converter: custom map first, Pint second.

    The ``convert`` method returns a ``(value, method)`` tuple where *method*
    is one of ``"identity"``, ``"custom"``, ``"custom_reverse"``, ``"pint"``.
    """

    def __init__(self, conversions_path: str | Path) -> None:
        self._custom: dict[ConversionKey, ConversionEntry] = {}
        path = Path(conversions_path)
        if not path.exists():
            logger.warning("Conversions file not found: %s", path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("conversions", []):
            key: ConversionKey = (
                entry["biomarker_id"],
                _lookup_key(entry["from_unit"]),
                _lookup_key(entry["to_unit"]),
            )
            self._custom[key] = entry

    def convert(
        self,
        biomarker_id: str,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> tuple[float, str]:
        """Convert *value* from *from_unit* to *to_unit*.

        Units are first resolved through :data:`UNIT_ALIASES`, then looked up
        in the custom conversion map.  If no custom entry exists, Pint is tried.

        Returns
        -------
        tuple[float, str]
            ``(converted_value, method)`` where method is ``"identity"``,
            ``"custom"``, ``"custom_reverse"``, or ``"pint"``.

        Raises
        ------
        ValueError
            If no conversion path exists.
        """
        from_resolved = normalize_unit(from_unit)
        to_resolved = normalize_unit(to_unit)
        from_key = _lookup_key(from_resolved)
        to_key = _lookup_key(to_resolved)

        if from_key == to_key:
            return value, "identity"

        # 1. Custom map – forward
        fwd: ConversionKey = (biomarker_id, from_key, to_key)
        if fwd in self._custom:
            return value * self._custom[fwd]["factor"], "custom"

        # 2. Custom map – reverse (only if bidirectional)
        rev: ConversionKey = (biomarker_id, to_key, from_key)
        if rev in self._custom:
            entry = self._custom[rev]
            if entry.get("bidirectional", True):
                return value / entry["factor"], "custom_reverse"

        # 3. Pint fallback
        try:
            quantity = ureg.Quantity(value, from_resolved)
            return quantity.to(to_resolved).magnitude, "pint"
        except Exception:
            pass

        raise ValueError(
            f"No conversion for {biomarker_id}: {from_unit} -> {to_unit}"
        )


def load_converter(
    conversions_path: str | Path | None = None,
) -> UnitConverter:
    """Convenience factory."""
    path = (
        conversions_path
        if conversions_path is not None
        else (get_default_runtime_data_dir() / "unit_conversions.json")
    )
    return UnitConverter(path)
