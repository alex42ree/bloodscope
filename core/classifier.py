"""Hybrid three-tier classification of biomarker values.

Normal band comes from lab-report-extracted reference ranges.
Optimal band comes from curated ranges.json (optimal-only entries).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from core.data_paths import get_default_runtime_data_dir
from core.lab_reference import LabReferenceIntervalKind, lab_reference_interval_kind
from core.schemas import Classification

logger = logging.getLogger(__name__)


def _validate_range(low: Optional[float], high: Optional[float]) -> bool:
    """Return True when both values are present and low < high."""
    if low is None or high is None:
        return False
    return low < high


def _value_in_lab_normal_one_sided(
    value: float,
    lab_low: Optional[float],
    lab_high: Optional[float],
    kind: LabReferenceIntervalKind,
) -> bool:
    """Whether value lies in the lab's normal region for one- or two-sided refs.

    Upper-only (e.g. ref ``< 5.7``): normal is values at or below the bound.
    Lower-only (e.g. ``> 40``): normal is values at or above the bound.
    """
    if kind == "both":
        return lab_low <= value <= lab_high  # type: ignore[operator]
    if kind == "upper_only":
        return value <= lab_high  # type: ignore[operator]
    if kind == "lower_only":
        return value >= lab_low  # type: ignore[operator]
    return False


def classify(
    value: float,
    biomarker_id: str,
    sex: str,
    age: int,
    ranges_df: pd.DataFrame,
    *,
    lab_low: Optional[float] = None,
    lab_high: Optional[float] = None,
) -> Classification:
    """Classify a biomarker value using the hybrid approach.

    Parameters
    ----------
    value:
        The (already unit-converted) numeric value.
    biomarker_id:
        Canonical biomarker identifier.
    sex:
        ``"male"`` or ``"female"``.
    age:
        Patient age in years.
    ranges_df:
        DataFrame with columns: biomarker_id, sex, age_min, age_max,
        optimal_low, optimal_high.  Loaded from ranges.json (optimal only).
    lab_low:
        Lower reference bound extracted from the lab report.
    lab_high:
        Upper reference bound extracted from the lab report.
    """
    lab_kind = lab_reference_interval_kind(lab_low, lab_high)
    lab_valid = lab_kind in ("both", "upper_only", "lower_only")

    opt_low: Optional[float] = None
    opt_high: Optional[float] = None
    if not ranges_df.empty:
        mask = (
            (ranges_df["biomarker_id"] == biomarker_id)
            & (ranges_df["sex"].isin([sex, "any"]))
            & (ranges_df["age_min"] <= age)
            & (ranges_df["age_max"] >= age)
        )
        matches = ranges_df[mask]

        if not matches.empty:
            sex_specific = matches[matches["sex"] == sex]
            row = sex_specific.iloc[0] if not sex_specific.empty else matches.iloc[0]
            opt_low = float(row["optimal_low"])
            opt_high = float(row["optimal_high"])

    opt_valid = _validate_range(opt_low, opt_high)

    if lab_valid and opt_valid:
        if lab_kind == "both":
            if opt_low < lab_low or opt_high > lab_high:  # type: ignore[operator]
                logger.warning(
                    "%s: optimal range [%.4g, %.4g] is outside lab normal [%.4g, %.4g] — "
                    "discarding optimal range as erroneous",
                    biomarker_id, opt_low, opt_high, lab_low, lab_high,
                )
                opt_valid = False
        elif lab_kind == "upper_only":
            if opt_high > lab_high:  # type: ignore[operator]
                logger.warning(
                    "%s: optimal range [%.4g, %.4g] extends above lab upper bound %.4g — "
                    "discarding optimal range as erroneous",
                    biomarker_id, opt_low, opt_high, lab_high,
                )
                opt_valid = False
        elif lab_kind == "lower_only":
            if opt_low < lab_low:  # type: ignore[operator]
                logger.warning(
                    "%s: optimal range [%.4g, %.4g] extends below lab lower bound %.4g — "
                    "discarding optimal range as erroneous",
                    biomarker_id, opt_low, opt_high, lab_low,
                )
                opt_valid = False

    if lab_valid:
        if not _value_in_lab_normal_one_sided(value, lab_low, lab_high, lab_kind):
            return Classification.OUT_OF_RANGE
        if opt_valid and opt_low <= value <= opt_high:  # type: ignore[operator]
            return Classification.OPTIMAL
        return Classification.NORMAL

    return Classification.UNKNOWN


def lookup_optimal(
    biomarker_id: str,
    sex: str,
    age: int,
    ranges_df: pd.DataFrame,
) -> tuple[Optional[float], Optional[float]]:
    """Look up optimal range for a biomarker. Returns (opt_low, opt_high)."""
    if ranges_df.empty:
        return None, None

    mask = (
        (ranges_df["biomarker_id"] == biomarker_id)
        & (ranges_df["sex"].isin([sex, "any"]))
        & (ranges_df["age_min"] <= age)
        & (ranges_df["age_max"] >= age)
    )
    matches = ranges_df[mask]
    if matches.empty:
        return None, None

    sex_specific = matches[matches["sex"] == sex]
    row = sex_specific.iloc[0] if not sex_specific.empty else matches.iloc[0]
    return float(row["optimal_low"]), float(row["optimal_high"])


def load_ranges(ranges_path: str | Path | None = None) -> pd.DataFrame:
    """Load ranges.json into a DataFrame ready for :func:`classify`."""
    path = Path(ranges_path) if ranges_path is not None else (get_default_runtime_data_dir() / "ranges.json")
    if not path.exists():
        logger.warning("Ranges file not found: %s — optimal classification unavailable", path)
        return pd.DataFrame(columns=[
            "biomarker_id", "sex", "age_min", "age_max", "unit",
            "optimal_low", "optimal_high", "source", "source_url",
        ])
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data.get("ranges", []))
