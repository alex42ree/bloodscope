"""Lab reference interval shape and display strings (shared by classifier, CLI, UI)."""

from __future__ import annotations

from typing import Literal, Optional

LabReferenceIntervalKind = Literal["none", "both", "upper_only", "lower_only"]


def lab_reference_interval_kind(
    lab_low: Optional[float],
    lab_high: Optional[float],
) -> LabReferenceIntervalKind:
    """Same rules as classification: both, upper_only, lower_only, or none (invalid / empty)."""
    if lab_low is not None and lab_high is not None:
        return "both" if lab_low < lab_high else "none"
    if lab_high is not None:
        return "upper_only"
    if lab_low is not None:
        return "lower_only"
    return "none"


def format_lab_reference_for_display(lab_low: Optional[float], lab_high: Optional[float]) -> str:
    """Human-readable lab reference for tables (aligned with CLI analyze output)."""
    kind = lab_reference_interval_kind(lab_low, lab_high)
    if kind == "none":
        if lab_low is None and lab_high is None:
            return "-"
        return "—"
    if kind == "upper_only":
        return f"< {lab_high}"
    if kind == "lower_only":
        return f"> {lab_low}"
    if lab_low == 0:
        return f"< {lab_high}"
    return f"{lab_low} - {lab_high}"
