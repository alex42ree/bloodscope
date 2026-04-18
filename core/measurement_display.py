"""Stable string formatting for reported measurement values (CLI, UI, Arrow-safe DataFrames)."""

from __future__ import annotations

from typing import Optional


def format_measurement_display(
    value: float,
    converted_value: Optional[float],
    value_modifier: Optional[str],
    *,
    use_converted: bool = True,
) -> str:
    """Format numeric measurement for display; always returns a str (no mixed-type columns)."""
    if use_converted and converted_value is not None:
        display_val = converted_value
    else:
        display_val = value
    mod = value_modifier or ""
    return f"{mod}{display_val}"
