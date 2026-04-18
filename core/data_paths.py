"""Default paths for runtime JSON (data/) and import configuration (data_generation/)."""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent


def get_default_runtime_data_dir() -> Path:
    """Directory with biomarkers.json, translations.json, ranges.json, etc.

    Override with env ``BLOODSCOPE_DATA_DIR`` (relative paths resolve against CWD).
    """
    env = os.getenv("BLOODSCOPE_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path("data")


def get_generation_config_dir() -> Path:
    """Versioned JSON inputs for ``init_*`` CLIs (curated list, LOINC tuning, languages, …)."""
    return _PKG_ROOT / "data_generation"
