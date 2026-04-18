"""Import biomarker definitions from a LOINC CSV file.

By default only the curated biomarkers listed in ``data_generation/curated_biomarkers.json`` are imported.
Pass ``--all`` to import every **ACTIVE** lab-relevant row from the LOINC table (``STATUS == ACTIVE``).
Pass ``--max-rows N`` to take the top *N* rows by LOINC ``COMMON_TEST_RANK``
(with accessory files for slugs when not in the curated map).

Each biomarker entry is enriched with LOINC unit metadata
(``loinc_ucum_unit``, ``loinc_example_units``, ``loinc_property``)
so that downstream tools (e.g. ``init_conversions.py``) can determine
which unit conversions are needed without re-reading the LOINC CSV.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import pandas as pd

from core.data_paths import get_generation_config_dir

LOINC_COLUMNS = {
    "LOINC_NUM",
    "COMPONENT",
    "LONG_COMMON_NAME",
    "EXAMPLE_UCUM_UNITS",
    "EXAMPLE_UNITS",
    "CLASS",
    "PROPERTY",
    "SYSTEM",
    "SCALE_TYP",
}

LOINC_COL_STATUS = "STATUS"
LOINC_COL_RANK = "COMMON_TEST_RANK"


def _slugify(text: str) -> str:
    """Convert a LOINC COMPONENT string into a snake_case id."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _consumer_name_slug(consumer_name: str, max_len: int = 48) -> str:
    """Normalize ConsumerName into a snake_case slug."""
    text = str(consumer_name).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text


def _class_to_category(cls: str) -> str:
    mapping = {
        "CHEM": "chemistry",
        "HEM/BC": "hematology",
        "COAG": "coagulation",
        "UA": "urinalysis",
        "SERO": "serology",
        "ALLERGY": "allergy",
        "DRUG/TOX": "toxicology",
    }
    return mapping.get(cls, "other")


def _parse_example_units(raw: str | float) -> list[str]:
    """Split the EXAMPLE_UNITS field (semicolon-separated) into a deduplicated list."""
    if pd.isna(raw) or not str(raw).strip():
        return []
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _loinc_unit_fields(row: pd.Series) -> dict:
    """Extract LOINC unit metadata from a row."""
    ucum_raw = row.get("EXAMPLE_UCUM_UNITS")
    ucum = str(ucum_raw).strip() if pd.notna(ucum_raw) else None
    if ucum:
        ucum_list = [u.strip() for u in ucum.split(";") if u.strip()]
        ucum = ucum_list[0] if ucum_list else None

    example_raw = row.get("EXAMPLE_UNITS", "")
    example_units = _parse_example_units(example_raw)

    prop_raw = row.get("PROPERTY")
    prop = str(prop_raw).strip() if pd.notna(prop_raw) else None

    return {
        "loinc_ucum_unit": ucum,
        "loinc_example_units": example_units,
        "loinc_property": prop,
    }


def _first_ucum_unit(row: pd.Series) -> str:
    ucum_raw = row.get("EXAMPLE_UCUM_UNITS")
    if pd.isna(ucum_raw) or not str(ucum_raw).strip():
        return ""
    ucum_list = [u.strip() for u in str(ucum_raw).split(";") if u.strip()]
    return ucum_list[0] if ucum_list else ""


def _standard_unit_from_row(row: pd.Series, ovr: dict) -> str:
    if "standard_unit" in ovr:
        return str(ovr["standard_unit"])
    return _first_ucum_unit(row)


def _decimal_places_from_overrides(ovr: dict, default: int = 2) -> int:
    if "decimal_places" in ovr and ovr["decimal_places"] is not None:
        return int(ovr["decimal_places"])
    return default


def _read_loinc_csv(loinc_csv: Path, *, load_status: bool, load_rank: bool) -> pd.DataFrame:
    """Load LOINC columns; ``load_status`` for ``--all`` / ``--max-rows``; ``load_rank`` adds COMMON_TEST_RANK."""
    header = pd.read_csv(loinc_csv, nrows=0, dtype=str).columns.tolist()
    want = set(LOINC_COLUMNS)
    if load_status:
        want.add(LOINC_COL_STATUS)
    if load_rank:
        want.add(LOINC_COL_RANK)
    usecols = [c for c in header if c in want]
    missing_base = LOINC_COLUMNS - set(usecols)
    if missing_base:
        raise click.ClickException(f"LOINC CSV missing required columns: {sorted(missing_base)}")
    if load_rank:
        for c in (LOINC_COL_STATUS, LOINC_COL_RANK):
            if c not in usecols:
                raise click.ClickException(
                    f"--max-rows requires column {c!r} in the LOINC CSV (full Loinc table)."
                )
    elif load_status:
        if LOINC_COL_STATUS not in usecols:
            raise click.ClickException(
                f"--all requires column {LOINC_COL_STATUS!r} in the LOINC CSV (full Loinc table)."
            )
    return pd.read_csv(loinc_csv, dtype=str, low_memory=False, usecols=usecols)


def _default_accessory_paths(loinc_csv: Path) -> tuple[Path, Path]:
    root = loinc_csv.resolve().parent.parent
    consumer = root / "AccessoryFiles" / "ConsumerName" / "ConsumerName.csv"
    hierarchy = (
        root / "AccessoryFiles" / "ComponentHierarchyBySystem" / "ComponentHierarchyBySystem.csv"
    )
    return consumer, hierarchy


def _load_import_overrides(path: Path | None) -> dict[str, dict]:
    p = path or (get_generation_config_dir() / "loinc_import_overrides.json")
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("overrides", {})
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _load_keyword_map(path: Path | None) -> dict[str, str]:
    p = path or (get_generation_config_dir() / "hierarchy_category_map.json")
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    m = data.get("keyword_to_category", {})
    return {str(k): str(v) for k, v in m.items()}


def _load_curated_biomarkers(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("biomarkers", [])
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lc = str(item.get("loinc_code", "")).strip()
        bid = str(item.get("id", "")).strip()
        if lc and bid:
            out.append({"loinc_code": lc, "id": bid})
    return out


def _load_loinc_import_config(path: Path) -> frozenset[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data.get("allowed_classes", [])
    return frozenset(str(c) for c in classes)


def _map_hierarchy_label(label: str, keyword_map: dict[str, str], cls: str) -> str:
    if not label or not keyword_map:
        return _class_to_category(cls)
    t = label.lower()
    for kw in sorted(keyword_map.keys(), key=len, reverse=True):
        if kw.lower() in t:
            return keyword_map[kw]
    return _class_to_category(cls)


def _is_loinc_num_code(code: str) -> bool:
    return bool(re.match(r"^\d+-\d+$", str(code)))


def _build_code_to_text(hierarchy_df: pd.DataFrame) -> dict[str, str]:
    """CODE -> CODE_TEXT (first row by SEQUENCE when duplicates exist)."""
    df = hierarchy_df.copy()
    df["_seq"] = pd.to_numeric(df["SEQUENCE"], errors="coerce").fillna(0)
    df = df.sort_values("_seq", na_position="last")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        c = str(row["CODE"])
        if c not in out:
            out[c] = str(row.get("CODE_TEXT") or "")
    return out


def _build_loinc_hierarchy_rows(hierarchy_df: pd.DataFrame) -> dict[str, pd.Series]:
    """LOINC_NUM-style CODE -> representative row (lowest SEQUENCE)."""
    mask = hierarchy_df["CODE"].astype(str).map(_is_loinc_num_code)
    sub = hierarchy_df[mask].copy()
    if sub.empty:
        return {}
    sub["_seq"] = pd.to_numeric(sub["SEQUENCE"], errors="coerce").fillna(0)
    sub = sub.sort_values("_seq", na_position="last")
    idx: dict[str, pd.Series] = {}
    for code, group in sub.groupby("CODE", sort=False):
        idx[str(code)] = group.iloc[0]
    return idx


def _resolve_hierarchy_category(
    loinc_code: str,
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    cls: str,
) -> tuple[str, int | None]:
    row = loinc_hierarchy_rows.get(loinc_code)
    if row is None:
        return _class_to_category(cls), None
    path = str(row.get("PATH_TO_ROOT") or "")
    if not path.strip():
        return _class_to_category(cls), None
    parts = [p for p in path.split(".") if p]
    if hierarchy_level < 0 or hierarchy_level >= len(parts):
        return _class_to_category(cls), None
    anc_code = parts[hierarchy_level]
    label = code_to_text.get(anc_code, "")
    if not label:
        return _class_to_category(cls), None
    cat = _map_hierarchy_label(label, keyword_map, cls)
    return cat, hierarchy_level


def _try_load_hierarchy(h_path: Path) -> tuple[dict[str, pd.Series], dict[str, str]]:
    if not h_path.exists():
        return {}, {}
    hdf = pd.read_csv(
        h_path,
        dtype=str,
        low_memory=False,
        usecols=lambda c: c in {"PATH_TO_ROOT", "SEQUENCE", "CODE", "CODE_TEXT"},
    )
    code_to_text = _build_code_to_text(hdf)
    loinc_hierarchy_rows = _build_loinc_hierarchy_rows(hdf)
    return loinc_hierarchy_rows, code_to_text


def _load_consumer_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
        usecols=lambda c: c in {"LoincNumber", "ConsumerName"},
    )
    return df.set_index("LoincNumber")["ConsumerName"].to_dict()


def _unique_biomarker_id(base: str, loinc_code: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    suffix = loinc_code.replace("-", "_")
    alt = f"{base}_{suffix}"
    if alt not in seen:
        return alt
    n = 2
    while f"{alt}_{n}" in seen:
        n += 1
    return f"{alt}_{n}"


def _seen_loinc_codes(biomarkers: list[dict]) -> set[str]:
    return {str(b["loinc_code"]) for b in biomarkers}


def _seen_biomarker_ids(biomarkers: list[dict]) -> set[str]:
    return {str(b["id"]) for b in biomarkers}


def _single_biomarker_all(
    row: pd.Series,
    curated_ids: dict[str, str],
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    import_overrides: dict[str, dict],
    seen_ids: set[str],
) -> dict:
    """One `--all` biomarker entry; updates ``seen_ids`` with the chosen ``id``."""
    loinc_code = str(row["LOINC_NUM"])
    cls = str(row.get("CLASS", "") or "")
    ovr = import_overrides.get(loinc_code, {})

    if loinc_code in curated_ids:
        bid = curated_ids[loinc_code]
    else:
        bid = _slugify(str(row.get("COMPONENT", loinc_code)))

    if bid in seen_ids:
        bid = f"{bid}_{loinc_code.replace('-', '_')}"
    seen_ids.add(bid)

    if "category" in ovr:
        category = str(ovr["category"])
    else:
        category, _ = _resolve_hierarchy_category(
            loinc_code,
            loinc_hierarchy_rows,
            code_to_text,
            hierarchy_level,
            keyword_map,
            cls,
        )

    unit = _standard_unit_from_row(row, ovr)
    dp = _decimal_places_from_overrides(ovr, default=2)

    entry = {
        "id": bid,
        "loinc_code": loinc_code,
        "en_name": str(row.get("LONG_COMMON_NAME", row.get("COMPONENT", bid))),
        "category": category,
        "standard_unit": unit,
        "description": str(row.get("LONG_COMMON_NAME", "")),
        "decimal_places": dp,
    }
    entry.update(_loinc_unit_fields(row))
    return entry


def _single_biomarker_ranked(
    row: pd.Series,
    consumer_map: dict[str, str] | None,
    curated_ids: dict[str, str],
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    import_overrides: dict[str, dict],
    seen_ids: set[str],
) -> dict:
    """One `--max-rows` biomarker entry; updates ``seen_ids`` with the chosen ``id``."""
    loinc_code = str(row["LOINC_NUM"])
    cls = str(row.get("CLASS", "") or "")
    ovr = import_overrides.get(loinc_code, {})
    curated_id = curated_ids.get(loinc_code)

    if ovr.get("id"):
        base_id = str(ovr["id"])
        id_source = "override"
    elif curated_id is not None:
        base_id = curated_id
        id_source = "curated"
    elif consumer_map and loinc_code in consumer_map:
        cn = consumer_map[loinc_code]
        if pd.notna(cn) and str(cn).strip():
            base_id = _consumer_name_slug(str(cn))
            id_source = "consumer_name"
        else:
            base_id = _slugify(str(row.get("COMPONENT", loinc_code)))
            id_source = "component"
    else:
        base_id = _slugify(str(row.get("COMPONENT", loinc_code)))
        id_source = "component"

    bid = _unique_biomarker_id(base_id, loinc_code, seen_ids)
    seen_ids.add(bid)

    if "category" in ovr:
        category = str(ovr["category"])
        h_level_used: int | None = None
    else:
        category, h_level_used = _resolve_hierarchy_category(
            loinc_code,
            loinc_hierarchy_rows,
            code_to_text,
            hierarchy_level,
            keyword_map,
            cls,
        )

    unit = _standard_unit_from_row(row, ovr)
    dp = _decimal_places_from_overrides(ovr, default=2)

    entry = {
        "id": bid,
        "loinc_code": loinc_code,
        "en_name": str(row.get("LONG_COMMON_NAME", row.get("COMPONENT", bid))),
        "category": category,
        "standard_unit": unit,
        "description": str(row.get("LONG_COMMON_NAME", "")),
        "decimal_places": dp,
        "slug_source": id_source,
        "hierarchy_level_used": h_level_used,
    }
    entry.update(_loinc_unit_fields(row))
    return entry


def _append_missing_curated(
    biomarkers: list[dict],
    curated_entries: list[dict[str, str]],
    df: pd.DataFrame,
    class_filter: frozenset[str],
    *,
    ranked_mode: bool,
    consumer_map: dict[str, str] | None,
    curated_ids: dict[str, str],
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    import_overrides: dict[str, dict],
) -> int:
    """Append curated LOINCs not already in ``biomarkers`` (ACTIVE + class filter). Returns count added."""
    seen_loinc = _seen_loinc_codes(biomarkers)
    seen_ids = _seen_biomarker_ids(biomarkers)

    work = df.set_index("LOINC_NUM", drop=False)
    if work.index.duplicated().any():
        work = work[~work.index.duplicated(keep="first")]

    added = 0
    for item in curated_entries:
        lc = item["loinc_code"]
        if lc in seen_loinc:
            continue
        if lc not in work.index:
            continue
        row = work.loc[lc]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cls = str(row.get("CLASS", "") or "")
        if cls not in class_filter:
            continue
        st_raw = row.get("STATUS")
        if pd.isna(st_raw) or str(st_raw).strip().upper() != "ACTIVE":
            continue

        if ranked_mode:
            entry = _single_biomarker_ranked(
                row,
                consumer_map,
                curated_ids,
                loinc_hierarchy_rows,
                code_to_text,
                hierarchy_level,
                keyword_map,
                import_overrides,
                seen_ids,
            )
        else:
            entry = _single_biomarker_all(
                row,
                curated_ids,
                loinc_hierarchy_rows,
                code_to_text,
                hierarchy_level,
                keyword_map,
                import_overrides,
                seen_ids,
            )
        biomarkers.append(entry)
        seen_loinc.add(lc)
        added += 1
    return added


def _build_curated(
    df: pd.DataFrame,
    curated_entries: list[dict[str, str]],
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    import_overrides: dict[str, dict],
) -> list[dict]:
    """Build biomarker entries from curated_biomarkers.json order."""
    biomarkers: list[dict] = []
    df_indexed = df.set_index("LOINC_NUM")

    for item in curated_entries:
        loinc_code = item["loinc_code"]
        bid = item["id"]
        if loinc_code not in df_indexed.index:
            click.echo(f"  Warning: LOINC code {loinc_code} ({bid}) not found in CSV — skipping")
            continue
        row = df_indexed.loc[loinc_code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        cls = str(row.get("CLASS", "") or "")
        ovr = import_overrides.get(loinc_code, {})
        if "category" in ovr:
            category = str(ovr["category"])
            h_level_used: int | None = None
        else:
            category, h_level_used = _resolve_hierarchy_category(
                loinc_code,
                loinc_hierarchy_rows,
                code_to_text,
                hierarchy_level,
                keyword_map,
                cls,
            )

        unit = _standard_unit_from_row(row, ovr)
        dp = _decimal_places_from_overrides(ovr, default=2)

        entry = {
            "id": bid,
            "loinc_code": loinc_code,
            "en_name": str(row.get("LONG_COMMON_NAME", row.get("COMPONENT", bid))),
            "category": category,
            "standard_unit": unit,
            "description": str(row.get("LONG_COMMON_NAME", "")),
            "decimal_places": dp,
            "hierarchy_level_used": h_level_used,
        }
        entry.update(_loinc_unit_fields(row))
        biomarkers.append(entry)

    return biomarkers


def _build_all(
    df: pd.DataFrame,
    class_filter: frozenset[str],
    curated_ids: dict[str, str],
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    import_overrides: dict[str, dict],
) -> list[dict]:
    """Build biomarker entries for all ACTIVE lab-relevant LOINC rows."""
    filtered = df[df["CLASS"].isin(class_filter)].copy()
    filtered = filtered[filtered["STATUS"].fillna("").str.upper() == "ACTIVE"]
    biomarkers: list[dict] = []
    seen_ids: set[str] = set()

    for _, row in filtered.iterrows():
        biomarkers.append(
            _single_biomarker_all(
                row,
                curated_ids,
                loinc_hierarchy_rows,
                code_to_text,
                hierarchy_level,
                keyword_map,
                import_overrides,
                seen_ids,
            )
        )

    return biomarkers


def _build_ranked(
    df: pd.DataFrame,
    max_rows: int,
    consumer_map: dict[str, str] | None,
    curated_ids: dict[str, str],
    loinc_hierarchy_rows: dict[str, pd.Series],
    code_to_text: dict[str, str],
    hierarchy_level: int,
    keyword_map: dict[str, str],
    import_overrides: dict[str, dict],
    class_filter: frozenset[str],
) -> list[dict]:
    """Top-N biomarkers by COMMON_TEST_RANK with optional accessory enrichment."""
    filtered = df[df["CLASS"].isin(class_filter)].copy()
    filtered = filtered[filtered["STATUS"].fillna("").str.upper() == "ACTIVE"]

    filtered["_rank"] = pd.to_numeric(filtered["COMMON_TEST_RANK"], errors="coerce").fillna(999999)
    ranked_pos = filtered[filtered["_rank"] > 0].sort_values("_rank", ascending=True)
    rank_zero = filtered[filtered["_rank"] == 0].sort_values("COMPONENT", ascending=True)
    combined = pd.concat([ranked_pos, rank_zero])
    subset = combined.head(max_rows)

    biomarkers: list[dict] = []
    seen_ids: set[str] = set()

    for _, row in subset.iterrows():
        biomarkers.append(
            _single_biomarker_ranked(
                row,
                consumer_map,
                curated_ids,
                loinc_hierarchy_rows,
                code_to_text,
                hierarchy_level,
                keyword_map,
                import_overrides,
                seen_ids,
            )
        )

    return biomarkers


@click.command()
@click.option("--loinc-csv", required=True, type=click.Path(exists=True), help="Path to the LOINC CSV file (Loinc.csv)")
@click.option("--output", required=True, type=click.Path(), help="Output path for biomarkers.json")
@click.option(
    "--all",
    "import_all",
    is_flag=True,
    default=False,
    help="Import all ACTIVE lab-relevant biomarkers (CLASS allowlist) instead of the curated list",
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    metavar="N",
    help="Import top N lab-relevant ACTIVE rows by COMMON_TEST_RANK (mutually exclusive with --all and default curated mode)",
)
@click.option(
    "--hierarchy-level",
    type=int,
    default=3,
    show_default=True,
    help="0-based index into PATH_TO_ROOT (ComponentHierarchyBySystem) for UI category labels",
)
@click.option(
    "--curated-json",
    type=click.Path(exists=True),
    default=None,
    help="Curated biomarker list (default: bloodscope/data_generation/curated_biomarkers.json)",
)
@click.option(
    "--import-config",
    type=click.Path(exists=True),
    default=None,
    help="allowed_classes for --all / --max-rows (default: bloodscope/data_generation/loinc_import_config.json)",
)
@click.option(
    "--consumer-names-csv",
    type=click.Path(exists=True),
    default=None,
    help="ConsumerName.csv (default: …/AccessoryFiles/ConsumerName/ConsumerName.csv next to Loinc table)",
)
@click.option(
    "--hierarchy-csv",
    type=click.Path(exists=True),
    default=None,
    help="ComponentHierarchyBySystem.csv (default: sibling under AccessoryFiles/)",
)
@click.option(
    "--hierarchy-map",
    type=click.Path(exists=True),
    default=None,
    help="JSON file: keyword_to_category map (default: bloodscope/data_generation/hierarchy_category_map.json)",
)
@click.option(
    "--overrides",
    "overrides_path",
    type=click.Path(exists=True),
    default=None,
    help="Sparse JSON overrides by LOINC (default: bloodscope/data_generation/loinc_import_overrides.json)",
)
@click.option(
    "--skip-curated",
    "skip_curated",
    is_flag=True,
    default=False,
    help="With --all or --max-rows: do not append missing LOINCs from curated_biomarkers.json",
)
def main(
    loinc_csv: str,
    output: str,
    import_all: bool,
    max_rows: int | None,
    hierarchy_level: int,
    curated_json: str | None,
    import_config: str | None,
    consumer_names_csv: str | None,
    hierarchy_csv: str | None,
    hierarchy_map: str | None,
    overrides_path: str | None,
    skip_curated: bool,
) -> None:
    """Import biomarker definitions from LOINC CSV into biomarkers.json."""
    if import_all and max_rows is not None:
        raise click.UsageError("Options --all and --max-rows are mutually exclusive.")
    if max_rows is not None and max_rows < 1:
        raise click.UsageError("--max-rows must be a positive integer.")
    if skip_curated and not import_all and max_rows is None:
        raise click.UsageError("--skip-curated is only valid with --all or --max-rows.")

    loinc_path = Path(loinc_csv)
    load_rank = max_rows is not None
    load_status = import_all or load_rank

    gen_dir = get_generation_config_dir()
    curated_path = Path(curated_json) if curated_json else (gen_dir / "curated_biomarkers.json")
    if not curated_path.exists():
        raise click.ClickException(f"Curated list not found: {curated_path}")
    curated_entries = _load_curated_biomarkers(curated_path)
    curated_ids = {e["loinc_code"]: e["id"] for e in curated_entries}

    config_path = Path(import_config) if import_config else (gen_dir / "loinc_import_config.json")
    if not config_path.exists():
        raise click.ClickException(f"Import config not found: {config_path}")
    class_filter = _load_loinc_import_config(config_path)

    ovr_p = Path(overrides_path) if overrides_path else None
    hm_p = Path(hierarchy_map) if hierarchy_map else None
    import_overrides = _load_import_overrides(ovr_p)
    keyword_map = _load_keyword_map(hm_p)

    default_consumer, default_hierarchy = _default_accessory_paths(loinc_path)
    h_path = Path(hierarchy_csv) if hierarchy_csv else default_hierarchy

    click.echo(f"Reading LOINC CSV from {loinc_path} ...")
    df = _read_loinc_csv(loinc_path, load_status=load_status, load_rank=load_rank)
    click.echo(f"  Loaded {len(df)} LOINC rows")

    if h_path.exists():
        click.echo(f"  Loading component hierarchy from {h_path}")
        loinc_hierarchy_rows, code_to_text = _try_load_hierarchy(h_path)
    else:
        click.echo(f"  Warning: hierarchy file not found at {h_path} — using CLASS-based categories")
        loinc_hierarchy_rows, code_to_text = {}, {}

    if import_all:
        click.echo("Importing ALL lab-relevant biomarkers ...")
        biomarkers = _build_all(
            df,
            class_filter,
            curated_ids,
            loinc_hierarchy_rows,
            code_to_text,
            hierarchy_level,
            keyword_map,
            import_overrides,
        )
        if not skip_curated:
            n_extra = _append_missing_curated(
                biomarkers,
                curated_entries,
                df,
                class_filter,
                ranked_mode=False,
                consumer_map=None,
                curated_ids=curated_ids,
                loinc_hierarchy_rows=loinc_hierarchy_rows,
                code_to_text=code_to_text,
                hierarchy_level=hierarchy_level,
                keyword_map=keyword_map,
                import_overrides=import_overrides,
            )
            if n_extra:
                click.echo(f"  Appended {n_extra} curated biomarker(s) not already in the export")
    elif max_rows is not None:
        click.echo(f"Importing top {max_rows} biomarkers by COMMON_TEST_RANK ...")
        c_path = Path(consumer_names_csv) if consumer_names_csv else default_consumer
        consumer_map: dict[str, str] | None = None
        if c_path.exists():
            click.echo(f"  Loading consumer names from {c_path}")
            consumer_map = _load_consumer_map(c_path)
        else:
            click.echo(f"  Warning: ConsumerName file not found at {c_path} — using COMPONENT-based slugs")

        biomarkers = _build_ranked(
            df,
            max_rows,
            consumer_map,
            curated_ids,
            loinc_hierarchy_rows,
            code_to_text,
            hierarchy_level,
            keyword_map,
            import_overrides,
            class_filter,
        )
        if not skip_curated:
            n_extra = _append_missing_curated(
                biomarkers,
                curated_entries,
                df,
                class_filter,
                ranked_mode=True,
                consumer_map=consumer_map,
                curated_ids=curated_ids,
                loinc_hierarchy_rows=loinc_hierarchy_rows,
                code_to_text=code_to_text,
                hierarchy_level=hierarchy_level,
                keyword_map=keyword_map,
                import_overrides=import_overrides,
            )
            if n_extra:
                click.echo(f"  Appended {n_extra} curated biomarker(s) not already in the top {max_rows} by rank")
    else:
        click.echo(f"Importing curated set of {len(curated_entries)} biomarkers ...")
        biomarkers = _build_curated(
            df,
            curated_entries,
            loinc_hierarchy_rows,
            code_to_text,
            hierarchy_level,
            keyword_map,
            import_overrides,
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"biomarkers": biomarkers}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote {len(biomarkers)} biomarkers to {output_path}")


if __name__ == "__main__":
    main()
