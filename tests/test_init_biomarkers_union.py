"""Curated union + --skip-curated behavior for init_biomarkers."""

import pandas as pd

from cli.init_biomarkers import _append_missing_curated, _build_ranked


def test_max_rows_union_appends_missing_curated():
    """Ranked top-1 then append curated LOINC not in that row."""
    df = pd.DataFrame(
        [
            {
                "LOINC_NUM": "1111-1",
                "COMPONENT": "Alpha",
                "LONG_COMMON_NAME": "Alpha long",
                "EXAMPLE_UCUM_UNITS": "mg/dL",
                "EXAMPLE_UNITS": "",
                "CLASS": "CHEM",
                "PROPERTY": "MCnc",
                "SYSTEM": "Ser",
                "SCALE_TYP": "Qn",
                "STATUS": "ACTIVE",
                "COMMON_TEST_RANK": "1",
            },
            {
                "LOINC_NUM": "2222-2",
                "COMPONENT": "Beta",
                "LONG_COMMON_NAME": "Beta long",
                "EXAMPLE_UCUM_UNITS": "g/L",
                "EXAMPLE_UNITS": "",
                "CLASS": "CHEM",
                "PROPERTY": "MCnc",
                "SYSTEM": "Ser",
                "SCALE_TYP": "Qn",
                "STATUS": "ACTIVE",
                "COMMON_TEST_RANK": "999",
            },
        ]
    )
    curated_entries = [
        {"loinc_code": "1111-1", "id": "alpha"},
        {"loinc_code": "2222-2", "id": "beta_curated"},
    ]
    curated_ids = {e["loinc_code"]: e["id"] for e in curated_entries}
    class_filter = frozenset({"CHEM"})

    biomarkers = _build_ranked(
        df,
        max_rows=1,
        consumer_map=None,
        curated_ids=curated_ids,
        loinc_hierarchy_rows={},
        code_to_text={},
        hierarchy_level=0,
        keyword_map={},
        import_overrides={},
        class_filter=class_filter,
    )
    assert len(biomarkers) == 1
    assert biomarkers[0]["loinc_code"] == "1111-1"

    n = _append_missing_curated(
        biomarkers,
        curated_entries,
        df,
        class_filter,
        ranked_mode=True,
        consumer_map=None,
        curated_ids=curated_ids,
        loinc_hierarchy_rows={},
        code_to_text={},
        hierarchy_level=0,
        keyword_map={},
        import_overrides={},
    )
    assert n == 1
    assert len(biomarkers) == 2
    codes = {b["loinc_code"] for b in biomarkers}
    assert codes == {"1111-1", "2222-2"}
    assert biomarkers[1]["id"] == "beta_curated"


def test_skip_curated_union_is_noop_when_empty():
    biomarkers: list[dict] = []
    df = pd.DataFrame(
        [
            {
                "LOINC_NUM": "1-1",
                "COMPONENT": "X",
                "LONG_COMMON_NAME": "X",
                "EXAMPLE_UCUM_UNITS": "mg/dL",
                "EXAMPLE_UNITS": "",
                "CLASS": "CHEM",
                "PROPERTY": "MCnc",
                "SYSTEM": "Ser",
                "SCALE_TYP": "Qn",
                "STATUS": "ACTIVE",
                "COMMON_TEST_RANK": "1",
            },
        ]
    )
    curated_entries = [{"loinc_code": "1-1", "id": "x"}]
    n = _append_missing_curated(
        biomarkers,
        curated_entries,
        df,
        frozenset({"CHEM"}),
        ranked_mode=False,
        consumer_map=None,
        curated_ids={"1-1": "x"},
        loinc_hierarchy_rows={},
        code_to_text={},
        hierarchy_level=0,
        keyword_map={},
        import_overrides={},
    )
    assert n == 1
    assert biomarkers[0]["id"] == "x"
