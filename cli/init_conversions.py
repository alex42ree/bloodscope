"""Generate unit_conversions.json from authoritative sources.

Dimensional conversions (same physical dimension, different scale) are
computed via the NLM UCUM API.  Molar conversions (mass <-> substance
amount) additionally require molecular weights, which are fetched from
the PubChem API.

Every entry records full provenance (source URL, molecular weight source)
so the resulting data file is independently auditable.

Usage
-----
::

    # Step 1: Discover which conversions are needed
    python -m cli.init_conversions discover \\
        --biomarkers data/biomarkers.json

    # Step 2: Generate unit_conversions.json
    python -m cli.init_conversions generate \\
        --biomarkers data/biomarkers.json \\
        --overrides data_generation/conversion_overrides.json \\
        --output data/unit_conversions.json
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import click

from core.data_paths import get_generation_config_dir

NLM_UCUM_BASE = "https://ucum.nlm.nih.gov/ucum-service/v1/ucumtransform"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

NO_CONVERSION_PROPERTIES = frozenset({
    "CCnc",       # catalytic concentration (U/L)
    "NFr", "MFr", "VFr",  # fractions (%)
    "Time",       # seconds
    "RelTime",    # INR
    "MRto",       # ratios
    "EntMass",    # pg (per-entity mass)
    "EntMCnc",    # g/dL (per-entity mass concentration)
    "EntMeanVol", # fL (mean volume)
    "DistWidth",  # % (distribution width)
    "ArVRat",     # mL/min (area-volume rate)
    "-",          # panels / no unit
})

TO_UCUM: dict[str, str] = {
    "x10^6/µL": "10*6/uL",
    "x10^3/µL": "10*3/uL",
    "µg/dL": "ug/dL",
    "µg/mL": "ug/mL",
    "µIU/mL": "u[IU]/mL",
    "µmol/L": "umol/L",
    "mIU/L": "m[IU]/L",
    "mIU/mL": "m[IU]/mL",
    "IU/mL": "[IU]/mL",
    "IU/L": "[IU]/L",
    "mEq/L": "meq/L",
    "ng/mL": "ng/mL",
    "pg/mL": "pg/mL",
}

KNOWN_EQUIVALENTS: dict[tuple[str, str], float] = {
    ("uIU/mL", "u[IU]/mL"): 1.0,
    ("u[IU]/mL", "uIU/mL"): 1.0,
    ("mU/L", "m[IU]/L"): 1.0,
    ("mcU/mL", "m[IU]/L"): 1.0,
    ("units/mL", "[IU]/mL"): 1.0,
    ("U IU/mL", "m[IU]/L"): 0.001,
    ("mmol/L", "meq/L"): 1.0,
    # mc-prefix equivalents (mc = µ in LOINC notation)
    ("mcg/dL", "ug/dL"): 1.0,
    ("mcg/mL", "ug/mL"): 1.0,
    ("mcg/mL", "mg/L"): 1.0,
    ("mcg/dL", "ug/mL"): 0.01,
    ("mcg/mL", "ug/dL"): 100.0,
    ("mcmol/dL", "umol/dL"): 1.0,
    ("mcmol/dL", "umol/L"): 10.0,
    ("mcmol/L", "umol/L"): 1.0,
    # IU inter-conversions
    ("[IU]/L", "m[IU]/mL"): 1.0,
    ("IU/L", "m[IU]/mL"): 1.0,
}

SKIP_DIMENSIONAL: set[tuple[str, str]] = {
    ("mg/dL", "umol/L"),
    ("mg/dL", "µmol/L"),
    ("mmol/L", "ug/dL"),
    ("mmol/L", "µg/dL"),
}

API_DELAY = 0.25


def _to_ucum(unit: str) -> str:
    """Convert our unit notation to UCUM-compatible notation for the NLM API."""
    return TO_UCUM.get(unit, unit)


def _normalize_pair(from_u: str, to_u: str) -> tuple[str, str]:
    return (_to_ucum(from_u), _to_ucum(to_u))


def _call_nlm_ucum(
    quantity: float,
    from_unit: str,
    to_unit: str,
    molecular_weight: float | None = None,
) -> tuple[float, str]:
    """Call NLM UCUM API and return (result_quantity, api_url)."""
    from_ucum = _to_ucum(from_unit)
    to_ucum = _to_ucum(to_unit)

    url = (
        f"{NLM_UCUM_BASE}/{quantity}"
        f"/from/{quote(from_ucum, safe='')}"
        f"/to/{quote(to_ucum, safe='')}"
    )
    if molecular_weight is not None:
        url += f"/MOLWEIGHT/{molecular_weight}"

    try:
        with urlopen(url, timeout=10) as resp:
            xml_data = resp.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"NLM API error for {url}: {exc}") from exc

    root = ET.fromstring(xml_data)
    result_el = root.find(".//ResultQuantity")
    if result_el is None or result_el.text is None:
        msg_el = root.find(".//Message")
        msg = msg_el.text if msg_el is not None and msg_el.text else "no result"
        raise RuntimeError(f"NLM UCUM API: {msg} ({url})")

    return float(result_el.text), url


def _fetch_pubchem_mw(cid: int) -> tuple[float, str]:
    """Fetch molecular weight from PubChem by CID."""
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/MolecularWeight/JSON"
    try:
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"PubChem error for CID {cid}: {exc}") from exc

    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        raise RuntimeError(f"PubChem returned no properties for CID {cid}")

    mw = float(props[0]["MolecularWeight"])
    source = f"PubChem CID {cid}"
    return mw, source


def _discover_dimensional(biomarkers: list[dict], skip_ids: set[str]) -> list[dict]:
    """Find dimensional conversions needed from LOINC example_units."""
    conversions = []
    for bm in biomarkers:
        bid = bm["id"]
        if bid in skip_ids:
            continue

        prop = bm.get("loinc_property", "")
        std_unit = bm["standard_unit"]
        example_units = bm.get("loinc_example_units", [])

        if prop in NO_CONVERSION_PROPERTIES:
            continue

        for alt_unit in example_units:
            alt_clean = alt_unit.strip()
            if not alt_clean or alt_clean == std_unit:
                continue
            norm_from, norm_to = _normalize_pair(alt_clean, std_unit)
            if norm_from == norm_to:
                continue
            if (norm_from, norm_to) in SKIP_DIMENSIONAL:
                continue
            conversions.append({
                "biomarker_id": bid,
                "from_unit": alt_clean,
                "to_unit": std_unit,
                "type": "dimensional",
            })

    return conversions


def _default_overrides_path() -> str | None:
    p = get_generation_config_dir() / "conversion_overrides.json"
    return str(p) if p.is_file() else None


def _load_overrides(path: str | None) -> tuple[set[str], list[dict]]:
    """Load override file. Returns (skip_ids, extra_conversions)."""
    if path is None:
        return set(), []
    p = Path(path)
    if not p.exists():
        return set(), []
    data = json.loads(p.read_text(encoding="utf-8"))
    skip_ids = {s["biomarker_id"] for s in data.get("skip", [])}
    extras = data.get("extra_conversions", [])
    return skip_ids, extras


def _resolve_conversion(
    from_unit: str,
    to_unit: str,
    molecular_weight: float | None = None,
) -> tuple[float, str, str]:
    """Resolve a single conversion factor.

    Returns (factor, source_label, source_url_or_note).
    Checks known equivalents first, then calls NLM API.
    """
    norm_from, norm_to = _normalize_pair(from_unit, to_unit)

    if norm_from == norm_to:
        return 1.0, "equivalent notation", "identity"

    equiv_key = (from_unit, norm_to)
    if equiv_key in KNOWN_EQUIVALENTS:
        f = KNOWN_EQUIVALENTS[equiv_key]
        return f, "known equivalent (UCUM notation)", f"{from_unit} = {f} × {to_unit}"

    equiv_key2 = (norm_from, norm_to)
    if equiv_key2 in KNOWN_EQUIVALENTS:
        f = KNOWN_EQUIVALENTS[equiv_key2]
        return f, "known equivalent (UCUM notation)", f"{from_unit} = {f} × {to_unit}"

    factor, api_url = _call_nlm_ucum(1.0, from_unit, to_unit, molecular_weight=molecular_weight)
    source = "NLM UCUM API (molar)" if molecular_weight else "NLM UCUM API (dimensional)"
    return factor, source, api_url


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Generate unit conversion data from authoritative sources."""


@cli.command()
@click.option("--biomarkers", required=True, type=click.Path(exists=True))
@click.option("--overrides", default=None, type=click.Path(exists=True),
              help="conversion_overrides.json (default: data_generation/conversion_overrides.json if present)")
def discover(biomarkers: str, overrides: str | None) -> None:
    """Analyse biomarkers.json and report which conversions are needed."""
    bm_data = json.loads(Path(biomarkers).read_text(encoding="utf-8"))
    bm_list = bm_data.get("biomarkers", [])
    skip_ids, extras = _load_overrides(overrides or _default_overrides_path())

    dim = _discover_dimensional(bm_list, skip_ids)

    no_conv = []
    for bm in bm_list:
        bid = bm["id"]
        prop = bm.get("loinc_property", "")
        if bid in skip_ids:
            continue
        if prop in NO_CONVERSION_PROPERTIES:
            no_conv.append(bid)
        elif not bm.get("loinc_example_units"):
            no_conv.append(bid)

    has_dim = {c["biomarker_id"] for c in dim}
    extra_ids = {e["biomarker_id"] for e in extras}
    rest = [bm["id"] for bm in bm_list
            if bm["id"] not in has_dim
            and bm["id"] not in extra_ids
            and bm["id"] not in skip_ids
            and bm["id"] not in set(no_conv)]

    click.echo(f"Total biomarkers: {len(bm_list)}")
    click.echo(f"  No conversion needed: {len(no_conv)}")
    click.echo(f"  Dimensional (from LOINC example_units): {len(dim)}")
    click.echo(f"  Molar (from overrides): {len(extras)}")
    click.echo(f"  Skipped (overrides): {len(skip_ids)}")
    click.echo(f"  No conversion data: {len(rest)}")

    if dim:
        click.echo("\nDimensional conversions:")
        for c in dim:
            click.echo(f"  {c['biomarker_id']:25s} {c['from_unit']:15s} -> {c['to_unit']}")

    if extras:
        click.echo(f"\nMolar conversions (from overrides, need PubChem MW + NLM API):")
        for e in extras:
            click.echo(f"  {e['biomarker_id']:25s} {e['from_unit']:15s} -> {e['to_unit']:15s} ({e.get('compound_name', '?')})")

    if rest:
        click.echo(f"\nBiomarkers with no conversion data:")
        for bid in rest:
            click.echo(f"  {bid}")


def _load_existing(output_path: str) -> tuple[list[dict], set[tuple[str, str, str]]]:
    """Load existing conversions from the output file.

    Returns (existing_entries, set_of_keys) where each key is
    ``(biomarker_id, from_unit, to_unit)``.
    """
    p = Path(output_path)
    if not p.exists():
        return [], set()
    data = json.loads(p.read_text(encoding="utf-8"))
    entries = data.get("conversions", [])
    keys = {(e["biomarker_id"], e["from_unit"], e["to_unit"]) for e in entries}
    return entries, keys


@cli.command()
@click.option("--biomarkers", required=True, type=click.Path(exists=True))
@click.option("--overrides", default=None, type=click.Path(exists=True),
              help="conversion_overrides.json (default: data_generation/conversion_overrides.json if present)")
@click.option("--output", required=True, type=click.Path())
@click.option("--merge", is_flag=True, default=False,
              help="Keep existing conversions in the output file; only generate missing ones.")
def generate(biomarkers: str, overrides: str | None, output: str, merge: bool) -> None:
    """Generate unit_conversions.json by calling PubChem + NLM UCUM APIs."""
    bm_data = json.loads(Path(biomarkers).read_text(encoding="utf-8"))
    bm_list = bm_data.get("biomarkers", [])
    skip_ids, extras = _load_overrides(overrides or _default_overrides_path())

    existing_entries: list[dict] = []
    existing_keys: set[tuple[str, str, str]] = set()
    if merge:
        existing_entries, existing_keys = _load_existing(output)
        if existing_keys:
            click.echo(f"Merge mode: {len(existing_keys)} existing conversions loaded, will skip those.\n")

    dim_conversions = _discover_dimensional(bm_list, skip_ids)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: list[dict] = list(existing_entries) if merge else []
    new_count = 0
    skipped = 0
    errors: list[str] = []

    click.echo(f"Processing {len(dim_conversions)} dimensional conversions...")
    for conv in dim_conversions:
        bid = conv["biomarker_id"]
        from_u = conv["from_unit"]
        to_u = conv["to_unit"]
        key = (bid, from_u, to_u)
        if key in existing_keys:
            click.echo(f"  {bid:25s} {from_u:15s} -> {to_u:15s} (already exists, skipped)")
            skipped += 1
            continue
        try:
            factor, source, source_url = _resolve_conversion(from_u, to_u)
            results.append({
                "biomarker_id": bid,
                "from_unit": from_u,
                "to_unit": to_u,
                "factor": round(factor, 10),
                "molecular_weight": None,
                "molecular_weight_source": None,
                "source": source,
                "source_url": source_url,
                "bidirectional": True,
                "generated_at": now,
            })
            new_count += 1
            click.echo(f"  {bid:25s} {from_u:15s} -> {to_u:15s} factor={factor:.6g}  [{source}]")
            if "NLM" in source:
                time.sleep(API_DELAY)
        except RuntimeError as exc:
            errors.append(f"  {bid}: {from_u} -> {to_u}: {exc}")
            click.echo(f"  {bid:25s} {from_u:15s} -> {to_u:15s} ERROR: {exc}")

    click.echo(f"\nProcessing {len(extras)} molar conversions (PubChem + NLM UCUM API)...")
    mw_cache: dict[int, tuple[float, str]] = {}
    for extra in extras:
        bid = extra["biomarker_id"]
        from_u = extra["from_unit"]
        to_u = extra["to_unit"]
        cid = extra.get("pubchem_cid")
        compound = extra.get("compound_name", "?")
        key = (bid, from_u, to_u)
        if key in existing_keys:
            click.echo(f"  {bid:25s} {from_u:15s} -> {to_u:15s} (already exists, skipped)")
            skipped += 1
            continue

        if cid is None:
            errors.append(f"  {bid}: no pubchem_cid specified")
            click.echo(f"  {bid:25s} SKIP: no pubchem_cid")
            continue

        try:
            if cid in mw_cache:
                mw, mw_source = mw_cache[cid]
            else:
                mw, mw_source = _fetch_pubchem_mw(cid)
                mw_cache[cid] = (mw, mw_source)
                time.sleep(API_DELAY)

            factor, source, api_url = _resolve_conversion(from_u, to_u, molecular_weight=mw)
            results.append({
                "biomarker_id": bid,
                "from_unit": from_u,
                "to_unit": to_u,
                "factor": round(factor, 10),
                "molecular_weight": mw,
                "molecular_weight_source": mw_source,
                "source": source,
                "source_url": api_url,
                "bidirectional": True,
                "generated_at": now,
            })
            new_count += 1
            click.echo(
                f"  {bid:25s} {from_u:15s} -> {to_u:15s} "
                f"MW={mw:<10.3f} factor={factor:.6g}  ({compound})"
            )
            time.sleep(API_DELAY)
        except RuntimeError as exc:
            errors.append(f"  {bid} ({compound}): {exc}")
            click.echo(f"  {bid:25s} {from_u:15s} -> {to_u:15s} ERROR: {exc}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"conversions": results}
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    click.echo(f"\nWrote {len(results)} total conversions to {output}")
    click.echo(f"  {new_count} new, {skipped} skipped (already existed)")
    if errors:
        click.echo(f"\n{len(errors)} errors:")
        for e in errors:
            click.echo(e)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
