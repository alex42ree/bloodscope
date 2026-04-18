# Data generation configuration

These JSON files are **inputs** to the `cli/init_*.py` scripts (LOINC import, translations, conversions, range sources). They are meant to be **versioned** in a public repository.

Generator **outputs** (`biomarkers.json`, `translations.json`, `ranges.json`, `unit_conversions.json`) go under [`../data/`](../data/) and are usually **not** committed (see the root `.gitignore`).

## Files

| File | Purpose |
|------|---------|
| `curated_biomarkers.json` | Curated LOINC → internal `id` list for the default `init_biomarkers` import |
| `loinc_import_config.json` | `allowed_classes` for `--all` / `--max-rows` |
| `loinc_import_overrides.json` | Per-LOINC overrides (category, unit, decimal places, …) |
| `hierarchy_category_map.json` | Keyword → UI category mapping |
| `conversion_overrides.json` | Skip list and extra molar conversions for `init_conversions` |
| `languages.json` | Supported language codes for `init_translations` |
| `range_sources.json` | URLs and priorities for `init_ranges` |

## LOINC

The LOINC table CSV is **not** shipped in this repo. Register at [loinc.org](https://loinc.org), accept the license, and download the full table package. You need at least `LoincTable/Loinc.csv`. For richer imports, also use the same release’s **AccessoryFiles** (e.g. `ConsumerName/ConsumerName.csv`, `ComponentHierarchyBySystem/ComponentHierarchyBySystem.csv`, `LinguisticVariants/`). See the root [README](../README.md) section **Building runtime data**.
