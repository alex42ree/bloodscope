# Runtime data (`data/`)

This directory holds the JSON files **BloodScope loads at runtime**: biomarker definitions, translations, optimal ranges, and unit conversions.

In a fresh clone these files may be missing: they are produced by the `cli/init_*.py` scripts (and optional `manage_*` tools) after you obtain LOINC and run the steps in the [root README](../README.md) (**Building runtime data**). Generator **configuration** lives in [`../data_generation/`](../data_generation/).

## Environment variable

- **`BLOODSCOPE_DATA_DIR`**: Path to a directory containing the same filenames (`biomarkers.json`, `translations.json`, `ranges.json`, `unit_conversions.json`). The app defaults to a `data` folder relative to the **current working directory** (typically the `bloodscope/` project root).

## Git

JSON files here are listed in `.gitignore` so large or LOINC-derived artifacts are not forced into a public repository. After you generate them locally, they can stay on disk for day-to-day use.
