# BloodScope

Python app that ingests **text-based** laboratory report PDFs, extracts biomarkers via the Anthropic API, maps names to canonical IDs, converts units, and classifies values as **optimal**, **normal**, or **out of range** using a **hybrid** model: the lab’s own reference interval from the PDF plus optional **optimal** bands from curated `ranges.json`.

---

## Important limitations

- **Not a medical device.** This software does not diagnose, treat, or replace professional medical advice.
- **Text PDFs only.** BloodScope does **not** support scanned or image-only PDFs. You need a searchable (“text”) PDF. If no text can be extracted, the app and CLI return a clear **scanned PDF / not supported** error.
- **Data processing.** Report content is sent to **Anthropic** for structured extraction (and optionally a second call for biomarker ID fallback). Review their terms and your privacy obligations before using real patient data.

### Optimal ranges — where the data comes from

Optimal (functional-medicine-style) bands in `data/ranges.json` are **not** universal clinical standards. They are narrower than typical lab reference ranges. This repository’s default pipeline for building those entries uses configured web/PDF sources listed in [`data_generation/range_sources.json`](data_generation/range_sources.json), for example:

- FMU Blood Tracking Form (PDF)
- Rupa Health optimal ranges (HTML)
- SiPhox Health article(s) (HTML)

---

## Features (as implemented)

### Streamlit UI (`app.py`)

- PDF upload with **size** and **page** limits (see Environment).
- Sidebar: sex and date-of-birth **from PDF** or **manual override**; **Analyze** button.
- Caption: text-PDF-only policy and limit summary.
- After run: metrics (totals by status), **categorized tables** (Reference vs Optimal columns, status coloring).
- Expanders: **Analysis log** (pipeline INFO for that run), **Patient context** (extracted vs effective demographics).
- **Unclassified biomarkers** section when names do not match the database.
- Short **disclaimer** footer.

### CLI analysis (`python -m cli.analyze`)

- Full pipeline on a PDF path: prints grouped tables, summary, patient context.
- Options: `--sex`, `--dob` (YYYY-MM-DD), `--output` JSON, `--data-dir`.
- Same PDF limits and scanned-PDF error behavior as the UI.
- Preflight: file exists, PDF limits, `ANTHROPIC_API_KEY`, required JSON data files.

### Data maintenance CLIs

Run from the `bloodscope` directory (or set `PYTHONPATH` appropriately). Prefer `python -m cli.<module>`.

| Module | Purpose |
|--------|---------|
| `cli.init_biomarkers` | LOINC CSV → `data/biomarkers.json` (curated, `--max-rows`, or `--all`) |
| `cli.init_translations` | LOINC Linguistic Variants → `data/translations.json` |
| `cli.init_ranges` | `discover` / `generate` optimal ranges from `data_generation/range_sources.json` → `data/ranges.json` |
| `cli.init_conversions` | `discover` / `generate` unit conversions (NLM UCUM + PubChem where needed) → `data/unit_conversions.json` |
| `cli.manage_translations` | `add`, `show`, `import`, `gaps`, `stats`, `check-duplicates`, `export-csv` |
| `cli.manage_ranges` | `set`, `show`, `import`, **`validate`** (informational: lists biomarkers **without** optimal rows; that is OK at runtime) |
| `cli.manage_conversions` | `add`, `list`, `show`, `validate`, `check-consistency` |

### Core modules (high level)

- `core/pipeline.py` — orchestration
- `core/extractor.py` — pdfplumber text + table fallback
- `core/pdf_validation.py` — `MAX_PDF_MB` / `MAX_PDF_SIZE_BYTES`, `MAX_PDF_PAGES`
- `core/exceptions.py` — e.g. `ScannedPdfNotSupportedError`, `PdfTooLargeError`, `PdfTooManyPagesError`
- `core/llm_parser.py`, `core/llm_biomarker_match.py`
- `core/matcher.py`, `core/converter.py`, `core/classifier.py`, `core/lab_reference.py`, `core/measurement_display.py`
- `core/schemas.py`, `core/data_paths.py`

---

## Setup

```bash
cd bloodscope
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum
```

### Data layout

- **`data_generation/`** — versioned **config** for import tools (committed). See [`data_generation/README.md`](data_generation/README.md).
- **`data/`** — **runtime JSON** used by the app. See [`data/README.md`](data/README.md). These files are usually **gitignored**; build them with the steps below.

Optional: set **`BLOODSCOPE_DATA_DIR`** to a directory that contains the four JSON files.

---

## Building runtime data (from LOINC)

Run all commands from the **`bloodscope/`** directory with your virtual environment activated. Replace `LOINC_ROOT` with the folder where you unpacked the LOINC release (the directory that contains `LoincTable` and `AccessoryFiles`).

### 1. Download LOINC

1. Create a free account at [loinc.org/downloads](https://loinc.org/downloads/).
2. Download the **LOINC** table package (CSV) for the release you want to use.
3. Unzip it. You should have:
   - `LOINC_ROOT/LoincTable/Loinc.csv` (required)
   - `LOINC_ROOT/AccessoryFiles/ConsumerName/ConsumerName.csv` (recommended for slugs / names)
   - `LOINC_ROOT/AccessoryFiles/ComponentHierarchyBySystem/ComponentHierarchyBySystem.csv` (recommended for categories)
   - `LOINC_ROOT/AccessoryFiles/LinguisticVariants/` (required for the translation import below)

### 2. Import biomarkers

Default mode imports the curated list from `data_generation/curated_biomarkers.json` (paths to accessory files are inferred next to `Loinc.csv` when you do not pass them explicitly):

```bash
python -m cli.init_biomarkers \
  --loinc-csv "$LOINC_ROOT/LoincTable/Loinc.csv" \
  --output data/biomarkers.json
```

Optional: `--max-rows N`, `--all`, or explicit `--consumer-names-csv` / `--hierarchy-csv` — see [`docs/BloodScope_Spec.md`](docs/BloodScope_Spec.md) §5.1.

### 3. Import translations

Builds `data/translations.json` from LOINC linguistic variant CSVs for the languages listed in `data_generation/languages.json`:

```bash
python -m cli.init_translations \
  --loinc-dir "$LOINC_ROOT/AccessoryFiles/LinguisticVariants" \
  --biomarkers data/biomarkers.json \
  --output data/translations.json
```

### 4. Generate unit conversions

Inspect what will be generated (optional):

```bash
python -m cli.init_conversions discover \
  --biomarkers data/biomarkers.json
```

Generate `data/unit_conversions.json`. This step calls the **NLM UCUM** and **PubChem** HTTP APIs (network required). Ensure `data_generation/conversion_overrides.json` is present for molar and skip rules:

```bash
python -m cli.init_conversions generate \
  --biomarkers data/biomarkers.json \
  --output data/unit_conversions.json
```

Re-run with `--merge` to fill in only missing rows after partial failures.

### 5. Generate optimal ranges

Requires **`ANTHROPIC_API_KEY`** (and network). Sources are listed in `data_generation/range_sources.json`.

```bash
python -m cli.init_ranges discover --biomarkers data/biomarkers.json

python -m cli.init_ranges generate \
  --biomarkers data/biomarkers.json \
  --sources data_generation/range_sources.json \
  --output data/ranges.json
```

Use `--merge` on `generate` to add missing biomarkers without overwriting existing rows. You can instead maintain ranges manually with `python -m cli.manage_ranges` (see the CLI table above).

### 6. Verify preparation

These files should exist before you run the app or `cli.analyze`:

- `data/biomarkers.json`
- `data/translations.json`
- `data/unit_conversions.json`
- `data/ranges.json`

### 7. Smoke test with the CLI

With `.env` configured (`ANTHROPIC_API_KEY` at minimum), analyze a **text-based** sample PDF. If you keep a demo file outside this repo (e.g. `challenge.pdf` next to a parent workspace), run:

```bash
python -m cli.analyze ../docs/challenge.pdf --data-dir data
```

Adjust the path to your PDF. On success you should see patient context, categorized biomarker tables, and a summary block. For a full JSON dump, add `--output result.json`.

---

## Environment variables

| Variable | Meaning |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for extraction (and fallback when enabled). |
| `CLAUDE_MODEL` | Model for `llm_parser` and `llm_biomarker_match`. |
| `MAX_PDF_MB` | Max upload size in MiB (default **10**). |
| `MAX_PDF_SIZE_BYTES` | Alternative exact byte cap (overrides `MAX_PDF_MB` if set). |
| `MAX_PDF_PAGES` | Max pages processed (default **100**). |
| `BIOMARKER_LLM_FALLBACK` | Set `0` to disable second-pass ID resolution. |
| `BIOMARKER_FALLBACK_TOP_K` | Shortlist size (default 25). |
| `BLOODSCOPE_DATA_DIR` | Directory for runtime JSON (default `./data` relative to CWD). |

See [`.env.example`](.env.example).

---

## Running

```bash
streamlit run app.py
```

```bash
python -m cli.analyze path/to/report.pdf
python -m cli.analyze path/to/report.pdf --sex male --dob 1990-05-15 --output result.json
```

After **Building runtime data**, use the sample command in step 7 (e.g. `../docs/challenge.pdf`) or any other searchable lab PDF.

---

## Tests

```bash
pytest tests/ -v
```

`tests/sample_pdfs/` is intentionally empty for now (placeholder for future anonymized PDFs).

---

## Specification

The consolidated specification is **[`docs/BloodScope_Spec.md`](docs/BloodScope_Spec.md)**. Older split documents are in [`docs/archive/`](docs/archive/).

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE).
