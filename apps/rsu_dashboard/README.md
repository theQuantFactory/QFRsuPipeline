# RSU Dashboard App

Production dashboard app using `qfpytoolbox` for ingestion, cleaning, analytics, and cache pre-baking.

## Prerequisites

- Python 3.10+ (3.13 supported in this repo)
- Dependencies installed in a virtual environment

From repository root:

```bash
python -m venv vrsuenv
vrsuenv\Scripts\activate
python -m pip install -e ".[dev,all]"
python -m pip install streamlit plotly duckdb
```

## Input data (supported formats)

Place raw files in:

`apps/rsu_dashboard/data/raw/`

Supported file formats: `.csv`, `.arrow`, `.xlsx`

Expected logical sources:

- `menage` (e.g. `menage.csv`)
- `scores` (either `scores.csv` or `score.csv`) **required**
- `programmes` (either consolidated `programmes.csv` with `menage_ano, programme`, or split files `amot.csv`, `asd.csv`, `amoa.csv`)
- `beneficiaire` (optional)

Notes:
- The pipeline auto-discovers aliases (`score`/`scores`, `programme`/`programmes`, `beneficiaire`/`beneficiaires`).
- You can mix formats across files (example: `menage.xlsx` + `score.csv` + `programmes.arrow`).

## Run the app (end-to-end)

From repository root:

```bash
# 1) Build parquet snapshots from raw inputs
python apps/rsu_dashboard/pipeline.py --input-dir apps/rsu_dashboard/data/raw --output-dir apps/rsu_dashboard/snapshots/csv

# 2) Pre-bake cache for fast dashboard startup
python apps/rsu_dashboard/prebake_dashboard.py

# 3) Start Streamlit dashboard
streamlit run apps/rsu_dashboard/dashboard.py
```

## Optional explicit paths

You can still pass explicit paths if needed:

```bash
python apps/rsu_dashboard/prebake_dashboard.py --snap-dir apps/rsu_dashboard/snapshots/csv --out apps/rsu_dashboard/snapshots/dashboard_cache.pkl
```

## Notes

- All compute logic is in `qfpytoolbox.rsu` and its helpers.
- The app layer only orchestrates pipeline, prebake, and UI.
- Main modules:
  - `qfpytoolbox.rsu`
  - `qfpytoolbox.rsu_loaders`
  - `qfpytoolbox.rsu_builder`
  - `qfpytoolbox.rsu_encoding`
