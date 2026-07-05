# GDP Nowcast — Canada

Monthly nowcast of Canadian gross domestic product (GDP, chained 2017 dollars, seasonally adjusted at annual rates) using employment, consumer price index, and manufacturing sales as predictors.

**Data source:** Statistics Canada Web Data Service (WDS) API — live data directly from the agency's REST endpoints.

---

## Project Structure

```
├── api.py                  # EXTRACT — StatCan download-link API
├── download.py             # EXTRACT — HTTP file download + ZIP extraction
├── loader.py               # EXTRACT — CSV loading (skips _MetaData sidecars)
├── clean.py                # EXTRACT — dimension filtering + date parsing
│
├── transform.py            ★ TRANSFORM + LOAD — orchestrates the full ETL pipeline
├── features.py             # TRANSFORM — lag, rolling mean, growth rate features
│
├── train_model.py          ★ MODEL — trains Random Forest & saves to models/
├── run_nowcast.ipynb       ★ MODEL — notebook with 5 models & nowcast output
│
├── agent/
│   ├── __init__.py
│   ├── tools.py            # Tool functions for the AI agent
│   └── graph.py            # LangGraph agent powered by DeepSeek
│
├── dashboard/
│   ├── __init__.py
│   └── app.py              ★ DASHBOARD — Streamlit interactive dashboard
│
├── models/
│   └── model_metrics.json   # CV metrics in JSON
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** Raw data (StatCan ZIP/CSV files), processed data (`training_data.csv`), and model binaries (`.pkl`) are excluded from git. Run `python transform.py` to download the data locally.

---

## The ETL-T Pipeline (Extract → Transform → Load → Model)

This project implements a complete ETL-T data pipeline. Here is each stage in detail:

### 1. EXTRACT — Pulling raw data from Statistics Canada

Four Python modules work together to extract data from the Statistics Canada WDS API:

| Module | What it does |
|--------|-------------|
| **`api.py`** | Calls the StatCan REST endpoint `getFullTableDownloadCSV/{productId}/en` with an 8-digit numeric product ID. The API returns a JSON array containing the direct download URL for the ZIP file. |
| **`download.py`** | Downloads the ZIP file via HTTP streaming to the `data/` folder. If the file is a ZIP archive, it extracts the contents into a subfolder. |
| **`loader.py`** | Scans the extracted folder for `.csv` files, ignoring any file whose name contains "MetaData" (these are metadata sidecars, not the actual data). Loads the data CSV into a pandas DataFrame. |
| **`clean.py`** | Takes the raw DataFrame and applies: (1) a geography filter to keep only rows where GEO contains "Canada", (2) exact-match filters on every dimension column (industry, seasonal adjustment, age group, etc.) so only one specific series remains, (3) date parsing that handles both string dates and integer years (e.g. 2023 → 2023-01-01), and (4) a deduplication warning if multiple rows remain per date. The result is a clean two-column DataFrame: `date` + `value`. |

**What is extracted:** Four separate StatCan tables (see the Data Warehouse tables section below), each reduced to a single time series with exactly one value per month.

### 2. TRANSFORM — Merging, aligning, and feature engineering

**`transform.py`** is the orchestrator that runs both the Extract and Transform stages. Here is exactly what it does, step by step:

**Step 1 — Extract each table:**
The script calls the four extraction modules for each of the four StatCan product IDs. Each table is downloaded, extracted, cleaned, and filtered to exactly one series.

**Step 2 — Merge on date:**
The three predictor tables (Labour Force Survey — Employment, Consumer Price Index — All-items, Monthly Survey of Manufacturing — Sales) are joined together with an **inner join** on the `date` column. This means only months where all three predictors exist are kept. Then the GDP (target) table is joined with a **left join**, so months where GDP has not yet been published are still kept in the dataset (these become the "nowcast" months where we predict without seeing the actual value).

**Step 3 — Feature engineering:**
For each of the three predictor series (employment, CPI, manufacturing sales), the script creates:

| Feature type | What it does | Number created |
|-------------|-------------|:------------:|
| **Lag features** | The value from 1, 2, and 3 months ago. This captures recent history and momentum. | 3 per predictor = 9 total |
| **Rolling means** | The average value over the last 3 months and last 6 months. This smooths out month-to-month noise and captures the medium-term trend. | 2 per predictor = 6 total |
| **Growth rate** | The month-over-month percentage change. This measures the rate of change rather than the level. | 1 per predictor = 3 total |

Total: **21 features** from the 3 predictor series, plus the raw values themselves — ready for the model.

**Step 4 — Drop incomplete rows:**
Rows where feature computation produced NaN values (the first few months where lag/rolling windows don't have enough history) are dropped. This leaves a clean, contiguous time series.

### 3. LOAD — Writing to the data warehouse

After the Transform step is complete, the resulting DataFrame is saved to:

```
data/processed/training_data.csv
```

This file acts as the **data warehouse** — a single, flat, ready-to-query table containing all 21 features, the GDP target column, and the date column.

You can also write to a **PostgreSQL / Supabase** database instead (see the [Supabase Setup](#-loading-to-supabase-postgresql) section).

### 4. MODELS — Five ML models compared

**`run_nowcast.ipynb`** reads the warehouse table (`training_data.csv`) and trains **5 models** side-by-side:

| Model | Type | Description |
|-------|------|-------------|
| **OLS Regression** | Linear | Baseline linear model with no regularisation |
| **Ridge Regression** | Linear | L2-regularised linear model — handles multicollinearity better than OLS |
| **SVR (RBF kernel)** | Non-linear | Support Vector Regression captures non-linear patterns (features are standardised) |
| **Gradient Boosting** | Tree ensemble | Sequential ensemble — each tree corrects the previous one's errors |
| **Random Forest** | Tree ensemble | Bagging ensemble of 100 trees — the primary production model |

Each model is evaluated with **4-fold TimeSeries cross-validation** (respecting temporal order — no look-ahead) and produces a nowcast for the latest month. The best model by CV MAE is highlighted automatically.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ETL (Extract → Transform → Load)
```bash
python transform.py
```

This downloads all four StatCan tables, cleans, filters, merges, engineers 21 features, and saves the result to `data/processed/training_data.csv`.

**Expected output:**
```
=======================================================
  ETL — Extract
=======================================================
  [gdp] Downloading 36-10-0434-01...
  [gdp] 352 rows, 1997-01-01 → 2026-04-01
  [lfs] Downloading 14-10-0287-01...
  [lfs] 605 rows, 1976-01-01 → 2026-05-01
  [cpi] Downloading 18-10-0004-01...
  [cpi] 1349 rows, 1914-01-01 → 2026-05-01
  [mfg] Downloading 16-10-0047-01...
  [mfg] 412 rows, 1992-01-01 → 2026-04-01

=======================================================
  ETL — Transform (merge + feature engineering)
=======================================================
  Merged frame:  412 rows, 1992-01-01 → 2026-04-01
  Feature-engineered: 21 features
  Total rows:          407
  Rows with target:    352
  Date range:          1992-06-01 → 2026-04-01

  ✅ Saved to data\processed\training_data.csv  (114,734 bytes)
```

### 3. Run the nowcast notebook
Open `run_nowcast.ipynb` in VS Code or Jupyter and run all cells — or execute from the command line:
```bash
jupyter nbconvert --to notebook --execute run_nowcast.ipynb --output run_nowcast_output.ipynb
```

The notebook loads the warehouse table, trains all 5 models, and prints a comparison table.

**Expected output (models comparison):**
```
Model                       Nowcast        Error     MAE%      CV MAE  CV MAE%
--------------------------------------------------------------------------
OLS Regression            2,416,653       63,750    2.71%      60,638    3.39%
Ridge Regression          2,416,779       63,876    2.71%      59,790    3.34%  ← best CV
SVR (RBF)                 1,775,918      576,985   24.52%     348,316   19.47%
Gradient Boosting         2,351,493        1,410    0.06%     105,850    5.92%
Random Forest             2,348,573        4,330    0.18%     107,944    6.03%
```

> **Note:** Results vary each time the pipeline runs with updated StatCan data. The values above are from a sample run.

### 4. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

Opens an interactive web dashboard with:
- **Metric cards** — nowcast values, CV error, OLS baseline, RF error
- **Multi-model chart** — all 5 model prediction lines vs actual GDP over time
- **Feature importance bar chart** — top 10 features driving the Random Forest
- **Model comparison table** — all 5 models side-by-side with nowcast, error, and CV MAE
- **TimeSeries CV summary** — training rows, date range, model type, last trained
- **Refresh & retrain buttons** — re-run `transform.py` to fetch new data or `train_model.py` to retrain
- **Senior Economist AI Agent** — chat sidebar to ask questions about the nowcast

---

## 💬 Senior Economist AI Agent

A conversational agent powered by **DeepSeek** + **LangGraph** lives in the dashboard sidebar. Ask questions like:

- *"What's the current GDP nowcast?"*
- *"Which model performs best?"*
- *"How does employment affect GDP?"*
- *"What was the GDP trend over the last 5 years?"*
- *"How accurate are the predictions?"*

To use it, set your DeepSeek API key in `.env`:
```
DEEPSEEK_API_KEY=sk-your_key_here
```

---

## 🗄 Loading to Supabase (PostgreSQL)

Instead of reading/writing from a local CSV, you can store the feature-engineered dataset in **Supabase PostgreSQL**.

### Step 1: Create a Supabase project

1. Go to [supabase.com](https://supabase.com) and create an account (free tier available)
2. Create a new project
3. Go to **Project Settings → Database** and copy the **Connection string (URI)**

### Step 2: Configure your `.env` file

The connection string looks like:
```
postgresql://postgres.xxxxx:password@aws-0-xx-xx-xx.pooler.supabase.com:6543/postgres
```

Set the individual components in `.env`:

```ini
DB_HOST=aws-0-xx-xx-xx.pooler.supabase.com
DB_PORT=6543
DB_NAME=postgres
DB_USER=postgres.xxxxx
DB_PASSWORD=your_password
```

### Step 3: Write data to Supabase

```bash
python transform.py --warehouse
```

This upserts the full dataset into a table called `gdp_nowcast_training`.

### Step 4: Train using Supabase data

```bash
python train_model.py --warehouse
```

### Step 5: Dashboard with Supabase

The dashboard automatically detects the PostgreSQL connection (`DATABASE_URL` or the `DB_*` variables) and loads from it. If Supabase is unavailable or unconfigured, it falls back to the local CSV.

---

## ☁️ Deploying the dashboard to Streamlit Cloud

The deployed app has no local CSV (`data/processed/*.csv` is git-ignored), so it **must** read from Supabase.

### 1. Use the Session Pooler connection string (IPv4)

> ⚠️ **Do not use the "Direct connection" string** (`db.<ref>.supabase.co:5432`). On the Supabase free tier that host is **IPv6-only**, and Streamlit Community Cloud is IPv4-only — the app will fail with a connection timeout.

In the Supabase Dashboard click **Connect** → choose **Session pooler**. The string looks like:

```
postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

### 2. Add it to Streamlit secrets

In the Streamlit Cloud app → **Settings → Secrets**, paste (TOML format):

```toml
DATABASE_URL = "postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres"
DEEPSEEK_API_KEY = "sk-..."
```

Streamlit exposes these as environment variables, which `warehouse.py` reads automatically — no code change needed.

### 3. Deploy

Point Streamlit Cloud at this repo with main file `dashboard/app.py`. The app will load the 407-row dataset directly from Supabase.

---

## Model Results (Sample — Latest Run)

| Model | Nowcast | Error | CV MAE |
|-------|---------|-------|--------|
| OLS Regression | $2,416,653M | $63,750M (2.71%) | $60,638M (3.39%) |
| Ridge Regression | $2,416,779M | $63,876M (2.71%) | $59,790M (3.34%) |
| SVR (RBF) | $1,775,918M | $576,985M (24.52%) | $348,316M (19.47%) |
| Gradient Boosting | $2,351,493M | $1,410M (0.06%) | $105,850M (5.92%) |
| **Random Forest** | **$2,348,573M** | **$4,330M (0.18%)** | **$107,944M (6.03%)** |

**Random Forest** achieves the lowest nowcast error (`$4,330M`, 0.18%), while **Ridge Regression** has the lowest cross-validated MAE (`$59,790M`, 3.34% of mean GDP).

> **Note:** Results will differ each time you run the pipeline due to updated StatCan data.

---

## Data Warehouse Tables

The Load step creates one central table (`training_data.csv`). However, the Extract stage works with four distinct StatCan tables. Here are their full official names and the exact filters used to isolate each series:

### Table 1: Gross domestic product (GDP) at basic prices, by industry, monthly (x 1,000)
**Product ID:** `36-10-0434-01`

| Filter column | Exact value selected |
|-------------|-------------------|
| Seasonal adjustment | `Seasonally adjusted at annual rates` |
| Prices | `Chained (2017) dollars` |
| North American Industry Classification System (NAICS) | `All industries [T001]` |
| GEO | `Canada` (fuzzy match) |

**Role:** Target variable. Monthly Canadian GDP in millions of chained 2017 dollars, seasonally adjusted at annual rates, all industries combined. **352 rows from 1997-01-01 to 2026-04-01.**

### Table 2: Labour force characteristics, monthly
**Product ID:** `14-10-0287-01`

| Filter column | Exact value selected |
|-------------|-------------------|
| Labour force characteristics | `Employment` |
| Gender | `Total - Gender` |
| Age group | `15 years and over` |
| Statistics | `Estimate` |
| Data type | `Seasonally adjusted` |

**Role:** Predictor. Total employment in Canada (thousands of persons), seasonally adjusted. This is the most important predictor in the model (39% feature importance). **605 rows from 1976-01-01 to 2026-05-01.**

### Table 3: Consumer price index, monthly
**Product ID:** `18-10-0004-01`

| Filter column | Exact value selected |
|-------------|-------------------|
| Products and product groups | `All-items` |

**Role:** Predictor. The headline Consumer Price Index (all-items, not seasonally adjusted). Captures inflation trends. **1,349 rows from 1914-01-01 to 2026-05-01.**

### Table 4: Monthly survey of manufacturing
**Product ID:** `16-10-0047-01`

| Filter column | Exact value selected |
|-------------|-------------------|
| Principal statistics | `Sales of goods manufactured (shipments)` |
| Seasonal adjustment | `Seasonally adjusted` |
| North American Industry Classification System (NAICS) | `Manufacturing [31-33]` |

**Role:** Predictor. Manufacturing sales in Canada (dollars), seasonally adjusted — a leading indicator of economic activity. **412 rows from 1992-01-01 to 2026-04-01.**

---

## Files Explained

| File | Purpose |
|------|---------|
| `api.py` | Calls Statistics Canada's REST API (`getFullTableDownloadCSV` endpoint) to retrieve the download URL for a given product ID. |
| `download.py` | Downloads a file (ZIP or CSV) from a URL via HTTP streaming. If the file is a ZIP archive, it extracts all contents into a target folder. |
| `loader.py` | Reads a StatCan CSV file from an extracted folder into a pandas DataFrame. Automatically skips any file containing "MetaData" in its name (these are metadata sidecars, not the actual data table). |
| `clean.py` | Filters a raw StatCan DataFrame by geography (GEO contains "Canada") and by exact dimension column filters. Parses dates (handles both string dates like "2023-01-01" and integer years like 2023). Reduces the table to just `date` and `value` columns with one row per date. |
| `transform.py` | **Orchestrator** — runs the full Extract-Transform-Load pipeline: downloads all 4 tables, cleans/filters each, merges on date, engineers 21 features (lags + rolling means + growth rates), and saves the result to `data/processed/training_data.csv`. Supports `--warehouse` to write to PostgreSQL. |
| `features.py` | Contains three reusable functions: `create_lag_features()` (shifts a column by N periods), `create_rolling_features()` (rolling window means), and `create_growth_rate()` (month-over-month percentage change). |
| `train_model.py` | Loads the feature-engineered dataset, trains a Random Forest with 4-fold TimeSeries CV, and saves the model + feature columns + metrics to `models/`. Supports `--warehouse` for PostgreSQL. |
| `run_nowcast.ipynb` | Jupyter notebook that loads the warehouse table, trains **5 models** (OLS, Ridge, SVR, Gradient Boosting, Random Forest) with TimeSeries CV, and outputs a comparison table plus feature importance. |
| `agent/tools.py` | Five tool functions (`get_nowcast`, `get_model_comparison`, `get_feature_importance`, `get_gdp_trend`, `get_data_summary`) used by the AI agent. |
| `agent/graph.py` | LangGraph ReAct agent powered by DeepSeek. Builds a state graph with agent/tool nodes and routes between them. |
| `dashboard/app.py` | **Streamlit dashboard** — interactive web app showing nowcast metrics, multi-model chart, feature importance, model comparison table, and the Senior Economist chat sidebar. |
| `warehouse.py` | PostgreSQL interface — `get_connection()`, `create_table_if_not_exists()`, `save_training_data()` (upsert), `load_training_data()`. |

---

## Dependencies

```
pandas, numpy, scikit-learn, requests
streamlit, plotly, joblib
langgraph, langchain-core, openai
sqlalchemy, python-dotenv
jupyter