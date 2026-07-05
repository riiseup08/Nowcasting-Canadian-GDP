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
├── dashboard/
│   ├── __init__.py
│   └── app.py              ★ DASHBOARD — Streamlit interactive dashboard
│
├── data/
│   ├── 36-10-0434-01.zip    # RAW — Gross domestic product (GDP) at basic prices, by industry, monthly
│   ├── 14-10-0287-01.zip    # RAW — Labour force characteristics, monthly
│   ├── 18-10-0004-01.zip    # RAW — Consumer price index, monthly
│   ├── 16-10-0047-01.zip    # RAW — Monthly survey of manufacturing
│   ├── 36-10-0434-01/       # RAW — Extracted CSV folder (GDP)
│   ├── 14-10-0287-01/       # RAW — Extracted CSV folder (Labour Force Survey)
│   ├── 18-10-0004-01/       # RAW — Extracted CSV folder (Consumer Price Index)
│   ├── 16-10-0047-01/       # RAW — Extracted CSV folder (Monthly Survey of Manufacturing)
│   └── processed/
│       └── training_data.csv # ★ WAREHOUSE — clean, merged, feature-engineered dataset
│
├── models/
│   ├── nowcast_model.pkl    # Trained Random Forest model (joblib)
│   ├── feature_cols.pkl     # Feature column names (joblib)
│   └── model_metrics.json   # CV metrics in JSON
│
├── requirements.txt
└── README.md
```

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

This is the "Load" step: the transformed data is stored in a persistent location so the modelling step can read it without re-running the entire ETL process.

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

### 3. Train the Random Forest model (optional — pre-trained model included)
```bash
python train_model.py
```

Trains a Random Forest with 4-fold TimeSeries CV, saves the model + feature columns + metrics to `models/`.

### 4. Run the nowcast notebook
Open `run_nowcast.ipynb` in VS Code or Jupyter and run all cells — or execute from the command line:
```bash
jupyter nbconvert --to notebook --execute run_nowcast.ipynb --output run_nowcast_output.ipynb
```

The notebook loads the warehouse table, trains all 5 models, and prints a comparison table.

**Expected output (models comparison):**
```
Model                   Nowcast        Error      MAE%     CV MAE   CV MAE%
-----------------------------------------------------------------------------
OLS Regression          2,416,653      63,750    2.71%    109,450    6.12%
Ridge Regression        2,396,588      43,685    1.86%    109,425    6.11%
SVR (RBF)               2,489,232     136,329    5.79%    275,462   15.39%
Gradient Boosting       2,367,198      14,295    0.61%    127,253    7.11%
Random Forest           2,348,573       4,330    0.18%    107,944    6.03%  ← best
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

Opens an interactive web dashboard with:
- **Metric cards** — nowcast values, CV error, OLS baseline, RF error
- **Actual vs Predicted chart** — time series with actual GDP, in-sample predictions, and nowcast marker
- **Feature importance bar chart** — top 10 features driving the Random Forest
- **Model comparison table** — all 5 models side-by-side with nowcast, error, and CV MAE
- **TimeSeries CV summary** — training rows, date range, model type, last trained
- **Refresh & retrain buttons** — re-run `transform.py` to fetch new data or `train_model.py` to retrain

---

## Model Results (Latest Month)

| Model | Nowcast | Error | CV MAE |
|-------|---------|-------|--------|
| OLS Regression | $2,416,653M | $63,750M (2.71%) | $109,450M (6.12%) |
| Ridge Regression | $2,396,588M | $43,685M (1.86%) | $109,425M (6.11%) |
| SVR (RBF) | $2,489,232M | $136,329M (5.79%) | $275,462M (15.39%) |
| Gradient Boosting | $2,367,198M | $14,295M (0.61%) | $127,253M (7.11%) |
| **Random Forest** | **$2,348,573M** | **$4,330M (0.18%)** | **$107,944M (6.03%)** |

**Random Forest** achieves the lowest CV MAE (`$107,944M`, 6.03% of mean GDP) and the closest nowcast to the actual published value (`$2,352,903M`).

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
| `transform.py` | **Orchestrator** — runs the full Extract-Transform-Load pipeline: downloads all 4 tables, cleans/filters each, merges on date, engineers 21 features (lags + rolling means + growth rates), and saves the result to `data/processed/training_data.csv`. |
| `features.py` | Contains three reusable functions: `create_lag_features()` (shifts a column by N periods), `create_rolling_features()` (rolling window means), and `create_growth_rate()` (month-over-month percentage change). |
| `train_model.py` | Loads the feature-engineered dataset, trains a Random Forest with 4-fold TimeSeries CV, and saves the model + feature columns + metrics to `models/`. |
| `run_nowcast.ipynb` | Jupyter notebook that loads the warehouse table, trains **5 models** (OLS, Ridge, SVR, Gradient Boosting, Random Forest) with TimeSeries CV, and outputs a comparison table plus feature importance. |
| `dashboard/app.py` | **Streamlit dashboard** — interactive web app showing nowcast metrics, actual vs predicted chart, feature importance, model comparison table, and data refresh/retrain controls. |

---

## Dependencies

```
pandas
numpy
scikit-learn
streamlit
plotly
joblib
sqlalchemy
python-dotenv
requests
openpyxl
jupyter