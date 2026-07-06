"""
transform.py — ETL Transform step
==================================
Extracts raw data from Statistics Canada (via the existing download/load/clean
modules), applies exact dimension filters, merges all series, engineers features
(lags, rolling means, growth rates), and saves the ready-to-model dataset to:

1. ``data/processed/training_data.csv``  (local dev cache)
2. ``PostgreSQL`` via ``warehouse.py``    (production warehouse)

Usage
-----
    python transform.py

Environment variables for PostgreSQL (optional — CSV is always saved):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import pandas as pd
import numpy as np
from functools import reduce
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

from api import get_download_link
from download import download_file, extract_zip
from loader import load_statcan_csv
from clean import clean_statcan
from features import create_lag_features, create_rolling_features, create_growth_rate

try:
    from warehouse import get_connection, save_training_data
    WAREHOUSE_AVAILABLE = True
except Exception:
    WAREHOUSE_AVAILABLE = False


# ── Series definitions ──────────────────────────────────────────────────────
# Each entry: (product_id, short_name, dict_of_filters)

SERIES: list[tuple[str, str, Optional[Dict[str, str]]]] = [
    # Target: monthly GDP (chained 2017 $, SAAR, all industries)
    ("36-10-0434-01", "gdp", {
        "Seasonal adjustment": "Seasonally adjusted at annual rates",
        "Prices": "Chained (2017) dollars",
        "North American Industry Classification System (NAICS)": "All industries [T001]",
    }),
    # Predictor: Employment (LFS)
    ("14-10-0287-01", "lfs", {
        "Labour force characteristics": "Employment",
        "Gender": "Total - Gender",
        "Age group": "15 years and over",
        "Statistics": "Estimate",
        "Data type": "Seasonally adjusted",
    }),
    # Predictor: CPI All-items
    ("18-10-0004-01", "cpi", {
        "Products and product groups": "All-items",
    }),
    # Predictor: Manufacturing sales
    ("16-10-0047-01", "mfg", {
        "Principal statistics": "Sales of goods manufactured (shipments)",
        "Seasonal adjustment": "Seasonally adjusted",
        "North American Industry Classification System (NAICS)": "Manufacturing [31-33]",
    }),
]

PROCESSED_DIR = Path("data/processed")
TRAINING_FILE = PROCESSED_DIR / "training_data.csv"


def fetch_table(table_id: str, name: str,
                filters: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Download, extract, clean, and filter one StatCan table."""
    print(f"  [{name}] Downloading {table_id}...", flush=True)
    link = get_download_link(table_id)
    file_ext = link.split(".")[-1].lower()
    local_file = f"data/{table_id}.{file_ext}"

    download_file(link, local_file)

    if file_ext == "zip":
        extract_zip(local_file, f"data/{table_id}")
        df = load_statcan_csv(f"data/{table_id}")
    else:
        df = pd.read_csv(local_file, low_memory=False)

    df = clean_statcan(df, geo_filter="Canada", filters=filters)
    result = df.rename(columns={"value": f"{name}_value"})
    print(f"  [{name}] {result.shape[0]} rows, "
          f"{result['date'].min().date()} -> {result['date'].max().date()}",
          flush=True)
    return result


def build_features(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Add lag, rolling, and growth features for each value column."""
    for col in value_cols:
        df = create_lag_features(df, col, lags=[1, 2, 3])
        df = create_rolling_features(df, col, windows=[3, 6])
        df = create_growth_rate(df, col)
    return df


def run_transform():
    """Orchestrate the full ETL transform pipeline."""

    print("=" * 55)
    print("  ETL - Extract")
    print("=" * 55)

    # ── Extract each series ──────────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}
    for table_id, name, filters in SERIES:
        tables[name] = fetch_table(table_id, name, filters)

    # ── Transform — merge ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ETL - Transform (merge + feature engineering)")
    print("=" * 55)

    # Predictors (outer join — keep the most recent months even when the
    # slowest indicator, manufacturing, hasn't been released yet). This is what
    # lets us produce a genuine forward nowcast for the latest month.
    predictors = [tables["lfs"], tables["cpi"], tables["mfg"]]
    feat_df = reduce(lambda a, b: pd.merge(a, b, on="date", how="outer"),
                     predictors)
    feat_df = feat_df.sort_values("date").reset_index(drop=True)

    # Carry the lagging indicator(s) forward for the trailing gap months so the
    # model has a complete feature row to nowcast on. Forward-fill only (never
    # back-fill — that would leak future information). Flag the frontier months
    # where manufacturing was estimated rather than published.
    predictor_value_cols = ["lfs_value", "cpi_value", "mfg_value"]
    feat_df["mfg_imputed"] = feat_df["mfg_value"].isna() & feat_df["lfs_value"].notna()
    feat_df[predictor_value_cols] = feat_df[predictor_value_cols].ffill()

    # Left-join GDP so we keep months where GDP hasn't been published yet
    df = pd.merge(feat_df, tables["gdp"], on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Merged frame:  {df.shape[0]} rows, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # ── Feature engineering on the predictors only ───────────────────────
    target = "gdp_value"
    predictor_cols = [c for c in df.columns
                      if c not in ["date", target] and c.endswith("_value")]

    df = build_features(df, predictor_cols)

    # Drop rows where feature computation produced NaNs (initial lags/rolls)
    feature_cols = [c for c in df.columns if c not in ["date", target]]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    train_count = df[target].notna().sum()
    print(f"  Feature-engineered: {len(feature_cols)} features")
    print(f"  Total rows:          {df.shape[0]}")
    print(f"  Rows with target:    {train_count}")
    print(f"  Date range:          {df['date'].min().date()} -> "
          f"{df['date'].max().date()}")

    # ── Save to CSV (local dev cache) ────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_FILE, index=False)
    print(f"\n  -> Saved to {TRAINING_FILE}  ({TRAINING_FILE.stat().st_size:,} bytes)")

    # ── Save to PostgreSQL (production warehouse) ────────────────────────
    if WAREHOUSE_AVAILABLE:
        try:
            conn = get_connection()
            save_training_data(conn, df)
            conn.close()
        except Exception as e:
            print(f"  !! Failed to write to PostgreSQL: {e}")
            print("  !! The CSV was still saved — check your DB connection settings.")
    else:
        print("  -- Skipping PostgreSQL (not configured or unavailable)")


if __name__ == "__main__":
    run_transform()