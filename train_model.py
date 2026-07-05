"""
train_model.py — Model training step
====================================
Reads the feature-engineered dataset (from CSV or PostgreSQL warehouse),
trains a Random Forest nowcast model with TimeSeries cross-validation,
and saves the trained model + feature column names to ``models/``.

Usage
-----
    python train_model.py              # reads from CSV (local dev)
    python train_model.py --warehouse  # reads from PostgreSQL (production)

Output
------
- ``models/nowcast_model.pkl``   — trained RandomForestRegressor (joblib)
- ``models/feature_cols.pkl``    — list of feature column names (joblib)
- ``models/model_metrics.json``  — CV MAE, MAE%, date range (JSON)
"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "nowcast_model.pkl"
FEATURE_COLS_FILE = MODELS_DIR / "feature_cols.pkl"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

TARGET = "gdp_value"


def load_data(use_warehouse: bool = False) -> pd.DataFrame:
    """Load the feature-engineered dataset from CSV or PostgreSQL."""
    if use_warehouse:
        from warehouse import get_connection, load_training_data
        print("  Loading from PostgreSQL warehouse...")
        conn = get_connection()
        df = load_training_data(conn)
        conn.close()
    else:
        csv_path = Path("data/processed/training_data.csv")
        print(f"  Loading from CSV ({csv_path})...")
        df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def train_model(df: pd.DataFrame):
    """Train a Random Forest model with TimeSeries CV and save artifacts."""

    feature_cols = [c for c in df.columns if c not in ["date", TARGET]]

    # Training set = rows where GDP target is published
    train = df[df[TARGET].notna()].copy()
    X = train[feature_cols]
    y = train[TARGET]

    print(f"\n  Training rows: {X.shape[0]}")
    print(f"  Features:      {len(feature_cols)}")
    print(f"  Date range:    {train['date'].min().date()} to {train['date'].max().date()}")

    # ── TimeSeries Cross-Validation ──────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=4)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = []

    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        scores.append(mean_absolute_error(y.iloc[te], preds))

    avg_mae = float(np.mean(scores))
    mae_pct = 100 * avg_mae / float(y.mean())

    # ── Refit on all training data ───────────────────────────────────────
    model.fit(X, y)

    # ── Save artifacts ───────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_cols, FEATURE_COLS_FILE)

    metrics = {
        "training_date": str(date.today()),
        "training_rows": X.shape[0],
        "n_features": len(feature_cols),
        "cv_avg_mae": avg_mae,
        "cv_mae_pct": mae_pct,
        "date_min": str(train["date"].min().date()),
        "date_max": str(train["date"].max().date()),
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Nowcast on latest row ────────────────────────────────────────────
    latest = df.iloc[[-1]]
    nowcast = float(model.predict(latest[feature_cols])[0])
    actual = latest[TARGET].iloc[0]
    actual_val = float(actual) if not pd.isna(actual) else None

    print(f"\n  {'=' * 45}")
    kind = "NOWCAST (GDP not yet published)" if actual_val is None else "LATEST-MONTH ESTIMATE"
    print(f"  {kind} for {latest['date'].iloc[0].date()}: {nowcast:,.0f}")
    if actual_val is not None:
        print(f"  Actual published:  {actual_val:,.0f}  (error {abs(nowcast - actual_val):,.0f})")
    print(f"  CV Avg MAE:        {avg_mae:,.1f}  ({mae_pct:.2f}% of mean GDP)")
    print(f"  {'=' * 45}")

    print(f"\n  -> Model saved:      {MODEL_FILE}")
    print(f"  -> Feature cols:     {FEATURE_COLS_FILE}")
    print(f"  -> Metrics:          {METRICS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GDP nowcast model")
    parser.add_argument("--warehouse", action="store_true",
                        help="Load data from PostgreSQL warehouse instead of CSV")
    args = parser.parse_args()

    df = load_data(use_warehouse=args.warehouse)
    train_model(df)