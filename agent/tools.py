"""
agent/tools.py — Tool functions for the Senior Economist agent
==============================================================
Each tool is a function decorated with @tool (from langchain_core.tools)
that the LangGraph agent can call to retrieve data about the GDP nowcast.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

# ── Paths (same as dashboard) ─────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "nowcast_model.pkl"
FEATURE_COLS_FILE = MODELS_DIR / "feature_cols.pkl"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
CSV_PATH = Path("data/processed/training_data.csv")
TARGET = "gdp_value"

# ── Lazy-loaded globals (loaded once per process) ────────────────────────────
_model = None
_feature_cols = None
_metrics = None
_df = None
_train = None


def _load_artifacts():
    global _model, _feature_cols, _metrics, _df, _train
    if _model is not None:
        return
    _model = joblib.load(MODEL_FILE)
    _feature_cols = joblib.load(FEATURE_COLS_FILE)
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            _metrics = json.load(f)
    _df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    _train = _df[_df[TARGET].notna()].copy()


def _run_all_models():
    """Fit 5 models and return (nowcasts_list, predictions_df)."""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    _load_artifacts()
    train = _train
    X = train[_feature_cols]
    y = train[TARGET]
    latest = _df.iloc[[-1]]
    X_now = latest[_feature_cols]

    tscv = TimeSeriesSplit(n_splits=4)

    def cv_mae_score(model, X, y):
        scores = []
        for tr, te in tscv.split(X):
            m = model.__class__(**model.get_params())
            m.fit(X.iloc[tr], y.iloc[tr])
            preds = m.predict(X.iloc[te])
            scores.append(mean_absolute_error(y.iloc[te], preds))
        avg = float(np.mean(scores))
        pct = 100 * avg / float(y.mean())
        return avg, pct

    nowcasts = []
    predictions = pd.DataFrame({"date": train["date"]})

    models = [
        ("OLS Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=42)),
    ]
    # SVR needs scaled data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_now_scaled = pd.DataFrame(scaler.transform(X_now), columns=X_now.columns, index=X_now.index)

    for name, model_obj in models:
        cv_mae_v, cv_pct_v = cv_mae_score(model_obj, X, y)
        model_obj.fit(X, y)
        nowcasts.append({"name": name, "pred": float(model_obj.predict(X_now)[0]),
                         "cv_mae": cv_mae_v, "cv_pct": cv_pct_v})
        predictions[name] = model_obj.predict(X)

    # SVR
    svr = SVR(kernel="rbf", C=100, gamma="scale")
    cv_mae_v, cv_pct_v = cv_mae_score(svr, X_scaled, y)
    svr.fit(X_scaled, y)
    nowcasts.append({"name": "SVR (RBF)", "pred": float(svr.predict(X_now_scaled)[0]),
                     "cv_mae": cv_mae_v, "cv_pct": cv_pct_v})
    predictions["SVR (RBF)"] = svr.predict(X_scaled)

    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    cv_mae_v, cv_pct_v = cv_mae_score(gbr, X, y)
    gbr.fit(X, y)
    nowcasts.append({"name": "Gradient Boosting", "pred": float(gbr.predict(X_now)[0]),
                     "cv_mae": cv_mae_v, "cv_pct": cv_pct_v})
    predictions["Gradient Boosting"] = gbr.predict(X)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_mae_v, cv_pct_v = cv_mae_score(rf, X, y)
    rf.fit(X, y)
    nowcasts.append({"name": "Random Forest", "pred": float(rf.predict(X_now)[0]),
                     "cv_mae": cv_mae_v, "cv_pct": cv_pct_v})
    predictions["Random Forest"] = rf.predict(X)

    return nowcasts, predictions


# ── Public tool functions ─────────────────────────────────────────────────────

def get_nowcast() -> str:
    """Return the current GDP nowcast for all 5 models."""
    _load_artifacts()
    nowcasts, _ = _run_all_models()
    latest = _df.iloc[[-1]]
    actual = latest[TARGET].iloc[0]
    actual_val = float(actual) if not pd.isna(actual) else None
    latest_date = latest["date"].iloc[0].date()

    lines = [f"**GDP Nowcast for {latest_date}**\n"]
    if actual_val is not None:
        lines.append(f"Actual published GDP: ${actual_val:,.0f}M\n")
    else:
        lines.append("Actual GDP: Not yet published (true forward nowcast)\n")

    for r in nowcasts:
        err = abs(r["pred"] - actual_val) if actual_val is not None else None
        err_str = f" (error: ${err:,.0f}M, {100*err/actual_val:.2f}%)" if err is not None else ""
        lines.append(f"- **{r['name']}**: ${r['pred']:,.0f}M{err_str}")

    # Best model
    best = min(nowcasts, key=lambda r: r["cv_mae"])
    lines.append(f"\n🏆 **Best model by CV MAE:** {best['name']} (${best['cv_mae']:,.0f}M, {best['cv_pct']:.2f}%)")
    return "\n".join(lines)


def get_model_comparison() -> str:
    """Return a full comparison table of all 5 models."""
    _load_artifacts()
    nowcasts, _ = _run_all_models()
    latest = _df.iloc[[-1]]
    actual = latest[TARGET].iloc[0]
    actual_val = float(actual) if not pd.isna(actual) else None

    lines = ["## Model Comparison\n"]
    lines.append(f"| {'Model':<22} | {'Nowcast':>12} | {'Error':>12} | {'Error %':>8} | {'CV MAE':>12} | {'CV MAE %':>8} |")
    lines.append("|" + "-" * 24 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 10 + "|")

    for r in nowcasts:
        err = abs(r["pred"] - actual_val) if actual_val is not None else float("nan")
        err_pct = 100 * err / actual_val if actual_val is not None and actual_val != 0 else float("nan")
        err_str = f"${err:,.0f}M" if not np.isnan(err) else "N/A"
        pct_str = f"{err_pct:.2f}%" if not np.isnan(err_pct) else "N/A"
        lines.append(f"| {r['name']:<22} | ${r['pred']:>9,.0f}M | {err_str:>12} | {pct_str:>8} | ${r['cv_mae']:>9,.0f}M | {r['cv_pct']:>7.2f}% |")

    best = min(nowcasts, key=lambda r: r["cv_mae"])
    lines.append(f"\n🏆 **Best model by CV MAE:** {best['name']}")
    return "\n".join(lines)


def get_feature_importance() -> str:
    """Return the top features driving the Random Forest model."""
    _load_artifacts()
    imp = pd.DataFrame({
        "feature": _feature_cols,
        "importance": _model.feature_importances_,
    }).sort_values("importance", ascending=False)

    lines = ["## Top 10 Features Driving GDP\n"]
    for i, row in imp.head(10).iterrows():
        lines.append(f"  {i+1}. **{row['feature']}** — {row['importance']:.4f} ({100*row['importance']:.2f}%)")

    # Group-level
    lines.append("\n### By predictor group:")
    for prefix, label in [("lfs", "Employment"), ("cpi", "CPI"), ("mfg", "Manufacturing")]:
        group_imp = imp[imp["feature"].str.startswith(prefix)]["importance"].sum()
        lines.append(f"  - **{label}**: {group_imp:.4f} ({100*group_imp:.2f}% of total)")

    lines.append("\n💡 **Interpretation:** Employment (LFS) dominates the model, "
                 "followed by CPI trends. Manufacturing has a smaller but meaningful contribution.")
    return "\n".join(lines)


def get_gdp_trend(period: str = "all") -> str:
    """Return historical GDP trend summary. Period can be 'all', '5y', '10y', '3y'."""
    _load_artifacts()
    train = _train.copy()

    if period == "5y":
        cutoff = train["date"].max() - pd.DateOffset(years=5)
        train = train[train["date"] >= cutoff]
    elif period == "10y":
        cutoff = train["date"].max() - pd.DateOffset(years=10)
        train = train[train["date"] >= cutoff]
    elif period == "3y":
        cutoff = train["date"].max() - pd.DateOffset(years=3)
        train = train[train["date"] >= cutoff]

    if train.empty:
        return "No data available for the requested period."

    latest_gdp = train[TARGET].iloc[-1]
    earliest_gdp = train[TARGET].iloc[0]
    change = latest_gdp - earliest_gdp
    pct_change = 100 * change / earliest_gdp
    avg = train[TARGET].mean()
    min_gdp = train[TARGET].min()
    max_gdp = train[TARGET].max()

    lines = [
        f"## GDP Trend ({period.upper() if period != 'all' else 'ALL'})",
        f"  **Period:** {train['date'].min().date()} to {train['date'].max().date()}",
        f"  **Start:** ${earliest_gdp:,.0f}M",
        f"  **End:** ${latest_gdp:,.0f}M",
        f"  **Change:** ${change:+,.0f}M ({pct_change:+.2f}%)",
        f"  **Average:** ${avg:,.0f}M",
        f"  **Min:** ${min_gdp:,.0f}M",
        f"  **Max:** ${max_gdp:,.0f}M",
    ]
    return "\n".join(lines)


def get_data_summary() -> str:
    """Return a summary of the training dataset."""
    _load_artifacts()
    latest = _df.iloc[[-1]]
    actual = latest[TARGET].iloc[0]
    actual_val = float(actual) if not pd.isna(actual) else None
    nowcast_rows = _df[TARGET].isna().sum()

    lines = [
        "## Data Summary",
        f"  **Total rows:** {_df.shape[0]}",
        f"  **Features:** {len(_feature_cols)}",
        f"  **Date range:** {_df['date'].min().date()} to {_df['date'].max().date()}",
        f"  **Training rows (GDP published):** {_train.shape[0]}",
        f"  **Nowcast rows (GDP missing):** {nowcast_rows}",
        f"  **Latest month:** {latest['date'].iloc[0].date()}",
    ]
    if actual_val is not None:
        lines.append(f"  **Latest GDP:** ${actual_val:,.0f}M")
    else:
        lines.append(f"  **Latest GDP:** Not yet published")

    if _metrics:
        lines.append(f"\n### Model Info (Random Forest)")
        lines.append(f"  **CV Avg MAE:** ${_metrics['cv_avg_mae']:,.0f}M ({_metrics['cv_mae_pct']:.2f}%)")
        lines.append(f"  **Trained:** {_metrics['training_date']}")
        lines.append(f"  **Trees:** {_metrics['n_estimators']}")
    return "\n".join(lines)