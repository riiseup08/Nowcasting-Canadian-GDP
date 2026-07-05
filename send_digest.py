"""
send_digest.py — Generate and send monthly email digest with LLM analysis
======================================================================
This script is designed to run in a GitHub Action after the monthly pipeline
update. It:

1. Reads the latest data from Supabase (PostgreSQL warehouse)
2. Loads the trained model and metrics
3. Collects nowcast, model comparison, feature importance, and trend data
4. Calls DeepSeek (the Senior Economist agent) to write a professional analysis
5. Generates an HTML email digest
6. Saves it to ``output/digest.html`` for the GitHub Action to send

Usage
-----
    python send_digest.py

Environment variables (set via GitHub Secrets):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD  (Supabase)
    DEEPSEEK_API_KEY                                    (DeepSeek)
    SMTP_FROM, SMTP_TO                                  (email addresses)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

# Load .env for local testing (ignored in CI — GitHub Secrets are used)
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "nowcast_model.pkl"
FEATURE_COLS_FILE = MODELS_DIR / "feature_cols.pkl"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
OUTPUT_DIR = Path("output")

TARGET = "gdp_value"


# ── 1. Load data (Supabase → CSV fallback) ───────────────────────────────────

def _warehouse_configured() -> bool:
    """True if a Supabase/PostgreSQL connection is configured via env."""
    return bool(os.getenv("DATABASE_URL") or os.getenv("DB_HOST"))

def load_data() -> pd.DataFrame:
    """Load the feature-engineered dataset. Try Supabase first, fall back to CSV."""
    CSV_PATH = Path("data/processed/training_data.csv")

    # 1. Try Supabase silently
    if _warehouse_configured():
        try:
            from warehouse import get_connection, load_training_data
            print("  [digest] Loading data from Supabase...")
            conn = get_connection()
            df = load_training_data(conn)
            conn.close()
            df = df.sort_values("date").reset_index(drop=True)
            if len(df):
                return df
        except Exception:
            pass

    # 2. Fall back to CSV — auto-run transform.py if missing
    if not CSV_PATH.exists():
        print("  [digest] CSV not found. Running transform.py to download data...")
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "transform.py"],
            capture_output=True, text=True,
        )
        if not CSV_PATH.exists():
            raise RuntimeError(
                f"transform.py failed to produce {CSV_PATH}.\n{result.stderr[-2000:]}"
            )
    print(f"  [digest] Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_model_artifacts():
    """Load the trained Random Forest model + feature columns + metrics."""
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)
    metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
    return model, feature_cols, metrics


# ── 2. Collect data for the digest ───────────────────────────────────────────

def collect_digest_data(df: pd.DataFrame, model, feature_cols: list, metrics: dict) -> dict:
    """Aggregate all the economic data needed for the digest."""
    train = df[df[TARGET].notna()].copy()
    latest = df.iloc[[-1]]
    latest_date = latest["date"].iloc[0]
    actual = latest[TARGET].iloc[0]
    actual_val = float(actual) if not pd.isna(actual) else None

    # --- Model predictions for latest month ---
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    X = train[feature_cols]
    y = train[TARGET]
    X_now = latest[feature_cols]

    tscv = TimeSeriesSplit(n_splits=4)

    def cv_mae_score(m, X, y):
        scores = []
        for tr, te in tscv.split(X):
            clf = m.__class__(**m.get_params())
            clf.fit(X.iloc[tr], y.iloc[tr])
            scores.append(mean_absolute_error(y.iloc[te], clf.predict(X.iloc[te])))
        return float(np.mean(scores)), 100 * float(np.mean(scores)) / float(y.mean())

    models_list = [
        ("OLS Regression", LinearRegression(), X, X_now),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=42), X, X_now),
    ]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_now_scaled = pd.DataFrame(scaler.transform(X_now), columns=X_now.columns, index=X_now.index)
    models_list.append(("SVR (RBF)", SVR(kernel="rbf", C=100, gamma="scale"), X_scaled, X_now_scaled))
    models_list.append(("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), X, X_now))
    models_list.append(("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X, X_now))

    nowcasts = []
    for name, m, X_tr, X_now_in in models_list:
        cv_mae, cv_pct = cv_mae_score(m, X_tr, y)
        m.fit(X_tr, y)
        pred = float(m.predict(X_now_in)[0])
        err = abs(pred - actual_val) if actual_val is not None else None
        err_pct = 100 * err / actual_val if actual_val is not None and actual_val != 0 else None
        nowcasts.append({
            "name": name,
            "pred": pred,
            "error": err,
            "error_pct": err_pct,
            "cv_mae": cv_mae,
            "cv_pct": cv_pct,
        })

    best_cv = min(nowcasts, key=lambda r: r["cv_mae"])

    # --- Feature importance ---
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    top_features = importance.head(10).to_dict(orient="records")

    # By predictor group
    group_imp = {}
    for prefix, label in [("lfs", "Employment"), ("cpi", "CPI"), ("mfg", "Manufacturing")]:
        group_imp[label] = float(importance[importance["feature"].str.startswith(prefix)]["importance"].sum())

    # --- GDP trend (recent 5 years) ---
    cutoff_5y = train["date"].max() - pd.DateOffset(years=5)
    recent = train[train["date"] >= cutoff_5y].copy()
    gdp_change = float(recent[TARGET].iloc[-1] - recent[TARGET].iloc[0])
    gdp_pct_change = 100 * gdp_change / float(recent[TARGET].iloc[0])

    # --- Nowcast row count ---
    nowcast_rows = int(df[TARGET].isna().sum())

    return {
        "latest_date": str(latest_date.date()),
        "actual_val": actual_val,
        "nowcasts": nowcasts,
        "best_cv_model": best_cv["name"],
        "best_cv_mae": best_cv["cv_mae"],
        "best_cv_pct": best_cv["cv_pct"],
        "top_features": top_features,
        "group_importance": group_imp,
        "gdp_trend_5y_change": gdp_change,
        "gdp_trend_5y_pct": gdp_pct_change,
        "gdp_mean": float(train[TARGET].mean()),
        "training_rows": int(train.shape[0]),
        "total_rows": int(df.shape[0]),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "nowcast_rows": nowcast_rows,
        "model_type": metrics.get("model_type", "RandomForestRegressor"),
        "n_estimators": metrics.get("n_estimators", 100),
        "training_date": metrics.get("training_date", str(datetime.now().date())),
    }


# ── 3. Call DeepSeek to generate the analysis ────────────────────────────────

def call_deepseek(digest_data: dict) -> str:
    """Send the digest data to DeepSeek for a professional economic analysis."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("  [digest] WARNING: DEEPSEEK_API_KEY not set. Using template analysis.")
        return _fallback_analysis(digest_data)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # Build the prompt with all the data
    nowcast_lines = []
    for r in digest_data["nowcasts"]:
        err_str = f" (error: ${r['error']:,.0f}M)" if r['error'] is not None else " (true nowcast)"
        nowcast_lines.append(f"  - {r['name']}: ${r['pred']:,.0f}M{err_str}, CV MAE: ${r['cv_mae']:,.0f}M ({r['cv_pct']:.2f}%)")

    features_lines = []
    for i, feat in enumerate(digest_data["top_features"][:8]):
        features_lines.append(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f} ({100*feat['importance']:.2f}%)")

    group_lines = []
    for label, pct in digest_data["group_importance"].items():
        group_lines.append(f"  - {label}: {100*pct:.1f}%")

    actual_str = f"${digest_data['actual_val']:,.0f}M" if digest_data['actual_val'] is not None else "not yet published"

    prompt = f"""You are a Senior Economist at the Bank of Canada writing a monthly GDP nowcast digest.

Write a professional, insightful analysis (3-4 paragraphs) based on the data below.
Use natural, fluent economic language. Do NOT use markdown formatting.
Highlight key trends, identify the most important drivers, and provide context.

LATEST MONTH: {digest_data['latest_date']}
ACTUAL GDP: {actual_str}
BEST MODEL (by CV MAE): {digest_data['best_cv_model']} (${digest_data['best_cv_mae']:,.0f}M, {digest_data['best_cv_pct']:.2f}%)

ALL MODELS:
{chr(10).join(nowcast_lines)}

TOP FEATURES (predictive power):
{chr(10).join(features_lines)}

FEATURE GROUPS:
{chr(10).join(group_lines)}

GDP TREND (last 5 years): change of ${digest_data['gdp_trend_5y_change']:+,.0f}M ({digest_data['gdp_trend_5y_pct']:+.2f}%)
TRAINING DATA: {digest_data['training_rows']} rows, {digest_data['date_min']} to {digest_data['date_max']}
NOWCAST ROWS (unpublished GDP): {digest_data['nowcast_rows']}

Write the analysis. Be professional, data-driven, and insightful. End with a brief outlook for the coming months."""

    messages = [
        {"role": "system", "content": "You are a Senior Economist at the Bank of Canada. Write professional, clear economic analysis. Use plain text without markdown formatting."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.4,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [digest] DeepSeek API error: {e}")
        return _fallback_analysis(digest_data)


def _fallback_analysis(data: dict) -> str:
    """Generate a basic analysis if DeepSeek is unavailable."""
    actual_str = f"${data['actual_val']:,.0f}M" if data['actual_val'] is not None else "not yet published"
    best = min(data["nowcasts"], key=lambda r: r["cv_mae"])
    return (
        f"According to the latest nowcast for {data['latest_date']}, the Random Forest model "
        f"projects GDP at ${best['pred']:,.0f}M. The actual published GDP is {actual_str}. "
        f"Among all five models, {best['name']} achieves the lowest cross-validated error "
        f"(${best['cv_mae']:,.0f}M, {best['cv_pct']:.2f}% of mean GDP). "
        f"Employment remains the primary economic driver, contributing "
        f"{100 * data['group_importance'].get('Employment', 0):.0f}% of model predictive power, "
        f"followed by CPI trends. Over the last five years, GDP has changed by "
        f"${data['gdp_trend_5y_change']:+,.0f}M ({data['gdp_trend_5y_pct']:+.2f}%)."
    )


# ── 4. Build HTML email ──────────────────────────────────────────────────────

def generate_html(digest_data: dict, llm_analysis: str) -> str:
    """Generate a professional HTML email digest."""
    from html import escape
    # Escape HTML special chars and convert newlines to <br> for email rendering
    analysis_html = escape(llm_analysis).replace("\n", "<br>")

    nowcast = digest_data["nowcasts"]
    best = min(nowcast, key=lambda r: r["cv_mae"])
    rf = [r for r in nowcast if r["name"] == "Random Forest"][0]
    month_name = datetime.strptime(digest_data["latest_date"], "%Y-%m-%d").strftime("%B %Y")

    # Build model table rows
    model_rows = ""
    for r in nowcast:
        err_str = f"${r['error']:,.0f}M" if r['error'] is not None else "N/A"
        err_pct = f"{r['error_pct']:.2f}%" if r['error_pct'] is not None else "N/A"
        star = "🌟 " if r["name"] == best["name"] else ""
        model_rows += f"""
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;">{star}{r['name']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;text-align:right;">${r['pred']:,.0f}M</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;text-align:right;">{err_str}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;text-align:right;">{err_pct}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #e2e8f0;text-align:right;">${r['cv_mae']:,.0f}M</td>
        </tr>"""

    # Build feature importance rows
    feat_rows = ""
    for i, feat in enumerate(digest_data["top_features"][:8]):
        feat_rows += f"""
        <tr>
            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;">{i+1}.</td>
            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;">{feat['feature']}</td>
            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;text-align:right;">{100*feat['importance']:.2f}%</td>
        </tr>"""

    # Feature group bars
    group_bars = ""
    colors = {"Employment": "#2563eb", "CPI": "#059669", "Manufacturing": "#d97706"}
    for label, pct in digest_data["group_importance"].items():
        pct_display = 100 * pct
        color = colors.get(label, "#64748b")
        group_bars += f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;color:#475569;margin-bottom:4px;">
                <span>{label}</span>
                <span>{pct_display:.1f}%</span>
            </div>
            <div style="background:#e2e8f0;border-radius:6px;height:10px;overflow:hidden;">
                <div style="background:{color};height:100%;width:{pct_display}%;border-radius:6px;"></div>
            </div>
        </div>"""

    actual_str = f"${digest_data['actual_val']:,.0f}M" if digest_data['actual_val'] is not None else "Not yet published"
    actual_label = "Actual GDP" if digest_data['actual_val'] is not None else "GDP Status"
    trend_sign = "▲" if digest_data['gdp_trend_5y_change'] >= 0 else "▼"
    trend_color = "#059669" if digest_data['gdp_trend_5y_change'] >= 0 else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GDP Nowcast Digest — {month_name}</title>
</head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f1f5f9;">
    <table width="100%" cellpadding="0" cellspacing="0" style="max-width:680px;margin:0 auto;background:#ffffff;">
        <!-- Header -->
        <tr>
            <td style="background:linear-gradient(135deg,#1e3a5f,#2563eb);padding:32px 40px;border-radius:0 0 0 0;">
                <h1 style="color:#ffffff;margin:0;font-size:28px;">📊 GDP Nowcast Digest</h1>
                <p style="color:#93c5fd;margin:8px 0 0 0;font-size:16px;">Canadian Economy — {month_name}</p>
            </td>
        </tr>

        <!-- Key metrics -->
        <tr>
            <td style="padding:24px 40px 8px 40px;">
                <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                        <td style="width:33%;text-align:center;padding:16px;background:#f8fafc;border-radius:8px;">
                            <p style="margin:0;font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">Nowcast (RF)</p>
                            <p style="margin:8px 0 0 0;font-size:24px;font-weight:700;color:#1e3a5f;">${rf['pred']:,.0f}M</p>
                        </td>
                        <td style="width:33%;text-align:center;padding:16px;background:#f8fafc;border-radius:8px;">
                            <p style="margin:0;font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">{actual_label}</p>
                            <p style="margin:8px 0 0 0;font-size:24px;font-weight:700;color:#1e3a5f;">{actual_str}</p>
                        </td>
                        <td style="width:33%;text-align:center;padding:16px;background:#f8fafc;border-radius:8px;">
                            <p style="margin:0;font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">5-Year Trend</p>
                            <p style="margin:8px 0 0 0;font-size:24px;font-weight:700;color:{trend_color};">{trend_sign} {digest_data['gdp_trend_5y_pct']:.1f}%</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>

        <!-- Analysis section -->
        <tr>
            <td style="padding:16px 40px;">
                <h2 style="font-size:18px;color:#1e3a5f;margin:0 0 12px 0;">🧠 Senior Economist Analysis</h2>
                <div style="background:#f8fafc;border-left:4px solid #2563eb;padding:16px 20px;border-radius:0 8px 8px 0;line-height:1.7;color:#334155;font-size:15px;">
                    {analysis_html}
                </div>
            </td>
        </tr>

        <!-- Model comparison table -->
        <tr>
            <td style="padding:16px 40px;">
                <h2 style="font-size:18px;color:#1e3a5f;margin:0 0 12px 0;">⚖️ Model Comparison</h2>
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:14px;">
                    <thead>
                        <tr style="background:#f1f5f9;">
                            <th style="padding:10px 12px;text-align:left;border-bottom:2px solid #cbd5e1;color:#475569;">Model</th>
                            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #cbd5e1;color:#475569;">Nowcast</th>
                            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #cbd5e1;color:#475569;">Error</th>
                            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #cbd5e1;color:#475569;">Error %</th>
                            <th style="padding:10px 12px;text-align:right;border-bottom:2px solid #cbd5e1;color:#475569;">CV MAE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {model_rows}
                    </tbody>
                </table>
                <p style="font-size:13px;color:#64748b;margin:8px 0 0 0;">🌟 Best model by cross-validated MAE</p>
            </td>
        </tr>

        <!-- Feature Importance -->
        <tr>
            <td style="padding:16px 40px;">
                <h2 style="font-size:18px;color:#1e3a5f;margin:0 0 12px 0;">🔥 Feature Importance (Top 8)</h2>
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:14px;">
                    <thead>
                        <tr style="background:#f1f5f9;">
                            <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #cbd5e1;color:#475569;width:30px;">#</th>
                            <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #cbd5e1;color:#475569;">Feature</th>
                            <th style="padding:8px 12px;text-align:right;border-bottom:2px solid #cbd5e1;color:#475569;">Weight</th>
                        </tr>
                    </thead>
                    <tbody>
                        {feat_rows}
                    </tbody>
                </table>
                <div style="margin-top:16px;padding:16px;background:#f8fafc;border-radius:8px;">
                    <p style="margin:0 0 8px 0;font-size:14px;font-weight:600;color:#475569;">Predictor Group Contribution</p>
                    {group_bars}
                </div>
            </td>
        </tr>

        <!-- Data summary -->
        <tr>
            <td style="padding:16px 40px;">
                <h2 style="font-size:18px;color:#1e3a5f;margin:0 0 12px 0;">📋 Data Summary</h2>
                <table width="100%" cellpadding="0" cellspacing="0" style="font-size:14px;">
                    <tr>
                        <td style="padding:6px 0;color:#475569;">Training rows</td>
                        <td style="padding:6px 0;text-align:right;font-weight:600;color:#1e293b;">{digest_data['training_rows']:,}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;color:#475569;">Date range</td>
                        <td style="padding:6px 0;text-align:right;font-weight:600;color:#1e293b;">{digest_data['date_min']} → {digest_data['date_max']}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;color:#475569;">Model</td>
                        <td style="padding:6px 0;text-align:right;font-weight:600;color:#1e293b;">{digest_data['model_type']} ({digest_data['n_estimators']} trees)</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;color:#475569;">Last trained</td>
                        <td style="padding:6px 0;text-align:right;font-weight:600;color:#1e293b;">{digest_data['training_date']}</td>
                    </tr>
                </table>
            </td>
        </tr>

        <!-- Footer -->
        <tr>
            <td style="padding:24px 40px;background:#f1f5f9;border-top:1px solid #e2e8f0;">
                <p style="margin:0;font-size:12px;color:#64748b;text-align:center;">
                    📊 <strong>GDP Nowcast Dashboard</strong><br>
                    Data source: Statistics Canada Web Data Service (WDS) API<br>
                    Predictors: Employment (LFS), CPI (All-items), Manufacturing Sales (Mfg)<br>
                    Models: OLS, Ridge, SVR, Gradient Boosting, Random Forest<br>
                    <br>
                    This digest was generated automatically on {datetime.now().strftime("%B %d, %Y at %H:%M UTC")}.
                </p>
            </td>
        </tr>
    </table>
</body>
</html>"""


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    """Generate the email digest and save to output/digest.html."""
    print("=" * 50)
    print("  GDP Nowcast — Monthly Email Digest Generator")
    print("=" * 50)

    # 1. Load data
    print("\n  [1/5] Loading data...")
    df = load_data()
    print(f"  -> {df.shape[0]} rows loaded")

    # 2. Load model
    print("  [2/5] Loading model artifacts...")
    model, feature_cols, metrics = load_model_artifacts()
    print(f"  -> {len(feature_cols)} features, model trained: {metrics.get('training_date', 'N/A')}")

    # 3. Collect digest data
    print("  [3/5] Collecting economic data...")
    digest_data = collect_digest_data(df, model, feature_cols, metrics)
    print(f"  -> Latest month: {digest_data['latest_date']}")
    print(f"  -> {len(digest_data['nowcasts'])} models evaluated")

    # 4. Call DeepSeek
    print("  [4/5] Generating LLM-powered analysis...")
    llm_analysis = call_deepseek(digest_data)
    print(f"  -> Analysis generated ({len(llm_analysis)} chars)")

    # 5. Generate HTML
    print("  [5/5] Generating HTML digest...")
    html = generate_html(digest_data, llm_analysis)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "digest.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  ✅ Digest saved to {output_path}  ({output_path.stat().st_size:,} bytes)")
    print("=" * 50)


if __name__ == "__main__":
    main()