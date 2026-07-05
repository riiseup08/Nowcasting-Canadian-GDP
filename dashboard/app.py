"""
dashboard/app.py — GDP Nowcast Streamlit Dashboard
==================================================
Interactive dashboard for the Canadian GDP nowcast model.

Usage
-----
    streamlit run dashboard/app.py

The dashboard loads the pre-trained Random Forest model from ``models/``
and the feature-engineered dataset from ``data/processed/training_data.csv``.
All 5 models (OLS, Ridge, SVR, Gradient Boosting, Random Forest) are evaluated
on the latest month, plotted over time, and compared side-by-side.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib
import json
import subprocess
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Senior Economist Agent (lazy import) ────────────────────────────────────
_agent_available = False
_agent_error = None
try:
    from agent.graph import run_agent
    _agent_available = True
except Exception as e:
    _agent_error = str(e)

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "nowcast_model.pkl"
FEATURE_COLS_FILE = MODELS_DIR / "feature_cols.pkl"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
CSV_PATH = Path("data/processed/training_data.csv")
TARGET = "gdp_value"


# ── Cached helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained Random Forest model + feature columns + metrics."""
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)
    metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
    return model, feature_cols, metrics


def _warehouse_configured() -> bool:
    """True if a Supabase/PostgreSQL connection is configured via env/secrets."""
    return bool(os.getenv("DATABASE_URL") or os.getenv("DB_HOST"))


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the feature-engineered dataset.

    Prefers the Supabase/PostgreSQL warehouse (used in production / Streamlit
    Cloud). Falls back to the local CSV cache for offline development.
    """
    # 1. Try the warehouse first when a connection is configured (silently)
    if _warehouse_configured():
        try:
            from warehouse import get_connection, load_training_data
            conn = get_connection()
            df = load_training_data(conn)
            conn.close()
            df = df.sort_values("date").reset_index(drop=True)
            if len(df):
                return df
        except Exception:
            pass  # Warehouse unavailable — fall through to CSV

    # 2. Fall back to the local CSV cache — auto-run transform.py if missing
    if not CSV_PATH.exists():
        with st.spinner("⏳ Downloading live data from Statistics Canada..."):
            result = subprocess.run(
                [sys.executable, "transform.py"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True,
            )
        if not CSV_PATH.exists():
            st.error(
                f"`transform.py` failed to produce {CSV_PATH}.\n\n"
                f"**stdout:**\n```\n{result.stdout[-2000:]}\n```\n\n"
                f"**stderr:**\n```\n{result.stderr[-2000:]}\n```"
            )
            st.stop()
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    return df


def compute_all_models(df: pd.DataFrame, feature_cols: list):
    """
    Fit 5 models and return:
      - nowcasts: list of dicts (name, pred, cv_mae, cv_pct) for latest month
      - predictions_df: DataFrame with date + one column per model's in-sample predictions
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    train = df[df[TARGET].notna()]
    X = train[feature_cols]
    y = train[TARGET]
    dates = train["date"]
    latest = df.iloc[[-1]]
    X_now = latest[feature_cols]

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
    predictions = pd.DataFrame({"date": dates})

    # 1. OLS
    ols = LinearRegression()
    cv_mae_val, cv_pct_val = cv_mae_score(ols, X, y)
    ols.fit(X, y)
    nowcasts.append({"name": "OLS Regression", "pred": float(ols.predict(X_now)[0]),
                     "cv_mae": cv_mae_val, "cv_pct": cv_pct_val})
    predictions["OLS Regression"] = ols.predict(X)

    # 2. Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    cv_mae_val, cv_pct_val = cv_mae_score(ridge, X, y)
    ridge.fit(X, y)
    nowcasts.append({"name": "Ridge Regression", "pred": float(ridge.predict(X_now)[0]),
                     "cv_mae": cv_mae_val, "cv_pct": cv_pct_val})
    predictions["Ridge Regression"] = ridge.predict(X)

    # 3. SVR (scaled)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_now_scaled = pd.DataFrame(scaler.transform(X_now), columns=X_now.columns, index=X_now.index)
    svr = SVR(kernel="rbf", C=100, gamma="scale")
    cv_mae_val, cv_pct_val = cv_mae_score(svr, X_scaled, y)
    svr.fit(X_scaled, y)
    nowcasts.append({"name": "SVR (RBF)", "pred": float(svr.predict(X_now_scaled)[0]),
                     "cv_mae": cv_mae_val, "cv_pct": cv_pct_val})
    predictions["SVR (RBF)"] = svr.predict(X_scaled)

    # 4. Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
    cv_mae_val, cv_pct_val = cv_mae_score(gbr, X, y)
    gbr.fit(X, y)
    nowcasts.append({"name": "Gradient Boosting", "pred": float(gbr.predict(X_now)[0]),
                     "cv_mae": cv_mae_val, "cv_pct": cv_pct_val})
    predictions["Gradient Boosting"] = gbr.predict(X)

    # 5. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_mae_val, cv_pct_val = cv_mae_score(rf, X, y)
    rf.fit(X, y)
    nowcasts.append({"name": "Random Forest", "pred": float(rf.predict(X_now)[0]),
                     "cv_mae": cv_mae_val, "cv_pct": cv_pct_val})
    predictions["Random Forest"] = rf.predict(X)

    return nowcasts, predictions


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GDP Nowcast — Canada",
    page_icon="📊",
    layout="wide",
)

st.title("📊 GDP Nowcast — Canada")
st.markdown(
    "Monthly nowcast of Canadian gross domestic product (chained 2017 dollars, "
    "seasonally adjusted at annual rates) using **employment**, **CPI**, and "
    "**manufacturing sales** as predictors.  "  
    "Five models are compared: OLS, Ridge, SVR, Gradient Boosting, Random Forest."
)

# ── Load artifacts ───────────────────────────────────────────────────────────
model, feature_cols, metrics = load_model()  # RF model for feature importance
df = load_data()

# ── Compute all models ───────────────────────────────────────────────────────
all_nowcasts, pred_df = compute_all_models(df, feature_cols)
rf_result = [r for r in all_nowcasts if r["name"] == "Random Forest"][0]
rf_nowcast = rf_result["pred"]
ols_result = [r for r in all_nowcasts if r["name"] == "OLS Regression"][0]
ols_pred = ols_result["pred"]

latest = df.iloc[[-1]]
actual = latest[TARGET].iloc[0]
actual_val = float(actual) if not pd.isna(actual) else None
latest_date = latest["date"].iloc[0]
if hasattr(latest_date, "date"):
    latest_date = latest_date.date()

train = df[df[TARGET].notna()].copy()

# ── Metric cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label=f"Best Nowcast — {latest_date}",
        value=f"${rf_nowcast:,.0f}M",
        delta=None if actual_val is None else f"Actual: ${actual_val:,.0f}M",
        delta_color="off",
    )

with col2:
    if metrics:
        cv_mae = metrics.get("cv_avg_mae", 0)
        cv_pct = metrics.get("cv_mae_pct", 0)
        st.metric(label="RF CV Avg MAE", value=f"${cv_mae:,.0f}M", delta=f"{cv_pct:.2f}% of mean GDP")
    else:
        st.metric(label="RF CV Avg MAE", value="Run train_model.py first")

with col3:
    st.metric(
        label="OLS Baseline",
        value=f"${ols_pred:,.0f}M",
        delta=None if actual_val is None else f"Error: ${abs(ols_pred - actual_val):,.0f}M",
        delta_color="off",
    )

with col4:
    if actual_val is not None:
        rf_err = abs(rf_nowcast - actual_val)
        st.metric(label=f"RF Error", value=f"${rf_err:,.0f}M", delta=f"{100*rf_err/actual_val:.2f}%")
    else:
        st.info("GDP not yet published — true forward nowcast.")

# ── Chart: all models over time ──────────────────────────────────────────────
st.subheader("Actual vs All Model Predictions")

model_colors = {
    "OLS Regression": "#1f77b4",
    "Ridge Regression": "#ff7f0e",
    "SVR (RBF)": "#2ca02c",
    "Gradient Boosting": "#d62728",
    "Random Forest": "#9467bd",
}

fig = go.Figure()

# Actual GDP line (boldest)
fig.add_trace(go.Scatter(
    x=train["date"], y=train[TARGET],
    mode="lines", name="Actual GDP",
    line=dict(color="black", width=3),
))

# One line per model
for name in ["OLS Regression", "Ridge Regression", "SVR (RBF)",
             "Gradient Boosting", "Random Forest"]:
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df[name],
        mode="lines", name=name,
        line=dict(color=model_colors[name], width=1.5, dash="dash"),
    ))

# Nowcast star marker (use RF as the primary nowcast)
fig.add_trace(go.Scatter(
    x=latest["date"], y=[rf_nowcast],
    mode="markers", name=f"Nowcast ({latest_date})",
    marker=dict(color="red", size=14, symbol="star"),
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="GDP (chained 2017 $M)",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ── Feature importance ───────────────────────────────────────────────────────
st.subheader("🔥 Feature Importance (Top 10) — Random Forest")

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).head(10)

fig_imp = px.bar(
    importance, x="importance", y="feature",
    orientation="h", text_auto=".1%",
    color="importance", color_continuous_scale="blues",
)
fig_imp.update_layout(
    yaxis=dict(autorange="reversed"),
    height=400,
    xaxis_title="Importance",
    yaxis_title="",
)
st.plotly_chart(fig_imp, use_container_width=True)

# ── Model comparison table ───────────────────────────────────────────────────
st.subheader("⚖️  Model Comparison (Latest Month)")

comp_data = []
for r in all_nowcasts:
    err = abs(r["pred"] - actual_val) if actual_val is not None else None
    err_pct = 100 * err / actual_val if actual_val is not None and actual_val != 0 else None
    comp_data.append({
        "Model": r["name"],
        "Nowcast ($M)": f"${r['pred']:,.0f}",
        "Error ($M)": f"${err:,.0f}" if err is not None else "N/A",
        "Error (%)": f"{err_pct:.2f}%" if err_pct is not None else "N/A",
        "CV MAE ($M)": f"${r['cv_mae']:,.0f}",
        "CV MAE (%)": f"{r['cv_pct']:.2f}%",
    })

comp_df = pd.DataFrame(comp_data)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# Find best model by CV MAE
best = min(all_nowcasts, key=lambda r: r["cv_mae"])
st.info(f"🏆 **Best model by CV MAE:** {best['name']} (${best['cv_mae']:,.0f}M, {best['cv_pct']:.2f}%)")

# ── TimeSeries CV details ────────────────────────────────────────────────────
st.subheader("📈 TimeSeries Cross-Validation Summary (Random Forest)")

if metrics:
    cv_info = [
        ("Training rows", f"{metrics.get('training_rows', 'N/A')}"),
        ("Features", f"{metrics.get('n_features', 'N/A')}"),
        ("Date range", f"{metrics.get('date_min', 'N/A')} to {metrics.get('date_max', 'N/A')}"),
        ("CV Avg MAE", f"${metrics.get('cv_avg_mae', 0):,.0f}M"),
        ("CV MAE (% of mean GDP)", f"{metrics.get('cv_mae_pct', 0):.2f}%"),
        ("Model type", f"{metrics.get('model_type', 'N/A')} ({metrics.get('n_estimators', 100)} trees)"),
        ("Last trained", metrics.get("training_date", "N/A")),
    ]
    for label, value in cv_info:
        st.write(f"- **{label}:** {value}")
else:
    st.info("Run `python train_model.py` to generate cross-validation metrics.")

# ── Data info & actions ──────────────────────────────────────────────────────
with st.expander("📋 Data summary & actions"):
    st.write(f"**Rows:** {df.shape[0]}  |  **Features:** {len(feature_cols)}")
    st.write(f"**Date range:** {df['date'].min().date()} to {df['date'].max().date()}")
    st.write(f"**Training rows:** {train.shape[0]} (GDP published)")
    st.write(f"**Nowcast rows:** {df[TARGET].isna().sum()} (GDP not yet published)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Refresh data (re-run transform)"):
            with st.spinner("Running transform.py..."):
                subprocess.run([sys.executable, "transform.py"],
                               cwd=Path(__file__).resolve().parent.parent)
            st.success("Data refreshed! Reloading...")
            st.cache_data.clear()
            st.rerun()

    with c2:
        if st.button("🔁 Retrain RF model"):
            with st.spinner("Training Random Forest with TimeSeries CV..."):
                subprocess.run([sys.executable, "train_model.py"],
                               cwd=Path(__file__).resolve().parent.parent)
            st.success("Model retrained! Reloading...")
            st.cache_resource.clear()
            st.rerun()

# ── Senior Economist Agent — Chat Sidebar ────────────────────────────────────
st.sidebar.title("💬 Senior Economist")
st.sidebar.markdown(
    "Ask questions about the GDP nowcast, model performance, "
    "economic trends, and more. Powered by **DeepSeek** + LangGraph."
)

if not _agent_available:
    st.sidebar.error(
        f"Agent failed to load. {_agent_error}\n\n"
        "Make sure `DEEPSEEK_API_KEY` is set in your `.env` file."
    )
else:
    # Initialise chat history
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Display chat messages
    for msg in st.session_state.agent_messages:
        with st.sidebar.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.sidebar.chat_input("Ask the Senior Economist..."):
        # Add user message
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.sidebar.chat_message("assistant"):
            with st.spinner("Analysing data..."):
                try:
                    response_text, updated_history = run_agent(
                        user_query=prompt,
                        history=st.session_state.agent_history,
                    )
                    st.session_state.agent_history = updated_history
                    st.markdown(response_text)
                    st.session_state.agent_messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    st.error(error_msg)
                    st.session_state.agent_messages.append({"role": "assistant", "content": error_msg})

    # Clear button
    if st.sidebar.button("🗑️ Clear conversation"):
        st.session_state.agent_history = []
        st.session_state.agent_messages = []
        st.rerun()

st.markdown("---")
st.caption(
    "Data source: Statistics Canada Web Data Service (WDS) API. "
    "Predictors: Labour Force Survey (employment), Consumer Price Index (all-items), "
    "Monthly Survey of Manufacturing (sales).  "
    "Models: OLS, Ridge, SVR (RBF), Gradient Boosting, Random Forest (100 trees).  "
    "Agent: Senior Economist powered by DeepSeek + LangGraph."
)
