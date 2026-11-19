"""
dashboard/streamlit_app.py
--------------------------
MODULE : Monitoring Dashboard
PURPOSE: Production-grade Streamlit monitoring app for the Self-Healing ML Pipeline.

FEATURES:
  - KPI cards (drift score, drifted features, samples, malignant rate, confidence)
  - Plotly gauge chart for drift score
  - Plotly feature drift bar chart (p-values per feature)
  - Plotly drift history line chart with threshold annotation
  - Plotly prediction probability histogram
  - Prediction distribution donut chart
  - Data validation panel
  - Training history (from training_meta.json)
  - Colour-coded pipeline log viewer
  - Sidebar: run pipeline, retrain, live config display

Run with:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import logging
import json
import subprocess
from datetime import datetime

import pandas    as pd
import numpy     as np
import plotly.graph_objects as go
import plotly.express       as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from pipeline.data_loader     import DataLoader
from pipeline.data_validator  import DataValidator
from pipeline.drift_detector  import DriftDetector
from pipeline.model_predictor import ModelPredictor
from pipeline.decision_engine import DecisionEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "Self-Healing ML Pipeline",
    page_icon             = "🧬",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 60%, #0e1520 100%);
    color: #e6edf3;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(88,166,255,0.1);
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem !important;
    color: #58a6ff !important;
    font-weight: 700;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.status-ok      { background:#0d3321; color:#3fb950; border:1px solid #238636; }
.status-warn    { background:#2d1f00; color:#d29922; border:1px solid #9e6a03; }
.status-retrain { background:#3d1c1c; color:#f85149; border:1px solid #da3633; }

/* ── Section headers ── */
.section-header {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #8b949e;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 6px;
}

/* ── Glass panel ── */
.glass-panel {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px 22px;
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
}

/* ── Log box ── */
.log-box {
    background: #0a0e1a;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    color: #8b949e;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.75;
}
.log-INFO    { color: #58a6ff; }
.log-WARNING { color: #d29922; }
.log-ERROR   { color: #f85149; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(56,139,253,0.3) !important;
}

/* ── Feature row ── */
.feature-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #161b22;
    font-size: 0.75rem;
}
.feature-name { flex:1; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }
.feature-stat { width: 72px; text-align: right; color: #8b949e; font-family: 'JetBrains Mono', monospace; }
.dot-drifted  { color: #f85149; font-size: 0.9rem; }
.dot-ok       { color: #3fb950; font-size: 0.9rem; }

/* ── Page title gradient ── */
.gradient-title {
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 40%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
}
.subtitle { color: #8b949e; font-size: 0.88rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_status_badge(action: str) -> str:
    """Return an HTML badge string for the given action."""
    if action == "continue":
        return '<span class="status-badge status-ok">✓ Healthy</span>'
    elif action == "retrain":
        return '<span class="status-badge status-retrain">⚡ Retraining Required</span>'
    return '<span class="status-badge status-warn">⚠ Unknown</span>'


def load_log_lines(n: int = 40) -> list:
    """Return the last N lines from pipeline.log."""
    if not os.path.exists(config.PIPELINE_LOG):
        return []
    with open(config.PIPELINE_LOG, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]


def colorize_log(line: str) -> str:
    """Wrap log line in a coloured HTML span."""
    line = line.strip().replace("<", "&lt;").replace(">", "&gt;")
    if "ERROR"   in line: return f'<span class="log-ERROR">{line}</span>'
    if "WARNING" in line: return f'<span class="log-WARNING">{line}</span>'
    return f'<span class="log-INFO">{line}</span>'


def load_training_history() -> list:
    """Load training run history from training_meta.json."""
    meta_file = getattr(
        config, "TRAINING_META_FILE",
        os.path.join(config.LOGS_DIR, "training_meta.json")
    )
    if not os.path.exists(meta_file):
        return []
    try:
        with open(meta_file, "r") as f:
            history = json.load(f)
        return history if isinstance(history, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Inter", color="#8b949e", size=11),
    margin        = dict(l=10, r=10, t=30, b=10),
)


def build_drift_gauge(drift_score: float) -> go.Figure:
    """Animated gauge showing drift score vs threshold."""
    color = (
        "#3fb950" if drift_score < 0.2 else
        "#d29922" if drift_score < config.DRIFT_THRESHOLD else
        "#f85149"
    )
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = drift_score,
        delta = {"reference": config.DRIFT_THRESHOLD, "valueformat": ".3f"},
        number= {"valueformat": ".3f", "font": {"color": color, "size": 36}},
        gauge = {
            "axis"      : {"range": [0, 1], "tickwidth": 1, "tickcolor": "#30363d",
                           "tickfont": {"color": "#8b949e", "size": 10}},
            "bar"       : {"color": color, "thickness": 0.25},
            "bgcolor"   : "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps"     : [
                {"range": [0, 0.2],                                  "color": "rgba(35,134,54,0.12)"},
                {"range": [0.2, config.DRIFT_THRESHOLD],            "color": "rgba(158,106,3,0.12)"},
                {"range": [config.DRIFT_THRESHOLD, 1],              "color": "rgba(218,54,51,0.12)"},
            ],
            "threshold" : {
                "line" : {"color": "#f85149", "width": 2},
                "thickness": 0.75,
                "value": config.DRIFT_THRESHOLD,
            },
        },
        title = {"text": "Drift Score", "font": {"color": "#8b949e", "size": 12}},
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=240)
    return fig


def build_feature_drift_bar(feature_results: dict) -> go.Figure:
    """Horizontal bar chart of p-values per feature — red = drifted."""
    items = [
        (feat, res["p_value"], res["drifted"])
        for feat, res in feature_results.items()
        if res["p_value"] is not None
    ]
    # Sort by p-value ascending (most drifted first)
    items.sort(key=lambda x: x[1])
    items = items[:20]   # top-20 most drifted features

    feats   = [i[0][:28] for i in items]
    pvals   = [i[1]       for i in items]
    drifted = [i[2]       for i in items]
    colors  = ["#f85149" if d else "#3fb950" for d in drifted]

    fig = go.Figure(go.Bar(
        x           = pvals,
        y           = feats,
        orientation = "h",
        marker_color= colors,
        text        = [f"{p:.4f}" for p in pvals],
        textposition= "outside",
        textfont    = dict(size=9, color="#8b949e"),
    ))
    fig.add_vline(
        x           = config.KS_P_VALUE_THRESHOLD,
        line_dash   = "dash",
        line_color  = "#da3633",
        annotation_text    = f"p={config.KS_P_VALUE_THRESHOLD}",
        annotation_position= "top right",
        annotation_font    = dict(color="#f85149", size=10),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title = "KS p-value",
        yaxis       = dict(autorange="reversed", tickfont=dict(size=9, color="#8b949e")),
        xaxis       = dict(range=[0, max(pvals + [0.1]) * 1.2],
                           tickfont=dict(size=9)),
        height      = max(280, len(items) * 22 + 60),
    )
    return fig


def build_drift_history_chart(drift_log_path: str) -> go.Figure | None:
    """Line chart of drift score over time with threshold annotation."""
    if not os.path.exists(drift_log_path):
        return None
    df = pd.read_csv(drift_log_path)
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").tail(30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x         = df["timestamp"],
        y         = df["drift_score"],
        mode      = "lines+markers",
        name      = "Drift Score",
        line      = dict(color="#58a6ff", width=2),
        marker    = dict(size=6, color="#58a6ff"),
        fill      = "tozeroy",
        fillcolor = "rgba(88,166,255,0.06)",
    ))
    fig.add_hline(
        y                  = config.DRIFT_THRESHOLD,
        line_dash          = "dash",
        line_color         = "#f85149",
        annotation_text    = f"Threshold ({config.DRIFT_THRESHOLD})",
        annotation_position= "top right",
        annotation_font    = dict(color="#f85149", size=10),
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title = "Time",
        yaxis_title = "Drift Score",
        yaxis       = dict(range=[0, 1.05]),
        height      = 260,
        showlegend  = False,
    )
    return fig


def build_prediction_histogram(probabilities: np.ndarray) -> go.Figure:
    """Histogram of P(class=1) for all predictions."""
    prob_1 = probabilities[:, 1]
    fig = go.Figure(go.Histogram(
        x        = prob_1,
        nbinsx   = 20,
        marker   = dict(
            color     = "rgba(88,166,255,0.7)",
            line      = dict(color="#58a6ff", width=1),
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title = "P(Malignant)",
        yaxis_title = "Count",
        title       = dict(text="Prediction Probability Distribution",
                           font=dict(size=12, color="#8b949e")),
        height      = 240,
    )
    return fig


def build_prediction_donut(n_benign: int, n_malignant: int) -> go.Figure:
    """Donut chart showing benign vs malignant split."""
    fig = go.Figure(go.Pie(
        labels    = ["Benign (0)", "Malignant (1)"],
        values    = [n_benign, n_malignant],
        hole      = 0.55,
        marker    = dict(colors=["#3fb950", "#f85149"],
                         line=dict(color="#0d1117", width=2)),
        textfont  = dict(size=11),
        hovertemplate = "%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        showlegend = True,
        legend     = dict(font=dict(color="#8b949e", size=10)),
        height     = 240,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED PIPELINE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def run_pipeline_analysis() -> dict:
    """Run the full analysis pipeline and return results (cached for 5 s)."""
    loader = DataLoader()
    try:
        incoming_df = loader.load_incoming()
        training_df = loader.load_training()
    except Exception as e:
        return {"error": str(e)}

    validator  = DataValidator()
    validation = validator.validate(incoming_df)

    detector = DriftDetector()
    drift    = detector.detect(training_df, incoming_df)

    predictor = ModelPredictor()
    model_ok  = False
    predictions   = np.array([])
    probabilities = np.array([]).reshape(0, 2)
    avg_confidence = None

    try:
        predictor.load_model()
        predictions, probabilities = predictor.predict(incoming_df)
        avg_confidence = predictor.get_avg_confidence(probabilities)
        model_ok = True
    except FileNotFoundError:
        pass

    engine   = DecisionEngine()
    decision = engine.decide(
        drift_score       = drift["drift_score"],
        validation_report = validation,
        predictions       = predictions,
        avg_confidence    = avg_confidence,
    )

    return {
        "validation":    validation,
        "drift":         drift,
        "decision":      decision,
        "predictions":   predictions,
        "probabilities": probabilities,
        "avg_confidence": avg_confidence,
        "model_ok":      model_ok,
        "incoming_df":   incoming_df,
        "training_df":   training_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Pipeline Control")
    st.markdown("---")

    st.markdown(f"**Project:** {config.PIPELINE_NAME}")
    st.markdown(f"**Version:** `{config.PIPELINE_VERSION}`")
    st.markdown(f"**Model:** `{config.MODEL_VERSION}`")
    st.markdown("---")

    # Auto-refresh toggle
    auto_refresh = st.toggle("🔁 Auto-refresh (10 s)", value=False)
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()

    st.markdown("")

    # Run full pipeline
    if st.button("▶️  Run Pipeline Now", use_container_width=True):
        with st.spinner("Running pipeline …"):
            try:
                main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
                result = subprocess.run(
                    [sys.executable, main_path],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    st.success("✅ Pipeline run complete!")
                else:
                    st.error(f"Pipeline error:\n{result.stderr[-500:]}")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Failed: {e}")

    # Retrain model
    if st.button("🏋️  Retrain Model", use_container_width=True):
        with st.spinner("Training …"):
            try:
                from training.train_model import ModelTrainer
                trainer = ModelTrainer()
                metrics = trainer.train()
                st.success(
                    f"✅ Retrained!  Acc: {metrics['accuracy']:.4f}  "
                    f"F1: {metrics['f1_weighted']:.4f}"
                )
                st.cache_data.clear()
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("**⚙️ Config**")
    st.markdown(f"- Drift threshold: `{config.DRIFT_THRESHOLD}`")
    st.markdown(f"- KS p-value cut-off: `{config.KS_P_VALUE_THRESHOLD}`")
    st.markdown(f"- Max missing: `{config.MAX_MISSING_FRACTION*100:.0f}%`")
    st.markdown(f"- Cooldown: `{getattr(config,'RETRAIN_COOLDOWN_MINUTES',10)} min`")
    st.markdown(f"- Confidence threshold: `{getattr(config,'CONFIDENCE_THRESHOLD',0.50)}`")
    st.markdown("---")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.72rem;'>"
        "Breast Cancer Wisconsin Dataset<br>"
        "Random Forest · KS Drift Detection</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="gradient-title">🧬 Self-Healing ML Pipeline</p>'
    f'<p class="subtitle">Monitoring Dashboard · {config.PIPELINE_NAME} v{config.PIPELINE_VERSION} · '
    f'Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── Load data ─────────────────────────────────────────────────────────────────
data = run_pipeline_analysis()

if "error" in data:
    st.error(f"Pipeline error: {data['error']}")
    st.stop()

drift      = data["drift"]
validation = data["validation"]
decision   = data["decision"]
preds      = data["predictions"]
probs      = data["probabilities"]
avg_conf   = data["avg_confidence"]

# ── Row 1 : KPI Cards ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Model Version", config.MODEL_VERSION)
with c2:
    drift_score = drift["drift_score"]
    delta_color = "off" if drift_score < config.DRIFT_THRESHOLD else "inverse"
    st.metric("Drift Score", f"{drift_score:.3f}",
              delta=f"threshold {config.DRIFT_THRESHOLD}")
with c3:
    n_dr = drift["n_features_drifted"]
    n_to = drift["n_features_total"]
    st.metric("Drifted Features", f"{n_dr} / {n_to}")
with c4:
    st.metric("Incoming Samples", validation["stats"].get("n_rows", "–"))
with c5:
    cls1_pct = f"{preds.mean():.1%}" if len(preds) > 0 else "–"
    st.metric("Malignant Rate", cls1_pct)
with c6:
    conf_val = f"{avg_conf:.3f}" if avg_conf is not None else "–"
    st.metric("Avg Confidence", conf_val)

st.markdown("")

# ── Row 2 : Status Banner ─────────────────────────────────────────────────────
action = decision["action"]
badge  = get_status_badge(action)
bg     = "#0d3321" if action == "continue" else "#3d1c1c"
border = "#238636" if action == "continue" else "#da3633"
cooldown_note = ""
if decision.get("cooldown_active"):
    cooldown_note = " &nbsp;⏳ <em style='color:#d29922;font-size:0.8rem;'>Cooldown active — retrain suppressed</em>"

st.markdown(
    f"""<div style='background:{bg};border:1px solid {border};
        border-radius:12px;padding:15px 22px;margin-bottom:10px;'>
        <div style='display:flex;align-items:center;gap:14px;'>
            {badge}
            <span style='color:#c9d1d9;font-size:0.88rem;'>{decision['reason']}{cooldown_note}</span>
            <span style='margin-left:auto;color:#8b949e;font-size:0.75rem;'>Rule #{decision['rule_id']}</span>
        </div></div>""",
    unsafe_allow_html=True,
)

st.markdown("")

# ── Row 3 : Drift Gauge + Feature Drift Chart ─────────────────────────────────
st.markdown('<p class="section-header">Drift Analysis</p>', unsafe_allow_html=True)
col_gauge, col_feat = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(
        build_drift_gauge(drift["drift_score"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with col_feat:
    feature_results = drift.get("feature_results", {})
    if feature_results:
        st.plotly_chart(
            build_feature_drift_bar(feature_results),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("No feature results available.")

st.markdown("")

# ── Row 4 : Predictions + Validation ─────────────────────────────────────────
st.markdown('<p class="section-header">Predictions & Validation</p>', unsafe_allow_html=True)
col_pred_donut, col_pred_hist, col_val = st.columns([1, 2, 1])

with col_pred_donut:
    if len(preds) > 0:
        n_benign    = int((preds == 0).sum())
        n_malignant = int((preds == 1).sum())
        st.plotly_chart(
            build_prediction_donut(n_benign, n_malignant),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.warning("No predictions yet.")

with col_pred_hist:
    if len(probs) > 0:
        st.plotly_chart(
            build_prediction_histogram(probs),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Run pipeline to generate predictions.")

with col_val:
    st.markdown('<p class="section-header">Data Validation</p>', unsafe_allow_html=True)
    vstats = validation["stats"]
    is_v   = validation["is_valid"]
    status_icon = "✅ PASS" if is_v else "❌ FAIL"
    st.markdown(f"**Status:** {status_icon}")
    st.markdown(f"- Rows: `{vstats.get('n_rows','–')}`")
    st.markdown(f"- Columns: `{vstats.get('n_columns','–')}`")
    st.markdown(f"- Missing: `{vstats.get('missing_pct','–')}%`")
    st.markdown(f"- Inf values: `{vstats.get('n_inf','–')}`")
    st.markdown(f"- Negatives: `{vstats.get('n_negative','–')}`")

    if validation["issues"]:
        st.markdown("**Issues:**")
        for issue in validation["issues"]:
            st.markdown(
                f"<span style='color:#f85149;font-size:0.78rem;'>⚠ {issue}</span>",
                unsafe_allow_html=True,
            )

st.markdown("")

# ── Row 5 : Drift History ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">Drift Score History</p>', unsafe_allow_html=True)
history_fig = build_drift_history_chart(config.DRIFT_LOG)
if history_fig:
    st.plotly_chart(history_fig, use_container_width=True,
                    config={"displayModeBar": False})
else:
    st.info("No drift history yet. Run `python main.py` to populate it.")

st.markdown("")

# ── Row 6 : Training History ──────────────────────────────────────────────────
st.markdown('<p class="section-header">Training Run History</p>', unsafe_allow_html=True)
training_history = load_training_history()
if training_history:
    rows = []
    for run in reversed(training_history[-10:]):   # last 10 runs
        rows.append({
            "Timestamp" : run.get("timestamp", "")[:19].replace("T", " "),
            "Accuracy"  : f"{run.get('accuracy', 0):.4f}",
            "Precision" : f"{run.get('precision_weighted', 0):.4f}",
            "Recall"    : f"{run.get('recall_weighted', 0):.4f}",
            "F1-Score"  : f"{run.get('f1_weighted', 0):.4f}",
            "Samples"   : run.get("n_train_samples", "–"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No training history yet. Run training first.")

st.markdown("")

# ── Row 7 : Pipeline Logs ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">Recent Pipeline Logs</p>', unsafe_allow_html=True)
log_lines = load_log_lines(n=40)
if log_lines:
    html_lines = "<br>".join(colorize_log(l) for l in log_lines)
    st.markdown(f'<div class="log-box">{html_lines}</div>', unsafe_allow_html=True)
else:
    st.info("No log entries yet. Run `python main.py` first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("")
st.markdown(
    "<hr style='border-color:#21262d;'>"
    "<p style='color:#8b949e;font-size:0.72rem;text-align:center;'>"
    "Self-Healing ML Pipeline &nbsp;·&nbsp; Breast Cancer Wisconsin Dataset &nbsp;·&nbsp; "
    "Random Forest &nbsp;·&nbsp; KS Drift Detection &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
