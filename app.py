import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, precision_score,
                             recall_score, f1_score)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShieldAI — Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — ALL FIXES APPLIED
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global dark background ── */
.stApp { background: linear-gradient(135deg, #1f0f18 0%, #2d1520 50%, #372028 100%) !important; }

/* ── Sidebar — compact, no wasted space ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f0f18 0%, #2d1520 100%) !important;
    border-right: 1px solid #5a3a50 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0px !important;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── DEPLOY BUTTON — make it bright & visible ── */
[data-testid="stToolbar"] {
    background: linear-gradient(90deg, #2d1520, #1f0f18) !important;
    border-bottom: 1px solid #5a3a50 !important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a,
header button,
header a {
    color: #ffffff !important;
    opacity: 1 !important;
}
/* Target the Deploy text button specifically */
.stDeployButton, .stDeployButton * {
    color: #4da3ff !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #1f0f18, #2d1520) !important;
    border-bottom: 1px solid #5a3a50 !important;
}
header[data-testid="stHeader"] button {
    color: #ffffff !important;
    background: #2979ff !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
/* ── Hamburger and top-right icons ── */
.stApp header button svg { fill: #ffffff !important; }

/* ── FILE UPLOADER — fix light/white box ── */
[data-testid="stFileUploader"] {
    background: #2d1520 !important;
    border: 1.5px dashed #2979ff !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] * {
    color: #e6edf3 !important;
}
[data-testid="stFileUploader"] section {
    background: #1a0a12 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] section > div {
    background: #1a0a12 !important;
}
/* "200MB per file" and "CSV" text */
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #c9d1d9 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #c9d1d9 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #1a0a12 !important;
    border: 1px dashed #2979ff !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #c9d1d9 !important;
}
/* Upload button inside dropzone */
[data-testid="stFileUploaderDropzone"] button {
    background: #2979ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
/* Small text like "200MB per file • CSV" */
.uploadedFileName, small, .css-small {
    color: #8b949e !important;
}

/* ── All app text ── */
.stApp, .stApp p, .stApp span, .stApp div,
.stApp label, .stApp h1, .stApp h2, .stApp h3 {
    color: #e6edf3 !important;
}

/* ── Page title ── */
.page-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #ffffff !important;
    text-shadow: 0 0 40px rgba(41,121,255,0.6);
    margin-bottom: 0;
    letter-spacing: -1px;
    line-height: 1.1;
}
.page-subtitle {
    font-size: 1rem;
    color: #8b949e !important;
    margin-top: 4px;
    margin-bottom: 16px;
}

/* ── Section headings ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff !important;
    background: linear-gradient(90deg, #2979ff28, transparent);
    border-left: 4px solid #2979ff;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
}

/* ── Metric cards ── */
.metric-card {
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    margin: 4px 0;
}
.card-blue   { background:#1f1535; border:1px solid #7b6b95; border-left:4px solid #9b8bbc; }
.card-green  { background:#1a2f1a; border:1px solid #5aab5a; border-left:4px solid #7cbd7c; }
.card-red    { background:#4a2028; border:1px solid #b87a7a; border-left:4px solid #e8a8a8; }
.card-yellow { background:#3a3a1a; border:1px solid #9b9b5a; border-left:4px solid #d1d17c; }
.card-blue   h2 { color:#a8a0d8 !important; font-size:2.1rem; font-weight:800; margin:0; }
.card-green  h2 { color:#7cbd7c !important; font-size:2.1rem; font-weight:800; margin:0; }
.card-red    h2 { color:#ff9999 !important; font-size:2.1rem; font-weight:800; margin:0; }
.card-yellow h2 { color:#e8e87c !important; font-size:2.1rem; font-weight:800; margin:0; }
.metric-card p  { color:#8b949e !important; font-size:0.82rem; margin:6px 0 0 0; }

/* ── Result badges ── */
.fraud-badge {
    background: #ff4b4b18;
    border: 2px solid #ff4b4b;
    color: #ff6b6b !important;
    border-radius: 50px;
    padding: 16px 40px;
    font-size: 1.7rem;
    font-weight: 800;
    text-align: center;
    display: block;
    margin: 10px 0;
    box-shadow: 0 0 20px #ff4b4b33;
}
.legit-badge {
    background: #00c85318;
    border: 2px solid #00c853;
    color: #00e676 !important;
    border-radius: 50px;
    padding: 16px 40px;
    font-size: 1.7rem;
    font-weight: 800;
    text-align: center;
    display: block;
    margin: 10px 0;
    box-shadow: 0 0 20px #00c85333;
}

/* ── Stats boxes ── */
.stats-box {
    background: #2d1520;
    border: 1px solid #5a3a50;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.stats-box h3 { color: #2979ff !important; font-size: 1.5rem; margin: 0; }
.stats-box p  { color: #8b949e !important; font-size: 0.8rem; margin: 4px 0 0 0; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #2979ff, #1565c0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(41,121,255,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    box-shadow: 0 6px 20px rgba(41,121,255,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Step boxes in About ── */
.step-box {
    background: linear-gradient(135deg,#2d1520,#1a0a12);
    border: 1px solid #5a3a50;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.step-num {
    background: #2979ff;
    color: white !important;
    border-radius: 50%;
    width: 30px; height: 30px; min-width: 30px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.95rem;
}
.step-text h4 { color: #ffffff !important; margin: 0 0 3px 0; font-size: 0.92rem; }
.step-text p  { color: #8b949e !important; margin: 0; font-size: 0.83rem; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(uploaded_bytes=None):
    if uploaded_bytes is not None:
        df = pd.read_csv(uploaded_bytes)
        if 'Class' in df.columns:
            y = df['Class'].values
            X = df.drop(columns=['Class']).values
        else:
            st.error("CSV must have a 'Class' column (0=Legit, 1=Fraud)")
            st.stop()
    else:
        try:
            df = pd.read_csv("sample_creditcard.csv")
            y  = df['Class'].values
            X  = df.drop(columns=['Class']).values
        except FileNotFoundError:
            np.random.seed(42)
            normal = np.random.normal(0, 1, (800, 10))
            fraud  = np.random.normal(4, 1, (40, 10))
            X = np.vstack((normal, fraud))
            y = np.array([0]*800 + [1]*40)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_feat   = X_scaled.shape[1]
    X_normal = X_scaled[y == 0]

    inputs  = Input(shape=(n_feat,))
    enc     = Dense(min(n_feat, 32), activation='relu')(inputs)
    enc     = Dropout(0.1)(enc)
    enc     = Dense(16, activation='relu')(enc)
    bottlen = Dense(8,  activation='relu')(enc)
    dec     = Dense(16, activation='relu')(bottlen)
    dec     = Dropout(0.1)(dec)
    dec     = Dense(min(n_feat, 32), activation='relu')(dec)
    outputs = Dense(n_feat, activation='linear')(dec)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_normal, X_normal,
        epochs=20, batch_size=16,
        validation_split=0.1,
        callbacks=[es], verbose=0
    )

    recon_all     = model.predict(X_scaled, verbose=0)
    errors_all    = np.mean((X_scaled - recon_all) ** 2, axis=1)
    normal_errors = errors_all[y == 0]
    threshold     = np.percentile(normal_errors, 95)
    predictions   = (errors_all > threshold).astype(int)

    return {
        "model": model, "scaler": scaler,
        "history": history.history,
        "X_scaled": X_scaled, "y": y,
        "errors": errors_all, "threshold": threshold,
        "predictions": predictions, "n_feat": n_feat,
    }


# ─────────────────────────────────────────────
# SIDEBAR — compact, no wasted space
# ─────────────────────────────────────────────
with st.sidebar:
    # App branding — tight padding
    st.markdown("""
    <div style="padding:20px 0 14px 0; text-align:center; border-bottom:1px solid #8b1538; margin-bottom:14px">
        <div style="font-size:3.2rem; line-height:1; filter:drop-shadow(0 0 12px #2979ff88)">🛡️</div>
        <div style="font-size:2rem; font-weight:900; color:#4da3ff !important;
                    letter-spacing:-1px; margin-top:8px; line-height:1.1;
                    text-shadow: 0 0 20px rgba(41,121,255,0.5)">ShieldAI</div>
        <div style="font-size:0.78rem; color:#8b949e !important; margin-top:5px;
                    letter-spacing:1px; text-transform:uppercase">
            Fraud Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation — no label gap
    page = st.radio("", [
        "🏠  Dashboard",
        "🔍  Single Transaction",
        "📂  Batch Prediction",
        "📊  Model Performance",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown('<div style="border-top:1px solid #8b1538; margin:12px 0 10px 0"></div>',
                unsafe_allow_html=True)

    # Upload section — compact
    st.markdown("""
    <p style="color:#c9d1d9 !important; font-size:0.88rem; font-weight:600; margin:0 0 4px 0">
        📁 Upload Transaction Data
    </p>
    <p style="color:#8b949e !important; font-size:0.76rem; margin:0 0 8px 0">
        Upload any CSV with transaction features + Class column
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.markdown('<div style="border-top:1px solid #8b1538; margin:12px 0 8px 0"></div>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e !important; font-size:0.72rem; text-align:center">Built with TensorFlow + Streamlit</p>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────
with st.spinner("⚡ ShieldAI is training on transaction data... please wait"):
    data = load_and_train(uploaded if uploaded else None)

model       = data["model"]
scaler      = data["scaler"]
history     = data["history"]
X_scaled    = data["X_scaled"]
y           = data["y"]
errors      = data["errors"]
threshold   = data["threshold"]
predictions = data["predictions"]
n_feat      = data["n_feat"]

total     = len(y)
n_fraud   = int(y.sum())
n_legit   = total - n_fraud
detected  = int(predictions.sum())
precision = precision_score(y, predictions, zero_division=0)
recall    = recall_score(y, predictions, zero_division=0)
f1        = f1_score(y, predictions, zero_division=0)
try:
    auc   = roc_auc_score(y, errors)
except Exception:
    auc   = 0.0

PLOT_BG  = "#1a0a12"
PLOT_GRD = "#2d1520"

# Normalize page name for comparison
page_clean = page.strip()


# ═══════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════
if "Dashboard" in page_clean:
    st.markdown('<h1 style="font-size:3rem; font-weight:900; color:#ffffff; text-shadow:0 0 40px rgba(41,121,255,0.7); letter-spacing:-1px; margin:0; line-height:1.1">🛡️ ShieldAI — Fraud Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.15rem; color:#8b949e; margin-top:6px; margin-bottom:16px">Real-time Autoencoder Anomaly Detection · Powered by Deep Learning</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card card-blue"><h2>{total:,}</h2><p>Total Transactions</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card card-green"><h2>{n_legit:,}</h2><p>Legitimate Transactions</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card card-red"><h2>{n_fraud:,}</h2><p>Actual Fraud Cases</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card card-yellow"><h2>{detected:,}</h2><p>Fraud Detected by Model</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance metrics with tooltips ──
    st.markdown('<p class="section-title">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val, tip in zip(
        [s1, s2, s3, s4],
        ["Precision", "Recall", "F1 Score", "AUC-ROC"],
        [precision, recall, f1, auc],
        [
            "Of all transactions flagged as fraud, how many were actually fraud",
            "Of all actual fraud cases, how many did the model successfully catch",
            "Balance between Precision and Recall — higher is better",
            "Overall model quality — 1.0 is perfect, 0.5 is random"
        ]
    ):
        col.metric(label=label, value=f"{val:.4f}", help=tip)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">📊 Transaction Distribution</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraudulent"],
            values=[n_legit, n_fraud],
            hole=0.55,
            marker_colors=["#2979ff", "#ff4b4b"],
            textinfo="label+percent",
            textfont=dict(color="white", size=13)
        ))
        fig.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font_color="white", height=310,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(font=dict(color="white"))
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<p class="section-title">📉 Avg Reconstruction Error: Fraud vs Legit</p>', unsafe_allow_html=True)
        avg_n = float(np.mean(errors[y == 0]))
        avg_f = float(np.mean(errors[y == 1]))
        fig2  = go.Figure(go.Bar(
            x=["Legitimate", "Fraudulent"],
            y=[avg_n, avg_f],
            marker_color=["#2979ff", "#ff4b4b"],
            text=[f"{avg_n:.4f}", f"{avg_f:.4f}"],
            textposition="outside",
            textfont=dict(color="white", size=13)
        ))
        fig2.add_hline(y=threshold, line_dash="dash", line_color="#ffd600",
                       annotation_text=f"Threshold = {threshold:.4f}",
                       annotation_font_color="#ffd600", annotation_font_size=12)
        fig2.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_GRD,
            font_color="white", yaxis_title="MSE Error", height=310,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(tickfont=dict(color="white", size=13)),
            yaxis=dict(tickfont=dict(color="white"))
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<p class="section-title">🔎 Sample Transaction Results (First 30)</p>', unsafe_allow_html=True)
    rows = []
    for i in range(min(30, total)):
        rows.append({
            "Txn #"    : i + 1,
            "Actual"   : "🔴 FRAUD" if y[i] == 1 else "🟢 LEGIT",
            "Predicted": "🔴 FRAUD" if predictions[i] == 1 else "🟢 LEGIT",
            "MSE Error": round(float(errors[i]), 6),
            "Result"   : "✅ Correct" if y[i] == predictions[i] else "❌ Wrong"
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", height=360)


# ═══════════════════════════════════════════════
# PAGE 2 — SINGLE TRANSACTION
# ═══════════════════════════════════════════════
elif "Single" in page_clean:
    st.markdown('<h1 style="font-size:3rem; font-weight:900; color:#ffffff; text-shadow:0 0 40px rgba(41,121,255,0.7); letter-spacing:-1px; margin:0; line-height:1.1">🔍 Single Transaction Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.15rem; color:#8b949e; margin-top:6px; margin-bottom:16px">Adjust feature sliders and click Predict — hover any feature label for details</p>', unsafe_allow_html=True)
    st.markdown("---")

    feature_names = (
        ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
         "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
         "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
        if n_feat == 30 else [f"Feature {i+1}" for i in range(n_feat)]
    )[:n_feat]

    feature_help = {
        "Time"  : "Seconds elapsed between this transaction and the first transaction in the dataset",
        "Amount": "Transaction amount in the original currency",
        **{f"V{i}": f"PCA component {i} — anonymized transaction feature derived from original data" for i in range(1, 29)}
    }

    st.markdown('<p class="section-title">⚙️ Set Transaction Features</p>', unsafe_allow_html=True)
    st.caption("💡 Hover over any feature label to see what it represents")

    feature_values = []
    cols_per_row   = 5
    for row_start in range(0, n_feat, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < n_feat:
                fname = feature_names[idx]
                tip   = feature_help.get(fname, f"Transaction feature {fname}")
                val   = col.slider(fname, -10.0, 10.0, 0.0, 0.1,
                                   key=f"f{idx}", help=tip)
                feature_values.append(val)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ PREDICT THIS TRANSACTION"):
        raw      = np.array(feature_values).reshape(1, -1)
        scaled   = scaler.transform(raw)
        recon    = model.predict(scaled, verbose=0)
        err      = float(np.mean((scaled - recon) ** 2))
        is_fraud = err > threshold

        col1, col2 = st.columns([1, 1])
        with col1:
            if is_fraud:
                st.markdown('<span class="fraud-badge">🔴 FRAUD DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="legit-badge">🟢 LEGITIMATE TRANSACTION</span>', unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-title">📊 Prediction Details</p>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("MSE Error",  f"{err:.6f}",
                      help="Reconstruction error — higher means the transaction looks more anomalous")
            d2.metric("Threshold",  f"{threshold:.6f}",
                      help="Errors above this are classified as FRAUD (95th percentile of normal errors)")
            d3.metric("Anomaly %",  f"{min(err/threshold, 3)*100:.1f}%",
                      delta="⚠ FRAUD" if is_fraud else "✓ LEGIT",
                      delta_color="inverse",
                      help="How anomalous this transaction is — above 100% means FRAUD")

        gauge_val = min(err / threshold, 3.0) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(gauge_val, 1),
            delta={"reference": 100, "valueformat": ".1f"},
            title={"text": "Anomaly Score  (100% = Fraud Threshold)", "font": {"color": "white", "size": 14}},
            number={"font": {"color": "white", "size": 32}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 300], "tickcolor": "white",
                         "tickfont": {"color": "white"}},
                "bar":  {"color": "#ff4b4b" if is_fraud else "#00c853"},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0,   100], "color": "#0d2112"},
                    {"range": [100, 300], "color": "#210d0d"},
                ],
                "threshold": {
                    "line": {"color": "#ffd600", "width": 4},
                    "thickness": 0.75, "value": 100
                }
            }
        ))
        fig.update_layout(paper_bgcolor=PLOT_BG, font_color="white", height=280)
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ═══════════════════════════════════════════════
elif "Batch" in page_clean:
    st.markdown('<h1 style="font-size:3rem; font-weight:900; color:#ffffff; text-shadow:0 0 40px rgba(41,121,255,0.7); letter-spacing:-1px; margin:0; line-height:1.1">📂 Batch Transaction Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.15rem; color:#8b949e; margin-top:6px; margin-bottom:16px">Use the sidebar uploader to batch scan all transactions at once</p>', unsafe_allow_html=True)
    st.markdown("---")

    if uploaded:
        df_batch   = pd.read_csv(uploaded)
        has_labels = 'Class' in df_batch.columns
        X_b        = df_batch.drop(columns=['Class'], errors='ignore').values
        X_b_scaled = scaler.transform(X_b)

        with st.spinner("⚡ Scanning all transactions..."):
            recon_b = model.predict(X_b_scaled, verbose=0)
            err_b   = np.mean((X_b_scaled - recon_b) ** 2, axis=1)
            pred_b  = (err_b > threshold).astype(int)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card card-blue"><h2>{len(pred_b):,}</h2><p>Total Scanned</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card card-green"><h2>{(pred_b==0).sum():,}</h2><p>Predicted LEGIT</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card card-red"><h2>{(pred_b==1).sum():,}</h2><p>Predicted FRAUD</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        results_df = df_batch.copy()
        results_df["MSE_Error"]  = err_b.round(6)
        results_df["Predicted"]  = ["🔴 FRAUD" if p == 1 else "🟢 LEGIT" for p in pred_b]
        if has_labels:
            results_df["Actual"] = ["🔴 FRAUD" if c == 1 else "🟢 LEGIT"
                                    for c in df_batch["Class"].values]
            results_df["Match"]  = results_df["Predicted"] == results_df["Actual"]

        st.markdown('<p class="section-title">📋 Prediction Results</p>', unsafe_allow_html=True)
        st.dataframe(results_df, width="stretch", height=420)

        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv_out, "shieldai_results.csv", "text/csv")
    else:
        st.info("👆 Upload a transaction CSV file using the sidebar uploader to get started.")
        st.markdown("""
**What format should my CSV be in?**
- Any CSV with numerical transaction feature columns
- Optionally include a `Class` column (0 = Legit, 1 = Fraud) for accuracy comparison
- Works with any real-world or Kaggle-format transaction dataset
        """)


# ═══════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif "Performance" in page_clean:
    st.markdown('<h1 style="font-size:3rem; font-weight:900; color:#ffffff; text-shadow:0 0 40px rgba(41,121,255,0.7); letter-spacing:-1px; margin:0; line-height:1.1">📊 Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Full evaluation of ShieldAI\'s Autoencoder fraud detection model</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-title">🏆 Key Performance Metrics</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color, tip in zip(
        [m1, m2, m3, m4],
        ["Precision", "Recall", "F1 Score", "AUC-ROC"],
        [precision, recall, f1, auc],
        ["card-blue", "card-green", "card-red", "card-yellow"],
        [
            "Of all flagged fraud, how many were actually fraud",
            "Of all real fraud, how many were caught",
            "Balance between Precision and Recall",
            "Overall model quality — 1.0 is perfect"
        ]
    ):
        col.markdown(
            f'<div class="metric-card {color}"><h2>{val:.4f}</h2><p>{label}</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">📊 Reconstruction Error Distribution</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors[y == 0], name="Legitimate",
            marker_color="#2979ff", opacity=0.75, nbinsx=60
        ))
        fig.add_trace(go.Histogram(
            x=errors[y == 1], name="Fraud",
            marker_color="#ff4b4b", opacity=0.9, nbinsx=30
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#ffd600",
                      annotation_text=f"Threshold = {threshold:.4f}",
                      annotation_font_color="#ffd600", annotation_font_size=12)
        fig.update_layout(
            barmode="overlay",
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_GRD,
            font_color="white", height=340,
            xaxis_title="MSE Reconstruction Error", yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
            margin=dict(t=10)
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<p class="section-title">🔲 Confusion Matrix</p>', unsafe_allow_html=True)
        cm    = confusion_matrix(y, predictions)
        z_txt = [
            [f"<b>TN</b><br>{cm[0][0]}<br>Correct Legit",
             f"<b>FP</b><br>{cm[0][1]}<br>Wrong Fraud Alert"],
            [f"<b>FN</b><br>{cm[1][0]}<br>Missed Fraud",
             f"<b>TP</b><br>{cm[1][1]}<br>Caught Fraud"]
        ]
        fig2 = go.Figure(go.Heatmap(
            z=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
            x=["<b>Predicted: Legit</b>", "<b>Predicted: Fraud</b>"],
            y=["<b>Actual: Legit</b>", "<b>Actual: Fraud</b>"],
            colorscale=[[0,"#0d2137"],[0.5,"#1565c0"],[1,"#2979ff"]],
            text=z_txt,
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            showscale=True
        ))
        fig2.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font_color="white", height=340,
            xaxis=dict(tickfont=dict(color="#4da3ff", size=12)),
            yaxis=dict(tickfont=dict(color="#4da3ff", size=12)),
            margin=dict(t=10)
        )
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-title">📉 Training Loss Curve</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            y=history.get("loss", []), name="Train Loss",
            line=dict(color="#2979ff", width=3),
            fill="tozeroy", fillcolor="rgba(41,121,255,0.1)"
        ))
        if "val_loss" in history:
            fig3.add_trace(go.Scatter(
                y=history["val_loss"], name="Val Loss",
                line=dict(color="#ff4b4b", width=2, dash="dash")
            ))
        fig3.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_GRD,
            font_color="white", height=300,
            xaxis_title="Epoch", yaxis_title="MSE Loss",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
            margin=dict(t=10)
        )
        st.plotly_chart(fig3, width="stretch")

    with col4:
        st.markdown('<p class="section-title">🔵 Error per Transaction (Sample)</p>', unsafe_allow_html=True)
        sample_n  = min(600, total)
        idx_s     = np.random.choice(total, sample_n, replace=False)
        legit_idx = idx_s[y[idx_s] == 0]
        fraud_idx = idx_s[y[idx_s] == 1]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=legit_idx, y=errors[legit_idx],
            mode="markers", name="Legit",
            marker=dict(color="#2979ff", size=5, opacity=0.5)
        ))
        fig4.add_trace(go.Scatter(
            x=fraud_idx, y=errors[fraud_idx],
            mode="markers", name="Fraud",
            marker=dict(color="#ff4b4b", size=10, symbol="triangle-up", opacity=1)
        ))
        fig4.add_hline(y=threshold, line_dash="dash", line_color="#ffd600",
                       annotation_text="Threshold", annotation_font_color="#ffd600")
        fig4.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_GRD,
            font_color="white", height=300,
            xaxis_title="Transaction Index", yaxis_title="MSE Error",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
            margin=dict(t=10)
        )
        st.plotly_chart(fig4, width="stretch")

    # Classification report — colored rows
    st.markdown('<p class="section-title">📋 Full Classification Report</p>', unsafe_allow_html=True)
    report    = classification_report(y, predictions,
                                      target_names=["Legit", "Fraud"],
                                      output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)

    def style_report(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        color_map = {
            "Legit"       : "background-color:#0d2137; color:#4da3ff",
            "Fraud"       : "background-color:#210d0d; color:#ff6b6b",
            "macro avg"   : "background-color:#21190d; color:#ffd600",
            "weighted avg": "background-color:#0d2112; color:#00e676",
        }
        for idx in df.index:
            styles.loc[idx, :] = color_map.get(idx, "background-color:#161b22; color:#c9d1d9")
        return styles

    st.dataframe(report_df.style.apply(style_report, axis=None), width="stretch")


# ═══════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════
elif "About" in page_clean:
    st.markdown('<h1 style="font-size:3rem; font-weight:900; color:#ffffff; text-shadow:0 0 40px rgba(41,121,255,0.7); letter-spacing:-1px; margin:0; line-height:1.1">ℹ️ About ShieldAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1.15rem; color:#8b949e; margin-top:6px; margin-bottom:16px">Intelligent fraud detection using Deep Learning Autoencoders</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">🧠 How It Works</p>', unsafe_allow_html=True)
        st.markdown("""
**Step 1:** Model learns normal transaction patterns from legitimate data

**Step 2:** For new transactions, the model measures "reconstruction error" — how different a transaction is from normal patterns

**Step 3:** Transactions with **high reconstruction error** are flagged as **FRAUD 🔴**

**Step 4:** Transactions with **low reconstruction error** are classified as **LEGIT 🟢**
        """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚙️ Key Features</p>', unsafe_allow_html=True)
        st.markdown("""
✓ **Autoencoder Architecture** — Deep neural network learns complex fraud patterns

✓ **Unsupervised Learning** — Detects anomalies without labeled fraud data

✓ **Real-time Detection** — Instantly classify transactions as fraud or legitimate

✓ **Adaptive Threshold** — Uses 95th percentile of normal errors for smart classification
        """)

    with col2:
        st.markdown('<p class="section-title">📊 Dataset</p>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.markdown('<div class="stats-box"><h3>284K+</h3><p>Total Transactions</p></div>', unsafe_allow_html=True)
        d2.markdown('<div class="stats-box"><h3>492</h3><p>Fraud Cases</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">🛠️ Tech Stack</p>', unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        for col, tech in zip([t1, t2, t3], ["TensorFlow", "Streamlit", "Plotly"]):
            col.markdown(
                f'<div class="stats-box"><h3 style="font-size:0.9rem;color:#2979ff !important">{tech}</h3></div>',
                unsafe_allow_html=True
            )
        st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
        t4, t5 = st.columns(2)
        for col, tech in zip([t4, t5], ["Scikit-learn", "Pandas"]):
            col.markdown(
                f'<div class="stats-box"><h3 style="font-size:0.9rem;color:#2979ff !important">{tech}</h3></div>',
                unsafe_allow_html=True
            )