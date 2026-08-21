import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ShieldAI — Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #1f0f18 0%,
            #2d1520 50%,
            #372028 100%
        ) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #1f0f18 0%,
            #2d1520 100%
        ) !important;
        border-right: 1px solid #5a3a50 !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 12px !important;
    }

    header[data-testid="stHeader"] {
        background: linear-gradient(
            90deg,
            #1f0f18,
            #2d1520
        ) !important;
        border-bottom: 1px solid #5a3a50 !important;
    }

    [data-testid="stFileUploader"] {
        background: #2d1520 !important;
        border: 1.5px dashed #2979ff !important;
        border-radius: 10px !important;
        padding: 8px !important;
    }

    [data-testid="stFileUploader"] section {
        background: #1a0a12 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #1a0a12 !important;
        border: 1px dashed #2979ff !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #2979ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff !important;
        background: linear-gradient(
            90deg,
            #2979ff28,
            transparent
        );
        border-left: 4px solid #2979ff;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 12px 0;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #8b949e !important;
        margin-top: 4px;
        margin-bottom: 16px;
    }

    .metric-card {
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        margin: 4px 0;
    }

    .card-blue {
        background: #1f1535;
        border: 1px solid #7b6b95;
        border-left: 4px solid #9b8bbc;
    }

    .card-green {
        background: #1a2f1a;
        border: 1px solid #5aab5a;
        border-left: 4px solid #7cbd7c;
    }

    .card-red {
        background: #4a2028;
        border: 1px solid #b87a7a;
        border-left: 4px solid #e8a8a8;
    }

    .card-yellow {
        background: #3a3a1a;
        border: 1px solid #9b9b5a;
        border-left: 4px solid #d1d17c;
    }

    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }

    .card-blue h2 {
        color: #a8a0d8 !important;
    }

    .card-green h2 {
        color: #7cbd7c !important;
    }

    .card-red h2 {
        color: #ff9999 !important;
    }

    .card-yellow h2 {
        color: #e8e87c !important;
    }

    .metric-card p {
        color: #8b949e !important;
        font-size: 0.82rem;
        margin: 6px 0 0 0;
    }

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
    }

    .detail-box {
        background: #2d1520;
        border: 1px solid #5a3a50;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .detail-label {
        color: #8b949e !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .detail-value {
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 3px;
    }

    .stButton > button {
        background: linear-gradient(
            135deg,
            #2979ff,
            #1565c0
        ) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "model.pkl"
AUTOENCODER_FILE = "autoencoder.keras"
SCALER_FILE = "scaler.pkl"
THRESHOLD_FILE = "threshold.pkl"
FEATURES_FILE = "features.pkl"
METRICS_FILE = "metrics.pkl"

PLOT_BG = "#1a0a12"
PLOT_GRID = "#2d1520"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():

    required_files = [
        MODEL_FILE,
        AUTOENCODER_FILE,
        SCALER_FILE,
        THRESHOLD_FILE,
        FEATURES_FILE,
        METRICS_FILE,
    ]

    missing_files = [
        filename
        for filename in required_files
        if not os.path.exists(filename)
    ]

    if missing_files:
        st.error(
            "Missing model files: "
            + ", ".join(missing_files)
        )
        st.info(
            "Make sure all trained model files are beside app.py."
        )
        st.stop()

    # Random Forest
    with open(
        MODEL_FILE,
        "rb"
    ) as file:
        model = pickle.load(file)

    # Autoencoder
    autoencoder = tf.keras.models.load_model(
        AUTOENCODER_FILE
    )

    # Scaler
    with open(
        SCALER_FILE,
        "rb"
    ) as file:
        scaler = pickle.load(file)

    # Threshold
    with open(
        THRESHOLD_FILE,
        "rb"
    ) as file:
        threshold = float(
            pickle.load(file)
        )

    # Feature names
    with open(
        FEATURES_FILE,
        "rb"
    ) as file:
        feature_names = pickle.load(file)

    # Saved test metrics
    with open(
        METRICS_FILE,
        "rb"
    ) as file:
        metrics = pickle.load(file)

    # Encoder
    encoder = tf.keras.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer(
            "latent"
        ).output,
    )

    return (
        model,
        encoder,
        scaler,
        threshold,
        feature_names,
        metrics,
    )


(
    model,
    encoder,
    scaler,
    threshold,
    feature_names,
    saved_metrics,
) = load_artifacts()


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):

    X = df.drop(
        columns=["Class"],
        errors="ignore",
    ).copy()

    missing_columns = [
        column
        for column in feature_names
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_columns)
        )

    # Preserve exact training order
    X = X[
        feature_names
    ]

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "CSV contains missing or non-numeric transaction values."
        )

    return X


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_transactions(df):

    X = prepare_data(
        df
    )

    # Scale
    X_scaled = scaler.transform(
        X.values
    )

    # Autoencoder latent features
    latent_features = encoder.predict(
        X_scaled,
        batch_size=1024,
        verbose=0,
    )

    # Hybrid feature representation
    hybrid_features = np.concatenate(
        [
            X_scaled,
            latent_features,
        ],
        axis=1,
    )

    # Random Forest fraud probability
    probabilities = model.predict_proba(
        hybrid_features
    )[:, 1]

    # Tuned threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return (
        X,
        probabilities,
        predictions,
    )


# ============================================================
# READ GLOBAL UPLOAD
# ============================================================

def get_uploaded_data():

    if uploaded is None:
        return None

    try:

        uploaded.seek(0)

        return pd.read_csv(
            uploaded
        )

    except Exception as error:

        st.error(
            f"Could not read uploaded CSV: {error}"
        )

        return None


# ============================================================
# CALCULATE UPLOADED CSV METRICS
# ============================================================

def calculate_uploaded_metrics(
    df,
    predictions,
    probabilities,
):

    if "Class" not in df.columns:
        return None

    actual = (
        df["Class"]
        .astype(int)
        .values
    )

    if len(
        np.unique(actual)
    ) < 2:
        return None

    precision = precision_score(
        actual,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            actual,
            probabilities,
        )
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(
        actual,
        predictions,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🛡️ ShieldAI"
    )

    st.caption(
        "Fraud Intelligence Platform"
    )

    st.divider()

    # RADIO NAVIGATION
    # Fresh unique key avoids the previous stale-radio error.

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Single Transaction",
            "📂 Batch Prediction",
            "📊 Model Performance",
            "ℹ️ About",
        ],
        key="shieldai_navigation_final_v5",
    )

    st.divider()

    st.subheader(
        "📁 Transaction Data"
    )

    st.caption(
        "Upload once. The same CSV is available across the app."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="shieldai_global_upload_final_v5",
    )

    if uploaded is not None:

        st.success(
            f"Loaded: {uploaded.name}"
        )

    else:

        st.caption(
            "No CSV uploaded. Showing saved test-set information."
        )

    st.divider()

    st.caption(
        "Built with Autoencoder + Random Forest"
    )


# ============================================================
# ACTIVE UPLOADED DATA
# ============================================================

active_df = get_uploaded_data()


# ============================================================
# SHARED UPLOAD PREDICTIONS
# ============================================================

uploaded_predictions = None
uploaded_probabilities = None
uploaded_features = None
uploaded_eval = None

if active_df is not None:

    try:

        (
            uploaded_features,
            uploaded_probabilities,
            uploaded_predictions,
        ) = predict_transactions(
            active_df
        )

        uploaded_eval = calculate_uploaded_metrics(
            active_df,
            uploaded_predictions,
            uploaded_probabilities,
        )

    except Exception:
        uploaded_predictions = None
        uploaded_probabilities = None
        uploaded_features = None
        uploaded_eval = None


# ============================================================
# DATASET STATUS
# ============================================================

def show_dataset_status():

    if active_df is None:

        st.info(
            "No CSV uploaded — showing saved held-out test-set information."
        )

    else:

        st.success(
            f"Active dataset: {len(active_df):,} transactions"
            + (
                " • Class column available"
                if "Class" in active_df.columns
                else " • No Class column"
            )
        )


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <h1 style="
            font-size:3rem;
            font-weight:900;
            color:#ffffff;
            margin:0;
        ">
            🛡️ ShieldAI — Fraud Intelligence
        </h1>

        <p class="page-subtitle">
            Autoencoder + Random Forest Fraud Detection
        </p>
        """,
        unsafe_allow_html=True,
    )

    show_dataset_status()

    st.markdown("---")

    # --------------------------------------------------------
    # BEFORE UPLOAD
    # --------------------------------------------------------

    if active_df is None:

        st.markdown(
            '<p class="section-title">🤖 Model Information</p>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(
            f"""
            <div class="metric-card card-blue">
                <h2>{len(feature_names)}</h2>
                <p>Input Features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c2.markdown(
            f"""
            <div class="metric-card card-green">
                <h2>{threshold:.3f}</h2>
                <p>Decision Threshold</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c3.markdown(
            f"""
            <div class="metric-card card-red">
                <h2>{saved_metrics['precision']:.3f}</h2>
                <p>Test Precision</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c4.markdown(
            f"""
            <div class="metric-card card-yellow">
                <h2>{saved_metrics['recall']:.3f}</h2>
                <p>Test Recall</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="section-title">🏆 Held-Out Test Performance</p>',
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Precision",
            f"{saved_metrics['precision']:.4f}",
        )

        p2.metric(
            "Recall",
            f"{saved_metrics['recall']:.4f}",
        )

        p3.metric(
            "F1 Score",
            f"{saved_metrics['f1']:.4f}",
        )

        p4.metric(
            "AUC-ROC",
            f"{saved_metrics['auc']:.4f}",
        )

        st.caption(
            f"Based on {int(saved_metrics['test_size']):,} "
            "unseen test transactions from the original dataset."
        )

    # --------------------------------------------------------
    # AFTER UPLOAD
    # --------------------------------------------------------

    else:

        if uploaded_predictions is not None:

            total = len(
                uploaded_predictions
            )

            fraud = int(
                uploaded_predictions.sum()
            )

            legit = total - fraud

            st.markdown(
                '<p class="section-title">📂 Uploaded Dataset</p>',
                unsafe_allow_html=True,
            )

            s1, s2, s3 = st.columns(3)

            s1.markdown(
                f"""
                <div class="metric-card card-blue">
                    <h2>{total:,}</h2>
                    <p>Total Transactions</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            s2.markdown(
                f"""
                <div class="metric-card card-green">
                    <h2>{legit:,}</h2>
                    <p>Predicted Legit</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            s3.markdown(
                f"""
                <div class="metric-card card-red">
                    <h2>{fraud:,}</h2>
                    <p>Predicted Fraud</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )

            if uploaded_eval is not None:

                st.markdown(
                    '<p class="section-title">📈 Uploaded Dataset Performance</p>',
                    unsafe_allow_html=True,
                )

                p1, p2, p3, p4 = st.columns(4)

                p1.metric(
                    "Precision",
                    f"{uploaded_eval['precision']:.4f}",
                )

                p2.metric(
                    "Recall",
                    f"{uploaded_eval['recall']:.4f}",
                )

                p3.metric(
                    "F1 Score",
                    f"{uploaded_eval['f1']:.4f}",
                )

                p4.metric(
                    "AUC-ROC",
                    f"{uploaded_eval['auc']:.4f}",
                )

                st.caption(
                    "These metrics are calculated from the currently uploaded CSV."
                )

            else:

                st.info(
                    "The uploaded CSV does not contain a usable Class column, "
                    "so precision, recall, F1 and AUC cannot be calculated."
                )

        else:

            st.error(
                "The uploaded CSV could not be processed."
            )


# ============================================================
# PAGE 2 — SINGLE TRANSACTION
# ============================================================

elif page == "🔍 Single Transaction":

    st.markdown(
        """
        <h1 style="
            font-size:3rem;
            font-weight:900;
            color:#ffffff;
            margin:0;
        ">
            🔍 Single Transaction Predictor
        </h1>

        <p class="page-subtitle">
            Enter one transaction and classify it with the trained model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    show_dataset_status()

    st.markdown("---")

    if active_df is not None:

        st.info(
            "A CSV is currently active. "
            "The form below is for an additional single-transaction prediction."
        )

    values = []

    for start in range(
        0,
        len(feature_names),
        5,
    ):

        columns = st.columns(5)

        for i, column in enumerate(
            columns
        ):

            index = start + i

            if index >= len(feature_names):

                continue

            name = feature_names[index]

            if name == "Amount":

                value = column.number_input(
                    "Amount",
                    min_value=0.0,
                    value=100.0,
                    key="single_amount_final_v5",
                )

            elif name == "Time":

                value = column.number_input(
                    "Time",
                    value=0.0,
                    key="single_time_final_v5",
                )

            else:

                value = column.number_input(
                    name,
                    value=0.0,
                    format="%.6f",
                    key=f"single_{index}_final_v5",
                )

            values.append(
                float(value)
            )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    if st.button(
        "⚡ PREDICT TRANSACTION",
        key="single_predict_final_v5",
    ):

        sample = pd.DataFrame(
            [values],
            columns=feature_names,
        )

        try:

            (
                _,
                probability,
                prediction,
            ) = predict_transactions(
                sample
            )

            fraud_probability = float(
                probability[0]
            )

            is_fraud = bool(
                prediction[0]
            )

            if is_fraud:

                st.markdown(
                    """
                    <div class="fraud-badge">
                        🔴 FRAUD DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="legit-badge">
                        🟢 LEGITIMATE TRANSACTION
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            a, b = st.columns(2)

            a.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%",
            )

            b.metric(
                "Decision Threshold",
                f"{threshold * 100:.2f}%",
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# PAGE 3 — BATCH PREDICTION
# ============================================================

elif page == "📂 Batch Prediction":

    st.markdown(
        """
        <h1 style="
            font-size:3rem;
            font-weight:900;
            color:#ffffff;
            margin:0;
        ">
            📂 Batch Transaction Prediction
        </h1>

        <p class="page-subtitle">
            Search, select, and inspect transactions from the active CSV.
        </p>
        """,
        unsafe_allow_html=True,
    )

    show_dataset_status()

    st.markdown("---")

    if active_df is None:

        st.info(
            "Upload a CSV using the Transaction Data uploader in the sidebar."
        )

    elif uploaded_predictions is None:

        st.error(
            "The uploaded CSV could not be processed."
        )

    else:

        results = active_df.copy()

        results.insert(
            0,
            "Transaction #",
            np.arange(
                1,
                len(results) + 1
            ),
        )

        results[
            "Fraud Probability"
        ] = uploaded_probabilities.round(
            6
        )

        results[
            "Predicted"
        ] = [
            "🔴 FRAUD"
            if p == 1
            else "🟢 LEGIT"
            for p in uploaded_predictions
        ]

        total = len(
            uploaded_predictions
        )

        fraud = int(
            uploaded_predictions.sum()
        )

        legit = total - fraud

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">📊 Prediction Summary</p>',
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Total Scanned",
            f"{total:,}",
        )

        s2.metric(
            "Predicted Legit",
            f"{legit:,}",
        )

        s3.metric(
            "Predicted Fraud",
            f"{fraud:,}",
        )

        # ----------------------------------------------------
        # VISUALIZATIONS
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">📈 Visual Analysis</p>',
            unsafe_allow_html=True,
        )

        chart1, chart2 = st.columns(2)

        with chart1:

            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "Legitimate",
                            "Fraudulent",
                        ],
                        values=[
                            legit,
                            fraud,
                        ],
                        hole=0.55,
                        textinfo="label+percent",
                        marker=dict(
                            colors=[
                                "#2979ff",
                                "#ff4b4b",
                            ]
                        ),
                    )
                ]
            )

            pie.update_layout(
                title="Transaction Distribution",
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                font_color="white",
                height=350,
            )

            st.plotly_chart(
                pie,
                use_container_width=True,
            )

        with chart2:

            bar = go.Figure(
                data=[
                    go.Bar(
                        x=[
                            "Legitimate",
                            "Fraudulent",
                        ],
                        y=[
                            legit,
                            fraud,
                        ],
                        text=[
                            legit,
                            fraud,
                        ],
                        textposition="outside",
                    )
                ]
            )

            bar.update_layout(
                title="Legitimate vs Fraud Transactions",
                xaxis_title="Prediction",
                yaxis_title="Transactions",
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_GRID,
                font_color="white",
                height=350,
            )

            st.plotly_chart(
                bar,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # UPLOADED DATASET METRICS
        # ----------------------------------------------------

        if uploaded_eval is not None:

            st.markdown(
                '<p class="section-title">📈 Uploaded Dataset Evaluation</p>',
                unsafe_allow_html=True,
            )

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "Precision",
                f"{uploaded_eval['precision']:.4f}",
            )

            e2.metric(
                "Recall",
                f"{uploaded_eval['recall']:.4f}",
            )

            e3.metric(
                "F1 Score",
                f"{uploaded_eval['f1']:.4f}",
            )

            e4.metric(
                "AUC-ROC",
                f"{uploaded_eval['auc']:.4f}",
            )

            # Confusion matrix

            cm = uploaded_eval[
                "confusion_matrix"
            ]

            st.markdown(
                '<p class="section-title">🔲 Confusion Matrix</p>',
                unsafe_allow_html=True,
            )

            matrix = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=[
                        "Predicted Legit",
                        "Predicted Fraud",
                    ],
                    y=[
                        "Actual Legit",
                        "Actual Fraud",
                    ],
                    colorscale=[
                        [0, "#0d2137"],
                        [0.5, "#1565c0"],
                        [1, "#2979ff"],
                    ],
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(
                        color="white",
                        size=16,
                    ),
                )
            )

            matrix.update_layout(
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                font_color="white",
                height=350,
            )

            st.plotly_chart(
                matrix,
                use_container_width=True,
            )

        else:

            st.info(
                "This CSV has no Class column, so uploaded-data "
                "evaluation metrics cannot be calculated."
            )

        # ----------------------------------------------------
        # SEARCH + SELECT
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">🔎 Search & Select Transactions</p>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Use the table search to search FRAUD or LEGIT, "
            "then select a row to inspect it."
        )

        selection = st.dataframe(
            results,
            use_container_width=True,
            height=500,
            hide_index=True,
            key="shieldai_results_table_final_v5",
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = []

        try:

            if hasattr(
                selection,
                "selection",
            ):

                selected_rows = (
                    selection.selection.rows
                )

        except Exception:

            selected_rows = []

        # ----------------------------------------------------
        # SELECTED TRANSACTION
        # ----------------------------------------------------

        if selected_rows:

            selected_index = int(
                selected_rows[0]
            )

            if (
                0
                <= selected_index
                < len(results)
            ):

                st.markdown(
                    '<p class="section-title">🔍 Selected Transaction Details</p>',
                    unsafe_allow_html=True,
                )

                selected_probability = float(
                    uploaded_probabilities[
                        selected_index
                    ]
                )

                selected_prediction = int(
                    uploaded_predictions[
                        selected_index
                    ]
                )

                d1, d2, d3 = st.columns(3)

                d1.markdown(
                    f"""
                    <div class="detail-box">
                        <div class="detail-label">
                            Transaction #
                        </div>
                        <div class="detail-value">
                            {selected_index + 1}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                d2.markdown(
                    f"""
                    <div class="detail-box">
                        <div class="detail-label">
                            Prediction
                        </div>
                        <div class="detail-value">
                            {
                                "🔴 FRAUD"
                                if selected_prediction == 1
                                else "🟢 LEGIT"
                            }
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                d3.markdown(
                    f"""
                    <div class="detail-box">
                        <div class="detail-label">
                            Fraud Probability
                        </div>
                        <div class="detail-value">
                            {selected_probability * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if "Class" in active_df.columns:

                    actual_value = int(
                        active_df.iloc[
                            selected_index
                        ]["Class"]
                    )

                    actual_label = (
                        "🔴 FRAUD"
                        if actual_value == 1
                        else "🟢 LEGIT"
                    )

                    st.markdown(
                        f"""
                        <div class="detail-box">
                            <div class="detail-label">
                                Actual Label
                            </div>
                            <div class="detail-value">
                                {actual_label}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<p class="section-title">📋 Transaction Features</p>',
                    unsafe_allow_html=True,
                )

                selected_features = (
                    uploaded_features.iloc[
                        selected_index
                    ]
                    .to_frame(
                        name="Value"
                    )
                )

                st.dataframe(
                    selected_features,
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">⬇️ Export</p>',
            unsafe_allow_html=True,
        )

        csv_data = (
            results
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            "⬇️ Download Prediction Results",
            csv_data,
            "shieldai_results.csv",
            "text/csv",
            key="shieldai_download_final_v5",
        )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        """
        <h1 style="
            font-size:3rem;
            font-weight:900;
            color:#ffffff;
            margin:0;
        ">
            📊 Model Performance
        </h1>

        <p class="page-subtitle">
            Performance according to the currently active dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

    show_dataset_status()

    st.markdown("---")

    # --------------------------------------------------------
    # BEFORE CSV UPLOAD
    # --------------------------------------------------------

    if active_df is None:

        st.markdown(
            '<p class="section-title">🏆 Held-Out Test Set Performance</p>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Precision",
            f"{saved_metrics['precision']:.4f}",
        )

        m2.metric(
            "Recall",
            f"{saved_metrics['recall']:.4f}",
        )

        m3.metric(
            "F1 Score",
            f"{saved_metrics['f1']:.4f}",
        )

        m4.metric(
            "AUC-ROC",
            f"{saved_metrics['auc']:.4f}",
        )

        st.info(
            f"These are the official held-out test-set metrics "
            f"from {int(saved_metrics['test_size']):,} unseen transactions."
        )

    # --------------------------------------------------------
    # AFTER CSV UPLOAD
    # --------------------------------------------------------

    else:

        st.markdown(
            '<p class="section-title">📂 Uploaded Dataset Performance</p>',
            unsafe_allow_html=True,
        )

        if uploaded_eval is not None:

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Precision",
                f"{uploaded_eval['precision']:.4f}",
            )

            m2.metric(
                "Recall",
                f"{uploaded_eval['recall']:.4f}",
            )

            m3.metric(
                "F1 Score",
                f"{uploaded_eval['f1']:.4f}",
            )

            m4.metric(
                "AUC-ROC",
                f"{uploaded_eval['auc']:.4f}",
            )

            st.success(
                "These metrics are calculated from the currently uploaded CSV."
            )

            st.warning(
                "These are uploaded-dataset evaluation metrics, "
                "not the official held-out test-set metrics."
            )

            # Confusion Matrix

            cm = uploaded_eval[
                "confusion_matrix"
            ]

            st.markdown(
                '<p class="section-title">🔲 Confusion Matrix</p>',
                unsafe_allow_html=True,
            )

            fig = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=[
                        "Predicted Legit",
                        "Predicted Fraud",
                    ],
                    y=[
                        "Actual Legit",
                        "Actual Fraud",
                    ],
                    colorscale=[
                        [0, "#0d2137"],
                        [0.5, "#1565c0"],
                        [1, "#2979ff"],
                    ],
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(
                        color="white",
                        size=16,
                    ),
                )
            )

            fig.update_layout(
                title="Uploaded Dataset Confusion Matrix",
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                font_color="white",
                height=400,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            total = len(
                uploaded_predictions
            )

            fraud = int(
                uploaded_predictions.sum()
            )

            legit = total - fraud

            st.warning(
                "The uploaded CSV does not contain a Class column. "
                "Precision, Recall, F1 and AUC cannot be calculated."
            )

            u1, u2, u3 = st.columns(3)

            u1.metric(
                "Transactions",
                f"{total:,}",
            )

            u2.metric(
                "Predicted Legit",
                f"{legit:,}",
            )

            u3.metric(
                "Predicted Fraud",
                f"{fraud:,}",
            )


# ============================================================
# PAGE 5 — ABOUT
# IMPORTANT: NO upload-dependent content here.
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <h1 style="
            font-size:3rem;
            font-weight:900;
            color:#ffffff;
            margin:0;
        ">
            ℹ️ About ShieldAI
        </h1>

        <p class="page-subtitle">
            Credit Card Fraud Detection using Autoencoder
            Feature Learning + Random Forest Classification
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    left, right = st.columns(2)

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<p class="section-title">🧠 Detection Pipeline</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **1. Autoencoder**

            Learns compact representations of transaction
            patterns using legitimate transaction data.

            **2. Latent Features**

            Extracted features from the Autoencoder are combined
            with the original standardized transaction features.

            **3. Random Forest**

            Uses the hybrid feature representation to classify
            transactions as legitimate or fraudulent.

            **4. Optimized Threshold**

            The fraud probability threshold is selected using
            validation data to balance precision and recall.
            """
        )

    # --------------------------------------------------------
    # TECHNOLOGY
    # No raw HTML here.
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<p class="section-title">🛠️ Technology Stack</p>',
            unsafe_allow_html=True,
        )

        technologies = [
            "🧠 TensorFlow / Keras",
            "🔍 Autoencoder",
            "🌲 Random Forest",
            "📊 Scikit-learn",
            "🐼 Pandas",
            "🔢 NumPy",
            "🚀 Streamlit",
            "📈 Plotly",
        ]

        for start in range(
            0,
            len(technologies),
            2,
        ):

            columns = st.columns(2)

            for index, column in enumerate(
                columns
            ):

                item_index = start + index

                if item_index >= len(
                    technologies
                ):
                    continue

                column.info(
                    technologies[item_index]
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

if active_df is None:

    st.caption(
        "ShieldAI • Saved held-out test-set information"
    )

else:

    st.caption(
        "ShieldAI • Active uploaded dataset • "
        f"{len(active_df):,} transactions"
    )