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
# FILES
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
# LOAD MODEL ARTIFACTS
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

    missing = [
        name
        for name in required_files
        if not os.path.exists(name)
    ]

    if missing:
        st.error(
            "Missing model files: "
            + ", ".join(missing)
        )
        st.stop()

    with open(
        MODEL_FILE,
        "rb"
    ) as file:
        model = pickle.load(file)

    autoencoder = tf.keras.models.load_model(
        AUTOENCODER_FILE
    )

    with open(
        SCALER_FILE,
        "rb"
    ) as file:
        scaler = pickle.load(file)

    with open(
        THRESHOLD_FILE,
        "rb"
    ) as file:
        threshold = float(
            pickle.load(file)
        )

    with open(
        FEATURES_FILE,
        "rb"
    ) as file:
        feature_names = pickle.load(file)

    with open(
        METRICS_FILE,
        "rb"
    ) as file:
        metrics = pickle.load(file)

    # Build encoder once.
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
# PREPARE DATA
# ============================================================

def prepare_data(df):

    X = df.drop(
        columns=["Class"],
        errors="ignore",
    ).copy()

    missing_columns = [
        name
        for name in feature_names
        if name not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_columns)
        )

    # Keep exact training feature order.
    X = X[feature_names]

    # Numeric conversion.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():
        raise ValueError(
            "CSV contains missing or non-numeric values."
        )

    return X


# ============================================================
# PURE MODEL PREDICTION
# ============================================================

def predict_dataframe(df):

    X = prepare_data(
        df
    )

    # IMPORTANT:
    # Pass the DataFrame, not X.values.
    # This removes the StandardScaler feature-name warning.
    X_scaled = scaler.transform(
        X
    )

    latent = encoder.predict(
        X_scaled,
        batch_size=4096,
        verbose=0,
    )

    hybrid = np.concatenate(
        [
            X_scaled,
            latent,
        ],
        axis=1,
    )

    probabilities = model.predict_proba(
        hybrid
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    return (
        probabilities,
        predictions,
    )


# ============================================================
# CACHED UPLOADED-DATA PREDICTION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=2,
)
def predict_uploaded_csv(
    csv_bytes,
):

    df = pd.read_csv(
        pd.io.common.BytesIO(
            csv_bytes
        )
    )

    probabilities, predictions = (
        predict_dataframe(
            df
        )
    )

    return (
        df,
        probabilities,
        predictions,
    )


# ============================================================
# UPLOADED CSV METRICS
# ============================================================

def calculate_metrics(
    df,
    predictions,
    probabilities,
):

    if "Class" not in df.columns:
        return None

    actual = (
        pd.to_numeric(
            df["Class"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )

    # Need both classes.
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
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "auc": float(
            auc
        ),
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

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Single Transaction",
            "📂 Batch Prediction",
            "📊 Model Performance",
            "ℹ️ About",
        ],
        key="shieldai_navigation_final_cloud",
    )

    st.divider()

    st.subheader(
        "📁 Transaction Data"
    )

    st.caption(
        "Upload once. The active CSV stays available "
        "while you navigate."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="shieldai_upload_final_cloud",
    )

    st.divider()

    if uploaded is None:

        st.caption(
            "No CSV uploaded — showing saved test-set information."
        )

    else:

        st.success(
            f"Loaded: {uploaded.name}"
        )

    st.divider()

    st.caption(
        "Built with Autoencoder + Random Forest"
    )


# ============================================================
# GET UPLOAD BYTES
# ============================================================

uploaded_bytes = None

if uploaded is not None:

    uploaded_bytes = uploaded.getvalue()


# ============================================================
# ACTIVE UPLOAD
# IMPORTANT:
# Prediction is cached. It will NOT retrain/recalculate on
# every navigation click.
# ============================================================

active_df = None
active_probabilities = None
active_predictions = None
active_eval = None

if uploaded_bytes is not None:

    try:

        (
            active_df,
            active_probabilities,
            active_predictions,
        ) = predict_uploaded_csv(
            uploaded_bytes
        )

        active_eval = calculate_metrics(
            active_df,
            active_predictions,
            active_probabilities,
        )

    except Exception as error:

        st.error(
            f"Uploaded CSV could not be processed: {error}"
        )


# ============================================================
# DATASET STATUS
# ============================================================

def dataset_status():

    if active_df is None:

        st.info(
            "No CSV uploaded — showing saved held-out test-set information."
        )

    else:

        class_text = (
            "Class column available"
            if "Class" in active_df.columns
            else "No Class column"
        )

        st.success(
            f"Active dataset: {len(active_df):,} transactions "
            f"• {class_text}"
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

    dataset_status()

    st.markdown("---")

    # --------------------------------------------------------
    # NO UPLOAD
    # --------------------------------------------------------

    if active_df is None:

        st.markdown(
            '<p class="section-title">🤖 Model Information</p>',
            unsafe_allow_html=True,
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Input Features",
            len(feature_names),
        )

        b.metric(
            "Decision Threshold",
            f"{threshold:.3f}",
        )

        c.metric(
            "Test Precision",
            f"{saved_metrics['precision']:.4f}",
        )

        d.metric(
            "Test Recall",
            f"{saved_metrics['recall']:.4f}",
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

        st.info(
            f"These are the official held-out test metrics "
            f"from {int(saved_metrics['test_size']):,} unseen transactions."
        )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    else:

        if active_predictions is None:

            st.error(
                "The uploaded CSV could not be processed."
            )

        else:

            total = len(
                active_predictions
            )

            fraud = int(
                active_predictions.sum()
            )

            legit = total - fraud

            st.markdown(
                '<p class="section-title">📂 Uploaded Dataset</p>',
                unsafe_allow_html=True,
            )

            a, b, c = st.columns(3)

            a.metric(
                "Total Transactions",
                f"{total:,}",
            )

            b.metric(
                "Predicted Legit",
                f"{legit:,}",
            )

            c.metric(
                "Predicted Fraud",
                f"{fraud:,}",
            )

            if active_eval is not None:

                st.markdown(
                    '<p class="section-title">📈 Uploaded Dataset Performance</p>',
                    unsafe_allow_html=True,
                )

                p1, p2, p3, p4 = st.columns(4)

                p1.metric(
                    "Precision",
                    f"{active_eval['precision']:.4f}",
                )

                p2.metric(
                    "Recall",
                    f"{active_eval['recall']:.4f}",
                )

                p3.metric(
                    "F1 Score",
                    f"{active_eval['f1']:.4f}",
                )

                p4.metric(
                    "AUC-ROC",
                    f"{active_eval['auc']:.4f}",
                )

                st.caption(
                    "These metrics are calculated from the currently uploaded CSV."
                )

            else:

                st.warning(
                    "This CSV has no usable Class labels, so "
                    "Precision, Recall, F1 and AUC cannot be calculated."
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
            Enter one transaction and classify it.
        </p>
        """,
        unsafe_allow_html=True,
    )

    dataset_status()

    st.markdown("---")

    values = []

    for start in range(
        0,
        len(feature_names),
        5,
    ):

        columns = st.columns(5)

        for offset, column in enumerate(
            columns
        ):

            index = start + offset

            if index >= len(
                feature_names
            ):

                continue

            name = feature_names[
                index
            ]

            if name == "Amount":

                value = column.number_input(
                    "Amount",
                    min_value=0.0,
                    value=100.0,
                    key="single_amount_cloud_final",
                )

            elif name == "Time":

                value = column.number_input(
                    "Time",
                    value=0.0,
                    key="single_time_cloud_final",
                )

            else:

                value = column.number_input(
                    name,
                    value=0.0,
                    format="%.6f",
                    key=f"single_{index}_cloud_final",
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
        key="single_predict_cloud_final",
    ):

        sample = pd.DataFrame(
            [values],
            columns=feature_names,
        )

        try:

            probabilities, predictions = (
                predict_dataframe(
                    sample
                )
            )

            fraud_probability = float(
                probabilities[0]
            )

            is_fraud = bool(
                predictions[0]
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

            x1, x2 = st.columns(2)

            x1.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%",
            )

            x2.metric(
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
            Search, select and inspect transactions from the active CSV.
        </p>
        """,
        unsafe_allow_html=True,
    )

    dataset_status()

    st.markdown("---")

    if active_df is None:

        st.info(
            "Upload a CSV from the sidebar."
        )

    elif active_predictions is None:

        st.error(
            "The uploaded CSV could not be processed."
        )

    else:

        # ----------------------------------------------------
        # RESULT DATAFRAME
        # ----------------------------------------------------

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
        ] = active_probabilities.round(
            6
        )

        results[
            "Predicted"
        ] = [
            "🔴 FRAUD"
            if p == 1
            else "🟢 LEGIT"
            for p in active_predictions
        ]

        total = len(
            active_predictions
        )

        fraud = int(
            active_predictions.sum()
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
        # CHARTS
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
                title="Legitimate vs Fraud",
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
        # UPLOADED CSV METRICS
        # ----------------------------------------------------

        if active_eval is not None:

            st.markdown(
                '<p class="section-title">📈 Uploaded Dataset Evaluation</p>',
                unsafe_allow_html=True,
            )

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "Precision",
                f"{active_eval['precision']:.4f}",
            )

            e2.metric(
                "Recall",
                f"{active_eval['recall']:.4f}",
            )

            e3.metric(
                "F1 Score",
                f"{active_eval['f1']:.4f}",
            )

            e4.metric(
                "AUC-ROC",
                f"{active_eval['auc']:.4f}",
            )

            cm = active_eval[
                "confusion_matrix"
            ]

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
                    text=cm,
                    texttemplate="%{text}",
                )
            )

            matrix.update_layout(
                title="Confusion Matrix",
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                font_color="white",
                height=350,
            )

            st.plotly_chart(
                matrix,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # SEARCHABLE TABLE
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">🔎 Search & Select Transactions</p>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Search FRAUD or LEGIT using the table search, "
            "then select a row."
        )

        selection = st.dataframe(
            results,
            use_container_width=True,
            height=500,
            hide_index=True,
            key="shieldai_batch_results_cloud_final",
            on_select="rerun",
            selection_mode="single-row",
        )

        # ----------------------------------------------------
        # SELECTED ROW
        # ----------------------------------------------------

        selected_rows = []

        try:

            selected_rows = (
                selection.selection.rows
            )

        except Exception:

            selected_rows = []

        if selected_rows:

            selected_index = int(
                selected_rows[0]
            )

            if (
                0
                <= selected_index
                < len(results)
            ):

                selected_probability = float(
                    active_probabilities[
                        selected_index
                    ]
                )

                selected_prediction = int(
                    active_predictions[
                        selected_index
                    ]
                )

                st.markdown(
                    '<p class="section-title">🔍 Selected Transaction Details</p>',
                    unsafe_allow_html=True,
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
                    active_df.iloc[
                        selected_index
                    ]
                    .drop(
                        labels=["Class"],
                        errors="ignore",
                    )
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
            key="shieldai_download_cloud_final",
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
            Performance for the currently active dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

    dataset_status()

    st.markdown("---")

    # --------------------------------------------------------
    # BEFORE UPLOAD
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
            f"Official held-out test results from "
            f"{int(saved_metrics['test_size']):,} unseen transactions."
        )

    # --------------------------------------------------------
    # AFTER UPLOAD
    # --------------------------------------------------------

    else:

        if active_eval is not None:

            st.markdown(
                '<p class="section-title">📂 Uploaded Dataset Performance</p>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Precision",
                f"{active_eval['precision']:.4f}",
            )

            m2.metric(
                "Recall",
                f"{active_eval['recall']:.4f}",
            )

            m3.metric(
                "F1 Score",
                f"{active_eval['f1']:.4f}",
            )

            m4.metric(
                "AUC-ROC",
                f"{active_eval['auc']:.4f}",
            )

            st.success(
                "These metrics are calculated from the currently uploaded CSV."
            )

            st.warning(
                "These are uploaded-dataset metrics, not the official "
                "held-out test-set metrics."
            )

            cm = active_eval[
                "confusion_matrix"
            ]

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
                    text=cm,
                    texttemplate="%{text}",
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
                active_predictions
            )

            fraud = int(
                active_predictions.sum()
            )

            legit = total - fraud

            st.warning(
                "The uploaded CSV does not contain a usable Class column. "
                "Metrics such as Precision, Recall, F1 and AUC cannot be calculated."
            )

            a, b, c = st.columns(3)

            a.metric(
                "Transactions",
                f"{total:,}",
            )

            b.metric(
                "Predicted Legit",
                f"{legit:,}",
            )

            c.metric(
                "Predicted Fraud",
                f"{fraud:,}",
            )


# ============================================================
# PAGE 5 — ABOUT
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

            Extracted Autoencoder features are combined
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