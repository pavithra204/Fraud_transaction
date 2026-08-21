import io
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
# OPTIONAL CPU THREAD LIMIT
# Helps reduce excessive CPU pressure on Streamlit Cloud.
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass


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

CHUNK_SIZE = 10_000


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
        filename
        for filename in required_files
        if not os.path.exists(filename)
    ]

    if missing:

        st.error(
            "Missing model files: "
            + ", ".join(missing)
        )

        st.stop()

    # Random Forest
    with open(
        MODEL_FILE,
        "rb",
    ) as file:

        model = pickle.load(file)

    # Autoencoder
    autoencoder = tf.keras.models.load_model(
        AUTOENCODER_FILE
    )

    # Scaler
    with open(
        SCALER_FILE,
        "rb",
    ) as file:

        scaler = pickle.load(file)

    # Threshold
    with open(
        THRESHOLD_FILE,
        "rb",
    ) as file:

        threshold = float(
            pickle.load(file)
        )

    # Features
    with open(
        FEATURES_FILE,
        "rb",
    ) as file:

        feature_names = pickle.load(file)

    # Metrics
    with open(
        METRICS_FILE,
        "rb",
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
# VALIDATE DATAFRAME
# ============================================================

def validate_dataframe(
    df,
):

    errors = []

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_columns = [
        name
        for name in feature_names
        if name not in df.columns
    ]

    if missing_columns:

        errors.append(
            {
                "type": "columns",
                "columns": missing_columns,
            }
        )

        return errors

    feature_df = df[
        feature_names
    ].copy()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_counts = (
        feature_df.isna().sum()
    )

    missing_counts = missing_counts[
        missing_counts > 0
    ]

    if not missing_counts.empty:

        errors.append(
            {
                "type": "missing",
                "details": missing_counts.to_dict(),
            }
        )

    # --------------------------------------------------------
    # Non-numeric values
    # --------------------------------------------------------

    non_numeric = []

    for column in feature_names:

        converted = pd.to_numeric(
            feature_df[column],
            errors="coerce",
        )

        invalid_mask = (
            converted.isna()
            & feature_df[column].notna()
        )

        if invalid_mask.any():

            non_numeric.append(
                column
            )

    if non_numeric:

        errors.append(
            {
                "type": "numeric",
                "columns": non_numeric,
            }
        )

    return errors


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
):

    X = df[
        feature_names
    ].copy()

    X = X.apply(
        pd.to_numeric,
        errors="raise",
    )

    return X


# ============================================================
# PREDICT SMALL/SINGLE DATAFRAME
# ============================================================

def predict_dataframe(
    df,
):

    errors = validate_dataframe(
        df
    )

    if errors:

        raise ValueError(
            "The dataset failed validation."
        )

    X = prepare_features(
        df
    )

    # Keep DataFrame column names.
    X_scaled = scaler.transform(
        X
    )

    latent = encoder.predict(
        X_scaled,
        batch_size=1024,
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
# DISPLAY VALIDATION ERRORS
# ============================================================

def show_validation_errors(
    errors,
):

    for error in errors:

        if error["type"] == "columns":

            st.error(
                "❌ Required transaction features are missing."
            )

            st.write(
                "Missing columns:"
            )

            st.code(
                "\n".join(
                    error["columns"]
                )
            )

        elif error["type"] == "missing":

            st.error(
                "❌ Missing values detected."
            )

            missing_df = pd.DataFrame(
                [
                    {
                        "Column": column,
                        "Missing Values": count,
                    }
                    for column, count
                    in error["details"].items()
                ]
            )

            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
            )

        elif error["type"] == "numeric":

            st.error(
                "❌ Non-numeric values detected."
            )

            st.write(
                "These columns contain invalid text/non-numeric values:"
            )

            st.code(
                "\n".join(
                    error["columns"]
                )
            )

    st.warning(
        "Please upload a corrected CSV. "
        "Prediction has been stopped."
    )


# ============================================================
# CHUNKED BATCH PREDICTION
# ============================================================

def run_chunked_prediction(
    df,
):

    errors = validate_dataframe(
        df
    )

    if errors:

        show_validation_errors(
            errors
        )

        return None

    total_rows = len(
        df
    )

    probability_chunks = []
    prediction_chunks = []

    progress = st.progress(
        0
    )

    status = st.empty()

    for start in range(
        0,
        total_rows,
        CHUNK_SIZE,
    ):

        end = min(
            start + CHUNK_SIZE,
            total_rows,
        )

        chunk = df.iloc[
            start:end
        ]

        probabilities, predictions = (
            predict_dataframe(
                chunk
            )
        )

        probability_chunks.append(
            probabilities
        )

        prediction_chunks.append(
            predictions
        )

        progress_value = (
            end / total_rows
        )

        progress.progress(
            progress_value
        )

        status.text(
            f"Processed {end:,} / {total_rows:,} transactions"
        )

    progress.empty()
    status.empty()

    probabilities = np.concatenate(
        probability_chunks
    )

    predictions = np.concatenate(
        prediction_chunks
    )

    return (
        probabilities,
        predictions,
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    df,
    predictions,
    probabilities,
):

    if "Class" not in df.columns:

        return None

    actual = pd.to_numeric(
        df["Class"],
        errors="coerce",
    )

    if actual.isna().any():

        return None

    actual = actual.astype(
        int
    ).to_numpy()

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
# INITIALIZE SESSION STATE
# ============================================================

if "active_file_name" not in st.session_state:
    st.session_state.active_file_name = None

if "active_file_bytes" not in st.session_state:
    st.session_state.active_file_bytes = None

if "active_df" not in st.session_state:
    st.session_state.active_df = None

if "active_probabilities" not in st.session_state:
    st.session_state.active_probabilities = None

if "active_predictions" not in st.session_state:
    st.session_state.active_predictions = None

if "active_metrics" not in st.session_state:
    st.session_state.active_metrics = None

if "active_validation_errors" not in st.session_state:
    st.session_state.active_validation_errors = None


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
        key="shieldai_navigation_final",
    )

    st.divider()

    st.subheader(
        "📁 Transaction Data"
    )

    st.caption(
        "Upload once. The same file stays active "
        "while navigating."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="shieldai_upload_final",
    )


# ============================================================
# HANDLE NEW UPLOAD
# ============================================================

if uploaded is not None:

    file_bytes = uploaded.getvalue()

    new_file = (
        st.session_state.active_file_bytes
        != file_bytes
    )

    if new_file:

        # Reset previous analysis.
        st.session_state.active_file_name = (
            uploaded.name
        )

        st.session_state.active_file_bytes = (
            file_bytes
        )

        st.session_state.active_df = None
        st.session_state.active_probabilities = None
        st.session_state.active_predictions = None
        st.session_state.active_metrics = None
        st.session_state.active_validation_errors = None

        try:

            df_uploaded = pd.read_csv(
                io.BytesIO(
                    file_bytes
                )
            )

            validation_errors = validate_dataframe(
                df_uploaded
            )

            if validation_errors:

                st.session_state.active_validation_errors = (
                    validation_errors
                )

                st.session_state.active_df = (
                    df_uploaded
                )

            else:

                st.session_state.active_df = (
                    df_uploaded
                )

        except Exception as error:

            st.session_state.active_validation_errors = [
                {
                    "type": "read",
                    "message": str(
                        error
                    ),
                }
            ]

# ============================================================
# NO UPLOAD
# ============================================================

if uploaded is None:

    # Keep previous session only if Streamlit retains it.
    # Otherwise show default state.

    pass


# ============================================================
# ACTIVE DATA
# ============================================================

active_df = (
    st.session_state.active_df
)

active_predictions = (
    st.session_state.active_predictions
)

active_probabilities = (
    st.session_state.active_probabilities
)

active_metrics = (
    st.session_state.active_metrics
)

active_validation_errors = (
    st.session_state.active_validation_errors
)


# ============================================================
# DATASET STATUS
# ============================================================

def show_dataset_status():

    if active_df is None:

        st.info(
            "No CSV uploaded — showing saved held-out test-set information."
        )

    else:

        if active_validation_errors:

            st.error(
                "Uploaded CSV is invalid."
            )

        else:

            class_text = (
                "Class column available"
                if "Class" in active_df.columns
                else "No Class column"
            )

            st.success(
                f"Active dataset: "
                f"{len(active_df):,} transactions • "
                f"{class_text}"
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
    # INVALID UPLOAD
    # --------------------------------------------------------

    if active_validation_errors:

        show_validation_errors(
            active_validation_errors
        )

    # --------------------------------------------------------
    # BEFORE UPLOAD
    # --------------------------------------------------------

    elif active_df is None:

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
            f"Official held-out test metrics from "
            f"{int(saved_metrics['test_size']):,} unseen transactions."
        )

    # --------------------------------------------------------
    # AFTER VALID UPLOAD
    # --------------------------------------------------------

    elif active_predictions is None:

        st.info(
            "CSV uploaded successfully. "
            "Open Batch Prediction to run the model."
        )

        st.markdown(
            '<p class="section-title">📂 Uploaded Dataset</p>',
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Transactions",
            f"{len(active_df):,}",
        )

        d2.metric(
            "Columns",
            len(active_df.columns),
        )

        d3.metric(
            "Model Features",
            len(feature_names),
        )

    # --------------------------------------------------------
    # AFTER BATCH ANALYSIS
    # --------------------------------------------------------

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

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Total Transactions",
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

        if active_metrics is not None:

            st.markdown(
                '<p class="section-title">📈 Uploaded Dataset Performance</p>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Precision",
                f"{active_metrics['precision']:.4f}",
            )

            m2.metric(
                "Recall",
                f"{active_metrics['recall']:.4f}",
            )

            m3.metric(
                "F1 Score",
                f"{active_metrics['f1']:.4f}",
            )

            m4.metric(
                "AUC-ROC",
                f"{active_metrics['auc']:.4f}",
            )

            st.caption(
                "Metrics calculated from the currently uploaded CSV."
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

    show_dataset_status()

    st.markdown("---")

    if active_validation_errors:

        show_validation_errors(
            active_validation_errors
        )

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

            index = (
                start
                + offset
            )

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
                    key="single_amount_final_cloud",
                )

            elif name == "Time":

                value = column.number_input(
                    "Time",
                    value=0.0,
                    key="single_time_final_cloud",
                )

            else:

                value = column.number_input(
                    name,
                    value=0.0,
                    format="%.6f",
                    key=f"single_{index}_final_cloud",
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
        key="single_predict_final_cloud",
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
            Process the active CSV in memory-efficient chunks.
        </p>
        """,
        unsafe_allow_html=True,
    )

    show_dataset_status()

    st.markdown("---")

    if active_df is None:

        st.info(
            "Upload a CSV using the sidebar."
        )

    elif active_validation_errors:

        show_validation_errors(
            active_validation_errors
        )

    elif active_predictions is None:

        st.markdown(
            '<p class="section-title">🚀 Ready for Analysis</p>',
            unsafe_allow_html=True,
        )

        st.write(
            f"Dataset contains "
            f"**{len(active_df):,} transactions**."
        )

        if st.button(
            "⚡ RUN BATCH FRAUD DETECTION",
            key="run_batch_final_cloud",
        ):

            try:

                result = run_chunked_prediction(
                    active_df
                )

                if result is not None:

                    (
                        probabilities,
                        predictions,
                    ) = result

                    st.session_state.active_probabilities = (
                        probabilities
                    )

                    st.session_state.active_predictions = (
                        predictions
                    )

                    st.session_state.active_metrics = (
                        calculate_metrics(
                            active_df,
                            predictions,
                            probabilities,
                        )
                    )

                    st.rerun()

            except Exception as error:

                st.error(
                    f"Batch prediction failed: {error}"
                )

    else:

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
        # UPLOADED METRICS
        # ----------------------------------------------------

        if active_metrics is not None:

            st.markdown(
                '<p class="section-title">📈 Uploaded Dataset Evaluation</p>',
                unsafe_allow_html=True,
            )

            e1, e2, e3, e4 = st.columns(4)

            e1.metric(
                "Precision",
                f"{active_metrics['precision']:.4f}",
            )

            e2.metric(
                "Recall",
                f"{active_metrics['recall']:.4f}",
            )

            e3.metric(
                "F1 Score",
                f"{active_metrics['f1']:.4f}",
            )

            e4.metric(
                "AUC-ROC",
                f"{active_metrics['auc']:.4f}",
            )

            cm = active_metrics[
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

        else:

            st.info(
                "This CSV has no Class column, so "
                "Precision/Recall/F1/AUC cannot be calculated."
            )

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">🔎 Search Transactions</p>',
            unsafe_allow_html=True,
        )

        filter_col, search_col = st.columns(
            [1, 2]
        )

        with filter_col:

            prediction_filter = st.selectbox(
                "Prediction",
                [
                    "All",
                    "Fraud only",
                    "Legit only",
                ],
                key="prediction_filter_final",
            )

        with search_col:

            search_text = st.text_input(
                "Search transaction number or prediction",
                placeholder="Example: 1250 or FRAUD",
                key="search_transaction_final",
            )

        # ----------------------------------------------------
        # BUILD DISPLAY TABLE
        # ----------------------------------------------------

        display_results = active_df.copy()

        display_results.insert(
            0,
            "Transaction #",
            np.arange(
                1,
                len(display_results) + 1
            ),
        )

        display_results[
            "Fraud Probability"
        ] = active_probabilities.round(
            6
        )

        display_results[
            "Predicted"
        ] = np.where(
            active_predictions == 1,
            "🔴 FRAUD",
            "🟢 LEGIT",
        )

        # Filter
        if prediction_filter == "Fraud only":

            display_results = display_results[
                active_predictions == 1
            ].copy()

        elif prediction_filter == "Legit only":

            display_results = display_results[
                active_predictions == 0
            ].copy()

        # Search
        if search_text.strip():

            query = (
                search_text
                .strip()
                .lower()
            )

            transaction_match = (
                display_results[
                    "Transaction #"
                ]
                .astype(str)
                .str.contains(
                    query,
                    na=False,
                )
            )

            prediction_match = (
                display_results[
                    "Predicted"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    na=False,
                )
            )

            display_results = (
                display_results[
                    transaction_match
                    | prediction_match
                ]
            )

        st.caption(
            f"Showing {min(len(display_results), 1000):,} "
            f"of {len(display_results):,} matching rows."
        )

        # Avoid rendering 284K rows in Streamlit.
        table_preview = display_results.head(
            1000
        )

        st.dataframe(
            table_preview,
            use_container_width=True,
            height=500,
            hide_index=True,
        )

        # ----------------------------------------------------
        # SELECT TRANSACTION NUMBER
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">🔍 Inspect Transaction</p>',
            unsafe_allow_html=True,
        )

        if len(display_results) > 0:

            transaction_options = (
                display_results[
                    "Transaction #"
                ]
                .head(1000)
                .astype(int)
                .tolist()
            )

            selected_transaction = st.selectbox(
                "Select Transaction #",
                transaction_options,
                key="selected_transaction_final",
            )

            selected_index = (
                int(
                    selected_transaction
                )
                - 1
            )

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

            d1, d2, d3 = st.columns(3)

            d1.markdown(
                f"""
                <div class="detail-box">
                    <div class="detail-label">
                        Transaction #
                    </div>
                    <div class="detail-value">
                        {selected_transaction}
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

            # Actual Class
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

            # Features
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

        else:

            st.warning(
                "No transactions matched your search."
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.markdown(
            '<p class="section-title">⬇️ Export</p>',
            unsafe_allow_html=True,
        )

        export_results = active_df.copy()

        export_results[
            "Fraud Probability"
        ] = active_probabilities

        export_results[
            "Predicted"
        ] = np.where(
            active_predictions == 1,
            "FRAUD",
            "LEGIT",
        )

        csv_data = (
            export_results
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
            key="download_final_cloud",
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
            Performance for the current dataset state.
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
            f"Official held-out test metrics from "
            f"{int(saved_metrics['test_size']):,} unseen transactions."
        )

    # --------------------------------------------------------
    # INVALID UPLOAD
    # --------------------------------------------------------

    elif active_validation_errors:

        show_validation_errors(
            active_validation_errors
        )

    # --------------------------------------------------------
    # UPLOAD BUT NOT ANALYZED
    # --------------------------------------------------------

    elif active_predictions is None:

        st.info(
            "The CSV is loaded and validated. "
            "Open Batch Prediction and click "
            "'RUN BATCH FRAUD DETECTION' to calculate "
            "the uploaded dataset results."
        )

        a, b, c = st.columns(3)

        a.metric(
            "Transactions",
            f"{len(active_df):,}",
        )

        b.metric(
            "Features",
            len(feature_names),
        )

        c.metric(
            "Decision Threshold",
            f"{threshold:.3f}",
        )

    # --------------------------------------------------------
    # UPLOADED DATASET ANALYSIS
    # --------------------------------------------------------

    else:

        if active_metrics is not None:

            st.markdown(
                '<p class="section-title">📂 Uploaded Dataset Performance</p>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Precision",
                f"{active_metrics['precision']:.4f}",
            )

            m2.metric(
                "Recall",
                f"{active_metrics['recall']:.4f}",
            )

            m3.metric(
                "F1 Score",
                f"{active_metrics['f1']:.4f}",
            )

            m4.metric(
                "AUC-ROC",
                f"{active_metrics['auc']:.4f}",
            )

            st.success(
                "These metrics are calculated from the currently uploaded CSV."
            )

            st.warning(
                "These are uploaded-dataset evaluation metrics, "
                "not the official held-out test metrics."
            )

            cm = active_metrics[
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
                "The uploaded CSV does not contain a usable Class column."
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
# STATIC PAGE
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

            Extracted Autoencoder features are combined with
            the original standardized transaction features.

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

                item_index = (
                    start + index
                )

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

elif active_validation_errors:

    st.caption(
        "ShieldAI • Uploaded file requires correction"
    )

else:

    st.caption(
        "ShieldAI • Active dataset • "
        f"{len(active_df):,} transactions"
    )