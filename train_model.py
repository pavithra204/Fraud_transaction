import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# FILES
# ============================================================

DATA_FILE = "creditcard.csv"

AUTOENCODER_FILE = "autoencoder.keras"
SCALER_FILE = "scaler.pkl"
MODEL_FILE = "model.pkl"
THRESHOLD_FILE = "threshold.pkl"
FEATURES_FILE = "features.pkl"
METRICS_FILE = "metrics.pkl"


# ============================================================
# AUTOENCODER
# ============================================================

def build_autoencoder(n_features):

    inputs = Input(shape=(n_features,))

    x = Dense(
        min(n_features, 32),
        activation="relu"
    )(inputs)

    x = Dropout(0.1)(x)

    x = Dense(
        16,
        activation="relu"
    )(x)

    latent = Dense(
        8,
        activation="relu",
        name="latent"
    )(x)

    x = Dense(
        16,
        activation="relu"
    )(latent)

    x = Dropout(0.1)(x)

    x = Dense(
        min(n_features, 32),
        activation="relu"
    )(x)

    outputs = Dense(
        n_features,
        activation="linear"
    )(x)

    autoencoder = Model(
        inputs,
        outputs
    )

    encoder = Model(
        inputs,
        latent
    )

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    return autoencoder, encoder


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHIELDAI - AUTOENCODER + RANDOM FOREST")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1/8] Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    if "Class" not in df.columns:
        raise ValueError(
            "creditcard.csv must contain a 'Class' column."
        )

    X = df.drop(
        columns=["Class"]
    )

    y = df["Class"].astype(int)

    print(
        f"Total transactions : {len(df):,}"
    )

    print(
        f"Legitimate         : {(y == 0).sum():,}"
    )

    print(
        f"Fraud              : {(y == 1).sum():,}"
    )

    print(
        f"Features            : {X.shape[1]}"
    )

    # ========================================================
    # 2. TRAIN / VALIDATION / TEST
    # ========================================================

    print("\n[2/8] Splitting data...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print(
        f"Train      : {len(X_train):,}"
    )

    print(
        f"Validation : {len(X_val):,}"
    )

    print(
        f"Test       : {len(X_test):,}"
    )

    # ========================================================
    # 3. SCALE FEATURES
    # ========================================================

    print("\n[3/8] Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================
    # 4. TRAIN AUTOENCODER ONLY ON LEGITIMATE TRAIN DATA
    # ========================================================

    print(
        "\n[4/8] Training Autoencoder on legitimate transactions..."
    )

    legitimate_mask = (
        y_train.values == 0
    )

    X_train_legitimate = (
        X_train_scaled[legitimate_mask]
    )

    print(
        f"Legitimate training rows : "
        f"{len(X_train_legitimate):,}"
    )

    autoencoder, encoder = (
        build_autoencoder(
            X_train_scaled.shape[1]
        )
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = autoencoder.fit(
        X_train_legitimate,
        X_train_legitimate,
        epochs=20,
        batch_size=256,
        validation_split=0.10,
        callbacks=[early_stopping],
        verbose=1,
    )

    # ========================================================
    # 5. GENERATE LATENT FEATURES
    # ========================================================

    print(
        "\n[5/8] Generating Autoencoder latent features..."
    )

    latent_train = encoder.predict(
        X_train_scaled,
        batch_size=1024,
        verbose=0,
    )

    latent_val = encoder.predict(
        X_val_scaled,
        batch_size=1024,
        verbose=0,
    )

    latent_test = encoder.predict(
        X_test_scaled,
        batch_size=1024,
        verbose=0,
    )

    # Hybrid features:
    # original scaled features + Autoencoder latent features

    hybrid_train = np.concatenate(
        [
            X_train_scaled,
            latent_train,
        ],
        axis=1,
    )

    hybrid_val = np.concatenate(
        [
            X_val_scaled,
            latent_val,
        ],
        axis=1,
    )

    hybrid_test = np.concatenate(
        [
            X_test_scaled,
            latent_test,
        ],
        axis=1,
    )

    print(
        f"Hybrid feature count : "
        f"{hybrid_train.shape[1]}"
    )

    # ========================================================
    # 6. TRAIN RANDOM FOREST
    # ========================================================

    print(
        "\n[6/8] Training Random Forest..."
    )

    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(
        hybrid_train,
        y_train,
    )

    # ========================================================
    # 7. THRESHOLD TUNING + FINAL TEST
    # ========================================================

    print(
        "\n[7/8] Tuning threshold..."
    )

    validation_probabilities = (
        rf_model.predict_proba(
            hybrid_val
        )[:, 1]
    )

    thresholds = np.arange(
        0.05,
        0.951,
        0.005,
    )

    # Primary objective:
    # Keep recall >= 80%, maximize precision.
    MIN_RECALL = 0.80

    best_threshold = None
    best_precision = -1.0
    best_recall = 0.0
    best_f1 = 0.0

    for threshold in thresholds:

        validation_predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            validation_predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            validation_predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            validation_predictions,
            zero_division=0,
        )

        if recall >= MIN_RECALL:

            if (
                precision > best_precision
                or (
                    precision == best_precision
                    and f1 > best_f1
                )
            ):

                best_threshold = float(
                    threshold
                )

                best_precision = float(
                    precision
                )

                best_recall = float(
                    recall
                )

                best_f1 = float(
                    f1
                )

    # Fallback
    if best_threshold is None:

        best_f1 = -1.0

        for threshold in thresholds:

            validation_predictions = (
                validation_probabilities
                >= threshold
            ).astype(int)

            precision = precision_score(
                y_val,
                validation_predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_val,
                validation_predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_val,
                validation_predictions,
                zero_division=0,
            )

            if f1 > best_f1:

                best_threshold = float(
                    threshold
                )

                best_precision = float(
                    precision
                )

                best_recall = float(
                    recall
                )

                best_f1 = float(
                    f1
                )

    print(
        f"Selected threshold : {best_threshold:.3f}"
    )

    # ------------------------------
    # Final untouched test set
    # ------------------------------

    test_probabilities = (
        rf_model.predict_proba(
            hybrid_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    print("\n" + "=" * 70)
    print("FINAL HYBRID MODEL TEST RESULTS")
    print("=" * 70)

    print(
        f"Threshold : {best_threshold:.3f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"AUC-ROC   : {auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Legit",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # 8. SAVE ARTIFACTS
    # ========================================================

    print(
        "\n[8/8] Saving model artifacts..."
    )

    autoencoder.save(
        AUTOENCODER_FILE
    )

    with open(
        SCALER_FILE,
        "wb"
    ) as file:
        pickle.dump(
            scaler,
            file
        )

    with open(
        MODEL_FILE,
        "wb"
    ) as file:
        pickle.dump(
            rf_model,
            file
        )

    with open(
        THRESHOLD_FILE,
        "wb"
    ) as file:
        pickle.dump(
            best_threshold,
            file
        )

    with open(
        FEATURES_FILE,
        "wb"
    ) as file:
        pickle.dump(
            list(X.columns),
            file
        )

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(best_threshold),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "total_dataset_size": int(len(df)),
        "legitimate_count": int(
            (y == 0).sum()
        ),
        "fraud_count": int(
            (y == 1).sum()
        ),
        "model_type": (
            "Autoencoder + Random Forest"
        ),
        "history": history.history,
    }

    with open(
        METRICS_FILE,
        "wb"
    ) as file:
        pickle.dump(
            metrics,
            file
        )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"✓ {AUTOENCODER_FILE}"
    )

    print(
        f"✓ {SCALER_FILE}"
    )

    print(
        f"✓ {MODEL_FILE}"
    )

    print(
        f"✓ {THRESHOLD_FILE}"
    )

    print(
        f"✓ {FEATURES_FILE}"
    )

    print(
        f"✓ {METRICS_FILE}"
    )

    print(
        "\nHybrid training completed successfully."
    )


if __name__ == "__main__":
    main()