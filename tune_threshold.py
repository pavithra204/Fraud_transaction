import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DATA_FILE = "creditcard.csv"

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
AUTOENCODER_FILE = "autoencoder.keras"

THRESHOLD_FILE = "threshold.pkl"
FEATURES_FILE = "features.pkl"
METRICS_FILE = "metrics.pkl"


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 65)
print("SHIELDAI - THRESHOLD OPTIMIZATION")
print("=" * 65)

df = pd.read_csv(DATA_FILE)

X = df.drop(
    columns=["Class"]
)

y = df["Class"].astype(int).values


with open(
    MODEL_FILE,
    "rb"
) as file:
    model = pickle.load(file)


with open(
    SCALER_FILE,
    "rb"
) as file:
    scaler = pickle.load(file)


with open(
    FEATURES_FILE,
    "rb"
) as file:
    feature_names = pickle.load(file)


import tensorflow as tf

autoencoder = tf.keras.models.load_model(
    AUTOENCODER_FILE
)

encoder = tf.keras.Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer("latent").output,
)


# ============================================================
# SAME SPLIT AS TRAINING
# ============================================================

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


# ============================================================
# PREPARE VALIDATION DATA
# ============================================================

X_val = X_val[
    feature_names
]

X_val_scaled = scaler.transform(
    X_val.values
)


latent_val = encoder.predict(
    X_val_scaled,
    batch_size=1024,
    verbose=0,
)


hybrid_val = np.concatenate(
    [
        X_val_scaled,
        latent_val,
    ],
    axis=1,
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

val_probabilities = model.predict_proba(
    hybrid_val
)[:, 1]


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

best_threshold = None
best_precision = -1.0
best_recall = 0.0
best_f1 = 0.0


# Precision-first:
# Recall must remain at least 80%.

for threshold in np.arange(
    0.05,
    0.951,
    0.001,
):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    if recall >= 0.80:

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


# ============================================================
# FALLBACK
# ============================================================

if best_threshold is None:

    print(
        "No threshold achieved recall >= 80%."
    )

    print(
        "Selecting best F1 threshold."
    )

    best_f1 = -1.0

    for threshold in np.arange(
        0.05,
        0.951,
        0.001,
    ):

        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
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


# ============================================================
# TEST SET WITH NEW THRESHOLD
# ============================================================

X_test = X_test[
    feature_names
]

X_test_scaled = scaler.transform(
    X_test.values
)


latent_test = encoder.predict(
    X_test_scaled,
    batch_size=1024,
    verbose=0,
)


hybrid_test = np.concatenate(
    [
        X_test_scaled,
        latent_test,
    ],
    axis=1,
)


test_probabilities = model.predict_proba(
    hybrid_test
)[:, 1]


test_predictions = (
    test_probabilities >= best_threshold
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


# ============================================================
# PRINT FINAL
# ============================================================

print("\n" + "=" * 65)
print("FINAL TUNED TEST RESULTS")
print("=" * 65)

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


# ============================================================
# SAVE NEW THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "wb"
) as file:

    pickle.dump(
        best_threshold,
        file
    )


# Update metrics.pkl
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
    "model_type": "Autoencoder + Random Forest",
}


with open(
    METRICS_FILE,
    "wb"
) as file:

    pickle.dump(
        metrics,
        file
    )


print("\nSaved:")
print("✓ threshold.pkl")
print("✓ metrics.pkl")
print(
    "\nNo model retraining was performed."
)