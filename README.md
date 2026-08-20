# 🛡️ Credit Card Fraud Detection Using Autoencoders

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🔍 Overview

An end-to-end credit card fraud detection system using an **Autoencoder neural network** 
for unsupervised anomaly detection. The model is trained exclusively on legitimate 
transactions and detects fraud as deviation from learned normal behavior.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app) ← replace after deployment

---

## 🧠 How It Works

```
Normal transactions → Train Autoencoder → Learn normal patterns
                                               ↓
New transaction → Encode → Bottleneck → Decode → Reconstruction
                                               ↓
              Compute MSE Error between Input and Reconstruction
| ℹ️ About | Project purpose, usage guidance, and dataset info |
              Error > Threshold (95th pct) → FRAUD 🔴
```

| Item | Value |

| Training approach | Autoencoder anomaly detection |
| Training data | Legitimate transactions |
| Decision rule | Reconstruction error above the 95th percentile |
| Output | Legitimate or suspicious, with an anomaly score |
| Important limitation | Results support review; they are not a final banking decision |
| 📊 Model Performance | Confusion matrix, ROC-AUC, F1, error distribution |
| ℹ️ About | Architecture, parameters, dataset info |

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
├── app.py                  ← Main Streamlit application
├── create_sample.py        ← Script to create sample from full Kaggle CSV
The repository includes `runtime.txt` for Python 3.11 and keeps the bundled sample
dataset relative to `app.py`, so the app does not depend on the cloud process's
working directory. Do not commit real customer statements, card numbers, or API keys.
├── sample_creditcard.csv   ← 5000-row sample (push this to GitHub)
├── requirements.txt        ← Dependencies
└── README.md
```

---

## ⚙️ Model Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| epochs | 20 | Training iterations |
| batch_size | 16 | Samples per weight update |
| optimizer | Adam | Adaptive learning rate |
| loss | MSE | Reconstruction quality measure |
| bottleneck | 8 neurons | Compressed representation |
| threshold | 95th percentile | Fraud/Legit decision boundary |
| activation | ReLU / Linear | Hidden / Output layers |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add real Kaggle dataset
#    Download creditcard.csv from Kaggle and run:
python create_sample.py

# 4. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

```
1. Push this repo to GitHub (public)
2. Go to share.streamlit.io
3. Sign in with GitHub
4. Click "New app"
5. Select repo → branch: main → file: app.py
6. Click Deploy → get your public URL ✅
```

---

## 📁 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total transactions:** 284,807
- **Fraud cases:** 492 (0.172%)
- **Features:** V1–V28 (PCA) + Time + Amount + Class

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras Autoencoder
- **App:** Streamlit
- **Charts:** Plotly
- **ML utilities:** Scikit-learn
- **Data:** Pandas + NumPy

---

## 📄 License

MIT License — free to use for academic and personal projects.
