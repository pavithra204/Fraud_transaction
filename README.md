# 🛡️ ShieldAI — Credit Card Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraudtransaction-fdzxzzvs4fvzhmranfxoc6.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.48.1-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end credit card fraud detection system combining **Autoencoder feature learning** with **Random Forest classification** and an optimized fraud-probability threshold.

## 🌐 Live Demo

**Streamlit App:**  
https://fraudtransaction-fdzxzzvs4fvzhmranfxoc6.streamlit.app/

---

## 🔍 Overview

ShieldAI is a machine-learning based credit card fraud detection application built with **TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Plotly, and Streamlit**.

The system uses a hybrid architecture:

```text
Transaction Features
        ↓
StandardScaler
        ↓
Autoencoder
        ↓
Latent Features
        +
Original Standardized Features
        ↓
Hybrid Feature Representation
        ↓
Random Forest Classifier
        ↓
Fraud Probability
        ↓
Optimized Decision Threshold
        ↓
LEGITIMATE / FRAUD
