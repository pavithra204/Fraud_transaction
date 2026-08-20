# 🚀 Deployment Guide - ShieldAI Fraud Detection

## **Option 1: Deploy to Streamlit Cloud (Recommended)**

### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add ShieldAI fraud detection app"
   git push origin main
   ```

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
   - Click "New app"
   - Connect your GitHub repository
   - Select branch: `main`
   - Set main file: `app.py`
   - Click "Deploy"

### Configuration for Streamlit Cloud:
- ✅ `requirements.txt` — Already configured
- ✅ `runtime.txt` — Python 3.11 specified
- ✅ `.streamlit/config.toml` — Settings configured

---

## **Option 2: Deploy to Heroku**

### Steps:
1. **Create `Procfile` in project root:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Heroku CLI:**
   ```bash
   heroku login
   heroku create your-shieldai-app
   git push heroku main
   ```

---

## **Option 3: Local Network Deployment**

### For other users to access on your network:

1. **Run app with network binding:**
   ```bash
   streamlit run app.py --server.address=0.0.0.0
   ```

2. **Find your IP address:**
   - Windows: `ipconfig` → IPv4 Address
   - Mac/Linux: `ifconfig` → inet

3. **Share link:** `http://YOUR_IP_ADDRESS:8501`

---

## **Verification Checklist:**

✅ No Python syntax errors
✅ All dependencies in `requirements.txt`
✅ Data files included (sample_creditcard.csv)
✅ Robust fallback to synthetic data
✅ Network accessible (0.0.0.0 binding)
✅ Configuration files present
✅ CSS/styling embedded (no external files)
✅ No hardcoded absolute paths
✅ .gitignore configured

---

## **File Structure for Deployment:**
```
credit_card_fraud_detect/
├── app.py                      ✅
├── requirements.txt            ✅
├── runtime.txt                 ✅
├── .gitignore                  ✅
├── README.md                   ✅
├── sample_creditcard.csv       ✅
├── .streamlit/
│   └── config.toml            ✅
└── Procfile                    (For Heroku only)
```

---

## **Troubleshooting:**

| Issue | Solution |
|-------|----------|
| **CSV not found** | App generates synthetic data automatically |
| **Port already in use** | `streamlit run app.py --server.port=8502` |
| **Network access denied** | Ensure firewall allows port 8501 |
| **TensorFlow issues** | TensorFlow automatically loads best available (CPU/GPU) |

---

## **Current Access:**
- 🔗 **Local:** `http://localhost:8501`
- 🔗 **Network:** `http://192.168.29.7:8501`
- 🔗 **External:** `http://49.37.134.198:8501` (if port forwarded)

**App is production-ready! Ready to push to GitHub or deploy.** 🚀
