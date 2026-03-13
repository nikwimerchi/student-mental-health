# Deployment Guide

## 1. Prepare project

1. Put your dataset at one of these paths:
   - data/Student Mental health.csv
   - Student Mental health.csv

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train and export model artifacts:

```powershell
python train_model.py
```

This creates:
- artifacts/depression_model.joblib
- artifacts/model_metadata.json
- artifacts/metrics_summary.json

## 2. Run locally

```powershell
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from your repo.
3. Set main file path to app.py.
4. Ensure your dataset file is included in the repo under data/.
5. runtime.txt already pins Python to 3.11.
6. .streamlit/config.toml already defines theme and headless server settings.
7. Deploy.

If artifacts are not committed, the app attempts first-run training automatically.

## 4. Render deployment

This repository includes render.yaml. On Render:

1. Create a new Blueprint service from your GitHub repo.
2. Render will read render.yaml automatically.
3. Ensure dataset is in the repo under data/Student Mental health.csv.
4. Deploy.

If you prefer manual service setup, use:
- Build command: pip install -r requirements.txt
- Start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

## Note

Predictions are for educational support only and are not a medical diagnosis.
