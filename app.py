from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")

st.title("Student Mental Health Predictor")
st.caption("Predict depression risk based on student profile inputs.")

artifacts_dir = Path("artifacts")
model_path = artifacts_dir / "depression_model.joblib"
metadata_path = artifacts_dir / "model_metadata.json"

if not model_path.exists() or not metadata_path.exists():
    st.error(
        "Model artifacts were not found. Run: python train_model.py, then restart this app."
    )
    st.stop()

model = joblib.load(model_path)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

features = metadata.get("features", [])
categorical_options = metadata.get("categorical_options", {})

st.subheader("Enter Student Information")

inputs: dict[str, object] = {}
for feature in features:
    if feature in categorical_options:
        options = categorical_options[feature]
        inputs[feature] = st.selectbox(feature.replace("_", " "), options=options)
    else:
        label = feature.replace("_", " ")
        if feature.lower() in {"age", "day", "month", "year", "hour"}:
            inputs[feature] = st.number_input(label, value=21.0)
        else:
            inputs[feature] = st.text_input(label, value="")

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    proba_text = "Probability unavailable"
    if hasattr(model, "predict_proba"):
        classes = [str(c) for c in model.classes_]
        probs = model.predict_proba(input_df)[0]
        if "yes" in classes:
            yes_prob = float(probs[classes.index("yes")])
            proba_text = f"Estimated depression probability: {yes_prob:.2%}"

    st.success(f"Predicted class: {prediction}")
    st.info(proba_text)

st.markdown("---")
st.caption("This tool is for educational use and not a clinical diagnosis.")
