from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


def get_risk_band(probability: float) -> str:
    if probability >= 0.8:
        return "Very High"
    if probability >= 0.6:
        return "High"
    if probability >= 0.35:
        return "Moderate"
    return "Low"


def get_support_actions(risk_band: str) -> list[str]:
    if risk_band == "Very High":
        return [
            "Arrange same-day check-in with a counselor or student support team.",
            "Encourage immediate connection with a trusted person (friend, family, mentor).",
            "Share urgent help options and local crisis contacts if safety concerns exist.",
            "Create a short 48-hour follow-up plan and monitor wellbeing closely.",
        ]
    if risk_band == "High":
        return [
            "Recommend scheduling a professional counseling appointment this week.",
            "Set a weekly wellbeing check-in with academic advisor or peer mentor.",
            "Reduce avoidable stressors: sleep schedule, workload planning, and breaks.",
            "Provide information on campus mental health resources and support groups.",
        ]
    if risk_band == "Moderate":
        return [
            "Encourage early preventive support, such as workshops or counseling intake.",
            "Track mood and stress patterns for 2-3 weeks.",
            "Promote healthy routines: sleep, activity, social connection, and hydration.",
            "Re-screen after a short interval if stress increases.",
        ]
    return [
        "Maintain healthy routines and social support.",
        "Use self-care tools for stress management during exams.",
        "Re-check periodically, especially after major life or study changes.",
    ]


def get_risk_css_class(risk_band: str) -> str:
    if risk_band == "Very High":
        return "risk-very-high"
    if risk_band == "High":
        return "risk-high"
    if risk_band == "Moderate":
        return "risk-moderate"
    return "risk-low"


def get_risk_text_class(risk_band: str) -> str:
    if risk_band == "Very High":
        return "risk-text-very-high"
    if risk_band == "High":
        return "risk-text-high"
    if risk_band == "Moderate":
        return "risk-text-moderate"
    return "risk-text-low"

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7fafc 0%, #eef7f1 100%);
    }
    .app-card {
        border: 1px solid #d9e4dd;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(30, 60, 45, 0.06);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0e3b2e;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        color: #315948;
        margin-top: 0;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.5rem;
    }
    .compact-stat {
        border: 1px solid #d9e4dd;
        border-radius: 12px;
        padding: 0.7rem;
        background: #ffffff;
    }
    .compact-stat .label {
        font-size: 0.8rem;
        color: #547566;
        margin-bottom: 0.1rem;
    }
    .compact-stat .value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #14382b;
    }
    .risk-panel {
        border-radius: 12px;
        padding: 0.85rem;
        margin-top: 0.75rem;
    }
    .risk-low {
        background: #ecfdf3;
        border: 1px solid #8fd1a5;
    }
    .risk-moderate {
        background: #fff8e8;
        border: 1px solid #e9c46a;
    }
    .risk-high {
        background: #fff1e7;
        border: 1px solid #f4a261;
    }
    .risk-very-high {
        background: #ffebeb;
        border: 1px solid #e76f51;
    }
    .risk-text-low {
        color: #1f7a53;
    }
    .risk-text-moderate {
        color: #9a6700;
    }
    .risk-text-high {
        color: #b45309;
    }
    .risk-text-very-high {
        color: #b42318;
    }
    .support-card {
        margin-top: 0.75rem;
        border: 1px solid #d9e4dd;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        background: #ffffff;
    }
    @media (max-width: 720px) {
        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='hero-title'>Student Mental Health Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-sub'>Predict depression likelihood from a student profile and review validation metrics.</p>",
    unsafe_allow_html=True,
)

artifacts_dir = Path("artifacts")
model_path = artifacts_dir / "depression_model.joblib"
metadata_path = artifacts_dir / "model_metadata.json"
metrics_path = artifacts_dir / "metrics_summary.json"

if not model_path.exists() or not metadata_path.exists():
    st.warning("Model artifacts not found. Attempting first-run training.")
    try:
        from train_model import train_and_save

        train_and_save()
    except Exception as ex:
        st.error(
            "Model artifacts were not found and automatic training failed. "
            "Run python train_model.py locally and commit artifacts folder.\n\n"
            f"Error: {ex}"
        )
        st.stop()

model = joblib.load(model_path)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metrics_summary = {}
if metrics_path.exists():
    metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))

features = metadata.get("features", [])
categorical_options = metadata.get("categorical_options", {})
saved_accuracy = metadata.get("accuracy")

tab_predict, tab_performance = st.tabs(["Predict", "Model Performance"])

with tab_predict:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("Student Profile")

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

    run_prediction = st.button("Predict", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_prediction:
        input_df = pd.DataFrame([inputs])
        prediction = str(model.predict(input_df)[0]).strip().lower()

        yes_prob = None
        if hasattr(model, "predict_proba"):
            classes = [str(c).strip().lower() for c in model.classes_]
            probs = model.predict_proba(input_df)[0]
            if "yes" in classes:
                yes_prob = float(probs[classes.index("yes")])

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Class", prediction)
        with col_b:
            if yes_prob is not None:
                st.metric("Depression Probability", f"{yes_prob:.2%}")
            else:
                st.metric("Depression Probability", "N/A")

        interpreted_prob = yes_prob
        if interpreted_prob is None:
            interpreted_prob = 0.75 if prediction == "yes" else 0.25

        risk_band = get_risk_band(interpreted_prob)
        risk_css_class = get_risk_css_class(risk_band)
        risk_text_class = get_risk_text_class(risk_band)
        actions = get_support_actions(risk_band)

        st.markdown(
            f"""
            <div class='result-grid'>
                <div class='compact-stat'>
                    <div class='label'>Predicted Class</div>
                    <div class='value'>{prediction}</div>
                </div>
                <div class='compact-stat'>
                    <div class='label'>Depression Probability</div>
                    <div class='value'>{interpreted_prob:.2%}</div>
                </div>
                <div class='compact-stat'>
                    <div class='label'>Risk Band</div>
                    <div class='value {risk_text_class}'>{risk_band}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class='risk-panel {risk_css_class}'>
                <strong>Risk Interpretation:</strong> <span class='{risk_text_class}'>{risk_band}</span> risk profile based on current model output.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Recommended Support Actions")
        actions_html = "".join([f"<li>{action}</li>" for action in actions])
        st.markdown(
            f"""
            <div class='support-card'>
                <ol>
                    {actions_html}
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediction == "yes":
            st.warning("Model indicates elevated depression risk. Consider supportive follow-up.")
        else:
            st.success("Model indicates lower depression risk for this profile.")

        st.info(
            "If there is immediate concern about self-harm or safety, contact local emergency or crisis services right away."
        )

with tab_performance:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("Validation Metrics")

    accuracy_to_show = metrics_summary.get("accuracy", saved_accuracy)
    if accuracy_to_show is not None:
        st.metric("Test Accuracy", f"{float(accuracy_to_show):.2%}")

    class_report = metrics_summary.get("classification_report", {})
    cm = metrics_summary.get("confusion_matrix", [])
    labels = metrics_summary.get("class_labels", [])

    report_rows = []
    for cls_name, values in class_report.items():
        if isinstance(values, dict) and {"precision", "recall", "f1-score"}.issubset(values.keys()):
            report_rows.append(
                {
                    "class": cls_name,
                    "precision": values.get("precision", 0.0),
                    "recall": values.get("recall", 0.0),
                    "f1_score": values.get("f1-score", 0.0),
                }
            )

    if report_rows:
        report_df = pd.DataFrame(report_rows)
        st.write("Per-class scores")
        st.dataframe(report_df, use_container_width=True)
        st.bar_chart(report_df.set_index("class")[["precision", "recall", "f1_score"]])
    else:
        st.info("No detailed metrics found yet. Re-run training with train_model.py to generate charts.")

    if cm and labels:
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        st.write("Confusion matrix")
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("This tool is for educational use and not a clinical diagnosis.")
