from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_CANDIDATES = [
    Path("data/Student Mental health.csv"),
    Path("Student Mental health.csv"),
    Path("/kaggle/input/student-mental-health/Student Mental health.csv"),
]


def load_dataset() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            print(f"Loaded dataset from: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Dataset not found. Put Student Mental health.csv in data/ or project root."
    )


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    df_model = df.rename(
        columns={
            "Choose your gender": "Choose_your_gender",
            "What is your course?": "What_is_your_course",
            "Your current year of Study": "Your_current_year_of_Study",
            "What is your CGPA?": "What_is_your_CGPA",
            "Marital status": "Marital_status",
            "Do you have Depression?": "Do_you_have_Depression",
            "Do you have Anxiety?": "Do_you_have_Anxiety",
            "Do you have Panic attack?": "Do_you_have_Panic_attack",
            "Did you seek any specialist for a treatment?": "Did_you_seek_any_specialist_for_a_treatment",
        }
    )

    target_col = "Do_you_have_Depression"
    if target_col not in df_model.columns:
        raise KeyError(f"Missing target column: {target_col}")

    drop_cols = [c for c in ["Timestamp", "Date", "day", "month", "year", "hour"] if c in df_model.columns]
    feature_df = df_model.drop(columns=drop_cols)

    feature_cols = [c for c in feature_df.columns if c != target_col]
    return feature_df, target_col, feature_cols


def train_and_save() -> None:
    raw_df = load_dataset()
    feature_df, target_col, feature_cols = build_training_frame(raw_df)

    X = feature_df[feature_cols].copy()
    y = feature_df[target_col].astype(str).str.strip().str.lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "depression_model.joblib"
    metadata_path = artifacts_dir / "model_metadata.json"
    metrics_path = artifacts_dir / "metrics_summary.json"

    joblib.dump(model, model_path)

    categorical_options: dict[str, list[str]] = {}
    for col in feature_cols:
        if X[col].dtype == "object":
            categorical_options[col] = sorted(X[col].dropna().astype(str).unique().tolist())

    metadata = {
        "target": target_col,
        "features": feature_cols,
        "classes": sorted(set(y.astype(str))),
        "categorical_options": categorical_options,
        "accuracy": round(float(accuracy), 4),
    }

    class_labels = sorted([str(c) for c in set(y.astype(str))])
    cm = confusion_matrix(y_test, preds, labels=class_labels)
    metrics_summary = {
        "accuracy": round(float(accuracy), 4),
        "test_samples": int(len(y_test)),
        "classification_report": report_dict,
        "class_labels": class_labels,
        "confusion_matrix": cm.tolist(),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print(f"Model saved to: {model_path.resolve()}")
    print(f"Metadata saved to: {metadata_path.resolve()}")
    print(f"Metrics saved to: {metrics_path.resolve()}")


if __name__ == "__main__":
    train_and_save()
