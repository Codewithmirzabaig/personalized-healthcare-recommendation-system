from __future__ import annotations

from pathlib import Path
import pickle
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "fever", "cough", "fatigue", "difficulty_breathing", "age", "gender",
    "blood_pressure", "cholesterol_level", "outcome_variable", "risk_level"
]

NUMERIC_COLUMNS = ["age", "blood_pressure", "cholesterol_level"]
CATEGORICAL_COLUMNS = [
    "fever", "cough", "fatigue", "difficulty_breathing", "gender", "outcome_variable", "risk_level"
]


def normalize_input(payload: dict) -> pd.DataFrame:
    """Convert frontend JSON into the exact schema expected by the disease model."""
    fever = "Yes" if int(payload.get("fever", 0)) == 1 else "No"
    cough = "Yes" if int(payload.get("cough", 0)) == 1 else "No"
    fatigue = "Yes" if int(payload.get("fatigue", 0)) == 1 else "No"
    breathing = "Yes" if int(payload.get("breathing", payload.get("difficulty_breathing", 0))) == 1 else "No"
    gender = "Male" if int(payload.get("gender", 0)) == 1 else "Female"
    bp = int(payload.get("bloodPressure", payload.get("blood_pressure", 1)))
    chol = int(payload.get("cholesterol", payload.get("cholesterol_level", 1)))
    age = int(payload.get("age", 30))
    symptom_count = sum([fever == "Yes", cough == "Yes", fatigue == "Yes", breathing == "Yes"])
    risk_level = "High" if symptom_count >= 3 and (age >= 60 or bp >= 2) else "Medium" if symptom_count >= 2 else "Low"

    return pd.DataFrame([{
        "fever": fever,
        "cough": cough,
        "fatigue": fatigue,
        "difficulty_breathing": breathing,
        "age": age,
        "gender": gender,
        "blood_pressure": bp,
        "cholesterol_level": chol,
        "outcome_variable": payload.get("outcome_variable", "Negative"),
        "risk_level": risk_level,
    }])


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
    ])


def train_and_save(data_path: Path, model_path: Path, encoder_path: Path) -> dict:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["disease"])
    X = df[FEATURE_COLUMNS].copy()
    y_raw = df["disease"].copy()

    # Keep common/known diseases and group rare labels into Other for a practical demo app.
    common = y_raw.value_counts()[y_raw.value_counts() >= 8].index
    y_grouped = y_raw.where(y_raw.isin(common), "Other")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_grouped)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "classes": list(encoder.classes_),
        "classification_report": classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0),
    }


def load_or_train_model(project_root: Path):
    model_path = project_root / "models" / "best_model.pkl"
    legacy_model_path = project_root / "models" / "best_model_legacy.pkl"
    encoder_path = project_root / "models" / "disease_encoder.pkl"
    data_path = project_root / "data" / "raw" / "Cleaned_Dataset.csv"

    try:
        model = joblib.load(model_path)
    except Exception:
        try:
            model = joblib.load(legacy_model_path)
        except Exception:
            train_and_save(data_path, model_path, encoder_path)
            model = joblib.load(model_path)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder
