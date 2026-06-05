from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

DEFAULT_TREATMENT = {
    "medicines": ["⚕️ Consult a licensed healthcare provider for diagnosis-specific treatment."],
    "advice": ["📞 Seek medical care if symptoms worsen or emergency symptoms appear."],
    "dosage_instructions": ["Follow a clinician's prescription and local medical guidance."],
    "foods_to_eat": [],
    "foods_to_avoid": [],
    "recovery_time": "Varies by condition and patient profile.",
    "when_to_see_doctor": "Immediately for severe symptoms or uncertainty."
}


def load_medicine_database(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def get_treatment_plan(disease: str, medicine_db: dict[str, Any]) -> dict[str, Any]:
    return medicine_db.get(disease, DEFAULT_TREATMENT)


def calculate_risk_level(features) -> str:
    row = features.iloc[0]
    symptom_count = sum(row[c] == "Yes" for c in ["fever", "cough", "fatigue", "difficulty_breathing"])
    if symptom_count >= 3 and (row["age"] >= 60 or row["blood_pressure"] >= 2):
        return "high"
    if symptom_count >= 2:
        return "medium"
    return "low"
