from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from model_utils import load_or_train_model, normalize_input
from medicine_recommender import calculate_risk_level, get_treatment_plan, load_medicine_database

app = Flask(__name__)
CORS(app)

model, label_encoder = load_or_train_model(PROJECT_ROOT)
medicine_db = load_medicine_database(PROJECT_ROOT / "models" / "medicine_database.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "medicine_database_loaded": bool(medicine_db),
    })


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "current_model": type(model).__name__,
        "diseases_count": len(label_encoder.classes_),
        "disease_classes": list(label_encoder.classes_),
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    payload = request.get_json(silent=True) or {}
    if not payload:
        return jsonify({"success": False, "error": "No JSON data provided"}), 400

    try:
        features = normalize_input(payload)
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        disease = str(label_encoder.classes_[prediction])
        confidence = round(float(probabilities[prediction] * 100), 2)

        top_indices = np.argsort(probabilities)[-5:][::-1]
        top5 = [
            {"disease": str(label_encoder.classes_[i]), "confidence": round(float(probabilities[i] * 100), 2)}
            for i in top_indices
        ]

        treatment = get_treatment_plan(disease, medicine_db)
        risk = calculate_risk_level(features)

        return jsonify({
            "success": True,
            "disease": disease,
            "confidence": confidence,
            "risk": risk,
            "top5": top5,
            "medicines": treatment.get("medicines", []),
            "dosage_instructions": treatment.get("dosage_instructions", []),
            "advice": treatment.get("advice", []),
            "foods_to_eat": treatment.get("foods_to_eat", []),
            "foods_to_avoid": treatment.get("foods_to_avoid", []),
            "recovery_time": treatment.get("recovery_time", "Not available"),
            "when_to_see_doctor": treatment.get("when_to_see_doctor", "Consult a doctor if symptoms persist."),
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": "This is an educational ML project, not medical advice. Consult a licensed clinician before taking medication."
        }), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
