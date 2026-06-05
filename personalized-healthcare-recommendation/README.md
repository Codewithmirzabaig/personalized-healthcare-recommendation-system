# Personalized Healthcare & Medicine Recommendation System

A Data Science and Machine Learning project that predicts a likely disease from patient symptoms and profile attributes, then recommends medicines, dosage guidance, lifestyle advice, food suggestions, risk level, and top disease probabilities through a Flask web application.

> Educational portfolio project only. This system is not a substitute for licensed medical advice, diagnosis, or treatment.

## Business Problem
Patients often need quick symptom-based guidance before consulting a clinician. This project demonstrates how machine learning can support healthcare triage by combining disease prediction, confidence scoring, and treatment recommendation logic in one API-driven application.

## Key Features
- Disease prediction from symptoms and patient attributes
- Personalized medicine recommendation database
- Risk level classification: low, medium, high
- Top 5 disease probability output
- Health advice, dosage instructions, foods to eat/avoid, recovery time
- Flask API backend with CORS support
- Web frontend using HTML templates
- Retraining fallback if legacy pickle model is incompatible
- Pytest-based API testing

## ML Inputs
- Fever
- Cough
- Fatigue
- Difficulty breathing
- Age
- Gender
- Blood pressure
- Cholesterol level
- Outcome variable
- Risk level

## Project Structure
```text
personalized-healthcare-recommendation/
├── app/
│   ├── app.py
│   └── templates/
├── data/
│   └── raw/Cleaned_Dataset.csv
├── models/
│   ├── best_model.pkl
│   ├── best_model_legacy.pkl
│   ├── disease_encoder.pkl
│   └── medicine_database.pkl
├── src/
│   ├── model_utils.py
│   ├── medicine_recommender.py
│   └── train_model.py
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
python app/app.py
```

Open:
```text
http://localhost:5000
```

## API Test
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"fever":1,"cough":1,"fatigue":1,"breathing":0,"age":35,"gender":1,"bloodPressure":1,"cholesterol":1}'
```

## Resume Bullet
Built a Personalized Healthcare & Medicine Recommendation System using Flask, Scikit-learn, NLP-style recommendation logic, and ML classification to predict diseases from patient symptoms and generate confidence scores, risk levels, top disease probabilities, medication guidance, and lifestyle recommendations through an API-driven web application.
