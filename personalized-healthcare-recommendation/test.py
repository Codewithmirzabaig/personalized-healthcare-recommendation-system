import requests

data = {
    "fever": 1,
    "cough": 1,
    "fatigue": 1,
    "breathing": 0,
    "age": 35,
    "gender": 1,
    "bloodPressure": 1,
    "cholesterol": 1
}

r = requests.post(
    "http://127.0.0.1:5000/predict",
    json=data
)

print(r.status_code)
print(r.json())