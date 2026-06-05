import requests
import json

url = "http://127.0.0.1:5000/predict"

payload = {
    "fever": 1,
    "cough": 1,
    "fatigue": 1,
    "breathing": 0,
    "age": 35,
    "gender": 1,
    "bloodPressure": 1,
    "cholesterol": 1,
    "model": "rf"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=headers,
        timeout=30
    )

    print("=" * 50)
    print("Status Code:", response.status_code)
    print("=" * 50)

    try:
        result = response.json()
        print(json.dumps(result, indent=4))
    except:
        print("Raw Response:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("ERROR: Flask server is not running.")
    print("Start the server first:")
    print("python app.py")

except Exception as e:
    print("ERROR:", str(e))