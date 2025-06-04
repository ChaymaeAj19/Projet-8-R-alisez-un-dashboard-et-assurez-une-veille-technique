import requests

API_URL = "https://projet-7-implementation.onrender.com/predict"

def get_prediction(client_id):
    try:
        payload = {"SK_ID_CURR": int(client_id)}
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
