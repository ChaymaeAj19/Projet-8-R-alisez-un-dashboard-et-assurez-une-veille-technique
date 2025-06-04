import requests

API_URL = "https://projet-7-implementation.onrender.com/predict"

def get_prediction(client_id):
    try:
        response = requests.get(API_URL, params={"client_id": client_id})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
