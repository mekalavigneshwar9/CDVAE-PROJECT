import requests

try:
    response = requests.get("http://127.0.0.1:8888/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"JSON Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
