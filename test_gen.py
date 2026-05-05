import requests
import time

try:
    print("Sending generation request...")
    start = time.time()
    response = requests.post("http://127.0.0.1:8888/api/generate", json={"seed": 5})
    end = time.time()
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if "error" in data:
            print(f"Error from API: {data['error']}")
        else:
            print(f"Generation took {end - start:.2f} seconds")
            print(f"Formula: {data['formula']}")
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
