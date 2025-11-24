import requests
import time

print("Triggering error alert test...")
for i in range(15):
    try:
        # Send invalid request to create errors
        response = requests.post("http://localhost:8001/predict", 
                               json={"wrong_field": "data"})
        print(f"Error {i+1}: Status {response.status_code}")
    except Exception as e:
        print(f"Error {i+1}: {e}")
    time.sleep(1)
print("Check Grafana alerts in 2-3 minutes!")

