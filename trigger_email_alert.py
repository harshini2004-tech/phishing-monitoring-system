import requests
import time

print("🚨 Triggering alert for email test...")

# Send enough invalid requests to trigger the error rate alert
for i in range(25):
    try:
        response = requests.post("http://localhost:8001/predict", 
                               json={"wrong_data": "cause_error"})
        print(f"Request {i+1}: Status {response.status_code}")
    except Exception as e:
        print(f"Request {i+1}: Error {e}")
    time.sleep(1)

print("📧 Check your Gmail inbox for alert emails!")

