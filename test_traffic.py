#!/usr/bin/env python3
import requests
import time

API_URL = "http://localhost:8000/predict"

test_cases = [
    {"content": "URGENT: Verify your account now!", "content_type": "email"},
    {"content": "Your package was delivered", "content_type": "message"},
    {"content": "https://google.com", "content_type": "url"},
]

print("Sending test requests...")
for i, test_case in enumerate(test_cases, 1):
    try:
        response = requests.post(API_URL, json=test_case, timeout=5)
        result = response.json()
        status = "✅" if response.status_code == 200 else "❌"
        print(f"{status} Request {i}: {result}")
    except Exception as e:
        print(f"❌ Request {i} failed: {e}")
    time.sleep(1)

print("Test completed!")
