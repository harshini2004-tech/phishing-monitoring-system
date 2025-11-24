#!/usr/bin/env python3
import requests
import time
import random
import threading

API_URL = "http://localhost:8001/predict"

test_cases = [
    # Phishing emails
    {"content": "URGENT: Your account will be suspended! Verify now: http://secure-login-verify.com", "content_type": "email"},
    {"content": "Security Alert: Unusual login detected. Confirm identity: http://bank-security.net", "content_type": "email"},
    {"content": "Your PayPal account needs verification: http://paypal-secure-login.com", "content_type": "email"},
    
    # Legitimate emails
    {"content": "Meeting reminder: Team sync at 3 PM tomorrow", "content_type": "email"},
    {"content": "Your monthly newsletter is available", "content_type": "email"},
    {"content": "Password changed successfully", "content_type": "email"},
    
    # Phishing messages
    {"content": "You won $1000! Claim prize: http://bit.ly/winprize", "content_type": "message"},
    {"content": "Free iPhone! Click: http://tinyurl.com/freegift", "content_type": "message"},
    {"content": "Bank alert: Card locked. Call: 555-0123", "content_type": "message"},
    
    # Legitimate messages
    {"content": "Your Uber is arriving", "content_type": "message"},
    {"content": "Package delivered", "content_type": "message"},
    {"content": "Your code is: 123456", "content_type": "message"},
    
    # Phishing URLs
    {"content": "http://secure-login-verify-account.com", "content_type": "url"},
    {"content": "https://facebook-security-confirm.net", "content_type": "url"},
    {"content": "http://microsoft-account-verify.com", "content_type": "url"},
    
    # Legitimate URLs
    {"content": "https://google.com", "content_type": "url"},
    {"content": "https://github.com", "content_type": "url"},
    {"content": "https://stackoverflow.com", "content_type": "url"},
]

def send_request():
    while True:
        test_case = random.choice(test_cases)
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=test_case, timeout=5)
            response_time = time.time() - start_time
            
            result = response.json()
            status = "✅" if response.status_code == 200 else "❌"
            
            print(f"{status} {test_case['content_type']:8} | Phishing: {result['is_phishing']:5} | "
                  f"Conf: {result['probability']:.3f} | Time: {response_time:.3f}s")
                  
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(random.uniform(1, 3))  # Random delay between 1-3 seconds

print("🚀 Starting traffic generator...")
print("Press Ctrl+C to stop\n")

# Start multiple threads for more traffic
threads = []
for i in range(3):  # 3 concurrent threads
    thread = threading.Thread(target=send_request, daemon=True)
    thread.start()
    threads.append(thread)

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping traffic generator...")
