#!/usr/bin/env python3
import requests
import time
import sys

def trigger_error_alert():
    """Trigger the High Error Rate alert"""
    print("🚨 Triggering High Error Rate Alert...")
    print("This should send an email to your Gmail in 2-3 minutes")
    
    # Send requests that will cause 422 errors (validation errors)
    error_count = 0
    for i in range(25):
        try:
            # Send malformed request to cause validation error
            response = requests.post("http://localhost:8001/predict", 
                                   json={
                                       "wrong_field": "invalid_data",
                                       "another_wrong": "more_invalid"
                                   })
            
            if response.status_code != 200:
                error_count += 1
                print(f"❌ Error {error_count}/25: Status {response.status_code}")
            else:
                print(f"✅ Request {i+1}: Success")
                
        except Exception as e:
            error_count += 1
            print(f"💥 Error {error_count}/25: {e}")
        
        time.sleep(2)  # 2 seconds between requests
    
    print(f"\n📊 Sent {error_count} error requests out of 25")
    print("⏳ Waiting for alert to trigger...")
    print("📧 Check your Gmail: harshininmurthy@gmail.com")

def trigger_phishing_alert():
    """Trigger the High Phishing Rate alert"""
    print("\n🎣 Triggering High Phishing Rate Alert...")
    
    phishing_content = [
        "URGENT: Your account will be suspended! Verify now!",
        "SECURITY ALERT: Unusual login detected! Confirm identity!",
        "Bank account verification required immediately!",
        "You won $5000 prize! Claim now!",
        "Free iPhone 15! Click to claim your gift!",
        "PayPal security update required! Verify account!",
        "Tax refund available! Claim your money now!",
        "Cryptocurrency investment opportunity! 500% returns!",
    ]
    
    for i in range(30):
        content = phishing_content[i % len(phishing_content)]
        try:
            response = requests.post("http://localhost:8001/predict", 
                                   json={
                                       "content": content,
                                       "content_type": "email"
                                   })
            result = response.json()
            print(f"🔴 Phishing {i+1}/30: {result['is_phishing']} (Conf: {result['probability']:.3f})")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(3)  # 3 seconds between requests
    
    print("⏳ Phishing alert may trigger in 5 minutes...")

if __name__ == "__main__":
    print("🔔 REAL ALERT DEMO STARTING")
    print("=" * 50)
    
    # Trigger error alert first
    trigger_error_alert()
    
    # Wait and then trigger phishing alert
    time.sleep(300)  # Wait 5 minutes
    trigger_phishing_alert()
    
    print("\n" + "=" * 50)
    print("🎬 Demo complete! Check your:")
    print("   📧 Gmail for alert emails")
    print("   📊 Grafana Alert Rules page for firing alerts")
    print("   🖥️ Dashboard for metrics")
