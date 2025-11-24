import requests
import time

print("⏳ Waiting for alert to return to Normal state...")
print("This happens when error rate drops below 5% for 2 minutes")

while True:
    try:
        response = requests.get(
            "http://localhost:9090/api/v1/query",
            params={'query': 'ALERTS{alertname="High Error Rate",alertstate="firing"}'}
        )
        data = response.json()
        
        if not data['data']['result']:
            print("✅ Alert is now NORMAL")
            break
        else:
            print("🔴 Alert still FIRING - waiting...")
            
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(30)  # Check every 30 seconds

print("🎬 Ready for demo! Alert is in Normal state")
