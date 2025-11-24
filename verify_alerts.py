import requests
import json

# Check current alert rules
try:
    # This might require Grafana API, but let's check the UI
    print("1. Go to: http://localhost:3000/alerting")
    print("2. Check 'Alert rules' - you should see:")
    print("   - High Error Rate")
    print("   - High Phishing Rate") 
    print("   - API Service Down")
    print("3. States should be 'Normal' (green)")
    
    # Check current error rate
    response = requests.get(
        "http://localhost:9090/api/v1/query",
        params={'query': '100 * (rate(phishing_requests_total{status_code!="200"}[2m]) / rate(phishing_requests_total[2m]))'}
    )
    data = response.json()
    if data['data']['result']:
        rate = float(data['data']['result'][0]['value'][1])
        print(f"4. Current error rate: {rate:.2f}%")
    else:
        print("4. No error rate data yet")
        
except Exception as e:
    print(f"Error: {e}")

