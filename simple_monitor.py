#!/usr/bin/env python3
from flask import Flask, jsonify, render_template_string
import requests
import time
import threading
from datetime import datetime
import json

app = Flask(__name__)

# Store metrics
metrics_data = {
    'total_requests': 0,
    'predictions': {'phishing': 0, 'legitimate': 0},
    'content_types': {'email': 0, 'message': 0, 'url': 0},
    'response_times': [],
    'last_updated': None,
    'error': None
}

def safe_float_convert(value):
    """Safely convert string to float, then to int"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0

def fetch_metrics():
    """Fetch metrics from the phishing API"""
    while True:
        try:
            response = requests.get('http://localhost:8000/metrics', timeout=5)
            if response.status_code == 200:
                lines = response.text.split('\n')
                
                # Reset counters
                metrics_data['predictions'] = {'phishing': 0, 'legitimate': 0}
                metrics_data['content_types'] = {'email': 0, 'message': 0, 'url': 0}
                metrics_data['error'] = None
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if line.startswith('phishing_requests_total'):
                        try:
                            value = line.split()[-1]
                            metrics_data['total_requests'] = safe_float_convert(value)
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing requests: {e}")
                            
                    elif line.startswith('phishing_predictions_total'):
                        try:
                            parts = line.split('{')
                            if len(parts) > 1:
                                labels = parts[1].split('}')[0]
                                value = line.split()[-1]
                                count = safe_float_convert(value)
                                
                                if 'is_phishing="True"' in labels:
                                    metrics_data['predictions']['phishing'] = count
                                elif 'is_phishing="False"' in labels:
                                    metrics_data['predictions']['legitimate'] = count
                                
                                if 'content_type="email"' in labels:
                                    metrics_data['content_types']['email'] = count
                                elif 'content_type="message"' in labels:
                                    metrics_data['content_types']['message'] = count
                                elif 'content_type="url"' in labels:
                                    metrics_data['content_types']['url'] = count
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing predictions: {e}")
                
                metrics_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"✅ Metrics updated: {metrics_data['total_requests']} requests")
                
        except Exception as e:
            error_msg = f"Error fetching metrics: {e}"
            metrics_data['error'] = error_msg
            print(error_msg)
        
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def dashboard():
    """Simple monitoring dashboard"""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phishing Detection Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
            .metric { background: white; padding: 20px; margin: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            .phishing { color: #e74c3c; font-weight: bold; }
            .legitimate { color: #27ae60; font-weight: bold; }
            .updated { color: #7f8c8d; font-size: 12px; margin-bottom: 20px; }
            .error { background: #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0; }
            h1 { color: #2d3436; }
            h2 { color: #636e72; margin: 5px 0; }
            h3 { color: #2d3436; margin-top: 0; }
            .stats { display: flex; justify-content: space-between; }
            .stat-item { text-align: center; }
        </style>
    </head>
    <body>
        <h1>🛡️ Phishing Detection Monitoring Dashboard</h1>
        <div class="updated">Last updated: {{ last_updated or 'Never' }}</div>
        
        {% if error %}
        <div class="error">
            <strong>⚠️ Error:</strong> {{ error }}
        </div>
        {% endif %}
        
        <div class="grid">
            <div class="metric">
                <h3>📊 Total Requests</h3>
                <h2>{{ total_requests }}</h2>
                <p>Total API requests made</p>
            </div>
            
            <div class="metric">
                <h3>🎯 Prediction Analysis</h3>
                <div class="stats">
                    <div class="stat-item">
                        <p class="phishing">Phishing</p>
                        <h2>{{ predictions.phishing }}</h2>
                    </div>
                    <div class="stat-item">
                        <p class="legitimate">Legitimate</p>
                        <h2>{{ predictions.legitimate }}</h2>
                    </div>
                </div>
            </div>
            
            <div class="metric">
                <h3>📝 Content Types</h3>
                <p>📧 Emails: {{ content_types.email }}</p>
                <p>💬 Messages: {{ content_types.message }}</p>
                <p>🔗 URLs: {{ content_types.url }}</p>
            </div>
            
            <div class="metric">
                <h3>🔧 Actions</h3>
                <p><a href="/metrics" target="_blank">📋 View Raw Metrics</a></p>
                <p><a href="/health" target="_blank">❤️ API Health</a></p>
                <p><a href="/api/metrics" target="_blank">📊 JSON API</a></p>
                <p><button onclick="location.reload()">🔄 Refresh Now</button></p>
            </div>
        </div>
        
        <div class="metric">
            <h3>ℹ️ System Status</h3>
            <p><strong>API:</strong> {% if total_requests > 0 %}✅ Running{% else %}❌ Not detected{% endif %}</p>
            <p><strong>Monitor:</strong> ✅ Active</p>
            <p><strong>Auto-refresh:</strong> Every 10 seconds</p>
        </div>
        
        <script>
            // Auto-refresh every 10 seconds
            setTimeout(() => location.reload(), 10000);
            
            // Add some visual feedback
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Phishing Monitor Dashboard Loaded');
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(template, **metrics_data)

@app.route('/metrics')
def show_metrics():
    """Show raw metrics from phishing API"""
    try:
        response = requests.get('http://localhost:8000/metrics')
        return f"<pre>{response.text}</pre>"
    except Exception as e:
        return f"<pre>Error fetching metrics: {e}</pre>"

@app.route('/health')
def health():
    """Check API health"""
    try:
        response = requests.get('http://localhost:8000/health')
        return jsonify({
            "api_health": response.json(),
            "monitor_health": "running",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/metrics')
def api_metrics():
    """JSON API for metrics"""
    return jsonify(metrics_data)

if __name__ == '__main__':
    # Start background thread to fetch metrics
    thread = threading.Thread(target=fetch_metrics, daemon=True)
    thread.start()
    
    print("🚀 Starting monitoring dashboard on http://localhost:5000")
    print("📊 Dashboard will auto-refresh every 10 seconds")
    print("⏳ Waiting for metrics from phishing API...")
    app.run(host='0.0.0.0', port=5000, debug=False)
