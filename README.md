 **Phishing Detection System**

A real-time cybersecurity platform that detects phishing attacks using machine learning with comprehensive monitoring, logging, and alerting capabilities.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**What This Project Does**

This system automatically detects phishing attempts in URLs and emails using AI, monitors all activity in real-time, and sends instant alerts when threats are found. It's a complete security monitoring solution that protects against phishing attacks before they can cause harm.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**How It Works**

1. **Detection**: Machine learning models analyze URLs and email content for phishing patterns
2. **Monitoring**: Tracks all system activity, performance, and threat detection rates
3. **Logging**: Stores all security events and detection results for analysis
4. **Alerting**: Sends immediate notifications via email/SMS when threats are detected

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




Start the detection service
uvicorn unified_app:app --host 0.0.0.0 --port 8001 --reload
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 Test with a suspicious URL
curl -X POST "http://localhost:8001/predict" \
  -d '{"content": "http://fake-bank-login.com", "content_type": "url"}'
  
**📊 Access Your Dashboard**
Detection API: http://localhost:8001

Monitoring: http://localhost:9090

Logs: http://localhost:3100

Alerts: Check your email/SMS
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


AI Detection: Identifies phishing URLs and emails automatically

Real-time Monitoring: Watch live threat detection and system health

Centralized Logging: All security events stored and searchable

Instant Alerts: Get notified immediately when threats are found

Web Dashboard: Beautiful Grafana interface to visualize everything
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Fake login pages & phishing websites

Suspicious email content and patterns

Malicious domains and redirect chains

Social engineering attempts

The system uses Prometheus for monitoring, Loki for logging, and integrates with email/SMS services for instant alerts when phishing threats are detected.
