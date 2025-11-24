# unified_app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import logging
import json
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi import Response
import os

# ==================== UNIFIED PHISHING DETECTOR CLASS ====================
class UnifiedPhishingDetector:
    def __init__(self):
        self.text_vectorizer = None
        self.model = None
        self.feature_names = []
        
    def extract_structural_features(self, content, content_type):
        features = {}
        features['length'] = len(content)
        features['num_special_chars'] = len(re.findall(r'[!@#$%^&*(),?":{}|<>]', content))
        features['num_digits'] = len(re.findall(r'\d', content))
        features['num_uppercase'] = len(re.findall(r'[A-Z]', content))
        features['uppercase_ratio'] = features['num_uppercase'] / max(1, features['length'])
        
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        features['num_urls'] = len(urls)
        
        if content_type == 'email':
            features['has_urgent_keywords'] = 1 if any(word in content.lower() for word in 
                                                     ['urgent', 'immediately', 'verify', 'confirm', 'security', 'account']) else 0
        elif content_type == 'message':
            features['has_shortened_url'] = 1 if any(domain in content for domain in 
                                                   ['bit.ly', 'tinyurl', 'goo.gl', 't.co']) else 0
        elif content_type == 'url':
            features['num_dots'] = content.count('.')
            features['num_hyphens'] = content.count('-')
            features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', content) else 0
            features['is_https'] = 1 if content.startswith('https') else 0
        
        return features
    
    def create_training_data(self):
        data = []
        
        # Phishing examples
        phishing_emails = [
            "URGENT: Your account will be suspended. Verify now: http://secure-login-verify.com",
            "Dear Customer, We detected suspicious activity. Confirm your identity: http://bank-security-update.net",
            "Security Alert: Unusual login attempt. Click here to secure: http://account-verification-center.com",
            "Your package delivery failed. Update shipping info: http://fedex-tracking-update.com",
            "Tax refund available. Claim your money: http://irs-refund-portal.com"
        ]
        
        legitimate_emails = [
            "Meeting reminder: Team sync at 3 PM tomorrow in Conference Room B",
            "Your monthly newsletter from Tech Company is here",
            "Password changed successfully for your account",
            "Project update: The new features have been deployed to staging",
            "Welcome to our service! Get started with these tips"
        ]
        
        phishing_messages = [
            "You've won $1000! Claim your prize: http://bit.ly/winprize",
            "URGENT: Your bank card is locked. Call now: 555-0123",
            "Free iPhone! Click here: http://tinyurl.com/freegift",
            "Account alert: Suspicious login. Verify: http://security-check.net",
            "Package delivery failed. Schedule redelivery: http://usps-reschedule.com"
        ]
        
        legitimate_messages = [
            "Your Uber is arriving in 3 minutes",
            "Your verification code is 123456",
            "Dental appointment reminder: Tomorrow at 2 PM",
            "Your Amazon order has been delivered",
            "Your food delivery is on the way"
        ]
        
        phishing_urls = [
            "http://secure-login-verify-account.com",
            "https://facebook-security-confirm.net",
            "http://paypal-verification-center.com",
            "https://apple-id-account-confirm.com",
            "http://microsoft-security-update.net"
        ]
        
        legitimate_urls = [
            "https://google.com/search?q=hello",
            "https://github.com/user/repo",
            "https://example.com/about",
            "http://wikipedia.org/science",
            "https://stackoverflow.com/questions"
        ]
        
        # Add to dataset
        for email in phishing_emails:
            data.append({'content': email, 'content_type': 'email', 'label': 1})
        for email in legitimate_emails:
            data.append({'content': email, 'content_type': 'email', 'label': 0})
        for msg in phishing_messages:
            data.append({'content': msg, 'content_type': 'message', 'label': 1})
        for msg in legitimate_messages:
            data.append({'content': msg, 'content_type': 'message', 'label': 0})
        for url in phishing_urls:
            data.append({'content': url, 'content_type': 'url', 'label': 1})
        for url in legitimate_urls:
            data.append({'content': url, 'content_type': 'url', 'label': 0})
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        structural_features = []
        for _, row in df.iterrows():
            features = self.extract_structural_features(row['content'], row['content_type'])
            structural_features.append(features)
        
        structural_df = pd.DataFrame(structural_features)
        
        if self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            text_features = self.text_vectorizer.fit_transform(df['content'])
        else:
            text_features = self.text_vectorizer.transform(df['content'])
        
        text_df = pd.DataFrame(text_features.toarray(), 
                             columns=[f"tfidf_{i}" for i in range(text_features.shape[1])])
        
        final_features = pd.concat([structural_df, text_df], axis=1)
        self.feature_names = list(final_features.columns)
        
        return final_features
    
    def train(self):
        print("Creating training data...")
        df = self.create_training_data()
        print(f"Training data created: {len(df)} samples")
        
        print("Preparing features...")
        X = self.prepare_features(df)
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        return self
    
    def predict(self, content, content_type):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        structural_features = self.extract_structural_features(content, content_type)
        structural_df = pd.DataFrame([structural_features])
        
        text_features = self.text_vectorizer.transform([content])
        text_df = pd.DataFrame(text_features.toarray(), 
                             columns=[f"tfidf_{i}" for i in range(text_features.shape[1])])
        
        features = pd.concat([structural_df, text_df], axis=1)
        
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.feature_names]
        
        probability = self.model.predict_proba(features)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability,
            'is_phishing': bool(prediction),
            'content_type': content_type
        }
    
    def save(self, filename='unified_phishing_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved as {filename}")

# ==================== FASTAPI APP SETUP ====================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phishing_api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('phishing_requests_total', 'Total requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('phishing_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('phishing_predictions_total', 'Total predictions', ['content_type', 'is_phishing'])

app = FastAPI(title="Phishing Detection API")

# Initialize model
model = None

# Check if model file exists, otherwise train a new one
if os.path.exists('unified_phishing_model.pkl'):
    try:
        with open('unified_phishing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully from file")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Training a new model...")
        model = UnifiedPhishingDetector()
        model.train()
        model.save()
else:
    print("🔄 No model file found. Training a new model...")
    model = UnifiedPhishingDetector()
    model.train()
    model.save()

class PredictionRequest(BaseModel):
    content: str
    content_type: str  # 'email', 'message', or 'url'

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_phishing: bool
    content_type: str
    model_version: str = "1.0.0"

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(endpoint=request.url.path, method=request.method).inc()
    REQUEST_DURATION.observe(duration)
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.4f}s")
    
    return response

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        return PredictionResponse(
            prediction=-1,
            probability=0.0,
            is_phishing=False,
            content_type=request.content_type
        )
    
    try:
        result = model.predict(request.content, request.content_type)
        
        # Log prediction
        PREDICTION_COUNT.labels(
            content_type=request.content_type,
            is_phishing=str(result['is_phishing'])
        ).inc()
        
        logger.info(f"Prediction - Content: {request.content[:50]}... - Phishing: {result['is_phishing']} - Confidence: {result['probability']:.3f}")
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            is_phishing=result['is_phishing'],
            content_type=result['content_type']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return PredictionResponse(
            prediction=-1,
            probability=0.0,
            is_phishing=False,
            content_type=request.content_type
        )

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY))

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    return {"message": "Phishing Detection API - Send POST requests to /predict"}

@app.post("/retrain")
async def retrain():
    """Endpoint to retrain the model"""
    global model
    try:
        logger.info("Starting model retraining...")
        model = UnifiedPhishingDetector()
        model.train()
        model.save()
        logger.info("Model retraining completed successfully")
        return {"status": "success", "message": "Model retrained successfully"}
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
