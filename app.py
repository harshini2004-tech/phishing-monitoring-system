from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import time
import logging
import json
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi import Response

# Load model
try:
    with open('unified_phishing_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model = None

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
