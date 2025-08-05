from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from typing import Optional, Dict
from ml_models.fraud_detector import FraudDetector
from database.db_manager import DatabaseManager
from monitoring.real_time_monitor import monitor

app = FastAPI(title="FoulPay Fraud Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detector and database
fraud_detector = FraudDetector()
db_manager = DatabaseManager()

class Transaction(BaseModel):
    amount: float
    time: str
    location: str
    merchant: str
    device: str
    account_age: int

class FeedbackData(BaseModel):
    transaction_id: str
    is_fraud: bool
    user_feedback: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and monitoring on startup"""
    fraud_detector.load_models()
    # Start monitoring in background task
    import asyncio
    asyncio.create_task(monitor.start_monitoring())

@app.get("/")
async def root():
    return {"message": "FoulPay Fraud Detection API is running"}

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """
    Predict if a transaction is fraudulent
    """
    try:
        # Convert transaction to feature vector
        features = fraud_detector.prepare_features(transaction)
        
        # Get enhanced predictions using credit card fraud analysis
        predictions = fraud_detector.predict_with_credit_card_analysis(transaction)
        
        # Calculate ensemble risk score (0-100)
        risk_score = fraud_detector.calculate_risk_score(predictions)
        
        # Generate detailed interpretation
        interpretation = get_detailed_interpretation(risk_score, predictions)
        
        # Store transaction in database
        transaction_id = await db_manager.store_transaction({
            **transaction.dict(),
            "risk_score": risk_score,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "transaction_id": transaction_id,
            "risk_score": int(risk_score),
            "interpretation": interpretation,
            "confidence": predictions.get("confidence", 0.85),
            "anomaly_detected": predictions.get("anomaly", False),
            "risk_factors": predictions.get("risk_factors", []),
            "risk_level": predictions.get("risk_level", "UNKNOWN"),
            "recommended_action": predictions.get("recommended_action", "REVIEW"),
            "requires_3ds": predictions.get("requires_3ds", False),
            "requires_manual_review": predictions.get("requires_manual_review", False)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackData):
    """
    Submit manual feedback for model retraining
    """
    try:
        # Store feedback in database
        await db_manager.store_feedback(feedback.dict())
        
        # Trigger model retraining if enough new feedback
        feedback_count = await db_manager.get_feedback_count()
        if feedback_count % 100 == 0:  # Retrain every 100 feedbacks
            fraud_detector.retrain_models()
        
        return {"message": "Feedback received", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        stats = await db_manager.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint with monitoring data
    """
    health_report = await monitor.generate_health_report()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": fraud_detector.models_loaded,
        "monitoring": health_report
    }

@app.get("/metrics")
async def get_real_time_metrics():
    """
    Get real-time system metrics
    """
    return monitor.get_current_metrics()

def get_interpretation(score: float) -> str:
    """Generate human-readable interpretation of risk score"""
    if score > 80:
        return "High risk: Transaction flagged for manual review. Multiple fraud indicators detected."
    elif score > 50:
        return "Medium risk: Unusual activity patterns detected. Monitor closely."
    elif score > 20:
        return "Low-medium risk: Minor anomalies detected but transaction appears mostly normal."
    else:
        return "Low risk: Transaction appears normal with standard patterns."

def get_detailed_interpretation(score: float, predictions: Dict) -> str:
    """Generate detailed interpretation based on risk factors"""
    base_interpretation = get_interpretation(score)
    
    risk_factors = predictions.get("risk_factors", [])
    if risk_factors:
        factors_text = ". Key concerns: " + ", ".join(risk_factors[:3])
        return base_interpretation + factors_text
    
    return base_interpretation

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)