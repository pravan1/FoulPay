#!/usr/bin/env python3
"""
FoulPay Fraud Detection API Server
Run this script to start the FastAPI server
"""

import uvicorn
import os
from main import app

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    
    print("🚀 Starting FoulPay Fraud Detection API...")
    print(f"📍 Server running at: http://{host}:{port}")
    print("📊 Health check available at: http://localhost:5000/health")
    print("📈 API documentation at: http://localhost:5000/docs")
    print("\n🔍 Features:")
    print("  • Real-time fraud detection")
    print("  • Machine learning ensemble models")
    print("  • Anomaly detection")
    print("  • User feedback collection")
    print("  • Transaction logging")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        reload=True,  # Enable hot reload in development
        log_level="info"
    )