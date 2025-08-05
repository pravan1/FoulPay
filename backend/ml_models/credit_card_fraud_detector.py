import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CreditCardFraudDetector:
    """
    Advanced credit card fraud detection based on real-world patterns
    """
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.models_loaded = False
        
        # Real credit card fraud indicators based on industry research
        self.high_risk_merchants = {
            'gas_station': 1.5,  # Gas stations have higher fraud rates
            'atm': 2.0,  # ATM withdrawals are high risk
            'online_gambling': 3.0,
            'wire_transfer': 2.5,
            'jewelry': 1.8,
            'electronics': 1.6,
            'cryptocurrency': 2.8
        }
        
        self.high_risk_countries = {
            'nigeria': 3.0,
            'romania': 2.5,
            'brazil': 2.0,
            'indonesia': 2.2,
            'venezuela': 2.8,
            'south_africa': 2.3
        }
        
        # Transaction velocity thresholds
        self.velocity_thresholds = {
            'hourly_count': 5,
            'hourly_amount': 5000,
            'daily_count': 20,
            'daily_amount': 10000
        }
        
    def extract_advanced_features(self, transaction_data: Dict) -> np.ndarray:
        """
        Extract advanced features based on real credit card fraud patterns
        """
        features = []
        
        # 1. Basic transaction features
        amount = transaction_data.get('amount', 0)
        hour = int(transaction_data.get('time', '12:00').split(':')[0])
        account_age_months = transaction_data.get('account_age', 12)
        
        # 2. Amount-based features
        features.extend([
            amount,
            np.log1p(amount),  # Log transform for better distribution
            amount ** 0.5,     # Square root transform
            1 if amount > 1000 else 0,  # High value transaction
            1 if amount > 5000 else 0,  # Very high value
            1 if amount % 100 == 0 else 0,  # Round amount (common in fraud)
            1 if amount < 10 else 0,  # Micro transaction (testing cards)
        ])
        
        # 3. Time-based features
        is_weekend = datetime.now().weekday() >= 5
        is_night = hour >= 22 or hour <= 5
        is_business_hours = 9 <= hour <= 17
        
        features.extend([
            hour,
            1 if is_weekend else 0,
            1 if is_night else 0,
            1 if is_business_hours else 0,
            np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
            np.cos(2 * np.pi * hour / 24)
        ])
        
        # 4. Location risk scoring
        location = transaction_data.get('location', '').lower()
        location_risk = 1.0
        for country, risk in self.high_risk_countries.items():
            if country in location:
                location_risk = risk
                break
        
        # Distance from home (simulated)
        is_international = 'international' in location or any(
            country in location for country in self.high_risk_countries
        )
        
        features.extend([
            location_risk,
            1 if is_international else 0,
            1 if 'vpn' in transaction_data.get('device', '').lower() else 0
        ])
        
        # 5. Merchant risk scoring
        merchant = transaction_data.get('merchant', '').lower()
        merchant_risk = 1.0
        merchant_category = 'standard'
        
        for category, risk in self.high_risk_merchants.items():
            if category in merchant:
                merchant_risk = risk
                merchant_category = category
                break
        
        # Check for suspicious merchant patterns
        is_new_merchant = 'new' in merchant or 'test' in merchant
        is_foreign_merchant = any(char.isalpha() and ord(char) > 127 for char in merchant)
        
        features.extend([
            merchant_risk,
            1 if is_new_merchant else 0,
            1 if is_foreign_merchant else 0,
            1 if merchant_category == 'atm' else 0,
            1 if merchant_category == 'online_gambling' else 0
        ])
        
        # 6. Device and channel features
        device = transaction_data.get('device', '').lower()
        device_risk_scores = {
            'mobile': 1.2,
            'tablet': 1.1,
            'desktop': 1.0,
            'atm': 1.5,
            'pos': 0.8,
            'unknown': 2.0
        }
        
        device_risk = device_risk_scores.get(device, 1.5)
        is_mobile = device in ['mobile', 'tablet']
        
        features.extend([
            device_risk,
            1 if is_mobile else 0,
            1 if device == 'unknown' else 0
        ])
        
        # 7. Account behavior features
        features.extend([
            account_age_months,
            1 if account_age_months < 3 else 0,  # New account
            1 if account_age_months < 1 else 0,  # Very new account
            np.log1p(account_age_months)
        ])
        
        # 8. Transaction pattern features
        # Simulate velocity checks (in real system, would query transaction history)
        velocity_score = self._calculate_velocity_score(amount, hour)
        
        features.extend([
            velocity_score,
            1 if velocity_score > 2 else 0  # High velocity flag
        ])
        
        # 9. Behavioral anomaly indicators
        unusual_time_location = is_night and is_international
        high_risk_combo = (merchant_risk > 1.5) and (amount > 1000)
        suspicious_pattern = (amount < 10) and is_new_merchant  # Card testing
        
        features.extend([
            1 if unusual_time_location else 0,
            1 if high_risk_combo else 0,
            1 if suspicious_pattern else 0
        ])
        
        return np.array(features)
    
    def _calculate_velocity_score(self, amount: float, hour: int) -> float:
        """
        Calculate velocity score based on transaction patterns
        """
        # Simulate velocity calculation (in production, would use actual transaction history)
        base_score = 1.0
        
        # High amount in short time
        if amount > 2000:
            base_score *= 1.5
            
        # Multiple transactions at odd hours
        if hour >= 23 or hour <= 5:
            base_score *= 1.3
            
        # Add some randomness to simulate real velocity checks
        base_score *= np.random.uniform(0.8, 1.2)
        
        return min(base_score, 3.0)
    
    def calculate_fraud_probability(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Calculate fraud probability using ensemble of models
        """
        # For demonstration, use rule-based scoring
        # In production, would use trained ML models
        
        fraud_score = 0.0
        confidence = 0.85
        risk_factors = []
        
        # Extract key features
        amount = features[0]
        is_high_value = features[3]
        is_round_amount = features[5]
        is_micro = features[6]
        is_night = features[9]
        location_risk = features[13]
        merchant_risk = features[17]
        is_new_account = features[25]
        velocity_score = features[28]
        
        # Rule-based scoring system based on real fraud patterns
        
        # Amount-based rules
        if is_high_value:
            fraud_score += 0.15
            risk_factors.append("High value transaction")
        
        if is_round_amount and amount > 500:
            fraud_score += 0.10
            risk_factors.append("Suspicious round amount")
        
        if is_micro:
            fraud_score += 0.20
            risk_factors.append("Card testing pattern detected")
        
        # Time-based rules
        if is_night:
            fraud_score += 0.10
            risk_factors.append("Unusual transaction time")
        
        # Location rules
        if location_risk > 2.0:
            fraud_score += 0.25
            risk_factors.append("High-risk location")
        
        # Merchant rules
        if merchant_risk > 2.0:
            fraud_score += 0.20
            risk_factors.append("High-risk merchant category")
        
        # Account rules
        if is_new_account:
            fraud_score += 0.15
            risk_factors.append("New account with limited history")
        
        # Velocity rules
        if velocity_score > 2.0:
            fraud_score += 0.25
            risk_factors.append("High transaction velocity")
        
        # Calculate final probability
        fraud_probability = min(fraud_score, 0.95)
        
        # Adjust confidence based on number of risk factors
        if len(risk_factors) >= 3:
            confidence = 0.90
        elif len(risk_factors) == 0:
            confidence = 0.95
            
        return {
            'probability': fraud_probability,
            'confidence': confidence,
            'risk_factors': risk_factors,
            'requires_3ds': fraud_probability > 0.3,  # 3D Secure recommendation
            'requires_manual_review': fraud_probability > 0.6
        }
    
    def get_risk_score(self, transaction_data: Dict) -> Dict[str, Any]:
        """
        Main method to get comprehensive risk assessment
        """
        # Extract features
        features = self.extract_advanced_features(transaction_data)
        
        # Calculate fraud probability
        fraud_result = self.calculate_fraud_probability(features)
        
        # Convert to 0-100 risk score
        risk_score = int(fraud_result['probability'] * 100)
        
        # Determine risk level based on new thresholds
        if risk_score > 80:
            risk_level = "HIGH"
            action = "BLOCK"
        elif risk_score > 50:
            risk_level = "MEDIUM"
            action = "MANUAL_REVIEW"
        else:
            risk_level = "LOW"
            action = "APPROVE"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recommended_action': action,
            'confidence': fraud_result['confidence'],
            'risk_factors': fraud_result['risk_factors'],
            'requires_3ds': fraud_result['requires_3ds'],
            'requires_manual_review': fraud_result['requires_manual_review'],
            'fraud_probability': fraud_result['probability']
        }