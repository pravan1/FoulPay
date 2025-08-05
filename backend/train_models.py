import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CreditCardFraudTrainer:
    """
    Train machine learning models on synthetic credit card fraud data
    that mimics real-world patterns
    """
    
    def __init__(self):
        self.models_dir = 'ml_models'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def generate_synthetic_fraud_data(self, n_samples=50000):
        """
        Generate synthetic credit card transaction data with realistic fraud patterns
        """
        print(f"Generating {n_samples} synthetic credit card transactions...")
        
        np.random.seed(42)
        
        # Initialize arrays
        data = {
            'amount': [],
            'hour': [],
            'day_of_week': [],
            'account_age_months': [],
            'merchant_category': [],
            'location_type': [],
            'device_type': [],
            'previous_transactions': [],
            'days_since_last_transaction': [],
            'is_fraud': []
        }
        
        # Merchant categories with fraud rates
        merchant_categories = {
            'grocery': 0.001,
            'gas_station': 0.02,
            'restaurant': 0.005,
            'online_shopping': 0.015,
            'atm_withdrawal': 0.03,
            'wire_transfer': 0.04,
            'jewelry': 0.025,
            'electronics': 0.02,
            'gambling': 0.05,
            'cryptocurrency': 0.06
        }
        
        # Location types with risk levels
        location_types = {
            'domestic': 0.005,
            'international': 0.03,
            'high_risk_country': 0.08,
            'vpn_detected': 0.05
        }
        
        # Device types
        device_types = ['mobile', 'desktop', 'tablet', 'atm', 'pos_terminal', 'unknown']
        
        for i in range(n_samples):
            # Account age (exponential distribution, newer accounts more likely)
            account_age = int(np.random.exponential(24))  # Average 24 months
            
            # Previous transactions (more for older accounts)
            prev_transactions = int(np.random.poisson(account_age * 5))
            
            # Days since last transaction
            days_since_last = np.random.choice([0, 1, 2, 3, 7, 14, 30, 60], 
                                             p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
            
            # Time patterns
            if np.random.random() < 0.8:  # 80% normal hours
                hour = np.random.choice(range(6, 23), p=self._get_normal_hour_distribution())
            else:  # 20% unusual hours
                hour = np.random.choice([0, 1, 2, 3, 4, 5, 23])
            
            day_of_week = np.random.randint(0, 7)
            
            # Merchant category
            merchant = np.random.choice(list(merchant_categories.keys()), 
                                      p=self._get_merchant_distribution())
            
            # Location
            if account_age < 3:  # New accounts more likely to have risky locations
                location = np.random.choice(list(location_types.keys()), 
                                          p=[0.7, 0.15, 0.1, 0.05])
            else:
                location = np.random.choice(list(location_types.keys()), 
                                          p=[0.85, 0.1, 0.03, 0.02])
            
            # Device
            device = np.random.choice(device_types, 
                                    p=[0.4, 0.3, 0.1, 0.1, 0.08, 0.02])
            
            # Amount patterns
            if merchant in ['grocery', 'restaurant', 'gas_station']:
                # Normal everyday transactions
                amount = np.random.lognormal(3.5, 0.8)
                amount = min(amount, 500)  # Cap at $500
            elif merchant in ['atm_withdrawal']:
                # ATM withdrawals often in round numbers
                amount = np.random.choice([20, 40, 50, 60, 80, 100, 200, 300, 400, 500], 
                                        p=[0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.05, 0.05])
            elif merchant in ['wire_transfer', 'cryptocurrency']:
                # Higher amounts
                amount = np.random.lognormal(6, 1.2)
            else:
                # General shopping
                amount = np.random.lognormal(4, 1)
            
            # Calculate fraud probability based on risk factors
            fraud_prob = self._calculate_fraud_probability(
                amount, hour, account_age, merchant, location, device,
                prev_transactions, days_since_last, merchant_categories, location_types
            )
            
            # Determine if fraud
            is_fraud = 1 if np.random.random() < fraud_prob else 0
            
            # Store data
            data['amount'].append(round(amount, 2))
            data['hour'].append(hour)
            data['day_of_week'].append(day_of_week)
            data['account_age_months'].append(account_age)
            data['merchant_category'].append(merchant)
            data['location_type'].append(location)
            data['device_type'].append(device)
            data['previous_transactions'].append(prev_transactions)
            data['days_since_last_transaction'].append(days_since_last)
            data['is_fraud'].append(is_fraud)
        
        df = pd.DataFrame(data)
        
        # Print statistics
        fraud_rate = df['is_fraud'].mean()
        print(f"Generated data with {fraud_rate*100:.2f}% fraud rate")
        print(f"Fraud transactions: {df['is_fraud'].sum()}")
        print(f"Normal transactions: {len(df) - df['is_fraud'].sum()}")
        
        return df
    
    def _get_normal_hour_distribution(self):
        """Get probability distribution for normal business hours"""
        hours = list(range(6, 23))
        # Peak hours: 10-12, 14-16, 18-20
        weights = []
        for h in hours:
            if h in [10, 11, 12]:
                weights.append(0.1)
            elif h in [14, 15, 16]:
                weights.append(0.08)
            elif h in [18, 19, 20]:
                weights.append(0.09)
            else:
                weights.append(0.04)
        
        # Normalize
        total = sum(weights)
        return [w/total for w in weights]
    
    def _get_merchant_distribution(self):
        """Get realistic merchant category distribution"""
        return [0.25, 0.15, 0.2, 0.15, 0.05, 0.02, 0.05, 0.08, 0.03, 0.02]
    
    def _calculate_fraud_probability(self, amount, hour, account_age, merchant, 
                                   location, device, prev_transactions, days_since_last,
                                   merchant_categories, location_types):
        """Calculate fraud probability based on multiple risk factors"""
        
        # Base fraud rate
        base_rate = merchant_categories[merchant]
        
        # Amount risk
        if amount > 5000:
            base_rate *= 3
        elif amount > 2000:
            base_rate *= 2
        elif amount < 5:  # Tiny amounts (card testing)
            base_rate *= 4
        
        # Time risk
        if hour in [0, 1, 2, 3, 4, 5]:
            base_rate *= 2.5
        elif hour in [22, 23]:
            base_rate *= 1.5
        
        # Account age risk
        if account_age < 1:
            base_rate *= 3
        elif account_age < 3:
            base_rate *= 2
        elif account_age < 6:
            base_rate *= 1.5
        
        # Location risk
        base_rate *= (1 + location_types[location] * 5)
        
        # Device risk
        if device == 'unknown':
            base_rate *= 2
        elif device in ['mobile', 'tablet'] and amount > 1000:
            base_rate *= 1.5
        
        # Velocity risk
        if prev_transactions < 10 and amount > 1000:
            base_rate *= 2
        
        if days_since_last == 0 and amount > 500:
            base_rate *= 1.8
        
        # Cap probability
        return min(base_rate, 0.95)
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        print("Preparing features...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['merchant_category', 'location_type', 'device_type']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col])
            label_encoders[col] = le
        
        # Create additional features
        df_features['amount_log'] = np.log1p(df_features['amount'])
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 5)).astype(int)
        df_features['amount_squared'] = df_features['amount'] ** 2
        df_features['amount_sqrt'] = np.sqrt(df_features['amount'])
        
        # High risk combinations
        df_features['high_risk_merchant'] = df_features['merchant_category'].isin(
            ['wire_transfer', 'cryptocurrency', 'gambling']
        ).astype(int)
        
        df_features['new_account_high_amount'] = (
            (df_features['account_age_months'] < 3) & (df_features['amount'] > 1000)
        ).astype(int)
        
        # Velocity features
        df_features['low_history_high_amount'] = (
            (df_features['previous_transactions'] < 10) & (df_features['amount'] > 500)
        ).astype(int)
        
        # Feature columns for training
        feature_cols = [
            'amount', 'amount_log', 'amount_squared', 'amount_sqrt',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'account_age_months', 'previous_transactions', 'days_since_last_transaction',
            'merchant_category_encoded', 'location_type_encoded', 'device_type_encoded',
            'high_risk_merchant', 'new_account_high_amount', 'low_history_high_amount'
        ]
        
        return df_features[feature_cols], df_features['is_fraud'], label_encoders
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models for ensemble"""
        print("Training models...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # 1. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        models['xgboost'] = xgb_model
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
        
        # 3. Isolation Forest for anomaly detection
        print("Training Isolation Forest...")
        # Train only on normal transactions
        normal_indices = y_train == 0
        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.05,  # Expected fraud rate
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train_scaled[normal_indices])
        models['isolation_forest'] = iso_forest
        
        # Evaluate models
        print("\nModel Evaluation:")
        print("="*50)
        
        for name, model in models.items():
            if name != 'isolation_forest':
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                print(f"\n{name.upper()}:")
                print(f"AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    print("\nTop 10 Important Features:")
                    print(importance_df)
        
        return models, scaler
    
    def save_models(self, models, scaler, label_encoders):
        """Save trained models and preprocessors"""
        print("\nSaving models...")
        
        # Save each model
        for name, model in models.items():
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_model.pkl'))
            print(f"Saved {name} model")
        
        # Save scaler
        joblib.dump(scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        print("Saved scaler")
        
        # Save label encoders
        joblib.dump(label_encoders, os.path.join(self.models_dir, 'label_encoders.pkl'))
        print("Saved label encoders")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'models': list(models.keys()),
            'features': list(scaler.feature_names_in_)
        }
        
        import json
        with open(os.path.join(self.models_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAll models saved to {self.models_dir}/")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting Credit Card Fraud Detection Training Pipeline")
        print("="*60)
        
        # Generate data
        df = self.generate_synthetic_fraud_data(n_samples=50000)
        
        # Prepare features
        X, y, label_encoders = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Training fraud rate: {y_train.mean()*100:.2f}%")
        print(f"Test fraud rate: {y_test.mean()*100:.2f}%")
        
        # Train models
        models, scaler = self.train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models(models, scaler, label_encoders)
        
        print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    trainer = CreditCardFraudTrainer()
    trainer.run_training_pipeline()