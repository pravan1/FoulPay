import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.utils import resample
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KaggleCreditCardTrainer:
    """
    Train models using the Kaggle credit card dataset
    """
    
    def __init__(self):
        self.models_dir = 'ml_models'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_and_analyze_data(self):
        """Load and analyze the Kaggle dataset"""
        print("Loading Kaggle credit card dataset...")
        
        # Load the dataset
        df = pd.read_csv(r'C:\Users\prava\Downloads\creditcard.csv')
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if this is the standard credit card dataset with V1-V28 features
        if 'V1' in df.columns:
            print("\nDetected standard credit card fraud dataset with PCA features")
            return self.process_standard_dataset(df)
        else:
            print("\nDetected custom credit card dataset")
            return self.process_custom_dataset(df)
    
    def process_standard_dataset(self, df):
        """Process the standard credit card dataset with V1-V28 features"""
        print("\nProcessing standard dataset...")
        
        # Basic statistics
        print(f"Class distribution:")
        print(df['Class'].value_counts())
        fraud_rate = df['Class'].mean()
        print(f"Fraud rate: {fraud_rate*100:.2f}%")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Class']]
        X = df[feature_cols]
        y = df['Class']
        
        # Scale Amount and Time features
        scaler = StandardScaler()
        if 'Amount' in X.columns:
            X['Amount_scaled'] = scaler.fit_transform(X[['Amount']])
        if 'Time' in X.columns:
            X['Time_scaled'] = scaler.fit_transform(X[['Time']])
            # Create hour of day feature
            X['Hour'] = (X['Time'] % (24*3600)) / 3600
        
        return X, y, None
    
    def process_custom_dataset(self, df):
        """Process custom dataset with categorical features"""
        print("\nProcessing custom dataset...")
        
        # Display basic info
        print(f"Class distribution:")
        print(df['Class'].value_counts())
        fraud_rate = df['Class'].mean()
        print(f"Fraud rate: {fraud_rate*100:.2f}%")
        
        # Prepare features
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Handle categorical variables
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[f'{col}_encoded'] = le.fit_transform(X[col])
            label_encoders[col] = le
            X = X.drop(col, axis=1)
        
        # Feature engineering
        if 'amount' in X.columns:
            X['amount_log'] = np.log1p(X['amount'])
            X['amount_squared'] = X['amount'] ** 2
            X['high_amount'] = (X['amount'] > X['amount'].quantile(0.9)).astype(int)
        
        if 'time' in X.columns:
            X['hour'] = X['time'] // 100
            X['is_night'] = ((X['hour'] >= 22) | (X['hour'] <= 5)).astype(int)
            X['is_weekend'] = 0  # Would need day of week info
        
        if 'account_age' in X.columns:
            X['new_account'] = (X['account_age'] < 6).astype(int)
            X['account_age_log'] = np.log1p(X['account_age'])
        
        return X, y, label_encoders
    
    def balance_dataset(self, X, y, method='smote'):
        """Balance the dataset using various techniques"""
        print(f"\nBalancing dataset using {method}...")
        
        if method == 'smote':
            # Use SMOTE for oversampling
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'smotetomek':
            # Use combined over and under sampling
            smt = SMOTETomek(random_state=42)
            X_balanced, y_balanced = smt.fit_resample(X, y)
        elif method == 'undersample':
            # Undersample majority class
            df_temp = pd.concat([X, y], axis=1)
            df_majority = df_temp[y == 0]
            df_minority = df_temp[y == 1]
            
            df_majority_downsampled = resample(df_majority, 
                                             replace=False,
                                             n_samples=len(df_minority)*2,
                                             random_state=42)
            
            df_balanced = pd.concat([df_majority_downsampled, df_minority])
            X_balanced = df_balanced.iloc[:, :-1]
            y_balanced = df_balanced.iloc[:, -1]
        else:
            X_balanced, y_balanced = X, y
        
        print(f"Balanced dataset shape: {X_balanced.shape}")
        print(f"New class distribution: {pd.Series(y_balanced).value_counts()}")
        
        return X_balanced, y_balanced
    
    def train_ensemble_models(self, X_train, X_test, y_train, y_test):
        """Train an ensemble of models"""
        print("\nTraining ensemble models...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        results = {}
        
        # 1. XGBoost with tuned parameters
        print("\n1. Training XGBoost...")
        xgb_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_scaled, y_train)
        models['xgboost'] = xgb_model
        
        # Evaluate
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
        results['xgboost'] = {
            'auc': roc_auc_score(y_test, y_proba_xgb),
            'predictions': y_pred_xgb,
            'probabilities': y_proba_xgb
        }
        
        # 2. Random Forest with balanced class weights
        print("\n2. Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
        
        # Evaluate
        y_pred_rf = rf_model.predict(X_test_scaled)
        y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
        results['random_forest'] = {
            'auc': roc_auc_score(y_test, y_proba_rf),
            'predictions': y_pred_rf,
            'probabilities': y_proba_rf
        }
        
        # 3. Gradient Boosting
        print("\n3. Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
        
        # Evaluate
        y_pred_gb = gb_model.predict(X_test_scaled)
        y_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]
        results['gradient_boosting'] = {
            'auc': roc_auc_score(y_test, y_proba_gb),
            'predictions': y_pred_gb,
            'probabilities': y_proba_gb
        }
        
        # 4. Ensemble prediction
        print("\n4. Creating ensemble predictions...")
        ensemble_proba = (y_proba_xgb * 0.4 + y_proba_rf * 0.3 + y_proba_gb * 0.3)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        results['ensemble'] = {
            'auc': roc_auc_score(y_test, ensemble_proba),
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"AUC-ROC Score: {result['auc']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, result['predictions']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, result['predictions'])
            print("\nConfusion Matrix:")
            print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
            
            if cm[1,1] + cm[1,0] > 0:
                recall = cm[1,1] / (cm[1,1] + cm[1,0])
                print(f"Fraud Detection Rate (Recall): {recall*100:.2f}%")
        
        # Feature importance analysis
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature names
        feature_names = X_train.columns.tolist()
        
        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        print("\nTop 15 Important Features (XGBoost):")
        for idx, row in xgb_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return models, scaler, results
    
    def save_trained_models(self, models, scaler, label_encoders=None):
        """Save the trained models"""
        print("\nSaving trained models...")
        
        # Save models
        for name, model in models.items():
            joblib.dump(model, os.path.join(self.models_dir, f'kaggle_{name}_model.pkl'))
            print(f"Saved {name} model")
        
        # Save scaler
        joblib.dump(scaler, os.path.join(self.models_dir, 'kaggle_scaler.pkl'))
        
        # Save label encoders if any
        if label_encoders:
            joblib.dump(label_encoders, os.path.join(self.models_dir, 'kaggle_label_encoders.pkl'))
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'models': list(models.keys()),
            'source': 'kaggle_credit_card_dataset'
        }
        
        import json
        with open(os.path.join(self.models_dir, 'kaggle_model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAll models saved to {self.models_dir}/")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting Kaggle Credit Card Fraud Detection Training")
        print("="*60)
        
        # Load and process data
        X, y, label_encoders = self.load_and_analyze_data()
        
        # Balance dataset (optional - comment out if not needed)
        # X, y = self.balance_dataset(X, y, method='smote')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Training fraud rate: {y_train.mean()*100:.2f}%")
        print(f"Test fraud rate: {y_test.mean()*100:.2f}%")
        
        # Train models
        models, scaler, results = self.train_ensemble_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_trained_models(models, scaler, label_encoders)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
        # Return best model info
        best_model = max(results.items(), key=lambda x: x[1]['auc'])
        print(f"\nBest model: {best_model[0]} with AUC: {best_model[1]['auc']:.4f}")

if __name__ == "__main__":
    trainer = KaggleCreditCardTrainer()
    trainer.run_training_pipeline()