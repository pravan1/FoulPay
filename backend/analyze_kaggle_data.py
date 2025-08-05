import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the credit card dataset
print("Loading credit card fraud dataset...")
df = pd.read_csv(r'C:\Users\prava\Downloads\creditcard.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Basic statistics
print(f"\nDataset info:")
print(df.info())

# Check class distribution
print(f"\nClass distribution:")
print(df['Class'].value_counts())
fraud_percentage = (df['Class'].sum() / len(df)) * 100
print(f"\nFraud percentage: {fraud_percentage:.2f}%")

# Analyze transaction amounts
print(f"\nTransaction amount statistics:")
print(df['Amount'].describe())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Analyze fraud vs normal transactions
fraud_df = df[df['Class'] == 1]
normal_df = df[df['Class'] == 0]

print(f"\nFraud transaction amount statistics:")
print(fraud_df['Amount'].describe())

print(f"\nNormal transaction amount statistics:")
print(normal_df['Amount'].describe())

# Analyze time patterns
if 'Time' in df.columns:
    print(f"\nTime feature statistics:")
    print(f"Min time: {df['Time'].min()}")
    print(f"Max time: {df['Time'].max()}")
    print(f"Time range suggests {df['Time'].max() / 3600:.1f} hours of data")

# Save sample for training
print(f"\nSaving processed dataset for model training...")
df.to_csv('data/creditcard_processed.csv', index=False)
print("Dataset saved to backend/data/creditcard_processed.csv")