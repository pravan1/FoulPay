import os
from typing import Dict, Any, Optional
from datetime import datetime
import json

# For Firebase (uncomment when using)
# import firebase_admin
# from firebase_admin import credentials, firestore

# For MongoDB (uncomment when using)
# from pymongo import MongoClient

class DatabaseManager:
    """
    Database manager that supports both Firebase and MongoDB
    Falls back to local JSON storage for development
    """
    
    def __init__(self):
        self.db_type = os.getenv('DB_TYPE', 'local')  # local, firebase, mongodb
        self.db = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection based on configuration"""
        if self.db_type == 'firebase':
            self._init_firebase()
        elif self.db_type == 'mongodb':
            self._init_mongodb()
        else:
            self._init_local_storage()
    
    def _init_firebase(self):
        """Initialize Firebase Firestore"""
        try:
            # Initialize Firebase Admin SDK
            # cred = credentials.Certificate("path/to/serviceAccountKey.json")
            # firebase_admin.initialize_app(cred)
            # self.db = firestore.client()
            print("Firebase initialization - please configure credentials")
            self._init_local_storage()  # Fallback
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            self._init_local_storage()  # Fallback
    
    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            # connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            # self.client = MongoClient(connection_string)
            # self.db = self.client['foulpay_fraud_detection']
            print("MongoDB initialization - please configure connection string")
            self._init_local_storage()  # Fallback
        except Exception as e:
            print(f"MongoDB initialization failed: {e}")
            self._init_local_storage()  # Fallback
    
    def _init_local_storage(self):
        """Initialize local JSON file storage"""
        self.db_type = 'local'
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data files
        self.transactions_file = os.path.join(self.data_dir, 'transactions.json')
        self.feedback_file = os.path.join(self.data_dir, 'feedback.json')
        
        # Create files if they don't exist
        for file_path in [self.transactions_file, self.feedback_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    async def store_transaction(self, transaction_data: Dict[str, Any]) -> str:
        """Store transaction data and return transaction ID"""
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        transaction_data['transaction_id'] = transaction_id
        
        if self.db_type == 'firebase':
            return await self._store_transaction_firebase(transaction_data)
        elif self.db_type == 'mongodb':
            return await self._store_transaction_mongodb(transaction_data)
        else:
            return await self._store_transaction_local(transaction_data)
    
    async def _store_transaction_firebase(self, transaction_data: Dict[str, Any]) -> str:
        """Store transaction in Firebase Firestore"""
        try:
            doc_ref = self.db.collection('transactions').document(transaction_data['transaction_id'])
            doc_ref.set(transaction_data)
            return transaction_data['transaction_id']
        except Exception as e:
            print(f"Firebase storage error: {e}")
            return await self._store_transaction_local(transaction_data)
    
    async def _store_transaction_mongodb(self, transaction_data: Dict[str, Any]) -> str:
        """Store transaction in MongoDB"""
        try:
            collection = self.db['transactions']
            collection.insert_one(transaction_data)
            return transaction_data['transaction_id']
        except Exception as e:
            print(f"MongoDB storage error: {e}")
            return await self._store_transaction_local(transaction_data)
    
    async def _store_transaction_local(self, transaction_data: Dict[str, Any]) -> str:
        """Store transaction in local JSON file"""
        try:
            # Read existing data
            with open(self.transactions_file, 'r') as f:
                transactions = json.load(f)
            
            # Add new transaction
            transactions.append(transaction_data)
            
            # Keep only last 10000 transactions to prevent file from growing too large
            if len(transactions) > 10000:
                transactions = transactions[-10000:]
            
            # Write back to file
            with open(self.transactions_file, 'w') as f:
                json.dump(transactions, f, indent=2)
            
            return transaction_data['transaction_id']
        except Exception as e:
            print(f"Local storage error: {e}")
            return transaction_data['transaction_id']
    
    async def store_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Store user feedback for model improvement"""
        feedback_data['feedback_id'] = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        feedback_data['timestamp'] = datetime.now().isoformat()
        
        if self.db_type == 'firebase':
            return await self._store_feedback_firebase(feedback_data)
        elif self.db_type == 'mongodb':
            return await self._store_feedback_mongodb(feedback_data)
        else:
            return await self._store_feedback_local(feedback_data)
    
    async def _store_feedback_firebase(self, feedback_data: Dict[str, Any]) -> bool:
        """Store feedback in Firebase"""
        try:
            doc_ref = self.db.collection('feedback').document(feedback_data['feedback_id'])
            doc_ref.set(feedback_data)
            return True
        except Exception as e:
            print(f"Firebase feedback storage error: {e}")
            return await self._store_feedback_local(feedback_data)
    
    async def _store_feedback_mongodb(self, feedback_data: Dict[str, Any]) -> bool:
        """Store feedback in MongoDB"""
        try:
            collection = self.db['feedback']
            collection.insert_one(feedback_data)
            return True
        except Exception as e:
            print(f"MongoDB feedback storage error: {e}")
            return await self._store_feedback_local(feedback_data)
    
    async def _store_feedback_local(self, feedback_data: Dict[str, Any]) -> bool:
        """Store feedback in local JSON file"""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_list = json.load(f)
            
            feedback_list.append(feedback_data)
            
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_list, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Local feedback storage error: {e}")
            return False
    
    async def get_feedback_count(self) -> int:
        """Get total number of feedback entries"""
        if self.db_type == 'local':
            try:
                with open(self.feedback_file, 'r') as f:
                    feedback_list = json.load(f)
                return len(feedback_list)
            except:
                return 0
        # Implement for other databases as needed
        return 0
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'database_type': self.db_type,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.db_type == 'local':
            try:
                # Transaction stats
                with open(self.transactions_file, 'r') as f:
                    transactions = json.load(f)
                
                # Feedback stats
                with open(self.feedback_file, 'r') as f:
                    feedback_list = json.load(f)
                
                # Calculate stats
                total_transactions = len(transactions)
                high_risk_count = len([t for t in transactions if t.get('risk_score', 0) > 80])
                
                stats.update({
                    'total_transactions': total_transactions,
                    'high_risk_transactions': high_risk_count,
                    'total_feedback_entries': len(feedback_list),
                    'high_risk_percentage': (high_risk_count / total_transactions * 100) if total_transactions > 0 else 0
                })
                
            except Exception as e:
                stats['error'] = str(e)
        
        return stats
    
    async def get_recent_transactions(self, limit: int = 100) -> list:
        """Get recent transactions for monitoring"""
        if self.db_type == 'local':
            try:
                with open(self.transactions_file, 'r') as f:
                    transactions = json.load(f)
                return transactions[-limit:] if transactions else []
            except:
                return []
        
        # Implement for other databases as needed
        return []