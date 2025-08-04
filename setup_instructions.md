# FoulPay Fraud Detection Setup Instructions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- Git

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment (optional)**
   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

4. **Start the API server**
   ```bash
   python run.py
   ```

   The API will be available at:
   - Main API: http://localhost:5000
   - Health check: http://localhost:5000/health
   - API docs: http://localhost:5000/docs
   - Real-time metrics: http://localhost:5000/metrics

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

   The app will be available at http://localhost:3000

## 🧠 ML Features Implemented

### 1. **Supervised Learning Models**
- **XGBoost Classifier** - Primary fraud detection model
- **Random Forest** - Backup classifier
- Trained on synthetic transaction data with realistic fraud patterns

### 2. **Unsupervised Anomaly Detection**
- **Isolation Forest** - Detects unusual transaction patterns
- Identifies novel fraud types not seen in training data
- Complements supervised models for comprehensive coverage

### 3. **Feature Engineering**
- Transaction amount (log-transformed and z-score normalized)
- Time-based features (hour, weekend/weekday, night transactions)
- Account age and behavioral patterns
- Location, merchant, and device encoding
- Anomaly indicators

### 4. **Ensemble Scoring**
- Combines supervised and unsupervised model outputs
- Risk score from 0-100 with confidence intervals
- Multiple risk levels: LOW, LOW-MEDIUM, MEDIUM, HIGH

## 📊 Database Options

The system supports three database backends:

### Local JSON Storage (Default)
- No setup required
- Perfect for development and testing
- Data stored in `backend/data/` directory

### Firebase Firestore
1. Create Firebase project
2. Generate service account key
3. Update database configuration in `.env`
4. Set `DB_TYPE=firebase`

### MongoDB Atlas
1. Create MongoDB Atlas cluster (free tier available)
2. Get connection string
3. Update configuration in `.env`
4. Set `DB_TYPE=mongodb`

## 🔍 Real-Time Monitoring

### Features
- **Live Fraud Rate Tracking** - Monitor fraud detection rates
- **Model Performance Metrics** - Track confidence scores and accuracy
- **Automated Alerts** - Get notified of unusual patterns
- **System Health Monitoring** - Database and API status

### Endpoints
- `/health` - Comprehensive health report
- `/metrics` - Real-time system metrics
- `/stats` - Transaction and fraud statistics

### Alert Thresholds
- High fraud rate: >15% of transactions flagged
- Low confidence: <70% average model confidence
- System errors: >5 errors in 10 minutes

## 🎯 Usage Workflow

1. **Transaction Input** - User enters transaction details
2. **Feature Processing** - System extracts and engineers features
3. **ML Prediction** - Ensemble models generate risk score
4. **Risk Assessment** - Display color-coded risk level with interpretation
5. **User Feedback** - Collect manual validation for model improvement
6. **Continuous Learning** - System learns from feedback to improve accuracy

## 🔧 API Endpoints

### Core Endpoints
- `POST /predict` - Fraud risk prediction
- `POST /feedback` - Submit user feedback
- `GET /stats` - System statistics
- `GET /health` - Health check with monitoring
- `GET /metrics` - Real-time metrics

### Sample Transaction Request
```json
{
  "amount": 1500.00,
  "time": "23:30",
  "location": "New York",
  "merchant": "Online Store",
  "device": "Mobile",
  "account_age": 6
}
```

### Sample Response
```json
{
  "transaction_id": "txn_20240804_123456_789",
  "risk_score": 75,
  "interpretation": "Medium risk: Unusual activity patterns detected. Monitor closely.",
  "confidence": 0.87,
  "anomaly_detected": true
}
```

## 🛡️ Security Features

- **CORS Protection** - Configured for frontend domains
- **Input Validation** - Pydantic models ensure data integrity
- **Error Handling** - Comprehensive exception management
- **Logging** - Detailed system and error logs

## 📈 Model Performance

The synthetic training data creates realistic fraud patterns:
- **Base fraud rate**: 5% (realistic for financial systems)
- **High-risk indicators**: Large amounts, night transactions, new accounts
- **Model accuracy**: Typically >90% on synthetic data
- **False positive rate**: <5% for legitimate transactions

## 🔄 Continuous Improvement

### Automated Retraining
- Triggers after every 100 user feedback submissions
- Incorporates real-world fraud examples
- Maintains model accuracy over time

### Data Collection
- All transactions logged with predictions
- User feedback stored for model improvement
- System metrics tracked for performance monitoring

## 🚨 Production Considerations

### Scaling
- Use Redis for session storage and caching
- Deploy with Docker containers
- Use load balancers for high availability

### Database
- Migrate to production database (PostgreSQL/MongoDB)
- Implement proper indexing for performance
- Set up backup and recovery procedures

### Monitoring
- Integrate with alerting systems (PagerDuty, Slack)
- Set up log aggregation (ELK stack)
- Monitor API response times and error rates

### Security
- Implement API authentication (JWT tokens)
- Use HTTPS in production
- Regular security audits and updates