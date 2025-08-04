import React, { useState, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation,
} from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';

import './styles/main.css';

import TransactionForm from './components/TransactionForm';
import ScoreDisplay from './components/ScoreDisplay';
import HomeScreen from './components/HomeScreen';
import LoadingScreen from './components/LoadingScreen';
import RiskScoreInfo from './components/RiskScoreInfo';
import FieldDef from './components/FieldDef';
import LoginPage from './components/LoginPage';
import LogoutButton from './components/LogoutButton';

function getInterpretation(score) {
  if (score > 80) return 'High risk: Likely fraudulent.';
  if (score > 50) return 'Medium risk: Unusual activity.';
  return 'Low risk: Transaction appears normal.';
}

function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth0();
  if (isLoading) return <LoadingScreen />;
  if (!isAuthenticated) return <Navigate to="/login" />;
  return children;
}

function AppRoutes({ score, interpretation, onSubmit, onBack }) {
  const location = useLocation();

  return (
    <>
      {location.pathname !== '/login' && <LogoutButton />}
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <HomeScreen onStart={() => (window.location.href = '/form')} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/form"
          element={
            <ProtectedRoute>
              <>
                <TransactionForm onSubmit={onSubmit} />
                {score !== null && (
                  <ScoreDisplay
                    score={score}
                    interpretation={interpretation}
                    confidence={transactionData?.confidence || 0.85}
                    anomaly_detected={transactionData?.anomaly_detected || false}
                    transaction_id={transactionData?.transaction_id}
                    onBack={onBack}
                  />
                )}
              </>
            </ProtectedRoute>
          }
        />
        <Route
          path="/risk-score-info"
          element={
            <ProtectedRoute>
              <RiskScoreInfo />
            </ProtectedRoute>
          }
        />
        <Route
          path="/field-def"
          element={
            <ProtectedRoute>
              <FieldDef />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </>
  );
}

function App() {
  const [score, setScore] = useState(null);
  const [interpretation, setInterpretation] = useState('');
  const [transactionData, setTransactionData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timeout = setTimeout(() => setLoading(false), 1600);
    return () => clearTimeout(timeout);
  }, []);

  const handleFormSubmit = async (form) => {
    try {
      let timeValue = form.time;

      // ✅ Fix: Only convert if not already in HH:MM format
      if (typeof timeValue !== 'string' || !timeValue.includes(':')) {
        const hours = Math.floor(timeValue / 60);
        const minutes = timeValue % 60;
        timeValue = `${hours.toString().padStart(2, '0')}:${minutes
          .toString()
          .padStart(2, '0')}`;
      }

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          amount: parseFloat(form.amount),
          time: timeValue,
          location: form.location,
          merchant: form.merchant,
          device: form.device,
          account_age: parseInt(form.account_age),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setScore(data.risk_score);
      setInterpretation(data.interpretation);
      setTransactionData(data); // Store full response data
    } catch (error) {
      console.error('Error calling prediction API:', error);
      const simulatedScore = Math.floor(Math.random() * 101);
      setScore(simulatedScore);
      setInterpretation(getInterpretation(simulatedScore));
    }
  };

  const handleReset = () => {
    setScore(null);
    setInterpretation('');
    setTransactionData(null);
  };

  if (loading) return <LoadingScreen />;

  return (
    <Router>
      <div className="foulpay-app-bg">
        <div className="foulpay-app-container">
          <AppRoutes
            score={score}
            interpretation={interpretation}
            onSubmit={handleFormSubmit}
            onBack={handleReset}
          />
        </div>
      </div>
    </Router>
  );
}

export default App;
