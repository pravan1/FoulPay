import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function HomeScreen({ onStart }) {
  const navigate = useNavigate();

  return (
    <div className="foulpay-app-container">
      <h1 className="foulpay-title">Welcome to FoulPay</h1>

      <div className="foulpay-description">
        <p>
          FoulPay is a machine learning-powered tool that detects unusual or potentially
          fraudulent financial transactions.
        </p>
        <p>
          Our system uses anomaly detection to evaluate how similar a transaction is to normal user behavior. Once
          you submit your inputs, you’ll receive a <strong>risk score</strong> based on the likelihood of fraud.
        </p>
      </div>

      <div
        className="foulpay-risk-info"
        role="button"
        onClick={() => navigate('/risk-score-info')}
        tabIndex={0}
      >
        <h3>🔍 What the Risk Score Means:</h3>
        <ul>
          <li><strong>0–50:</strong> Low risk — the transaction appears normal.</li>
          <li><strong>51–80:</strong> Medium risk — some unusual behavior detected.</li>
          <li><strong>81–100:</strong> High risk — likely fraudulent, needs review.</li>
        </ul>
      </div>

      <div
        className="foulpay-field-info"
        role="button"
        onClick={() => navigate('/field-def')}
        tabIndex={0}
      >
        <h3>📝 What Each Field Means:</h3>
        <ul>
          <li><strong>Amount:</strong> The transaction total in U.S. dollars.</li>
          <li><strong>Time:</strong> Time of transaction in 24-hour format (e.g., 13 for 1 PM).</li>
          <li><strong>Location:</strong> City or region where the transaction was made.</li>
          <li><strong>Merchant:</strong> Type of merchant (e.g., Crypto, Grocery, Electronics).</li>
          <li><strong>Device:</strong> Device used (Mobile, Laptop, Tablet).</li>
          <li><strong>Account Age:</strong> How long the account has been active (in months).</li>
        </ul>
      </div>

      <button className="foulpay-btn" onClick={onStart}>
        Start Detection
      </button>
    </div>
  );
}
