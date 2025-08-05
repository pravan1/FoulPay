import React, { useState } from 'react';
import '../styles/main.css';

export default function ScoreDisplay({ score, interpretation, confidence, anomaly_detected, transaction_id, risk_factors, risk_level, recommended_action, requires_3ds, requires_manual_review }) {
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  
  let color = '#1ecb5a';
  let displayRiskLevel = risk_level || 'LOW';
  
  if (score > 80) {
    color = '#ff3b30';
    displayRiskLevel = risk_level || 'HIGH';
  } else if (score > 50) {
    color = '#ffcc00';
    displayRiskLevel = risk_level || 'MEDIUM';
  } else if (score > 20) {
    color = '#ff9500';
    displayRiskLevel = risk_level || 'LOW-MEDIUM';
  }
  
  const actionColors = {
    'BLOCK': '#ff3b30',
    'MANUAL_REVIEW': '#ffcc00',
    'ADDITIONAL_AUTH': '#ff9500',
    'MONITOR': '#007aff',
    'APPROVE': '#1ecb5a'
  };

  const handleFeedback = async (isFraud) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transaction_id: transaction_id,
          is_fraud: isFraud,
          user_feedback: `User marked transaction as ${isFraud ? 'fraudulent' : 'legitimate'}`
        }),
      });
      
      if (response.ok) {
        setFeedbackSubmitted(true);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div className="foulpay-score-container">
      <div className="foulpay-score-header">
        <div className="foulpay-score-label">Fraud Risk Assessment</div>
        <div className="foulpay-risk-level" style={{ color }}>{displayRiskLevel} RISK</div>
      </div>
      
      <div className="foulpay-score-main">
        <div className="foulpay-score-value" style={{ color }}>{score}</div>
        <div className="foulpay-score-max">/100</div>
      </div>
      
      <div className="foulpay-score-interpretation">{interpretation}</div>
      
      <div className="foulpay-score-details">
        <div className="foulpay-detail-item">
          <span className="foulpay-detail-label">Confidence:</span>
          <span className="foulpay-detail-value">{(confidence * 100).toFixed(1)}%</span>
        </div>
        
        {anomaly_detected && (
          <div className="foulpay-anomaly-alert">
            <span className="foulpay-anomaly-icon">⚠️</span>
            <span>Anomalous transaction pattern detected</span>
          </div>
        )}
        
        {recommended_action && (
          <div className="foulpay-detail-item">
            <span className="foulpay-detail-label">Action:</span>
            <span className="foulpay-detail-value" style={{ color: actionColors[recommended_action] || '#007aff' }}>
              {recommended_action.replace('_', ' ')}
            </span>
          </div>
        )}
        
        {requires_3ds && (
          <div className="foulpay-detail-item">
            <span className="foulpay-detail-label">3D Secure:</span>
            <span className="foulpay-detail-value" style={{ color: '#ff9500' }}>Required</span>
          </div>
        )}
      </div>
      
      {risk_factors && risk_factors.length > 0 && (
        <div className="foulpay-risk-factors">
          <div className="foulpay-risk-factors-label">Risk Factors:</div>
          <ul className="foulpay-risk-factors-list">
            {risk_factors.map((factor, index) => (
              <li key={index} className="foulpay-risk-factor-item">{factor}</li>
            ))}
          </ul>
        </div>
      )}
      
      {!feedbackSubmitted && (
        <div className="foulpay-feedback-section">
          <div className="foulpay-feedback-label">Was this assessment accurate?</div>
          <div className="foulpay-feedback-buttons">
            <button 
              className="foulpay-feedback-btn foulpay-feedback-fraud"
              onClick={() => handleFeedback(true)}
            >
              Mark as Fraud
            </button>
            <button 
              className="foulpay-feedback-btn foulpay-feedback-legitimate"
              onClick={() => handleFeedback(false)}
            >
              Mark as Legitimate
            </button>
          </div>
        </div>
      )}
      
      {feedbackSubmitted && (
        <div className="foulpay-feedback-thanks">
          Thank you for your feedback! This helps improve our fraud detection.
        </div>
      )}
    </div>
  );
} 