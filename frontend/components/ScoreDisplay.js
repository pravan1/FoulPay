import React from 'react';
import '../styles/main.css';

export default function ScoreDisplay({ score, interpretation }) {
  let color = '#1ecb5a';
  if (score > 80) color = '#ff3b30';
  else if (score > 50) color = '#ffcc00';

  return (
    <div className="foulpay-score-container">
      <div className="foulpay-score-label">Fraud Risk Score</div>
      <div className="foulpay-score-value" style={{ color }}>{score}</div>
      <div className="foulpay-score-interpretation">{interpretation}</div>
    </div>
  );
} 