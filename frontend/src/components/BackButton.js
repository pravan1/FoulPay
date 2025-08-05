import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function BackButton({ to = -1, label = '← Back' }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(to)}
      className="foulpay-btn foulpay-btn-secondary"
      style={{ maxWidth: 160, marginBottom: '1.5rem' }}
    >
      {label}
    </button>
  );
}
