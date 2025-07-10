import React, { useState } from 'react';
import '../styles/main.css';
import BackButton from './BackButton'; // ✅ Ensure this file exists

const initialState = {
  amount: '',
  time: '',
  location: '',
  merchant: '',
  device: '',
  account_age: '',
};

export default function TransactionForm({ onSubmit }) {
  const [form, setForm] = useState(initialState);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.amount || !form.time || !form.location || !form.merchant || !form.device || !form.account_age) {
      setError('Please fill in all fields.');
      return;
    }
    setError('');
    onSubmit(form);
    setForm(initialState);
  };

  return (
    <div className="foulpay-app-container foulpay-page-with-footer">
      <div className="foulpay-content">
        <form className="foulpay-form" onSubmit={handleSubmit}>
          <h2 className="foulpay-title">FoulPay Transaction</h2>

          <div className="foulpay-field-group">
            <label>Amount ($)</label>
            <input type="number" name="amount" value={form.amount} onChange={handleChange} min="0" step="0.01" required />
          </div>

          <div className="foulpay-field-group">
            <label>Time (HH:MM)</label>
            <input type="time" name="time" value={form.time} onChange={handleChange} required />
          </div>

          <div className="foulpay-field-group">
            <label>Location</label>
            <input type="text" name="location" value={form.location} onChange={handleChange} required />
          </div>

          <div className="foulpay-field-group">
            <label>Merchant</label>
            <input type="text" name="merchant" value={form.merchant} onChange={handleChange} required />
          </div>

          <div className="foulpay-field-group">
            <label>Device</label>
            <input type="text" name="device" value={form.device} onChange={handleChange} required />
          </div>

          <div className="foulpay-field-group">
            <label>Account Age (months)</label>
            <input type="number" name="account_age" value={form.account_age} onChange={handleChange} min="0" required />
          </div>

          {error && <div className="foulpay-error">{error}</div>}

          <button className="foulpay-btn" type="submit">Check Risk</button>
        </form>
      </div>

      <div className="foulpay-footer">
        <BackButton />
      </div>
    </div>
  );
}
