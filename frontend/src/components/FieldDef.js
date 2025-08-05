import React from 'react';
import BackButton from './BackButton'; // Make sure this is in the same folder

export default function FieldDef() {
  return (
    <div className="foulpay-app-container foulpay-page-with-footer">
      <div className="foulpay-content">
        <h1 className="foulpay-title">Transaction Field Definitions</h1>

        <div className="foulpay-description">
          <p>
            Each field below is used as a feature in our machine learning model to detect fraud. Understanding them helps
            you provide accurate inputs and interpret the results.
          </p>
        </div>

        <div className="foulpay-field-info">
          <h3>💰 Amount</h3>
          <ul>
            <li>Total value of the transaction in U.S. dollars.</li>
            <li>Unusual spikes in amount may signal fraud.</li>
          </ul>
        </div>

        <div className="foulpay-field-info">
          <h3>🕒 Time</h3>
          <ul>
            <li>Hour of the day in 24-hour format (0–23).</li>
            <li>Unusual times (e.g., late night) may raise risk.</li>
          </ul>
        </div>

        <div className="foulpay-field-info">
          <h3>📍 Location</h3>
          <ul>
            <li>City or region where the transaction occurred.</li>
            <li>New or distant locations from past history may be suspicious.</li>
          </ul>
        </div>

        <div className="foulpay-field-info">
          <h3>🏪 Merchant</h3>
          <ul>
            <li>Category of merchant (e.g., Crypto, Grocery, Travel).</li>
            <li>High-risk categories (e.g., Crypto) are more fraud-prone.</li>
          </ul>
        </div>

        <div className="foulpay-field-info">
          <h3>📱 Device</h3>
          <ul>
            <li>Device type used for the transaction (Mobile, Laptop, etc.).</li>
            <li>Unknown or new devices may increase suspicion.</li>
          </ul>
        </div>

        <div className="foulpay-field-info">
          <h3>📆 Account Age</h3>
          <ul>
            <li>How long the account has existed, in months.</li>
            <li>Newer accounts have a higher baseline fraud risk.</li>
          </ul>
        </div>
      </div>

      <div className="foulpay-footer">
        <BackButton />
      </div>
    </div>
  );
}
