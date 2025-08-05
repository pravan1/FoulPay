import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

export default function LoginPage() {
  const { loginWithRedirect, isLoading } = useAuth0();

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      {isLoading ? (
        <>
          <div className="loading-spinner" />
          <p className="loading-text">Loading...</p>
        </>
      ) : (
        <>
          <h1 className="foulpay-title">FoulPay</h1>
          <button
            className="foulpay-btn"
            onClick={() => loginWithRedirect()}
            aria-label="Sign in"
            style={{ marginTop: '1rem' }}
          >
            Sign in
          </button>
        </>
      )}
    </div>
  );
}
