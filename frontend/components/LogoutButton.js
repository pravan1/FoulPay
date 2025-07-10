import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

export default function LogoutButton() {
  const { logout } = useAuth0();

  return (
    <button
      className="foulpay-btn-secondary logout-btn"
      onClick={() => logout({ returnTo: window.location.origin })}
      aria-label="Log out of FoulPay"
    >
      Log Out
    </button>
  );
}
