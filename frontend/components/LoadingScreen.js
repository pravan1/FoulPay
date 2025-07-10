import React, { useEffect, useState } from 'react';

export default function LoadingScreen({ onFinish }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        const next = prev + 2;
        if (next >= 100) {
          clearInterval(interval);
          setTimeout(onFinish, 500);
        }
        return next;
      });
    }, 30);

    return () => clearInterval(interval);
  }, [onFinish]);

  return React.createElement(
    'div',
    { className: 'loading-screen' },
    React.createElement(
      'div',
      { className: 'loading-bar-container' },
      React.createElement('div', {
        className: 'loading-bar',
        style: { width: `${progress}%` },
      })
    ),
    React.createElement(
      'h2',
      { className: 'loading-text' },
      'FoulPay is preparing your experience...'
    )
  );
}
