import React, { useState, useEffect } from 'react';
import { subscribeServerWake } from '../utils/api';

export default function ServerWakeBanner() {
  const [waking, setWaking] = useState(false);

  useEffect(() => {
    return subscribeServerWake(setWaking);
  }, []);

  if (!waking) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="bg-amber-500/10 dark:bg-amber-500/15 border-b border-amber-500/30 text-amber-800 dark:text-amber-200 px-4 py-2 text-xs sm:text-sm text-center flex items-center justify-center gap-2 transition-all duration-300"
    >
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
        <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
      </span>
      <span>The demo server is waking up, this can take about a minute. Retrying automatically...</span>
    </div>
  );
}
