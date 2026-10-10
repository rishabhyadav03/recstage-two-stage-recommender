// Base URL for API calls. In development, defaults to '' so Vite proxy routes /api to the backend.
// In production or custom environments, set VITE_API_URL in .env (e.g. https://username-space.hf.space).
const rawBase = import.meta.env.VITE_API_URL ?? '';
// Strip trailing slash if present to avoid double-slashes
export const API_BASE = rawBase.replace(/\/+$/, '');

export function apiUrl(path) {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${cleanPath}`;
}

// Global server wake-up listener & state
let isWakingUp = false;
const wakeListeners = new Set();

export function subscribeServerWake(listener) {
  wakeListeners.add(listener);
  listener(isWakingUp);
  return () => wakeListeners.delete(listener);
}

export function setServerWake(val) {
  if (isWakingUp !== val) {
    isWakingUp = val;
    wakeListeners.forEach((fn) => fn(isWakingUp));
  }
}

// Transparent fetch wrapper with 5s timeout detection and automatic retries for cold starts
if (typeof window !== 'undefined' && window.fetch && !window.__recstage_fetch_patched) {
  window.__recstage_fetch_patched = true;
  const nativeFetch = window.fetch.bind(window);

  window.fetch = async function (input, init) {
    const urlStr = typeof input === 'string' ? input : (input?.url || '');
    if (!urlStr.includes('/api/')) {
      return nativeFetch(input, init);
    }

    let timer = setTimeout(() => {
      setServerWake(true);
    }, 5000);

    let attempt = 0;
    const maxRetries = 20; // Allow up to ~60s of retries for free tier cold boot
    const retryDelayMs = 3000;

    while (true) {
      try {
        const res = await nativeFetch(input, init);
        if (res.status === 502 || res.status === 503 || res.status === 504) {
          throw new Error(`Server returned status ${res.status}`);
        }
        clearTimeout(timer);
        setServerWake(false);
        return res;
      } catch (err) {
        attempt++;
        setServerWake(true);
        if (attempt >= maxRetries) {
          clearTimeout(timer);
          throw err;
        }
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      }
    }
  };
}
