// Base URL for API calls. In development, defaults to '' so Vite proxy routes /api to the backend.
// In production or custom environments, set VITE_API_URL in .env (e.g. https://api.example.com).
export const API_BASE = import.meta.env.VITE_API_URL ?? '';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}
