export const AGE_MAP = {
  1: "Under 18",
  18: "18-24",
  25: "25-34",
  35: "35-44",
  45: "45-49",
  50: "50-55",
  56: "56+",
};

export function formatAge(age) {
  return AGE_MAP[age] || (age ? `Age ${age}` : "Age not specified");
}

export function formatGender(gender) {
  if (gender === 'M') return 'Male';
  if (gender === 'F') return 'Female';
  return 'Gender not specified';
}

export function formatRating(val) {
  if (val === undefined || val === null || isNaN(val)) return '—';
  return Number(val).toFixed(2);
}

export function formatLatency(val) {
  if (val === undefined || val === null || isNaN(val)) return '—';
  return Number(val).toFixed(1) + ' ms';
}

export function formatCount(val) {
  if (val === undefined || val === null || isNaN(val)) return '0';
  return Number(val).toLocaleString();
}
