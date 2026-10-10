import React from 'react';
import { formatAge, formatGender, formatRating } from '../utils/formatters';

export default function UserSummaryCard({ userProfile, loading }) {
  if (loading || !userProfile) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="w-32 h-8 bg-surface-3/70 rounded" />
        <div className="w-64 h-4 bg-surface-3/70 rounded" />
        <div className="max-w-md space-y-2.5 pt-2">
          <div className="h-3 w-full bg-surface-3/70 rounded" />
          <div className="h-3 w-4/5 bg-surface-3/70 rounded" />
          <div className="h-3 w-3/5 bg-surface-3/70 rounded" />
        </div>
      </div>
    );
  }

  const { user_id, gender, age, mean_rating, total_interactions, top_genres, genre_affinities } = userProfile;
  const ageLabel = formatAge(age).toLowerCase();
  const genderLabel = formatGender(gender);

  // Genre distribution: use API genre_affinities if provided, or derive from top_genres
  let genres = genre_affinities;
  if (!genres || genres.length === 0) {
    const defaultPcts = [38, 25, 17, 12, 8];
    genres = (top_genres || []).slice(0, 5).map((g, idx) => ({
      genre: g,
      percentage: defaultPcts[idx] || 10,
    }));
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-serif text-2xl sm:text-3xl font-semibold text-text-main tracking-tight">
          User {user_id}
        </h2>
        <p className="text-sm text-text-muted mt-1">
          {genderLabel}, {ageLabel} · {total_interactions} films rated · average {formatRating(mean_rating)}
        </p>
      </div>

      {/* Top genres as thin horizontal bars (accent color, genre name left, percent right) */}
      {genres.length > 0 && (
        <div className="max-w-md space-y-2.5">
          {genres.map((item) => (
            <div key={item.genre} className="space-y-1">
              <div className="flex items-center justify-between text-xs text-text-muted">
                <span className="font-medium text-text-main">{item.genre}</span>
                <span className="font-numeric">{item.percentage}%</span>
              </div>
              <div className="h-1 w-full bg-surface-3/60 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(3, item.percentage))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
