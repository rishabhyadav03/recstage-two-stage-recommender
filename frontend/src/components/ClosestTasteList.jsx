import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Film } from 'lucide-react';

function cleanTitle(title) {
  if (!title) return '';
  let clean = title.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

export default function ClosestTasteList({
  items = [],
  loading = false,
  onHover = () => {},
  maxItems = 5,
}) {
  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="space-y-3">
        <div className="w-48 h-5 bg-surface-3/70 rounded animate-pulse" />
        <div className="divide-y divide-border">
          {Array.from({ length: maxItems }).map((_, i) => (
            <div key={i} className="flex items-center gap-[16px] py-[12px] animate-pulse">
              <div className="w-[48px] h-[72px] rounded-md bg-surface-3/70 flex-shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="h-4 w-3/4 bg-surface-3/70 rounded" />
                <div className="h-3 w-1/2 bg-surface-3/70 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const displayItems = (items || []).slice(0, maxItems);

  if (displayItems.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      {/* Small serif heading */}
      <h3 className="font-serif text-lg font-semibold text-text-main tracking-tight">
        Closest to this user's taste
      </h3>

      {/* Compact list of films by retrieval score */}
      <div id="closest-taste-list" className="divide-y divide-border">
        {displayItems.map((item) => (
          <div
            key={item.movie_id}
            role="button"
            tabIndex={0}
            onClick={() => navigate(`/explore?item=${item.movie_id}`)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                navigate(`/explore?item=${item.movie_id}`);
              }
            }}
            onMouseEnter={() => onHover(item.movie_id)}
            onMouseLeave={() => onHover(null)}
            className="taste-row flex items-center gap-[16px] py-[12px] cursor-pointer group transition-colors"
          >
            {/* Poster: 48px wide (2:3, so 72px tall), rounded-md */}
            <div className="w-[48px] h-[72px] min-w-[48px] min-h-[72px] rounded-md overflow-hidden bg-surface-3 flex-shrink-0 border border-border/40 flex items-center justify-center shadow-xs">
              {item.poster_url ? (
                <img
                  src={item.poster_url}
                  alt={item.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                  loading="lazy"
                />
              ) : (
                <Film size={18} className="text-text-muted/60" />
              )}
            </div>

            {/* Movie metadata */}
            <div className="min-w-0 flex-1">
              {/* Title: about 18px serif on one line with ellipsis */}
              <h4
                className="font-serif text-[18px] leading-snug font-semibold text-text-main truncate group-hover:text-accent transition-colors"
                title={item.title}
              >
                {cleanTitle(item.title)}
              </h4>

              {/* Genre line and "Taste match" line: about 14px on the same row */}
              <div className="flex items-center justify-between gap-2 text-[14px] text-text-muted mt-1 leading-normal">
                <span className="truncate">
                  {item.year || ''} · {item.group || 'Film'}
                </span>
                <span className="font-numeric flex-shrink-0 text-text-muted">
                  Taste match {item.match_pct}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
