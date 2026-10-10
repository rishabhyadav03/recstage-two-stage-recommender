import React, { useState } from 'react';

/**
 * MoviePoster component with 2:3 aspect ratio, lazy loading, soft fade-in,
 * and a clean typographic flat-color fallback when no poster is available.
 */
export default function MoviePoster({
  src,
  title,
  year,
  className = '',
  aspectRatio = '2/3',
  sizes = '(max-width: 768px) 50vw, 342px',
  loading = 'lazy',
}) {
  const [loaded, setLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  const showFallback = !src || hasError;

  return (
    <div
      className={`relative overflow-hidden rounded-lg bg-surface-2 select-none ${className}`}
      style={{ aspectRatio }}
    >
      {/* Fallback flat color block with typographic title */}
      {showFallback ? (
        <div className="absolute inset-0 flex flex-col justify-between p-3.5 bg-surface-2 text-text-muted border border-border/40">
          <span className="text-[11px] font-medium tracking-wider uppercase opacity-70">
            {year || 'Film'}
          </span>
          <div className="my-auto">
            <span className="text-sm font-semibold text-text-main line-clamp-3 leading-snug">
              {title}
            </span>
          </div>
          <span className="text-[10px] opacity-60">No poster</span>
        </div>
      ) : (
        <>
          {/* Soft skeleton placeholder while image is loading */}
          {!loaded && (
            <div className="absolute inset-0 bg-surface-3 animate-pulse" />
          )}

          {/* Real image with lazy/eager loading and smooth fade-in */}
          <img
            src={src}
            alt={year ? `${title} (${year})` : title}
            loading={loading}
            decoding="async"
            sizes={sizes}
            onLoad={() => setLoaded(true)}
            onError={() => setHasError(true)}
            className={`w-full h-full object-cover transition-opacity duration-300 ${
              loaded ? 'opacity-100' : 'opacity-0'
            }`}
          />
        </>
      )}
    </div>
  );
}
