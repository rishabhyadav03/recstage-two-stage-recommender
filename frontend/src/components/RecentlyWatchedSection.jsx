import React, { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import MoviePoster from './MoviePoster';

function cleanMovieTitle(title) {
  if (!title) return '';
  let clean = title.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

export default function RecentlyWatchedSection({ watchHistory, loading }) {
  const scrollRef = useRef(null);

  const handleScroll = (direction) => {
    if (scrollRef.current) {
      const scrollAmount = direction === 'left' ? -400 : 400;
      scrollRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="w-48 h-6 bg-surface-3/70 rounded animate-pulse" />
        <div className="flex gap-5 overflow-hidden">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="min-w-[185px] max-w-[185px] flex-shrink-0 animate-pulse space-y-2">
              <div className="w-full aspect-[2/3] bg-surface-3/70 rounded-lg" />
              <div className="h-4 w-3/4 bg-surface-3/70 rounded mt-2" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!watchHistory || watchHistory.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Section Header: Title and Desktop Navigation Arrows */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-text-main tracking-tight">
          Because you watched
        </h3>

        {/* Desktop Left/Right Navigation Arrows */}
        <div className="hidden md:flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => handleScroll('left')}
            className="w-8 h-8 rounded-full border border-border bg-surface-2 hover:bg-surface-3 flex items-center justify-center text-text-muted hover:text-text-main transition-colors cursor-pointer"
            aria-label="Scroll watch history left"
          >
            <ChevronLeft size={16} />
          </button>
          <button
            type="button"
            onClick={() => handleScroll('right')}
            className="w-8 h-8 rounded-full border border-border bg-surface-2 hover:bg-surface-3 flex items-center justify-center text-text-muted hover:text-text-main transition-colors cursor-pointer"
            aria-label="Scroll watch history right"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {/* Horizontal Scroll-Snap Rail of w185 posters, hidden scrollbar, no card borders */}
      <div
        ref={scrollRef}
        className="flex gap-5 overflow-x-auto scroll-smooth snap-x snap-mandatory focus:outline-none [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden pb-2"
        tabIndex={0}
        aria-label="Because you watched rail"
      >
        {watchHistory.map((item, idx) => {
          const displayTitle = cleanMovieTitle(item.title);
          return (
            <div
              key={idx}
              className="min-w-[185px] max-w-[185px] flex-shrink-0 snap-start flex flex-col group select-none"
            >
              {/* Poster (w185, 2:3 aspect ratio, no card border) */}
              <div className="w-full aspect-[2/3] rounded-lg overflow-hidden bg-surface-2 shadow-soft border border-border/40">
                <MoviePoster
                  src={item.poster_url_small || item.poster_url}
                  title={displayTitle}
                  year={item.release_year}
                  loading="eager"
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Title below as plain text */}
              <div className="pt-2">
                <span
                  className="text-xs font-medium text-text-main line-clamp-1 leading-snug group-hover:text-accent transition-colors block"
                  title={item.title}
                >
                  {displayTitle}
                </span>
                <span className="text-[11px] text-text-muted block mt-0.5">
                  {item.release_year || '—'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
