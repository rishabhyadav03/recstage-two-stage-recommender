import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Compass } from 'lucide-react';
import { formatRating } from '../utils/formatters';

function formatMovieTitle(title) {
  if (!title) return '';
  let clean = title.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

export default function RecommendCard({ item, rank, onExploreSimilar, animationIndex = 0 }) {
  const [overlayOpen, setOverlayOpen] = useState(false);
  const cardRef = useRef(null);

  const cleanTitle = formatMovieTitle(item.title);
  const posterSrc = item.poster_url || item.poster_url_large;
  const reason = item.why || item.reason || 'Matched based on latent representations and contextual signals.';
  const displayGenres = item.genres?.slice(0, 2).join(', ') || '';
  const metaLine = item.release_year 
    ? `${item.release_year}${displayGenres ? ` · ${displayGenres}` : ''}`
    : displayGenres;

  // Close overlay when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (cardRef.current && !cardRef.current.contains(event.target)) {
        setOverlayOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, []);

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.25, 
        delay: Math.min(animationIndex * 0.03, 0.35),
        ease: [0.25, 1, 0.5, 1] 
      }}
      className="group relative flex flex-col focus:outline-none select-none"
    >
      {/* Poster Container with 4px lift on hover / focus */}
      <div 
        className="relative w-full aspect-[2/3] rounded-lg overflow-hidden bg-surface-2 shadow-soft border border-border/40 transition-transform duration-200 ease-out group-hover:-translate-y-1 group-focus-within:-translate-y-1 cursor-pointer"
        tabIndex={0}
        role="button"
        aria-label={`${cleanTitle}, Rank ${rank}`}
        onClick={() => setOverlayOpen(prev => !prev)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setOverlayOpen(prev => !prev);
          }
        }}
      >
        {/* Poster Image or Typographic Fallback */}
        {posterSrc ? (
          <img
            src={posterSrc}
            alt={cleanTitle}
            loading="eager"
            decoding="async"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col justify-between p-3 bg-surface-2 text-text-muted">
            <span className="text-[10px] uppercase tracking-wider opacity-70">
              {item.release_year || 'Film'}
            </span>
            <span className="text-xs font-semibold text-text-main line-clamp-3 leading-snug">
              {cleanTitle}
            </span>
            <span className="text-[10px] opacity-60">No poster</span>
          </div>
        )}

        {/* Soft dark scrim at bottom behind numeral (gradient from rgba(0,0,0,0.65) to transparent 40% up) */}
        <div 
          className="absolute inset-x-0 bottom-0 h-[40%] bg-gradient-to-t from-[rgba(0,0,0,0.65)] to-transparent pointer-events-none z-10"
          aria-hidden="true"
        />

        {/* Rank Numeral: fully INSIDE poster at bottom 0.75rem, left 0.75rem, line-height 1 */}
        <div 
          className="absolute bottom-3 left-3 z-20 font-serif italic font-bold text-[2.5rem] leading-[1] text-white select-none pointer-events-none [text-shadow:_0_1px_8px_rgba(0,0,0,0.5)]"
          aria-hidden="true"
        >
          {rank}
        </div>

        {/* Dark Overlay fading in on Hover / Keyboard Focus / Touch Tap */}
        <div
          className={`absolute inset-0 bg-black/85 backdrop-blur-xs p-3.5 flex flex-col justify-between text-white transition-opacity duration-200 z-[15] rounded-lg ${
            overlayOpen
              ? 'opacity-100 pointer-events-auto'
              : 'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 pointer-events-none group-hover:pointer-events-auto group-focus-within:pointer-events-auto'
          }`}
        >
          {/* Reason in one plain human sentence: small WHY FOR YOU label, up to 5 lines with ellipsis */}
          <div className="pt-1">
            <span className="text-[10px] font-semibold tracking-wider uppercase text-amber-300 block mb-1">
              Why for you
            </span>
            <p className="text-xs text-white/95 line-clamp-5 leading-relaxed font-normal">
              {reason}
            </p>
          </div>

          {/* Action: "Similar →" only, aligned right with left space reserved for numeral */}
          <div className="pt-2 border-t border-white/15 flex items-center justify-end text-xs pl-12">
            {onExploreSimilar && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onExploreSimilar(item);
                }}
                className="text-white/80 hover:text-white font-medium cursor-pointer inline-flex items-center gap-1 hover:underline underline-offset-2"
                title="Explore similar films"
              >
                <Compass size={12} />
                <span>Similar &rarr;</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Item Info under Poster: Title, then "1996 · Drama, Thriller", then "★ 4.21" */}
      <div className="pt-2">
        {/* Title */}
        <h3 
          className="text-sm font-semibold text-text-main line-clamp-1 leading-snug group-hover:text-accent transition-colors"
          title={item.title}
        >
          {cleanTitle}
        </h3>

        {/* Muted "1996 · Drama, Thriller" line */}
        <div className="text-xs text-text-muted mt-0.5 line-clamp-1">
          {metaLine || 'Film'}
        </div>

        {/* "★ 4.21" line */}
        <div className="text-xs font-semibold text-text-main font-numeric flex items-center gap-1 mt-0.5">
          <span className="text-amber-400">★</span>
          <span>{formatRating(item.avg_rating)}</span>
        </div>
      </div>
    </motion.div>
  );
}
