import React from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { Compass, Sparkles } from 'lucide-react';

function formatMovieTitle(title) {
  if (!title) return '';
  let clean = title.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

export default function RecommendHero({ 
  item, 
  loading, 
  userId, 
  mode 
}) {
  const shouldReduceMotion = useReducedMotion();

  const posterUrl = item?.poster_url_large || 
    (item?.poster_url ? item.poster_url.replace('/w342/', '/w500/') : null);
  const backdropUrl = item?.backdrop_url || null;
  const overview = item?.overview || null;
  const reason = item?.why || item?.reason || null;
  const cleanTitle = item?.title ? formatMovieTitle(item.title) : '';

  // Format the metadata line: "1952 · Musical, Romance · ★ 4.28"
  const metaParts = [];
  if (item?.release_year) metaParts.push(item.release_year);
  if (item?.genres && item.genres.length > 0) metaParts.push(item.genres.join(', '));
  if (item?.avg_rating) metaParts.push(`★ ${Number(item.avg_rating).toFixed(2)}`);
  const metaLine = metaParts.join(' · ');

  const transitionConfig = shouldReduceMotion
    ? { duration: 0 }
    : { duration: 0.35, ease: [0.25, 1, 0.5, 1] };

  return (
    <section 
      aria-label="Top recommendation spotlight"
      className="relative left-1/2 right-1/2 -ml-[50vw] -mr-[50vw] w-screen mb-0"
    >
      {/* Blurred Backdrop Background (fades smoothly to page background in light and dark) */}
      {backdropUrl && !loading && (
        <div 
          className="absolute inset-0 overflow-hidden pointer-events-none select-none" 
          aria-hidden="true"
        >
          <img
            src={backdropUrl}
            alt=""
            className="w-full h-full object-cover object-center filter blur-xs sm:blur-sm md:blur-[6px] opacity-25 dark:opacity-35 scale-105 transition-opacity duration-500"
          />
          {/* Subtle multi-directional gradients fading to page background */}
          <div className="absolute inset-0 bg-gradient-to-b from-background via-background/60 to-background" />
          <div className="absolute inset-0 bg-gradient-to-r from-background via-background/75 to-background/40" />
        </div>
      )}

      {/* Main Content Container (re-aligned to max-w-6xl with proper grid padding) */}
      <div className="relative z-10 mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 pt-2 sm:pt-4 pb-8">
        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div
              key="hero-skeleton"
              initial={shouldReduceMotion ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={shouldReduceMotion ? undefined : { opacity: 0 }}
              transition={transitionConfig}
              className="flex flex-col sm:flex-row items-center sm:items-start gap-6 sm:gap-8 md:gap-12"
              aria-label="Loading top recommendation"
            >
              {/* Poster Skeleton: matches 2:3 aspect ratio */}
              <div 
                className="w-48 sm:w-56 md:w-64 lg:w-72 shrink-0 aspect-[2/3] rounded-md bg-surface-3/70 animate-pulse shadow-soft border border-border/40" 
              />

              {/* Info Skeleton */}
              <div className="flex-1 w-full space-y-4 pt-1 text-center sm:text-left">
                {/* Badge skeleton */}
                <div className="h-5 w-32 rounded-full bg-surface-3/70 animate-pulse mx-auto sm:mx-0" />
                {/* Title skeleton */}
                <div className="h-12 sm:h-14 w-3/4 max-w-lg rounded-md bg-surface-3/70 animate-pulse mx-auto sm:mx-0" />
                {/* Meta line skeleton */}
                <div className="h-5 w-56 rounded bg-surface-3/70 animate-pulse mx-auto sm:mx-0" />
                {/* Overview skeleton (3 lines clamped shape) */}
                <div className="space-y-2 pt-1 max-w-2xl mx-auto sm:mx-0">
                  <div className="h-4 w-full rounded bg-surface-3/70 animate-pulse" />
                  <div className="h-4 w-11/12 rounded bg-surface-3/70 animate-pulse" />
                  <div className="h-4 w-4/5 rounded bg-surface-3/70 animate-pulse" />
                </div>
                {/* Reason skeleton */}
                <div className="h-5 w-3/4 max-w-xl rounded bg-surface-3/70 animate-pulse mx-auto sm:mx-0 pt-1" />
                {/* Actions skeleton */}
                <div className="flex items-center justify-center sm:justify-start gap-6 pt-3">
                  <div className="h-5 w-24 rounded bg-surface-3/70 animate-pulse" />
                  <div className="h-5 w-28 rounded bg-surface-3/70 animate-pulse" />
                </div>
              </div>
            </motion.div>
          ) : item ? (
            <motion.div
              key={`${userId}-${mode}-${item.item_id}`}
              initial={shouldReduceMotion ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={shouldReduceMotion ? undefined : { opacity: 0 }}
              transition={transitionConfig}
              className="flex flex-col sm:flex-row items-center sm:items-start gap-6 sm:gap-8 md:gap-12"
            >
              {/* Left: Large Poster (w500, 2:3, rounded-md, soft shadow) */}
              <div className="relative w-48 sm:w-56 md:w-64 lg:w-72 shrink-0 aspect-[2/3] rounded-md overflow-hidden shadow-soft border border-border/60 bg-surface-2 group">
                {posterUrl ? (
                  <img
                    src={posterUrl}
                    alt={item.title}
                    loading="eager"
                    className="w-full h-full object-cover select-none"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col justify-between p-4 bg-surface-2 text-text-muted">
                    <span className="text-[11px] font-medium tracking-wider uppercase opacity-70">
                      {item.release_year || 'Film'}
                    </span>
                    <div className="my-auto">
                      <span className="text-sm font-semibold text-text-main line-clamp-3 leading-snug">
                        {cleanTitle || item.title}
                      </span>
                    </div>
                    <span className="text-[10px] opacity-60">No poster</span>
                  </div>
                )}

                {/* Top-1 Rank Badge */}
                <div className="absolute top-3 left-3 px-2 py-0.5 rounded-md bg-black/60 backdrop-blur-xs text-white text-xs font-semibold font-numeric border border-white/15 z-10 flex items-center gap-1">
                  <span>#1</span>
                </div>
              </div>

              {/* Right: Movie Details */}
              <div className="flex-1 w-full space-y-3.5 text-center sm:text-left">
                {/* Eyebrow badge */}
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-accent/10 border border-accent/25 text-accent text-[11px] font-semibold tracking-wide uppercase">
                  <Sparkles size={11} />
                  <span>Top Recommendation</span>
                </div>

                {/* Title in serif display font (48-56px on desktop, tight tracking) */}
                <h2 
                  className="text-[34px] sm:text-[42px] md:text-[48px] lg:text-[54px] font-serif font-semibold text-text-main tracking-tight leading-[1.08]"
                  title={item.title}
                >
                  {cleanTitle || item.title}
                </h2>

                {/* Metadata Line: "1952 · Musical, Romance · ★ 4.28" */}
                {metaLine && (
                  <div className="text-sm sm:text-base font-medium text-text-muted flex items-center justify-center sm:justify-start gap-1.5 flex-wrap">
                    <span>{metaLine}</span>
                  </div>
                )}

                {/* TMDB Overview clamped to 3 lines (hidden gracefully if null) */}
                {overview && (
                  <p className="text-sm sm:text-[15px] text-text-muted leading-relaxed line-clamp-3 max-w-2xl">
                    {overview}
                  </p>
                )}

                {/* Reason in one plain human sentence from API reason field */}
                {reason && (
                  <p className="text-sm sm:text-[15px] text-text-main leading-relaxed font-normal max-w-2xl pt-0.5">
                    {reason}
                  </p>
                )}

                {/* Actions: "More like this" Explore link */}
                <div className="pt-2 flex items-center justify-center sm:justify-start">
                  <Link
                    to={`/explore?item=${item.item_id}`}
                    className="text-sm sm:text-base font-medium text-text-muted hover:text-text-main underline underline-offset-4 decoration-border hover:decoration-text-muted transition-colors cursor-pointer inline-flex items-center gap-1.5"
                    title={`Explore films similar to ${cleanTitle || item.title}`}
                  >
                    <Compass size={15} />
                    <span>More like this</span>
                    <span aria-hidden="true">&rarr;</span>
                  </Link>
                </div>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </section>
  );
}
