import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search, X, Clock, ArrowLeft } from 'lucide-react';
import { formatRating } from '../utils/formatters';

function cleanMovieTitle(title) {
  if (!title) return '';
  let clean = title.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

export default function ExplorePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialItemId = searchParams.get('item');

  const [query, setQuery] = useState('Star');
  const [results, setResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);

  const [selectedMovie, setSelectedMovie] = useState(null);
  const [similarItems, setSimilarItems] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarLatency, setSimilarLatency] = useState(null);

  const splitViewRef = useRef(null);

  // Load similar films for a given movie
  const loadSimilarForMovie = useCallback(async (movie, shouldScroll = false) => {
    if (!movie?.item_id) return;
    setSelectedMovie(movie);
    setSimilarLoading(true);

    try {
      const res = await fetch(`/api/similar/${movie.item_id}?n=10`);
      if (res.ok) {
        const data = await res.json();
        // Enrich selected movie if data has more details
        setSelectedMovie((prev) => ({
          ...prev,
          ...movie,
          overview: data?.overview || movie.overview,
          backdrop_url: data?.backdrop_url || movie.backdrop_url,
          poster_url_large: data?.poster_url_large || movie.poster_url_large,
          poster_url: data?.poster_url || movie.poster_url,
        }));
        setSimilarItems(data?.similar_items ?? []);
        setSimilarLatency(data?.query_ms ?? null);
      } else {
        setSimilarItems([]);
      }
    } catch (e) {
      console.error("Load similar failed:", e);
      setSimilarItems([]);
    } finally {
      setSimilarLoading(false);
      if (shouldScroll && splitViewRef.current) {
        splitViewRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, []);

  const loadItemById = useCallback(async (itemId) => {
    try {
      const res = await fetch(`/api/similar/${itemId}?n=10`);
      if (res.ok) {
        const data = await res.json();
        setSelectedMovie({
          item_id: data?.item_id ?? itemId,
          title: data?.title ?? 'Selected Movie',
          genres: data?.genres ?? [],
          poster_url: data?.poster_url ?? null,
          poster_url_large: data?.poster_url_large ?? null,
          backdrop_url: data?.backdrop_url ?? null,
          overview: data?.overview ?? null,
          release_year: data?.release_year ?? null,
          avg_rating: data?.avg_rating ?? null,
          popularity_rank: data?.popularity_rank ?? null,
        });
        setSimilarItems(data?.similar_items ?? []);
        setSimilarLatency(data?.query_ms ?? null);
      }
    } catch (e) {
      console.error("Load item failed:", e);
    }
  }, []);

  const executeSearch = useCallback(async (q) => {
    if (!q || q.trim().length === 0) {
      setResults([]);
      return;
    }
    setSearchLoading(true);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=15`);
      if (res.ok) {
        const data = await res.json();
        const searchResults = data?.results ?? [];
        setResults(searchResults);
        // Default select first movie if none selected yet
        if (searchResults.length > 0 && !selectedMovie) {
          loadSimilarForMovie(searchResults[0]);
        }
      } else {
        setResults([]);
      }
    } catch (e) {
      console.error("Search failed:", e);
      setResults([]);
    } finally {
      setSearchLoading(false);
    }
  }, [selectedMovie, loadSimilarForMovie]);

  // Initial load
  useEffect(() => {
    if (initialItemId) {
      loadItemById(initialItemId);
    } else {
      executeSearch('Star');
    }
  }, [initialItemId, loadItemById, executeSearch]);

  // Debounced search
  useEffect(() => {
    const handler = setTimeout(() => {
      if (query.trim().length >= 2) {
        executeSearch(query.trim());
      }
    }, 250);
    return () => clearTimeout(handler);
  }, [query, executeSearch]);

  const handleSelectMovie = (movie) => {
    loadSimilarForMovie(movie, true);
  };

  const handleClearSelection = () => {
    setSelectedMovie(null);
    setSimilarItems([]);
    setSearchParams({});
  };

  return (
    <div>
      {/* 1. Large Serif Heading & Editorial Intro */}
      <div>
        <h1 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-semibold text-text-main tracking-tight">
          Explore the catalog
        </h1>
        <p className="text-sm sm:text-base text-text-muted mt-1 max-w-2xl leading-relaxed">
          Search across 3,883 films and inspect latent vector neighbors queried in real time via FAISS vector index.
        </p>
      </div>

      {/* 2. Oversized Underlined Search Input */}
      <div className="mt-6 relative border-b-2 border-border/80 focus-within:border-accent transition-colors pb-1">
        <div className="flex items-center gap-3">
          <Search size={24} className="text-text-muted/60 flex-shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search movie title (e.g. Matrix, Toy Story, Star Wars)..."
            className="w-full bg-transparent text-xl sm:text-2xl md:text-3xl font-serif text-text-main placeholder:text-text-muted/40 outline-none py-2"
          />
          {query && (
            <button
              type="button"
              onClick={() => {
                setQuery('');
                setResults([]);
              }}
              className="text-text-muted hover:text-text-main p-1 transition-colors cursor-pointer"
              aria-label="Clear search"
            >
              <X size={20} />
            </button>
          )}
        </div>
      </div>

      {/* 3. Selected Movie Split View (When Selected) */}
      {selectedMovie && (
        <section
          ref={splitViewRef}
          aria-labelledby="selected-film-title"
          className="mt-8 sm:mt-10 space-y-6"
        >
          {/* Back / Deselect Link */}
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={handleClearSelection}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-text-muted hover:text-text-main transition-colors cursor-pointer"
            >
              <ArrowLeft size={14} />
              <span>Back to all results</span>
            </button>

            {similarLatency != null && (
              <span className="text-xs text-text-muted font-numeric">
                FAISS latency: {similarLatency} ms
              </span>
            )}
          </div>

          {/* Split View: Big Poster Left, Editorial Meta Right */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 lg:gap-12 items-start">
            {/* Left: Big Poster (w500, 2:3) */}
            <div className="md:col-span-4 lg:col-span-3">
              <div className="w-full aspect-[2/3] rounded-lg overflow-hidden bg-surface-2 shadow-soft border border-border/40">
                {selectedMovie.poster_url_large || selectedMovie.poster_url ? (
                  <img
                    src={selectedMovie.poster_url_large || selectedMovie.poster_url}
                    alt={cleanMovieTitle(selectedMovie.title)}
                    loading="eager"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col justify-between p-4 bg-surface-2 text-text-muted">
                    <span className="text-xs uppercase tracking-wider">{selectedMovie.release_year || 'Film'}</span>
                    <span className="text-base font-semibold text-text-main line-clamp-3">{cleanMovieTitle(selectedMovie.title)}</span>
                    <span className="text-xs opacity-60">No poster available</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Title, Meta, and TMDB Overview */}
            <div className="md:col-span-8 lg:col-span-9 space-y-4">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-accent font-numeric">
                  Selected Film
                </span>
                <h2
                  id="selected-film-title"
                  className="font-serif text-3xl sm:text-4xl lg:text-5xl font-semibold text-text-main tracking-tight leading-tight mt-1"
                >
                  {cleanMovieTitle(selectedMovie.title)}
                </h2>
              </div>

              {/* Single Meta Line: Year · Genres · Rating */}
              <p className="text-sm sm:text-base text-text-muted font-medium">
                {selectedMovie.release_year && <span>{selectedMovie.release_year}</span>}
                {(selectedMovie.genres ?? []).length > 0 && (
                  <span> · {(selectedMovie.genres ?? []).join(', ')}</span>
                )}
                {selectedMovie.avg_rating != null && (
                  <span> · ★ {formatRating(selectedMovie.avg_rating)}</span>
                )}
                {selectedMovie.popularity_rank != null && (
                  <span className="font-numeric text-xs text-text-muted/80"> · Pop #{selectedMovie.popularity_rank}</span>
                )}
              </p>

              {/* TMDB Overview */}
              {selectedMovie.overview ? (
                <p className="text-sm sm:text-base text-text-main/90 leading-relaxed max-w-2xl pt-2">
                  {selectedMovie.overview}
                </p>
              ) : (
                <p className="text-sm text-text-muted italic pt-2">
                  Classic catalog title indexed in the MovieLens 1M embedding space.
                </p>
              )}
            </div>
          </div>

          {/* 4. Similar Films Poster Grid */}
          <div className="mt-8 sm:mt-10 pt-6 border-t border-border/60 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-serif text-2xl font-semibold text-text-main tracking-tight">
                  Similar films
                </h3>
                <p className="text-xs sm:text-sm text-text-muted mt-1">
                  Top 10 nearest vector neighbors in the Two-Tower latent space.
                </p>
              </div>
            </div>

            {similarLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-7 lg:gap-8 animate-pulse">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="space-y-2">
                    <div className="w-full aspect-[2/3] bg-surface-3/70 rounded-lg" />
                    <div className="h-4 w-3/4 bg-surface-3/70 rounded" />
                    <div className="h-3 w-1/2 bg-surface-3/70 rounded" />
                  </div>
                ))}
              </div>
            ) : (similarItems ?? []).length === 0 ? (
              <div className="py-8 text-sm text-text-muted">
                No similar films found.
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-7 lg:gap-8">
                {(similarItems ?? []).map((item) => {
                  const displayTitle = cleanMovieTitle(item.title);
                  const genres = item.genres?.slice(0, 2).join(', ') || '';
                  const metaLine = item.release_year 
                    ? `${item.release_year}${genres ? ` · ${genres}` : ''}`
                    : genres;

                  return (
                    <div
                      key={item.item_id}
                      onClick={() => handleSelectMovie(item)}
                      className="group cursor-pointer flex flex-col select-none"
                    >
                      {/* Poster (lifts 4px on hover, no card borders) */}
                      <div className="w-full aspect-[2/3] rounded-lg overflow-hidden bg-surface-2 shadow-soft border border-border/40 transition-transform duration-200 group-hover:-translate-y-1">
                        {item.poster_url || item.poster_url_large ? (
                          <img
                            src={item.poster_url || item.poster_url_large}
                            alt={displayTitle}
                            loading="eager"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex flex-col justify-between p-3 bg-surface-2 text-text-muted">
                            <span className="text-[10px] uppercase opacity-70">{item.release_year || 'Film'}</span>
                            <span className="text-xs font-semibold text-text-main line-clamp-3">{displayTitle}</span>
                            <span className="text-[10px] opacity-60">No poster</span>
                          </div>
                        )}
                      </div>

                      {/* Title & Metadata */}
                      <div className="pt-2">
                        <h4
                          className="text-sm font-semibold text-text-main line-clamp-1 group-hover:text-accent transition-colors"
                          title={item.title}
                        >
                          {displayTitle}
                        </h4>
                        <div className="text-xs text-text-muted mt-0.5 line-clamp-1">
                          {metaLine || 'Film'}
                        </div>
                        <div className="text-xs font-semibold text-text-main font-numeric flex items-center gap-1 mt-0.5">
                          <span className="text-amber-400">★</span>
                          <span>{formatRating(item.avg_rating)}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>
      )}

      {/* 5. Search Results Poster Grid (Shown below or when browsing) */}
      <section aria-labelledby="search-results-heading" className="mt-8 sm:mt-10 pt-6 border-t border-border/60 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 id="search-results-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
              {query ? `Search results for "${query}"` : 'Browse catalog'}
            </h2>
            <p className="text-xs sm:text-sm text-text-muted mt-1">
              {results.length} {results.length === 1 ? 'film' : 'films'} found
            </p>
          </div>
        </div>

        {searchLoading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-7 lg:gap-8 animate-pulse">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
              <div key={i} className="space-y-2">
                <div className="w-full aspect-[2/3] bg-surface-3/70 rounded-lg" />
                <div className="h-4 w-3/4 bg-surface-3/70 rounded" />
                <div className="h-3 w-1/2 bg-surface-3/70 rounded" />
              </div>
            ))}
          </div>
        ) : results.length === 0 ? (
          <div className="py-8 text-center text-sm text-text-muted">
            No movies matched "{query}". Try another title like "Godfather", "Matrix", or "Toy Story".
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-7 lg:gap-8">
            {results.map((movie) => {
              const displayTitle = cleanMovieTitle(movie.title);
              const genres = movie.genres?.slice(0, 2).join(', ') || '';
              const metaLine = movie.release_year 
                ? `${movie.release_year}${genres ? ` · ${genres}` : ''}`
                : genres;
              const isSelected = selectedMovie?.item_id === movie.item_id;

              return (
                <div
                  key={movie.item_id}
                  onClick={() => handleSelectMovie(movie)}
                  className={`group cursor-pointer flex flex-col select-none ${
                    isSelected ? 'opacity-100' : 'hover:opacity-100'
                  }`}
                >
                  {/* Poster (lifts 4px on hover, no card borders) */}
                  <div className={`w-full aspect-[2/3] rounded-lg overflow-hidden bg-surface-2 shadow-soft transition-all duration-200 group-hover:-translate-y-1 ${
                    isSelected ? 'ring-2 ring-accent' : 'border border-border/40'
                  }`}>
                    {movie.poster_url || movie.poster_url_large ? (
                      <img
                        src={movie.poster_url || movie.poster_url_large}
                        alt={displayTitle}
                        loading="eager"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col justify-between p-3 bg-surface-2 text-text-muted">
                        <span className="text-[10px] uppercase opacity-70">{movie.release_year || 'Film'}</span>
                        <span className="text-xs font-semibold text-text-main line-clamp-3">{displayTitle}</span>
                        <span className="text-[10px] opacity-60">No poster</span>
                      </div>
                    )}
                  </div>

                  {/* Title & Metadata */}
                  <div className="pt-2">
                    <h3
                      className={`text-sm font-semibold line-clamp-1 transition-colors ${
                        isSelected ? 'text-accent' : 'text-text-main group-hover:text-accent'
                      }`}
                      title={movie.title}
                    >
                      {displayTitle}
                    </h3>
                    <div className="text-xs text-text-muted mt-0.5 line-clamp-1">
                      {metaLine || 'Film'}
                    </div>
                    <div className="text-xs font-semibold text-text-main font-numeric flex items-center gap-1 mt-0.5">
                      <span className="text-amber-400">★</span>
                      <span>{formatRating(movie.avg_rating)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
