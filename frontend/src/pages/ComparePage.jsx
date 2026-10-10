import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Dices, Sparkles, AlertCircle, RefreshCw, Film } from 'lucide-react';
import UserCombobox from '../components/UserCombobox';
import ErrorBoundary from '../components/ErrorBoundary';
import { apiUrl } from '../utils/api';

function cleanTitle(raw) {
  if (!raw) return '';
  let clean = raw.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

export default function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // URL state sync
  const [userId, setUserId] = useState(() => Number(searchParams.get('user')) || 303);
  const [k, setK] = useState(() => Number(searchParams.get('k')) || 10);
  const [activeMobileTab, setActiveMobileTab] = useState('two_stage'); // 'popularity' | 'retrieval_only' | 'two_stage'
  const [hoveredMovieId, setHoveredMovieId] = useState(null);

  // Data fetching state
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Precomputed winners state
  const [winningUsers, setWinningUsers] = useState([]);
  const [winningUserIndex, setWinningUserIndex] = useState(0);

  // Fetch winning users list once
  useEffect(() => {
    let isMounted = true;
    async function loadWinningUsers() {
      try {
        const res = await fetch(apiUrl('/api/compare/winning-users'));
        if (res.ok) {
          const list = await res.json();
          if (isMounted && Array.isArray(list) && list.length > 0) {
            setWinningUsers(list);
          }
        }
      } catch (e) {
        console.warn('Could not load precomputed winning users:', e);
      }
    }
    loadWinningUsers();
    return () => { isMounted = false; };
  }, []);

  // Fetch comparison data for current user and k
  const fetchCompareData = useCallback(async (uid, count) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl(`/api/compare/${uid}?k=${count}`));
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error('Failed to load compare data:', err);
      setError(err.message || 'Failed to connect to backend server');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCompareData(userId, k);
    setSearchParams({ user: String(userId), k: String(k) }, { replace: true });
  }, [userId, k, fetchCompareData, setSearchParams]);

  // Handlers
  const handleRandomUser = () => {
    const randomId = Math.floor(Math.random() * 6040) + 1;
    setUserId(randomId);
  };

  const handleTryWinningUser = () => {
    if (winningUsers.length === 0) {
      setUserId(303);
      return;
    }
    const nextIdx = (winningUserIndex + 1) % winningUsers.length;
    setWinningUserIndex(nextIdx);
    setUserId(winningUsers[nextIdx].user_id);
  };

  // Held-out summary sentence computation (strictly from real data)
  const heldOutSummary = useMemo(() => {
    if (!data?.held_out) return null;
    const h = data.held_out;
    const movieTitle = cleanTitle(h.title);

    // Two-stage description
    const tsPart = h.rank_in_two_stage
      ? `ranked #${h.rank_in_two_stage} by Two-stage`
      : `not in the Two-stage top\u00a0${k}`;

    // Retrieval description
    let retPart = '';
    if (h.rank_in_retrieval) {
      retPart = `#${h.rank_in_retrieval} by Retrieval`;
    } else if (h.retrieval_candidate_rank) {
      retPart = `#${h.retrieval_candidate_rank} by Retrieval`;
    } else {
      retPart = `not in the Retrieval top\u00a0${k}`;
    }

    // Popularity description
    const popPart = h.rank_in_popularity
      ? `#${h.rank_in_popularity} by Popularity`
      : `not in the Popularity top\u00a0${k}`;

    return `The film this user actually watched next (${movieTitle}) was ${tsPart}, ${retPart}, and ${popPart}.`;
  }, [data, k]);

  return (
    <ErrorBoundary>
      <div className="w-full">
        {/* Controls row: 1.5rem padding above (pt-6), 1.5rem below (pb-6) */}
        <div className="-mt-8 sm:-mt-10 pt-6 pb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm text-text-muted border-b border-border/40">
          {/* User selector & Winning user button */}
          <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
            <span className="sr-only sm:not-sr-only">Showing comparison for</span>
            <UserCombobox 
              selectedUserId={userId} 
              onSelectUser={setUserId} 
              showRandom={false}
              inline={true}
            />

            <div className="flex flex-col gap-0.5">
              <button
                type="button"
                onClick={handleTryWinningUser}
                className="h-8 px-2.5 bg-accent-subtle hover:bg-accent-subtle/80 border border-accent/30 rounded-lg text-xs font-semibold text-accent transition-colors cursor-pointer inline-flex items-center gap-1.5 shadow-xs"
                title="Pick a user where Two-stage ranks their next watched film higher than Retrieval"
              >
                <Sparkles size={13} className="text-accent shrink-0" />
                <span>Try a user where Two-stage wins</span>
              </button>
              <span className="text-[10px] text-text-muted/80 pl-1 leading-tight">Selected examples, not typical users.</span>
            </div>
          </div>

          {/* Right controls: Random user and Show 5 / 10 */}
          <div className="flex items-center gap-2.5 self-start sm:self-auto">
            <button
              type="button"
              onClick={handleRandomUser}
              className="h-8 px-2.5 bg-surface-2 hover:bg-surface-3 border border-border rounded-lg text-xs font-medium text-text-muted hover:text-text-main transition-colors cursor-pointer inline-flex items-center gap-1.5"
              title="Pick a random user"
            >
              <Dices size={13} />
              <span>Random user</span>
            </button>

            <div className="flex items-center gap-1.5 text-xs text-text-muted">
              <label htmlFor="compare-k-select">Show:</label>
              <select
                id="compare-k-select"
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="h-8 px-2 bg-surface-2 border border-border rounded-lg text-xs font-medium text-text-main outline-none focus:border-accent cursor-pointer"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
              </select>
            </div>
          </div>
        </div>

        {/* Item 4: "Compare" heading (2.25rem, serif), then 8px, then intro sentence */}
        <div style={{ marginTop: '24px' }}>
          <h1 style={{ margin: 0, fontSize: '2.25rem', lineHeight: '2.5rem' }} className="font-serif font-bold tracking-tight text-text-main">
            Compare
          </h1>
          <p style={{ marginTop: '8px', marginBottom: 0 }} className="text-sm text-text-muted">
            The same user, three ways: most popular, nearest by taste, and re-ranked.
          </p>
        </div>

        {/* Item 4: Top of page spacing rhythm:
            Intro sentence, then 24px, then held-out sentence, then 6px, then its muted note, then 40px, then the three column headers */}
        {data?.held_out && (
          <div style={{ marginTop: '24px' }} className="flex items-start gap-2.5">
            <span className="text-accent font-bold text-base shrink-0 leading-normal select-none" aria-hidden="true">
              ✓
            </span>
            <div>
              <p style={{ margin: 0 }} className="text-base font-normal text-text-main leading-snug">
                {heldOutSummary}
              </p>
              <p style={{ marginTop: '6px', marginInline: 0, marginBottom: 0 }} className="text-xs text-text-muted">
                This film was held back from training and ranking, and is shown only as a check.
              </p>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div style={{ marginTop: '24px' }} className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
            <button
              onClick={() => fetchCompareData(userId, k)}
              className="px-3 py-1 bg-surface border border-border rounded-lg text-xs font-semibold hover:bg-surface-2 transition-colors cursor-pointer flex items-center gap-1.5"
            >
              <RefreshCw size={12} />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Mobile Tab Switcher (< 768px) */}
        <div className="md:hidden mt-6 flex border border-border rounded-lg bg-surface-2 p-1 text-xs font-semibold">
          <button
            type="button"
            onClick={() => setActiveMobileTab('popularity')}
            className={`flex-1 py-1.5 rounded-md text-center transition-colors ${
              activeMobileTab === 'popularity'
                ? 'bg-surface text-text-main shadow-xs font-bold'
                : 'text-text-muted hover:text-text-main'
            }`}
          >
            Popularity
          </button>
          <button
            type="button"
            onClick={() => setActiveMobileTab('retrieval_only')}
            className={`flex-1 py-1.5 rounded-md text-center transition-colors ${
              activeMobileTab === 'retrieval_only'
                ? 'bg-surface text-text-main shadow-xs font-bold'
                : 'text-text-muted hover:text-text-main'
            }`}
          >
            Retrieval only
          </button>
          <button
            type="button"
            onClick={() => setActiveMobileTab('two_stage')}
            className={`flex-1 py-1.5 rounded-md text-center transition-colors ${
              activeMobileTab === 'two_stage'
                ? 'bg-accent text-white shadow-xs font-bold'
                : 'text-text-muted hover:text-text-main'
            }`}
          >
            Two-stage
          </button>
        </div>

        {/* Item 4: 40px gap before the three column headers. Equal column widths & 2rem gaps */}
        {loading ? (
          <CompareSkeleton k={k} />
        ) : data ? (
          <div 
            style={{ marginTop: '40px', columnGap: '2rem' }} 
            className="grid grid-cols-1 md:grid-cols-3 items-start w-full"
          >
            {/* Column 1: Popularity */}
            <div className={`flex flex-col w-full min-w-0 ${activeMobileTab !== 'popularity' ? 'hidden md:flex' : 'flex'}`}>
              <ColumnHeader
                title="Popularity"
                description="Most watched by everyone"
                badge="Baseline"
              />
              {/* Item 5: Description to first row: 12px (6px padding-bottom + 6px margin-top) */}
              <div style={{ marginTop: '6px' }} className="divide-y divide-border border-b border-border">
                {data.popularity.map((item) => (
                  <CompareItemRow
                    key={item.movie_id}
                    item={item}
                    isHeldOut={item.movie_id === data.held_out?.movie_id}
                    isHovered={hoveredMovieId === item.movie_id}
                    onHover={setHoveredMovieId}
                    columnType="popularity"
                  />
                ))}
              </div>
              <StatsStrip stats={data.stats.popularity} label="Popularity Stats" />
            </div>

            {/* Column 2: Retrieval only */}
            <div className={`flex flex-col w-full min-w-0 ${activeMobileTab !== 'retrieval_only' ? 'hidden md:flex' : 'flex'}`}>
              <ColumnHeader
                title="Retrieval only"
                description="Closest to this user's taste"
                badge="Stage 1"
              />
              {/* Item 5: Description to first row: 12px (6px padding-bottom + 6px margin-top) */}
              <div style={{ marginTop: '6px' }} className="divide-y divide-border border-b border-border">
                {data.retrieval_only.map((item) => (
                  <CompareItemRow
                    key={item.movie_id}
                    item={item}
                    isHeldOut={item.movie_id === data.held_out?.movie_id}
                    isHovered={hoveredMovieId === item.movie_id}
                    onHover={setHoveredMovieId}
                    columnType="retrieval"
                  />
                ))}
              </div>
              <StatsStrip stats={data.stats.retrieval_only} label="Retrieval Stats" />
            </div>

            {/* Column 3: Two-stage */}
            <div className={`flex flex-col w-full min-w-0 ${activeMobileTab !== 'two_stage' ? 'hidden md:flex' : 'flex'}`}>
              <ColumnHeader
                title="Two-stage"
                description="Re-ordered by a second model"
                badge="Stage 1 + 2"
                highlightBadge={true}
              />
              {/* Item 5: Description to first row: 12px (6px padding-bottom + 6px margin-top) */}
              <div style={{ marginTop: '6px' }} className="divide-y divide-border border-b border-border">
                {data.two_stage.map((item) => (
                  <CompareItemRow
                    key={item.movie_id}
                    item={item}
                    isHeldOut={item.movie_id === data.held_out?.movie_id}
                    isHovered={hoveredMovieId === item.movie_id}
                    onHover={setHoveredMovieId}
                    columnType="two_stage"
                    showMovement={true}
                  />
                ))}
              </div>
              <StatsStrip stats={data.stats.two_stage} label="Two-stage Stats" />
            </div>
          </div>
        ) : null}
      </div>
    </ErrorBoundary>
  );
}

// ---------------------------------------------------------------------------
// Column Header Component
// Item 5: Column header to description 4px
// ---------------------------------------------------------------------------
function ColumnHeader({ title, description, badge, highlightBadge = false }) {
  return (
    <div style={{ paddingBottom: '6px', borderBottom: '1px solid var(--border)' }}>
      <div className="flex items-center justify-between gap-2">
        <h2 style={{ margin: 0 }} className="font-serif text-xl sm:text-2xl font-bold tracking-tight text-text-main leading-tight">
          {title}
        </h2>
        <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${
          highlightBadge 
            ? 'bg-accent text-white border-accent'
            : 'bg-surface-2 text-text-muted border-border'
        }`}>
          {badge}
        </span>
      </div>
      <p style={{ marginTop: '4px', marginInline: 0, marginBottom: 0 }} className="text-xs text-text-muted leading-snug">
        {description}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Compare Item Row Component
// Item 1, 2, 3: Strict inline styles for layout, sizes, gaps and font sizes
// ---------------------------------------------------------------------------
function CompareItemRow({ item, isHeldOut, isHovered, onHover, columnType, showMovement = false }) {
  const title = cleanTitle(item.title);
  const posterSrc = item.poster_url || item.poster_url_large;
  const genres = item.genres?.slice(0, 2).join(', ') || '';

  // Movement marker calculation relative to retrieval list / candidate pool
  let movementMarker = null;
  if (showMovement) {
    const candRank = item.retrieval_candidate_rank;
    if (candRank === null || candRank === undefined) {
      movementMarker = { label: 'new', type: 'new' };
    } else if (candRank > item.rank) {
      movementMarker = { label: `↑ from #${candRank}`, type: 'up' };
    } else if (candRank < item.rank) {
      movementMarker = { label: `↓ from #${candRank}`, type: 'down' };
    } else {
      movementMarker = { label: '= unchanged', type: 'same' };
    }
  }

  return (
    <div
      onMouseEnter={() => onHover(item.movie_id)}
      onMouseLeave={() => onHover(null)}
      /* Item 1 & 3: row container display:flex; align-items:center; column-gap: 1.25rem; vertical padding 14px; fixed height 136px */
      style={{
        display: 'flex',
        alignItems: 'center',
        columnGap: '1.25rem',
        paddingTop: '14px',
        paddingBottom: '14px',
        height: '136px',
        minHeight: '136px',
        maxHeight: '136px',
        boxSizing: 'border-box',
        background: 'transparent',
      }}
      className="group relative cursor-pointer"
    >
      {/* Item 3: Poster 72px wide (2:3, so 108px tall) */}
      <div 
        style={{
          width: '72px',
          minWidth: '72px',
          maxWidth: '72px',
          height: '108px',
          minHeight: '108px',
          maxHeight: '108px',
          position: 'relative',
          borderRadius: '6px',
          overflow: 'hidden',
          flexShrink: 0,
          backgroundColor: 'var(--surface-3)',
          userSelect: 'none',
        }}
      >
        {posterSrc ? (
          <img
            src={posterSrc}
            alt={title}
            loading="lazy"
            decoding="async"
            style={{ width: '100%', height: '108px', objectFit: 'cover', display: 'block' }}
          />
        ) : (
          <div className="w-full h-full flex flex-col justify-center items-center p-1 text-center bg-surface-2 text-text-muted">
            <Film size={16} className="opacity-40 mb-1" />
            <span className="text-[9px] line-clamp-2 leading-tight">{title}</span>
          </div>
        )}

        {/* Soft bottom scrim behind numeral */}
        <div 
          style={{
            position: 'absolute',
            inset: 0,
            top: 'auto',
            height: '42%',
            background: 'linear-gradient(to top, rgba(0, 0, 0, 0.75) 0%, transparent 100%)',
            pointerEvents: 'none',
            zIndex: 10,
          }}
          aria-hidden="true"
        />

        {/* Rank Numeral inside bottom-left corner */}
        <div 
          style={{
            position: 'absolute',
            bottom: '4px',
            left: '6px',
            zIndex: 20,
            fontFamily: 'var(--font-serif)',
            fontStyle: 'italic',
            fontWeight: 'bold',
            fontSize: '18px',
            lineHeight: 1,
            color: '#ffffff',
            textShadow: '0 1px 4px rgba(0, 0, 0, 0.8)',
            pointerEvents: 'none',
          }}
          aria-hidden="true"
        >
          {item.rank}
        </div>

        {/* Held-Out Badge on Poster */}
        {isHeldOut && (
          <div 
            style={{
              position: 'absolute',
              top: '3px',
              left: '3px',
              right: '3px',
              zIndex: 30,
              backgroundColor: '#D94F30',
              color: '#ffffff',
              fontSize: '7.5px',
              fontWeight: 'bold',
              padding: '1.5px 2px',
              borderRadius: '2.5px',
              textAlign: 'center',
              lineHeight: 1.1,
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
            }}
            title="User actually watched this film next in held-out test split"
          >
            ✓ Watched next
          </div>
        )}
      </div>

      {/* Item 1: Text block with flex-1 and min-width: 0 */}
      <div 
        style={{
          flex: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          overflow: 'hidden',
        }}
      >
        {/* Item 2: Title about 20px, serif, one line with ellipsis, ink-colored */}
        <h3 
          title={title}
          style={{
            fontFamily: 'var(--font-serif)',
            fontSize: '20px',
            lineHeight: '24px',
            color: 'var(--text)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            minWidth: 0,
            margin: 0,
          }}
          className={`transition-colors ${
            isHovered 
              ? 'underline decoration-accent underline-offset-2' 
              : 'group-hover:underline group-hover:decoration-accent group-hover:underline-offset-2'
          }`}
        >
          {title}
        </h3>

        {/* Item 2: Movement marker about 14px, weight 500, accent for up, muted for down or new. Sits 0.125rem under title */}
        {movementMarker ? (
          <p 
            style={{
              fontSize: '14px',
              fontWeight: 500,
              lineHeight: '18px',
              marginTop: '2px',
              marginBottom: 0,
              color: movementMarker.type === 'up' ? 'var(--accent)' : 'var(--text-muted)',
            }}
          >
            {movementMarker.label}
          </p>
        ) : (
          <div style={{ height: '0px' }} />
        )}

        {/* Item 2: Genre line about 15px, muted */}
        <p 
          style={{
            fontSize: '15px',
            lineHeight: '19px',
            marginTop: '3px',
            marginBottom: 0,
            color: 'var(--text-muted)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {item.year}{genres ? ` · ${genres}` : ''}
        </p>

        {/* Item 2: Stat line about 15px, muted */}
        {columnType === 'popularity' && item.watch_count ? (
          <p 
            style={{
              fontSize: '15px',
              lineHeight: '19px',
              marginTop: '3px',
              marginBottom: 0,
              color: 'var(--text-muted)',
            }}
          >
            Watched by {item.watch_count.toLocaleString()} users
          </p>
        ) : columnType === 'retrieval' ? (
          <p 
            style={{
              fontSize: '15px',
              lineHeight: '19px',
              marginTop: '3px',
              marginBottom: 0,
              color: 'var(--text-muted)',
            }}
          >
            Taste match {Math.round(item.score * 100)}%
          </p>
        ) : columnType === 'two_stage' ? (
          <p 
            style={{
              fontSize: '15px',
              lineHeight: '19px',
              marginTop: '3px',
              marginBottom: 0,
              color: 'var(--text-muted)',
            }}
          >
            Match score {item.score.toFixed(2)}
          </p>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats Strip Component (Real computed metrics per list)
// ---------------------------------------------------------------------------
function StatsStrip({ stats, label }) {
  if (!stats) return null;

  return (
    <div style={{ marginTop: '16px' }} className="p-3.5 rounded-xl border border-border/60 bg-surface-2/70 text-xs">
      <div className="flex items-center justify-between mb-2.5 pb-1.5 border-b border-border/40">
        <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">
          {label}
        </span>
        <span className="text-[11px] font-semibold text-text-main">
          {stats.latency_ms.toFixed(1)} ms
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="space-y-0.5">
          <span className="text-text-muted block text-[10px]">Avg Pop Rank</span>
          <span className="font-semibold text-text-main">
            #{stats.avg_popularity_rank.toFixed(1)}
          </span>
        </div>

        <div className="space-y-0.5">
          <span className="text-text-muted block text-[10px]">Genre Diversity</span>
          <span className="font-semibold text-text-main">
            {(stats.genre_diversity * 100).toFixed(1)}%
          </span>
        </div>

        <div className="space-y-0.5">
          <span className="text-text-muted block text-[10px]">Long-Tail Share</span>
          <span className="font-semibold text-text-main">
            {(stats.long_tail_share * 100).toFixed(1)}%
          </span>
        </div>

        <div className="space-y-0.5">
          <span className="text-text-muted block text-[10px]">Stage Latency</span>
          <span className="font-semibold text-text-main">
            {stats.latency_ms.toFixed(1)} ms
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeletons matching exact row & column dimensions (136px height, 72x108 poster)
// ---------------------------------------------------------------------------
function CompareSkeleton({ k = 10 }) {
  return (
    <div style={{ marginTop: '40px', columnGap: '2rem' }} className="grid grid-cols-1 md:grid-cols-3 items-start w-full animate-pulse">
      {[1, 2, 3].map((col) => (
        <div key={col} className="w-full min-w-0">
          <div style={{ paddingBottom: '8px', borderBottom: '1px solid var(--border)' }} className="space-y-1.5">
            <div className="h-6 w-32 bg-surface-2 rounded-md" />
            <div className="h-3 w-48 bg-surface-2 rounded-md" />
          </div>
          <div style={{ marginTop: '12px' }} className="divide-y divide-border border-b border-border">
            {Array.from({ length: k }).map((_, i) => (
              <div 
                key={i} 
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  columnGap: '1.25rem',
                  paddingTop: '14px',
                  paddingBottom: '14px',
                  height: '136px',
                  boxSizing: 'border-box'
                }}
              >
                <div style={{ width: '72px', height: '108px', flexShrink: 0 }} className="rounded-md bg-surface-2" />
                <div className="flex-1 min-w-0 space-y-2">
                  <div className="h-5 w-3/4 bg-surface-2 rounded" />
                  <div className="h-4 w-1/2 bg-surface-2 rounded" />
                  <div className="h-4 w-1/3 bg-surface-2 rounded" />
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '16px' }} className="p-3.5 rounded-xl border border-border/60 bg-surface-2/70 h-24" />
        </div>
      ))}
    </div>
  );
}
