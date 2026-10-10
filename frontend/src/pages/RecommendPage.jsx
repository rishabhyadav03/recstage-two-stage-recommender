import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { RefreshCw, AlertCircle, Dices } from 'lucide-react';
import UserCombobox from '../components/UserCombobox';
import UserSummaryCard from '../components/UserSummaryCard';
import RecentlyWatchedSection from '../components/RecentlyWatchedSection';
import RecommendCard from '../components/RecommendCard';
import RecommendHero from '../components/RecommendHero';
import EmbeddingMap from '../components/EmbeddingMap';
import ClosestTasteList from '../components/ClosestTasteList';
import ErrorBoundary from '../components/ErrorBoundary';
import { formatLatency } from '../utils/formatters';
import { apiUrl } from '../utils/api';

export default function RecommendPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // State
  const [userId, setUserId] = useState(() => Number(searchParams.get('user')) || 1);
  const [mode, setMode] = useState(() => searchParams.get('mode') || 'two_stage'); // 'two_stage' | 'retrieval_only' | 'popularity'
  const [topN, setTopN] = useState(() => Number(searchParams.get('topN')) || 10);

  const [userProfile, setUserProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const [mapUserState, setMapUserState] = useState(null);
  const [mapUserLoading, setMapUserLoading] = useState(false);
  const [highlightMovieId, setHighlightMovieId] = useState(null);

  const [recsData, setRecsData] = useState(null);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recsError, setRecsError] = useState(null);

  const topMovie = recsData?.recommendations?.[0] || null;
  const isHeroLoading = recsLoading || (!recsData && !recsError);
  // Skip rank #1 in the grid if already spotlighted in the hero
  const gridItems = topMovie && recsData?.recommendations 
    ? recsData.recommendations.slice(1) 
    : (recsData?.recommendations || []);

  const fetchUserProfile = useCallback(async (id) => {
    setProfileLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/users/${id}`));
      if (res.ok) {
        const data = await res.json();
        setUserProfile(data);
      }
    } catch (err) {
      console.error("Failed to load user profile:", err);
    } finally {
      setProfileLoading(false);
    }
  }, []);

  const fetchMapUserState = useCallback(async (id) => {
    setMapUserLoading(true);
    try {
      const res = await fetch(apiUrl(`/api/map/user/${id}`));
      if (res.ok) {
        const data = await res.json();
        setMapUserState(data);
      }
    } catch (err) {
      console.error("Failed to load user map state:", err);
    } finally {
      setMapUserLoading(false);
    }
  }, []);

  const fetchRecommendations = useCallback(async (id, m, n) => {
    setRecsLoading(true);
    setRecsError(null);
    try {
      const res = await fetch(apiUrl(`/api/recommend/${id}?n=${n}&mode=${m}`));
      if (res.ok) {
        const data = await res.json();
        setRecsData(data);
      } else {
        const errData = await res.json().catch(() => ({}));
        setRecsError(errData.detail || `Server returned ${res.status}`);
      }
    } catch (err) {
      console.error("Failed to load recommendations:", err);
      setRecsError("Failed to reach recommendation API. Make sure the backend server is running.");
    } finally {
      setRecsLoading(false);
    }
  }, []);

  // Fetch user profile and map state on userId or mode change
  useEffect(() => {
    fetchUserProfile(userId);
    fetchMapUserState(userId);
  }, [userId, mode, fetchUserProfile, fetchMapUserState]);

  // Fetch recommendations on userId, mode, or topN change
  useEffect(() => {
    if (mode === 'popularity') {
      fetchRecommendations(userId, 'retrieval_only', topN);
    } else {
      fetchRecommendations(userId, mode, topN);
    }
  }, [userId, mode, topN, fetchRecommendations]);

  const handleExploreSimilar = (movie) => {
    navigate(`/explore?item=${movie.item_id}`);
  };

  return (
    <div>
      {/* 2. Controls line directly under header (1.5rem above and below, no box or border) */}
      <div className="-mt-8 sm:-mt-10 pt-6 pb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm text-text-muted">
        {/* Left / Row 1 on mobile: User select and Mode */}
        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          <span className="sr-only sm:not-sr-only">Showing recommendations for</span>
          <UserCombobox 
            selectedUserId={userId} 
            onSelectUser={setUserId} 
            showRandom={false}
            inline={true}
          />
          <span className="sr-only sm:not-sr-only">using</span>
          <div 
            className="inline-flex p-0.5 bg-surface-2 border border-border rounded-lg text-xs font-medium"
            role="radiogroup"
            aria-label="Recommendation mode"
          >
            <button
              type="button"
              role="radio"
              aria-checked={mode === 'popularity'}
              onClick={() => setMode('popularity')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                mode === 'popularity'
                  ? 'bg-surface text-text-main font-semibold shadow-xs border border-border'
                  : 'text-text-muted hover:text-text-main'
              }`}
              title="Global non-personalized baseline"
            >
              Popularity
            </button>

            <button
              type="button"
              role="radio"
              aria-checked={mode === 'retrieval_only'}
              onClick={() => setMode('retrieval_only')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                mode === 'retrieval_only'
                  ? 'bg-surface text-text-main font-semibold shadow-xs border border-border'
                  : 'text-text-muted hover:text-text-main'
              }`}
              title="FAISS top-candidate vector search"
            >
              Retrieval
            </button>

            <button
              type="button"
              role="radio"
              aria-checked={mode === 'two_stage'}
              onClick={() => setMode('two_stage')}
              className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                mode === 'two_stage'
                  ? 'bg-accent text-white font-semibold shadow-xs'
                  : 'text-text-muted hover:text-text-main'
              }`}
              title="Two-Tower retrieval followed by LightGBM LambdaRank re-ranking"
            >
              Two-stage
            </button>
          </div>
        </div>

        {/* Right / Row 2 on mobile: Random user and Show 10 / 20 select */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          <button
            type="button"
            onClick={() => {
              const randomId = Math.floor(Math.random() * 6040) + 1;
              setUserId(randomId);
            }}
            className="h-8 px-2.5 bg-surface-2 hover:bg-surface-3 border border-border rounded-lg text-xs font-medium text-text-muted hover:text-text-main transition-colors cursor-pointer inline-flex items-center gap-1.5"
            title="Pick a random user"
          >
            <Dices size={13} />
            <span>Random user</span>
          </button>

          <div className="flex items-center gap-1.5 text-xs text-text-muted">
            <label htmlFor="topn-select">Show:</label>
            <select
              id="topn-select"
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="h-8 px-2 bg-surface-2 border border-border rounded-lg text-xs font-medium text-text-main outline-none focus:border-accent cursor-pointer"
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
            </select>
          </div>
        </div>
      </div>

      {/* 3. Hero Section for #1 Recommendation */}
      <RecommendHero
        item={topMovie}
        loading={isHeroLoading}
        userId={userId}
        mode={mode}
      />

      {mode === 'popularity' && (
        <div className="mt-5 p-3.5 bg-surface-2 border border-border rounded-xl text-xs text-text-muted flex items-center gap-2">
          <AlertCircle size={15} className="text-accent flex-shrink-0" />
          <span>
            <strong>Note:</strong> Showing baseline catalog rankings. The production Two-Tower + LambdaRank models demonstrate significant recall and NDCG gains over this baseline (detailed on the Results page).
          </span>
        </div>
      )}

      {/* 4. Recommended For You Poster Grid Section */}
      <section aria-labelledby="recommended-heading" className="mt-8 sm:mt-10 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 id="recommended-heading" className="text-xl font-semibold text-text-main tracking-tight">
              Recommended for you
            </h2>
            <p className="text-sm text-text-muted mt-1">
              Picked from 200 candidates and re-ordered for you.
            </p>
          </div>
        </div>

        {/* Error State */}
        {recsError && (
          <div className="p-6 bg-surface border border-red-500/30 rounded-xl text-center space-y-3">
            <p className="text-sm text-red-500">{recsError}</p>
            <button
              onClick={() => fetchRecommendations(userId, mode === 'popularity' ? 'retrieval_only' : mode, topN)}
              className="px-4 py-2 bg-surface-2 border border-border rounded-lg text-xs font-medium text-text-main hover:bg-surface-3 transition-colors inline-flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw size={14} />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Loading Skeletons: 2 cols mobile, 3 cols tablet, 5 cols desktop, matching poster shape */}
        {recsLoading && !recsError && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-7 lg:gap-8">
            {Array.from({ length: Math.max(1, topN - (topMovie ? 1 : 0)) }).map((_, i) => (
              <div key={i} className="flex flex-col animate-pulse">
                {/* Poster Skeleton */}
                <div className="w-full aspect-[2/3] rounded-lg bg-surface-3/70 shadow-soft border border-border/40" />
                {/* Title Skeleton */}
                <div className="h-4 w-4/5 bg-surface-3/70 rounded mt-2.5" />
                {/* Meta Line Skeleton */}
                <div className="h-3 w-3/5 bg-surface-3/70 rounded mt-1" />
                {/* Rating Skeleton */}
                <div className="h-3 w-1/4 bg-surface-3/70 rounded mt-1" />
              </div>
            ))}
          </div>
        )}

        {/* Results Grid: 2 cols on mobile, 3 on tablet, 5 on desktop, 24-32px gaps, skipping rank #1 */}
        {!recsLoading && !recsError && gridItems.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-7 lg:gap-8">
            {gridItems.map((item, index) => (
              <RecommendCard
                key={item.item_id}
                item={item}
                rank={topMovie ? index + 2 : index + 1}
                animationIndex={index}
                onExploreSimilar={handleExploreSimilar}
              />
            ))}
          </div>
        )}
      </section>

      {/* 4. Taste Profile (User Summary Editorial Block & Embedding Map) */}
      <section aria-labelledby="user-profile-heading" className="mt-8 sm:mt-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 items-start">
          {/* Left Column container on desktop, unrolls via contents on mobile */}
          <div id="taste-left-column" className="contents lg:block lg:col-span-5">
            <div className="order-1">
              <UserSummaryCard userProfile={userProfile} loading={profileLoading} />
            </div>
            <div className="order-3 mt-8">
              <ClosestTasteList
                items={mapUserState?.top_retrieved}
                loading={mapUserLoading}
                onHover={setHighlightMovieId}
              />
            </div>
          </div>

          {/* Right Column container */}
          <div id="taste-right-column" className="order-2 lg:col-span-7">
            <ErrorBoundary fallback={<div className="p-4 text-xs text-text-muted">Embedding map could not load.</div>}>
              <EmbeddingMap
                userId={userId}
                mode={mode}
                userState={mapUserState}
                highlightMovieId={highlightMovieId}
              />
            </ErrorBoundary>
          </div>
        </div>
      </section>

      {/* 5. History Rail (Because You Watched) */}
      <section aria-labelledby="because-watched-heading" className="mt-8 sm:mt-10">
        <RecentlyWatchedSection watchHistory={userProfile?.watch_history} loading={profileLoading} />
      </section>

      {/* 6. Pipeline Note at very bottom */}
      {recsData && !recsLoading && (
        <div className="mt-8 sm:mt-10 pt-4 border-t border-border text-xs text-text-muted">
          <p>
            200 candidates retrieved in {Number(recsData.retrieval_ms).toFixed(1)} ms, ranked in {(mode === 'two_stage' ? Number(recsData.ranking_ms) : 0).toFixed(1)} ms. {Number(recsData.total_ms).toFixed(1)} ms end to end.
          </p>
        </div>
      )}
    </div>
  );
}
