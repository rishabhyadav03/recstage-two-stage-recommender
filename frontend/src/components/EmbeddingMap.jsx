import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { RefreshCw, Film } from 'lucide-react';

export const GENRE_GROUPS = [
  {
    name: "Animation/Children's",
    lightColor: '#D97706', // Warm Amber
    darkColor: '#F59E0B',
  },
  {
    name: 'Drama',
    lightColor: '#4F46E5', // Indigo
    darkColor: '#818CF8',
  },
  {
    name: 'Comedy',
    lightColor: '#059669', // Emerald
    darkColor: '#34D399',
  },
  {
    name: 'Action/Adventure/Thriller',
    lightColor: '#0284C7', // Sky Blue
    darkColor: '#38BDF8',
  },
  {
    name: 'Romance/Musical',
    lightColor: '#9333EA', // Purple
    darkColor: '#C084FC',
  },
  {
    name: 'Other',
    lightColor: '#64748B', // Slate
    darkColor: '#94A3B8',
  },
];

// Vermilion accent for user & recommendations
const ACCENT_LIGHT = '#D94F30';
const ACCENT_DARK = '#E8664A';

function cleanTitle(title) {
  if (!title) return '';
  let clean = title.replace(/\s*\(\d{4}\)$/, '').trim();
  const match = clean.match(/^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$/i);
  if (match) {
    clean = `${match[2]} ${match[1]}`;
  }
  return clean;
}

function drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius, fill, stroke) {
  let rot = (Math.PI / 2) * 3;
  let x = cx;
  let y = cy;
  const step = Math.PI / spikes;

  ctx.beginPath();
  ctx.moveTo(cx, cy - outerRadius);
  for (let i = 0; i < spikes; i++) {
    x = cx + Math.cos(rot) * outerRadius;
    y = cy + Math.sin(rot) * outerRadius;
    ctx.lineTo(x, y);
    rot += step;

    x = cx + Math.cos(rot) * innerRadius;
    y = cy + Math.sin(rot) * innerRadius;
    ctx.lineTo(x, y);
    rot += step;
  }
  ctx.lineTo(cx, cy - outerRadius);
  ctx.closePath();

  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

export default function EmbeddingMap({
  userId = 1,
  mode = 'two_stage',
  userState: externalUserState,
  onUserStateLoaded,
  highlightMovieId = null,
}) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const navigate = useNavigate();

  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);

  // Data states
  const [points, setPoints] = useState([]);
  const [mapLoading, setMapLoading] = useState(true);
  const [mapError, setMapError] = useState(null);

  const [internalUserState, setInternalUserState] = useState(null);
  const [userLoading, setUserLoading] = useState(false);

  const userState = externalUserState !== undefined ? externalUserState : internalUserState;

  // Hover state
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Animation glide state
  const currentPosRef = useRef(null);
  const targetPosRef = useRef(null);
  const animFrameRef = useRef(null);

  // Color lookup map
  const colorMap = useMemo(() => {
    const map = {};
    for (const g of GENRE_GROUPS) {
      map[g.name] = isDark ? g.darkColor : g.lightColor;
    }
    return map;
  }, [isDark]);

  const defaultDotColor = isDark ? '#94A3B8' : '#64748B';
  const accentColor = isDark ? ACCENT_DARK : ACCENT_LIGHT;

  // 1. Fetch base embedding map (loaded once, cached by browser)
  const fetchMap = useCallback(async () => {
    setMapLoading(true);
    setMapError(null);
    try {
      const res = await fetch('/api/map');
      if (!res.ok) {
        if (res.status === 503) {
          throw new Error('Embedding map not built yet. Run `python ml/build_embedding_map.py` to generate it.');
        }
        throw new Error(`Failed to load embedding map (status ${res.status})`);
      }
      const data = await res.json();
      setPoints(data);
    } catch (err) {
      console.error('Error fetching embedding map:', err);
      setMapError(err.message || 'Could not load embedding map.');
    } finally {
      setMapLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMap();
  }, [fetchMap]);

  // 2. Fetch user-specific map projection if not provided externally
  const fetchUserState = useCallback(async (uid) => {
    if (!uid) return;
    setUserLoading(true);
    try {
      const res = await fetch(`/api/map/user/${uid}`);
      if (!res.ok) {
        throw new Error(`Failed to load user state (status ${res.status})`);
      }
      const data = await res.json();
      setInternalUserState(data);
      if (onUserStateLoaded) onUserStateLoaded(data);
    } catch (err) {
      console.error('Error fetching user map state:', err);
    } finally {
      setUserLoading(false);
    }
  }, [onUserStateLoaded]);

  useEffect(() => {
    if (externalUserState === undefined) {
      fetchUserState(userId);
    }
  }, [userId, mode, externalUserState, fetchUserState]);

  // Points by movie_id lookup
  const pointLookup = useMemo(() => {
    const lookup = new Map();
    for (const p of points) {
      lookup.set(p.movie_id, p);
    }
    return lookup;
  }, [points]);

  // Compute view bounds from 1st to 99th percentile of x and y with 4% padding
  const bounds = useMemo(() => {
    if (!points || points.length === 0) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }
    const xs = points.map((p) => p.x).sort((a, b) => a - b);
    const ys = points.map((p) => p.y).sort((a, b) => a - b);
    const n = points.length;
    const p1X = xs[Math.floor(0.01 * n)];
    const p99X = xs[Math.floor(0.99 * n)];
    const p1Y = ys[Math.floor(0.01 * n)];
    const p99Y = ys[Math.floor(0.99 * n)];

    const rx = p99X - p1X;
    const ry = p99Y - p1Y;

    let minX = p1X - 0.04 * rx;
    let maxX = p99X + 0.04 * rx;
    let minY = p1Y - 0.04 * ry;
    let maxY = p99Y + 0.04 * ry;

    // Ensure the user star and its ring stay inside the frame after bounds change
    const userPos = userState?.user_xy;
    if (userPos && userPos.length >= 2) {
      const [ux, uy] = userPos;
      if (ux - 0.04 * rx < minX) minX = ux - 0.04 * rx;
      if (ux + 0.04 * rx > maxX) maxX = ux + 0.04 * rx;
      if (uy - 0.04 * ry < minY) minY = uy - 0.04 * ry;
      if (uy + 0.04 * ry > maxY) maxY = uy + 0.04 * ry;
    }

    return { minX, maxX, minY, maxY };
  }, [points, userState?.user_xy]);

  // Sets for quick overlay check
  const watchedSet = useMemo(() => {
    return new Set(userState?.history_ids || []);
  }, [userState]);

  const recommendedSet = useMemo(() => {
    return new Set(userState?.recommended_ids || []);
  }, [userState]);

  // Screen reader recommended movie list
  const recommendedMovies = useMemo(() => {
    if (!userState?.recommended_ids) return [];
    return userState.recommended_ids
      .map((id) => pointLookup.get(id))
      .filter(Boolean);
  }, [userState, pointLookup]);

  // Render base dots to offscreen canvas using percentile bounds
  const renderBaseDotsToOffscreen = useCallback(
    (width, height) => {
      if (width <= 0 || height <= 0 || points.length === 0) return;

      let offscreen = offscreenCanvasRef.current;
      if (!offscreen) {
        offscreen = document.createElement('canvas');
        offscreenCanvasRef.current = offscreen;
      }

      const dpr = window.devicePixelRatio || 1;
      offscreen.width = width * dpr;
      offscreen.height = height * dpr;

      const ctx = offscreen.getContext('2d');
      ctx.clearRect(0, 0, offscreen.width, offscreen.height);
      ctx.scale(dpr, dpr);

      // Margins inside canvas for dot breathing room
      const pad = 16;
      const plotW = width - pad * 2;
      const plotH = height - pad * 2;

      // Base dots: r=2.5 with 0.55 opacity (dimmed to 0.35 when user state is active)
      const baseAlpha = userState ? 0.35 : 0.55;
      ctx.globalAlpha = baseAlpha;

      const { minX, maxX, minY, maxY } = bounds;
      const spanX = maxX - minX || 1;
      const spanY = maxY - minY || 1;

      for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        // Points outside bounds are not drawn
        if (pt.x < minX || pt.x > maxX || pt.y < minY || pt.y > maxY) continue;

        const normX = (pt.x - minX) / spanX;
        const normY = (pt.y - minY) / spanY;
        const px = pad + normX * plotW;
        const py = pad + normY * plotH;

        ctx.fillStyle = colorMap[pt.group] || defaultDotColor;
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.globalAlpha = 1.0;
    },
    [points, bounds, colorMap, defaultDotColor, userState]
  );

  // Main draw function: draws offscreen base dots + overlays
  const drawCanvas = useCallback(
    (starPos) => {
      const canvas = canvasRef.current;
      const offscreen = offscreenCanvasRef.current;
      if (!canvas || !offscreen) return;

      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      // Ensure canvas pixel dimensions match bounding client rect
      const targetW = Math.round(rect.width * dpr);
      const targetH = Math.round(rect.height * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(dpr, dpr);

      const width = rect.width;
      const height = rect.height;

      // 1. Draw cached base dots
      ctx.drawImage(offscreen, 0, 0, width, height);

      const pad = 16;
      const plotW = width - pad * 2;
      const plotH = height - pad * 2;

      const { minX, maxX, minY, maxY } = bounds;
      const spanX = maxX - minX || 1;
      const spanY = maxY - minY || 1;

      // 2. Draw Watched films as neutral rings
      if (watchedSet.size > 0) {
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = isDark ? 'rgba(237, 231, 221, 0.7)' : 'rgba(27, 24, 21, 0.65)';
        for (const mid of watchedSet) {
          const pt = pointLookup.get(mid);
          if (pt && pt.x >= minX && pt.x <= maxX && pt.y >= minY && pt.y <= maxY) {
            const px = pad + ((pt.x - minX) / spanX) * plotW;
            const py = pad + ((pt.y - minY) / spanY) * plotH;
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
      }

      // 3. Draw Recommended films as vermilion filled dots with white outline
      if (recommendedSet.size > 0) {
        ctx.lineWidth = 1.5;
        ctx.fillStyle = accentColor;
        ctx.strokeStyle = isDark ? '#1B1917' : '#FFFFFF';
        for (const mid of recommendedSet) {
          const pt = pointLookup.get(mid);
          if (pt && pt.x >= minX && pt.x <= maxX && pt.y >= minY && pt.y <= maxY) {
            const px = pad + ((pt.x - minX) / spanX) * plotW;
            const py = pad + ((pt.y - minY) / spanY) * plotH;
            ctx.beginPath();
            ctx.arc(px, py, 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }
      }

      // 4. Draw Highlighted movie from Closest list hover (thin accent ring & dot enlarged)
      if (highlightMovieId) {
        const pt = pointLookup.get(highlightMovieId);
        if (pt && pt.x >= minX && pt.x <= maxX && pt.y >= minY && pt.y <= maxY) {
          const px = pad + ((pt.x - minX) / spanX) * plotW;
          const py = pad + ((pt.y - minY) / spanY) * plotH;

          ctx.save();
          // Enlarged dot
          ctx.beginPath();
          ctx.arc(px, py, 6, 0, Math.PI * 2);
          ctx.fillStyle = colorMap[pt.group] || accentColor;
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = isDark ? '#131110' : '#FFFFFF';
          ctx.stroke();

          // Thin accent ring
          ctx.beginPath();
          ctx.arc(px, py, 9.5, 0, Math.PI * 2);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = accentColor;
          ctx.stroke();
          ctx.restore();
        }
      }

      // 5. Draw User Position (vermilion star with soft dashed circle around nearest area)
      if (starPos) {
        const rawPx = pad + ((starPos[0] - minX) / spanX) * plotW;
        const rawPy = pad + ((starPos[1] - minY) / spanY) * plotH;

        // User star and its ring must stay inside the frame after bounds change
        const ringR = 36;
        const starPx = Math.max(ringR + 2, Math.min(width - ringR - 2, rawPx));
        const starPy = Math.max(ringR + 2, Math.min(height - ringR - 2, rawPy));

        // Soft dashed circle
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = isDark ? 'rgba(232, 102, 74, 0.45)' : 'rgba(217, 79, 48, 0.4)';
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(starPx, starPy, 26, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Outer glow/pulse ring
        ctx.strokeStyle = isDark ? 'rgba(232, 102, 74, 0.25)' : 'rgba(217, 79, 48, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(starPx, starPy, 36, 0, Math.PI * 2);
        ctx.stroke();

        // Vermilion 5-pointed Star
        drawStar(
          ctx,
          starPx,
          starPy,
          5,
          8,
          4,
          accentColor,
          isDark ? '#131110' : '#FFFFFF'
        );
      }

      ctx.restore();
    },
    [bounds, watchedSet, recommendedSet, pointLookup, highlightMovieId, colorMap, isDark, accentColor]
  );

  // Resize handler & base dots re-render
  const handleResizeAndRedraw = useCallback(() => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    renderBaseDotsToOffscreen(rect.width, rect.height);
    drawCanvas(currentPosRef.current || userState?.user_xy);
  }, [renderBaseDotsToOffscreen, drawCanvas, userState]);

  useEffect(() => {
    handleResizeAndRedraw();
  }, [handleResizeAndRedraw, theme, points, bounds, userState, highlightMovieId]);

  useEffect(() => {
    const handleResize = () => {
      handleResizeAndRedraw();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResizeAndRedraw]);

  // Gliding transition for user star
  useEffect(() => {
    if (!userState?.user_xy) return;

    const newTarget = userState.user_xy;
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (!currentPosRef.current || prefersReducedMotion) {
      currentPosRef.current = newTarget;
      targetPosRef.current = newTarget;
      drawCanvas(newTarget);
      return;
    }

    const startPos = [...currentPosRef.current];
    targetPosRef.current = newTarget;
    const duration = 400; // ms transition
    const startTime = performance.now();

    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }

    const animate = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);

      const curX = startPos[0] + (newTarget[0] - startPos[0]) * ease;
      const curY = startPos[1] + (newTarget[1] - startPos[1]) * ease;
      currentPosRef.current = [curX, curY];

      drawCanvas([curX, curY]);

      if (progress < 1) {
        animFrameRef.current = requestAnimationFrame(animate);
      } else {
        currentPosRef.current = newTarget;
      }
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [userState, drawCanvas]);

  // Mouse hover detection (nearest dot within 10px)
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const pad = 16;
    const plotW = rect.width - pad * 2;
    const plotH = rect.height - pad * 2;

    const { minX, maxX, minY, maxY } = bounds;
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;

    let nearest = null;
    let minDistSq = 100; // 10px radius squared

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY) continue;
      const px = pad + ((p.x - minX) / spanX) * plotW;
      const py = pad + ((p.y - minY) / spanY) * plotH;
      const dx = px - mouseX;
      const dy = py - mouseY;
      const distSq = dx * dx + dy * dy;

      if (distSq < minDistSq) {
        minDistSq = distSq;
        nearest = { ...p, px, py };
      }
    }

    if (nearest) {
      setHoveredPoint(nearest);
      setTooltipPos({ x: nearest.px, y: nearest.py });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  const handleClick = () => {
    if (hoveredPoint?.movie_id) {
      navigate(`/explore?item=${hoveredPoint.movie_id}`);
    }
  };

  // Loading skeleton
  if (mapLoading) {
    return (
      <div className="space-y-3">
        <div>
          <div className="w-56 h-6 bg-surface-3/70 rounded" />
          <div className="w-80 h-3.5 bg-surface-3/70 rounded mt-1.5" />
        </div>
        <div className="w-full h-[20rem] lg:h-[26rem] bg-surface-2/60 border border-border/60 rounded-xl animate-pulse" />
      </div>
    );
  }

  // Error state
  if (mapError) {
    return (
      <div className="space-y-3">
        <div>
          <h3 className="font-serif text-lg sm:text-xl font-semibold text-text-main tracking-tight">
            Where this user's taste lives
          </h3>
          <p className="text-xs sm:text-sm text-text-muted mt-0.5">
            Each dot is a film. Films that people tend to like together sit close to each other.
          </p>
        </div>
        <div className="w-full h-[20rem] lg:h-[26rem] bg-surface border border-border rounded-xl flex flex-col items-center justify-center p-6 text-center space-y-3">
          <p className="text-sm text-text-muted max-w-sm">
            {mapError}
          </p>
          <button
            onClick={fetchMap}
            className="px-3.5 py-1.5 bg-surface-2 hover:bg-surface-3 border border-border rounded-lg text-xs font-medium text-text-main transition-colors inline-flex items-center gap-1.5 cursor-pointer"
          >
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      {/* Small serif heading & editorial subtitle */}
      <div>
        <h3 className="font-serif text-lg sm:text-xl font-semibold text-text-main tracking-tight">
          Where this user's taste lives
        </h3>
        <p className="text-xs sm:text-sm text-text-muted mt-0.5">
          Each dot is a film. Films that people tend to like together sit close to each other.
        </p>
      </div>

      {/* Canvas container with hairline border: 26rem on desktop, 20rem on mobile */}
      <div
        ref={containerRef}
        className="relative w-full h-[20rem] lg:h-[26rem] border border-border/70 rounded-xl overflow-hidden bg-surface-2/30"
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`2D UMAP embedding map with films. User ${userId} is positioned at the vermilion star with their recommendations highlighted.`}
          tabIndex={0}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className="w-full h-full block cursor-default"
          style={{ cursor: hoveredPoint ? 'pointer' : 'default' }}
        />

        {/* Hover Tooltip (within 10px of nearest dot) */}
        {hoveredPoint && (
          <div
            className="absolute z-20 pointer-events-none transition-transform duration-75 ease-out shadow-popover border border-border/80 bg-surface/95 backdrop-blur-md rounded-lg p-2.5 max-w-xs flex items-center gap-2.5"
            style={{
              left: `${Math.min(
                Math.max(12, tooltipPos.x + 12),
                (containerRef.current?.clientWidth || 300) - 230
              )}px`,
              top: `${Math.min(
                Math.max(12, tooltipPos.y - 45),
                (containerRef.current?.clientHeight || 200) - 80
              )}px`,
            }}
          >
            {/* Small w92 poster */}
            <div className="w-10 h-15 rounded bg-surface-3/80 flex-shrink-0 overflow-hidden border border-border/40 flex items-center justify-center">
              {hoveredPoint.poster_path ? (
                <img
                  src={`https://image.tmdb.org/t/p/w92${hoveredPoint.poster_path}`}
                  alt=""
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              ) : (
                <Film size={18} className="text-text-muted/60" />
              )}
            </div>

            {/* Details */}
            <div className="min-w-0 flex-1">
              <div
                className="font-serif text-xs font-semibold text-text-main truncate"
                title={hoveredPoint.title}
              >
                {cleanTitle(hoveredPoint.title)}
              </div>
              <div className="text-[11px] text-text-muted mt-0.5">
                {hoveredPoint.year || ''}
              </div>
              <div className="flex items-center gap-1.5 mt-1 text-[11px] text-text-muted">
                <span
                  className="w-2 h-2 rounded-full inline-block flex-shrink-0"
                  style={{
                    backgroundColor:
                      colorMap[hoveredPoint.group] || defaultDotColor,
                  }}
                />
                <span className="truncate">{hoveredPoint.group}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend: ONE row under the map */}
      <div className="flex items-center gap-x-3 sm:gap-x-4 overflow-x-auto whitespace-nowrap text-xs text-text-muted py-0.5 no-scrollbar">
        {GENRE_GROUPS.map((group) => (
          <div key={group.name} className="inline-flex items-center gap-1.5 flex-shrink-0">
            <span
              className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0"
              style={{
                backgroundColor: isDark ? group.darkColor : group.lightColor,
              }}
            />
            <span className="text-text-muted hover:text-text-main transition-colors">
              {group.name}
            </span>
          </div>
        ))}
        {/* User indicator in legend */}
        <div className="inline-flex items-center gap-1.5 flex-shrink-0">
          <span
            className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0"
            style={{ backgroundColor: accentColor }}
          />
          <span className="text-text-main font-medium">User & Top Recs</span>
        </div>
      </div>

      {/* Caption under legend */}
      <p className="text-[11px] text-text-muted/80 leading-relaxed">
        2D projection of 64-dimensional embeddings (UMAP). Distances are approximate: films that look close are usually, but not always, similar. A few outlying films are not shown.
      </p>

      {/* Keyboard accessible screen-reader list */}
      <div className="sr-only">
        <h4>Recommended films for User {userId} on the embedding map:</h4>
        <ul>
          {recommendedMovies.map((m) => (
            <li key={m.movie_id}>
              <button
                type="button"
                onClick={() => navigate(`/explore?item=${m.movie_id}`)}
              >
                {m.title} ({m.year}) - {m.group}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
