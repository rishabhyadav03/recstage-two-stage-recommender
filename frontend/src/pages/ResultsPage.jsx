import React, { useState, useEffect } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  CartesianGrid 
} from 'recharts';
import { useTheme } from '../context/ThemeContext';
import { apiUrl } from '../utils/api';

export default function ResultsPage() {
  const { theme } = useTheme();
  const [metrics, setMetrics] = useState(null);

  // Fetch metrics dynamically from /api/metrics
  useEffect(() => {
    let isMounted = true;
    async function fetchMetrics() {
      try {
        const res = await fetch(apiUrl('/api/metrics'));
        if (res.ok) {
          const data = await res.json();
          if (isMounted) setMetrics(data);
        }
      } catch (err) {
        console.error('Failed to load metrics:', err);
      }
    }
    fetchMetrics();
    return () => { isMounted = false; };
  }, []);

  // Theme-aware chart colors in warm editorial palette
  const isDark = theme === 'dark';
  const gridColor = isDark ? '#2E2924' : '#E8E1D5';
  const textColor = isDark ? '#9A9187' : '#736B61';
  const barPrimary = isDark ? '#E8664A' : '#D94F30';
  const barSecondary = isDark ? '#3D362F' : '#D5CCC0';
  const tooltipBg = isDark ? '#1C1917' : '#FAF8F5';
  const tooltipBorder = isDark ? '#3D362F' : '#E8E1D5';

  // Extract model stats with fallbacks
  const pop = metrics?.["Popularity"] || {
    "Recall@10": 0.0477, "Recall@50": 0.1779, "Recall@100": 0.2759,
    "NDCG@10": 0.0222, "MAP@10": 0.0147, "Coverage@10": 0.0389, "latency_p50_ms": 0.057
  };

  const bpr = metrics?.["Matrix Factorization (BPR)"] || {
    "Recall@10": 0.0671, "Recall@50": 0.2329, "Recall@100": 0.3545,
    "NDCG@10": 0.0315, "MAP@10": 0.0208, "Coverage@10": 0.1635, "latency_p50_ms": 0.709
  };

  const tt = metrics?.["Two-Tower Retrieval"] || {
    "Recall@10": 0.1622, "Recall@50": 0.3876, "Recall@100": 0.5069,
    "NDCG@10": 0.0838, "MAP@10": 0.0602, "Coverage@10": 0.3593, "latency_p50_ms": 0.537
  };

  const ts = metrics?.["Two-Stage (Two-Tower + Ranker)"] || {
    "Recall@10": 0.1684, "Recall@50": 0.3955, "Recall@100": 0.5145,
    "NDCG@10": 0.0859, "MAP@10": 0.0610, "Coverage@10": 0.3837, "latency_p50_ms": 3.29, "Hit@10": 0.1684
  };

  const ablSim = metrics?.["Ablation: Ranker w/o Similarity"] || {
    "NDCG@10": 0.0893, "Recall@10": 0.1704, "Hit@10": 0.1704, "Coverage@10": 0.3817, "latency_p50_ms": 1.55
  };

  const ablUser = metrics?.["Ablation: Ranker w/o User Features"] || {
    "NDCG@10": 0.0868, "Recall@10": 0.1706, "Hit@10": 0.1706, "Coverage@10": 0.3860, "latency_p50_ms": 1.18
  };

  const cold = metrics?.["error_analysis"]?.["cold_start_users"] || {
    "num_users": 760, "max_interactions": 34, "NDCG@10": 0.1020, "Hit@10": 0.2000
  };

  // Chart 1: Recall@K comparison
  const recallChartData = [
    { name: 'Popularity', 'Recall@10': pop["Recall@10"], 'Recall@100': pop["Recall@100"] },
    { name: 'BPR-MF', 'Recall@10': bpr["Recall@10"], 'Recall@100': bpr["Recall@100"] },
    { name: 'Two-Tower', 'Recall@10': tt["Recall@10"], 'Recall@100': tt["Recall@100"] },
    { name: 'Two-Stage', 'Recall@10': ts["Recall@10"], 'Recall@100': ts["Recall@100"] },
  ];

  // Chart 2: NDCG@10 comparison
  const ndcgChartData = [
    { name: 'Popularity', 'NDCG@10': pop["NDCG@10"] },
    { name: 'BPR-MF', 'NDCG@10': bpr["NDCG@10"] },
    { name: 'Two-Tower', 'NDCG@10': tt["NDCG@10"] },
    { name: 'Two-Stage', 'NDCG@10': ts["NDCG@10"] },
  ];

  const totalUsers = ts.num_evaluated_users || 3563;

  return (
    <div>
      {/* 1. Large Serif Heading & Editorial Intro */}
      <div>
        <h1 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-semibold text-text-main tracking-tight">
          Benchmark results
        </h1>
        <p className="text-sm sm:text-base text-text-muted mt-1 max-w-2xl leading-relaxed">
          Empirical offline evaluation over {totalUsers.toLocaleString()} test users using strict leave-last-out temporal testing on MovieLens 1M (ratings ≥ 4 as positives).
        </p>
      </div>

      {/* 2. Primary Model Comparison Table (Clean typographic table, hairline row borders, best values in bold accent) */}
      <section aria-labelledby="metrics-table-heading" className="mt-8 sm:mt-10 space-y-4">
        <div>
          <h2 id="metrics-table-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
            Overall offline metrics
          </h2>
          <p className="text-sm text-text-muted mt-1">
            Comparing candidate retrieval and precision re-ranking against global and collaborative baselines.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-border/80 text-text-muted">
                <th className="py-3 px-2 font-medium">Model</th>
                <th className="py-3 px-3 font-medium text-right">Recall@10</th>
                <th className="py-3 px-3 font-medium text-right">Recall@50</th>
                <th className="py-3 px-3 font-medium text-right">Recall@100</th>
                <th className="py-3 px-3 font-medium text-right">NDCG@10</th>
                <th className="py-3 px-3 font-medium text-right">MAP@10</th>
                <th className="py-3 px-3 font-medium text-right">Coverage</th>
                <th className="py-3 px-3 font-medium text-right">p50 Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              <tr>
                <td className="py-3.5 px-2 font-medium text-text-main">Popularity Baseline</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{pop["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{pop["Recall@50"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{pop["Recall@100"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{pop["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{pop["MAP@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((pop["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{(pop["latency_p50_ms"] || 0.1).toFixed(1)} ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 font-medium text-text-main">Matrix Factorization (BPR)</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{bpr["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{bpr["Recall@50"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{bpr["Recall@100"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{bpr["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{bpr["MAP@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((bpr["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{(bpr["latency_p50_ms"] || 0.7).toFixed(1)} ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 font-medium text-text-main">Two-Tower DNN Retrieval</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{tt["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{tt["Recall@50"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{tt["Recall@100"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{tt["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{tt["MAP@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((tt["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{(tt["latency_p50_ms"] || 0.5).toFixed(1)} ms</td>
              </tr>
              <tr className="bg-accent-subtle/30">
                <td className="py-3.5 px-2 font-semibold text-text-main flex items-center gap-2">
                  <span>Two-Stage (Two-Tower + Ranker)</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent text-white uppercase font-bold tracking-wider">Top</span>
                </td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{ts["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{ts["Recall@50"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{ts["Recall@100"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{ts["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{ts["MAP@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">{((ts["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{(ts["latency_p50_ms"] || 3.3).toFixed(1)} ms</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-xs text-text-muted mt-2 leading-relaxed">
          <strong className="text-text-main">Paired bootstrap check (1,000 resamples, 95% CI):</strong> Mean difference for Two-Stage vs Two-Tower is +0.0021 on NDCG@10 (95% CI [−0.0019, +0.0058]) and +0.0062 on Recall@10 (95% CI [−0.0020, +0.0143]). Because both intervals span zero, the test-set difference is not statistically distinguishable from zero.
        </p>
      </section>

      {/* 3. Visual Comparisons: Recall@K & NDCG@10 Charts in Warm Palette with Muted Gridlines */}
      <section aria-labelledby="charts-heading" className="mt-8 sm:mt-10 pt-6 border-t border-border/60 space-y-4">
        <div>
          <h2 id="charts-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
            Visual comparisons
          </h2>
          <p className="text-sm text-text-muted mt-1">
            Comparing retrieval capacity across diverse cutoffs and ranked position decay.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          {/* Chart 1: Recall@K */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-text-muted">
              <span className="font-medium text-text-main">Recall at K=10 and K=100</span>
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-accent inline-block" />
                  Recall@10
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-border inline-block" />
                  Recall@100
                </span>
              </div>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={recallChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke={gridColor} vertical={false} />
                  <XAxis dataKey="name" stroke={textColor} fontSize={12} tickLine={false} />
                  <YAxis stroke={textColor} fontSize={12} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: tooltipBg, 
                      borderColor: tooltipBorder, 
                      borderRadius: '8px', 
                      fontSize: '12px',
                      color: isDark ? '#EDE7DD' : '#1B1815'
                    }} 
                  />
                  <Bar isAnimationActive={false} dataKey="Recall@10" fill={barPrimary} radius={[3, 3, 0, 0]} />
                  <Bar isAnimationActive={false} dataKey="Recall@100" fill={barSecondary} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart 2: NDCG@10 */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-text-muted">
              <span className="font-medium text-text-main">Normalized Discounted Cumulative Gain (NDCG@10)</span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={ndcgChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke={gridColor} vertical={false} />
                  <XAxis dataKey="name" stroke={textColor} fontSize={12} tickLine={false} />
                  <YAxis stroke={textColor} fontSize={12} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: tooltipBg, 
                      borderColor: tooltipBorder, 
                      borderRadius: '8px', 
                      fontSize: '12px',
                      color: isDark ? '#EDE7DD' : '#1B1815'
                    }} 
                  />
                  <Bar isAnimationActive={false} dataKey="NDCG@10" fill={barPrimary} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Ablation Studies Table Below */}
      <section aria-labelledby="ablation-heading" className="mt-8 sm:mt-10 pt-6 border-t border-border/60 space-y-4">
        <div>
          <h2 id="ablation-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
            Ablation studies
          </h2>
          <p className="text-sm text-text-muted mt-1">
            Evaluating the influence of specific feature subsets on the Stage-2 LightGBM ranker.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-border/80 text-text-muted">
                <th className="py-3 px-2 font-medium">Ablation Variant</th>
                <th className="py-3 px-3 font-medium text-right">NDCG@10</th>
                <th className="py-3 px-3 font-medium text-right">Recall@10</th>
                <th className="py-3 px-3 font-medium text-right">Hit@10</th>
                <th className="py-3 px-3 font-medium text-right">Coverage</th>
                <th className="py-3 px-3 font-medium text-right">Latency p50</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              <tr>
                <td className="py-3.5 px-2 font-medium text-text-main">Full Two-Stage Ranker</td>
                <td className="py-3.5 px-3 text-right font-numeric font-medium text-text-main">{ts["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{ts["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((ts["Hit@10"] || ts["Recall@10"] || 0) * 100).toFixed(2)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((ts["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{(ts["latency_p50_ms"] || 3.3).toFixed(1)} ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 text-text-muted">Ranker w/o Similarity Features</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{ablSim["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{ablSim["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((ablSim["Hit@10"] || 0) * 100).toFixed(2)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((ablSim["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{(ablSim["latency_p50_ms"] || 1.6).toFixed(1)} ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 text-text-muted">Ranker w/o User Features</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{ablUser["NDCG@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{ablUser["Recall@10"]?.toFixed(4)}</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((ablUser["Hit@10"] || 0) * 100).toFixed(2)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{((ablUser["Coverage@10"] || 0) * 100).toFixed(1)}%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">{(ablUser["latency_p50_ms"] || 1.2).toFixed(1)} ms</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-xs sm:text-sm text-text-muted leading-relaxed pt-2">
          <strong className="text-text-main">Ablation interpretation:</strong> Variants without cosine similarity (NDCG@10 0.0893, 95% CI Δ [−0.0061, −0.0002]) and without user features (NDCG@10 0.0868, 95% CI Δ [−0.0033, +0.0016]) perform equal to or slightly higher than the full ranker. Because Stage-1 retrieval rank carries 71.5% of the feature importance, the ranker mostly preserves retrieval order with small adjustments. A simpler ranker (e.g. w/o user features) is just as effective and faster in production (1.18 ms vs 3.29 ms p50 latency).
        </p>
      </section>

      {/* 5. Error Analysis: Cold-Start Segment (Clean typography with hairline borders) */}
      <section aria-labelledby="cold-start-heading" className="mt-8 sm:mt-10 pt-6 border-t border-border/60 space-y-4">
        <div>
          <h2 id="cold-start-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
            Error analysis: Cold-start segment
          </h2>
          <p className="text-sm text-text-muted mt-1">
            Evaluating performance on users in the bottom 20% of interaction volume (≤ {cold["max_interactions"] || 34} ratings).
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="space-y-1">
            <div className="text-xs text-text-muted font-medium">User segment size</div>
            <div className="font-serif text-3xl font-bold text-text-main font-numeric">
              {cold["num_users"] || 760}
            </div>
            <div className="text-xs text-text-muted">Bottom 20% (≤{cold["max_interactions"] || 34} interactions)</div>
          </div>

          <div className="space-y-1 sm:border-l sm:border-border/60 sm:pl-6">
            <div className="text-xs text-text-muted font-medium">Cold-start NDCG@10</div>
            <div className="font-serif text-3xl font-bold text-text-main font-numeric">
              {cold["NDCG@10"]?.toFixed(4)}
            </div>
            <div className="text-xs text-text-muted">vs {ts["NDCG@10"]?.toFixed(4)} overall</div>
          </div>

          <div className="space-y-1 sm:border-l sm:border-border/60 sm:pl-6">
            <div className="text-xs text-text-muted font-medium">Cold-start Hit@10</div>
            <div className="font-serif text-3xl font-bold text-text-main font-numeric">
              {((cold["Hit@10"] || 0) * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-text-muted">vs {((ts["Hit@10"] || ts["Recall@10"] || 0) * 100).toFixed(2)}% overall</div>
          </div>
        </div>

        <p className="text-xs text-text-muted mt-2 leading-relaxed">
          <strong className="text-text-main">Cold-start cohort note:</strong> Infrequent viewers (≤34 interactions, bottom 20% of users, 760 users) score higher NDCG@10 (0.1020 vs 0.0859 overall) because their held-out test items are predominantly mainstream blockbuster titles with high training support (averaging 1,037 training ratings vs 824 for warmer users), which are easier for the model to rank accurately than the long tail.
        </p>
      </section>
    </div>
  );
}
