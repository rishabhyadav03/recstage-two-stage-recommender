import React from 'react';
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

export default function ResultsPage() {
  const { theme } = useTheme();

  // Theme-aware chart colors in warm editorial palette
  const isDark = theme === 'dark';
  const gridColor = isDark ? '#2E2924' : '#E8E1D5';
  const textColor = isDark ? '#9A9187' : '#736B61';
  const barPrimary = isDark ? '#E8664A' : '#D94F30';
  const barSecondary = isDark ? '#3D362F' : '#D5CCC0';
  const tooltipBg = isDark ? '#1C1917' : '#FAF8F5';
  const tooltipBorder = isDark ? '#3D362F' : '#E8E1D5';

  // Chart 1: Recall@K comparison
  const recallChartData = [
    { name: 'Popularity', 'Recall@10': 0.0477, 'Recall@100': 0.2759 },
    { name: 'BPR-MF', 'Recall@10': 0.0671, 'Recall@100': 0.3545 },
    { name: 'Two-Tower', 'Recall@10': 0.1622, 'Recall@100': 0.5069 },
    { name: 'Two-Stage', 'Recall@10': 0.0166, 'Recall@100': 0.1782 },
  ];

  // Chart 2: NDCG@10 comparison
  const ndcgChartData = [
    { name: 'Popularity', 'NDCG@10': 0.0222 },
    { name: 'BPR-MF', 'NDCG@10': 0.0315 },
    { name: 'Two-Tower', 'NDCG@10': 0.0838 },
    { name: 'Two-Stage', 'NDCG@10': 0.0072 },
  ];

  return (
    <div>
      {/* 1. Large Serif Heading & Editorial Intro */}
      <div>
        <h1 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-semibold text-text-main tracking-tight">
          Benchmark results
        </h1>
        <p className="text-sm sm:text-base text-text-muted mt-1 max-w-2xl leading-relaxed">
          Empirical offline evaluation over 3,563 test users using strict leave-last-out temporal testing on MovieLens 1M (ratings ≥ 4 as positives).
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
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0477</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.1779</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.2759</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0222</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0147</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">3.9%</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.1 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 font-medium text-text-main">Matrix Factorization (BPR)</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0671</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.2329</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.3545</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0315</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0208</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">16.4%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.7 ms</td>
              </tr>
              <tr className="bg-accent-subtle/30">
                <td className="py-3.5 px-2 font-semibold text-text-main flex items-center gap-2">
                  <span>Two-Tower DNN Retrieval</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent text-white uppercase font-bold tracking-wider">Top</span>
                </td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.1622</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.3876</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.5069</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.0838</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.0602</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">35.9%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.5 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 font-medium text-text-main">Two-Stage (Two-Tower + Ranker)</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0166</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0929</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.1782</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0072</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0045</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">22.7%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">4.8 ms</td>
              </tr>
            </tbody>
          </table>
        </div>
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
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0072</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0166</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">1.66%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">22.7%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">4.8 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-2 text-text-muted">Ranker w/o Similarity Features</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0076</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">0.0168</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">1.68%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">22.6%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">4.6 ms</td>
              </tr>
              <tr className="bg-accent-subtle/30 font-medium">
                <td className="py-3.5 px-2 text-text-main flex items-center gap-2">
                  <span>Ranker w/o User Features</span>
                  <span className="text-xs text-accent font-semibold">(+82% NDCG gain)</span>
                </td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.0131</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">0.0250</td>
                <td className="py-3.5 px-3 text-right font-numeric font-bold text-accent">2.50%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">20.2%</td>
                <td className="py-3.5 px-3 text-right font-numeric text-text-muted">4.9 ms</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-xs sm:text-sm text-text-muted leading-relaxed pt-2">
          <strong className="text-text-main">Ablation finding:</strong> Removing static user demographic profile features improved ranker NDCG from 0.0072 to 0.0131, confirming that sparse demographics induced minor training over-fitting, whereas real-time item priors and session cross-features generalize effectively to future holdout interactions.
        </p>
      </section>

      {/* 5. Error Analysis: Cold-Start Segment (Clean typography with hairline borders) */}
      <section aria-labelledby="cold-start-heading" className="mt-8 sm:mt-10 pt-6 border-t border-border/60 space-y-4">
        <div>
          <h2 id="cold-start-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
            Error analysis: Cold-start segment
          </h2>
          <p className="text-sm text-text-muted mt-1">
            Evaluating performance on users in the bottom 20% of interaction volume (≤ 34 ratings).
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="space-y-1">
            <div className="text-xs text-text-muted font-medium">User segment size</div>
            <div className="font-serif text-3xl font-bold text-text-main font-numeric">
              760
            </div>
            <div className="text-xs text-text-muted">Bottom 20% (≤34 interactions)</div>
          </div>

          <div className="space-y-1 sm:border-l sm:border-border/60 sm:pl-6">
            <div className="text-xs text-text-muted font-medium">Cold-start NDCG@10</div>
            <div className="font-serif text-3xl font-bold text-text-main font-numeric">
              0.0057
            </div>
            <div className="text-xs text-text-muted">vs 0.0072 overall (-20.8%)</div>
          </div>

          <div className="space-y-1 sm:border-l sm:border-border/60 sm:pl-6">
            <div className="text-xs text-text-muted font-medium">Cold-start Hit@10</div>
            <div className="font-serif text-3xl font-bold text-text-main font-numeric">
              1.45%
            </div>
            <div className="text-xs text-text-muted">vs 1.66% overall</div>
          </div>
        </div>
      </section>
    </div>
  );
}
