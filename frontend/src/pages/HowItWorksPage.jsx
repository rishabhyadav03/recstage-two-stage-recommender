import React, { useEffect, useState } from 'react';
import { formatLatency } from '../utils/formatters';
import { apiUrl } from '../utils/api';

const FEATURE_META = {
  stage1_rank_inv: { category: 'Pipeline Signal', role: 'Reciprocal position of candidate in FAISS Stage-1 output', defaultImp: 0.715 },
  sim_last_items_max: { category: 'Contextual Cross', role: "Max cosine similarity to user's recent watches", defaultImp: 0.054 },
  item_popularity: { category: 'Item Prior', role: 'Log rating count across training corpus', defaultImp: 0.051 },
  recency: { category: 'Temporal Prior', role: 'Normalized release year of the movie', defaultImp: 0.043 },
  two_tower_sim: { category: 'Latent Representation', role: 'Raw cosine dot product between UserTower and ItemTower embeddings', defaultImp: 0.035 },
  sim_last_items_mean: { category: 'Contextual Cross', role: "Mean similarity across the user's active history", defaultImp: 0.033 },
  user_activity: { category: 'User Profile', role: 'Total ratings count for the user', defaultImp: 0.025 },
  item_mean_rating: { category: 'Quality Prior', role: 'Historical average star rating of the candidate item', defaultImp: 0.023 },
  genre_affinity: { category: 'User-Item Affinity', role: 'Dot product between user genre history and item genre vector', defaultImp: 0.014 },
  user_mean_rating: { category: 'User Calibration', role: 'Mean rating bias of the user', defaultImp: 0.007 },
};

export default function HowItWorksPage() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch(apiUrl('/api/metrics'))
      .then((r) => r.json())
      .then((d) => setMetrics(d))
      .catch((e) => console.error("Could not fetch metrics:", e));
  }, []);

  const retrievalP50 = metrics?.['Two-Tower Retrieval']?.latency_p50_ms ?? 0.5;
  const rankerP50 = metrics?.['Two-Stage (Two-Tower + Ranker)']?.ranking_latency_p50_ms ?? 2.8;
  const totalP50 = metrics?.['Two-Stage (Two-Tower + Ranker)']?.latency_p50_ms ?? 3.3;

  const importances = metrics?.error_analysis?.feature_importances;
  const featureRows = Object.entries(FEATURE_META).map(([key, meta]) => {
    const rawVal = importances && importances[key] != null ? importances[key] : meta.defaultImp;
    return {
      key,
      category: meta.category,
      role: meta.role,
      pct: (rawVal * 100).toFixed(1) + '%',
      rawVal,
    };
  }).sort((a, b) => b.rawVal - a.rawVal);

  return (
    <div>
      {/* 1. Large Serif Heading & Editorial Intro */}
      <div>
        <h1 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-semibold text-text-main tracking-tight">
          How the system works
        </h1>
        <p className="text-sm sm:text-base text-text-muted mt-1 max-w-2xl leading-relaxed">
          An industrial two-stage recommendation architecture: balancing extreme throughput retrieval across large catalogs with expressive contextual re-ranking.
        </p>
      </div>

      {/* 2. Editorial Three-Step Story (Joined by thin lines, no boxed cards) */}
      <section aria-labelledby="architecture-story" className="mt-8 sm:mt-10 pt-6 border-t border-border/70">
        <div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12 relative">
            {/* Step 01: Catalog */}
            <div className="flex flex-col justify-between space-y-4">
              <div>
                <div className="font-serif text-4xl sm:text-5xl font-bold text-accent tracking-tight">
                  01
                </div>
                <h2 className="font-serif text-2xl font-semibold text-text-main tracking-tight mt-1">
                  Catalog
                </h2>
                <p className="text-sm text-text-muted mt-2 leading-relaxed">
                  A corpus of 3,883 films. Evaluating every movie with complex neural rankers and cross-features on every user interaction is computationally impossible under interactive latencies.
                </p>
              </div>

              <div className="pt-4 border-t border-border/50 text-xs text-text-muted font-numeric">
                <span>3,883 candidate items in catalog</span>
              </div>
            </div>

            {/* Step 02: Retrieval */}
            <div className="flex flex-col justify-between space-y-4 md:border-l md:border-border/60 md:pl-10">
              <div>
                <div className="font-serif text-4xl sm:text-5xl font-bold text-accent tracking-tight">
                  02
                </div>
                <h2 className="font-serif text-2xl font-semibold text-text-main tracking-tight mt-1">
                  Retrieval
                </h2>
                <p className="text-sm text-text-muted mt-2 leading-relaxed">
                  The UserTower maps user history into a 64-dimensional latent embedding. An IVF-Flat FAISS vector index rapidly prunes the corpus, retrieving the top 200 candidates in sub-millisecond time.
                </p>
              </div>

              <div className="pt-4 border-t border-border/50 text-xs text-text-muted font-numeric flex items-center justify-between">
                <span>Top 200 candidates</span>
                <span className="font-medium text-text-main">
                  {retrievalP50 != null ? `${Number(retrievalP50).toFixed(1)} ms` : '0.5 ms'} p50
                </span>
              </div>
            </div>

            {/* Step 03: Ranking */}
            <div className="flex flex-col justify-between space-y-4 md:border-l md:border-border/60 md:pl-10">
              <div>
                <div className="font-serif text-4xl sm:text-5xl font-bold text-accent tracking-tight">
                  03
                </div>
                <h2 className="font-serif text-2xl font-semibold text-text-main tracking-tight mt-1">
                  Ranking
                </h2>
                <p className="text-sm text-text-muted mt-2 leading-relaxed">
                  A LightGBM LambdaRank decision tree re-scores the 200 candidates across 10 cross-features (user recency, genre alignment, and past interaction similarity) to deliver the final Top 10 recommendations.
                </p>
              </div>

              <div className="pt-4 border-t border-border/50 text-xs text-text-muted font-numeric flex items-center justify-between">
                <span>Final Top 10 items</span>
                <span className="font-medium text-text-main">
                  {rankerP50 != null ? `${Number(rankerP50).toFixed(1)} ms` : '2.8 ms'} p50
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Quiet Pipeline Note */}
        <div className="mt-6 pt-4 border-t border-border/50 text-xs text-text-muted flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <span>
            End-to-end recommendation generation completes in{' '}
            <strong className="text-text-main font-numeric">
              {totalP50 != null ? `${Number(totalP50).toFixed(1)} ms` : '3.3 ms'}
            </strong>{' '}
            (p50) on a single CPU core.
          </span>
          <span className="text-text-muted/70">
            Zero temporal data leakage verified across splits.
          </span>
        </div>
      </section>

      {/* 3. Stage-2 Feature Store Breakdown (Clean typographic table with hairline dividers) */}
      <section aria-labelledby="features-heading" className="mt-8 sm:mt-10 space-y-4">
        <div>
          <h2 id="features-heading" className="font-serif text-2xl font-semibold text-text-main tracking-tight">
            Stage-2 Feature Store
          </h2>
          <p className="text-sm text-text-muted mt-1">
            Ten cross-features engineered across user history, item metadata, and latent vector geometry.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-border/80 text-text-muted">
                <th className="py-3 px-2 font-medium">Feature</th>
                <th className="py-3 px-3 font-medium">Category</th>
                <th className="py-3 px-3 font-medium text-right">Importance</th>
                <th className="py-3 px-3 font-medium hidden sm:table-cell">Engineering Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {featureRows.map((feat, idx) => (
                <tr key={feat.key}>
                  <td className="py-3 px-2 font-medium text-text-main font-mono text-xs">{feat.key}</td>
                  <td className="py-3 px-3 text-text-muted">{feat.category}</td>
                  <td className={`py-3 px-3 text-right font-numeric ${idx === 0 ? 'font-semibold text-accent' : 'font-medium'}`}>
                    {feat.pct}
                  </td>
                  <td className="py-3 px-3 text-text-muted hidden sm:table-cell">{feat.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 4. Why Two Stages? Editorial Narrative */}
      <section className="mt-8 sm:mt-10 pt-6 border-t border-border/50 space-y-4 max-w-3xl text-sm text-text-muted leading-relaxed">
        <h3 className="font-serif text-xl font-semibold text-text-main tracking-tight">
          Why Two Stages?
        </h3>
        <p>
          In modern content applications, millions of items cannot be scored by a rich gradient-boosted tree or cross-network in real time. The first stage trades complex interactions for speed: representing items and users as independent vectors allows vector indexes (such as FAISS) to perform logarithmic-time candidate pruning.
        </p>
        <p>
          Once the candidate space is narrowed from thousands to hundreds, the second stage evaluates non-linear cross features (such as session-level recency, genre alignment, and rating distributions) to rank the final recommendations for the end user.
        </p>
      </section>
    </div>
  );
}
