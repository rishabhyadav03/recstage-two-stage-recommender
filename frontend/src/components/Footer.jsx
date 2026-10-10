import React from 'react';

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-border py-6 text-center text-[13px] text-text-muted">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div>
          <span className="font-medium text-text-main">RecStage</span> — Two-Stage Recommendation Architecture on MovieLens 1M.
        </div>
        <div>
          <a 
            href="https://grouplens.org/datasets/movielens/1m/" 
            target="_blank" 
            rel="noreferrer"
            className="hover:text-text-main underline decoration-border hover:decoration-text-muted transition-colors"
          >
            Dataset
          </a>
        </div>
      </div>
    </footer>
  );
}
