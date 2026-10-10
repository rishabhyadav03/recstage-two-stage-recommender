import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Header from './components/Header';
import Footer from './components/Footer';

import ErrorBoundary from './components/ErrorBoundary';
import ServerWakeBanner from './components/ServerWakeBanner';

import RecommendPage from './pages/RecommendPage';
import ComparePage from './pages/ComparePage';
import ExplorePage from './pages/ExplorePage';
import HowItWorksPage from './pages/HowItWorksPage';
import ResultsPage from './pages/ResultsPage';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col bg-background text-text-main transition-colors duration-150 overflow-x-clip">
          <Header />
          <ServerWakeBanner />
          
          <main className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 sm:pt-10 pb-8 sm:pb-10">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<RecommendPage />} />
                <Route path="/compare" element={<ComparePage />} />
                <Route path="/explore" element={<ExplorePage />} />
                <Route path="/how-it-works" element={<HowItWorksPage />} />
                <Route path="/results" element={<ResultsPage />} />
              </Routes>
            </ErrorBoundary>
          </main>

          <Footer />
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}
