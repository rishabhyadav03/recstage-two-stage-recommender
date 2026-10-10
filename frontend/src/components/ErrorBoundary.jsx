import React from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, RotateCcw, Home } from 'lucide-react';

/**
 * Top-level ErrorBoundary that catches client-side rendering crashes
 * and displays a helpful recovery UI while keeping the header intact.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an unhandled error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    // Force soft refresh of the view
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="py-12 sm:py-16 px-4">
          <div className="max-w-md mx-auto bg-surface border border-border rounded-2xl p-6 sm:p-8 text-center shadow-card space-y-5">
            <div className="w-12 h-12 mx-auto rounded-xl bg-accent-subtle border border-accent-border flex items-center justify-center text-accent">
              <AlertTriangle size={24} />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-semibold text-text-main tracking-tight">
                Something went wrong
              </h2>
              <p className="text-sm text-text-muted leading-relaxed">
                An unexpected error occurred while loading this view. You can retry or head back to the main recommendations.
              </p>
              {this.state.error?.message && (
                <div className="mt-3 p-2.5 bg-surface-2 rounded-lg border border-border text-xs text-text-muted font-mono break-words text-left overflow-x-auto max-h-24">
                  {this.state.error.message}
                </div>
              )}
            </div>

            <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
              <Link
                to="/"
                onClick={this.handleReset}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-accent text-white hover:bg-accent-hover transition-colors shadow-soft"
              >
                <Home size={16} />
                <span>Back to Recommend</span>
              </Link>
              <button
                type="button"
                onClick={this.handleRetry}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-surface border border-border text-text-main hover:bg-surface-2 transition-colors cursor-pointer"
              >
                <RotateCcw size={16} />
                <span>Retry</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
