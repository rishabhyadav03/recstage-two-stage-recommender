import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Sun, Moon, Menu, X } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function Header() {
  const { theme, toggleTheme } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { to: '/', label: 'Recommend' },
    { to: '/compare', label: 'Compare' },
    { to: '/explore', label: 'Explore' },
    { to: '/how-it-works', label: 'How it works' },
    { to: '/results', label: 'Results' },
  ];

  return (
    <header className="sticky top-0 z-50 h-14 sm:h-16 w-full border-b border-border bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex h-full max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Wordmark (Far Left Child) */}
        <NavLink 
          to="/" 
          className="flex items-center gap-[10px] text-lg font-semibold tracking-tight text-text-main transition-colors hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 rounded-sm"
          aria-label="RecStage home"
        >
          <img 
            src="/logo.png" 
            alt="RecStage logo" 
            width="36" 
            height="36" 
            className="h-[36px] w-[36px] rounded-[22%] border border-border object-contain shrink-0" 
          />
          <span className="tracking-tight">RecStage</span>
        </NavLink>

        {/* Right Child Group: Nav links (immediately left of toggle) + Theme Toggle + Mobile menu button */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Desktop Nav Links */}
          <nav className="hidden md:flex items-center gap-1 sm:gap-1.5" aria-label="Main navigation">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) =>
                  `px-3 py-1.5 text-[14px] font-medium rounded-md transition-colors whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 ${
                    isActive
                      ? 'text-accent bg-accent-subtle font-semibold'
                      : 'text-text-muted hover:text-text-main hover:bg-surface-2'
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            type="button"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-surface text-text-muted hover:text-text-main hover:bg-surface-2 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2"
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>

          {/* Hamburger Menu on mobile (< 768px), next to the toggle */}
          <button
            onClick={() => setMobileMenuOpen(prev => !prev)}
            type="button"
            className="flex md:hidden h-9 w-9 items-center justify-center rounded-lg border border-border bg-surface text-text-muted hover:text-text-main hover:bg-surface-2 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2"
            aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          >
            {mobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile Nav Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-b border-border bg-surface px-4 py-3 space-y-1 shadow-popover">
          {navLinks.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) =>
                `block px-3 py-2 text-[15px] font-medium rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 ${
                  isActive
                    ? 'text-accent bg-accent-subtle font-semibold'
                    : 'text-text-muted hover:text-text-main hover:bg-surface-2'
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      )}
    </header>
  );
}
