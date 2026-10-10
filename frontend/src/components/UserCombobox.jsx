import React, { useState, useRef, useEffect } from 'react';
import { Search, Dices, ChevronDown, Check } from 'lucide-react';

const SUGGESTED_USERS = [
  { id: 1, label: 'User 1', desc: 'Sci-Fi & Drama enthusiast' },
  { id: 2, label: 'User 2', desc: 'Action & Thriller fan' },
  { id: 42, label: 'User 42', desc: 'Classic & Film-Noir cinephile' },
  { id: 760, label: 'User 760', desc: 'Cold-start user (≤34 interactions)' },
];

export default function UserCombobox({ selectedUserId, onSelectUser, showRandom = true, inline = false }) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const wrapperRef = useRef(null);

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (id) => {
    onSelectUser(id);
    setIsOpen(false);
    setQuery('');
  };

  const handleRandom = () => {
    // 6040 users total in ML-1M
    const randomId = Math.floor(Math.random() * 6040) + 1;
    onSelectUser(randomId);
    setIsOpen(false);
  };

  const handleQuerySubmit = (e) => {
    e.preventDefault();
    const parsed = parseInt(query, 10);
    if (!isNaN(parsed) && parsed >= 1 && parsed <= 6040) {
      handleSelect(parsed);
    }
  };

  return (
    <div className="relative inline-flex items-center gap-2" ref={wrapperRef}>
      {/* Combobox Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(prev => !prev)}
        className={inline 
          ? "flex items-center justify-between gap-1.5 h-8 px-2.5 bg-surface-2 hover:bg-surface-3 border border-border rounded-lg text-xs font-semibold text-text-main transition-colors cursor-pointer"
          : "flex items-center justify-between gap-3 h-10 px-3.5 bg-surface border border-border rounded-lg text-sm text-text-main hover:bg-surface-2 transition-colors cursor-pointer min-w-[170px]"
        }
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span>User {selectedUserId}</span>
        <ChevronDown size={13} className="text-text-muted" />
      </button>

      {/* Random User Button */}
      {showRandom && (
        <button
          type="button"
          onClick={handleRandom}
          className="flex items-center gap-1.5 h-10 px-3 bg-surface border border-border rounded-lg text-xs font-medium text-text-muted hover:text-text-main hover:bg-surface-2 transition-colors cursor-pointer"
          title="Pick a random user"
          aria-label="Pick a random user"
        >
          <Dices size={15} />
          <span className="hidden sm:inline">Random user</span>
        </button>
      )}

      {/* Dropdown Popover */}
      {isOpen && (
        <div className="absolute top-12 left-0 z-50 w-72 bg-surface border border-border rounded-xl shadow-popover p-2 text-sm">
          {/* Quick Search Form */}
          <form onSubmit={handleQuerySubmit} className="mb-2">
            <div className="relative flex items-center">
              <Search size={14} className="absolute left-2.5 text-text-muted" />
              <input
                type="number"
                min="1"
                max="6040"
                placeholder="Jump to User ID (1–6040)..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                autoFocus
                className="w-full h-8 pl-8 pr-2.5 text-xs bg-surface-2 border border-border rounded-md text-text-main placeholder:text-text-muted outline-none focus:border-accent"
              />
            </div>
          </form>

          {/* Quick presets list */}
          <div className="text-[11px] font-semibold uppercase tracking-wider text-text-muted px-2 py-1">
            Suggested Profiles
          </div>

          <div className="space-y-0.5">
            {SUGGESTED_USERS.map((u) => {
              const isSelected = selectedUserId === u.id;
              return (
                <button
                  key={u.id}
                  type="button"
                  onClick={() => handleSelect(u.id)}
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-left transition-colors cursor-pointer ${
                    isSelected ? 'bg-accent-subtle text-accent font-medium' : 'hover:bg-surface-2 text-text-main'
                  }`}
                >
                  <div>
                    <div className="text-xs font-medium">{u.label}</div>
                    <div className="text-[11px] text-text-muted">{u.desc}</div>
                  </div>
                  {isSelected && <Check size={14} className="text-accent" />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
