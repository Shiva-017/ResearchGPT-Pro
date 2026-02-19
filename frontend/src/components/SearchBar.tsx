// src/components/SearchBar.tsx

"use client";

import { Search, Loader2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
  initialQuery?: string;
}

export default function SearchBar({
  onSearch,
  isLoading,
  initialQuery = "",
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed.length >= 3) onSearch(trimmed);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative group">
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-brand-600/20 via-brand-500/20 to-brand-400/20 blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500" />

        <div className="relative flex items-center rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] focus-within:border-[var(--border-focus)] focus-within:ring-1 focus-within:ring-brand-500/30">
          <Search className="ml-5 h-5 w-5 text-[var(--text-muted)] shrink-0" />

          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search 25,000+ research papers..."
            className="flex-1 bg-transparent px-4 py-4 text-base text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none"
          />

          <button
            type="submit"
            disabled={isLoading || query.trim().length < 3}
            className="mr-2 flex items-center gap-2 rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Searching
              </>
            ) : (
              "Search"
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
