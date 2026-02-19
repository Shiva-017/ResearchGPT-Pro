// src/components/SearchFilters.tsx

"use client";

import { SlidersHorizontal, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

export interface FilterState {
  use_hyde: boolean;
  use_rerank: boolean;
  alpha: number;
  top_k: number;
  year_min: number | null;
  year_max: number | null;
  chunk_type: string | null;
}

interface SearchFiltersProps {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
}

export const DEFAULT_FILTERS: FilterState = {
  use_hyde: true,
  use_rerank: true,
  alpha: 0.7,
  top_k: 5,
  year_min: null,
  year_max: null,
  chunk_type: null,
};

export default function SearchFilters({
  filters,
  onChange,
}: SearchFiltersProps) {
  const [open, setOpen] = useState(false);

  const update = (patch: Partial<FilterState>) =>
    onChange({ ...filters, ...patch });

  return (
    <div className="w-full">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
      >
        <SlidersHorizontal className="h-4 w-4" />
        Search Settings
        {open ? (
          <ChevronUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        )}
      </button>

      {open && (
        <div className="mt-3 grid grid-cols-2 gap-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4 sm:grid-cols-3 lg:grid-cols-6 animate-slideDown">
          {/* HyDE toggle */}
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={filters.use_hyde}
              onChange={(e) => update({ use_hyde: e.target.checked })}
              className="accent-brand-500 h-4 w-4"
            />
            <span className="text-[var(--text-secondary)]">HyDE</span>
          </label>

          {/* Rerank toggle */}
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={filters.use_rerank}
              onChange={(e) => update({ use_rerank: e.target.checked })}
              className="accent-brand-500 h-4 w-4"
            />
            <span className="text-[var(--text-secondary)]">Rerank</span>
          </label>

          {/* Alpha slider */}
          <div className="flex flex-col gap-1">
            <span className="text-xs text-[var(--text-muted)]">
              Dense/Sparse: {filters.alpha.toFixed(1)}
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filters.alpha}
              onChange={(e) => update({ alpha: parseFloat(e.target.value) })}
              className="accent-brand-500"
            />
          </div>

          {/* Top K */}
          <div className="flex flex-col gap-1">
            <span className="text-xs text-[var(--text-muted)]">Results</span>
            <select
              value={filters.top_k}
              onChange={(e) => update({ top_k: parseInt(e.target.value) })}
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] px-2 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-brand-500"
            >
              {[5, 10, 15, 20, 30].map((n) => (
                <option key={n} value={n}>
                  {n} papers
                </option>
              ))}
            </select>
          </div>

          {/* Year min */}
          <div className="flex flex-col gap-1">
            <span className="text-xs text-[var(--text-muted)]">Year from</span>
            <input
              type="number"
              min="2000"
              max="2026"
              value={filters.year_min ?? ""}
              placeholder="Any"
              onChange={(e) =>
                update({
                  year_min: e.target.value ? parseInt(e.target.value) : null,
                })
              }
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] px-2 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-brand-500 w-full"
            />
          </div>

          {/* Year max */}
          <div className="flex flex-col gap-1">
            <span className="text-xs text-[var(--text-muted)]">Year to</span>
            <input
              type="number"
              min="2000"
              max="2026"
              value={filters.year_max ?? ""}
              placeholder="Any"
              onChange={(e) =>
                update({
                  year_max: e.target.value ? parseInt(e.target.value) : null,
                })
              }
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] px-2 py-1.5 text-sm text-[var(--text-primary)] outline-none focus:border-brand-500 w-full"
            />
          </div>
        </div>
      )}
    </div>
  );
}
