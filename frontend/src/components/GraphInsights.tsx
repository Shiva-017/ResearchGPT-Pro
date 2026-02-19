// src/components/GraphInsights.tsx

"use client";

import { GitBranch, Users, Star, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface GraphPaper {
  id: string;
  title: string;
  year: number;
  citations: number;
  shared_connections?: number;
  cited_by_count?: number;
  field?: string;
}

interface GraphExpert {
  name: string;
  papers_in_cluster: number;
  total_citations: number;
}

interface GraphInsightsProps {
  insights: {
    connected_papers?: GraphPaper[];
    seminal_papers?: GraphPaper[];
    top_experts?: GraphExpert[];
    papers_with_context?: number;
  };
}

export default function GraphInsights({ insights }: GraphInsightsProps) {
  const [open, setOpen] = useState(true);

  const hasContent =
    (insights.connected_papers?.length || 0) > 0 ||
    (insights.seminal_papers?.length || 0) > 0 ||
    (insights.top_experts?.length || 0) > 0;

  if (!hasContent) return null;

  return (
    <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.03]">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left"
      >
        <GitBranch className="h-4 w-4 text-emerald-400 shrink-0" />
        <span className="text-sm font-medium text-emerald-300">
          Knowledge Graph Insights
        </span>
        <span className="text-xs text-[var(--text-muted)] ml-1">
          — citation network, key researchers, foundational papers
        </span>
        <span className="ml-auto">
          {open ? (
            <ChevronUp className="h-3 w-3 text-emerald-400" />
          ) : (
            <ChevronDown className="h-3 w-3 text-emerald-400" />
          )}
        </span>
      </button>

      {open && (
        <div className="border-t border-emerald-500/10 px-4 py-3 space-y-4 animate-slideDown">
          {/* Seminal papers */}
          {insights.seminal_papers && insights.seminal_papers.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Star className="h-3.5 w-3.5 text-amber-400" />
                <span className="text-xs font-semibold text-amber-300 uppercase tracking-wider">
                  Foundational Papers
                </span>
              </div>
              <div className="space-y-1.5">
                {insights.seminal_papers.map((p) => (
                  <div
                    key={p.id}
                    className="flex items-baseline gap-2 text-sm text-[var(--text-secondary)]"
                  >
                    <span className="shrink-0 text-xs text-amber-400/60">▸</span>
                    <span>
                      {p.title?.slice(0, 65)}{(p.title?.length || 0) > 65 ? "…" : ""}
                    </span>
                    <span className="shrink-0 text-xs text-[var(--text-muted)]">
                      ({p.year}) {p.citations} cites
                      {p.cited_by_count ? ` · cited by ${p.cited_by_count} results` : ""}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connected papers */}
          {insights.connected_papers && insights.connected_papers.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <GitBranch className="h-3.5 w-3.5 text-emerald-400" />
                <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
                  Related (from citation graph)
                </span>
              </div>
              <div className="space-y-1.5">
                {insights.connected_papers.slice(0, 5).map((p) => (
                  <div
                    key={p.id}
                    className="flex items-baseline gap-2 text-sm text-[var(--text-secondary)]"
                  >
                    <span className="shrink-0 text-xs text-emerald-400/60">▸</span>
                    <span>
                      {p.title?.slice(0, 65)}{(p.title?.length || 0) > 65 ? "…" : ""}
                    </span>
                    <span className="shrink-0 text-xs text-[var(--text-muted)]">
                      ({p.year}) {p.citations} cites
                      {p.shared_connections ? ` · ${p.shared_connections} shared links` : ""}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Key researchers */}
          {insights.top_experts && insights.top_experts.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Users className="h-3.5 w-3.5 text-sky-400" />
                <span className="text-xs font-semibold text-sky-300 uppercase tracking-wider">
                  Key Researchers
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {insights.top_experts.map((e) => (
                  <span
                    key={e.name}
                    className="rounded-lg border border-sky-500/20 bg-sky-500/5 px-2.5 py-1 text-xs text-sky-300"
                  >
                    {e.name}
                    <span className="text-[var(--text-muted)] ml-1">
                      {e.papers_in_cluster}p · {e.total_citations}c
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
