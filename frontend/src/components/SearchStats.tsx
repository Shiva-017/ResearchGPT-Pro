// src/components/SearchStats.tsx

"use client";

import { Clock, Zap, DollarSign, Hash } from "lucide-react";

interface SearchStatsProps {
  timings: Record<string, number>;
  tokens_used: number;
  estimated_cost: number;
  total_results: number;
  queryType?: string;
}

const AGENT_LABELS: Record<string, { label: string; color: string }> = {
  SEARCH: { label: "Search", color: "text-blue-400" },
  GRAPH: { label: "Graph", color: "text-emerald-400" },
  TRENDING: { label: "Trending", color: "text-amber-400" },
  PATH: { label: "Path", color: "text-purple-400" },
  FOLLOWUP: { label: "Follow-up", color: "text-sky-400" },
  COMPARE: { label: "Compare", color: "text-rose-400" },
  GAPS: { label: "Gaps", color: "text-orange-400" },
  DIRECT: { label: "Direct", color: "text-gray-400" },
};

export default function SearchStats({
  timings,
  tokens_used,
  estimated_cost,
  total_results,
  queryType,
}: SearchStatsProps) {
  const totalMs = timings.total_ms ?? 0;
  const agent = queryType ? AGENT_LABELS[queryType] : null;

  const stages = [
    { key: "classify_ms", label: "Route", color: "text-pink-400" },
    { key: "hyde_ms", label: "HyDE", color: "text-purple-400" },
    { key: "embed_ms", label: "Embed", color: "text-blue-400" },
    { key: "pinecone_ms", label: "Pinecone", color: "text-emerald-400" },
    { key: "rerank_ms", label: "Rerank", color: "text-amber-400" },
    { key: "graph_total_ms", label: "Graph", color: "text-teal-400" },
    { key: "answer_ms", label: "Answer", color: "text-cyan-400" },
  ];

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs text-[var(--text-muted)]">
      {agent && (
        <span className={`font-semibold ${agent.color}`}>
          ▸ {agent.label} agent
        </span>
      )}

      <span className="flex items-center gap-1">
        <Hash className="h-3 w-3" />
        {total_results} paper{total_results !== 1 ? "s" : ""}
      </span>

      <span className="flex items-center gap-1">
        <Clock className="h-3 w-3" />
        {totalMs.toFixed(0)}ms total
      </span>

      {/* Stage breakdown */}
      {stages.map(
        ({ key, label, color }) =>
          timings[key] != null && (
            <span key={key} className={`${color}`}>
              {label}: {timings[key].toFixed(0)}ms
            </span>
          )
      )}

      <span className="flex items-center gap-1">
        <Zap className="h-3 w-3" />
        {tokens_used.toLocaleString()} tokens
      </span>

      <span className="flex items-center gap-1">
        <DollarSign className="h-3 w-3" />${estimated_cost.toFixed(6)}
      </span>
    </div>
  );
}
