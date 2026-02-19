// src/components/PaperCard.tsx

"use client";

import {
  FileText,
  ExternalLink,
  Code,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Layers,
} from "lucide-react";
import { useState } from "react";
import { PaperResult } from "@/lib/types";

interface PaperCardProps {
  paper: PaperResult;
  rank: number;
}

export default function PaperCard({ paper, rank }: PaperCardProps) {
  const [expanded, setExpanded] = useState(false);

  const scoreDisplay = paper.rerank_score
    ? paper.rerank_score.toFixed(3)
    : paper.best_score.toFixed(3);

  const scoreLabel = paper.rerank_score ? "rerank" : "similarity";

  // Parse categories string into array
  const cats = paper.categories
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean);

  // Parse authors
  const authorList = paper.authors.split(",").map((a) => a.trim());
  const displayAuthors =
    authorList.length > 3
      ? `${authorList.slice(0, 3).join(", ")} +${authorList.length - 3} more`
      : authorList.join(", ");

  return (
    <div
      className="group rounded-xl border border-[var(--border)] bg-[var(--bg-card)] hover:border-brand-500/40 hover:bg-[var(--bg-card-hover)] animate-fadeIn"
      style={{ animationDelay: `${rank * 60}ms` }}
    >
      <div className="p-5">
        {/* Header row: rank, score, badges */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600/20 text-xs font-bold text-brand-400">
              {rank}
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              {scoreLabel}: {scoreDisplay}
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {paper.has_code && (
              <span className="flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-400">
                <Code className="h-3 w-3" /> Code
              </span>
            )}
            {paper.is_survey && (
              <span className="flex items-center gap-1 rounded-md bg-amber-500/10 px-2 py-0.5 text-xs text-amber-400">
                <BookOpen className="h-3 w-3" /> Survey
              </span>
            )}
            <span className="rounded-md bg-brand-500/10 px-2 py-0.5 text-xs text-brand-300">
              {paper.year}
            </span>
          </div>
        </div>

        {/* Title */}
        <h3 className="text-[15px] font-semibold leading-snug text-[var(--text-primary)] mb-2">
          {paper.title}
        </h3>

        {/* Authors */}
        <p className="text-xs text-[var(--text-muted)] mb-2">{displayAuthors}</p>

        {/* Categories */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {cats.slice(0, 5).map((cat) => (
            <span
              key={cat}
              className="rounded-md border border-[var(--border)] px-2 py-0.5 text-[11px] text-[var(--text-muted)]"
            >
              {cat}
            </span>
          ))}
          {cats.length > 5 && (
            <span className="text-[11px] text-[var(--text-muted)]">
              +{cats.length - 5}
            </span>
          )}
        </div>

        {/* Abstract (truncated) */}
        <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
          {expanded
            ? paper.abstract
            : paper.abstract.length > 250
            ? paper.abstract.slice(0, 250) + "…"
            : paper.abstract}
        </p>

        {/* Expand / Actions row */}
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {paper.abstract.length > 250 && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300"
              >
                {expanded ? (
                  <>
                    <ChevronUp className="h-3 w-3" /> Less
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-3 w-3" /> More
                  </>
                )}
              </button>
            )}

            {/* Chunk info */}
            <span className="flex items-center gap-1 text-xs text-[var(--text-muted)]">
              <Layers className="h-3 w-3" />
              {paper.matched_chunks.length} chunk
              {paper.matched_chunks.length > 1 ? "s" : ""} matched
              {paper.matched_chunks.map((c) => (
                <span
                  key={c.chunk_id}
                  className={`ml-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    c.chunk_type === "problem"
                      ? "bg-rose-500/10 text-rose-400"
                      : "bg-sky-500/10 text-sky-400"
                  }`}
                >
                  {c.chunk_type}
                </span>
              ))}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <a
              href={`https://arxiv.org/abs/${paper.paper_id.replace("arxiv:", "")}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-brand-400"
            >
              <FileText className="h-3 w-3" /> arXiv
            </a>
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-brand-400"
            >
              <ExternalLink className="h-3 w-3" /> PDF
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
