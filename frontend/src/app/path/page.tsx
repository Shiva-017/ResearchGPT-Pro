// src/app/path/page.tsx

"use client";

import { useState, useRef } from "react";
import {
  GraduationCap,
  Navigation,
  Loader2,
  AlertCircle,
  ExternalLink,
  FileText,
  ArrowRight,
  Link2,
  Unlink,
  Clock,
  CheckCircle,
  MapPin,
  Flag,
} from "lucide-react";
import {
  findResearchPath,
  ResearchPathResponse,
  PathStep,
} from "@/lib/api";

export default function PathPage() {
  const [startTopic, setStartTopic] = useState("");
  const [endTopic, setEndTopic] = useState("");
  const [numSteps, setNumSteps] = useState(4);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchPathResponse | null>(null);

  const handleFind = async () => {
    if (startTopic.trim().length < 2 || endTopic.trim().length < 2) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await findResearchPath(startTopic.trim(), endTopic.trim(), numSteps);
      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
      }
    } catch (err: any) {
      setError(err.message || "Failed to find path");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleFind();
  };

  return (
    <main className="min-h-screen">
      <div className="mx-auto max-w-4xl px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <a href="/" className="flex items-center gap-2 hover:opacity-80">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700">
              <GraduationCap className="h-4 w-4 text-white" />
            </div>
            <span className="text-sm font-bold tracking-tight">
              ResearchGPT<span className="text-brand-400"> Pro</span>
            </span>
          </a>
          <span className="text-[var(--text-muted)] text-sm">›</span>
          <div className="flex items-center gap-1.5">
            <Navigation className="h-4 w-4 text-brand-400" />
            <span className="text-sm font-semibold text-brand-300">Research GPS</span>
          </div>
        </div>

        <p className="text-sm text-[var(--text-muted)] mb-8 ml-11">
          Find the optimal learning path between two research topics through the citation graph.
        </p>

        {/* Input form */}
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-[1fr,auto,1fr] gap-4 items-end">
            {/* Start topic */}
            <div>
              <label className="flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)] mb-2">
                <MapPin className="h-3 w-3 text-emerald-400" />
                I currently know about
              </label>
              <input
                type="text"
                value={startTopic}
                onChange={(e) => setStartTopic(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. CNNs, convolutional neural networks"
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-card)] px-4 py-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-emerald-500"
              />
            </div>

            {/* Arrow */}
            <div className="flex items-center justify-center pb-1">
              <ArrowRight className="h-5 w-5 text-[var(--text-muted)]" />
            </div>

            {/* End topic */}
            <div>
              <label className="flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)] mb-2">
                <Flag className="h-3 w-3 text-rose-400" />
                I want to learn about
              </label>
              <input
                type="text"
                value={endTopic}
                onChange={(e) => setEndTopic(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. GPU kernel programming, CUDA optimization"
                className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-card)] px-4 py-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-rose-500"
              />
            </div>
          </div>

          {/* Options row */}
          <div className="flex items-center justify-between mt-4">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                Steps:
                <select
                  value={numSteps}
                  onChange={(e) => setNumSteps(parseInt(e.target.value))}
                  className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] px-2 py-1 text-sm text-[var(--text-primary)] outline-none"
                >
                  {[3, 4, 5, 6].map((n) => (
                    <option key={n} value={n}>{n} papers</option>
                  ))}
                </select>
              </label>
            </div>

            <button
              onClick={handleFind}
              disabled={isLoading || startTopic.trim().length < 2 || endTopic.trim().length < 2}
              className="flex items-center gap-2 rounded-xl bg-brand-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Finding path...
                </>
              ) : (
                <>
                  <Navigation className="h-4 w-4" /> Find Path
                </>
              )}
            </button>
          </div>
        </div>

        {/* Suggestions */}
        {!result && !isLoading && (
          <div className="mb-8">
            <p className="text-xs text-[var(--text-muted)] mb-3">Try these paths:</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {[
                { from: "CNNs", to: "GPU kernel optimization" },
                { from: "Logistic regression", to: "Large language models" },
                { from: "Graph theory", to: "Graph neural networks" },
                { from: "Bayesian statistics", to: "Diffusion models" },
              ].map((s) => (
                <button
                  key={s.from + s.to}
                  onClick={() => {
                    setStartTopic(s.from);
                    setEndTopic(s.to);
                  }}
                  className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-xs text-[var(--text-secondary)] hover:border-brand-500/40 text-left"
                >
                  <span className="text-emerald-400">{s.from}</span>
                  <ArrowRight className="h-3 w-3 text-[var(--text-muted)] shrink-0" />
                  <span className="text-rose-400">{s.to}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-6 flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-500/5 px-4 py-3">
            <AlertCircle className="h-5 w-5 text-rose-400 shrink-0" />
            <p className="text-sm text-rose-300">{error}</p>
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex flex-col items-center py-16 animate-fadeIn">
            <Loader2 className="h-8 w-8 text-brand-400 animate-spin mb-4" />
            <p className="text-sm text-[var(--text-secondary)]">
              Finding the optimal learning path...
            </p>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Searching embeddings, traversing citation graph, generating explanations
            </p>
          </div>
        )}

        {/* Results — Timeline */}
        {result && result.path.length > 0 && (
          <div className="animate-fadeIn">
            {/* Stats bar */}
            <div className="flex flex-wrap items-center gap-4 mb-6 text-xs text-[var(--text-muted)]">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {result.timings.total_ms?.toFixed(0)}ms
              </span>
              <span>{result.total_papers_on_path} papers</span>
              <span className="flex items-center gap-1">
                <Link2 className="h-3 w-3" />
                {(result.citation_coverage * 100).toFixed(0)}% citation coverage
              </span>
            </div>

            {/* Timeline */}
            <div className="relative">
              {result.path.map((step, i) => {
                const isStart = i === 0;
                const isEnd = i === result.path.length - 1;
                const isMiddle = !isStart && !isEnd;

                return (
                  <div
                    key={step.paper_id + i}
                    className="relative flex gap-4 pb-8 animate-fadeIn"
                    style={{ animationDelay: `${i * 100}ms` }}
                  >
                    {/* Timeline line + node */}
                    <div className="flex flex-col items-center shrink-0 w-10">
                      {/* Node */}
                      <div
                        className={`relative z-10 flex h-10 w-10 items-center justify-center rounded-full border-2 ${
                          isStart
                            ? "border-emerald-500 bg-emerald-500/20 text-emerald-400"
                            : isEnd
                            ? "border-rose-500 bg-rose-500/20 text-rose-400"
                            : "border-brand-500 bg-brand-500/20 text-brand-400"
                        }`}
                      >
                        {isStart ? (
                          <MapPin className="h-4 w-4" />
                        ) : isEnd ? (
                          <Flag className="h-4 w-4" />
                        ) : (
                          <span className="text-xs font-bold">{i}</span>
                        )}
                      </div>

                      {/* Connecting line */}
                      {!isEnd && (
                        <div className="flex-1 flex flex-col items-center mt-1">
                          {step.has_citation_link ? (
                            <div className="w-0.5 flex-1 bg-gradient-to-b from-brand-500/60 to-brand-500/20" />
                          ) : (
                            <div className="w-0.5 flex-1 border-l-2 border-dashed border-[var(--border)]" />
                          )}
                          {/* Link indicator */}
                          <div className="my-1">
                            {step.has_citation_link ? (
                              <Link2 className="h-3 w-3 text-brand-400" />
                            ) : (
                              <Unlink className="h-3 w-3 text-[var(--text-muted)]" />
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Paper card */}
                    <div className="flex-1 rounded-xl border border-[var(--border)] bg-[var(--bg-card)] hover:border-brand-500/30 p-4 mb-1">
                      {/* Label */}
                      <div className="flex items-center justify-between mb-2">
                        <span
                          className={`text-[10px] font-bold uppercase tracking-wider ${
                            isStart
                              ? "text-emerald-400"
                              : isEnd
                              ? "text-rose-400"
                              : "text-brand-400"
                          }`}
                        >
                          {isStart ? "📍 You are here" : isEnd ? "🎯 Your goal" : `Step ${i}`}
                        </span>
                        <div className="flex items-center gap-2">
                          {step.citations > 0 && (
                            <span className="text-[10px] text-[var(--text-muted)]">
                              {step.citations} citations
                            </span>
                          )}
                          <span className="rounded-md bg-brand-500/10 px-1.5 py-0.5 text-[10px] text-brand-300">
                            {step.year}
                          </span>
                        </div>
                      </div>

                      {/* Title */}
                      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-1.5 leading-snug">
                        {step.title}
                      </h3>

                      {/* Why read */}
                      {step.why_read && (
                        <p className="text-xs text-brand-300 bg-brand-500/5 rounded-lg px-3 py-2 mb-2 border border-brand-500/10">
                          💡 {step.why_read}
                        </p>
                      )}

                      {/* Abstract */}
                      <p className="text-xs text-[var(--text-secondary)] leading-relaxed mb-2">
                        {step.abstract.length > 200
                          ? step.abstract.slice(0, 200) + "…"
                          : step.abstract}
                      </p>

                      {/* Footer */}
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-[var(--text-muted)]">
                          {step.authors?.split(",").slice(0, 2).join(", ")}
                          {(step.authors?.split(",").length || 0) > 2 ? " et al." : ""}
                        </span>
                        <div className="flex items-center gap-2">
                          <a
                            href={`https://arxiv.org/abs/${step.paper_id.replace("arxiv:", "")}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-brand-400"
                          >
                            <FileText className="h-3 w-3" /> arXiv
                          </a>
                          {step.pdf_url && (
                            <a
                              href={step.pdf_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-brand-400"
                            >
                              <ExternalLink className="h-3 w-3" /> PDF
                            </a>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Completion badge */}
              <div className="flex items-center gap-3 ml-10 pt-2">
                <CheckCircle className="h-5 w-5 text-emerald-400" />
                <span className="text-sm text-emerald-300">
                  Path complete — {result.total_papers_on_path} papers from{" "}
                  <strong>{result.start_topic}</strong> to{" "}
                  <strong>{result.end_topic}</strong>
                </span>
              </div>
            </div>
          </div>
        )}

        {/* No path */}
        {result && result.path.length === 0 && !error && (
          <div className="flex flex-col items-center py-16">
            <Unlink className="h-10 w-10 text-[var(--text-muted)] mb-4" />
            <p className="text-[var(--text-secondary)]">
              No path found between these topics
            </p>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Try broader topics or increase the number of steps
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
