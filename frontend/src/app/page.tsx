// src/app/page.tsx

"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { GraduationCap, AlertCircle, Send, Loader2, Navigation } from "lucide-react";
import {
  streamChat,
  ChatMessage as APIChatMessage,
  SourcesEvent,
} from "@/lib/api";
import { PaperResult, GraphInsights as GraphInsightsType } from "@/lib/types";
import PaperCard from "@/components/PaperCard";
import AnswerCard from "@/components/AnswerCard";
import GraphInsights from "@/components/GraphInsights";
import SearchStats from "@/components/SearchStats";
import SearchFilters, {
  FilterState,
  DEFAULT_FILTERS,
} from "@/components/SearchFilters";

// ── Types ───────────────────────────────────────────────────────────

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  papers?: PaperResult[];
  timings?: Record<string, number>;
  graphInsights?: GraphInsightsType;
  queryType?: string;
  isStreaming?: boolean;
}

// ── Main ────────────────────────────────────────────────────────────

export default function Home() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = useCallback(async () => {
    const query = input.trim();
    if (query.length < 3 || isLoading) return;

    setInput("");
    setError(null);
    setIsLoading(true);

    const userTurnId = `user-${Date.now()}`;
    const assistantTurnId = `asst-${Date.now()}`;

    // Add user message
    setTurns((prev) => [
      ...prev,
      { id: userTurnId, role: "user", content: query },
    ]);

    // Add empty assistant message (will fill via streaming)
    setTurns((prev) => [
      ...prev,
      { id: assistantTurnId, role: "assistant", content: "", isStreaming: true },
    ]);

    // Build history for API (previous turns only)
    const history: APIChatMessage[] = turns
      .filter((t) => t.content)
      .map((t) => ({ role: t.role, content: t.content }));

    await streamChat(
      {
        message: query,
        history,
        top_k: filters.top_k,
        use_hyde: filters.use_hyde,
        use_rerank: filters.use_rerank,
        alpha: filters.alpha,
        year_min: filters.year_min,
        year_max: filters.year_max,
      },
      {
        onSources: (data: SourcesEvent) => {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantTurnId
                ? {
                    ...t,
                    papers: data.papers,
                    timings: data.timings,
                    graphInsights: data.graph_insights,
                    queryType: data.query_type,
                  }
                : t
            )
          );
        },
        onToken: (token: string) => {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantTurnId
                ? { ...t, content: t.content + token }
                : t
            )
          );
        },
        onDone: () => {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantTurnId
                ? { ...t, isStreaming: false }
                : t
            )
          );
          setIsLoading(false);
        },
        onError: (err: string) => {
          setError(err);
          setTurns((prev) =>
            prev.map((t) =>
              t.id === assistantTurnId
                ? { ...t, content: "_Failed to generate answer._", isStreaming: false }
                : t
            )
          );
          setIsLoading(false);
        },
      }
    );
  }, [input, isLoading, turns, filters]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (q: string) => {
    setInput(q);
    // Small delay so user sees the input before sending
    setTimeout(() => {
      setInput(q);
      const fakeEvent = { trim: () => q, length: q.length };
      // Trigger send directly
    }, 50);
  };

  const isEmpty = turns.length === 0;

  return (
    <main className="flex flex-col h-screen">
      {/* ── Chat area ────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4">
          {/* Landing state */}
          {isEmpty && (
            <div className="flex flex-col items-center pt-[20vh] animate-fadeIn">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-lg shadow-brand-500/20">
                  <GraduationCap className="h-6 w-6 text-white" />
                </div>
                <h1 className="text-3xl font-bold tracking-tight">
                  ResearchGPT
                  <span className="text-brand-400"> Pro</span>
                </h1>
              </div>
              <p className="text-sm text-[var(--text-muted)] text-center max-w-md mb-8">
                Ask anything about computer science research. I&apos;ll search
                25,000+ papers and give you an answer with citations.
              </p>

              {/* Research GPS link */}
              <a
                href="/path"
                className="flex items-center gap-2 rounded-xl border border-brand-500/20 bg-brand-500/5 px-4 py-2.5 text-sm text-brand-300 hover:bg-brand-500/10 hover:border-brand-500/30 mb-6"
              >
                <Navigation className="h-4 w-4" />
                Research GPS — Find a learning path between topics
              </a>

              {/* Suggestions */}
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "How do transformers handle long documents?",
                  "What are the latest approaches to graph neural networks?",
                  "Compare different methods for federated learning",
                  "What techniques reduce LLM inference cost?",
                  "Explain diffusion models for image generation",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setInput(q);
                    }}
                    className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-sm text-[var(--text-secondary)] hover:border-brand-500/40 hover:text-brand-300 text-left"
                  >
                    {q}
                  </button>
                ))}
              </div>

              {/* Filters */}
              <div className="w-full mt-8 max-w-2xl">
                <SearchFilters filters={filters} onChange={setFilters} />
              </div>
            </div>
          )}

          {/* Compact header after first message */}
          {!isEmpty && (
            <div className="sticky top-0 z-10 bg-[var(--bg-primary)]/80 backdrop-blur-md border-b border-[var(--border)] py-3 mb-4 -mx-4 px-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700">
                    <GraduationCap className="h-3.5 w-3.5 text-white" />
                  </div>
                  <span className="text-sm font-bold tracking-tight">
                    ResearchGPT<span className="text-brand-400"> Pro</span>
                  </span>
                </div>
                <SearchFilters filters={filters} onChange={setFilters} />
              </div>
            </div>
          )}

          {/* Chat turns */}
          {turns.map((turn) => (
            <div key={turn.id} className="mb-6 animate-fadeIn">
              {turn.role === "user" ? (
                /* ── User message ── */
                <div className="flex justify-end mb-2">
                  <div className="max-w-[80%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-3 text-sm text-white">
                    {turn.content}
                  </div>
                </div>
              ) : (
                /* ── Assistant message ── */
                <div className="space-y-3">
                  {/* Stats */}
                  {turn.timings && (
                    <SearchStats
                      timings={turn.timings}
                      tokens_used={0}
                      estimated_cost={0}
                      total_results={turn.papers?.length || 0}
                      queryType={turn.queryType}
                    />
                  )}

                  {/* Streamed answer */}
                  {(turn.content || turn.isStreaming) && (
                    <AnswerCard
                      answer={turn.content + (turn.isStreaming ? "▌" : "")}
                    />
                  )}

                  {/* Graph insights */}
                  {turn.graphInsights && (
                    <GraphInsights insights={turn.graphInsights} />
                  )}

                  {/* Source papers */}
                  {turn.papers && turn.papers.length > 0 && (
                    <>
                      <div className="flex items-center gap-2 pt-1">
                        <div className="h-px flex-1 bg-[var(--border)]" />
                        <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                          Sources ({turn.papers.length})
                        </span>
                        <div className="h-px flex-1 bg-[var(--border)]" />
                      </div>
                      <div className="space-y-2">
                        {turn.papers.map((paper, i) => (
                          <PaperCard
                            key={paper.paper_id}
                            paper={paper}
                            rank={i + 1}
                          />
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Error */}
          {error && (
            <div className="mb-4 flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-500/5 px-4 py-3">
              <AlertCircle className="h-5 w-5 text-rose-400 shrink-0" />
              <p className="text-sm text-rose-300">{error}</p>
            </div>
          )}

          <div ref={chatEndRef} className="h-4" />
        </div>
      </div>

      {/* ── Input bar ────────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-[var(--border)] bg-[var(--bg-primary)]">
        <div className="mx-auto max-w-4xl px-4 py-4">
          <div className="relative flex items-center rounded-2xl border border-[var(--border)] bg-[var(--bg-secondary)] focus-within:border-[var(--border-focus)] focus-within:ring-1 focus-within:ring-brand-500/30">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isEmpty
                  ? "Ask about research papers..."
                  : "Ask a follow-up question..."
              }
              disabled={isLoading}
              className="flex-1 bg-transparent px-5 py-3.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || input.trim().length < 3}
              className="mr-2 flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-[var(--text-muted)]">
            Answers are AI-generated from retrieved papers. Always verify with
            the original source.
          </p>
        </div>
      </div>
    </main>
  );
}
