// src/lib/api.ts

import { SearchRequest, SearchResponse, PaperResult, GraphInsights } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

// ── Original search (non-streaming) ─────────────────────────────────

export async function searchPapers(
  request: SearchRequest
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`Search failed (${res.status}): ${error}`);
  }
  return res.json();
}

// ── Chat (streaming) ────────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatMessage[];
  top_k?: number;
  use_hyde?: boolean;
  use_rerank?: boolean;
  alpha?: number;
  year_min?: number | null;
  year_max?: number | null;
}

export interface SourcesEvent {
  type: "sources";
  papers: PaperResult[];
  timings: Record<string, number>;
  hyde_abstract: string | null;
  total_results: number;
  graph_insights?: GraphInsights;
  query_type?: string;
  path_result?: any;
}

export interface TokenEvent {
  type: "token";
  content: string;
}

export interface DoneEvent {
  type: "done";
}

type SSEEvent = SourcesEvent | TokenEvent | DoneEvent;

/**
 * Stream chat response via SSE.
 *
 * Calls onSources() immediately when papers arrive,
 * then onToken() for each streamed word,
 * then onDone() when complete.
 */
export async function streamChat(
  request: ChatRequest,
  callbacks: {
    onSources: (data: SourcesEvent) => void;
    onToken: (token: string) => void;
    onDone: () => void;
    onError: (error: string) => void;
  }
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!res.ok) {
      const error = await res.text();
      callbacks.onError(`Chat failed (${res.status}): ${error}`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      callbacks.onError("No response body");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event: SSEEvent = JSON.parse(jsonStr);

          if (event.type === "sources") {
            callbacks.onSources(event as SourcesEvent);
          } else if (event.type === "token") {
            callbacks.onToken((event as TokenEvent).content);
          } else if (event.type === "done") {
            callbacks.onDone();
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }

    // Ensure onDone is called
    callbacks.onDone();
  } catch (err: any) {
    callbacks.onError(err.message || "Connection failed");
  }
}

// ── Research Path ─────────────────────────────────────────────────

export interface PathStep {
  paper_id: string;
  title: string;
  year: number;
  abstract: string;
  citations: number;
  authors: string;
  pdf_url: string;
  why_read: string;
  position: number;
  has_citation_link: boolean;
}

export interface ResearchPathResponse {
  start_topic: string;
  end_topic: string;
  path: PathStep[];
  total_papers_on_path: number;
  citation_coverage: number;
  timings: Record<string, number>;
  error?: string;
}

export async function findResearchPath(
  startTopic: string,
  endTopic: string,
  numSteps: number = 4
): Promise<ResearchPathResponse> {
  const params = new URLSearchParams({
    start_topic: startTopic,
    end_topic: endTopic,
    num_steps: numSteps.toString(),
  });
  const res = await fetch(`${API_BASE}/research-path?${params}`, {
    method: "POST",
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`Path finding failed (${res.status}): ${error}`);
  }
  return res.json();
}

// ── Health ───────────────────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}
