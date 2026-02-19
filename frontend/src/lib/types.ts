// src/lib/types.ts

/**
 * TypeScript types matching the backend Pydantic models.
 */

export interface SearchRequest {
  query: string;
  top_k?: number;
  use_hyde?: boolean;
  use_rerank?: boolean;
  alpha?: number;
  year_min?: number | null;
  year_max?: number | null;
  categories?: string[] | null;
  chunk_type?: string | null;
}

export interface ChunkResult {
  chunk_id: string;
  chunk_type: "problem" | "method" | string;
  score: number;
  rerank_score: number | null;
}

export interface PaperResult {
  paper_id: string;
  title: string;
  abstract: string;
  authors: string;
  year: number;
  categories: string;
  pdf_url: string;
  field: string;
  citation_tier: string;
  has_code: boolean;
  is_survey: boolean;
  best_score: number;
  rerank_score: number | null;
  matched_chunks: ChunkResult[];
}

export interface GraphPaper {
  id: string;
  title: string;
  year: number;
  citations: number;
  shared_connections?: number;
  cited_by_count?: number;
  field?: string;
}

export interface GraphExpert {
  name: string;
  papers_in_cluster: number;
  total_citations: number;
}

export interface GraphInsights {
  connected_papers?: GraphPaper[];
  seminal_papers?: GraphPaper[];
  top_experts?: GraphExpert[];
  papers_with_context?: number;
}

export interface SearchResponse {
  query: string;
  answer: string | null;
  hyde_abstract: string | null;
  total_results: number;
  papers: PaperResult[];
  timings: Record<string, number>;
  tokens_used: number;
  estimated_cost: number;
  graph_insights?: GraphInsights;
}
