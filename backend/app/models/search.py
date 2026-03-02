# backend/app/models/search.py

"""
Pydantic models for search requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ── Request ──────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """What the user sends."""
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of papers to return")
    use_hyde: bool = Field(default=True, description="Use HyDE for query expansion")
    use_rerank: bool = Field(default=True, description="Use cross-encoder reranking")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense vs sparse weight (1=pure dense)")

    # Optional filters
    year_min: Optional[int] = Field(default=None, description="Filter: minimum year")
    year_max: Optional[int] = Field(default=None, description="Filter: maximum year")
    categories: Optional[List[str]] = Field(default=None, description="Filter: arXiv categories")
    chunk_type: Optional[str] = Field(default=None, description="Filter by chunk type (deprecated in v2)")


# ── Response ─────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    """A single chunk hit from Pinecone."""
    chunk_id: str
    chunk_type: str = "full"     # v2: "full" | legacy: "problem" | "method"
    section_heading: str = ""    # e.g. "3.2 Multi-Head Attention"
    score: float                 # similarity score
    rerank_score: Optional[float] = None  # cross-encoder score (if reranked)
    snippet: str = ""            # actual chunk text (for LLM context)


class PaperResult(BaseModel):
    """A paper with its best-matching chunks grouped together."""
    paper_id: str
    title: str
    abstract: str
    authors: str
    year: int
    categories: str
    pdf_url: str
    field: str
    citation_tier: str
    has_code: bool
    is_survey: bool

    # Relevance
    best_score: float            # highest score among its chunks
    rerank_score: Optional[float] = None
    matched_chunks: List[ChunkResult]


class SearchResponse(BaseModel):
    """What the API returns."""
    query: str
    answer: Optional[str] = None          # LLM-generated answer from retrieved papers
    hyde_abstract: Optional[str] = None   # the generated hypothetical (for transparency)
    total_results: int
    papers: List[PaperResult]

    # Performance / debug
    timings: Dict[str, float] = Field(default_factory=dict)
    tokens_used: int = 0
    estimated_cost: float = 0.0


# ── Chat models ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    """Chat endpoint — supports conversation history + search."""
    message: str = Field(..., min_length=1, max_length=1000)
    history: List[ChatMessage] = Field(default_factory=list, description="Previous messages")
    top_k: int = Field(default=5, ge=1, le=20)
    use_hyde: bool = True
    use_rerank: bool = True
    alpha: float = 0.7
    year_min: Optional[int] = None
    year_max: Optional[int] = None
