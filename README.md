# ResearchGPT Pro

**An AI-powered academic research assistant combining hybrid vector-graph retrieval with conversational AI.**

ResearchGPT Pro searches 25,000+ computer science papers using a 7-stage retrieval pipeline (HyDE query expansion, hybrid dense+sparse search, cross-encoder reranking, knowledge graph enrichment) and delivers streaming AI answers with citations. It also features a Research GPS that finds optimal learning paths between research topics using semantic interpolation over the citation graph.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Concepts](#core-concepts)
3. [Data Pipeline: Ingestion](#data-pipeline-ingestion)
4. [Search Pipeline: Retrieval + Answer Generation](#search-pipeline-retrieval--answer-generation)
5. [Knowledge Graph: Neo4j](#knowledge-graph-neo4j)
6. [HybridRAG: Vector + Graph Intelligence](#hybridrag-vector--graph-intelligence)
7. [Research GPS: Learning Path Finder](#research-gps-learning-path-finder)
8. [Frontend Architecture](#frontend-architecture)
9. [Tech Stack](#tech-stack)
10. [Interview Prep: Key Questions & Answers](#interview-prep-key-questions--answers)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 14)                     │
│  Chat Interface  │  Paper Cards  │  Graph Insights  │  Research GPS │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSE (streaming) + REST
┌────────────────────────────▼────────────────────────────────────┐
│                      BACKEND (FastAPI)                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  SearchService                            │  │
│  │  HyDE → Embed → Pinecone → Rerank → Group → Graph → LLM │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  HybridRAG   │  │ ResearchPath │  │  EmbeddingService    │  │
│  │  (graph      │  │ Finder       │  │  (OpenAI wrapper)    │  │
│  │   enrichment)│  │ (GPS)        │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────┬───────────────────┬───────────────────┬────────────────┘
         │                   │                   │
┌────────▼─────┐  ┌─────────▼────────┐  ┌──────▼──────────┐
│   Pinecone   │  │     Neo4j        │  │   OpenAI API    │
│   (Vectors)  │  │   (Graph)        │  │   (LLM + Embed) │
│              │  │                  │  │                 │
│  50K chunks  │  │  102K nodes      │  │  GPT-4o-mini    │
│  1536-dim    │  │  590K edges      │  │  text-embed-3   │
│  hybrid      │  │  citations +     │  │  -small         │
│  dense+sparse│  │  co-authorship   │  │                 │
└──────────────┘  └──────────────────┘  └─────────────────┘
```

---

## Core Concepts

### 1. Dense vs Sparse Vectors

**Dense vectors** (from OpenAI embeddings): Every dimension has a value. Captures semantic meaning — "transformers are slow" and "attention is expensive" get similar vectors even though they share no words.

**Sparse vectors** (from BM25): Mostly zeros, with values only for words that appear. Captures exact keyword matches — "FlashAttention" gets a high score for queries containing that exact term.

**Why both?** Dense search finds meaning, sparse search finds exact terms. Combined with `alpha` weighting (70% dense, 30% sparse), they complement each other. This requires the `dotproduct` metric in Pinecone (not `cosine`) because sparse vectors use magnitude to encode importance.

### 2. HyDE (Hypothetical Document Embedding)

**Problem:** User queries are short and informal ("how to make LLMs faster"). Paper abstracts are long and technical. Their embeddings aren't directly comparable.

**Solution:** Ask GPT-4o-mini to generate a fake abstract that would answer the query. Embed that fake abstract instead. Now the search is comparing abstract-shaped text against abstract-shaped text — much better alignment.

**Implementation:** The final search vector is a weighted blend: `0.4 × original_query + 0.6 × hyde_abstract`. This keeps keyword grounding from the original while getting semantic precision from HyDE.

### 3. Cross-Encoder Reranking

**Problem:** Pinecone compares vectors numerically — fast but rough. It can't read the actual text.

**Solution:** After Pinecone returns 20-40 candidates, a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) reads each (query, document) pair together and gives a precise relevance score. This catches cases where vector similarity was misleading.

**Key difference from bi-encoder:** A bi-encoder embeds query and document separately then compares numbers. A cross-encoder feeds both into the same model simultaneously — slower but far more accurate.

### 4. Chunk Strategy (2 Chunks Per Paper)

Each paper is split into two embedding chunks:

- **Problem chunk:** Title (repeated for weight) + field + year + first 1/3 of abstract (problem statement)
- **Method chunk:** Title (repeated) + field + year + remaining 2/3 of abstract (approach + results)

**Why?** A query about "what problems exist in X" matches problem chunks better. A query about "how to implement Y" matches method chunks better. One vector per paper would blur this distinction.

**Filler cleaning:** Before splitting, ~25 boilerplate phrases ("we propose", "state-of-the-art", "extensive experiments") are stripped from abstracts so embeddings capture actual content.

### 5. Knowledge Graph (Neo4j)

A citation graph with:
- **Paper nodes** (102K): id, title, year, citations, field
- **Author nodes** (62K): name
- **Category nodes** (140): arXiv categories
- **CITES edges** (147K): paper-to-paper citation relationships
- **COAUTHORED_WITH edges** (440K): author-to-author collaboration links
- **AUTHORED_BY edges**: paper-to-author
- **IN_CATEGORY edges**: paper-to-category

This enables queries vector search can't do: "what cites this paper", "who collaborates with X", "what's the citation chain from A to B".

### 6. HybridRAG

After Pinecone returns ranked papers, Neo4j enriches each result with:
- What it cites and what cites it (citation context)
- Related highly-cited papers Pinecone missed (graph expansion)
- Seminal papers multiple results all cite (common ancestors)
- Key researchers in the topic cluster (author expertise)

This enriched context goes to the LLM, producing answers that mention citation lineage, foundational work, and key researchers — not just isolated abstracts.

---

## Data Pipeline: Ingestion

### Stage Flow

```
ArXiv API → Validate → Deduplicate → Fit BM25 → Embed (2 chunks) → Upsert Pinecone
                                                                          │
Semantic Scholar API → Enrich citations ──────────────────────── Ingest Neo4j
```

### Stage 1: Fetch (ArXiv)

Uses the `arxiv` Python library with `arxiv.Client` for built-in pagination and rate limiting (3 req/sec). Papers are cached to `backend/data/raw/` as JSON — re-runs skip already-fetched categories.

**Scale:** 625 papers × 40 CS categories = ~25,000 papers. Takes ~3-4 hours due to ArXiv rate limits.

### Stage 2: Validate

Pydantic v2 schema validation: title length (10-1000 chars), abstract minimum 50 chars, at least one author, year between 1990 and current year. Invalid papers are logged and skipped.

### Stage 3: Deduplicate

ArXiv papers have version numbers (e.g., `2301.12345v1`, `2301.12345v2`). The deduplicator normalizes IDs by stripping the version suffix and keeps only the latest version.

### Stage 4: Fit BM25

The `pinecone-text` BM25Encoder is fitted on the full corpus (title + abstract for each paper). This creates a vocabulary-specific sparse encoder saved to disk as `bm25_encoder.pkl`. Must be fitted before upserting.

### Stage 5: Embed

For each paper, `build_embedding_chunks()` creates 2 chunks. Each chunk is embedded via OpenAI `text-embedding-3-small` (1536 dimensions, $0.02/1M tokens). Embeddings are cached to disk — re-runs skip already-embedded chunks.

**Cost:** ~$0.50 for 25K papers (50K chunks).

### Stage 6: Upsert to Pinecone

Each chunk is upserted with:
- **Dense vector:** 1536-dim OpenAI embedding
- **Sparse vector:** BM25 encoding of chunk text
- **Metadata:** title, abstract, authors, year, categories, field, chunk_type, paper_id, has_code, is_survey, citation_tier

Batched at 100 vectors per upsert with checkpointing every 500. Progress is resumable if interrupted.

### Stage 7: Citation Enrichment (Semantic Scholar)

Uses the S2 batch API (`POST /paper/batch`, 500 papers per request) to fetch citation counts, references, and citing papers. Results cached per-paper to disk. Takes ~2 minutes for 25K papers via batch endpoint.

### Stage 8: Neo4j Graph Ingestion

Creates Paper, Author, and Category nodes with AUTHORED_BY, IN_CATEGORY, and CITES relationships. Then builds COAUTHORED_WITH edges by finding authors who share papers. Batched at 500 papers per transaction.

### Fault Tolerance

- **Checkpoint manager:** Tracks completed/embedded/failed papers in newline-delimited text files + JSONL error log
- **Set-based dedup:** In-memory sets for O(1) lookup, backed by text files
- **Resume:** Re-running without `--fresh` flag picks up where it left off at any stage
- **Retry:** Exponential backoff on API errors via `tenacity`

---

## Search Pipeline: Retrieval + Answer Generation

### Full 7-Stage Pipeline

```
User: "How do transformers handle long documents?"
  │
  ▼
Stage 1: HyDE (~150ms)
  GPT-4o-mini generates a hypothetical abstract
  "A method for extending transformer attention to sequences of 
   length 16K+ using sliding window and global attention patterns..."
  │
  ▼
Stage 2: Embed (~100ms)
  Embed original query AND HyDE abstract
  Blend: 0.4 × query_vec + 0.6 × hyde_vec
  │
  ▼
Stage 3: Pinecone Hybrid Search (~50ms)
  Dense: blended vector
  Sparse: BM25 encoding of original query
  Alpha: 0.7 (70% dense, 30% sparse)
  Returns: 20-40 candidate chunks
  │
  ▼
Stage 4: Cross-Encoder Rerank (~200ms)
  ms-marco-MiniLM-L-6-v2 reads each (query, chunk) pair
  Re-scores and re-sorts. Top 5 papers selected.
  │
  ▼
Stage 5: Group Chunks → Papers
  Merge problem + method chunks from same paper
  Paper score = best chunk score
  │
  ▼
Stage 6: Graph Enrichment (~200ms)
  For each paper: citations, references, author's other work
  Find connected papers Pinecone missed
  Find seminal papers (common ancestors)
  Find key researchers in cluster
  │
  ▼
Stage 7: Stream Answer (~500ms)
  GPT-4o-mini with enriched context
  Mentions citation relationships, foundational work, researchers
  Streamed token-by-token via SSE
```

**Total latency:** ~1.2 seconds for the full pipeline.

### Streaming Architecture

The `/api/v1/chat` endpoint uses Server-Sent Events (SSE):

1. **Sources event:** Sent immediately after stages 1-6 complete. Contains papers, timings, graph insights.
2. **Token events:** Each word/token from GPT streamed as it generates.
3. **Done event:** Signals completion.

The frontend renders papers instantly while the answer types in character-by-character.

### Conversation History

The chat endpoint accepts a `history` array of previous messages. The last 6 turns are included in the LLM context, enabling follow-up questions like "tell me more about paper [2]" or "compare those approaches."

---

## Knowledge Graph: Neo4j

### Schema

```
(:Paper {id, title, year, arxiv_id, s2_id, citation_count, field})
(:Author {name})
(:Category {name})

(Paper)-[:CITES]->(Paper)
(Paper)-[:AUTHORED_BY]->(Author)
(Paper)-[:IN_CATEGORY]->(Category)
(Author)-[:COAUTHORED_WITH {paper_count}]->(Author)
```

### Graph Queries Used in Production

| Query | What it does | Used by |
|---|---|---|
| `get_paper_context()` | Citations + references + author's other work for a paper | HybridRAG enrichment |
| `get_connected_papers()` | Highly-cited papers linked to a set via citations | Graph expansion |
| `get_common_ancestor()` | Papers cited by multiple results (seminal papers) | Foundational paper detection |
| `get_author_expertise()` | Top authors in a citation cluster | Expert identification |
| `get_trending_papers()` | Recent papers with high citation velocity | Trend detection |
| `get_research_gaps()` | Highly-cited old papers with no recent follow-up | Gap analysis |
| `get_field_bridges()` | Papers spanning two categories | Cross-domain discovery |
| `get_research_lineage()` | Citation tree: ancestors + descendants | Research GPS |
| `get_citation_path()` | Shortest citation chain between two papers | Path validation |

### Why Neo4j Over a Relational DB?

Citation networks are inherently graph-structured. A query like "find all papers within 3 citation hops of paper X" requires recursive JOINs in SQL (slow, complex) but is a single Cypher query in Neo4j (`(p)-[:CITES*1..3]-()`). The COAUTHORED_WITH relationship discovery (finding authors who share papers) is also a natural graph traversal pattern.

---

## HybridRAG: Vector + Graph Intelligence

### What It Adds

Without HybridRAG, the LLM sees 5 isolated abstracts. With it:

```
Before (vector only):
  [1] Paper A — abstract...
  [2] Paper B — abstract...
  [3] Paper C — abstract...

After (vector + graph):
  [1] Paper A (232 citations)
      Abstract: ...
      Builds on: Paper X, Paper Y
      Follow-up work: Paper Z improved this by 2x
      Author also wrote: Paper W

  [2] Paper B (89 citations)
      Abstract: ...
      Cited by Paper A (same research thread)

  === RELATED PAPERS (from citation graph) ===
  - Paper Q (150 cites, connected to 3 of the above)

  === FOUNDATIONAL PAPERS ===
  - Paper R (500 cites, cited by 4 of the retrieved papers)

  === KEY RESEARCHERS ===
  Researcher X (8 papers, 1200 citations), Researcher Y (5 papers)
```

The LLM can now write: "Paper A [1], which builds on earlier work by X, was later extended by Z. The foundational paper R underpins most of this research area, and Researcher X has been the primary contributor."

---

## Research GPS: Learning Path Finder

### Algorithm: Semantic Stepping Stones

**Problem:** Find the optimal reading order from topic A to topic B.

**Approach:** Instead of following random citation chains, generate intermediate semantic waypoints between A and B, find real papers at each waypoint, and validate citation connectivity.

```
Step 1: Embed "CNNs" → vector_A
        Embed "GPU kernels" → vector_B

Step 2: Generate 4 waypoints via linear interpolation:
        wp1 = 0.2×A + 0.8×B  (mostly CNN)
        wp2 = 0.4×A + 0.6×B  (CNN→efficiency)
        wp3 = 0.6×A + 0.4×B  (efficiency→GPU)
        wp4 = 0.8×A + 0.2×B  (mostly GPU)

Step 3: For each waypoint, search Pinecone → candidate papers

Step 4: Select best candidate per waypoint:
        Score = 0.4×similarity + 0.3×citation_boost + 0.3×graph_bonus
        graph_bonus: +0.3 if paper cites the previous step's paper

Step 5: Validate citation links via Neo4j

Step 6: GPT generates "why read this" for each step
```

**Why this works:** Each waypoint paper is semantically closer to the goal than the last AND filters for citation connectivity. The result is a reading order where each paper naturally builds on concepts from the previous one.

**Latency:** ~1.2 seconds total.

---

## Frontend Architecture

### Tech Stack

Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide icons.

### Pages

| Route | Component | Purpose |
|---|---|---|
| `/` | `page.tsx` | Chat interface — main search + conversation |
| `/path` | `path/page.tsx` | Research GPS — learning path finder |

### Components

| Component | Purpose |
|---|---|
| `AnswerCard` | Renders LLM answer with markdown, **bold**, citation badges `[1]` |
| `PaperCard` | Paper result with title, abstract, authors, chunks, badges, arXiv/PDF links |
| `GraphInsights` | Collapsible panel: foundational papers, connected papers, key researchers |
| `SearchStats` | Timing breakdown per pipeline stage |
| `SearchFilters` | Toggle HyDE, rerank, alpha slider, year filter, top_k |
| `HydePreview` | Expandable view of the generated hypothetical abstract |

### API Communication

- **Chat:** SSE (Server-Sent Events) via `POST /api/v1/chat` — sources arrive instantly, answer streams token-by-token
- **Search:** REST via `POST /api/v1/search` — full response in one shot
- **Research GPS:** REST via `POST /api/v1/research-path`
- **Proxy:** `next.config.js` rewrites `/api/*` to FastAPI on port 8000

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Vector DB** | Pinecone (serverless, free tier) | Managed, hybrid search support, 2GB free |
| **Graph DB** | Neo4j Aura (free tier) | Native graph queries, 200K nodes free |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim, cheapest ($0.02/1M tokens) |
| **LLM** | GPT-4o-mini | Fast, cheap, good at synthesis |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Runs locally, no API cost |
| **Sparse encoder** | BM25 (pinecone-text) | Classic keyword matching |
| **Backend** | FastAPI (Python) | Async, auto-docs, SSE support |
| **Frontend** | Next.js 14 + Tailwind | React Server Components, dark theme |
| **Data sources** | ArXiv API + Semantic Scholar API | Free, comprehensive CS coverage |

---

## Project Structure

```
ResearchGPT-Pro/
├── backend/
│   ├── app/                         # FastAPI application
│   │   ├── config.py                # Settings (env vars)
│   │   ├── main.py                  # FastAPI app + CORS
│   │   ├── core/                    # Utilities
│   │   │   ├── logger.py
│   │   │   ├── rate_limiter.py
│   │   │   └── exceptions.py
│   │   ├── db/                      # Database clients
│   │   │   ├── pinecone_client.py   # Hybrid search + upsert
│   │   │   └── neo4j_client.py      # Graph queries
│   │   ├── models/                  # Pydantic models
│   │   │   ├── paper.py
│   │   │   └── search.py
│   │   ├── routes/                  # API endpoints
│   │   │   └── search.py            # /search, /chat, /research-path
│   │   └── services/                # Business logic
│   │       ├── search_service.py    # 7-stage search pipeline
│   │       ├── hybrid_rag.py        # Vector + graph enrichment
│   │       ├── research_path.py     # Research GPS algorithm
│   │       └── embedding_service.py # OpenAI embedding wrapper
│   │
│   ├── scripts/
│   │   ├── run_ingestion.py         # Paper ingestion pipeline
│   │   ├── run_graph_ingestion.py   # Citation + Neo4j ingestion
│   │   ├── check_progress.py        # Monitor ingestion progress
│   │   └── ingestion/               # Pipeline components
│   │       ├── pipeline.py          # Orchestrator
│   │       ├── checkpoint_manager.py
│   │       ├── categories.py
│   │       ├── fetchers/
│   │       │   ├── arxiv_fetcher.py
│   │       │   └── semantic_scholar.py
│   │       └── processors/
│   │           ├── embedder.py
│   │           ├── validator.py
│   │           └── deduplicator.py
│   │
│   └── data/                        # Runtime data (gitignored)
│       ├── raw/                     # Cached ArXiv papers
│       ├── processed/               # S2 citation cache
│       ├── checkpoints/             # Pipeline state
│       └── logs/                    # Log files
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Chat interface
│   │   │   ├── path/page.tsx        # Research GPS
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── AnswerCard.tsx
│   │   │   ├── PaperCard.tsx
│   │   │   ├── GraphInsights.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   ├── SearchFilters.tsx
│   │   │   ├── SearchStats.tsx
│   │   │   └── HydePreview.tsx
│   │   └── lib/
│   │       ├── api.ts               # API client + SSE streaming
│   │       └── types.ts             # TypeScript types
│   ├── next.config.js               # API proxy to FastAPI
│   ├── tailwind.config.ts
│   └── package.json
│
├── requirements.txt
├── README.md                        # This file
├── SETUP.md                         # Setup guide
└── .env                             # API keys (not committed)
```

---

## License

MIT
