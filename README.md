# ResearchGPT Pro

**An AI-powered academic research assistant combining full-text retrieval, hybrid vector-graph search, and conversational AI.**

ResearchGPT Pro searches computer science papers using an 8-stage retrieval pipeline: full-text PDF parsing via GROBID, structure-aware section chunking, HyDE query expansion, hybrid dense+sparse search, cross-encoder reranking, parent-child chunk expansion, knowledge graph enrichment, and streaming AI answers with citations. It also features a Research GPS that finds optimal learning paths between research topics using semantic interpolation over the citation graph.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Concepts](#core-concepts)
3. [Data Pipeline: Full-Text Ingestion](#data-pipeline-full-text-ingestion)
4. [Search Pipeline: Retrieval + Answer Generation](#search-pipeline-retrieval--answer-generation)
5. [Knowledge Graph: Neo4j](#knowledge-graph-neo4j)
6. [HybridRAG: Vector + Graph Intelligence](#hybridrag-vector--graph-intelligence)
7. [Research GPS: Learning Path Finder](#research-gps-learning-path-finder)
8. [Frontend Architecture](#frontend-architecture)
9. [Tech Stack](#tech-stack)

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
│  │                  SearchService (8 stages)                  │  │
│  │  HyDE → Embed → Pinecone → Rerank → Group → Expand →    │  │
│  │  Graph Enrich → Stream Answer                             │  │
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
│  ~13K chunks │  │  102K nodes      │  │  GPT-4o-mini    │
│  1536-dim    │  │  590K edges      │  │  text-embed-3   │
│  hybrid      │  │  citations +     │  │  -small         │
│  dense+sparse│  │  co-authorship   │  │                 │
└──────────────┘  └──────────────────┘  └─────────────────┘
         ▲
         │
┌────────┴─────────────────────────────────────────────────────┐
│              FULL-TEXT INGESTION PIPELINE                      │
│  ArXiv API → Download PDFs → GROBID Parse → Section Chunk    │
│  → Embed → BM25 Fit → Upsert                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Full-Text Section Chunking (GROBID)

Previous approach embedded only paper abstracts (~200-300 words per paper). The current pipeline downloads actual PDFs, parses them with GROBID (a ML-based document parser), and chunks the full text by section.

**How GROBID works:** It uses a cascade of CRF sequence labeling models that read font size, position, bold/italic, and spacing to identify document structure (title, abstract, section headings, paragraphs, references). It outputs structured TEI XML with labeled sections.

**Chunking strategy — Structure-aware recursive with contextual prefixes:**
- Respects natural document boundaries: section → paragraph → sentence
- Target: 500 tokens per chunk, max 600, min 100
- 50-token overlap at split points
- Contextual prefix on every chunk (Anthropic's contextual retrieval approach):
  ```
  [Computer Science | 2025] Attention Is All You Need
  Section: 3.2 Multi-Head Attention
  Instead of performing a single attention function...
  ```
- Skips low-value sections: Related Work, Acknowledgements, References

**Result:** ~15-20 chunks per paper covering methodology, experiments, results, ablations — not just the abstract.

### 2. Parent-Child Chunk Expansion

A search for "binding shortcuts in VLMs" might match only the abstract chunk. But the user wants methodology details from Section 4. After identifying relevant papers, a second Pinecone query filtered by `paper_id` fetches the most query-relevant section chunks from each paper. This ensures the LLM sees methods and results, not just the abstract that happened to match.

### 3. Dense vs Sparse Vectors

**Dense vectors** (from OpenAI embeddings): Captures semantic meaning — "transformers are slow" and "attention is expensive" get similar vectors even though they share no words.

**Sparse vectors** (from BM25): Captures exact keyword matches — "FlashAttention" gets a high score for queries containing that exact term.

Combined with `alpha` weighting (70% dense, 30% sparse) using `dotproduct` metric in Pinecone.

### 4. HyDE (Hypothetical Document Embedding)

User queries are short and informal. Paper sections are long and technical. HyDE asks GPT-4o-mini to generate a fake abstract that would answer the query, then embeds that. The final search vector is `0.4 × query + 0.6 × hyde`.

### 5. Cross-Encoder Reranking

After Pinecone returns 20-40 candidates, Cohere's reranker reads each (query, chunk_snippet) pair and gives a precise relevance score. With full-text chunks, the reranker now scores against actual section content rather than just title+abstract.

### 6. Knowledge Graph (Neo4j)

A citation graph with 102K nodes (papers, authors, categories) and 590K edges (citations, co-authorship). Enables queries vector search can't do: "what cites this paper", "who collaborates with X", "what's the citation chain from A to B".

---

## Data Pipeline: Full-Text Ingestion

### Pipeline Flow

```
ArXiv API → Validate → Deduplicate → Download PDFs → GROBID Parse
    → Section Chunk → Embed → BM25 Fit → Upsert Pinecone

Semantic Scholar API → Enrich citations → Ingest Neo4j
```

### Step 1: Fetch Papers (ArXiv API)
Fetches paper metadata (title, abstract, authors, categories, PDF URL) from ArXiv. Cached to `backend/data/raw/` as JSON. ~625 papers × 40 CS categories.

### Step 2: Validate + Deduplicate
Pydantic schema validation + ArXiv version deduplication (keeps latest version only).

### Step 3: Download PDFs
Rate-limited downloader (1 req/3s per ArXiv guidelines) with disk caching to `backend/data/pdfs/`. Skips already-downloaded papers. Skips oversized PDFs (>20MB) that crash GROBID.

### Step 4: GROBID Parse
Sends each PDF to a local GROBID server (Docker) and parses the TEI XML response into structured sections. Each section gets a heading, body text, and classified type (method, experiment, conclusion, etc.).

**Section classification:** Keyword matching against ~90 known heading patterns, with IMRAD position-based fallback for unrecognized headings. Sections like Related Work, Acknowledgements, and References are filtered out.

**Fault tolerance:** Auto-detects GROBID crashes (3 consecutive failures), waits for recovery, returns partial results.

### Step 5: Section Chunking
The `SectionChunker` splits parsed sections into 400-600 token chunks at paragraph and sentence boundaries, with 50-token overlap and contextual prefixes containing paper title, field, year, and section heading.

### Step 6: Embed
OpenAI `text-embedding-3-small` (1536 dimensions). Batched at 100, cached to disk, resumable. ~$2-4 for 1,000 papers.

### Step 7: BM25 Fit + Pinecone Upsert
BM25 fitted on enriched chunk texts (includes title, authors, keywords, section content). Each chunk upserted with dense vector, sparse vector, and metadata including `chunk_snippet` (first 800 chars of chunk text for LLM context).

---

## Search Pipeline: Retrieval + Answer Generation

### Full 8-Stage Pipeline

```
User: "What model architecture was used in the binding shortcuts paper?"
  │
  ▼
Stage 1: HyDE (~150ms) — Generate hypothetical abstract
  ▼
Stage 2: Embed (~100ms) — Blend 0.4×query + 0.6×hyde
  ▼
Stage 3: Pinecone Hybrid Search (~50ms) — 20-40 candidate chunks
  ▼
Stage 4: Cross-Encoder Rerank (~200ms) — Score against chunk text
  ▼
Stage 5: Group Chunks → Papers — Merge chunks by paper_id
  ▼
Stage 5b: Parent-Child Expansion (~100ms) — Fetch method/result 
  chunks from each identified paper, filtered by paper_id
  ▼
Stage 6: Graph Enrichment (~200ms) — Citations, ancestors, experts
  ▼
Stage 7: Stream Answer (~500ms) — GPT-4o-mini with section-level
  context, streamed via SSE
```

**Total latency:** ~1.2 seconds. The LLM now sees actual section content ("12-layer decoder-only Transformer, 128 hidden dim, AdamW lr=1e-4") instead of just abstracts.

### Context Building

The context sent to the LLM includes actual matched chunk snippets:

```
[1] Seeing to Generalize (2026)
    Authors: Nicolas Buzeta, Felipe del Rio...
    Matched section [D.1 Model Architecture]: We use a 12-layer 
    decoder-only Transformer with hidden dimension 128, FFN dimension 
    512, 4 attention heads, RoPE positional encoding...
    Matched section [4. VLM Gains]: The text-only model achieves 
    perfect in-distribution accuracy but OOD drops to 37.2%...
    Citations: 0
    Builds on: Gur-Arieh et al. 2025
```

---

## Knowledge Graph: Neo4j

### Schema

```
(:Paper)-[:CITES]->(:Paper)
(:Paper)-[:AUTHORED_BY]->(:Author)
(:Paper)-[:IN_CATEGORY]->(:Category)
(:Author)-[:COAUTHORED_WITH {paper_count}]->(:Author)
```

102K paper nodes, 62K author nodes, 140 category nodes, 147K CITES edges, 440K COAUTHORED_WITH edges.

### Key Graph Queries

| Query | Purpose |
|---|---|
| `get_paper_context()` | Citations + references + author's other work |
| `get_connected_papers()` | Related papers Pinecone missed via citation links |
| `get_common_ancestor()` | Seminal papers cited by multiple results |
| `get_author_expertise()` | Top researchers in a citation cluster |
| `get_trending_papers()` | Recent papers with high citation velocity |
| `get_research_gaps()` | Highly-cited old papers with no recent follow-up |
| `get_research_lineage()` | Full citation tree (ancestors + descendants) |

---

## HybridRAG: Vector + Graph Intelligence

After Pinecone returns ranked papers with section-level snippets, Neo4j enriches each result with citation context, graph-expanded related papers, common ancestors (seminal papers), and author expertise. The LLM can then write answers mentioning citation lineage and key researchers.

---

## Research GPS: Learning Path Finder

Finds optimal reading order from topic A to topic B using semantic interpolation in embedding space, constrained by the citation graph. Generates 4 waypoints via linear interpolation, finds real papers at each waypoint, validates citation connectivity, and generates "why read this" explanations.

---

## Frontend Architecture

Next.js 14 (App Router), TypeScript, Tailwind CSS. Chat interface with SSE streaming, paper cards with section-level matched chunks, graph insights panel, and Research GPS page.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **PDF Parsing** | GROBID (Docker) | ML-based academic PDF → structured sections |
| **Vector DB** | Pinecone (serverless) | Hybrid dense+sparse search, free tier |
| **Graph DB** | Neo4j Aura | Citation graph queries, free tier |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim, $0.02/1M tokens |
| **LLM** | GPT-4o-mini | Answer generation + HyDE |
| **Reranker** | Cohere rerank-v3.5 | Cross-encoder reranking via API |
| **Sparse encoder** | BM25 (pinecone-text) | Keyword matching |
| **Backend** | FastAPI (Python) | Async, SSE streaming |
| **Frontend** | Next.js 14 + Tailwind | Dark theme, streaming chat |
| **Data sources** | ArXiv + Semantic Scholar | Papers + citations |

---

## Project Structure

```
ResearchGPT-Pro/
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── db/
│   │   │   ├── pinecone_client.py    # Hybrid search + upsert + chunk_snippet metadata
│   │   │   └── neo4j_client.py
│   │   ├── models/
│   │   │   └── search.py             # ChunkResult with snippet + section_heading
│   │   ├── routes/
│   │   │   └── search.py
│   │   └── services/
│   │       ├── search_service.py     # 8-stage pipeline + parent-child expansion
│   │       ├── hybrid_rag.py         # Context builder uses chunk snippets
│   │       ├── research_path.py
│   │       └── embedding_service.py
│   │
│   ├── scripts/
│   │   ├── run_ingestion.py          # Abstract-only ingestion (legacy)
│   │   ├── run_fulltext_ingestion.py # Full-text GROBID pipeline ★
│   │   ├── reembed.py                # Migration script v1→v2
│   │   ├── run_graph_ingestion.py
│   │   ├── check_progress.py
│   │   └── ingestion/
│   │       ├── pipeline.py
│   │       ├── checkpoint_manager.py
│   │       ├── categories.py
│   │       ├── fetchers/
│   │       │   ├── arxiv_fetcher.py
│   │       │   └── semantic_scholar.py
│   │       └── processors/
│   │           ├── embedder.py        # v2: single enriched chunk + keyword extraction
│   │           ├── validator.py
│   │           ├── deduplicator.py
│   │           └── fulltext/          # ★ New: full-text processing
│   │               ├── pdf_downloader.py
│   │               ├── grobid_parser.py
│   │               └── section_chunker.py
│   │
│   └── data/
│       ├── raw/          # Cached ArXiv JSON
│       ├── pdfs/         # Downloaded PDFs
│       ├── checkpoints/  # Pipeline state + embedding cache
│       └── logs/
│
├── frontend/             # Next.js 14 app
├── docker-compose.yml    # GROBID service
├── requirements.txt
├── README.md
├── SETUP.md
└── .env
```

---

## License

MIT
