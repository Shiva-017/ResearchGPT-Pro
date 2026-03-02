# ResearchGPT Pro — Setup Guide

Step-by-step instructions to get ResearchGPT Pro running from scratch.

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **Docker Desktop** (for GROBID PDF parser)
- **Git**

You'll also need free accounts for:
- [OpenAI](https://platform.openai.com/) — API key for embeddings + GPT-4o-mini
- [Pinecone](https://www.pinecone.io/) — free tier, vector database
- [Cohere](https://cohere.com/) — free tier, reranker API
- [Neo4j Aura](https://neo4j.com/cloud/aura-free/) — free tier, graph database (optional, for graph features)

---

## 1. Clone and Install

```bash
git clone https://github.com/yourusername/ResearchGPT-Pro.git
cd ResearchGPT-Pro

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

---

## 2. Configure Environment

Create a `.env` file in the project root:

```env
# OpenAI (required)
OPENAI_API_KEY=sk-...

# Pinecone (required)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=research-papers

# Cohere (required for reranking)
COHERE_API_KEY=your-cohere-api-key

# Neo4j Aura (optional — graph features disabled without it)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

### Getting the keys

- **OpenAI:** https://platform.openai.com/api-keys → Create new secret key
- **Pinecone:** https://app.pinecone.io → API Keys (left sidebar)
- **Cohere:** https://dashboard.cohere.com/api-keys → Free tier, 1000 rerank calls/month
- **Neo4j Aura:** https://neo4j.com/cloud/aura-free/ → Create instance, save the password (shown only once)

---

## 3. Start GROBID

GROBID parses PDFs into structured sections. It runs as a Docker container:

```bash
# Start GROBID (runs in background)
docker compose up -d

# Wait ~30-60 seconds, then verify it's running:
curl http://localhost:8070/api/isalive
# Or open http://localhost:8070 in your browser
```

**If `docker compose` fails**, run directly:
```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 --memory=4g lfoppiano/grobid:0.8.2-crf
```

The `-crf` image is ~500MB, CPU-only, and fast enough for our use case.

---

## 4. Ingest Papers (Full-Text Pipeline)

### Quick Start — 100 papers (test run, ~15 min)

```bash
python -m backend.scripts.run_fulltext_ingestion --papers 100 --fresh
```

This downloads PDFs, parses with GROBID, chunks sections, embeds, and uploads to Pinecone.

### Production — 1,000 papers (~90 min, ~$2-4)

```bash
python -m backend.scripts.run_fulltext_ingestion --papers 1000 --fresh
```

### Custom categories

```bash
python -m backend.scripts.run_fulltext_ingestion --papers 500 --categories cs.AI cs.CV cs.CL --fresh
```

### Resume after interruption

PDFs and embeddings are cached. If interrupted, resume without `--fresh`:
```bash
python -m backend.scripts.run_fulltext_ingestion --papers 1000 --skip-download
```

### Expected output

```
FULL-TEXT INGESTION COMPLETE
============================================================
Papers processed: 1000 → 950 with PDF → 900 parsed
Chunks total    : 13,000
Chunks embedded : 13,000
Avg chunks/paper: 14.4
Embedding cost  : $2.50
Pinecone vectors: 13,000
```

### What the pipeline does

| Step | Action | Time |
|---|---|---|
| 1 | Load papers from ArXiv cache | instant |
| 2 | Validate + deduplicate | instant |
| 3 | Download PDFs (rate limited 1/3s) | ~50 min |
| 4 | GROBID parse PDFs → structured sections | ~30 min |
| 5 | Chunk sections (400-600 tokens, paragraph boundaries) | ~1 min |
| 6 | Embed chunks (OpenAI text-embedding-3-small) | ~5 min |
| 7 | Fit BM25 + upsert to Pinecone | ~3 min |

### Pinecone free tier budget

Free tier = 100K vectors. With ~15 chunks/paper:
- 1,000 papers → ~15K vectors ✅
- 3,000 papers → ~45K vectors ✅
- 6,000 papers → ~90K vectors ⚠️ (near limit)

---

## 5. Ingest Citation Graph (Optional)

If you have Neo4j configured:

```bash
python -m backend.scripts.run_graph_ingestion
```

This fetches citations from Semantic Scholar and builds the knowledge graph. Takes ~10-15 minutes, costs nothing.

Without Neo4j, the system still works — you just won't get citation context, related papers, or author expertise in answers.

---

## 6. Start the Application

### Terminal 1 — Backend

```bash
uvicorn backend.app.main:app --reload
```

API at http://localhost:8000. Swagger docs at http://localhost:8000/docs.

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Frontend at http://localhost:3000.

### Terminal 3 — GROBID (if not using docker compose)

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 --memory=4g lfoppiano/grobid:0.8.2-crf
```

---

## 7. Test It

### Queries that test full-text retrieval

These should return specific details from paper sections, not just abstract summaries:

```
"what optimizer and learning rate were used for training in the binding shortcuts paper"
"what image encoders were compared for visual training in VLM experiments"
"which papers report results on the GLUE benchmark"
```

### Queries that test broad topic search

```
"How do transformers handle long documents?"
"Compare federated learning approaches"
"What techniques reduce LLM inference cost?"
```

### Research GPS

Go to http://localhost:3000/path:
- I know: "CNNs" → I want to learn: "GPU kernel optimization"

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/search` | Search papers (full response) |
| POST | `/api/v1/chat` | Chat with streaming answer (SSE) |
| POST | `/api/v1/research-path` | Learning path between topics |
| GET | `/api/v1/health` | Health check |

---

## Troubleshooting

### GROBID won't start / docker errors

```bash
# Restart Docker Desktop, then:
docker pull lfoppiano/grobid:0.8.2-crf
docker run --rm --init --ulimit core=0 -p 8070:8070 --memory=4g lfoppiano/grobid:0.8.2-crf
```

### GROBID crashes during parsing

Large PDFs (>20MB) can exhaust GROBID's memory. The parser auto-skips these. If GROBID dies mid-run, restart it and re-run with `--skip-download`.

### `<|endoftext|>` error during chunking

Some papers contain literal special tokens. Already fixed — `tiktoken` is called with `disallowed_special=()`.

### Pinecone "index configuration does not support sparse values"

Your index was created with `cosine` metric. Delete it in Pinecone dashboard and re-run with `--fresh` (it recreates with `dotproduct`).

### Search returns only abstract-level answers

Ensure you ran `run_fulltext_ingestion` (not `run_ingestion`). Check that `chunk_snippet` exists in your Pinecone vectors via the dashboard. If missing, re-run with `--fresh`.

### Module not found errors

```bash
python -m pip install -r requirements.txt
```

---

## Costs

| Component | Cost |
|---|---|
| Pinecone | Free tier (100K vectors) |
| Neo4j Aura | Free tier (200K nodes) |
| Cohere reranker | Free tier (1000 calls/month) |
| GROBID | Free (Docker, runs locally) |
| OpenAI embeddings (1K papers) | ~$2-4 one-time |
| OpenAI GPT-4o-mini (per query) | ~$0.0003 |
| **Total setup (1K papers)** | **~$2-4** |
| **Per query** | **~$0.0003** |
