# ResearchGPT Pro — Setup Guide

Step-by-step instructions to get ResearchGPT Pro running from scratch.

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **Git**

You'll also need free accounts for:
- [OpenAI](https://platform.openai.com/) — API key for embeddings + GPT-4o-mini
- [Pinecone](https://www.pinecone.io/) — free tier, vector database
- [Neo4j Aura](https://neo4j.com/cloud/aura-free/) — free tier, graph database

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

# Neo4j Aura (required for graph features)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

### Getting the keys

**OpenAI:** Go to https://platform.openai.com/api-keys → Create new secret key

**Pinecone:** Go to https://app.pinecone.io → API Keys (left sidebar)

**Neo4j Aura:**
1. Go to https://neo4j.com/cloud/aura-free/
2. Create a free instance
3. **Save the password immediately** (shown only once)
4. Copy the connection URI from the instance dashboard

---

## 3. Ingest Papers

### Step 3a: Ingest into Pinecone (embeddings)

```bash
# Ingest 25,000 papers across all CS categories
python backend/scripts/run_ingestion.py --preset cs-all --papers 625 --fresh

# Or start smaller for testing
python backend/scripts/run_ingestion.py --categories cs.AI --papers 100 --fresh
```

**Time:** ~4-6 hours for 25K papers (ArXiv rate limits are the bottleneck)
**Cost:** ~$0.50 (OpenAI embeddings)

If interrupted, resume without `--fresh`:
```bash
python backend/scripts/run_ingestion.py --preset cs-all --papers 625
```

Check progress anytime:
```bash
python backend/scripts/check_progress.py
```

### Step 3b: Ingest into Neo4j (citations + graph)

After Pinecone ingestion completes:

```bash
python backend/scripts/run_graph_ingestion.py
```

This:
1. Loads papers from cache (instant)
2. Fetches citations from Semantic Scholar (~2-5 minutes via batch API)
3. Creates nodes + edges in Neo4j (~5 minutes)
4. Builds co-authorship graph (~2 minutes)

**Time:** ~10-15 minutes total
**Cost:** Free (Semantic Scholar API is free)

---

## 4. Start the Application

Open **two terminals:**

### Terminal 1 — Backend API

```bash
# From project root
uvicorn backend.app.main:app --reload
```

The API runs at http://localhost:8000. Swagger docs at http://localhost:8000/docs.

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

The frontend runs at http://localhost:3000.

---

## 5. Use It

### Chat Search

Go to http://localhost:3000 and ask a question:
- "How do transformers handle long documents?"
- "Compare federated learning approaches"
- "What techniques reduce LLM inference cost?"

The system will: generate a HyDE abstract → embed → search Pinecone (hybrid) → rerank with cross-encoder → enrich with Neo4j graph → stream an AI answer with citations.

### Research GPS

Go to http://localhost:3000/path (or click the "Research GPS" button on the landing page):
- I know: "CNNs" → I want to learn: "GPU kernel optimization"
- I know: "Logistic regression" → I want to learn: "Large language models"

The system finds an optimal reading path between the two topics.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/search` | Search papers (non-streaming, full response) |
| POST | `/api/v1/chat` | Chat with streaming answer (SSE) |
| POST | `/api/v1/research-path` | Find learning path between topics |
| GET | `/api/v1/health` | Health check |

### Example: Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms", "top_k": 5}'
```

### Example: Research Path

```bash
curl -X POST "http://localhost:8000/api/v1/research-path?start_topic=CNNs&end_topic=GPU%20kernels&num_steps=4"
```

---

## Configuration Options

### Ingestion Presets

```bash
# Core AI/ML only (6 categories)
python backend/scripts/run_ingestion.py --preset cs-core --papers 1000

# All CS (40 categories)
python backend/scripts/run_ingestion.py --preset cs-all --papers 625

# Everything including Math, Stats, Bio, Finance (75+ categories)
python backend/scripts/run_ingestion.py --preset comprehensive --papers 300
```

### Search Filters (via API or UI)

| Parameter | Default | Description |
|---|---|---|
| `top_k` | 5 | Number of papers to return |
| `use_hyde` | true | Enable HyDE query expansion |
| `use_rerank` | true | Enable cross-encoder reranking |
| `alpha` | 0.7 | Dense vs sparse weight (1.0 = pure dense) |
| `year_min` | null | Filter by minimum year |
| `year_max` | null | Filter by maximum year |

---

## Troubleshooting

### "No module named 'loguru'" (or any module)

Your pip and uvicorn might be using different Python versions. Fix:
```bash
# Check which Python is running
python --version

# Install for the correct Python
python -m pip install -r requirements.txt

# Or use a specific version
py -3.12 -m pip install -r requirements.txt
py -3.12 -m uvicorn backend.app.main:app --reload
```

### "Index configuration does not support sparse values"

Your Pinecone index was created with `cosine` metric but hybrid search needs `dotproduct`. Delete the index in the Pinecone dashboard and re-run ingestion with `--fresh`.

### Pinecone or Neo4j connection errors

Check your `.env` file. Make sure there are no extra spaces or quotes around the values.

### ArXiv rate limiting during ingestion

Normal. The pipeline automatically handles rate limits with backoff. If it stops completely, wait 5 minutes and re-run (it resumes from checkpoint).

### First search is slow (~30 seconds)

The cross-encoder model downloads on first use (~100MB). Subsequent searches are fast (~1-2 seconds).

---

## Costs

| Component | Cost |
|---|---|
| Pinecone | Free tier (2GB, unlimited reads) |
| Neo4j Aura | Free tier (200K nodes, 400K relationships) |
| OpenAI embeddings (25K papers) | ~$0.50 one-time |
| OpenAI GPT-4o-mini (per query) | ~$0.0003 |
| Semantic Scholar API | Free |
| Cross-encoder reranker | Free (runs locally) |
| **Total for setup** | **~$0.50** |
| **Per query** | **~$0.0003** |
