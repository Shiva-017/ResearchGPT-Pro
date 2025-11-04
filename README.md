# ResearchGPT-Pro

A hybrid RAG (Retrieval-Augmented Generation) system for academic paper search using Pinecone, Neo4j, and PostgreSQL. The system ingests papers from arXiv, generates embeddings, and stores them in a vector database for semantic search.

## Features

- **Hybrid Search**: Combines vector similarity search with graph-based relationships
- **Multi-Source**: Supports ArXiv data ingestion
- **Fault-Tolerant**: Automatic checkpointing and resume capability
- **Cost-Effective**: Optimized for free tier usage
- **Scalable**: Built with FastAPI and modern async patterns
- **Rate-Limited**: Respects ArXiv API limits (3 requests/second)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   ArXiv     │────▶│   Pipeline   │────▶│  Pinecone   │
│   Papers    │     │  (Embedding) │     │  (Vectors)  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   PostgreSQL │
                    │  (Metadata)  │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    Neo4j     │
                    │   (Graph)    │
                    └──────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- (Optional) Neo4j and PostgreSQL for full functionality

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ResearchGPT-Pro
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Set up databases** (optional, for full functionality)
   ```bash
   docker-compose up -d
   ```

## Ingestion Pipeline

The ingestion pipeline is a fault-tolerant system that fetches papers from arXiv, validates them, generates embeddings, and stores them in Pinecone.

### Pipeline Stages

1. **Fetch**: Downloads papers from arXiv with rate limiting
2. **Validate**: Validates paper structure and content
3. **Deduplicate**: Removes duplicate papers (handles version numbers)
4. **Embed**: Generates embeddings using OpenAI's `text-embedding-3-small`
5. **Store**: Uploads vectors to Pinecone with metadata

### Usage

#### Basic Usage

```bash
# Ingest papers from a single category
python backend/scripts/run_ingestion.py --categories cs.AI --papers 1000

# Ingest from multiple categories
python backend/scripts/run_ingestion.py --categories cs.AI cs.CL cs.LG --papers 5000

# Resume from checkpoint (if interrupted)
python backend/scripts/run_ingestion.py --categories cs.AI --papers 1000

# Start fresh (ignore checkpoints)
python backend/scripts/run_ingestion.py --categories cs.AI --papers 1000 --fresh
```

#### Large-Scale Ingestion (100K papers)

```bash
# Ingest 100,000 papers across multiple categories
python backend/scripts/run_ingestion.py \
  --categories cs.AI cs.CL cs.LG cs.CV cs.NE cs.SI cs.IR cs.CY \
  --papers 12500
```

**Estimated Time**: ~15-17 hours for 100K papers
- Fetching: ~9-10 hours (ArXiv rate limits)
- Embedding: ~3-4 hours (OpenAI API)
- Uploading: ~3 hours (Pinecone rate limits)

**Cost**: ~$2 for 100K papers (text-embedding-3-small at $0.02 per 1M tokens)

### Pipeline Features

#### Fault Tolerance

- **Checkpointing**: Automatically saves progress after each stage
- **Resume Capability**: Can resume from any checkpoint if interrupted
- **Error Handling**: Gracefully handles API errors and continues processing
- **State Management**: Tracks completed, embedded, and failed papers

#### Rate Limiting

- **ArXiv API**: Respects 3 requests/second limit (0.34s between requests)
- **OpenAI API**: Handles rate limits with exponential backoff
- **Pinecone**: Respects 10 writes/second limit for free tier

#### Smart Error Handling

- **UnexpectedEmptyPageError**: Automatically stops when category has fewer papers than requested
- **Rate Limit Errors**: Waits and retries with exponential backoff
- **Network Errors**: Retries up to 3 times with increasing delays
- **Partial Failures**: Continues processing even if some papers fail

### Checkpoint System

The pipeline maintains checkpoints in `backend/data/checkpoints/`:

- **ingestion_state.json**: Tracks completed, embedded, and failed papers
- **embeddings/**: Stores individual embedding files (one per paper)
- **pinecone_stored_*.json**: Tracks papers uploaded to Pinecone

### Monitoring Progress

#### View Logs

```bash
# View latest log
tail -f backend/data/logs/ingestion_*.log

# View last 50 lines
tail -n 50 backend/data/logs/ingestion_*.log
```

#### Check Progress

```bash
# Check checkpoint status
python backend/scripts/check_progress.py

# Retry failed papers
python backend/scripts/retry_failed.py
```

#### Test with Single Paper

```bash
# Test the pipeline with just 1 paper
python backend/scripts/test_single_paper.py
```

### Cost Estimation

#### Embedding Costs (OpenAI)

| Model | Dimensions | Cost per 1M tokens | Cost for 100K papers |
|-------|-----------|-------------------|---------------------|
| text-embedding-3-small | 1536 | $0.02 | ~$1.00 |
| text-embedding-3-large | 3072 | $0.13 | ~$6.50 |

**Recommendation**: Use `text-embedding-3-small` for cost-effective ingestion.

#### Pinecone Costs (Serverless Free Tier)

- **Storage**: 2 GB (~100K vectors at 1536 dims) - **FREE**
- **Queries**: Unlimited reads - **FREE**
- **Writes**: Rate limited (10 req/sec) - **FREE**
- **Indexes**: 5 serverless indexes - **FREE**

**Note**: Free tier is sufficient for 100K papers. Additional storage costs $0.096/GB/month.

### Configuration

#### Environment Variables

```env
# Required
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=research-papers

# Optional
LOG_LEVEL=INFO
```

#### Pipeline Parameters

- **Batch Size**: 100 papers per ArXiv fetch batch (configurable)
- **Embedding Batch Size**: 100 papers per OpenAI API call
- **Pinecone Batch Size**: 100 vectors per upsert (respects free tier limits)
- **Checkpoint Interval**: Every 500 papers uploaded

### Troubleshooting

#### Common Issues

1. **ArXiv Rate Limiting**
   - **Symptom**: `UnexpectedEmptyPageError` or slow fetching
   - **Solution**: Pipeline automatically handles this. Wait or reduce papers per category.

2. **OpenAI Quota Exceeded**
   - **Symptom**: `insufficient_quota` error
   - **Solution**: Check OpenAI account billing and add credits.

3. **Pinecone Index Not Ready**
   - **Symptom**: Index creation timeout
   - **Solution**: Check Pinecone dashboard, index is created automatically.

4. **Memory Issues with Large Batches**
   - **Symptom**: Process killed or slow performance
   - **Solution**: Reduce batch sizes or process categories separately.

#### Resuming Failed Ingestion

If ingestion is interrupted:

1. **Check what was completed**:
   ```bash
   python backend/scripts/check_progress.py
   ```

2. **Resume automatically**:
   ```bash
   # Just run again without --fresh flag
   python backend/scripts/run_ingestion.py --categories cs.AI --papers 1000
   ```

3. **Retry failed papers**:
   ```bash
   python backend/scripts/retry_failed.py
   ```

### Data Structure

#### Paper Format

```python
{
    "id": "arxiv:2510.27688",
    "title": "Paper Title",
    "abstract": "Paper abstract...",
    "authors": ["Author 1", "Author 2"],
    "year": 2025,
    "published": "2025-10-27",
    "categories": ["cs.AI", "cs.LG"],
    "primary_category": "cs.AI",
    "pdf_url": "https://arxiv.org/pdf/2510.27688.pdf",
    "source": "arxiv",
    "citation_count": 0,
    "embedding": [0.123, 0.456, ...]  # 1536 dimensions
}
```

#### Pinecone Metadata

Each vector in Pinecone includes:
- **Display fields**: title, abstract, authors, year
- **Filter fields**: categories, citation_count, field
- **Helper fields**: has_code, is_survey, author_count
- **Identifiers**: arxiv_id, source

## API (FastAPI)

### Endpoints

#### Search Papers

```bash
POST /api/v1/search
{
    "query": "machine learning",
    "top_k": 10,
    "use_graph": true,
    "rerank": true
}
```

#### Get Paper by ID

```bash
GET /api/v1/papers/{paper_id}
```

#### List Papers

```bash
GET /api/v1/papers?limit=10&offset=0&query=transformer
```

### Running the API

```bash
# Development
uvicorn backend.app.main:app --reload

# Production
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
research-gpt-pro/
├── backend/
│   ├── app/                    # FastAPI application
│   │   ├── core/              # Utilities (rate limiter, logger, exceptions)
│   │   ├── db/                # Database clients (Pinecone, Neo4j, PostgreSQL)
│   │   ├── services/          # Business logic (embedding, search, RAG)
│   │   ├── models/            # Pydantic models
│   │   └── routes/            # API endpoints
│   │
│   ├── scripts/
│   │   ├── ingestion/         # Ingestion pipeline
│   │   │   ├── fetchers/     # Data fetchers (ArXiv)
│   │   │   ├── processors/   # Data processors (validator, embedder, deduplicator)
│   │   │   ├── storers/      # Database storers
│   │   │   └── pipeline.py   # Main pipeline orchestrator
│   │   │
│   │   ├── run_ingestion.py  # Entry point for ingestion
│   │   ├── check_progress.py # Check ingestion progress
│   │   └── retry_failed.py   # Retry failed papers
│   │
│   ├── data/
│   │   ├── raw/              # Raw fetched papers (cached)
│   │   ├── processed/        # Processed papers
│   │   ├── checkpoints/      # Pipeline checkpoints
│   │   └── logs/             # Log files
│   │
│   └── tests/                # Unit tests
│
├── frontend/                  # Frontend application (coming soon)
└── infrastructure/            # Docker and deployment configs
```

## Development

### Running Tests

```bash
pytest backend/tests/
```

### Code Style

The project follows PEP 8 style guidelines. Key libraries used:
- **FastAPI**: Modern web framework
- **Pydantic v2**: Data validation
- **Tenacity**: Retry logic
- **Loguru**: Structured logging
- **TQDM**: Progress bars

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- ArXiv for providing open access to research papers
- OpenAI for embedding models
- Pinecone for vector database infrastructure

