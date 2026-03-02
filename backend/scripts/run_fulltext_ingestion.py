# backend/scripts/run_fulltext_ingestion.py

"""
Full-text ingestion pipeline — GROBID edition.

Scope: 1,000-2,000 papers from top categories (cs.AI, cs.CL, cs.LG)
       to stay within Pinecone free tier (~15K-30K vectors).

Pipeline:
  1. Load raw papers from cache
  2. Select top categories + limit paper count
  3. Download PDFs from arXiv
  4. Parse PDFs with GROBID (Docker)
  5. Chunk sections intelligently
  6. Embed chunks with OpenAI
  7. Fit BM25 on chunk texts
  8. Clear old Pinecone data + upsert

Prerequisites:
  - GROBID running: docker run --rm -p 8070:8070 grobid/grobid:0.8.1
  - .env with OPENAI_API_KEY, PINECONE_API_KEY

Usage:
  cd ResearchGPT-Pro
  python -m backend.scripts.run_fulltext_ingestion
  python -m backend.scripts.run_fulltext_ingestion --papers 500 --categories cs.AI cs.CL
  python -m backend.scripts.run_fulltext_ingestion --papers 2000 --skip-download  # if PDFs already cached

Estimated (1,000 papers):
  - Download:  ~50 min (arXiv rate limit)
  - GROBID:    ~30 min
  - Embedding: ~5 min, ~$2-4
  - Total:     ~90 min
  - Vectors:   ~10K-20K
"""

import sys
import os
import json
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datetime import datetime
from typing import List, Dict
from loguru import logger

from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient
from backend.scripts.ingestion.checkpoint_manager import IngestionState
from backend.scripts.ingestion.processors.validator import PaperValidator
from backend.scripts.ingestion.processors.deduplicator import Deduplicator
from backend.scripts.ingestion.processors.fulltext.pdf_downloader import PdfDownloader
from backend.scripts.ingestion.processors.fulltext.grobid_parser import GrobidParser
from backend.scripts.ingestion.processors.fulltext.section_chunker import SectionChunker
from backend.scripts.ingestion.processors.embedder import FaultTolerantEmbedder


# Top categories for scoped ingestion
DEFAULT_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]


def load_papers_for_categories(
    categories: List[str],
    raw_dir: str = "backend/data/raw",
    max_per_category: int = 500,
) -> List[Dict]:
    """Load papers from cached arxiv JSON files for specific categories."""
    all_papers = []

    for cat in categories:
        safe_cat = cat.replace(".", "_")
        pattern = os.path.join(raw_dir, f"arxiv_{safe_cat}_*.json")
        files = glob.glob(pattern)

        if not files:
            logger.warning(f"No cached file for category {cat}")
            continue

        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                papers = json.load(f)
            all_papers.extend(papers[:max_per_category])
            logger.info(f"  {cat}: loaded {min(len(papers), max_per_category)} papers")

    return all_papers


def main():
    parser = argparse.ArgumentParser(description="Full-text ingestion with GROBID")
    parser.add_argument("--papers", type=int, default=1000, help="Max papers to process")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--skip-download", action="store_true", help="Skip PDF download (use cached)")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--fresh", action="store_true", help="Clear all state and start fresh")
    args = parser.parse_args()

    start = datetime.now()

    print("\n" + "=" * 60)
    print("RESEARCHGPT PRO — FULL-TEXT INGESTION (GROBID)")
    print("=" * 60)
    print(f"Categories:  {args.categories}")
    print(f"Max papers:  {args.papers}")
    print(f"GROBID:      {args.grobid_url}")
    print()

    # ── Step 1: Load papers ──────────────────────────────────────────
    logger.info("STEP 1: Loading papers from cache…")
    per_cat = args.papers // len(args.categories) + 1
    papers_raw = load_papers_for_categories(
        args.categories, max_per_category=per_cat
    )
    logger.info(f"Loaded {len(papers_raw)} raw papers")

    # ── Step 2: Validate + dedup ─────────────────────────────────────
    logger.info("\nSTEP 2: Validate + deduplicate…")
    validator = PaperValidator()
    papers = validator.validate_papers(papers_raw)
    dedup = Deduplicator()
    papers = dedup.deduplicate(papers)

    # Limit to requested count
    papers = papers[:args.papers]
    logger.info(f"Using {len(papers)} papers")

    # ── Step 3: Download PDFs ────────────────────────────────────────
    if args.skip_download:
        logger.info("\nSTEP 3: Skipping download (--skip-download)")
        # Check for existing PDFs
        downloader = PdfDownloader()
        for paper in papers:
            path = downloader._pdf_path(paper["id"])
            if path.exists() and path.stat().st_size > 1000:
                paper["pdf_local_path"] = str(path)
        papers_with_pdf = [p for p in papers if "pdf_local_path" in p]
        failed_download = [p for p in papers if "pdf_local_path" not in p]
        downloader.cleanup()
    else:
        logger.info("\nSTEP 3: Downloading PDFs…")
        downloader = PdfDownloader()
        papers_with_pdf, failed_download = downloader.download_papers(papers)
        downloader.cleanup()

    logger.info(f"PDFs ready: {len(papers_with_pdf)} | Missing: {len(failed_download)}")

    if not papers_with_pdf:
        logger.error("No PDFs available. Aborting.")
        return

    # ── Step 4: Parse with GROBID ────────────────────────────────────
    logger.info("\nSTEP 4: Parsing PDFs with GROBID…")
    grobid = GrobidParser(grobid_url=args.grobid_url)
    parsed_results = grobid.parse_papers(papers_with_pdf)
    logger.info(f"Parsed: {len(parsed_results)} papers")

    if not parsed_results:
        logger.error("No papers parsed successfully. Check GROBID.")
        return

    # Print sample
    if parsed_results:
        sample = parsed_results[0]
        logger.info(f"\nSample: {sample['paper_id']}")
        for sec in sample["sections"][:5]:
            logger.info(f"  [{sec['section_type']:12s}] {sec['heading'][:50]} ({len(sec['text'])} chars)")

    # ── Step 5: Chunk sections ───────────────────────────────────────
    logger.info("\nSTEP 5: Chunking sections…")
    chunker = SectionChunker(
        target_tokens=500,
        max_tokens=600,
        min_tokens=100,
        overlap_tokens=50,
    )
    all_chunks = chunker.chunk_papers(papers_with_pdf, parsed_results)

    if not all_chunks:
        logger.error("No chunks produced. Aborting.")
        return

    # Preview a chunk
    logger.info(f"\nSample chunk:\n{all_chunks[0]['chunk_text'][:300]}…\n")

    # ── Step 6: Embed chunks ─────────────────────────────────────────
    logger.info("STEP 6: Embedding chunks…")

    if args.fresh:
        state = IngestionState()
        state.clear()
    else:
        state = IngestionState()

    embedder = FaultTolerantEmbedder(
        api_key=settings.openai_api_key,
        state_manager=state,
    )

    # The embedder expects chunks with chunk_id and chunk_text
    # Our chunks already have those from SectionChunker
    embedded_chunks = []
    batch_size = 100

    from tqdm import tqdm

    # Filter out already-embedded
    embedded_ids = state.get_embedded_ids()
    to_embed = [c for c in all_chunks if c["chunk_id"] not in embedded_ids]
    cached = [c for c in all_chunks if c["chunk_id"] in embedded_ids]

    logger.info(f"Already embedded: {len(cached)} | To embed: {len(to_embed)}")

    # Load cached embeddings
    for chunk in cached:
        emb = state.load_embedding(chunk["chunk_id"])
        if emb:
            chunk["embedding"] = emb
            embedded_chunks.append(chunk)

    # Embed remaining chunks using the embedder's batch API
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError

    client = OpenAI(api_key=settings.openai_api_key)
    total_tokens = 0

    for i in tqdm(range(0, len(to_embed), batch_size), desc="Embedding"):
        batch = to_embed[i:i + batch_size]
        texts = [c["chunk_text"] for c in batch]

        try:
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small",
            )
            total_tokens += response.usage.total_tokens

            chunk_ids = []
            for chunk, emb_data in zip(batch, response.data):
                chunk["embedding"] = emb_data.embedding
                state.save_embedding(chunk["chunk_id"], emb_data.embedding)
                embedded_chunks.append(chunk)
                chunk_ids.append(chunk["chunk_id"])

            state.mark_embedded(chunk_ids)

        except (APIError, RateLimitError, APIConnectionError) as e:
            logger.error(f"Batch API error at {i}: {e}")
            # Fallback to single
            for chunk in batch:
                try:
                    r = client.embeddings.create(
                        input=chunk["chunk_text"],
                        model="text-embedding-3-small",
                    )
                    total_tokens += r.usage.total_tokens
                    chunk["embedding"] = r.data[0].embedding
                    state.save_embedding(chunk["chunk_id"], chunk["embedding"])
                    state.mark_embedded([chunk["chunk_id"]])
                    embedded_chunks.append(chunk)
                except Exception as e2:
                    logger.error(f"Single embed failed {chunk['chunk_id']}: {e2}")

    cost = (total_tokens / 1_000_000) * 0.02
    logger.info(f"Embedding done: {len(embedded_chunks)} chunks, {total_tokens:,} tokens, ${cost:.4f}")

    # ── Step 7: BM25 + Pinecone ──────────────────────────────────────
    logger.info("\nSTEP 7: BM25 + Pinecone upsert…")
    pinecone = PineconeClient(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=1536,
    )

    # Re-fit BM25 on new chunk texts
    bm25_corpus = [c["chunk_text"] for c in embedded_chunks]
    pinecone.fit_bm25(bm25_corpus)

    # Clear old vectors
    if args.fresh:
        try:
            pinecone.index.delete(delete_all=True)
            logger.info("Old vectors deleted.")
            import time
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Could not delete old vectors: {e}")

        stored_path = os.path.join(
            pinecone.checkpoint_dir,
            f"pinecone_stored_{pinecone.index_name}.json",
        )
        if os.path.exists(stored_path):
            os.remove(stored_path)

    # Upsert
    chunks_ok = [c for c in embedded_chunks if c.get("embedding")]
    result = pinecone.upsert_chunks(chunks_ok, skip_existing=not args.fresh)

    # Mark papers as completed
    completed_ids = list({c["id"] for c in chunks_ok})
    state.mark_completed(completed_ids)

    # ── Final report ─────────────────────────────────────────────────
    duration = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 60)
    print("FULL-TEXT INGESTION COMPLETE")
    print("=" * 60)
    print(f"Duration        : {duration:.0f}s ({duration/60:.1f} min)")
    print(f"Categories      : {args.categories}")
    print(f"Papers processed: {len(papers)} → {len(papers_with_pdf)} with PDF → {len(parsed_results)} parsed")
    print(f"Chunks total    : {len(all_chunks)}")
    print(f"Chunks embedded : {len(embedded_chunks)}")
    print(f"Chunks uploaded : {result['upserted']}")
    print(f"Avg chunks/paper: {len(all_chunks)/max(len(parsed_results),1):.1f}")
    print(f"Tokens          : {total_tokens:,}")
    print(f"Embedding cost  : ${cost:.4f}")
    print(f"Pinecone vectors: {result['upserted']}")
    print("=" * 60 + "\n")

    # Section type breakdown
    from collections import Counter
    type_counts = Counter(c["chunk_type"] for c in all_chunks)
    print("Chunk type breakdown:")
    for ctype, count in type_counts.most_common():
        print(f"  {ctype:15s}: {count:5d} ({count/len(all_chunks)*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
