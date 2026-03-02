# backend/scripts/reembed.py

"""
Re-embedding migration script.

Migrates from v1 (dual problem/method chunks) to v2 (single enriched chunk).
This script:
  1. Loads all raw papers from cached JSON files
  2. Validates and deduplicates them
  3. Re-embeds with the new single-chunk strategy
  4. Clears the old Pinecone index and upserts fresh data
  5. Re-fits BM25 on the new chunk texts

Usage:
  cd ResearchGPT-Pro
  python -m backend.scripts.reembed

Estimated cost: ~$0.25-0.50 for 25K papers (single chunk = half the tokens)
Estimated time: ~15-20 minutes
"""

import sys
import os
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datetime import datetime
from loguru import logger

from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient
from backend.scripts.ingestion.checkpoint_manager import IngestionState
from backend.scripts.ingestion.processors.embedder import FaultTolerantEmbedder, build_embedding_chunks
from backend.scripts.ingestion.processors.validator import PaperValidator
from backend.scripts.ingestion.processors.deduplicator import Deduplicator


def load_all_raw_papers(raw_dir: str = "backend/data/raw") -> list:
    """Load all papers from cached arxiv JSON files."""
    papers = []
    json_files = sorted(glob.glob(os.path.join(raw_dir, "arxiv_*.json")))

    if not json_files:
        logger.error(f"No arxiv JSON files found in {raw_dir}")
        return []

    logger.info(f"Found {len(json_files)} category files in {raw_dir}")

    for filepath in json_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                batch = json.load(f)
            papers.extend(batch)
            logger.info(f"  {filename}: {len(batch)} papers")
        except Exception as e:
            logger.error(f"  {filename}: FAILED — {e}")

    logger.info(f"Total raw papers loaded: {len(papers):,}")
    return papers


def main():
    start = datetime.now()

    print("\n" + "=" * 60)
    print("RESEARCHGPT PRO — RE-EMBEDDING MIGRATION (v1 → v2)")
    print("=" * 60)
    print()
    print("v1: Dual chunks (problem/method split) — 2 vectors per paper")
    print("v2: Single enriched chunk — 1 vector per paper, better coherence")
    print()

    # ── Step 1: Load raw papers ──────────────────────────────────────
    logger.info("STEP 1: Loading raw papers from cache…")
    papers_raw = load_all_raw_papers()
    if not papers_raw:
        logger.error("No papers found. Run the ingestion pipeline first.")
        return

    # ── Step 2: Validate ─────────────────────────────────────────────
    logger.info("\nSTEP 2: Validating papers…")
    validator = PaperValidator()
    papers = validator.validate_papers(papers_raw)
    logger.info(f"Valid: {len(papers):,}")

    # ── Step 3: Deduplicate ──────────────────────────────────────────
    logger.info("\nSTEP 3: Deduplicating…")
    dedup = Deduplicator()
    papers = dedup.deduplicate(papers)
    logger.info(f"Unique: {len(papers):,}")

    # ── Step 4: Preview new chunking ─────────────────────────────────
    logger.info("\nSTEP 4: Preview — new chunk format")
    sample = papers[0]
    sample_chunks = build_embedding_chunks(sample)
    print("\n--- Sample chunk ---")
    print(sample_chunks[0]['chunk_text'][:500])
    print(f"--- (chunk_id: {sample_chunks[0]['chunk_id']}) ---\n")

    # ── Step 5: Reset checkpoint state ───────────────────────────────
    logger.info("STEP 5: Resetting checkpoint state for fresh re-embedding…")
    state = IngestionState()
    state.clear()
    logger.info("Checkpoints cleared.")

    # ── Step 6: Embed with new strategy ──────────────────────────────
    logger.info("\nSTEP 6: Embedding papers with v2 strategy…")
    embedder = FaultTolerantEmbedder(
        api_key=settings.openai_api_key,
        state_manager=state,
    )
    chunks = embedder.embed_papers(papers, batch_size=100)

    chunks_ok   = [c for c in chunks if c.get("embedding")]
    chunks_fail = [c for c in chunks if not c.get("embedding")]

    logger.info(f"Embedded: {len(chunks_ok):,} | Failed: {len(chunks_fail):,}")

    if not chunks_ok:
        logger.error("No chunks embedded. Aborting.")
        return

    # ── Step 7: Re-fit BM25 on new chunk texts ──────────────────────
    logger.info("\nSTEP 7: Re-fitting BM25 on new chunk texts…")
    pinecone = PineconeClient(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=1536,
    )

    # BM25 should be fitted on the chunk_text (which now includes
    # title, authors, keywords, and abstract)
    bm25_corpus = [c['chunk_text'] for c in chunks_ok]
    pinecone.fit_bm25(bm25_corpus)

    # ── Step 8: Clear old vectors and upsert ─────────────────────────
    logger.info("\nSTEP 8: Clearing old Pinecone vectors and upserting…")

    # Delete all vectors in the index
    try:
        pinecone.index.delete(delete_all=True)
        logger.info("Old vectors deleted from Pinecone.")
        import time
        time.sleep(5)  # Wait for deletion to propagate
    except Exception as e:
        logger.warning(f"Could not delete old vectors: {e}")
        logger.info("Proceeding with upsert (will overwrite existing IDs)…")

    # Clear the stored checkpoint file too
    stored_path = os.path.join(
        pinecone.checkpoint_dir,
        f"pinecone_stored_{pinecone.index_name}.json",
    )
    if os.path.exists(stored_path):
        os.remove(stored_path)

    # Upsert new chunks
    result = pinecone.upsert_chunks(chunks_ok, skip_existing=False)

    # Mark papers as completed
    completed_ids = list({c["id"] for c in chunks_ok})
    state.mark_completed(completed_ids)

    # ── Final report ─────────────────────────────────────────────────
    duration = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 60)
    print("RE-EMBEDDING MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Duration       : {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Papers         : {len(papers):,}")
    print(f"Chunks (v2)    : {len(chunks_ok):,}  (was ~{len(papers)*2:,} in v1)")
    print(f"Uploaded       : {result['upserted']:,}")
    print(f"Failed         : {result.get('failed', 0) + len(chunks_fail):,}")
    print(f"Tokens         : {embedder.total_tokens:,}")
    print(f"Cost           : ${embedder.total_cost:.4f}")
    print(f"Vectors saved  : ~{len(papers):,} (50% reduction from v1)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
