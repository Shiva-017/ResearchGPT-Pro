# backend/scripts/ingestion/pipeline.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from typing import List, Dict
from datetime import datetime
from loguru import logger

from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient
from backend.scripts.ingestion.checkpoint_manager import IngestionState
from backend.scripts.ingestion.fetchers.arxiv_fetcher import ArxivFetcher
from backend.scripts.ingestion.processors.embedder import FaultTolerantEmbedder
from backend.scripts.ingestion.processors.validator import PaperValidator
from backend.scripts.ingestion.processors.deduplicator import Deduplicator


class IngestionPipeline:
    """
    Fault-tolerant ingestion pipeline.

    Stages:
      1. Fetch      — ArXiv papers
      2. Validate   — schema + content checks
      3. Deduplicate— remove version duplicates
      4. BM25 fit   — fit sparse encoder on full corpus (once)
      5. Embed      — expand to chunks, generate embeddings
      6. Upsert     — push chunks to Pinecone (hybrid sparse-dense)
    """

    def __init__(self, resume: bool = True):
        logger.info("Initialising ingestion pipeline…")

        self.state = IngestionState()
        if not resume:
            logger.warning("Starting fresh — ignoring checkpoints")
            self.state.clear()

        self.pinecone = PineconeClient(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=1536,
        )

        self.arxiv_fetcher = ArxivFetcher()
        self.validator     = PaperValidator()
        self.deduplicator  = Deduplicator()
        self.embedder      = FaultTolerantEmbedder(
            api_key=settings.openai_api_key,
            state_manager=self.state,
        )

        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------

    def run(
        self,
        categories: List[str] = ["cs.AI"],
        papers_per_category: int = 1000,
    ) -> Dict:
        start = datetime.now()

        try:
            # ── Stage 1: Fetch ──────────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 1: FETCHING PAPERS FROM ARXIV")
            logger.info("=" * 60)
            papers_raw = self.arxiv_fetcher.fetch_by_categories(
                categories=categories,
                papers_per_category=papers_per_category,
            )
            fetched_count = len(papers_raw)

            # ── Stage 2: Validate ───────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 2: VALIDATING")
            logger.info("=" * 60)
            papers = self.validator.validate_papers(papers_raw)

            # ── Stage 3: Deduplicate ────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 3: DEDUPLICATING")
            logger.info("=" * 60)
            papers = self.deduplicator.deduplicate(papers)

            # ── Stage 4: Fit BM25 ───────────────────────────────────────
            # Fit on title + abstract of the full corpus.
            # Skipped if encoder already exists on disk.
            if self.pinecone.bm25 is None:
                logger.info("\n" + "=" * 60)
                logger.info("STAGE 4: FITTING BM25 ENCODER")
                logger.info("=" * 60)
                corpus = [
                    f"{p['title']} {p['abstract']}" for p in papers
                ]
                self.pinecone.fit_bm25(corpus)
            else:
                logger.info("STAGE 4: BM25 encoder already fitted — skipping")

            # ── Stage 5: Embed ──────────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 5: GENERATING EMBEDDINGS (chunks)")
            logger.info("=" * 60)
            chunks = self.embedder.embed_papers(papers)

            chunks_ok   = [c for c in chunks if c.get("embedding")]
            chunks_fail = [c for c in chunks if not c.get("embedding")]

            if chunks_fail:
                logger.warning(f"{len(chunks_fail)} chunks have no embedding — skipping upload")

            # ── Stage 6: Upsert ─────────────────────────────────────────
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 6: UPLOADING CHUNKS TO PINECONE")
            logger.info("=" * 60)

            if not chunks_ok:
                logger.warning("No embedded chunks to upload.")
                result = {"upserted": 0, "skipped": 0, "failed": len(chunks_fail), "failed_ids": []}
            else:
                result = self.pinecone.upsert_chunks(chunks_ok, skip_existing=True)

                # Mark original paper IDs as completed
                # (a paper is "done" when ALL its chunks are uploaded)
                uploaded_chunk_ids = set(result.get("failed_ids", []))
                completed_paper_ids = list({
                    c["id"] for c in chunks_ok
                    if c["chunk_id"] not in uploaded_chunk_ids
                })
                self.state.mark_completed(completed_paper_ids)

            # ── Final report ────────────────────────────────────────────
            duration = (datetime.now() - start).total_seconds()
            stats = {
                "duration_seconds":   duration,
                "categories":         categories,
                "papers_fetched":     fetched_count,
                "papers_validated":   len(papers),
                "chunks_total":       len(chunks),
                "chunks_uploaded":    result["upserted"],
                "chunks_skipped":     result.get("skipped", 0),
                "chunks_failed":      result.get("failed", 0) + len(chunks_fail),
                "embedding_cost":     self.embedder.total_cost,
                "total_tokens":       self.embedder.total_tokens,
            }
            self._print_report(stats)
            return stats

        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted — progress saved, run again to resume.")
            raise
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.info("Progress saved — run again to resume.")
            raise

    # ------------------------------------------------------------------

    def _print_report(self, s: Dict):
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Categories : {', '.join(s['categories'])}")
        print(f"Duration   : {s['duration_seconds']:.1f}s")
        print(f"\nPapers     : fetched={s['papers_fetched']}  validated={s['papers_validated']}")
        print(f"Chunks     : total={s['chunks_total']}  uploaded={s['chunks_uploaded']}  "
              f"skipped={s['chunks_skipped']}  failed={s['chunks_failed']}")
        print(f"\nCost       : tokens={s['total_tokens']:,}  cost=${s['embedding_cost']:.4f}")
        print("=" * 60 + "\n")
