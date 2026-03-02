# backend/app/db/pinecone_client.py

"""
Pinecone client with:
- Chunk-aware upsert  (chunk_id as vector ID, chunk_type in metadata)
- Hybrid sparse-dense search (BM25 + cosine)
- Fault-tolerant batched upsert with checkpointing
"""

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict, Optional, Set
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
import os
import pickle


class PineconeClient:

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        checkpoint_dir: str = "backend/data/checkpoints",
    ):
        self.pc             = Pinecone(api_key=api_key)
        self.index_name     = index_name
        self.dimension      = dimension
        self.checkpoint_dir = checkpoint_dir
        self.bm25_path      = os.path.join(checkpoint_dir, "bm25_encoder.pkl")

        os.makedirs(checkpoint_dir, exist_ok=True)

        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

        # BM25 encoder — loaded if available, else None until fit() is called
        self.bm25: Optional[BM25Encoder] = self._load_bm25()

        logger.info(f"Pinecone index '{self.index_name}' ready | BM25: {'loaded' if self.bm25 else 'not fitted yet'}")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_index_exists(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name in existing:
            return

        logger.info(f"Creating index '{self.index_name}' (dimension={self.dimension})")
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric="dotproduct",        # required for hybrid sparse-dense
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        deadline = time.time() + 300
        while not self.pc.describe_index(self.index_name).status["ready"]:
            if time.time() > deadline:
                raise TimeoutError("Index creation timed out (5 min)")
            logger.info("Waiting for index to be ready…")
            time.sleep(3)

        logger.info(f"Index '{self.index_name}' created")

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def fit_bm25(self, corpus: List[str]):
        """
        Fit BM25 encoder on a corpus of strings and save to disk.
        Call this once after fetching all papers, before upserting.
        corpus should be [title + " " + abstract, ...] for each paper.
        """
        logger.info(f"Fitting BM25 on {len(corpus)} documents…")
        self.bm25 = BM25Encoder()
        self.bm25.fit(corpus)
        self._save_bm25()
        logger.info("BM25 fitted and saved.")

    def _save_bm25(self):
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def _load_bm25(self) -> Optional[BM25Encoder]:
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                return pickle.load(f)
        return None

    # ------------------------------------------------------------------
    # Metadata preparation
    # ------------------------------------------------------------------

    def prepare_metadata(self, chunk: Dict) -> Dict:
        """Build Pinecone metadata from a chunk dict."""
        authors    = chunk.get("authors", [])
        categories = chunk.get("categories", [])

        return {
            # --- display fields ---
            "title":            chunk.get("title", "")[:500],
            "abstract":         chunk.get("abstract", "")[:1000],
            "authors":          ", ".join(authors[:10])[:500],
            "first_author":     authors[0] if authors else "Unknown",
            "year":             chunk.get("year", 0),
            "pdf_url":          chunk.get("pdf_url", ""),

            # --- chunk fields ---
            "chunk_type":       chunk.get("chunk_type", "full"),
            "paper_id":         chunk.get("id", ""),               # original paper ID
            "section_heading":  chunk.get("section_heading", "")[:200],
            "is_fulltext":      chunk.get("is_fulltext", False),
            "chunk_snippet":    chunk.get("chunk_text", "")[:800],  # actual chunk text for LLM context

            # --- filter fields ---
            "primary_category": categories[0] if categories else "unknown",
            "categories":       ",".join(categories[:10])[:200],
            "field":            self._extract_field(categories[0] if categories else ""),
            "citation_count":   chunk.get("citation_count", 0),
            "citation_tier":    self._citation_tier(chunk.get("citation_count", 0)),

            # --- helper flags ---
            "has_code":         "github" in chunk.get("abstract", "").lower(),
            "is_survey":        any(
                                    w in chunk.get("title", "").lower()
                                    for w in ["survey", "review", "overview"]
                                ),
            "author_count":     len(authors),

            # --- identifiers ---
            "arxiv_id":         chunk.get("arxiv_id", ""),
            "source":           chunk.get("source", "arxiv"),
        }

    def _extract_field(self, category: str) -> str:
        prefix = category.split(".")[0] if "." in category else category
        return {
            "cs":      "Computer Science",
            "math":    "Mathematics",
            "physics": "Physics",
            "stat":    "Statistics",
            "eess":    "Electrical Engineering",
            "q-bio":   "Quantitative Biology",
            "q-fin":   "Quantitative Finance",
            "econ":    "Economics",
        }.get(prefix, "Other")

    def _citation_tier(self, count: int) -> str:
        if count >= 100: return "high"
        if count >= 10:  return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _upsert_batch(self, vectors: List[Dict]) -> int:
        result = self.index.upsert(vectors=vectors)
        return result.upserted_count

    def upsert_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 100,
        checkpoint_every: int = 500,
        skip_existing: bool = True,
    ) -> Dict:
        """
        Upsert embedded chunks into Pinecone.

        Each chunk must have:
          - chunk_id   : unique vector ID
          - embedding  : List[float]
          - chunk_text : used for BM25 sparse vector (if bm25 fitted)
          + all paper metadata fields

        Returns stats dict.
        """
        logger.info(f"Upserting {len(chunks)} chunks to Pinecone…")

        skipped = 0
        if skip_existing:
            existing = self._get_stored_chunk_ids()
            original_len = len(chunks)
            chunks = [c for c in chunks if c["chunk_id"] not in existing]
            skipped = original_len - len(chunks)
            if skipped:
                logger.info(f"Skipping {skipped} already-stored chunks")
            if not chunks:
                logger.info("All chunks already stored.")
                return {"upserted": 0, "skipped": skipped, "failed": 0, "failed_ids": []}

        total_upserted = 0
        failed_ids: List[str] = []
        start = time.time()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            vectors = []
            batch_chunk_ids = []

            for chunk in batch:
                if not chunk.get("embedding"):
                    logger.warning(f"Chunk {chunk['chunk_id']} has no embedding — skipping")
                    failed_ids.append(chunk["chunk_id"])
                    continue

                vec = {
                    "id":       chunk["chunk_id"],
                    "values":   chunk["embedding"],
                    "metadata": self.prepare_metadata(chunk),
                }

                # Add sparse vector if BM25 is fitted
                if self.bm25:
                    try:
                        vec["sparse_values"] = self.bm25.encode_documents(
                            chunk.get("chunk_text", chunk.get("title", ""))
                        )
                    except Exception as e:
                        logger.warning(f"BM25 encode failed for {chunk['chunk_id']}: {e}")

                vectors.append(vec)
                batch_chunk_ids.append(chunk["chunk_id"])

            if not vectors:
                continue

            try:
                count = self._upsert_batch(vectors)
                total_upserted += count

                if total_upserted % checkpoint_every == 0 or i + batch_size >= len(chunks):
                    ok_ids = [cid for cid in batch_chunk_ids if cid not in failed_ids]
                    self._save_checkpoint(ok_ids, total_upserted)

                time.sleep(0.1)   # respect free-tier 10 writes/sec

                if total_upserted % 500 == 0 or i + batch_size >= len(chunks):
                    elapsed = time.time() - start
                    rate    = total_upserted / elapsed if elapsed else 0
                    pct     = total_upserted / len(chunks) * 100
                    logger.info(
                        f"Progress: {total_upserted}/{len(chunks)} "
                        f"({pct:.1f}%) — {rate:.1f} chunks/sec"
                    )

            except Exception as e:
                logger.error(f"Batch upsert failed at index {i}: {e}")
                failed_ids.extend(batch_chunk_ids)

        logger.info(f"Upsert complete — {total_upserted} chunks | failed: {len(failed_ids)}")
        return {
            "upserted":   total_upserted,
            "skipped":    skipped,
            "failed":     len(failed_ids),
            "failed_ids": failed_ids,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
        query_text: Optional[str] = None,
        alpha: float = 0.7,
    ):
        """
        Query Pinecone.

        If BM25 is fitted AND query_text is provided, performs hybrid
        sparse-dense search with weighting alpha (dense) / (1-alpha) (sparse).
        Otherwise falls back to pure dense search.

        Args:
            vector:     Dense query embedding.
            top_k:      Results to return.
            filter:     Pinecone metadata filter dict.
            query_text: Raw query string for BM25 sparse vector.
            alpha:      Dense weight (0-1). 0.7 = 70% dense, 30% BM25.
        """
        kwargs = dict(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
        )
        if filter:
            kwargs["filter"] = filter

        if self.bm25 and query_text:
            try:
                sparse = self.bm25.encode_queries(query_text)
                # Scale vectors for hybrid weighting
                kwargs["vector"]        = [v * alpha for v in vector]
                kwargs["sparse_vector"] = {
                    "indices": sparse["indices"],
                    "values":  [v * (1 - alpha) for v in sparse["values"]],
                }
            except Exception as e:
                logger.warning(f"BM25 query encode failed, falling back to dense: {e}")

        return self.index.query(**kwargs)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _get_stored_chunk_ids(self) -> Set[str]:
        path = os.path.join(self.checkpoint_dir, f"pinecone_stored_{self.index_name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return set(json.load(f).get("stored_ids", []))
        return set()

    def _save_checkpoint(self, new_ids: List[str], total: int):
        path = os.path.join(self.checkpoint_dir, f"pinecone_stored_{self.index_name}.json")
        data = {"stored_ids": []}
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)

        existing = set(data["stored_ids"])
        data["stored_ids"].extend([cid for cid in new_ids if cid not in existing])
        data["total_count"]   = total
        data["last_updated"]  = time.time()

        with open(path, "w") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return self.index.describe_index_stats()
