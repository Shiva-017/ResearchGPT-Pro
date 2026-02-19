# backend/scripts/ingestion/fetchers/semantic_scholar.py

"""
Semantic Scholar API fetcher for citation enrichment.

Uses the BATCH endpoint: POST /paper/batch
  - Up to 500 papers per request
  - 19K papers = ~40 requests instead of 19K individual ones
  - Free tier: 1 request/second without key

For each paper, fetches:
  - Citation count
  - List of papers it cites (references)  
  - List of papers that cite it (citations)
"""

import httpx
import time
import json
import os
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm


class SemanticScholarFetcher:
    """
    Enrich papers with citation data from Semantic Scholar.
    Uses batch endpoint for efficiency + disk cache.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    BATCH_SIZE = 500  # S2 batch limit
    PAPER_FIELDS = "paperId,externalIds,citationCount,influentialCitationCount,citations.paperId,citations.externalIds,references.paperId,references.externalIds"

    def __init__(
        self,
        cache_dir: str = "backend/data/processed",
        api_key: Optional[str] = None,
        delay: float = 1.5,  # seconds between batch requests
    ):
        self.cache_dir = os.path.join(cache_dir, "s2_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.api_key = api_key
        self.delay = delay
        self.last_request = 0.0

        # Stats
        self.fetched = 0
        self.cached = 0
        self.failed = 0
        self.not_found = 0

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key

        self.client = httpx.Client(headers=headers, timeout=60.0)
        logger.info(f"SemanticScholar fetcher ready (batch mode, delay={delay}s)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Enrich papers with citation data using batch endpoint.

        Adds to each paper:
          - s2_id, citation_count, influential_citations
          - references: list of arXiv IDs this paper cites
          - cited_by: list of arXiv IDs that cite this paper
        """
        logger.info(f"Enriching {len(papers)} papers with Semantic Scholar…")

        # Build lookup: arxiv_id → paper dict
        paper_map = {}
        to_fetch = []
        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id:
                continue

            # Check cache
            cached = self._load_cache(arxiv_id)
            if cached is not None:
                self._apply_enrichment(paper, cached)
                self.cached += 1
            else:
                paper_map[arxiv_id] = paper
                to_fetch.append(arxiv_id)

        logger.info(f"Cached: {self.cached} | Need to fetch: {len(to_fetch)}")

        if not to_fetch:
            logger.info("All papers already cached!")
            return papers

        # Fetch in batches of 500
        total_batches = (len(to_fetch) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        logger.info(f"Fetching in {total_batches} batches of {self.BATCH_SIZE}…")

        for i in tqdm(range(0, len(to_fetch), self.BATCH_SIZE), desc="S2 batches"):
            batch_ids = to_fetch[i: i + self.BATCH_SIZE]
            results = self._fetch_batch(batch_ids)

            if results is None:
                # Total failure for this batch
                self.failed += len(batch_ids)
                continue

            # Match results back to papers
            for s2_paper in results:
                if s2_paper is None:
                    self.not_found += 1
                    continue

                # Find the arxiv_id for this result
                ext_ids = s2_paper.get("externalIds") or {}
                arxiv_id = ext_ids.get("ArXiv")

                if arxiv_id and arxiv_id in paper_map:
                    self._save_cache(arxiv_id, s2_paper)
                    self._apply_enrichment(paper_map[arxiv_id], s2_paper)
                    self.fetched += 1
                else:
                    # S2 returned a paper we can't match back
                    self.not_found += 1

            # Cache misses (papers S2 didn't return = not found)
            returned_arxiv_ids = set()
            for s2_paper in results:
                if s2_paper:
                    ext = s2_paper.get("externalIds") or {}
                    if ext.get("ArXiv"):
                        returned_arxiv_ids.add(ext["ArXiv"])

            for aid in batch_ids:
                if aid not in returned_arxiv_ids and not self._load_cache(aid):
                    self._save_cache(aid, {"not_found": True})
                    self.not_found += 1

        logger.info(
            f"Enrichment complete: "
            f"fetched={self.fetched}, cached={self.cached}, "
            f"not_found={self.not_found}, failed={self.failed}"
        )
        return papers

    # ------------------------------------------------------------------
    # Batch fetch
    # ------------------------------------------------------------------

    def _fetch_batch(self, arxiv_ids: List[str], retries: int = 3) -> Optional[List]:
        """
        Fetch a batch of papers via POST /paper/batch.
        Returns list of paper dicts (some may be None if not found).
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/paper/batch"
        params = {"fields": self.PAPER_FIELDS}
        body = {"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]}

        for attempt in range(retries):
            try:
                resp = self.client.post(url, params=params, json=body)

                if resp.status_code == 429:
                    wait = min(60 * (attempt + 1), 300)
                    logger.warning(f"S2 rate limited, waiting {wait}s (attempt {attempt + 1}/{retries})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as e:
                logger.warning(f"S2 batch error: {e.response.status_code} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep(10 * (attempt + 1))
            except Exception as e:
                logger.warning(f"S2 batch failed: {e} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep(10 * (attempt + 1))

        logger.error(f"S2 batch failed after {retries} retries")
        return None

    # ------------------------------------------------------------------
    # Apply enrichment
    # ------------------------------------------------------------------

    def _apply_enrichment(self, paper: Dict, data: Dict):
        """Apply S2 data to a paper dict."""
        if data.get("not_found"):
            return

        paper["s2_id"] = data.get("paperId", "")
        paper["citation_count"] = data.get("citationCount", 0)
        paper["influential_citations"] = data.get("influentialCitationCount", 0)
        paper["references"] = self._extract_arxiv_ids(data.get("references", []))
        paper["cited_by"] = self._extract_arxiv_ids(data.get("citations", []))

    def _extract_arxiv_ids(self, papers_list: List) -> List[str]:
        """Extract arXiv IDs from S2 paper list."""
        ids = []
        for p in papers_list or []:
            if p is None:
                continue
            ext = p.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv")
            if arxiv_id:
                ids.append(arxiv_id)
        return ids

    # ------------------------------------------------------------------
    # Rate limit + cache
    # ------------------------------------------------------------------

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()

    def _cache_path(self, arxiv_id: str) -> str:
        safe_id = arxiv_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_id}.json")

    def _load_cache(self, arxiv_id: str) -> Optional[Dict]:
        path = self._cache_path(arxiv_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _save_cache(self, arxiv_id: str, data: Dict):
        path = self._cache_path(arxiv_id)
        with open(path, "w") as f:
            json.dump(data, f)

    def get_stats(self) -> Dict:
        return {
            "fetched": self.fetched,
            "cached": self.cached,
            "not_found": self.not_found,
            "failed": self.failed,
        }
