# backend/scripts/ingestion/fetchers/arxiv_fetcher.py

"""
Fault-tolerant arXiv paper fetcher.

Uses the arxiv library's built-in pagination — one Search per category,
iterate through results() which handles page offsets internally.
"""

import arxiv
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import time
import json
import os

try:
    from arxiv import UnexpectedEmptyPageError
except ImportError:
    UnexpectedEmptyPageError = Exception


class ArxivFetcher:
    """Fault-tolerant arXiv paper fetcher."""

    def __init__(
        self,
        rate_limit: float = 0.34,   # 3 req/sec = 0.33s between
        cache_dir: str = "backend/data/raw",
    ):
        self.rate_limit = rate_limit
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"ArxivFetcher ready (rate limit {rate_limit}s ≈ {1/rate_limit:.0f} req/s)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_papers(
        self,
        category: str = "cs.AI",
        max_results: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch papers from a single arXiv category.

        Uses one arxiv.Search with the full max_results and iterates
        via the library's built-in pagination (Client.results()).
        """
        # ── Cache check ──────────────────────────────────────────────
        date_suffix = f"_{start_date}_{end_date}" if start_date and end_date else ""
        cache_file = os.path.join(
            self.cache_dir,
            f"arxiv_{category.replace('.', '_')}_{max_results}{date_suffix}.json",
        )
        if os.path.exists(cache_file):
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, "r") as f:
                papers = json.load(f)
            logger.info(f"Cached: {len(papers)} papers")
            return papers

        # ── Build query ──────────────────────────────────────────────
        query = f"cat:{category}"
        if start_date and end_date:
            query += f" AND submittedDate:[{start_date} TO {end_date}]"

        logger.info(f"Fetching up to {max_results} papers | query: {query}")

        # ── Single search, iterate with built-in pagination ──────────
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        # arxiv.Client handles rate limiting and pagination internally.
        # page_size controls how many results per API call (default 100).
        client = arxiv.Client(
            page_size=100,
            delay_seconds=self.rate_limit,   # delay between API pages
            num_retries=3,
        )

        papers = []
        try:
            for result in tqdm(
                client.results(search),
                total=max_results,
                desc=f"Fetching {category}",
            ):
                paper = self._arxiv_result_to_dict(result)
                papers.append(paper)

                # Progress log every 200 papers
                if len(papers) % 200 == 0:
                    logger.info(f"  {category}: {len(papers)}/{max_results} fetched")

        except UnexpectedEmptyPageError:
            # Category has fewer papers than requested — totally normal
            logger.info(
                f"  {category}: exhausted at {len(papers)} papers "
                f"(requested {max_results})"
            )
        except Exception as e:
            error_str = str(e).lower()
            if "empty" in error_str or "unexpectedemptypage" in error_str:
                logger.info(f"  {category}: exhausted at {len(papers)} papers")
            elif "429" in str(e) or "rate limit" in error_str:
                logger.warning(f"  Rate limited at {len(papers)} papers, saving what we have")
            else:
                logger.error(f"  Fetch error at {len(papers)} papers: {e}")
                if not papers:
                    raise

        logger.info(f"  {category}: {len(papers)} papers fetched")

        # ── Cache ────────────────────────────────────────────────────
        if papers:
            with open(cache_file, "w") as f:
                json.dump(papers, f, indent=2)
            logger.info(f"  Cached to {cache_file}")

        return papers

    def fetch_by_categories(
        self,
        categories: List[str],
        papers_per_category: int = 1000,
    ) -> List[Dict]:
        """Fetch from multiple categories, deduplicate across categories."""
        logger.info(
            f"Fetching from {len(categories)} categories "
            f"({papers_per_category} each, target ~{len(categories) * papers_per_category:,})"
        )

        all_papers = []
        for i, category in enumerate(categories, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{i}/{len(categories)}] Category: {category}")
            logger.info(f"{'=' * 60}")

            papers = self.fetch_papers(
                category=category,
                max_results=papers_per_category,
            )
            all_papers.extend(papers)

        # Cross-category dedup
        seen = set()
        unique = []
        for paper in all_papers:
            if paper["id"] not in seen:
                seen.add(paper["id"])
                unique.append(paper)

        dupes = len(all_papers) - len(unique)
        logger.info(f"\nTotal fetched : {len(all_papers):,}")
        logger.info(f"Unique papers : {len(unique):,}")
        logger.info(f"Cross-cat dupes removed: {dupes:,}")

        return unique

    def fetch_recent_papers(
        self,
        category: str = "cs.AI",
        days: int = 7,
    ) -> List[Dict]:
        """Fetch papers from the last N days (incremental updates)."""
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return self.fetch_papers(
            category=category,
            max_results=500,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _arxiv_result_to_dict(self, result: arxiv.Result) -> Dict:
        """Convert arxiv.Result → standard dict."""
        entry_id = result.entry_id.split("/")[-1]
        arxiv_id = entry_id.split("v")[0]

        return {
            "id": f"arxiv:{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": result.title.strip(),
            "abstract": result.summary.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "published_date": result.published.isoformat(),
            "year": result.published.year,
            "categories": result.categories,
            "primary_category": result.primary_category,
            "pdf_url": result.pdf_url,
            "source": "arxiv",
            "citation_count": 0,
        }
