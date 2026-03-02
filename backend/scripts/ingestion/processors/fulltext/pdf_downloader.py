# backend/scripts/ingestion/processors/fulltext/pdf_downloader.py

"""
Fault-tolerant PDF downloader for arXiv papers.

Downloads PDFs with:
  - Rate limiting (respect arXiv's 1 req/3s guideline)
  - Disk caching (skip already-downloaded PDFs)
  - Retry with exponential backoff
  - Progress tracking
"""

import os
import time
import httpx
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class PdfDownloader:
    """Download arXiv PDFs to a local directory."""

    def __init__(
        self,
        output_dir: str = "backend/data/pdfs",
        rate_limit: float = 3.0,      # seconds between requests (arXiv guideline)
        timeout: float = 60.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.timeout = timeout

        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "ResearchGPT-Pro/1.0 (academic research tool; mailto:your@email.com)"
            },
        )

        logger.info(f"PdfDownloader ready → {self.output_dir} (rate: 1 req/{rate_limit}s)")

    def _pdf_path(self, paper_id: str) -> Path:
        """Get local path for a paper PDF."""
        # paper_id is like "arxiv:2504.12345" → "2504.12345.pdf"
        safe_id = paper_id.replace("arxiv:", "").replace("/", "_")
        return self.output_dir / f"{safe_id}.pdf"

    def is_downloaded(self, paper_id: str) -> bool:
        """Check if a paper is already downloaded."""
        path = self._pdf_path(paper_id)
        return path.exists() and path.stat().st_size > 1000  # > 1KB = valid PDF

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
    def _download_one(self, pdf_url: str, dest: Path) -> bool:
        """Download a single PDF. Retries on failure."""
        response = self.client.get(pdf_url)
        response.raise_for_status()

        if len(response.content) < 1000:
            raise ValueError(f"PDF too small ({len(response.content)} bytes)")

        dest.write_bytes(response.content)
        return True

    def download_papers(
        self,
        papers: List[Dict],
        max_papers: Optional[int] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Download PDFs for a list of papers.

        Returns:
            (successful_papers, failed_papers)
            Each successful paper gets a 'pdf_local_path' field added.
        """
        if max_papers:
            papers = papers[:max_papers]

        # Filter out already-downloaded
        to_download = []
        already_done = []
        for paper in papers:
            path = self._pdf_path(paper["id"])
            if self.is_downloaded(paper["id"]):
                paper["pdf_local_path"] = str(path)
                already_done.append(paper)
            else:
                to_download.append(paper)

        logger.info(
            f"PDFs: {len(already_done)} cached, {len(to_download)} to download"
        )

        successful = list(already_done)
        failed = []

        for paper in tqdm(to_download, desc="Downloading PDFs"):
            pdf_url = paper.get("pdf_url", "")
            if not pdf_url:
                failed.append(paper)
                continue

            dest = self._pdf_path(paper["id"])

            try:
                self._download_one(pdf_url, dest)
                paper["pdf_local_path"] = str(dest)
                successful.append(paper)
            except Exception as e:
                logger.warning(f"Failed to download {paper['id']}: {e}")
                failed.append(paper)

            # Rate limit
            time.sleep(self.rate_limit)

        logger.info(
            f"Download complete: {len(successful)} OK, {len(failed)} failed"
        )
        return successful, failed

    def cleanup(self):
        """Close HTTP client."""
        self.client.close()
