# backend/scripts/ingestion/fetchers/arxiv_fetcher.py

import arxiv
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import time
import json
import os

# Import for handling ArXiv API errors
try:
    from arxiv import UnexpectedEmptyPageError
except ImportError:
    # Fallback for older versions
    UnexpectedEmptyPageError = Exception

class ArxivFetcher:
    """
    Fault-tolerant arXiv paper fetcher
    """
    
    def __init__(
        self,
        rate_limit: float = 0.34,  # 3 requests/second = 0.33s between
        cache_dir: str = "backend/data/raw"
    ):
        """
        Args:
            rate_limit: Seconds between requests (ArXiv limit: 3 req/sec)
            cache_dir: Directory to cache fetched papers
        """
        self.rate_limit = rate_limit
        self.cache_dir = cache_dir
        self.last_request_time = 0.0  # Track last request time for rate limiting
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"ArxivFetcher initialized (rate limit: {rate_limit}s = ~{1/rate_limit:.1f} req/sec)")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_not_exception_type(UnexpectedEmptyPageError)
    )
    def _fetch_single_batch(
        self,
        query: str,
        max_results: int
    ) -> List[arxiv.Result]:
        """
        Fetch a single batch with retry logic and rate limiting
        
        Note: ArXiv API limit is 3 requests/second. The arxiv library
        may make multiple internal requests for large max_results.
        UnexpectedEmptyPageError means we've hit the end of available results.
        
        Args:
            query: Search query
            max_results: Number of results to fetch (keep small to avoid internal pagination)
        
        Returns:
            List of arxiv.Result objects
        """
        # Enforce rate limit BEFORE making request
        self._enforce_rate_limit()
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(search.results())
        return results
    
    def fetch_papers(
        self,
        category: str = "cs.AI",
        max_results: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch papers from arXiv
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.CL')
            max_results: Maximum papers to fetch
            start_date: Filter start date (YYYYMMDD)
            end_date: Filter end date (YYYYMMDD)
        
        Returns:
            List of paper dictionaries
        """
        # Check cache first (include date range in cache key to avoid collisions)
        date_suffix = ""
        if start_date and end_date:
            date_suffix = f"_{start_date}_{end_date}"
        cache_file = os.path.join(
            self.cache_dir,
            f"arxiv_{category.replace('.', '_')}_{max_results}{date_suffix}.json"
        )
        
        if os.path.exists(cache_file):
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                papers = json.load(f)
            logger.info(f"Loaded {len(papers)} papers from cache")
            return papers
        
        # Build query
        query = f"cat:{category}"
        if start_date and end_date:
            query += f" AND submittedDate:[{start_date} TO {end_date}]"
        
        logger.info(f"Fetching {max_results} papers from arXiv")
        logger.info(f"Query: {query}")
        
        # Fetch papers - ArXiv API has strict limits (3 req/sec)
        # Use smaller batches to avoid internal pagination issues
        # ArXiv library internally paginates, so smaller batches = fewer internal pages
        papers = []
        batch_size = 100  # Even smaller batches to minimize internal pagination issues
        total_fetched = 0
        
        try:
            # Fetch in batches with proper rate limiting
            for batch_start in tqdm(range(0, max_results, batch_size), desc=f"Fetching {category}"):
                batch_max = min(batch_size, max_results - batch_start)
                
                try:
                    # Rate limiting is handled inside _fetch_single_batch
                    # Fetch batch
                    results = self._fetch_single_batch(
                        query=query,
                        max_results=batch_max
                    )
                    
                    # Convert to dict format
                    for result in results:
                        paper = self._arxiv_result_to_dict(result)
                        papers.append(paper)
                        total_fetched += 1
                    
                    # If we got fewer results than requested, we've hit the limit
                    if len(results) < batch_max:
                        logger.info(f"Only got {len(results)} results (requested {batch_max}), may have hit API limit")
                        break
                    
                    # Additional rate limiting after batch (redundant but safe)
                    # This ensures we don't exceed 3 req/sec even with internal pagination
                    self._enforce_rate_limit()
                    
                except UnexpectedEmptyPageError as e:
                    # ArXiv API returned empty page - this category has fewer papers than requested
                    # This is normal and expected, not an error
                    logger.info(f"ArXiv returned empty page at batch {batch_start} - category may have fewer papers")
                    logger.info(f"Successfully fetched {total_fetched} papers from {category} (requested {max_results})")
                    break
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a rate limit error
                    if "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                        logger.warning(f"Rate limit hit at batch {batch_start}, waiting longer before retry...")
                        # Wait longer when rate limited
                        time.sleep(5)
                        # Continue to retry via tenacity decorator
                        raise
                    elif "empty" in error_str.lower() or "UnexpectedEmptyPageError" in error_str:
                        # Another form of empty page error
                        logger.info(f"Empty page error at batch {batch_start} - stopping fetch for this category")
                        logger.info(f"Successfully fetched {total_fetched} papers from {category}")
                        break
                    else:
                        logger.error(f"Failed to fetch batch starting at {batch_start}: {e}")
                        # For other errors, try to continue with next batch
                        # But if we've been failing a lot, maybe stop
                        if batch_start > 0 and len(papers) == 0:
                            logger.warning(f"No papers fetched yet, stopping after repeated failures")
                            break
                        continue
            
        except Exception as e:
            logger.error(f"Failed to fetch papers: {e}")
            # If we got some papers, return them rather than failing completely
            if papers:
                logger.info(f"Returning {len(papers)} papers fetched before error")
                return papers
            raise
        
        logger.info(f"Fetched {len(papers)} papers")
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(papers, f, indent=2)
        
        logger.info(f"Cached to {cache_file}")
        
        return papers
    
    def fetch_by_categories(
        self,
        categories: List[str],
        papers_per_category: int = 1000
    ) -> List[Dict]:
        """
        Fetch papers from multiple categories
        
        Args:
            categories: List of arXiv categories
            papers_per_category: Papers to fetch per category
        
        Returns:
            Combined list of unique papers
        """
        logger.info(f"Fetching from {len(categories)} categories")
        
        all_papers = []
        
        for category in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {category}")
            logger.info(f"{'='*60}")
            
            papers = self.fetch_papers(
                category=category,
                max_results=papers_per_category
            )
            
            all_papers.extend(papers)
        
        # Remove duplicates
        seen = set()
        unique_papers = []
        
        for paper in all_papers:
            if paper['id'] not in seen:
                seen.add(paper['id'])
                unique_papers.append(paper)
        
        logger.info(f"\n Total papers: {len(all_papers)}")
        logger.info(f"Unique papers: {len(unique_papers)}")
        logger.info(f"Removed {len(all_papers) - len(unique_papers)} duplicates")
        
        return unique_papers
    
    def _arxiv_result_to_dict(self, result: arxiv.Result) -> Dict:
        """
        Convert arxiv.Result to standard dictionary
        
        Args:
            result: arxiv.Result object
        
        Returns:
            Paper dictionary
        """
        # Extract arXiv ID (remove version number)
        entry_id = result.entry_id.split('/')[-1]
        arxiv_id = entry_id.split('v')[0]  # Remove version
        
        return {
            'id': f"arxiv:{arxiv_id}",
            'arxiv_id': arxiv_id,
            'title': result.title.strip(),
            'abstract': result.summary.strip().replace('\n', ' '),
            'authors': [author.name for author in result.authors],
            'published': result.published.strftime('%Y-%m-%d'),
            'published_date': result.published.isoformat(),
            'year': result.published.year,
            'categories': result.categories,
            'primary_category': result.primary_category,
            'pdf_url': result.pdf_url,
            'source': 'arxiv',
            'citation_count': 0  # Will be enriched later from Semantic Scholar
        }
    
    def fetch_recent_papers(
        self,
        category: str = "cs.AI",
        days: int = 7
    ) -> List[Dict]:
        """
        Fetch papers from the last N days (for incremental updates)
        
        Args:
            category: arXiv category
            days: Number of days to look back
        
        Returns:
            List of recent papers
        """
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching papers from last {days} days")
        
        return self.fetch_papers(
            category=category,
            max_results=500, 
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )