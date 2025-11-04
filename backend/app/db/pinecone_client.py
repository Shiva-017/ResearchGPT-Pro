# backend/app/db/pinecone_client.py (COMPLETE, FAULT-TOLERANT VERSION)

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional, Set
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import json
import os

class PineconeClient:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        checkpoint_dir: str = "backend/data/checkpoints"
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        logger.info(f"Pinecone index '{self.index_name}' initialized")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index '{self.index_name}'")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Wait for index to be ready with timeout (max 5 minutes)
            max_wait_time = 300  # 5 minutes
            wait_start = time.time()
            
            while not self.pc.describe_index(self.index_name).status['ready']:
                elapsed = time.time() - wait_start
                if elapsed > max_wait_time:
                    raise TimeoutError(
                        f"Index creation timed out after {max_wait_time}s. "
                        f"Please check Pinecone dashboard."
                    )
                logger.info(f"Waiting for index to be ready... ({elapsed:.0f}s)")
                time.sleep(2)
            
            logger.info(f"Index '{self.index_name}' created")
    
    def prepare_metadata(self, paper: Dict) -> Dict:
        """Prepare comprehensive metadata"""
        authors = paper.get('authors', [])
        categories = paper.get('categories', [])
        
        return {
            # Display fields
            'title': paper['title'][:500],
            'abstract': paper['abstract'][:1000],
            'authors': ', '.join(authors[:10])[:500],
            'first_author': authors[0] if authors else 'Unknown',
            'year': paper.get('year', 0),
            'pdf_url': paper.get('pdf_url', ''),
            
            # Filter fields
            'primary_category': categories[0] if categories else 'unknown',
            'categories': ','.join(categories[:10])[:200],
            'field': self._extract_field(categories[0] if categories else ''),
            'citation_count': paper.get('citation_count', 0),
            'citation_tier': self._get_citation_tier(paper.get('citation_count', 0)),
            'published_month': self._extract_month(paper.get('published', '')),
            
            # Helpers
            'has_code': 'github' in paper.get('abstract', '').lower(),
            'is_survey': any(w in paper['title'].lower() 
                           for w in ['survey', 'review', 'overview']),
            'author_count': len(authors),
            
            # Identifiers
            'arxiv_id': paper.get('arxiv_id', ''),
            'source': paper.get('source', 'arxiv'),
        }
    
    def _extract_field(self, category: str) -> str:
        """Extract field from category"""
        field_map = {
            'cs': 'Computer Science',
            'math': 'Mathematics',
            'physics': 'Physics',
            'stat': 'Statistics'
        }
        prefix = category.split('.')[0] if '.' in category else category
        return field_map.get(prefix, 'Other')
    
    def _get_citation_tier(self, count: int) -> str:
        """Get citation tier"""
        if count >= 100:
            return 'high'
        elif count >= 10:
            return 'medium'
        return 'low'
    
    def _extract_month(self, date_str: str) -> int:
        """Extract month from date"""
        try:
            return int(date_str.split('-')[1]) if '-' in date_str else 0
        except:
            return 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30)
    )
    def _upsert_batch_with_retry(
        self,
        vectors: List[Dict]
    ) -> int:
        """
        Upsert single batch with automatic retry
        
        Returns:
            Number of vectors upserted
        """
        result = self.index.upsert(vectors=vectors)
        return result.upserted_count
    
    def upsert_papers(
        self,
        papers: List[Dict],
        batch_size: int = 100,
        checkpoint_every: int = 500,
        skip_existing: bool = True
    ) -> Dict:
        """
        Upsert papers into Pinecone with fault tolerance
        
        Args:
            papers: List of papers with 'embedding' field
            batch_size: Vectors per batch (max 100 for free tier rate limits)
            checkpoint_every: Save progress every N papers
            skip_existing: Check and skip already stored papers
        
        Returns:
            Dict with statistics
        """
        logger.info(f"Upserting {len(papers)} papers to Pinecone...")
        
        # Check what's already stored (if skip_existing)
        skipped_count = 0
        if skip_existing:
            existing_ids = self._get_stored_paper_ids()
            papers = [p for p in papers if p['id'] not in existing_ids]
            skipped_count = len(existing_ids)
            
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} already stored papers")
            
            if not papers:
                logger.info("All papers already stored!")
                return {
                    'upserted': 0,
                    'skipped': skipped_count,
                    'failed': 0,
                    'failed_ids': []
                }
        
        total_upserted = 0
        failed_ids = []
        start_time = time.time()
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            # Prepare vectors
            vectors = []
            batch_ids = []
            
            for p in batch:
                # Validate paper has embedding
                if 'embedding' not in p:
                    logger.warning(f"Paper {p['id']} has no embedding, skipping")
                    failed_ids.append(p['id'])
                    continue
                
                vectors.append({
                    'id': p['id'],
                    'values': p['embedding'],
                    'metadata': self.prepare_metadata(p)
                })
                batch_ids.append(p['id'])
            
            if not vectors:
                continue
            
            try:
                # Upsert with retry
                count = self._upsert_batch_with_retry(vectors)
                total_upserted += count
                
                # Save checkpoint (track cumulative IDs, not just current batch)
                if total_upserted % checkpoint_every == 0 or i + batch_size >= len(papers):
                    # Get all successfully uploaded IDs from this batch
                    successful_batch_ids = [pid for pid in batch_ids if pid not in failed_ids]
                    self._save_checkpoint(successful_batch_ids, total_upserted)
                
                # Rate limiting (10 writes/sec for free tier)
                time.sleep(0.1)
                
                # Progress logging (use actual paper count, not batch index)
                if total_upserted % 500 == 0 or i + batch_size >= len(papers):
                    elapsed = time.time() - start_time
                    rate = total_upserted / elapsed if elapsed > 0 else 0
                    percentage = (total_upserted / len(papers)) * 100 if len(papers) > 0 else 0
                    
                    logger.info(
                        f"Progress: {total_upserted}/{len(papers)} "
                        f"({percentage:.1f}%) - "
                        f"Rate: {rate:.1f} papers/sec"
                    )
                
            except Exception as e:
                logger.error(f"Failed to upsert batch at index {i}: {e}")
                failed_ids.extend(batch_ids)
                # Continue to next batch instead of failing completely
                continue
        
        # Final logging (OUTSIDE the loop - this was your bug!)
        logger.info(f"Total upserted: {total_upserted}")
        
        if failed_ids:
            logger.warning(f"Failed: {len(failed_ids)} papers")
        
        return {
            'upserted': total_upserted,
            'skipped': skipped_count,
            'failed': len(failed_ids),
            'failed_ids': failed_ids
        }  
    
    def _get_stored_paper_ids(self) -> Set[str]:
        """
        Check which paper IDs are already in Pinecone
        
        Returns:
            Set of paper IDs already stored
        """
        # Try to load from checkpoint file first (faster)
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"pinecone_stored_{self.index_name}.json"
        )
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get('stored_ids', []))
        
        return set()
    
    def _save_checkpoint(self, new_ids: List[str], total_count: int):
        """
        Save checkpoint of stored paper IDs
        
        Args:
            new_ids: Newly stored paper IDs
            total_count: Total papers stored so far
        """
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"pinecone_stored_{self.index_name}.json"
        )
        
        # Load existing
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                # Ensure stored_ids is a list
                if 'stored_ids' not in data:
                    data['stored_ids'] = []
        else:
            data = {'stored_ids': []}
        
        # Add new IDs with deduplication
        existing_set = set(data['stored_ids'])
        new_unique_ids = [pid for pid in new_ids if pid not in existing_set]
        data['stored_ids'].extend(new_unique_ids)
        
        # Ensure we track all IDs (cumulative, not just current batch)
        data['total_count'] = total_count
        data['last_updated'] = time.time()
        
        # Save
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
        
        logger.debug(f"Checkpoint saved: {total_count} papers ({len(new_unique_ids)} new)")
    
    def query(
        self,
        vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ):
        """Query similar vectors"""
        return self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata
        )
    
    def get_stats(self):
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        return stats