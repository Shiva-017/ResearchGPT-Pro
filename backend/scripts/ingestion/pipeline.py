# backend/scripts/ingestion/pipeline.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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
    Complete fault-tolerant ingestion pipeline
    Orchestrates: Fetch → Validate → Deduplicate → Embed → Store
    """
    
    def __init__(self, resume: bool = True):
        """
        Args:
            resume: Resume from checkpoint (default: True)
        """
        logger.info("Initializing ingestion pipeline...")
        
        # State management
        self.state = IngestionState()
        
        if not resume:
            logger.warning("Starting fresh (ignoring checkpoints)")
            self.state.clear()
        
        # Initialize components
        self.pinecone = PineconeClient(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=1536
        )
        
        self.arxiv_fetcher = ArxivFetcher()
        self.validator = PaperValidator()
        self.deduplicator = Deduplicator()
        
        self.embedder = FaultTolerantEmbedder(
            api_key=settings.openai_api_key,
            state_manager=self.state
        )
        
        logger.info("Pipeline initialized")
    
    def run(
        self,
        categories: List[str] = ["cs.AI"],
        papers_per_category: int = 1000
    ) -> Dict:
        """
        Run complete pipeline with fault tolerance
        
        Args:
            categories: arXiv categories to fetch
            papers_per_category: Papers per category
        
        Returns:
            Pipeline statistics
        """
        start_time = datetime.now()
        
        try:
            logger.info("\n" + "="*60)
            logger.info("STAGE 1: FETCHING PAPERS FROM ARXIV")
            logger.info("="*60)
            
            papers_raw = self.arxiv_fetcher.fetch_by_categories(
                categories=categories,
                papers_per_category=papers_per_category
            )
            papers_fetched_count = len(papers_raw)
            
            logger.info("\n" + "="*60)
            logger.info("STAGE 2: VALIDATING PAPERS")
            logger.info("="*60)
            
            papers = self.validator.validate_papers(papers_raw)
            
            logger.info("\n" + "="*60)
            logger.info("STAGE 3: REMOVING DUPLICATES")
            logger.info("="*60)
            
            papers = self.deduplicator.deduplicate(papers)
            
            logger.info("\n" + "="*60)
            logger.info("STAGE 4: GENERATING EMBEDDINGS")
            logger.info("="*60)
            
            papers = self.embedder.embed_papers(papers)
            
            # Validate that all papers have embeddings before upload
            papers_with_embeddings = [p for p in papers if 'embedding' in p and p['embedding']]
            papers_without_embeddings = [p for p in papers if 'embedding' not in p or not p['embedding']]
            
            if papers_without_embeddings:
                logger.warning(f"{len(papers_without_embeddings)} papers missing embeddings, skipping upload")
                for paper in papers_without_embeddings:
                    logger.warning(f"  - {paper['id']}: {paper.get('title', 'Unknown')[:50]}")
            
            logger.info("\n" + "="*60)
            logger.info("STAGE 5: UPLOADING TO PINECONE")
            logger.info("="*60)
            
            # Let Pinecone client handle skip_existing (removes redundant filtering)
            # Only filter papers that don't have embeddings
            if not papers_with_embeddings:
                logger.warning("No papers with embeddings to upload!")
                result = {
                    'upserted': 0,
                    'skipped': 0,
                    'failed': len(papers_without_embeddings),
                    'failed_ids': [p['id'] for p in papers_without_embeddings]
                }
            else:
                result = self.pinecone.upsert_papers(
                    papers_with_embeddings,
                    skip_existing=True  # Let Pinecone client handle this
                )
                
                # Mark uploaded papers as completed (only successful ones)
                uploaded_ids = [
                    p['id'] for p in papers_with_embeddings
                    if p['id'] not in result.get('failed_ids', [])
                ]
                self.state.mark_completed(uploaded_ids)
            
            # Calculate final stats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_stats = {
                'duration_seconds': duration,
                'categories': categories,
                'papers_fetched': papers_fetched_count,  # Original count before processing
                'papers_validated': len(papers),
                'papers_with_embeddings': len(papers_with_embeddings),
                'papers_uploaded': result['upserted'],
                'papers_skipped': result.get('skipped', 0),
                'papers_failed': result.get('failed', 0) + len(papers_without_embeddings),
                'embedding_cost': self.embedder.total_cost,
                'total_tokens': self.embedder.total_tokens
            }
            
            self._print_final_report(final_stats)
            
            return final_stats
            
        except KeyboardInterrupt:
            logger.warning("\n Pipeline interrupted (Ctrl+C)")
            logger.info("Progress saved. Run again to resume.")
            raise
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.info("Progress saved. Run again to resume.")
            raise
    
    def _print_final_report(self, stats: Dict):
        """Print final pipeline report"""
        print("\n" + "="*60)
        print("PIPELINE COMPLETED!")
        print("="*60)
        print(f"Categories: {', '.join(stats['categories'])}")
        print(f"Duration: {stats['duration_seconds']:.2f}s")
        print(f"\nPapers:")
        print(f"  Fetched: {stats['papers_fetched']}")
        print(f"  Validated: {stats.get('papers_validated', stats['papers_fetched'])}")
        print(f"  With embeddings: {stats.get('papers_with_embeddings', 0)}")
        print(f"  Uploaded: {stats['papers_uploaded']}")
        print(f"  Skipped: {stats['papers_skipped']}")
        print(f"  Failed: {stats['papers_failed']}")
        print(f"\nCost:")
        print(f"  Tokens used: {stats['total_tokens']:,}")
        print(f"  Total cost: ${stats['embedding_cost']:.4f}")
        print("="*60 + "\n")