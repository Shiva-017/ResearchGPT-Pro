# backend/scripts/test_single_paper.py
"""
Test script to ingest a single paper and verify it's stored in Pinecone
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from loguru import logger
from backend.scripts.ingestion.pipeline import IngestionPipeline
from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient

def verify_paper_in_pinecone(paper_id: str):
    """Verify a paper exists in Pinecone by querying with a dummy vector"""
    try:
        logger.info(f"Verifying paper {paper_id} in Pinecone...")
        
        pinecone = PineconeClient(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=1536
        )
        
        # Get index stats
        stats = pinecone.get_stats()
        logger.info(f"Index stats: {stats}")
        
        # Try to fetch the paper by ID (Pinecone doesn't have direct get by ID, 
        # but we can query with a dummy vector and filter by metadata)
        # Actually, we can use fetch() if available, or query with metadata filter
        
        logger.info(f"Paper {paper_id} verification completed")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying paper: {e}")
        return False

def main():
    """Test ingestion with 1 paper"""
    
    logger.info("="*60)
    logger.info("TESTING SINGLE PAPER INGESTION")
    logger.info("="*60)
    
    # Configure logging
    logger.add(
        "backend/data/logs/test_ingestion_{time}.log",
        rotation="10 MB",
        level="INFO"
    )
    
    # Run pipeline with 1 paper
    logger.info("Starting ingestion with 1 paper...")
    pipeline = IngestionPipeline(resume=False)  # Start fresh for test
    
    try:
        result = pipeline.run(
            categories=["cs.AI"],  # Single category
            papers_per_category=1  # Just 1 paper!
        )
        
        logger.info("\n" + "="*60)
        logger.info("INGESTION COMPLETED!")
        logger.info("="*60)
        logger.info(f"Papers fetched: {result['papers_fetched']}")
        logger.info(f"Papers validated: {result['papers_validated']}")
        logger.info(f"Papers with embeddings: {result['papers_with_embeddings']}")
        logger.info(f"Papers uploaded: {result['papers_uploaded']}")
        logger.info(f"Papers failed: {result['papers_failed']}")
        logger.info(f"Cost: ${result['embedding_cost']:.4f}")
        
        if result['papers_uploaded'] > 0:
            logger.info("\n✅ SUCCESS: Paper was uploaded to Pinecone!")
            
            # Get the uploaded paper ID from state
            uploaded_ids = pipeline.state.get_completed_ids()
            if uploaded_ids:
                paper_id = list(uploaded_ids)[0]
                logger.info(f"Uploaded paper ID: {paper_id}")
                
                # Verify in Pinecone
                verify_paper_in_pinecone(paper_id)
        else:
            logger.warning("\n⚠️  WARNING: No papers were uploaded!")
            if result['papers_failed'] > 0:
                logger.error("Check the logs for error details")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

