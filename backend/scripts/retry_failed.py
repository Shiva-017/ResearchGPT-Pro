# backend/scripts/retry_failed.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from loguru import logger
from backend.scripts.ingestion.checkpoint_manager import IngestionState
from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient
from backend.scripts.ingestion.processors.embedder import FaultTolerantEmbedder

def main():
    """Retry failed papers from previous runs"""
    
    logger.info("Retrying failed papers...")
    
    # Load state
    state = IngestionState()
    failed_papers = state.state.get('failed_papers', [])
    
    if not failed_papers:
        logger.info("No failed papers to retry!")
        return
    
    logger.info(f"Found {len(failed_papers)} failed papers")
    
    # Show failures
    print("\nFailed papers:")
    for i, failure in enumerate(failed_papers[:10], 1):
        print(f"{i}. {failure['paper_id']}: {failure['error']}")
    
    if len(failed_papers) > 10:
        print(f"... and {len(failed_papers) - 10} more")
    
    # Ask to retry
    response = input("\nRetry these papers? (yes/no): ")
    
    if response.lower() != 'yes':
        logger.info("Cancelled")
        return
    
    # Initialize services
    pinecone = PineconeClient(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name
    )
    
    embedder = FaultTolerantEmbedder(
        api_key=settings.openai_api_key,
        state_manager=state
    )
    
    # TODO: Load paper data and retry
    # (Implementation depends on how you stored failed papers)
    
    logger.info("Retry logic to be implemented based on your needs")

if __name__ == "__main__":
    main()