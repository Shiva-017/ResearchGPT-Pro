# backend/scripts/check_progress.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.scripts.ingestion.checkpoint_manager import IngestionState
from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient

def main():
    """Check current ingestion progress"""
    
    # Load state
    state = IngestionState()
    stats = state.get_stats()
    
    # Get Pinecone stats
    pinecone = PineconeClient(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name
    )
    pinecone_stats = pinecone.get_stats()
    
    print("\n" + "="*60)
    print("INGESTION PROGRESS")
    print("="*60)
    
    print("\nLocal State:")
    print(f"  Completed: {stats['total_completed']}")
    print(f"  Embedded: {stats['total_embedded']}")
    print(f"  Failed: {stats['total_failed']}")
    
    if stats.get('started_at'):
        print(f"  Started: {stats['started_at']}")
    if stats.get('last_checkpoint'):
        print(f"  Last checkpoint: {stats['last_checkpoint']}")
    
    print("\nPinecone Index:")
    print(f"  Total vectors: {pinecone_stats['total_vector_count']}")
    print(f"  Index fullness: {pinecone_stats['index_fullness']*100:.2f}%")
    
    # Calculate storage
    vectors = pinecone_stats['total_vector_count']
    storage_gb = (vectors * 1536 * 4 + vectors * 2000) / (1024**3)
    print(f"  Storage used: {storage_gb:.3f} GB / 2.0 GB")
    print(f"  Remaining: {2 - storage_gb:.3f} GB")
    
    # Show failed papers
    if stats['total_failed'] > 0:
        print(f"\n Failed Papers ({stats['total_failed']}):")
        
        failed = state.state.get('failed_papers', [])
        for failure in failed[:10]:
            print(f"  - {failure['paper_id']}: {failure['error'][:50]}")
        
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    print("="*60 + "\n")
    
    # Options
    print("Options:")
    print("  1. Continue ingestion (run: python backend/scripts/run_ingestion.py)")
    print("  2. Start fresh (run: python backend/scripts/run_ingestion.py --fresh)")
    print("  3. Retry failed papers (run: python backend/scripts/retry_failed.py)")
    print("  4. Clear state (type 'clear' below)")
    
    choice = input("\nEnter choice (or press Enter to exit): ").strip().lower()
    
    if choice == 'clear':
        confirm = input("Are you sure? This will delete all checkpoints (yes/no): ")
        if confirm.lower() == 'yes':
            state.clear()
            print("State cleared!")

if __name__ == "__main__":
    main()