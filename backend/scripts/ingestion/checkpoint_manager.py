# backend/scripts/ingestion/checkpoint_manager.py

import json
import os
import shutil
from typing import Dict, List, Set
from datetime import datetime
from loguru import logger

class IngestionState:
    """Track ingestion state across failures"""
    
    def __init__(self, checkpoint_dir: str = "backend/data/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.state_file = os.path.join(checkpoint_dir, "ingestion_state.json")
        self.embeddings_dir = os.path.join(checkpoint_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Load existing state or create new
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from disk"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                logger.info(f"Loaded existing state: {state['completed_count']} papers completed")
                return state
        else:
            logger.info("Starting fresh ingestion")
            return {
                'started_at': datetime.now().isoformat(),
                'completed_papers': [],
                'failed_papers': [],
                'embedded_papers': [],
                'completed_count': 0,
                'last_checkpoint': None
            }
    
    def _save_state(self):
        """Persist state to disk"""
        self.state['last_checkpoint'] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_paper_completed(self, paper_id: str) -> bool:
        """Check if paper already processed"""
        return paper_id in self.state['completed_papers']
    
    def is_paper_embedded(self, paper_id: str) -> bool:
        """Check if paper has embedding saved"""
        return paper_id in self.state['embedded_papers']
    
    def save_embedding(self, paper_id: str, embedding: List[float]):
        """Save embedding to disk (before Pinecone upload)"""
        embedding_file = os.path.join(
            self.embeddings_dir,
            f"{paper_id.replace(':', '_')}.json"
        )
        
        with open(embedding_file, 'w') as f:
            json.dump(embedding, f)
    
    def load_embedding(self, paper_id: str) -> List[float]:
        """Load embedding from disk"""
        embedding_file = os.path.join(
            self.embeddings_dir,
            f"{paper_id.replace(':', '_')}.json"
        )
        
        if os.path.exists(embedding_file):
            with open(embedding_file, 'r') as f:
                return json.load(f)
        return None
    
    def mark_embedded(self, paper_ids: List[str]):
        """Mark papers as embedded"""
        self.state['embedded_papers'].extend(paper_ids)
        self._save_state()
    
    def mark_completed(self, paper_ids: List[str]):
        """Mark papers as fully processed"""
        self.state['completed_papers'].extend(paper_ids)
        self.state['completed_count'] = len(self.state['completed_papers'])
        self._save_state()
        
        logger.info(f"Checkpoint saved: {self.state['completed_count']} papers")
    
    def mark_failed(self, paper_id: str, error: str):
        """Mark paper as failed"""
        self.state['failed_papers'].append({
            'paper_id': paper_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self._save_state()
    
    def get_completed_ids(self) -> Set[str]:
        """Get set of completed paper IDs"""
        return set(self.state['completed_papers'])
    
    def get_embedded_ids(self) -> Set[str]:
        """Get set of embedded paper IDs"""
        return set(self.state['embedded_papers'])
    
    def clear(self):
        """Clear state (start fresh)"""
        # Remove state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        
        # Remove all embedding files
        if os.path.exists(self.embeddings_dir):
            shutil.rmtree(self.embeddings_dir)
            os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Reset state
        self.state = {
            'started_at': datetime.now().isoformat(),
            'completed_papers': [],
            'failed_papers': [],
            'embedded_papers': [],
            'completed_count': 0,
            'last_checkpoint': None
        }
        
        logger.info("State and embeddings cleared")