# backend/scripts/ingestion/checkpoint_manager.py

import json
import os
import shutil
from typing import Dict, List, Set, Optional
from datetime import datetime
from loguru import logger


class IngestionState:
    """
    Track ingestion state across failures.
    Uses set-based deduplication to avoid memory bloat at 100K+ papers.
    IDs are stored as newline-delimited text files for fast load/append.
    """

    def __init__(self, checkpoint_dir: str = "backend/data/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.state_file       = os.path.join(checkpoint_dir, "ingestion_state.json")
        self.completed_file   = os.path.join(checkpoint_dir, "completed_ids.txt")
        self.embedded_file    = os.path.join(checkpoint_dir, "embedded_ids.txt")
        self.failed_file      = os.path.join(checkpoint_dir, "failed_papers.jsonl")
        self.embeddings_dir   = os.path.join(checkpoint_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # In-memory sets for fast lookup — loaded once at startup
        self._completed: Set[str] = self._load_id_set(self.completed_file)
        self._embedded: Set[str]  = self._load_id_set(self.embedded_file)

        # Lightweight metadata (no large lists)
        self._meta = self._load_meta()

        logger.info(
            f"State loaded — completed: {len(self._completed)}, "
            f"embedded: {len(self._embedded)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_id_set(self, filepath: str) -> Set[str]:
        """Load IDs from a newline-delimited text file into a set."""
        if not os.path.exists(filepath):
            return set()
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _append_ids(self, filepath: str, ids: List[str]):
        """Append new IDs to the text file (no duplicates written)."""
        with open(filepath, "a", encoding="utf-8") as f:
            for pid in ids:
                f.write(pid + "\n")

    def _load_meta(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "started_at": datetime.now().isoformat(),
            "last_checkpoint": None,
        }

    def _save_meta(self):
        self._meta["last_checkpoint"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self._meta, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_paper_completed(self, paper_id: str) -> bool:
        return paper_id in self._completed

    def is_paper_embedded(self, paper_id: str) -> bool:
        return paper_id in self._embedded

    def get_completed_ids(self) -> Set[str]:
        return set(self._completed)

    def get_embedded_ids(self) -> Set[str]:
        return set(self._embedded)

    def mark_embedded(self, paper_ids: List[str]):
        """Mark papers as having embeddings saved to disk."""
        new_ids = [pid for pid in paper_ids if pid not in self._embedded]
        if not new_ids:
            return
        self._embedded.update(new_ids)
        self._append_ids(self.embedded_file, new_ids)
        self._save_meta()

    def mark_completed(self, paper_ids: List[str]):
        """Mark papers as fully processed (uploaded to Pinecone)."""
        new_ids = [pid for pid in paper_ids if pid not in self._completed]
        if not new_ids:
            return
        self._completed.update(new_ids)
        self._append_ids(self.completed_file, new_ids)
        self._save_meta()
        logger.info(f"Checkpoint saved — total completed: {len(self._completed)}")

    def mark_failed(self, paper_id: str, error: str):
        """Append a failed paper record to the JSONL log."""
        record = {
            "paper_id": paper_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.failed_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_failed_papers(self) -> List[Dict]:
        """Load all failed paper records."""
        if not os.path.exists(self.failed_file):
            return []
        records = []
        with open(self.failed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def save_embedding(self, paper_id: str, embedding: List[float]):
        """Persist an embedding vector to disk."""
        path = os.path.join(
            self.embeddings_dir,
            f"{paper_id.replace(':', '_').replace('/', '_')}.json"
        )
        with open(path, "w") as f:
            json.dump(embedding, f)

    def load_embedding(self, paper_id: str) -> Optional[List[float]]:
        """Load an embedding vector from disk cache."""
        path = os.path.join(
            self.embeddings_dir,
            f"{paper_id.replace(':', '_').replace('/', '_')}.json"
        )
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    # ------------------------------------------------------------------
    # Stats & reset
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        failed = self.get_failed_papers()
        return {
            "total_completed": len(self._completed),
            "total_embedded":  len(self._embedded),
            "total_failed":    len(failed),
            "started_at":      self._meta.get("started_at"),
            "last_checkpoint": self._meta.get("last_checkpoint"),
        }

    def clear(self):
        """Wipe all state and embedding cache (start fresh)."""
        for path in [self.state_file, self.completed_file,
                     self.embedded_file, self.failed_file]:
            if os.path.exists(path):
                os.remove(path)

        if os.path.exists(self.embeddings_dir):
            shutil.rmtree(self.embeddings_dir)
            os.makedirs(self.embeddings_dir, exist_ok=True)

        self._completed = set()
        self._embedded  = set()
        self._meta = {
            "started_at": datetime.now().isoformat(),
            "last_checkpoint": None,
        }
        logger.info("State and embeddings cleared.")
