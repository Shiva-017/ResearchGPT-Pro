# backend/app/services/embedding_service.py

"""
Thin wrapper around OpenAI embeddings for query-time use.
Keeps the embedding model/config in one place.
"""

from openai import OpenAI
from typing import List
from loguru import logger


class EmbeddingService:
    """Generate embeddings for search queries."""

    MODEL = "text-embedding-3-small"
    DIMENSION = 1536
    COST_PER_M = 0.02  # USD per 1M tokens

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.total_tokens = 0

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string. Returns 1536-dim vector."""
        response = self.client.embeddings.create(
            input=text,
            model=self.MODEL,
        )
        self.total_tokens += response.usage.total_tokens
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in one API call."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.MODEL,
        )
        self.total_tokens += response.usage.total_tokens
        return [d.embedding for d in response.data]

    @property
    def cost(self) -> float:
        return (self.total_tokens / 1_000_000) * self.COST_PER_M
