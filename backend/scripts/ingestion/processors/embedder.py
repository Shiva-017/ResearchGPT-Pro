# backend/scripts/ingestion/processors/embedder.py (FAULT-TOLERANT VERSION)

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError
from typing import List, Dict
from tqdm import tqdm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import time

class FaultTolerantEmbedder:
    def __init__(self, api_key: str, state_manager):
        """
        Args:
            api_key: OpenAI API key
            state_manager: IngestionState instance
        """
        # Initialize OpenAI client - explicitly set api_key to avoid proxy issues
        self.client = OpenAI(
            api_key=api_key,
            # Explicitly disable proxies if not needed
        )
        self.model = "text-embedding-3-small"
        self.state = state_manager
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _generate_embedding_with_retry(self, text: str) -> List[float]:
        """Generate single embedding with retry logic"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        self.total_tokens += response.usage.total_tokens
        
        return response.data[0].embedding
    
    def embed_papers(
        self,
        papers: List[Dict],
        batch_size: int = 100,
        save_every: int = 100
    ) -> List[Dict]:
        """
        Generate embeddings with fault tolerance
        
        Args:
            papers: List of papers
            batch_size: Papers per API call
            save_every: Save checkpoint every N papers
        
        Returns:
            Papers with embeddings
        """
        # Filter out papers already embedded
        embedded_ids = self.state.get_embedded_ids()
        papers_to_embed = [
            p for p in papers 
            if p['id'] not in embedded_ids
        ]
        
        logger.info(f"Already embedded: {len(embedded_ids)}")
        logger.info(f"Need to embed: {len(papers_to_embed)}")
        
        if not papers_to_embed:
            logger.info("All papers already embedded!")
            return self._load_embeddings_from_cache(papers)
        
        papers_with_embeddings = []
        embedded_count = 0
        
        for i in tqdm(range(0, len(papers_to_embed), batch_size), desc="Embedding"):
            batch = papers_to_embed[i:i+batch_size]
            
            try:
                # Generate embeddings for batch
                texts = [f"{p['title']}\n\n{p['abstract']}" for p in batch]
                
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                # Track cost
                self.total_tokens += response.usage.total_tokens
                
                # Add embeddings to papers AND save to disk
                batch_ids = []
                for paper, emb_data in zip(batch, response.data):
                    paper['embedding'] = emb_data.embedding
                    
                    # CRITICAL: Save embedding to disk immediately!
                    self.state.save_embedding(paper['id'], emb_data.embedding)
                    
                    batch_ids.append(paper['id'])
                    papers_with_embeddings.append(paper)
                    embedded_count += 1
                
                # Mark as embedded in state
                self.state.mark_embedded(batch_ids)
                
                # Checkpoint every N papers
                if embedded_count % save_every == 0:
                    logger.info(f"Checkpoint: {embedded_count} papers embedded")
                
            except (APIError, RateLimitError, APIConnectionError) as e:
                logger.error(f"OpenAI API error in batch embedding: {e}")
                
                # Try one-by-one for failed batch with exponential backoff
                for paper in batch:
                    try:
                        embedding = self._generate_embedding_with_retry(
                            f"{paper['title']}\n\n{paper['abstract']}"
                        )
                        paper['embedding'] = embedding
                        self.state.save_embedding(paper['id'], embedding)
                        self.state.mark_embedded([paper['id']])
                        papers_with_embeddings.append(paper)
                        embedded_count += 1
                        
                    except (APIError, RateLimitError, APIConnectionError) as e2:
                        logger.error(f"Failed to embed {paper['id']} after retries: {e2}")
                        self.state.mark_failed(paper['id'], str(e2))
                    except Exception as e2:
                        logger.error(f"Unexpected error embedding {paper['id']}: {e2}")
                        self.state.mark_failed(paper['id'], f"Unexpected error: {str(e2)}")
            except Exception as e:
                logger.error(f"Unexpected error in batch embedding: {e}")
                # Mark entire batch as failed
                for paper in batch:
                    self.state.mark_failed(paper['id'], f"Batch error: {str(e)}")
        
        # Calculate final cost
        self.total_cost = (self.total_tokens / 1_000_000) * 0.02
        
        logger.info(f"Embedded {embedded_count} new papers")
        logger.info(f"Cost: ${self.total_cost:.4f}")
        
        # Load ALL embeddings (including previously cached)
        all_papers = self._load_embeddings_from_cache(papers)
        
        return all_papers
    
    def _load_embeddings_from_cache(self, papers: List[Dict]) -> List[Dict]:
        """
        Load embeddings from disk cache
        
        Args:
            papers: Papers without embeddings
        
        Returns:
            Papers with embeddings loaded from cache
        """
        logger.info("Loading embeddings from cache...")
        
        papers_with_embeddings = []
        
        for paper in papers:
            # Try to load from cache
            embedding = self.state.load_embedding(paper['id'])
            
            if embedding:
                paper['embedding'] = embedding
                papers_with_embeddings.append(paper)
            else:
                logger.warning(f"No cached embedding for {paper['id']}")
        
        logger.info(f"Loaded {len(papers_with_embeddings)} embeddings from cache")
        
        return papers_with_embeddings