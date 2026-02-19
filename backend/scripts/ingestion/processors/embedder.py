# backend/scripts/ingestion/processors/embedder.py

"""
Fault-tolerant embedder with semantically enriched text construction.

Key improvements over v1:
- Title is repeated to amplify the highest-signal field
- Abstract filler phrases are stripped before embedding
- Each paper is split into TWO chunks (problem vs method/results)
  → 2 vectors per paper, stored with chunk_type metadata
  → retrieval precision improves significantly for specific queries
- Embeddings cached to disk; already-embedded papers are skipped
"""

import re
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import time


# ---------------------------------------------------------------------------
# Filler phrases that pollute embedding space (appear in almost every paper)
# ---------------------------------------------------------------------------
_FILLER_PHRASES = [
    "in this paper", "in this work", "in this study",
    "we propose", "we present", "we introduce", "we develop",
    "we show", "we demonstrate", "we evaluate",
    "extensive experiments", "experimental results show",
    "demonstrate that", "experiments demonstrate",
    "outperforms", "outperform", "state-of-the-art", "sota",
    "baselines", "strong baselines",
    "to this end", "in order to", "towards this goal",
    "to address this", "to tackle this",
    "promising results", "significant improvement",
    "novel approach", "novel method", "novel framework",
    "effective and efficient",
]

_FILLER_RE = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in _FILLER_PHRASES) + r')\b',
    flags=re.IGNORECASE
)


def _clean_abstract(text: str) -> str:
    """Strip filler phrases and normalise whitespace."""
    cleaned = _FILLER_RE.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned


def _split_abstract(abstract: str) -> Tuple[str, str]:
    """
    Split abstract into (problem_part, method_result_part).
    Heuristic: first ~1/3 of sentences = problem statement,
               remaining = method + results.
    Returns two non-empty strings.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if s.strip()]
    if len(sentences) <= 2:
        # Too short to split meaningfully
        return abstract, abstract

    split_idx = max(1, len(sentences) // 3)
    problem_part = ' '.join(sentences[:split_idx])
    method_part  = ' '.join(sentences[split_idx:])
    return problem_part, method_part


def build_embedding_chunks(paper: Dict) -> List[Dict]:
    """
    Build 1-2 enriched text chunks from a paper dict.

    Each chunk is a copy of the paper dict with two extra fields:
      - chunk_id   : "{paper_id}_problem" or "{paper_id}_method"
      - chunk_text : the string to embed
      - chunk_type : "problem" | "method"

    Why two chunks?
      - Problem chunk → hits queries about what problem a paper solves
      - Method chunk  → hits queries about how it was solved / what results
    """
    title    = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    category = paper.get('primary_category', '')
    year     = str(paper.get('year', ''))

    clean_abs = _clean_abstract(abstract)
    problem_part, method_part = _split_abstract(clean_abs)

    base = {
        **paper,
        # Remove raw embedding field if present — will be recomputed
        'embedding': None,
    }

    chunks = []

    # --- Chunk 1: Problem / motivation ---
    chunks.append({
        **base,
        'chunk_id':   f"{paper['id']}_problem",
        'chunk_type': 'problem',
        'chunk_text': (
            f"Title: {title}\n"
            f"Title: {title}\n"           # intentional repeat — boosts title weight
            f"Field: {category}\n"
            f"Year: {year}\n"
            f"Problem: {problem_part}"
        ),
    })

    # --- Chunk 2: Method + results ---
    # Only add if method part is meaningfully different from problem part
    if method_part != problem_part:
        chunks.append({
            **base,
            'chunk_id':   f"{paper['id']}_method",
            'chunk_type': 'method',
            'chunk_text': (
                f"Title: {title}\n"
                f"Title: {title}\n"
                f"Field: {category}\n"
                f"Year: {year}\n"
                f"Method and Results: {method_part}"
            ),
        })

    return chunks


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class FaultTolerantEmbedder:
    """
    Generate OpenAI embeddings for paper chunks with:
    - Disk-based caching (skip already-embedded chunks)
    - Batch API calls with exponential-backoff retry
    - Per-paper fallback to single calls on batch failure
    """

    MODEL      = "text-embedding-3-small"
    COST_PER_M = 0.02   # USD per 1M tokens

    def __init__(self, api_key: str, state_manager):
        self.client        = OpenAI(api_key=api_key)
        self.state         = state_manager
        self.total_tokens  = 0
        self.total_cost    = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def embed_papers(
        self,
        papers: List[Dict],
        batch_size: int = 100,
    ) -> List[Dict]:
        """
        Expand papers → chunks, embed each chunk, return enriched list.

        Returns a flat list of chunk dicts (may be 2x len(papers)).
        Each chunk dict contains:
          - all original paper fields
          - chunk_id, chunk_type, chunk_text
          - embedding  (List[float], 1536-dim)
        """
        # 1. Expand papers into chunks
        all_chunks: List[Dict] = []
        for paper in papers:
            all_chunks.extend(build_embedding_chunks(paper))

        logger.info(f"Papers: {len(papers)} → Chunks: {len(all_chunks)}")

        # 2. Skip already-embedded chunks
        embedded_ids = self.state.get_embedded_ids()
        to_embed   = [c for c in all_chunks if c['chunk_id'] not in embedded_ids]
        cached     = [c for c in all_chunks if c['chunk_id'] in embedded_ids]

        logger.info(f"Already embedded: {len(cached)} | Need to embed: {len(to_embed)}")

        # 3. Embed pending chunks
        newly_embedded = self._embed_chunks(to_embed, batch_size)

        # 4. Load cached embeddings from disk
        for chunk in cached:
            emb = self.state.load_embedding(chunk['chunk_id'])
            if emb:
                chunk['embedding'] = emb
            else:
                # Cache miss — re-queue (shouldn't happen normally)
                logger.warning(f"Cache miss for {chunk['chunk_id']}, re-embedding")
                newly_embedded.extend(self._embed_chunks([chunk], batch_size))

        # 5. Compute cost
        self.total_cost = (self.total_tokens / 1_000_000) * self.COST_PER_M

        all_results = newly_embedded + [c for c in cached if c.get('embedding')]
        logger.info(
            f"Embedding complete — {len(all_results)} chunks | "
            f"tokens: {self.total_tokens:,} | cost: ${self.total_cost:.4f}"
        )
        return all_results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_chunks(
        self,
        chunks: List[Dict],
        batch_size: int,
    ) -> List[Dict]:
        """Embed a list of chunks in batches, returns chunks with embeddings."""
        embedded: List[Dict] = []

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i: i + batch_size]
            texts = [c['chunk_text'] for c in batch]

            try:
                batch_embedded = self._call_api_batch(texts, batch)
                embedded.extend(batch_embedded)

            except (APIError, RateLimitError, APIConnectionError) as e:
                logger.error(f"Batch API error at index {i}: {e} — falling back to single")
                for chunk in batch:
                    result = self._embed_single_with_retry(chunk)
                    if result:
                        embedded.append(result)

            except Exception as e:
                logger.error(f"Unexpected batch error at index {i}: {e}")
                for chunk in batch:
                    self.state.mark_failed(chunk['chunk_id'], f"Batch error: {e}")

        return embedded

    def _call_api_batch(self, texts: List[str], chunks: List[Dict]) -> List[Dict]:
        """Single batched API call, saves each embedding to disk."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.MODEL,
        )
        self.total_tokens += response.usage.total_tokens

        results = []
        chunk_ids = []
        for chunk, emb_data in zip(chunks, response.data):
            chunk['embedding'] = emb_data.embedding
            self.state.save_embedding(chunk['chunk_id'], emb_data.embedding)
            results.append(chunk)
            chunk_ids.append(chunk['chunk_id'])

        self.state.mark_embedded(chunk_ids)
        return results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _embed_single_with_retry(self, chunk: Dict) -> Optional[Dict]:
        """Embed a single chunk with retry; returns None on permanent failure."""
        try:
            response = self.client.embeddings.create(
                input=chunk['chunk_text'],
                model=self.MODEL,
            )
            self.total_tokens += response.usage.total_tokens
            chunk['embedding'] = response.data[0].embedding
            self.state.save_embedding(chunk['chunk_id'], chunk['embedding'])
            self.state.mark_embedded([chunk['chunk_id']])
            return chunk
        except Exception as e:
            logger.error(f"Failed to embed {chunk['chunk_id']}: {e}")
            self.state.mark_failed(chunk['chunk_id'], str(e))
            return None
