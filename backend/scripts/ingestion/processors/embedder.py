# backend/scripts/ingestion/processors/embedder.py

"""
Production-grade embedder v2 — single enriched chunk per paper.

Why v2?
────────
v1 split each abstract into problem/method halves. Research shows this
hurts retrieval for short documents like abstracts (150-300 words):
  - Heuristic 1/3 split is brittle and often breaks mid-thought
  - Creates artificially tiny chunks (~50-150 tokens) that lose context
  - 2x vectors in Pinecone = 2x cost, 2x noise for the reranker

v2 strategy (backed by NVIDIA 2024 benchmark + Anthropic contextual retrieval):
  - ONE enriched chunk per paper — full abstract stays coherent
  - Structured metadata prefix — contextualizes the chunk for both
    dense (semantic) and sparse (BM25) retrieval
  - Automated keyword extraction — boosts exact-match BM25 queries
  - Filler phrase removal — kept from v1 (proven effective)
  - 50% fewer vectors, better coherence, lower embedding cost

Chunk format:
  [{field} | {year}] {title}
  Authors: {top authors}
  Keywords: {extracted keywords}
  {cleaned abstract}
"""

import re
from collections import Counter
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from typing import List, Dict, Optional, Set
from tqdm import tqdm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import time


# ---------------------------------------------------------------------------
# Filler phrases that pollute embedding space
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
    flags=re.IGNORECASE,
)

# Common stopwords to exclude from keyword extraction
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "we", "our", "they", "their", "them",
    "which", "what", "who", "whom", "also", "using", "based", "two",
    "one", "first", "new", "show", "use", "approach", "method", "methods",
    "results", "paper", "work", "model", "models", "data", "problem",
    "proposed", "existing", "however", "can", "well", "many", "set",
    "given", "different", "respectively", "across", "per", "via",
}

# Phrases that indicate methodology — boost these as keywords
_METHOD_INDICATORS = {
    "transformer", "attention", "convolution", "cnn", "rnn", "lstm", "gru",
    "bert", "gpt", "diffusion", "gan", "vae", "reinforcement learning",
    "contrastive learning", "self-supervised", "semi-supervised",
    "federated", "graph neural", "knowledge graph", "knowledge distillation",
    "fine-tuning", "pre-training", "zero-shot", "few-shot", "prompt",
    "retrieval", "augmented", "generation", "embedding", "encoder",
    "decoder", "segmentation", "detection", "classification", "regression",
    "optimization", "gradient", "backpropagation", "normalization",
    "regularization", "dropout", "pruning", "quantization", "distillation",
    "meta-learning", "multi-task", "transfer learning", "domain adaptation",
    "adversarial", "robustness", "fairness", "interpretability",
    "explainability", "causal", "bayesian", "variational",
}


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def _clean_abstract(text: str) -> str:
    """Strip filler phrases and normalise whitespace."""
    cleaned = _FILLER_RE.sub('', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned


def _extract_keywords(title: str, abstract: str, top_k: int = 8) -> List[str]:
    """
    Extract key terms from title + abstract using frequency analysis
    plus domain-aware boosting.

    Returns up to top_k keywords sorted by relevance.
    """
    text = f"{title} {abstract}".lower()

    # Check for multi-word method indicators first
    found_methods = []
    for method in _METHOD_INDICATORS:
        if method in text:
            found_methods.append(method)

    # Tokenise into words, filter stopwords
    words = re.findall(r'\b[a-z][a-z\-]{2,}\b', text)
    words = [w for w in words if w not in _STOPWORDS and len(w) > 2]

    # Count frequency
    freq = Counter(words)

    # Boost words that appear in the title (title words are high signal)
    title_words = set(re.findall(r'\b[a-z][a-z\-]{2,}\b', title.lower()))
    title_words -= _STOPWORDS
    for tw in title_words:
        if tw in freq:
            freq[tw] *= 3

    # Boost method indicators
    for method in found_methods:
        single_word = method.replace(" ", "-")
        freq[single_word] = freq.get(single_word, 0) + 5

    # Get top keywords, prefer multi-word methods first
    keywords = []
    for method in found_methods[:4]:
        keywords.append(method)

    for word, _ in freq.most_common(top_k * 2):
        if word not in ' '.join(keywords) and len(keywords) < top_k:
            keywords.append(word)

    return keywords[:top_k]


def _format_field(category: str) -> str:
    """Convert arXiv category prefix to readable field name."""
    prefix = category.split(".")[0] if "." in category else category
    return {
        "cs":      "Computer Science",
        "math":    "Mathematics",
        "physics": "Physics",
        "stat":    "Statistics",
        "eess":    "Electrical Engineering",
        "q-bio":   "Quantitative Biology",
        "q-fin":   "Quantitative Finance",
        "econ":    "Economics",
    }.get(prefix, "Other")


# ---------------------------------------------------------------------------
# Chunk builder — single enriched chunk per paper
# ---------------------------------------------------------------------------

def build_embedding_chunks(paper: Dict) -> List[Dict]:
    """
    Build ONE enriched chunk from a paper dict.

    Returns a list with a single chunk dict containing:
      - chunk_id   : "{paper_id}"  (no _problem/_method suffix)
      - chunk_text : the enriched string to embed
      - chunk_type : "full"

    Chunk text format:
      [{field} | {year}] {title}
      Authors: {top 3 authors}
      Keywords: {extracted keywords}
      {cleaned abstract}
    """
    title    = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    categories = paper.get('categories', [])
    primary_cat = categories[0] if categories else paper.get('primary_category', '')
    year     = str(paper.get('year', ''))
    authors  = paper.get('authors', [])

    # Clean the abstract
    clean_abs = _clean_abstract(abstract)

    # Extract keywords
    keywords = _extract_keywords(title, clean_abs)

    # Format the chunk
    field = _format_field(primary_cat)
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" (+{len(authors) - 3} more)"
    keyword_str = ", ".join(keywords) if keywords else ""

    # Build enriched chunk text
    parts = [
        f"[{field} | {year}] {title}",
    ]
    if author_str:
        parts.append(f"Authors: {author_str}")
    if keyword_str:
        parts.append(f"Keywords: {keyword_str}")
    parts.append(clean_abs)

    chunk_text = "\n".join(parts)

    base = {
        **paper,
        'embedding': None,
    }

    return [{
        **base,
        'chunk_id':   paper['id'],          # e.g. "arxiv:2504.12345"
        'chunk_type': 'full',
        'chunk_text': chunk_text,
    }]


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

        Returns a flat list of chunk dicts (1 per paper in v2).
        Each chunk dict contains:
          - all original paper fields
          - chunk_id, chunk_type, chunk_text
          - embedding  (List[float], 1536-dim)
        """
        # 1. Expand papers into chunks (1 per paper in v2)
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
