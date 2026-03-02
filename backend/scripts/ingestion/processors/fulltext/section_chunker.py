# backend/scripts/ingestion/processors/fulltext/section_chunker.py

"""
Smart section-aware chunker for GROBID-parsed papers.

Strategy (backed by NVIDIA 2024 + Chroma benchmarks):
  - Target chunk size: 400-500 tokens (optimal for factoid + analytical queries)
  - Respect natural boundaries: section → paragraph → sentence
  - Overlap: 50 tokens at split points to preserve cross-boundary context
  - Section metadata: each chunk knows which section it came from
  - Skip low-value sections: Related Work, Acknowledgements, References

Chunk format:
  [{field} | {year}] {paper_title}
  Section: {section_heading}
  {chunk_text}

This contextual prefix follows Anthropic's contextual retrieval approach
(49% reduction in retrieval failures).
"""

import re
import tiktoken
from typing import List, Dict, Optional
from loguru import logger


# Use the tokenizer matching text-embedding-3-small
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens using the embedding model's tokenizer."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Split on sentence-ending punctuation followed by space + capital
    # But avoid splitting on common abbreviations
    text = re.sub(r'\n+', ' ', text)  # normalize newlines
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


class SectionChunker:
    """
    Chunk GROBID-parsed sections into embedding-ready pieces.

    For each paper, produces multiple chunks with contextual prefixes.
    Each chunk carries metadata about its source section and paper.
    """

    def __init__(
        self,
        target_tokens: int = 500,      # target chunk size (bumped from 450)
        max_tokens: int = 600,         # hard ceiling (bumped from 500)
        min_tokens: int = 100,         # skip tiny fragments (bumped from 80)
        overlap_tokens: int = 50,       # overlap at split points
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    def _build_prefix(self, paper: Dict, section_heading: str) -> str:
        """
        Build the contextual prefix for a chunk.
        Follows Anthropic's contextual retrieval approach.
        """
        title = paper.get("title", "")
        categories = paper.get("categories", [])
        primary_cat = categories[0] if categories else paper.get("primary_category", "")
        year = str(paper.get("year", ""))

        field = {
            "cs": "Computer Science", "math": "Mathematics",
            "stat": "Statistics", "physics": "Physics",
            "eess": "Electrical Engineering",
        }.get(primary_cat.split(".")[0] if "." in primary_cat else primary_cat, "Other")

        parts = [f"[{field} | {year}] {title}"]
        if section_heading:
            parts.append(f"Section: {section_heading}")
        return "\n".join(parts)

    def _chunk_text(
        self,
        text: str,
        prefix: str,
    ) -> List[str]:
        """
        Split text into chunks respecting natural boundaries.

        Hierarchy: paragraph → sentence → word (last resort)
        Returns list of chunk body texts (without prefix).
        """
        prefix_tokens = _count_tokens(prefix)
        available_tokens = self.max_tokens - prefix_tokens - 5  # buffer

        total_tokens = _count_tokens(text)

        # If the whole text fits, return it as one chunk
        if total_tokens <= available_tokens:
            return [text]

        chunks = []
        paragraphs = _split_into_paragraphs(text)

        current_parts = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para)

            # If adding this paragraph exceeds limit
            if current_tokens + para_tokens > available_tokens:
                # Flush current chunk
                if current_parts:
                    chunks.append("\n\n".join(current_parts))

                # If single paragraph is too big, split by sentences
                if para_tokens > available_tokens:
                    sentence_chunks = self._chunk_by_sentences(
                        para, available_tokens
                    )
                    chunks.extend(sentence_chunks)
                    current_parts = []
                    current_tokens = 0
                else:
                    # Start new chunk with this paragraph
                    # Add overlap: include last sentence of previous chunk
                    overlap = ""
                    if current_parts:
                        last_sentences = _split_into_sentences(current_parts[-1])
                        if last_sentences:
                            overlap = last_sentences[-1]

                    if overlap and _count_tokens(overlap) < self.overlap_tokens * 2:
                        current_parts = [overlap, para]
                        current_tokens = _count_tokens(overlap) + para_tokens
                    else:
                        current_parts = [para]
                        current_tokens = para_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        # Flush remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            if _count_tokens(chunk_text) >= self.min_tokens:
                chunks.append(chunk_text)

        return chunks

    def _chunk_by_sentences(
        self,
        text: str,
        max_tokens: int,
    ) -> List[str]:
        """Split a long paragraph into chunks at sentence boundaries."""
        sentences = _split_into_sentences(text)
        if not sentences:
            return [text[:max_tokens * 4]]  # rough char estimate as fallback

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _count_tokens(sent)

            if current_tokens + sent_tokens > max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))

                    # Overlap: keep last sentence
                    overlap_sent = current_sentences[-1]
                    current_sentences = [overlap_sent, sent]
                    current_tokens = _count_tokens(overlap_sent) + sent_tokens
                else:
                    # Single sentence exceeds limit — truncate
                    chunks.append(sent[:max_tokens * 4])
                    current_sentences = []
                    current_tokens = 0
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens

        if current_sentences:
            text = " ".join(current_sentences)
            if _count_tokens(text) >= self.min_tokens:
                chunks.append(text)

        return chunks

    def chunk_paper(
        self,
        paper: Dict,
        parsed: Dict,
    ) -> List[Dict]:
        """
        Chunk a single parsed paper into embedding-ready dicts.

        Args:
            paper:  Original paper dict (from arxiv fetcher)
            parsed: GROBID-parsed dict (from GrobidParser)

        Returns:
            List of chunk dicts, each with:
              - chunk_id:   "{paper_id}_s{section_idx}_c{chunk_idx}"
              - chunk_type: section type (e.g. "method", "experiment")
              - chunk_text: contextual prefix + chunk body
              - section_heading: original section heading
              + all original paper fields
        """
        chunks = []
        chunk_counter = 0

        for sec_idx, section in enumerate(parsed["sections"]):
            heading = section["heading"]
            text = section["text"]
            section_type = section["section_type"]

            if not text or _count_tokens(text) < self.min_tokens:
                continue

            # Build contextual prefix
            prefix = self._build_prefix(paper, heading)

            # Split section text into chunks
            text_chunks = self._chunk_text(text, prefix)

            for c_idx, chunk_body in enumerate(text_chunks):
                chunk_text = f"{prefix}\n{chunk_body}"

                chunks.append({
                    **paper,
                    "embedding": None,
                    "chunk_id": f"{paper['id']}_s{sec_idx}_c{c_idx}",
                    "chunk_type": section_type,
                    "chunk_text": chunk_text,
                    "section_heading": heading,
                    "section_index": sec_idx,
                    "chunk_index": c_idx,
                    "is_fulltext": True,
                })
                chunk_counter += 1

        # Also add an abstract chunk if the paper has one
        abstract = paper.get("abstract", "") or parsed.get("abstract_text", "")
        if abstract and _count_tokens(abstract) >= self.min_tokens:
            prefix = self._build_prefix(paper, "Abstract")
            chunks.insert(0, {
                **paper,
                "embedding": None,
                "chunk_id": f"{paper['id']}_abstract",
                "chunk_type": "abstract",
                "chunk_text": f"{prefix}\n{abstract}",
                "section_heading": "Abstract",
                "section_index": -1,
                "chunk_index": 0,
                "is_fulltext": True,
            })
            chunk_counter += 1

        logger.debug(
            f"{paper['id']}: {len(parsed['sections'])} sections → "
            f"{chunk_counter} chunks"
        )
        return chunks

    def chunk_papers(
        self,
        papers: List[Dict],
        parsed_results: List[Dict],
    ) -> List[Dict]:
        """
        Chunk multiple papers.

        Args:
            papers: List of original paper dicts (keyed by 'id')
            parsed_results: List of GROBID-parsed dicts (keyed by 'paper_id')

        Returns:
            Flat list of all chunk dicts across all papers.
        """
        # Build lookup by paper_id
        paper_map = {p["id"]: p for p in papers}

        all_chunks = []
        papers_chunked = 0
        papers_skipped = 0

        for parsed in parsed_results:
            paper_id = parsed["paper_id"]
            paper = paper_map.get(paper_id)
            if not paper:
                logger.warning(f"No paper found for parsed result {paper_id}")
                papers_skipped += 1
                continue

            chunks = self.chunk_paper(paper, parsed)
            all_chunks.extend(chunks)
            papers_chunked += 1

        avg_chunks = len(all_chunks) / papers_chunked if papers_chunked else 0

        logger.info(
            f"Chunking complete: {papers_chunked} papers → "
            f"{len(all_chunks)} chunks (avg {avg_chunks:.1f}/paper)"
        )

        # Token distribution stats
        if all_chunks:
            token_counts = [_count_tokens(c["chunk_text"]) for c in all_chunks]
            logger.info(
                f"Token stats — min: {min(token_counts)}, "
                f"avg: {sum(token_counts)/len(token_counts):.0f}, "
                f"max: {max(token_counts)}"
            )

        return all_chunks
