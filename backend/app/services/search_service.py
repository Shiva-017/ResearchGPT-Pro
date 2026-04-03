# backend/app/services/search_service.py

"""
Search service with:
  - HyDE (Hypothetical Document Embedding) for query expansion
  - Hybrid Pinecone search (dense + BM25 sparse)
  - Cross-encoder reranking for precision
  - Chunk → paper grouping

Pipeline:
  Query → HyDE → Embed → Pinecone (hybrid) → Rerank → Group → Return
"""

import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from openai import OpenAI
from loguru import logger

from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.hybrid_rag import HybridRAG
from backend.app.models.search import (
    SearchRequest,
    SearchResponse,
    PaperResult,
    ChunkResult,
    ChatRequest,
    ChatMessage,
)


# ── HyDE Prompt ──────────────────────────────────────────────────────

_HYDE_SYSTEM = (
    "You are a research paper abstract generator. "
    "Given a research question or topic, write a short hypothetical "
    "abstract (3-5 sentences) of a paper that would perfectly answer it. "
    "Be specific and technical. Do not include a title. "
    "Do not say 'this paper' — just describe the work directly."
)

_ANSWER_SYSTEM = (
    "You are ResearchGPT, an expert AI research assistant with access to "
    "25,000+ CS papers and a knowledge graph of citations, authors, and research lineages. "
    "Rules:\n"
    "- Answer the user's question directly and clearly\n"
    "- Cite papers using [1], [2], etc. matching the order provided\n"
    "- Be specific: mention method names, results, and key findings\n"
    "- If the context includes citation relationships, mention them "
    "(e.g., 'Paper [2] builds on [1]' or 'This was followed up by...')\n"
    "- If foundational/seminal papers are listed, mention them as important context\n"
    "- If key researchers are listed, mention who is driving the field\n"
    "- If related papers from the citation graph are shown, use them to give a broader picture\n"
    "- If papers disagree, note the different perspectives\n"
    "- Use markdown: **bold** for key terms, bullet points for lists\n"
    "- Keep the answer focused and under 350 words"
)

_ANSWER_USER = (
    "Question: {query}\n\n"
    "Retrieved Papers:\n{context}\n\n"
    "Synthesize an answer from these papers. Cite using [1], [2], etc."
)

_HYDE_USER = (
    "Write a hypothetical abstract for a paper that answers this question:\n\n"
    "{query}"
)

_REWRITE_SYSTEM = (
    "You rewrite follow-up questions into standalone search queries. "
    "The user is having a conversation about research papers. "
    "Their latest message may reference previous context (e.g., 'those papers', "
    "'the same topic', 'who are the top authors', 'start from basics'). "
    "Rewrite it into a complete, specific search query that includes the topic.\n\n"
    "Rules:\n"
    "- Output ONLY the rewritten query, nothing else\n"
    "- Keep it under 20 words\n"
    "- Include the specific topic from conversation context\n"
    "- If the message is already self-contained, return it unchanged"
)

_REWRITE_USER = (
    "Conversation so far:\n{history_text}\n\n"
    "Latest message: {message}\n\n"
    "Rewritten search query:"
)


# ── Cohere Reranker (API-based, no local model needed) ───────────────

import cohere

_cohere_client = None


def _get_cohere_client():
    """Lazy-init Cohere client."""
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
        logger.info("Cohere rerank client initialized.")
    return _cohere_client


# ── Search Service ───────────────────────────────────────────────────

class SearchService:
    """
    Full search pipeline:
      1. HyDE  — generate hypothetical abstract
      2. Embed — embed original query + HyDE abstract
      3. Search — hybrid Pinecone query
      4. Rerank — cross-encoder reranking
      5. Group — merge chunks back into papers
    """

    def __init__(self):
        self.llm = OpenAI(api_key=settings.openai_api_key)
        self.embedder = EmbeddingService(api_key=settings.openai_api_key)
        self.pinecone = PineconeClient(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=1536,
        )
        self.rag = HybridRAG()
        logger.info(f"SearchService initialized (graph: {'ON' if self.rag.graph_available else 'OFF'}).")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, request: SearchRequest) -> SearchResponse:
        """Run the full search pipeline."""
        timings: Dict[str, float] = {}
        hyde_abstract: Optional[str] = None
        t_start = time.time()

        # ── Stage 1: HyDE ────────────────────────────────────────────
        if request.use_hyde:
            t0 = time.time()
            hyde_abstract = self._generate_hyde(request.query)
            timings["hyde_ms"] = (time.time() - t0) * 1000
            logger.info(f"HyDE generated ({timings['hyde_ms']:.0f}ms)")

        # ── Stage 2: Embed ───────────────────────────────────────────
        t0 = time.time()
        query_vector, hyde_vector = self._embed_query(request.query, hyde_abstract)
        timings["embed_ms"] = (time.time() - t0) * 1000

        # Combine: average the original query vector and HyDE vector
        # This keeps keyword grounding from the original query
        if hyde_vector:
            search_vector = [
                (q * 0.4 + h * 0.6) for q, h in zip(query_vector, hyde_vector)
            ]
        else:
            search_vector = query_vector

        # ── Stage 3: Pinecone search ─────────────────────────────────
        t0 = time.time()
        # Fetch more candidates than needed so reranker has room to work
        fetch_k = request.top_k * 4 if request.use_rerank else request.top_k * 2
        fetch_k = min(fetch_k, 80)  # cap to control Pinecone latency

        pinecone_filter = self._build_filter(request)

        results = self.pinecone.query(
            vector=search_vector,
            top_k=fetch_k,
            filter=pinecone_filter,
            query_text=request.query,   # for BM25 sparse
            alpha=request.alpha,
        )
        timings["pinecone_ms"] = (time.time() - t0) * 1000

        matches = results.matches if hasattr(results, "matches") else []
        logger.info(
            f"Pinecone returned {len(matches)} chunks ({timings['pinecone_ms']:.0f}ms)"
        )

        if not matches:
            return SearchResponse(
                query=request.query,
                hyde_abstract=hyde_abstract,
                total_results=0,
                papers=[],
                timings=timings,
                tokens_used=self.embedder.total_tokens,
                estimated_cost=self.embedder.cost,
            )

        # ── Stage 4: Cross-encoder rerank ────────────────────────────
        if request.use_rerank and len(matches) > 1:
            t0 = time.time()
            matches = self._rerank(request.query, matches)
            timings["rerank_ms"] = (time.time() - t0) * 1000
            logger.info(f"Reranked {len(matches)} chunks ({timings['rerank_ms']:.0f}ms)")

        # ── Stage 5: Group chunks → papers ───────────────────────────
        t0 = time.time()
        papers = self._group_chunks_to_papers(matches, top_k=request.top_k)
        timings["group_ms"] = (time.time() - t0) * 1000

        # ── Stage 5b: Parent-child chunk expansion ───────────────────
        # The initial search may only find the abstract chunk of a paper.
        # This step fetches additional chunks from the same papers
        # (methods, experiments, etc.) so the LLM sees actual detail.
        t0 = time.time()
        papers = self._expand_paper_chunks(papers, search_vector)
        timings["expand_ms"] = (time.time() - t0) * 1000

        # ── Stage 6: Graph enrichment ────────────────────────────
        t0 = time.time()
        enrichment = self.rag.enrich(papers, request.query)
        graph_timings = enrichment.get("timings", {})
        timings.update(graph_timings)

        # ── Stage 7: Generate answer with enriched context ─────────
        answer = None
        if papers:
            t0 = time.time()
            answer = self._generate_answer_from_context(
                request.query, enrichment["context_str"]
            )
            timings["answer_ms"] = (time.time() - t0) * 1000
            logger.info(f"Answer generated ({timings['answer_ms']:.0f}ms)")

        timings["total_ms"] = (time.time() - t_start) * 1000

        logger.info(
            f"Search complete: {len(papers)} papers in {timings['total_ms']:.0f}ms"
        )

        return SearchResponse(
            query=request.query,
            answer=answer,
            hyde_abstract=hyde_abstract,
            total_results=len(papers),
            papers=papers,
            timings=timings,
            tokens_used=self.embedder.total_tokens,
            estimated_cost=self.embedder.cost,
        )

    # ------------------------------------------------------------------
    # Stage 1: HyDE
    # ------------------------------------------------------------------

    def _generate_hyde(self, query: str) -> str:
        """
        Generate a hypothetical abstract that would answer the query.
        Uses GPT-4o-mini for speed and cost (~$0.0001 per call).
        """
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM},
                    {"role": "user", "content": _HYDE_USER.format(query=query)},
                ],
                max_tokens=200,  # short abstract only
                temperature=0.7,  # slight variation for diversity
            )
            abstract = response.choices[0].message.content.strip()
            logger.debug(f"HyDE abstract: {abstract[:100]}…")
            return abstract

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e} — falling back to raw query")
            return None

    # ------------------------------------------------------------------
    # Stage 6: Answer generation
    # ------------------------------------------------------------------

    def _generate_answer_from_context(
        self,
        query: str,
        context: str,
    ) -> Optional[str]:
        """
        Generate a synthesized answer from enriched context.
        Context includes graph data (citations, related papers, experts).
        """
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _ANSWER_SYSTEM},
                    {"role": "user", "content": _ANSWER_USER.format(
                        query=query, context=context
                    )},
                ],
                max_tokens=600,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Stage 2: Embed
    # ------------------------------------------------------------------

    def _embed_query(
        self,
        query: str,
        hyde_abstract: Optional[str],
    ) -> Tuple[List[float], Optional[List[float]]]:
        """
        Embed the original query and optionally the HyDE abstract.
        Returns (query_vector, hyde_vector or None).
        """
        if hyde_abstract:
            vectors = self.embedder.embed_batch([query, hyde_abstract])
            return vectors[0], vectors[1]
        else:
            return self.embedder.embed_text(query), None

    # ------------------------------------------------------------------
    # Stage 3: Pinecone filter
    # ------------------------------------------------------------------

    def _build_filter(self, request: SearchRequest) -> Optional[Dict]:
        """Build Pinecone metadata filter from request params."""
        conditions = []

        if request.year_min is not None:
            conditions.append({"year": {"$gte": request.year_min}})
        if request.year_max is not None:
            conditions.append({"year": {"$lte": request.year_max}})
        # chunk_type filter deprecated in v2 (single enriched chunk)
        if getattr(request, 'chunk_type', None):
            conditions.append({"chunk_type": {"$eq": request.chunk_type}})
        if request.categories:
            # Match if primary_category is in the list
            conditions.append(
                {"primary_category": {"$in": request.categories}}
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ------------------------------------------------------------------
    # Stage 4: Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, matches: list) -> list:
        """
        Re-score Pinecone results using Cohere Rerank API.

        Cohere reads (query, document) pairs and returns precise
        relevance scores — same idea as a cross-encoder but hosted,
        so zero RAM / no PyTorch needed locally.

        Free tier: 1,000 rerank calls/month.
        """
        co = _get_cohere_client()

        # Build document texts for Cohere
        # Use chunk_snippet (fulltext section) if available, otherwise title+abstract
        docs = []
        for match in matches:
            meta = match.metadata if hasattr(match, "metadata") else {}
            snippet = meta.get("chunk_snippet", "")
            if snippet:
                doc_text = snippet
            else:
                doc_text = (
                    f"{meta.get('title', '')}. "
                    f"{meta.get('abstract', '')}"
                )
            docs.append(doc_text)

        try:
            response = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=len(docs),
            )

            # Map Cohere results back to matches with scores
            reranked = []
            for result in response.results:
                match = matches[result.index]
                match._rerank_score = float(result.relevance_score)
                reranked.append(match)

            return reranked

        except Exception as e:
            logger.warning(f"Cohere rerank failed: {e} — using Pinecone scores")
            return matches

    # ------------------------------------------------------------------
    # Stage 5: Group chunks → papers
    # ------------------------------------------------------------------

    def _group_chunks_to_papers(
        self,
        matches: list,
        top_k: int,
    ) -> List[PaperResult]:
        """
        Group chunks back into papers.

        Multiple chunks (problem, method) from the same paper are
        merged into a single PaperResult. The paper's score is the
        best score among its chunks.
        """
        paper_map: Dict[str, Dict] = {}

        for match in matches:
            meta = match.metadata if hasattr(match, "metadata") else {}
            paper_id = meta.get("paper_id", match.id)

            chunk = ChunkResult(
                chunk_id=match.id,
                chunk_type=meta.get("chunk_type", "unknown"),
                section_heading=meta.get("section_heading", ""),
                score=float(match.score) if hasattr(match, "score") else 0.0,
                rerank_score=getattr(match, "_rerank_score", None),
                snippet=meta.get("chunk_snippet", ""),
            )

            if paper_id not in paper_map:
                paper_map[paper_id] = {
                    "paper_id": paper_id,
                    "title": meta.get("title", ""),
                    "abstract": meta.get("abstract", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", 0),
                    "categories": meta.get("categories", ""),
                    "pdf_url": meta.get("pdf_url", ""),
                    "field": meta.get("field", ""),
                    "citation_tier": meta.get("citation_tier", "low"),
                    "has_code": meta.get("has_code", False),
                    "is_survey": meta.get("is_survey", False),
                    "best_score": chunk.rerank_score or chunk.score,
                    "rerank_score": chunk.rerank_score,
                    "matched_chunks": [chunk],
                }
            else:
                paper_map[paper_id]["matched_chunks"].append(chunk)
                # Update best score
                candidate = chunk.rerank_score or chunk.score
                if candidate > paper_map[paper_id]["best_score"]:
                    paper_map[paper_id]["best_score"] = candidate
                    paper_map[paper_id]["rerank_score"] = chunk.rerank_score

        # Sort papers by best score descending
        sorted_papers = sorted(
            paper_map.values(),
            key=lambda p: p["best_score"],
            reverse=True,
        )

        # Convert to PaperResult models and trim to top_k
        return [PaperResult(**p) for p in sorted_papers[:top_k]]

    # ------------------------------------------------------------------
    # Stage 5b: Parent-child chunk expansion
    # ------------------------------------------------------------------

    def _expand_paper_chunks(
        self,
        papers: List[PaperResult],
        search_vector: List[float],
    ) -> List[PaperResult]:
        """
        For each top paper, fetch additional chunks from Pinecone
        filtered by paper_id. This ensures the LLM sees methods,
        experiments, and results — not just the abstract that
        happened to match the initial query.

        Uses the same search vector to rank which chunks from
        each paper are most relevant to the query.
        """
        if not papers:
            return papers

        expanded = []
        for paper in papers[:5]:  # only expand top 5 to keep latency low
            existing_chunk_ids = {c.chunk_id for c in paper.matched_chunks}

            # If we already have 3+ chunks, no need to expand
            if len(paper.matched_chunks) >= 3:
                expanded.append(paper)
                continue

            try:
                # Query Pinecone filtered to this paper's chunks
                results = self.pinecone.query(
                    vector=search_vector,
                    top_k=6,
                    filter={"paper_id": {"$eq": paper.paper_id}},
                    include_metadata=True,
                )

                new_chunks = []
                for match in (results.matches if hasattr(results, "matches") else []):
                    if match.id not in existing_chunk_ids:
                        meta = match.metadata if hasattr(match, "metadata") else {}
                        new_chunks.append(ChunkResult(
                            chunk_id=match.id,
                            chunk_type=meta.get("chunk_type", "unknown"),
                            section_heading=meta.get("section_heading", ""),
                            score=float(match.score) if hasattr(match, "score") else 0.0,
                            snippet=meta.get("chunk_snippet", ""),
                        ))

                # Add the best new chunks (up to 3 total per paper)
                slots = max(0, 3 - len(paper.matched_chunks))
                if new_chunks and slots > 0:
                    all_chunks = list(paper.matched_chunks) + new_chunks[:slots]
                    paper = paper.model_copy(update={"matched_chunks": all_chunks})
                    logger.debug(
                        f"Expanded {paper.paper_id}: "
                        f"{len(existing_chunk_ids)} → {len(all_chunks)} chunks"
                    )

            except Exception as e:
                logger.warning(f"Chunk expansion failed for {paper.paper_id}: {e}")

            expanded.append(paper)

        # Append remaining papers (beyond top-5) without chunk expansion
        expanded.extend(papers[5:])
        return expanded

    # ------------------------------------------------------------------
    # Chat with streaming
    # ------------------------------------------------------------------

    def _rewrite_query(
        self,
        message: str,
        history: list,
    ) -> str:
        """
        Rewrite a follow-up message into a standalone search query
        using conversation history for context.

        Example:
          History: "Tell me about GPU kernels"
          Follow-up: "who has the most cited papers?"
          Rewritten: "most cited papers about GPU kernel programming"
        """
        if not history:
            return message

        # Build history text (last 4 turns for context window)
        history_lines = []
        for msg in history[-4:]:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            history_lines.append(f"{role}: {content[:150]}")
        history_text = "\n".join(history_lines)

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": _REWRITE_USER.format(
                        history_text=history_text, message=message
                    )},
                ],
                max_tokens=50,
                temperature=0.0,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.debug(f"Query rewrite: '{message}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return message

    def chat_search(self, request: ChatRequest) -> tuple:
        """
        Run retrieval stages (HyDE → embed → Pinecone → rerank → group).
        Returns (papers, context_str, timings) without generating the answer.
        The answer is streamed separately via stream_chat_answer().
        """
        timings = {}
        t_start = time.time()

        # Stage 0: Rewrite follow-up question into a standalone search query
        search_query = request.message
        if request.history:
            t0 = time.time()
            search_query = self._rewrite_query(request.message, request.history)
            timings["rewrite_ms"] = (time.time() - t0) * 1000

        # Reuse the core search pipeline for chat retrieval
        search_req = SearchRequest(
            query=search_query,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
            use_rerank=request.use_rerank,
            alpha=request.alpha,
            year_min=request.year_min,
            year_max=request.year_max,
        )

        hyde_abstract = None

        # Stage 1: HyDE (uses rewritten query for better matching)
        if request.use_hyde:
            t0 = time.time()
            hyde_abstract = self._generate_hyde(search_query)
            timings["hyde_ms"] = (time.time() - t0) * 1000

        # Stage 2: Embed
        t0 = time.time()
        query_vector, hyde_vector = self._embed_query(search_query, hyde_abstract)
        timings["embed_ms"] = (time.time() - t0) * 1000

        if hyde_vector:
            search_vector = [(q * 0.4 + h * 0.6) for q, h in zip(query_vector, hyde_vector)]
        else:
            search_vector = query_vector

        # Stage 3: Pinecone
        t0 = time.time()
        fetch_k = min(request.top_k * 4 if request.use_rerank else request.top_k * 2, 100)
        pinecone_filter = self._build_filter(search_req)
        results = self.pinecone.query(
            vector=search_vector, top_k=fetch_k,
            filter=pinecone_filter, query_text=search_query, alpha=request.alpha,
        )
        timings["pinecone_ms"] = (time.time() - t0) * 1000
        matches = results.matches if hasattr(results, "matches") else []

        # Stage 4: Rerank
        if request.use_rerank and len(matches) > 1:
            t0 = time.time()
            matches = self._rerank(search_query, matches)
            timings["rerank_ms"] = (time.time() - t0) * 1000

        # Stage 5: Group
        papers = self._group_chunks_to_papers(matches, top_k=request.top_k)

        # Stage 5b: Parent-child expansion
        papers = self._expand_paper_chunks(papers, search_vector)

        # Stage 6: Graph enrichment (HybridRAG)
        t0 = time.time()
        enrichment = self.rag.enrich(papers, search_query)
        context_str = enrichment["context_str"]
        graph_insights = enrichment.get("graph_insights", {})
        graph_timings = enrichment.get("timings", {})
        timings.update(graph_timings)
        timings["total_ms"] = (time.time() - t_start) * 1000

        return papers, context_str, timings, hyde_abstract, graph_insights

    def stream_chat_answer(
        self,
        query: str,
        context: str,
        history: list,
    ):
        """
        Generator that yields answer tokens as they stream from GPT.
        Used with SSE (Server-Sent Events).
        """
        # Build messages
        messages = [
            {"role": "system", "content": _CHAT_SYSTEM},
        ]

        # Append conversation history (last 6 turns to limit context size)
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        # Current turn with retrieved context
        messages.append({
            "role": "user",
            "content": _ANSWER_USER.format(query=query, context=context),
        })

        try:
            stream = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\n\n_Error generating answer: {e}_"


# ── Chat system prompt ────────────────────────────────────────────

_CHAT_SYSTEM = (
    "You are ResearchGPT, an expert AI research assistant with access to "
    "25,000+ CS papers and a knowledge graph with 100K+ nodes (papers, authors, "
    "categories) and 150K+ citation relationships.\n\n"
    "Your context includes not just paper abstracts, but also:\n"
    "- Citation relationships (what each paper builds on and who cited it)\n"
    "- Related papers from the citation graph that vector search missed\n"
    "- Foundational/seminal papers that multiple results build on\n"
    "- Key researchers driving the field\n\n"
    "Rules:\n"
    "- Answer directly and conversationally\n"
    "- Cite papers using [1], [2], etc.\n"
    "- Mention citation relationships when relevant (e.g., '[2] extends [1]')\n"
    "- Reference foundational papers and key researchers when they appear in context\n"
    "- Use markdown: **bold** for key terms, bullet points for lists\n"
    "- For follow-ups, refer to papers from earlier in the conversation\n"
    "- Keep answers under 350 words unless user asks to elaborate\n"
    "- If user asks about trends, gaps, authors, or lineage, leverage the graph data"
)
