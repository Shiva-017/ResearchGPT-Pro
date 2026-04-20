# backend/app/services/hybrid_rag.py

"""
HybridRAG — combines Pinecone vector search with Neo4j graph intelligence.

This is the brain of ResearchGPT Pro. It takes Pinecone search results
and enriches them with graph context before sending to the LLM.

What it adds over pure vector search:
  1. Citation context   — what each paper cites and is cited by
  2. Graph expansion    — finds related papers Pinecone missed
  3. Common ancestors   — seminal papers the results all build on
  4. Author expertise   — who are the key researchers
  5. Research lineage   — the evolution chain of ideas
  6. Trend detection    — what's gaining traction recently
  7. Research gaps      — underexplored directions
  8. Reading lists      — curated paths from foundational to recent
"""

import time
from typing import List, Dict, Optional
from loguru import logger

from backend.app.config import settings
from backend.app.db.neo4j_client import Neo4jClient
from backend.app.models.search import PaperResult


class HybridRAG:
    """
    Orchestrates Pinecone results + Neo4j graph enrichment.

    Usage:
        rag = HybridRAG()
        enriched_context = rag.enrich(papers, query)
        # enriched_context is a string ready to inject into LLM prompt
    """

    def __init__(self):
        self.neo4j: Optional[Neo4jClient] = None
        self._connect()

    def _connect(self):
        """Try to connect to Neo4j. If unavailable, graph features are disabled."""
        if not settings.neo4j_uri:
            logger.warning("Neo4j URI not set — graph features disabled")
            return
        try:
            self.neo4j = Neo4jClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
            )
            logger.info("HybridRAG: Neo4j connection established")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e} — graph features disabled")
            self.neo4j = None

    @property
    def graph_available(self) -> bool:
        return self.neo4j is not None

    # ------------------------------------------------------------------
    # Main enrichment pipeline
    # ------------------------------------------------------------------

    def enrich(
        self,
        papers: List[PaperResult],
        query: str,
    ) -> Dict:
        """
        Take Pinecone results and enrich with graph intelligence.

        Returns a dict with:
          - context_str:      enriched text for LLM (replaces plain abstracts)
          - graph_insights:   structured graph data for frontend
          - timings:          performance metrics
        """
        if not self.graph_available or not papers:  # fast path: skip graph calls
            return {
                "context_str": self._build_basic_context(papers),
                "graph_insights": {},
                "timings": {},
            }

        timings = {}
        paper_ids = [p.paper_id for p in papers]

        # ── 1. Paper context (citations, references, author work) ────
        t0 = time.time()
        paper_contexts = {}
        for pid in paper_ids:
            try:
                ctx = self.neo4j.get_paper_context(pid)
                if ctx:
                    paper_contexts[pid] = ctx
            except Exception as e:
                logger.debug(f"Paper context failed for {pid}: {e}")
        timings["paper_context_ms"] = (time.time() - t0) * 1000

        # ── 2. Graph expansion (related papers Pinecone missed) ──────
        t0 = time.time()
        try:
            connected = self.neo4j.get_connected_papers(paper_ids, max_extra=5)
        except Exception:
            connected = []
        timings["graph_expand_ms"] = (time.time() - t0) * 1000

        # ── 3. Common ancestors (seminal papers) ─────────────────────
        t0 = time.time()
        try:
            ancestors = self.neo4j.get_common_ancestor(paper_ids, limit=4)  # top seminal papers
        except Exception:
            ancestors = []
        timings["ancestors_ms"] = (time.time() - t0) * 1000

        # ── 4. Author expertise ──────────────────────────────────────
        t0 = time.time()
        try:
            experts = self.neo4j.get_author_expertise(paper_ids, limit=5)  # top domain experts
        except Exception:
            experts = []
        timings["experts_ms"] = (time.time() - t0) * 1000

        # ── Build enriched context for LLM ───────────────────────────
        context_str = self._build_enriched_context(
            papers, paper_contexts, connected, ancestors, experts
        )

        # ── Compile graph insights for frontend ──────────────────────
        graph_insights = {
            "connected_papers": connected[:6],
            "seminal_papers": ancestors,
            "top_experts": experts[:5],
            "papers_with_context": len(paper_contexts),
        }

        total_graph_ms = sum(timings.values())
        timings["graph_total_ms"] = total_graph_ms
        logger.info(f"Graph enrichment complete: {total_graph_ms:.0f}ms")

        return {
            "context_str": context_str,
            "graph_insights": graph_insights,
            "timings": timings,
        }

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_basic_context(self, papers: List[PaperResult]) -> str:
        """Build context using actual matched chunk text, not just abstracts."""
        parts = []
        for i, paper in enumerate(papers[:5], start=1):
            entry = (
                f"[{i}] {paper.title} ({paper.year})\n"
                f"    Authors: {paper.authors[:100]}"
            )

            # Use the actual matched chunk snippets — this is the whole
            # point of fulltext indexing. The LLM sees methodology details,
            # experimental results, etc. instead of just abstracts.
            snippets_added = 0
            for chunk in paper.matched_chunks:
                if chunk.snippet and snippets_added < 3:
                    heading = f" [{chunk.section_heading}]" if chunk.section_heading else ""
                    entry += f"\n    Matched section{heading}: {chunk.snippet[:400]}"
                    snippets_added += 1

            # Fallback to abstract if no chunk snippets available
            if snippets_added == 0:
                abstract = paper.abstract[:400] + ("..." if len(paper.abstract) > 400 else "")
                entry += f"\n    Abstract: {abstract}"

            parts.append(entry)
        return "\n\n".join(parts)

    def _build_enriched_context(
        self,
        papers: List[PaperResult],
        paper_contexts: Dict,
        connected: List[Dict],
        ancestors: List[Dict],
        experts: List[Dict],
    ) -> str:
        """
        Build rich context string that gives the LLM much more
        to work with than plain abstracts.
        """
        sections = []

        # ── Retrieved papers with graph context ──────────────────────
        sections.append("=== RETRIEVED PAPERS ===")
        for i, paper in enumerate(papers[:5], 1):
            entry = (
                f"[{i}] {paper.title} ({paper.year})\n"
                f"    Authors: {paper.authors[:100]}"
            )

            # Use actual matched chunk text for richer LLM context
            snippets_added = 0
            for chunk in paper.matched_chunks:
                if chunk.snippet and snippets_added < 3:
                    heading = f" [{chunk.section_heading}]" if chunk.section_heading else ""
                    entry += f"\n    Matched section{heading}: {chunk.snippet[:350]}"
                    snippets_added += 1

            if snippets_added == 0:
                abstract = paper.abstract[:350] + ("..." if len(paper.abstract) > 350 else "")
                entry += f"\n    Abstract: {abstract}"

            # Add graph context if available
            ctx = paper_contexts.get(paper.paper_id)
            if ctx:
                citations = ctx.get("citation_count", 0)
                if citations:
                    entry += f"\n    Citations: {citations}"

                refs = ctx.get("references", [])
                if refs:
                    ref_titles = [r["title"][:60] for r in refs[:3] if r.get("title")]
                    if ref_titles:
                        entry += f"\n    Builds on: {'; '.join(ref_titles)}"

                cited_by = ctx.get("cited_by", [])
                if cited_by:
                    cb_titles = [r["title"][:60] for r in cited_by[:3] if r.get("title")]
                    if cb_titles:
                        entry += f"\n    Follow-up work: {'; '.join(cb_titles)}"

                authors_ctx = ctx.get("authors", [])
                for author_info in authors_ctx[:2]:
                    other = author_info.get("other_papers", [])
                    if other:
                        other_titles = [p["title"][:50] for p in other[:2] if p.get("title")]
                        if other_titles:
                            entry += f"\n    {author_info['name']} also wrote: {'; '.join(other_titles)}"

            sections.append(entry)

        # ── Related papers Pinecone missed ───────────────────────────
        if connected:
            sections.append("\n=== RELATED PAPERS (from citation graph) ===")
            for c in connected[:5]:
                sections.append(
                    f"- {c['title'][:70]} ({c.get('year', '?')}) "
                    f"— {c.get('citations', 0)} citations, "
                    f"connected to {c.get('shared_connections', 0)} of the above papers"
                )

        # ── Seminal papers ───────────────────────────────────────────
        if ancestors:
            sections.append("\n=== FOUNDATIONAL PAPERS (cited by multiple results) ===")
            for a in ancestors[:3]:
                sections.append(
                    f"- {a['title'][:70]} ({a.get('year', '?')}) "
                    f"— {a.get('citations', 0)} citations, "
                    f"cited by {a.get('cited_by_count', 0)} of the retrieved papers"
                )

        # ── Key researchers ──────────────────────────────────────────
        if experts:
            sections.append("\n=== KEY RESEARCHERS IN THIS AREA ===")
            expert_strs = []
            for e in experts[:5]:
                expert_strs.append(
                    f"{e['name']} ({e['papers_in_cluster']} papers, "
                    f"{e.get('total_citations', 0)} citations)"
                )
            sections.append(", ".join(expert_strs))

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Standalone graph queries (for special query types)
    # ------------------------------------------------------------------

    def get_trending(self, field: Optional[str] = None) -> str:
        """Format trending papers as LLM context."""
        if not self.graph_available:
            return ""
        try:
            trending = self.neo4j.get_trending_papers(field=field, limit=10)
            if not trending:
                return "No trending papers found."
            lines = ["=== TRENDING PAPERS (recent + high citation velocity) ==="]
            for i, p in enumerate(trending, 1):
                lines.append(
                    f"[{i}] {p['title'][:70]} ({p['year']}) "
                    f"— {p['citations']} citations"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Trending query failed: {e}")
            return ""

    def get_lineage(self, paper_id: str) -> str:
        """Format research lineage as LLM context."""
        if not self.graph_available:
            return ""
        try:
            lineage = self.neo4j.get_research_lineage(paper_id)
            lines = [f"=== RESEARCH LINEAGE FOR {paper_id} ==="]

            ancestors = lineage.get("ancestors", [])
            if ancestors:
                lines.append("\nBuilds on (predecessors):")
                for a in ancestors[:8]:
                    lines.append(
                        f"  {'  ' * (a.get('depth', 1) - 1)}← {a['title'][:60]} "
                        f"({a.get('year', '?')}, {a.get('citations', 0)} cites)"
                    )

            descendants = lineage.get("descendants", [])
            if descendants:
                lines.append("\nInfluenced (follow-up work):")
                for d in descendants[:8]:
                    lines.append(
                        f"  {'  ' * (d.get('depth', 1) - 1)}→ {d['title'][:60]} "
                        f"({d.get('year', '?')}, {d.get('citations', 0)} cites)"
                    )

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Lineage query failed: {e}")
            return ""

    def get_gaps(self, category: str) -> str:
        """Format research gaps as LLM context."""
        if not self.graph_available:
            return ""
        try:
            gaps = self.neo4j.get_research_gaps(category, limit=8)
            if not gaps:
                return f"No clear research gaps found in {category}."
            lines = [f"=== POTENTIAL RESEARCH GAPS IN {category} ===",
                     "(Highly cited older papers with little recent follow-up)"]
            for g in gaps:
                lines.append(
                    f"- {g['title'][:70]} ({g['year']}) "
                    f"— {g['citations']} citations but only "
                    f"{g.get('recent_citers', 0)} recent papers cite it"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Gaps query failed: {e}")
            return ""

    def get_field_bridge(self, cat1: str, cat2: str) -> str:
        """Format cross-field papers as LLM context."""
        if not self.graph_available:
            return ""
        try:
            bridges = self.neo4j.get_field_bridges(cat1, cat2, limit=8)
            if not bridges:
                return f"No papers bridging {cat1} and {cat2} found."
            lines = [f"=== PAPERS BRIDGING {cat1} AND {cat2} ==="]
            for b in bridges:
                lines.append(
                    f"- {b['title'][:70]} ({b.get('year', '?')}) "
                    f"— {b.get('citations', 0)} citations"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Bridge query failed: {e}")
            return ""
