# backend/app/services/agent_router.py

"""
LangGraph multi-agent router.

Classifies user queries and routes to the appropriate agent:

  SEARCH      → "What is attention?" → full 7-stage search pipeline
  GRAPH       → "Who are top NLP researchers?" → Neo4j graph query directly
  TRENDING    → "What's trending in 2024?" → Neo4j trending query
  PATH        → "Path from CNNs to transformers" → Research GPS
  FOLLOWUP    → "Tell me more about paper [2]" → use conversation context, no new search
  COMPARE     → "Compare paper [1] and [3]" → use conversation context
  GAPS        → "What's underexplored in cs.AI?" → Neo4j gap detection
  DIRECT      → "What does HyDE stand for?" → LLM answers from knowledge, no retrieval

This replaces the one-size-fits-all pipeline with intelligent routing.
"""

from __future__ import annotations

import time
import json
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from loguru import logger
from openai import OpenAI

from langgraph.graph import StateGraph, END

from backend.app.config import settings
from backend.app.services.search_service import SearchService
from backend.app.services.hybrid_rag import HybridRAG
from backend.app.services.research_path import ResearchPathFinder
from backend.app.models.search import ChatRequest, ChatMessage, PaperResult


# ── Query types ──────────────────────────────────────────────────────

QUERY_TYPES = [
    "SEARCH",     # needs vector search + graph enrichment
    "GRAPH",      # pure graph query (authors, citations, networks)
    "TRENDING",   # what's hot recently
    "PATH",       # learning path between topics
    "FOLLOWUP",   # refers to previous results, no new search
    "COMPARE",    # compare specific papers from results
    "GAPS",       # research gap detection
    "DIRECT",     # can be answered without retrieval
]


# ── Agent State ──────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State passed through the LangGraph."""
    # Input
    query: str
    history: List[Dict]
    chat_request: Dict

    # Classification
    query_type: str
    extracted_params: Dict       # e.g. {author: "X"}, {cat1: "cs.AI", cat2: "cs.CV"}

    # Results
    papers: List[Dict]
    context: str
    graph_insights: Dict
    timings: Dict
    hyde_abstract: Optional[str]

    # Path results (for PATH queries)
    path_result: Optional[Dict]


# ── Classifier Prompt ────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = """You classify research assistant queries into types. 
Reply with ONLY a JSON object, no markdown, no explanation.

Types:
- SEARCH: needs paper search (questions about topics, methods, techniques)
- GRAPH: asks about authors, citations, collaborators, "who works on X", "what cites Y"
- TRENDING: asks what's trending, hot, recent, gaining traction
- PATH: asks for a learning path, roadmap, "how to get from X to Y", "what should I read to learn X"
- FOLLOWUP: refers to previous results ("tell me more about paper [2]", "expand on that", "what about the third one")
- COMPARE: asks to compare specific papers or approaches from results ("compare [1] and [3]")
- GAPS: asks about research gaps, underexplored areas, what's missing
- DIRECT: simple factual question LLM can answer without searching ("what does RAG stand for")

Extract any relevant parameters:
- For GRAPH: {author, paper_id, field}
- For PATH: {start_topic, end_topic}
- For GAPS: {category}
- For TRENDING: {field}
- For others: {}

Respond as: {"type": "SEARCH", "params": {}}"""

_CLASSIFIER_USER = """Query: "{query}"
Conversation has {history_len} previous messages.
{history_hint}

Classify this query."""


# ── Agent Router ─────────────────────────────────────────────────────

class AgentRouter:
    """
    LangGraph-based multi-agent router.

    Usage:
        router = AgentRouter()
        result = router.run(chat_request)
        # result contains: papers, context, timings, path_result, etc.
    """

    def __init__(self):
        self.llm = OpenAI(api_key=settings.openai_api_key)
        self.search_service = SearchService()
        self.rag = self.search_service.rag
        self.path_finder: Optional[ResearchPathFinder] = None

        # Build the LangGraph
        self.graph = self._build_graph()

        logger.info("AgentRouter initialized with LangGraph")

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("classify", self._classify_node)
        builder.add_node("search_agent", self._search_agent)
        builder.add_node("graph_agent", self._graph_agent)
        builder.add_node("trending_agent", self._trending_agent)
        builder.add_node("path_agent", self._path_agent)
        builder.add_node("followup_agent", self._followup_agent)
        builder.add_node("compare_agent", self._compare_agent)
        builder.add_node("gaps_agent", self._gaps_agent)
        builder.add_node("direct_agent", self._direct_agent)

        # Entry point
        builder.set_entry_point("classify")

        # Conditional routing from classifier
        builder.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "SEARCH": "search_agent",
                "GRAPH": "graph_agent",
                "TRENDING": "trending_agent",
                "PATH": "path_agent",
                "FOLLOWUP": "followup_agent",
                "COMPARE": "compare_agent",
                "GAPS": "gaps_agent",
                "DIRECT": "direct_agent",
            },
        )

        # All agents go to END
        for node in ["search_agent", "graph_agent", "trending_agent",
                      "path_agent", "followup_agent", "compare_agent",
                      "gaps_agent", "direct_agent"]:
            builder.add_edge(node, END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, request: ChatRequest) -> AgentState:
        """Run the full agent pipeline and return the state."""
        initial_state: AgentState = {
            "query": request.message,
            "history": [{"role": m.role, "content": m.content} for m in request.history],
            "chat_request": {
                "message": request.message,
                "top_k": request.top_k,
                "use_hyde": request.use_hyde,
                "use_rerank": request.use_rerank,
                "alpha": request.alpha,
                "year_min": request.year_min,
                "year_max": request.year_max,
            },
            "query_type": "",
            "extracted_params": {},
            "papers": [],
            "context": "",
            "graph_insights": {},
            "timings": {},
            "hyde_abstract": None,
            "path_result": None,
        }

        result = self.graph.invoke(initial_state)
        return result

    def stream_answer(self, state: AgentState):
        """Stream the LLM answer given the agent's context."""
        return self.search_service.stream_chat_answer(
            query=state["query"],
            context=state["context"],
            history=[ChatMessage(**m) for m in state["history"]],
        )

    # ------------------------------------------------------------------
    # Classifier node
    # ------------------------------------------------------------------

    def _classify_node(self, state: AgentState) -> dict:
        """Classify the query type using GPT."""
        t0 = time.time()

        history_hint = ""
        if state["history"]:
            last_asst = [m for m in state["history"] if m["role"] == "assistant"]
            if last_asst:
                history_hint = f"Last assistant response started with: \"{last_asst[-1]['content'][:100]}...\""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": _CLASSIFIER_USER.format(
                        query=state["query"],
                        history_len=len(state["history"]),
                        history_hint=history_hint,
                    )},
                ],
                max_tokens=100,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)

            query_type = parsed.get("type", "SEARCH").upper()
            params = parsed.get("params", {})

            # Validate
            if query_type not in QUERY_TYPES:
                query_type = "SEARCH"

        except Exception as e:
            logger.warning(f"Classification failed: {e} — defaulting to SEARCH")
            query_type = "SEARCH"
            params = {}

        classify_ms = (time.time() - t0) * 1000
        logger.info(f"Query classified as {query_type} ({classify_ms:.0f}ms) params={params}")

        return {
            "query_type": query_type,
            "extracted_params": params,
            "timings": {**state["timings"], "classify_ms": classify_ms},
        }

    def _route_query(self, state: AgentState) -> str:
        """Route to the correct agent based on classification."""
        return state["query_type"]

    # ------------------------------------------------------------------
    # Agent nodes
    # ------------------------------------------------------------------

    def _search_agent(self, state: AgentState) -> dict:
        """Full 7-stage search pipeline."""
        request = ChatRequest(
            message=state["query"],
            history=[ChatMessage(**m) for m in state["history"]],
            **{k: v for k, v in state["chat_request"].items() if k != "message"},
        )
        papers, context, timings, hyde, graph_insights = self.search_service.chat_search(request)

        return {
            "papers": [p.model_dump() for p in papers],
            "context": context,
            "timings": {**state["timings"], **timings},
            "hyde_abstract": hyde,
            "graph_insights": graph_insights,
        }

    def _graph_agent(self, state: AgentState) -> dict:
        """Handle graph-specific queries (authors, citations, networks)."""
        t0 = time.time()
        params = state["extracted_params"]
        context_parts = []

        if not self.rag.graph_available:
            return {"context": "Graph database is not connected.", "timings": state["timings"]}

        # Author query
        author = params.get("author")
        if author:
            try:
                network = self.rag.neo4j.get_author_network(author)
                if network:
                    context_parts.append(f"=== AUTHOR: {network['name']} ===")
                    context_parts.append(f"Total papers: {network['paper_count']}")
                    if network.get("collaborators"):
                        context_parts.append("Top collaborators:")
                        for c in network["collaborators"][:10]:
                            context_parts.append(
                                f"  - {c['name']} ({c['papers_together']} papers together)"
                            )
            except Exception as e:
                logger.warning(f"Author query failed: {e}")

        # Field query — top authors
        field = params.get("field")
        if field and not author:
            try:
                top = self.rag.neo4j.get_top_authors_in_field(field, limit=10)
                if top:
                    context_parts.append(f"=== TOP RESEARCHERS IN {field} ===")
                    for i, a in enumerate(top, 1):
                        context_parts.append(
                            f"[{i}] {a['name']} — {a['papers']} papers, {a['total_citations']} citations"
                        )
            except Exception as e:
                logger.warning(f"Field authors query failed: {e}")

        # Paper citation query
        paper_id = params.get("paper_id")
        if paper_id:
            try:
                cit = self.rag.neo4j.get_paper_citations(paper_id)
                if cit:
                    context_parts.append(f"=== PAPER: {cit.get('title', paper_id)} ===")
                    context_parts.append(f"Citations: {cit.get('citation_count', 0)}")
                    refs = cit.get("references", [])
                    if refs:
                        context_parts.append(f"References ({len(refs)}):")
                        for r in refs[:5]:
                            context_parts.append(f"  - {r.get('title', r['id'])} ({r.get('year', '?')})")
                    cited = cit.get("cited_by", [])
                    if cited:
                        context_parts.append(f"Cited by ({len(cited)}):")
                        for c in cited[:5]:
                            context_parts.append(f"  - {c.get('title', c['id'])} ({c.get('year', '?')})")
            except Exception as e:
                logger.warning(f"Paper citation query failed: {e}")

        # Fallback: if no specific params, do a general field search
        if not context_parts:
            try:
                top = self.rag.neo4j.get_top_authors_in_field("Computer Science", limit=10)
                if top:
                    context_parts.append("=== TOP CS RESEARCHERS ===")
                    for i, a in enumerate(top, 1):
                        context_parts.append(
                            f"[{i}] {a['name']} — {a['papers']} papers, {a['total_citations']} citations"
                        )
            except Exception:
                context_parts.append("No graph data available for this query.")

        graph_ms = (time.time() - t0) * 1000
        return {
            "context": "\n".join(context_parts),
            "timings": {**state["timings"], "graph_agent_ms": graph_ms},
        }

    def _trending_agent(self, state: AgentState) -> dict:
        """Handle trending/hot topic queries."""
        t0 = time.time()
        params = state["extracted_params"]
        field = params.get("field")

        context = self.rag.get_trending(field=field)
        if not context:
            context = "No trending data available. The graph may not have recent papers."

        return {
            "context": context,
            "timings": {**state["timings"], "trending_ms": (time.time() - t0) * 1000},
        }

    def _path_agent(self, state: AgentState) -> dict:
        """Handle learning path queries."""
        t0 = time.time()
        params = state["extracted_params"]

        start = params.get("start_topic", "")
        end = params.get("end_topic", "")

        if not start or not end:
            # Try to infer from query
            return {
                "context": (
                    "I can find a learning path for you! Please specify:\n"
                    "- What you currently know (starting topic)\n"
                    "- What you want to learn (goal topic)\n\n"
                    "For example: 'Find a path from CNNs to GPU kernel optimization'"
                ),
                "timings": {**state["timings"], "path_ms": (time.time() - t0) * 1000},
            }

        # Lazy-load path finder
        if self.path_finder is None:
            self.path_finder = ResearchPathFinder()

        result = self.path_finder.find_path(start, end, num_steps=4)

        # Format as context for LLM
        if result.get("path"):
            lines = [f"=== LEARNING PATH: {start} → {end} ==="]
            for i, step in enumerate(result["path"]):
                label = "START" if i == 0 else ("GOAL" if i == len(result["path"]) - 1 else f"Step {i}")
                citation_link = "✓ cites next" if step.get("has_citation_link") else ""
                lines.append(
                    f"\n[{label}] {step['title']} ({step.get('year', '?')})"
                    f"\n  Citations: {step.get('citations', 0)}"
                    f"\n  Why: {step.get('why_read', 'N/A')}"
                    f"\n  {citation_link}"
                )
            lines.append(f"\nPath coverage: {result.get('citation_coverage', 0):.0%} citation-connected")
            context = "\n".join(lines)
        else:
            context = f"Could not find a path from '{start}' to '{end}'. Try broader topics."

        return {
            "context": context,
            "path_result": result,
            "timings": {**state["timings"], "path_ms": (time.time() - t0) * 1000},
        }

    def _followup_agent(self, state: AgentState) -> dict:
        """Handle follow-up questions using conversation context."""
        # No new search — just build context from the conversation history
        context_parts = ["=== PREVIOUS CONVERSATION CONTEXT ==="]
        for msg in state["history"][-4:]:
            role = msg["role"].upper()
            content = msg["content"][:500]
            context_parts.append(f"\n{role}: {content}")

        context_parts.append(f"\n\nNew question: {state['query']}")
        context_parts.append("Answer based on the papers and information discussed above.")

        return {
            "context": "\n".join(context_parts),
            "timings": {**state["timings"], "followup_ms": 0},
        }

    def _compare_agent(self, state: AgentState) -> dict:
        """Handle comparison queries using conversation context."""
        context_parts = ["=== COMPARISON REQUEST ==="]
        context_parts.append(f"User wants to compare: {state['query']}")
        context_parts.append("\n=== PREVIOUS CONTEXT ===")

        for msg in state["history"][-4:]:
            role = msg["role"].upper()
            content = msg["content"][:500]
            context_parts.append(f"\n{role}: {content}")

        context_parts.append("\nCompare the relevant papers/approaches mentioned above. "
                           "Highlight key differences in methods, results, and use cases.")

        return {
            "context": "\n".join(context_parts),
            "timings": {**state["timings"], "compare_ms": 0},
        }

    def _gaps_agent(self, state: AgentState) -> dict:
        """Handle research gap queries."""
        t0 = time.time()
        params = state["extracted_params"]
        category = params.get("category", "cs.AI")

        context = self.rag.get_gaps(category)
        if not context:
            context = f"No clear research gaps found for {category}."

        return {
            "context": context,
            "timings": {**state["timings"], "gaps_ms": (time.time() - t0) * 1000},
        }

    def _direct_agent(self, state: AgentState) -> dict:
        """Handle direct questions that don't need retrieval."""
        return {
            "context": (
                "This is a general knowledge question. "
                "Answer from your training knowledge. "
                f"Question: {state['query']}"
            ),
            "timings": {**state["timings"], "direct_ms": 0},
        }
