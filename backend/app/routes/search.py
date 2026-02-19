# backend/app/routes/search.py

"""
Search + Chat + Research Path API endpoints.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.app.models.search import SearchRequest, SearchResponse, ChatRequest, PaperResult
from backend.app.services.search_service import SearchService
from backend.app.services.agent_router import AgentRouter
from backend.app.services.research_path import ResearchPathFinder

router = APIRouter(prefix="/api/v1", tags=["search"])

# Singletons
_search_service = None
_agent_router = None
_path_finder = None


def _get_service() -> SearchService:
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


def _get_router() -> AgentRouter:
    global _agent_router
    if _agent_router is None:
        _agent_router = AgentRouter()
    return _agent_router


def _get_path_finder() -> ResearchPathFinder:
    global _path_finder
    if _path_finder is None:
        _path_finder = ResearchPathFinder()
    return _path_finder


@router.post("/search", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    """Search for research papers (non-streaming, full response)."""
    try:
        service = _get_service()
        return service.search(request)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
def chat_stream(request: ChatRequest):
    """
    Chat endpoint with intelligent routing + streaming answer.

    LangGraph classifies the query and routes to the right agent:
      SEARCH   → full 7-stage retrieval pipeline
      GRAPH    → Neo4j author/citation queries
      TRENDING → recent high-impact papers
      PATH     → Research GPS learning path
      FOLLOWUP → uses conversation context (no new search)
      COMPARE  → compares papers from previous results
      GAPS     → research gap detection
      DIRECT   → LLM answers without retrieval

    Returns SSE:
      1. {"type": "sources", ...}  — papers, timings, query_type
      2. {"type": "token", ...}    — streamed answer tokens
      3. {"type": "done"}
    """
    try:
        agent = _get_router()

        # Run the agent graph (classify → route → execute)
        state = agent.run(request)

        def event_generator():
            # Event 1: Send results + metadata
            papers_data = state.get("papers", [])
            sources_data = {
                "type": "sources",
                "papers": papers_data,
                "timings": state.get("timings", {}),
                "hyde_abstract": state.get("hyde_abstract"),
                "total_results": len(papers_data),
                "graph_insights": state.get("graph_insights", {}),
                "query_type": state.get("query_type", "SEARCH"),
                "path_result": state.get("path_result"),
            }
            yield f"data: {json.dumps(sources_data)}\n\n"

            # Events 2+: Stream answer tokens
            for token in agent.stream_answer(state):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research-path")
def find_research_path(
    start_topic: str,
    end_topic: str,
    num_steps: int = 4,
):
    """Find the optimal learning path between two research topics."""
    try:
        finder = _get_path_finder()
        return finder.find_path(
            start_topic=start_topic,
            end_topic=end_topic,
            num_steps=num_steps,
        )
    except Exception as e:
        logger.error(f"Research path failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "ResearchGPT-Pro"}
