# backend/app/services/research_path.py

"""
Research GPS — find the optimal learning path between two topics.

Algorithm:
  1. Embed start topic + end topic
  2. Generate N waypoint vectors (semantic interpolation)
  3. For each waypoint, query Pinecone for candidate papers
  4. Score candidates by: semantic position + citation count + graph connectivity
  5. Validate citation links via Neo4j
  6. Build the final path: start → step1 → step2 → ... → goal
  7. LLM generates "why read this" for each step
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger

from backend.app.config import settings
from backend.app.db.pinecone_client import PineconeClient
from backend.app.db.neo4j_client import Neo4jClient
from backend.app.services.embedding_service import EmbeddingService
from openai import OpenAI


class ResearchPathFinder:
    """
    Find the optimal learning path between two research topics.
    Combines semantic interpolation (Pinecone) with citation
    validation (Neo4j) for educationally meaningful paths.
    """

    def __init__(self):
        self.llm = OpenAI(api_key=settings.openai_api_key)
        self.embedder = EmbeddingService(api_key=settings.openai_api_key)
        self.pinecone = PineconeClient(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=1536,
        )
        self.neo4j: Optional[Neo4jClient] = None

        if settings.neo4j_uri:
            try:
                self.neo4j = Neo4jClient(
                    uri=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password,
                )
            except Exception as e:
                logger.warning(f"Neo4j unavailable for paths: {e}")

        logger.info("ResearchPathFinder initialized")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def find_path(
        self,
        start_topic: str,
        end_topic: str,
        num_steps: int = 4,
        candidates_per_step: int = 10,
    ) -> Dict:
        """
        Find the optimal learning path from start_topic to end_topic.

        Returns:
            {
                "start_topic": str,
                "end_topic": str,
                "path": [
                    {
                        "step": 0,
                        "paper_id": str,
                        "title": str,
                        "year": int,
                        "abstract": str,
                        "citations": int,
                        "why_read": str,         # LLM-generated explanation
                        "position": float,       # 0.0 = start, 1.0 = end
                        "has_citation_link": bool # linked to next step via citations
                    }, ...
                ],
                "total_papers_on_path": int,
                "citation_coverage": float,  # % of steps linked by citations
                "timings": {...}
            }
        """
        timings = {}
        t_start = time.time()

        # ── Step 1: Embed both topics ────────────────────────────────
        t0 = time.time()
        vectors = self.embedder.embed_batch([
            f"research topic: {start_topic}",
            f"research topic: {end_topic}",
        ])
        start_vec = np.array(vectors[0])
        end_vec = np.array(vectors[1])
        timings["embed_ms"] = (time.time() - t0) * 1000

        # ── Step 2: Find anchor papers for start and end ─────────────
        t0 = time.time()
        start_paper = self._find_anchor_paper(start_vec.tolist(), start_topic)
        end_paper = self._find_anchor_paper(end_vec.tolist(), end_topic)
        timings["anchors_ms"] = (time.time() - t0) * 1000

        if not start_paper or not end_paper:
            return {
                "start_topic": start_topic,
                "end_topic": end_topic,
                "path": [],
                "error": "Could not find anchor papers for one or both topics",
                "timings": timings,
            }

        # ── Step 3: Generate waypoint vectors ────────────────────────
        # Linearly interpolate between start and end embeddings
        # positions: 0.0 = start, 1.0 = end
        # We already have anchors at 0.0 and 1.0, so waypoints are in between
        positions = np.linspace(0, 1, num_steps + 2)[1:-1]  # exclude endpoints
        waypoint_vecs = [
            ((1 - t) * start_vec + t * end_vec).tolist()
            for t in positions
        ]

        # ── Step 4: Search for candidates at each waypoint ───────────
        t0 = time.time()
        all_candidates = []
        seen_ids = {start_paper["paper_id"], end_paper["paper_id"]}

        for i, (wvec, pos) in enumerate(zip(waypoint_vecs, positions)):
            candidates = self._search_waypoint(
                wvec, candidates_per_step, seen_ids, pos
            )
            all_candidates.append(candidates)
            # Add to seen to avoid duplicates across steps
            for c in candidates:
                seen_ids.add(c["paper_id"])

        timings["waypoint_search_ms"] = (time.time() - t0) * 1000

        # ── Step 5: Score and select best paper per waypoint ─────────
        t0 = time.time()
        path_papers = [start_paper]

        for i, candidates in enumerate(all_candidates):
            if not candidates:
                continue

            # Score: combine semantic position accuracy + citation count
            best = self._select_best_candidate(
                candidates, path_papers[-1]["paper_id"] if path_papers else None
            )
            if best:
                path_papers.append(best)

        path_papers.append(end_paper)
        timings["selection_ms"] = (time.time() - t0) * 1000

        # ── Step 6: Validate citation links via Neo4j ────────────────
        t0 = time.time()
        path_papers = self._validate_citation_links(path_papers)
        timings["citation_check_ms"] = (time.time() - t0) * 1000

        # ── Step 7: Generate "why read this" for each step ───────────
        t0 = time.time()
        path_papers = self._generate_explanations(
            path_papers, start_topic, end_topic
        )
        timings["explain_ms"] = (time.time() - t0) * 1000

        timings["total_ms"] = (time.time() - t_start) * 1000

        # Calculate citation coverage
        links = sum(1 for p in path_papers[:-1] if p.get("has_citation_link"))
        coverage = links / max(len(path_papers) - 1, 1)

        logger.info(
            f"Path found: {len(path_papers)} steps, "
            f"{coverage:.0%} citation coverage, "
            f"{timings['total_ms']:.0f}ms"
        )

        return {
            "start_topic": start_topic,
            "end_topic": end_topic,
            "path": path_papers,
            "total_papers_on_path": len(path_papers),
            "citation_coverage": round(coverage, 2),
            "timings": timings,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _find_anchor_paper(
        self, vector: List[float], topic: str
    ) -> Optional[Dict]:
        """Find the best anchor paper for a topic (most cited relevant paper)."""
        results = self.pinecone.query(
            vector=vector,
            top_k=10,
            query_text=topic,
            alpha=0.7,
        )
        matches = results.matches if hasattr(results, "matches") else []

        if not matches:
            return None

        # Pick the most cited among top matches
        best = None
        best_score = -1
        for match in matches[:10]:
            meta = match.metadata if hasattr(match, "metadata") else {}
            citations = meta.get("citation_count", 0) or 0
            sim_score = float(match.score) if hasattr(match, "score") else 0
            # Combined score: similarity * 0.6 + citation_boost * 0.4
            citation_boost = min(citations / 100, 1.0)
            combined = sim_score * 0.6 + citation_boost * 0.4

            if combined > best_score:
                best_score = combined
                best = {
                    "paper_id": meta.get("paper_id", match.id),
                    "title": meta.get("title", ""),
                    "year": meta.get("year", 0),
                    "abstract": meta.get("abstract", "")[:300],
                    "citations": citations,
                    "authors": meta.get("authors", ""),
                    "field": meta.get("field", ""),
                    "pdf_url": meta.get("pdf_url", ""),
                    "similarity": sim_score,
                }

        return best

    def _search_waypoint(
        self,
        vector: List[float],
        top_k: int,
        seen_ids: set,
        position: float,
    ) -> List[Dict]:
        """Search Pinecone at a waypoint vector, filtering already-seen papers."""
        results = self.pinecone.query(
            vector=vector,
            top_k=top_k * 2,  # fetch extra to account for filtering
        )
        matches = results.matches if hasattr(results, "matches") else []

        candidates = []
        for match in matches:
            meta = match.metadata if hasattr(match, "metadata") else {}
            pid = meta.get("paper_id", match.id)
            if pid in seen_ids:
                continue

            candidates.append({
                "paper_id": pid,
                "title": meta.get("title", ""),
                "year": meta.get("year", 0),
                "abstract": meta.get("abstract", "")[:300],
                "citations": meta.get("citation_count", 0) or 0,
                "authors": meta.get("authors", ""),
                "field": meta.get("field", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "similarity": float(match.score) if hasattr(match, "score") else 0,
                "position": round(position, 2),
            })

            if len(candidates) >= top_k:
                break

        return candidates

    def _select_best_candidate(
        self,
        candidates: List[Dict],
        prev_paper_id: Optional[str],
    ) -> Optional[Dict]:
        """
        Select the best paper from candidates for a waypoint.
        Prefers: high citations + graph connection to previous step.
        """
        if not candidates:
            return None

        # Check graph connectivity if Neo4j available
        connected_ids = set()
        if self.neo4j and prev_paper_id:
            try:
                ctx = self.neo4j.get_paper_citations(prev_paper_id)
                refs = {r["id"] for r in ctx.get("references", []) if r.get("id")}
                cited = {r["id"] for r in ctx.get("cited_by", []) if r.get("id")}
                connected_ids = refs | cited
            except Exception:
                pass

        # Score candidates
        best = None
        best_score = -1

        for c in candidates:
            citations = c.get("citations", 0)
            sim = c.get("similarity", 0)
            citation_boost = min(citations / 50, 1.0)
            graph_bonus = 0.3 if c["paper_id"] in connected_ids else 0.0

            score = sim * 0.4 + citation_boost * 0.3 + graph_bonus + 0.01
            if score > best_score:
                best_score = score
                best = c
                best["has_citation_link"] = c["paper_id"] in connected_ids

        return best

    def _validate_citation_links(self, path: List[Dict]) -> List[Dict]:
        """Check which consecutive papers are linked by citations."""
        if not self.neo4j or len(path) < 2:
            for p in path:
                p.setdefault("has_citation_link", False)
            return path

        for i in range(len(path) - 1):
            current_id = path[i]["paper_id"]
            next_id = path[i + 1]["paper_id"]

            try:
                ctx = self.neo4j.get_paper_citations(current_id)
                all_connected = set()
                for r in ctx.get("references", []):
                    if r.get("id"):
                        all_connected.add(r["id"])
                for r in ctx.get("cited_by", []):
                    if r.get("id"):
                        all_connected.add(r["id"])

                path[i]["has_citation_link"] = next_id in all_connected
            except Exception:
                path[i]["has_citation_link"] = False

        # Last paper doesn't need a forward link
        path[-1]["has_citation_link"] = False

        return path

    def _generate_explanations(
        self,
        path: List[Dict],
        start_topic: str,
        end_topic: str,
    ) -> List[Dict]:
        """Generate 'why read this' for each paper on the path."""
        if not path:
            return path

        # Build context for a single LLM call
        papers_desc = []
        for i, p in enumerate(path):
            step_label = "START" if i == 0 else ("GOAL" if i == len(path) - 1 else f"Step {i}")
            papers_desc.append(
                f"[{step_label}] {p['title']} ({p.get('year', '?')})\n"
                f"  Abstract: {p.get('abstract', '')[:200]}"
            )

        prompt = (
            f"A researcher knows about '{start_topic}' and wants to learn '{end_topic}'.\n"
            f"Here is their learning path through research papers:\n\n"
            + "\n\n".join(papers_desc)
            + "\n\nFor each paper, write a 1-sentence explanation of WHY it's "
            f"an important stepping stone from '{start_topic}' to '{end_topic}'. "
            "Focus on what new concept or skill this paper introduces that bridges "
            "the gap. Format as:\n"
            "START: ...\nStep 1: ...\nStep 2: ...\n...\nGOAL: ...\n"
            "Keep each explanation under 25 words."
        )

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research mentor helping plan learning paths."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            text = response.choices[0].message.content.strip()

            # Parse explanations
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            explanations = []
            for line in lines:
                # Remove "START:", "Step N:", "GOAL:" prefix
                for prefix in ["START:", "GOAL:"] + [f"Step {i}:" for i in range(20)]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                explanations.append(line)

            # Assign to papers
            for i, p in enumerate(path):
                if i < len(explanations):
                    p["why_read"] = explanations[i]
                else:
                    p["why_read"] = ""

        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            for p in path:
                p["why_read"] = ""

        return path
