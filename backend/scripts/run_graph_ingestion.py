# backend/scripts/run_graph_ingestion.py

"""
Graph ingestion pipeline:
  1. Load papers from arXiv raw cache (already fetched)
  2. Enrich with Semantic Scholar (citations, references)
  3. Ingest into Neo4j (papers, authors, categories, citations)
  4. Build co-authorship edges

Usage:
    python backend/scripts/run_graph_ingestion.py
    python backend/scripts/run_graph_ingestion.py --skip-enrich    # skip S2 if already done
    python backend/scripts/run_graph_ingestion.py --limit 1000     # test with fewer papers
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
import glob
import time
from loguru import logger

from backend.app.config import settings
from backend.app.db.neo4j_client import Neo4jClient
from backend.scripts.ingestion.fetchers.semantic_scholar import SemanticScholarFetcher
from backend.scripts.ingestion.processors.validator import PaperValidator
from backend.scripts.ingestion.processors.deduplicator import Deduplicator


def separator(title: str, char="=", width=60):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def load_papers_from_cache(cache_dir: str = "backend/data/raw") -> list:
    """Load all cached arXiv papers from raw JSON files."""
    papers = []
    json_files = glob.glob(os.path.join(cache_dir, "arxiv_*.json"))

    if not json_files:
        logger.error(f"No cached papers found in {cache_dir}")
        logger.info("Run the ingestion pipeline first:")
        logger.info("  python backend/scripts/run_ingestion.py --preset cs-all --papers 625")
        return []

    logger.info(f"Found {len(json_files)} cache files")

    for filepath in json_files:
        with open(filepath, "r") as f:
            batch = json.load(f)
            papers.extend(batch)
        logger.info(f"  Loaded {len(batch):,} papers from {os.path.basename(filepath)}")

    return papers


def main():
    parser = argparse.ArgumentParser(description="Graph ingestion pipeline")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip Semantic Scholar enrichment (use if already done)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of papers (for testing)")
    parser.add_argument("--skip-coauthor", action="store_true",
                        help="Skip co-authorship edge building")
    args = parser.parse_args()

    logger.add(
        "backend/data/logs/graph_ingestion_{time}.log",
        rotation="10 MB",
        level="INFO",
    )

    start = time.time()

    # ── Step 1: Load papers from cache ───────────────────────────────
    separator("STEP 1: LOADING PAPERS FROM CACHE")

    papers = load_papers_from_cache()
    if not papers:
        return

    # Validate + dedup (same as ingestion pipeline)
    validator = PaperValidator()
    papers = validator.validate_papers(papers)

    deduplicator = Deduplicator()
    papers = deduplicator.deduplicate(papers)

    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers")

    print(f"  Total unique papers: {len(papers):,}")

    # ── Step 2: Enrich with Semantic Scholar ─────────────────────────
    if not args.skip_enrich:
        separator("STEP 2: ENRICHING WITH SEMANTIC SCHOLAR")
        print(f"  Papers to enrich: {len(papers):,}")
        print(f"  Estimated time: ~{len(papers) * 3.5 / 60:.0f} minutes")
        print(f"  (Results cached — re-run is instant for already-fetched papers)")
        print()

        s2 = SemanticScholarFetcher()
        papers = s2.enrich_papers(papers)

        stats = s2.get_stats()
        print(f"\n  S2 stats:")
        print(f"    Fetched from API : {stats['fetched']:,}")
        print(f"    Loaded from cache: {stats['cached']:,}")
        print(f"    Not found on S2  : {stats['not_found']:,}")
        print(f"    Failed           : {stats['failed']:,}")

        # Show citation stats
        has_citations = sum(1 for p in papers if p.get("citation_count", 0) > 0)
        has_refs = sum(1 for p in papers if p.get("references"))
        has_cited = sum(1 for p in papers if p.get("cited_by"))
        total_refs = sum(len(p.get("references", [])) for p in papers)
        total_cited = sum(len(p.get("cited_by", [])) for p in papers)

        print(f"\n  Citation coverage:")
        print(f"    Papers with citations : {has_citations:,} / {len(papers):,}")
        print(f"    Papers with references: {has_refs:,}")
        print(f"    Papers with cited_by  : {has_cited:,}")
        print(f"    Total reference edges : {total_refs:,}")
        print(f"    Total cited_by edges  : {total_cited:,}")
    else:
        separator("STEP 2: SKIPPED (--skip-enrich)")

    # ── Step 3: Connect to Neo4j ─────────────────────────────────────
    separator("STEP 3: CONNECTING TO NEO4J")

    if not settings.neo4j_uri:
        logger.error("NEO4J_URI not set in .env!")
        logger.info("Add these to your .env file:")
        logger.info("  NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io")
        logger.info("  NEO4J_USERNAME=neo4j")
        logger.info("  NEO4J_PASSWORD=your-password")
        return

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    print("  ✅ Connected to Neo4j")

    # Create indexes
    neo4j.create_indexes()
    print("  ✅ Indexes created")

    # ── Step 4: Ingest into graph ────────────────────────────────────
    separator("STEP 4: INGESTING INTO NEO4J GRAPH")
    print(f"  Papers to ingest: {len(papers):,}")

    neo4j.ingest_papers_batch(papers, batch_size=500)
    print("  ✅ Papers, authors, categories, and citations ingested")

    # ── Step 5: Build co-authorship ──────────────────────────────────
    if not args.skip_coauthor:
        separator("STEP 5: BUILDING CO-AUTHORSHIP GRAPH")
        neo4j.build_coauthorship()
        print("  ✅ Co-authorship edges built")
    else:
        separator("STEP 5: SKIPPED (--skip-coauthor)")

    # ── Step 6: Stats ────────────────────────────────────────────────
    separator("STEP 6: GRAPH STATISTICS")
    stats = neo4j.get_graph_stats()
    print(f"  Paper nodes       : {stats.get('papers', 0):,}")
    print(f"  Author nodes      : {stats.get('authors', 0):,}")
    print(f"  Category nodes    : {stats.get('categories', 0):,}")
    print(f"  CITES edges       : {stats.get('citation_edges', 0):,}")
    print(f"  COAUTHORED edges  : {stats.get('coauthor_edges', 0):,}")

    # Show top cited papers
    print("\n  Top 10 most cited papers:")
    top = neo4j.get_influential_papers(limit=10)
    for i, p in enumerate(top, 1):
        print(f"    [{i:2d}] {p['citations']:>5d} citations — {p['title'][:60]}… ({p['year']})")

    duration = time.time() - start

    separator("COMPLETE", char="█")
    print(f"  Duration: {duration / 60:.1f} minutes")
    print(f"  Graph is ready for queries!")
    print()

    neo4j.close()


if __name__ == "__main__":
    main()
