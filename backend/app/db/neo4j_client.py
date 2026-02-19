# backend/app/db/neo4j_client.py

"""
Neo4j client for the research paper knowledge graph.

Schema:
  (:Paper {id, title, year, arxiv_id, s2_id, citation_count, field})
  (:Author {name})
  (:Category {name})

  (Paper)-[:AUTHORED_BY]->(Author)
  (Paper)-[:CITES]->(Paper)
  (Paper)-[:IN_CATEGORY]->(Category)
  (Author)-[:COAUTHORED_WITH {paper_count}]->(Author)
"""

from neo4j import GraphDatabase
from typing import List, Dict, Optional
from loguru import logger


class Neo4jClient:

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        # Verify connection
        self.driver.verify_connectivity()
        logger.info(f"Neo4j connected: {uri}")

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def create_indexes(self):
        """Create indexes and constraints for performance."""
        queries = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX paper_arxiv IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)",
            "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX paper_field IF NOT EXISTS FOR (p:Paper) ON (p.field)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    # Some constraints may already exist
                    logger.debug(f"Index/constraint: {e}")
        logger.info("Neo4j indexes created.")

    # ------------------------------------------------------------------
    # Ingestion (batch)
    # ------------------------------------------------------------------

    def ingest_papers_batch(self, papers: List[Dict], batch_size: int = 500):
        """
        Ingest a batch of enriched papers into the graph.

        Each paper dict should have:
          id, arxiv_id, title, year, authors, categories,
          primary_category, citation_count, references, cited_by
        """
        with self.driver.session() as session:
            for i in range(0, len(papers), batch_size):
                batch = papers[i: i + batch_size]
                # 1. Create Paper nodes
                session.execute_write(self._create_paper_nodes, batch)
                # 2. Create Author nodes + AUTHORED_BY
                session.execute_write(self._create_author_relationships, batch)
                # 3. Create Category nodes + IN_CATEGORY
                session.execute_write(self._create_category_relationships, batch)
                # 4. Create CITES relationships
                session.execute_write(self._create_citation_relationships, batch)

                logger.info(f"  Graph batch {i + len(batch)}/{len(papers)} ingested")

    @staticmethod
    def _create_paper_nodes(tx, papers: List[Dict]):
        tx.run(
            """
            UNWIND $papers AS p
            MERGE (paper:Paper {id: p.id})
            SET paper.arxiv_id       = p.arxiv_id,
                paper.title          = p.title,
                paper.year           = p.year,
                paper.citation_count = p.citation_count,
                paper.field          = p.field,
                paper.s2_id          = p.s2_id
            """,
            papers=[
                {
                    "id": p["id"],
                    "arxiv_id": p.get("arxiv_id", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", 0),
                    "citation_count": p.get("citation_count", 0),
                    "field": _extract_field(p.get("primary_category", "")),
                    "s2_id": p.get("s2_id", ""),
                }
                for p in papers
            ],
        )

    @staticmethod
    def _create_author_relationships(tx, papers: List[Dict]):
        # Flatten papers → (paper_id, author_name) pairs
        rows = []
        for p in papers:
            for author in p.get("authors", [])[:20]:  # cap at 20 authors
                rows.append({"paper_id": p["id"], "author": author.strip()})

        if not rows:
            return

        tx.run(
            """
            UNWIND $rows AS r
            MERGE (a:Author {name: r.author})
            WITH a, r
            MATCH (p:Paper {id: r.paper_id})
            MERGE (p)-[:AUTHORED_BY]->(a)
            """,
            rows=rows,
        )

    @staticmethod
    def _create_category_relationships(tx, papers: List[Dict]):
        rows = []
        for p in papers:
            for cat in p.get("categories", [])[:10]:
                rows.append({"paper_id": p["id"], "category": cat})

        if not rows:
            return

        tx.run(
            """
            UNWIND $rows AS r
            MERGE (c:Category {name: r.category})
            WITH c, r
            MATCH (p:Paper {id: r.paper_id})
            MERGE (p)-[:IN_CATEGORY]->(c)
            """,
            rows=rows,
        )

    @staticmethod
    def _create_citation_relationships(tx, papers: List[Dict]):
        # CITES: this paper → referenced papers
        rows = []
        for p in papers:
            for ref_arxiv_id in p.get("references", []):
                rows.append({
                    "from_id": p["id"],
                    "to_id": f"arxiv:{ref_arxiv_id}",
                })
            for citing_arxiv_id in p.get("cited_by", []):
                rows.append({
                    "from_id": f"arxiv:{citing_arxiv_id}",
                    "to_id": p["id"],
                })

        if not rows:
            return

        tx.run(
            """
            UNWIND $rows AS r
            MERGE (from:Paper {id: r.from_id})
            MERGE (to:Paper {id: r.to_id})
            MERGE (from)-[:CITES]->(to)
            """,
            rows=rows,
        )

    # ------------------------------------------------------------------
    # Build co-authorship edges (run once after all papers ingested)
    # ------------------------------------------------------------------

    def build_coauthorship(self):
        """
        Create COAUTHORED_WITH edges between authors who share papers.
        Run once after full ingestion.
        """
        logger.info("Building co-authorship graph…")
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
                WHERE id(a1) < id(a2)
                WITH a1, a2, count(p) AS papers_together
                MERGE (a1)-[r:COAUTHORED_WITH]-(a2)
                SET r.paper_count = papers_together
                RETURN count(r) AS edges_created
                """
            )
            count = result.single()["edges_created"]
            logger.info(f"Co-authorship edges: {count}")

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_paper_citations(self, paper_id: str) -> Dict:
        """Get papers that cite this paper and papers it references."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper {id: $paper_id})
                OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                RETURN p.title AS title,
                       p.citation_count AS citation_count,
                       collect(DISTINCT {id: ref.id, title: ref.title, year: ref.year}) AS references,
                       collect(DISTINCT {id: citing.id, title: citing.title, year: citing.year}) AS cited_by
                """,
                paper_id=paper_id,
            )
            record = result.single()
            if not record:
                return {}
            return {
                "title": record["title"],
                "citation_count": record["citation_count"],
                "references": [r for r in record["references"] if r["id"]],
                "cited_by": [r for r in record["cited_by"] if r["id"]],
            }

    def get_author_network(self, author_name: str, depth: int = 1) -> Dict:
        """Get an author's collaborators and their shared papers."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Author {name: $name})
                OPTIONAL MATCH (a)-[r:COAUTHORED_WITH]-(coauthor:Author)
                OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
                RETURN a.name AS name,
                       count(DISTINCT p) AS paper_count,
                       collect(DISTINCT {
                           name: coauthor.name,
                           papers_together: r.paper_count
                       }) AS collaborators
                ORDER BY r.paper_count DESC
                """,
                name=author_name,
            )
            record = result.single()
            if not record:
                return {}
            return {
                "name": record["name"],
                "paper_count": record["paper_count"],
                "collaborators": sorted(
                    [c for c in record["collaborators"] if c["name"]],
                    key=lambda x: x["papers_together"] or 0,
                    reverse=True,
                ),
            }

    def get_top_authors_in_field(self, field: str, limit: int = 20) -> List[Dict]:
        """Top authors by paper count in a field."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper {field: $field})-[:AUTHORED_BY]->(a:Author)
                WITH a, count(p) AS papers, sum(p.citation_count) AS total_citations
                RETURN a.name AS name, papers, total_citations
                ORDER BY total_citations DESC, papers DESC
                LIMIT $limit
                """,
                field=field,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_influential_papers(self, field: str = None, limit: int = 20) -> List[Dict]:
        """Most cited papers, optionally filtered by field."""
        with self.driver.session() as session:
            if field:
                result = session.run(
                    """
                    MATCH (p:Paper {field: $field})
                    WHERE p.citation_count > 0
                    RETURN p.id AS id, p.title AS title, p.year AS year,
                           p.citation_count AS citations, p.field AS field
                    ORDER BY p.citation_count DESC
                    LIMIT $limit
                    """,
                    field=field, limit=limit,
                )
            else:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE p.citation_count > 0
                    RETURN p.id AS id, p.title AS title, p.year AS year,
                           p.citation_count AS citations, p.field AS field
                    ORDER BY p.citation_count DESC
                    LIMIT $limit
                    """,
                    limit=limit,
                )
            return [dict(r) for r in result]

    def get_citation_path(
        self, from_id: str, to_id: str, max_depth: int = 4
    ) -> List[Dict]:
        """Find shortest citation path between two papers."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (a:Paper {id: $from_id})-[:CITES*1..%d]-(b:Paper {id: $to_id})
                )
                RETURN [n IN nodes(path) | {id: n.id, title: n.title, year: n.year}] AS path
                """ % max_depth,
                from_id=from_id,
                to_id=to_id,
            )
            record = result.single()
            return record["path"] if record else []

    # ------------------------------------------------------------------
    # Advanced graph queries
    # ------------------------------------------------------------------

    def get_paper_context(self, paper_id: str) -> Dict:
        """
        Rich context for a single paper: citations, references,
        author's other work, related highly-cited papers.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper {id: $paper_id})
                
                // Papers this one cites
                OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
                WITH p, collect(DISTINCT {id: ref.id, title: ref.title, year: ref.year, citations: ref.citation_count})[..5] AS refs
                
                // Papers that cite this one
                OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                WITH p, refs, collect(DISTINCT {id: citing.id, title: citing.title, year: citing.year, citations: citing.citation_count})[..5] AS cited_by
                
                // Authors and their other top papers
                OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(other:Paper)
                WHERE other.id <> p.id AND other.citation_count > 0
                WITH p, refs, cited_by, a,
                     collect(DISTINCT {id: other.id, title: other.title, year: other.year, citations: other.citation_count})[..3] AS author_papers
                WITH p, refs, cited_by,
                     collect(DISTINCT {name: a.name, other_papers: author_papers})[..5] AS authors
                
                RETURN p.id AS id, p.title AS title, p.year AS year,
                       p.citation_count AS citation_count, p.field AS field,
                       refs, cited_by, authors
                """,
                paper_id=paper_id,
            )
            record = result.single()
            if not record:
                return {}
            return {
                "id": record["id"],
                "title": record["title"],
                "year": record["year"],
                "citation_count": record["citation_count"],
                "field": record["field"],
                "references": [r for r in record["refs"] if r["id"]],
                "cited_by": [r for r in record["cited_by"] if r["id"]],
                "authors": [a for a in record["authors"] if a["name"]],
            }

    def get_connected_papers(
        self, paper_ids: List[str], max_extra: int = 10
    ) -> List[Dict]:
        """
        Find highly-cited papers connected to a set of papers
        (via citations) that aren't already in the set.
        This is the "graph expansion" step.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $ids AS pid
                MATCH (p:Paper {id: pid})
                
                // Get papers connected via citation in either direction
                OPTIONAL MATCH (p)-[:CITES]-(connected:Paper)
                WHERE NOT connected.id IN $ids
                  AND connected.citation_count > 0
                  AND connected.title IS NOT NULL
                
                WITH connected, count(DISTINCT p) AS shared_connections
                WHERE connected IS NOT NULL
                
                RETURN connected.id AS id,
                       connected.title AS title,
                       connected.year AS year,
                       connected.citation_count AS citations,
                       connected.field AS field,
                       shared_connections
                ORDER BY shared_connections DESC, connected.citation_count DESC
                LIMIT $limit
                """,
                ids=paper_ids,
                limit=max_extra,
            )
            return [dict(r) for r in result]

    def get_common_ancestor(self, paper_ids: List[str], limit: int = 5) -> List[Dict]:
        """
        Find papers that multiple of the given papers all cite.
        These are likely seminal/foundational papers.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $ids AS pid
                MATCH (p:Paper {id: pid})-[:CITES]->(ancestor:Paper)
                WHERE ancestor.title IS NOT NULL
                WITH ancestor, count(DISTINCT p) AS cited_by_count
                WHERE cited_by_count >= 2
                RETURN ancestor.id AS id,
                       ancestor.title AS title,
                       ancestor.year AS year,
                       ancestor.citation_count AS citations,
                       cited_by_count
                ORDER BY cited_by_count DESC, ancestor.citation_count DESC
                LIMIT $limit
                """,
                ids=paper_ids,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_trending_papers(
        self, field: Optional[str] = None, min_year: int = 2024, limit: int = 10
    ) -> List[Dict]:
        """
        Papers published recently that are accumulating citations fast.
        Trending = recent + high citation velocity.
        """
        with self.driver.session() as session:
            if field:
                result = session.run(
                    """
                    MATCH (p:Paper {field: $field})
                    WHERE p.year >= $min_year AND p.citation_count > 5
                    RETURN p.id AS id, p.title AS title, p.year AS year,
                           p.citation_count AS citations, p.field AS field
                    ORDER BY p.citation_count DESC
                    LIMIT $limit
                    """,
                    field=field, min_year=min_year, limit=limit,
                )
            else:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE p.year >= $min_year AND p.citation_count > 5
                    RETURN p.id AS id, p.title AS title, p.year AS year,
                           p.citation_count AS citations, p.field AS field
                    ORDER BY p.citation_count DESC
                    LIMIT $limit
                    """,
                    min_year=min_year, limit=limit,
                )
            return [dict(r) for r in result]

    def get_field_bridges(self, cat1: str, cat2: str, limit: int = 10) -> List[Dict]:
        """
        Find papers that bridge two categories
        (have relationships to papers in both).
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper)-[:IN_CATEGORY]->(c1:Category {name: $cat1})
                MATCH (p)-[:IN_CATEGORY]->(c2:Category {name: $cat2})
                WHERE p.title IS NOT NULL
                RETURN p.id AS id, p.title AS title, p.year AS year,
                       p.citation_count AS citations
                ORDER BY p.citation_count DESC
                LIMIT $limit
                """,
                cat1=cat1, cat2=cat2, limit=limit,
            )
            return [dict(r) for r in result]

    def get_research_lineage(self, paper_id: str, depth: int = 3) -> Dict:
        """
        Get the "family tree" — what a paper builds on and what built on it.
        Returns a tree structure.
        """
        with self.driver.session() as session:
            # Ancestors (papers this one cites, recursively)
            ancestors = session.run(
                """
                MATCH (start:Paper {id: $paper_id})
                MATCH path = (start)-[:CITES*1..%d]->(ancestor:Paper)
                WHERE ancestor.title IS NOT NULL
                WITH ancestor, length(path) AS depth
                RETURN DISTINCT ancestor.id AS id, ancestor.title AS title,
                       ancestor.year AS year, ancestor.citation_count AS citations,
                       depth
                ORDER BY depth, ancestor.citation_count DESC
                LIMIT 15
                """ % depth,
                paper_id=paper_id,
            )

            # Descendants (papers that cite this one, recursively)
            descendants = session.run(
                """
                MATCH (start:Paper {id: $paper_id})
                MATCH path = (descendant:Paper)-[:CITES*1..%d]->(start)
                WHERE descendant.title IS NOT NULL
                WITH descendant, length(path) AS depth
                RETURN DISTINCT descendant.id AS id, descendant.title AS title,
                       descendant.year AS year, descendant.citation_count AS citations,
                       depth
                ORDER BY depth, descendant.year DESC
                LIMIT 15
                """ % depth,
                paper_id=paper_id,
            )

            return {
                "ancestors": [dict(r) for r in ancestors],
                "descendants": [dict(r) for r in descendants],
            }

    def get_reading_list(
        self, paper_ids: List[str], limit: int = 10
    ) -> List[Dict]:
        """
        Build a curated reading list from a set of papers:
        seminal → foundational → recent.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                // Start from given papers
                UNWIND $ids AS pid
                MATCH (p:Paper {id: pid})
                
                // Get the full citation neighborhood
                OPTIONAL MATCH (p)-[:CITES*1..2]-(related:Paper)
                WHERE related.title IS NOT NULL
                
                // Combine starting papers + related
                WITH collect(DISTINCT p) + collect(DISTINCT related) AS all_papers
                UNWIND all_papers AS paper
                
                WITH DISTINCT paper
                RETURN paper.id AS id,
                       paper.title AS title,
                       paper.year AS year,
                       paper.citation_count AS citations,
                       paper.field AS field
                ORDER BY paper.year ASC, paper.citation_count DESC
                LIMIT $limit
                """,
                ids=paper_ids,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_author_expertise(
        self, topic_paper_ids: List[str], limit: int = 10
    ) -> List[Dict]:
        """
        Find top authors for a topic based on papers in a citation cluster.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $ids AS pid
                MATCH (p:Paper {id: pid})-[:CITES*0..1]-(related:Paper)
                MATCH (related)-[:AUTHORED_BY]->(a:Author)
                WITH a, count(DISTINCT related) AS papers_in_cluster,
                     sum(related.citation_count) AS total_citations
                WHERE papers_in_cluster >= 2
                RETURN a.name AS name,
                       papers_in_cluster,
                       total_citations
                ORDER BY papers_in_cluster DESC, total_citations DESC
                LIMIT $limit
                """,
                ids=topic_paper_ids,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_research_gaps(
        self, category: str, min_citations: int = 20, limit: int = 10
    ) -> List[Dict]:
        """
        Find highly-cited papers in a category that have few papers
        citing them from recent years — potential underexplored directions.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper)-[:IN_CATEGORY]->(:Category {name: $category})
                WHERE p.citation_count >= $min_citations AND p.year <= 2023
                
                // Count recent papers citing this one
                OPTIONAL MATCH (recent:Paper)-[:CITES]->(p)
                WHERE recent.year >= 2025
                WITH p, count(recent) AS recent_citers
                WHERE recent_citers <= 2
                
                RETURN p.id AS id, p.title AS title, p.year AS year,
                       p.citation_count AS citations,
                       recent_citers
                ORDER BY p.citation_count DESC
                LIMIT $limit
                """,
                category=category,
                min_citations=min_citations,
                limit=limit,
            )
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> Dict:
        """Get node and relationship counts."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Paper) WITH count(p) AS papers
                MATCH (a:Author) WITH papers, count(a) AS authors
                MATCH (c:Category) WITH papers, authors, count(c) AS categories
                OPTIONAL MATCH ()-[cites:CITES]->() WITH papers, authors, categories, count(cites) AS citation_edges
                OPTIONAL MATCH ()-[co:COAUTHORED_WITH]-() WITH papers, authors, categories, citation_edges, count(co) AS coauthor_edges
                RETURN papers, authors, categories, citation_edges, coauthor_edges
                """
            )
            record = result.single()
            return dict(record) if record else {}


# ── Helper ───────────────────────────────────────────────────────────

def _extract_field(category: str) -> str:
    prefix = category.split(".")[0] if "." in category else category
    return {
        "cs": "Computer Science",
        "math": "Mathematics",
        "physics": "Physics",
        "stat": "Statistics",
        "eess": "Electrical Engineering",
        "q-bio": "Quantitative Biology",
        "q-fin": "Quantitative Finance",
        "econ": "Economics",
    }.get(prefix, "Other")
