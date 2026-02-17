# Development Log

Tracking incremental improvements and fixes.

- [2026-02-02] refactor: extract context builder into separate method
- [2026-02-02] fix: skip graph enrichment when Neo4j times out
- [2026-02-07] chore: normalize newlines in frontend api module
- [2026-02-08] refactor: extract rewrite prompt constants to top
- [2026-02-08] fix: cap connected paper results to avoid memory spike
- [2026-02-08] chore: align indentation in search pipeline
- [2026-02-10] fix: guard against None abstract in paper context
- [2026-02-10] refactor: extract query validation into helper
- [2026-02-10] chore: remove stale TODO comment
- [2026-02-11] fix: default category filter to None not empty list
- [2026-02-11] perf: cache Neo4j client connection across requests
- [2026-02-11] fix: add fallback for missing chunk_snippet field
- [2026-02-14] docs: add note about BM25 sparse vector behaviour
- [2026-02-14] chore: add blank line between logical blocks
- [2026-02-14] docs: add inline comments to hybrid search stages
- [2026-02-16] refactor: rename internal var for clarity in reranker
- [2026-02-17] fix: skip expansion for papers already fully chunked
- [2026-02-17] chore: minor formatting in rate limiter module
