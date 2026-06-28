"""Safe (idempotent, read-only) tool names for speculative prefetch.

Tools in this set are safe to speculatively cache during decode because
they produce no side effects — calling them early and reusing their
results across turns is sound.

This set is deliberately conservative: only tools whose implementations
are known to be pure reads belong here. Side-effecting tools
(``web_search_write``, ``send_email``, etc.) are excluded.

The module-level attribute is settable so tests can ``mock.patch.object``
it without touching the frozenset itself (AGENTS.md: mock-safe module
design).
"""

from __future__ import annotations

SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "web_search_read",
        "file_read",
        "memory_lookup",
        "document_search",
        "web_scrape",
        "vector_search",
    }
)
