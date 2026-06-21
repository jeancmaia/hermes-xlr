# Hermes-NIM-XLR — common dev tasks
#
# Uses `uv` for Python environment management (cross-platform).
# On Windows, run via Git Bash or WSL, or `nmake` (limited support).
# On any platform: `uv run pytest` and `uv run ruff` work directly.

.PHONY: test lint check

# ── Tests ──────────────────────────────────────────────────────────

test:  ## Run the full test suite
	uv run pytest

# ── Lint ───────────────────────────────────────────────────────────

lint:  ## Lint-check all source files
	uv run ruff check src/

# ── Combined ───────────────────────────────────────────────────────

check: test lint  ## Run tests then lint