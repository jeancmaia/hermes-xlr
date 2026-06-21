---
name: pr-ready
description: Run CI-equivalent checks plus project-specific guardrails before pushing or opening a PR. Use before `git push` or when the user says "check if this is ready", "validate before PR", or "run the pre-push checks".
---

# PR-ready guardrail

Run the **exact same checks CI runs**, plus project-specific static analysis
that catches recurring anti-patterns in this repo.

## Step 1 — CI mirror

Run these in order. Stop on first failure.

```bash
uv run ruff check .          # NOT just src/ — CI checks everything
uv run ruff format --check .
uv run pytest
```

If any fails, fix it and re-run before continuing to Step 2.

## Step 2 — Project-specific guardrails

Run each check below. Flag every violation found.

### 2a — Optional dependency declarations

Find every `try/except ImportError` block across the codebase:

```bash
rg -n "except ImportError" src/
```

For each one, verify the imported package appears in `pyproject.toml` under
`[project.optional-dependencies]`. If it's missing, flag: **undeclared optional
dependency**.

### 2b — Unguarded external-tool parsing

Find every `int()` or `float()` call in the codebase that converts output from
an external source (subprocess stdout, file read, API response). These must
live inside a `try/except (ValueError, TypeError)` block.

Look for:
- `int(gpu.get(` → must be inside try/except
- `int(result.stdout` → must be inside try/except
- `float(some_parsed_value` → must be inside try/except

If a conversion is unguarded, flag: **unguarded external-tool int/float
conversion — will crash on malformed input**.

### 2c — Mock-hostile module-level sentinels

Find module-level constants that gate behavior based on optional imports:

```bash
rg -n "_HAS_|_SENTINEL|_AVAILABLE" src/
```

For each one, check whether it is evaluated at import time with `hasattr()` on
an optional module. If it is, flag: **import-time sentinel — stale under
mock.patch; use runtime `hasattr()` or a settable flag instead**.

### 2d — Data blobs in business-logic files

Scan for static dictionaries or lists with >20 entries defined in non-data
modules (any file not named `_<something>.py`):

```bash
rg -n "^\w+\s*:\s*dict\[.*\]\s*=\s*\{" src/ --after-context 5
```

Flag: **large static data structure in business-logic file — extract to a
separate `_data.py` module**.

### 2e — Spec.md references in docstrings

Find any `spec.md` references in docstrings:

```bash
rg -n "spec\.md" src/
```

If any match, flag: **spec.md reference in docstring — use HER-N issue keys
for internal cross-references instead, or remove the reference**.

## Step 3 — Summary

Report a clean/not-clean verdict. If clean, the branch is ready to push.
If violations found, list each one with file:line and the fix needed.