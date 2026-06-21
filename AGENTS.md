# AGENTS.md — Hermes-NIM-XLR

Context primer for harness agents working in this repository. Read this first: it tells you where
work is tracked, how to work here, and the rules you must not break.

## Source of working

Work is tracked in **Linear**, accessed through the **Linear MCP server** (`linear-server`).

- **Team:** Hermes-XLR
- **Project:** Hermes-NIM-XLR — Optimization-First Agent Runtime

Use the Linear MCP to read and update issues, milestones, and status under this project. Treat Linear
as the source of truth for what is planned and in progress — when you start or finish a unit of work,
reflect it there so the tracker doesn't drift from the repo.

Work is organized as sprint milestones **S0 → S7**, with issues broken out under each (issue keys are
prefixed `HER-`). Pick up the relevant issue, work to its Definition of Done, and update its status as
you go rather than freelancing outside the plan.

## Workflow

- **Always work on a new branch.** Never commit directly to `main`. Branch first, then make changes.
- Keep changes scoped and reviewable; open a PR back to `main` when a unit of work is done.
- **Branch naming:** `<type>/<ticket>-<slug>`, e.g. `feat/her-1-repo-python-package-skeleton`. Use a
  Conventional-Commits type (`feat`, `fix`, `hotfix`, `chore`, `docs`, `refactor`, `test`) and the
  lowercased `HER-` ticket key. Don't use the bare username-prefixed name Linear suggests by default.

### Environment & conventions

- **Platform:** Windows; primary shell is PowerShell. Don't assume a POSIX-only environment — prefer
  cross-platform tooling, and write shell snippets that work on Windows.
- **Python 3.11+.**
- **Exact-pin dependencies.** No floating version ranges — pin every dependency for reproducibility and
  supply-chain safety.

## Invariants — never break these

These are the load-bearing rules of the runtime. They are silently breakable, and breaking one defeats
the project's core purpose. Honor them regardless of what you're changing:

1. **Don't break the prefix cache.** Add nothing per-turn ahead of the volatile prompt tier — no
   `time.time()`, no sub-day timestamps, no per-turn entropy injected into the prompt. The whole
   prefill-reuse win depends on a byte-stable prefix.
2. **Tool calls stay native.** Consume the provider's structured `tool_calls`; never regex- or
   XML-scrape tool calls out of text.
3. **The transport is stateless translation.** Caching, retry, credentials, metrics, and streaming live
   on the agent — not in the transport. The transport only translates request/response formats.
4. **Latency hiding is additive and safe.** Read-only prefetch only; persistence is async; never trigger
   a side-effecting action speculatively. The model decides, the runtime executes.

Two more discipline rules worth stating:

- **Benchmarks measure real work** — no hardcoded sleeps; every number is an honest A/B against a vanilla
  baseline on real inference.
- **Use real engine interfaces** — tune the engine's actual config fields; don't invent flags.

## What this project is

A Windows + NVIDIA, optimization-first agent runtime that accelerates the `hermes-agent` framework.
It plugs into the agent at a single seam — the provider transport — detects the host GPU, and emits an
execution plan that saturates the available silicon, with no code change. Status: design draft — no
runtime code has landed yet.

The core idea: an agent turn is dominated by GPU inference, which splits into a compute-bound *prefill*
and a memory-bound *decode*. The runtime wins on four levers — (1) never recompute a cached prefix,
(2) move fewer bytes per token (quantization), (3) emit more tokens per memory-read (speculative
decoding), and (4) hide non-inference work behind decode.

The whole surface is: a capability mapper, an `XLRTransport` over a pluggable engine-backend seam, a
tuned per-backend engine config, and a measurement harness.

## Pre-push checklist

Before pushing or opening a PR, run the **exact same commands CI runs** — not the narrower Makefile
targets:

    uv run ruff check .          # NOT `ruff check src/` — CI lints everything
    uv run ruff format --check .
    uv run pytest

Use **`/pr-ready`** (the project skill at `.claude/skills/pr-ready.md`) for a complete guardrail
pass that mirrors CI plus runs project-specific static checks.

## Project conventions

These are conventions discovered through review. Follow them; they prevent recurring
anti-patterns in this repo.

### Optional dependency policy

Any import guarded by `try/except ImportError` **must** still be declared in `pyproject.toml` under
`[project.optional-dependencies]`, even if the code gracefully degrades without it. The pyproject.toml
entry is the contract — it tells tooling, CI, and future maintainers that the dependency is intentional.

### Safe external parsing

Every `int()`, `float()`, or type conversion from external tool output (subprocess stdout, file reads,
API responses, etc.) **must** live inside a `try/except (ValueError, TypeError)` block. Never assume an
external tool produces well-formed output — corrupted or empty fields are common and must not crash the
probe.

### Mock-safe module design

Module-level constants that gate behavior (feature flags, capability sentinels like
`_HAS_NVML_CC`) must remain **dynamically re-evaluable** at test time. Prefer runtime `hasattr()`
checks over import-time sentinels computed from optional imports. If you need a sentinel, expose it
as a settable module attribute so `mock.patch.object` can override it in tests.

### Data extraction pattern

Static lookup tables larger than ~20 entries belong in a separate `_<name>.py` file, not in the
main business-logic module. This keeps the logic file skimmable and the data independently testable
and updatable without touching the probe code.

### Docstring style

No `spec.md` cross-references in docstrings. Use `HER-N` issue keys only when the cross-reference
is essential for a future maintainer reading the code in isolation. Spec references drift out of
date; Linear issue keys are traceable.

## Reference

For anything about the `hermes-agent` framework itself — its loop, transport contract, prompt tiering,
or invariants — consult **`ref/hermes-architecture-presentation.md`**. It is the authoritative source on
the framework; rely on it rather than guessing about framework internals.
