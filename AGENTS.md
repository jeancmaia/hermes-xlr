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

## Reference

For anything about the `hermes-agent` framework itself — its loop, transport contract, prompt tiering,
or invariants — consult **`ref/hermes-architecture-presentation.md`**. It is the authoritative source on
the framework; rely on it rather than guessing about framework internals.
