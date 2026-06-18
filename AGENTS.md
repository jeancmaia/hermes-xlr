# AGENTS.md — Hermes-NIM-XLR

Context primer for harness agents working in this repository. Read this first: it tells you where
work is tracked and how to work here.

## Source of working

Work is tracked in **Linear**, accessed through the **Linear MCP server** (`linear-server`).

- **Team:** Hermes-XLR
- **Project:** Hermes-NIM-XLR — Optimization-First Agent Runtime

Use the Linear MCP to read and update issues, milestones, and status under this project. Treat Linear
as the source of truth for what is planned and in progress — when you start or finish a unit of work,
reflect it there so the tracker doesn't drift from the repo.

## Workflow

- **Always work on a new branch.** Never commit directly to `main`. Branch first, then make changes.
- Keep changes scoped and reviewable; open a PR back to `main` when a unit of work is done.

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
