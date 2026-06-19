<div align="center">

# Hermes-XLR

**A Windows-first, optimization-first agent runtime for NVIDIA GPUs — maximize a local GPU and Hermes, _simultaneously_.**

[![Status](https://img.shields.io/badge/status-design%20draft%20(v0.1)-orange)](#status--roadmap)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#)
[![Local engine](https://img.shields.io/badge/local%20engine-llama.cpp%20native%20Windows-orange)](https://github.com/ggml-org/llama.cpp)
[![Scale engine](https://img.shields.io/badge/scale%20engine-TensorRT--LLM-76B900)](https://github.com/NVIDIA/TensorRT-LLM)
[![Platform](https://img.shields.io/badge/platform-Windows%20%2B%20NVIDIA-76B900)](#hardware--os-support)
[![Built on](https://img.shields.io/badge/built%20on-hermes--agent%200.16-555)](https://github.com/NousResearch/hermes-agent)

</div>

Hermes-XLR is a **Windows-first** acceleration layer for the [`hermes-agent`](https://github.com/NousResearch/hermes-agent)
framework — it runs **natively on Windows + NVIDIA** (default engine `llama.cpp`, no WSL2). It plugs in at a
single seam — the provider transport — detects whatever NVIDIA GPU it is running on, and emits an execution plan
that **saturates that silicon**, from a 6 GB laptop to a 24 GB desktop — without forking Hermes or touching its
core.

> **Status: design draft (v0.1).** This repository is the **thesis + architecture**. No runtime code has landed
> yet — see [Roadmap](#status--roadmap).

---

## Table of contents

- [Why](#why)
- [Highlights](#highlights)
- [How it works: where the milliseconds are](#how-it-works-where-the-milliseconds-are)
- [The four levers](#the-four-levers)
- [Architecture](#architecture)
- [Scope & non-goals](#scope--non-goals)
- [Hardware & OS support](#hardware--os-support)
- [Status & roadmap](#status--roadmap)
- [Built on](#built-on)
- [License](#license)

## Why

An agent turn is dominated by GPU inference, and GPU inference is **two** workloads with opposite characteristics:
a _compute-bound prefill_ (process the prompt) and a _memory-bound decode_ (generate the answer). You cannot
out-engineer the decode wall in software — so Hermes-XLR doesn't try to. It wins by **(1)** never recomputing a
cached prefix, **(2)** moving fewer bytes per token, **(3)** emitting more tokens per memory-read, and **(4)**
hiding the non-inference work behind the decode it can't avoid — and it adapts those tactics to whatever
Windows NVIDIA GPU it finds, a 6 GB laptop through a 24 GB desktop.


## How it works: where the milliseconds are

"Speed up" is Amdahl's law — optimize the dominant term. So, honestly, where does a turn's time go?
(≈150-token reply, 6 GB-class GPU, illustrative.)

```
  capability mapping   ▏ 0 ms      (runs once at startup, never on the turn)
  transport glue       ▏ ~3 ms     (build request, parse SSE — Python, <0.1%)
  prefill (cache hit)  ██ 20–80 ms (engine, compiled C++/CUDA)
  decode (150 tok)     ████████████████████████  3,000–5,000 ms   ← THE BUDGET (engine, C++/CUDA)
```

The milliseconds live in the GPU, in compiled CUDA we **delegate** to the engine (`llama.cpp` natively on
Windows, TensorRT-LLM for performance/scale) — our own code is ~0.1% of a turn. So you beat the budget by
**shrinking inference**, not by speeding up orchestration. That is the whole game, and it is the four levers.

## The four levers

| # | Lever | Attacks | Mechanism | Target |
|---|---|---|---|---|
| 1 | **Prefix-cache reuse** | prefill / TTFT | reuse KV blocks for the unchanged prompt prefix; prefill only the new suffix | skip re-prefill of a multi-thousand-token static prompt _every turn_ |
| 2 | **Quantization (INT4 + KV)** | decode | fewer weight bytes read per token (decode is bandwidth-bound) | ~2–4× the decode ceiling, and the model _fits_ 6 GB |
| 3 | **Speculative decoding** | decode | draft proposes K tokens, target verifies all K in one pass | ~1.5–3× decode throughput |
| 4 | **Latency hiding** | the rest | async persistence, safe read-only prefetch, constrained decoding | overlap non-inference work with decode; emit fewer wasted tokens |

> Targets are **targets** — validated by a real A/B benchmark, never hardcoded.

**The marriage with Hermes (lever 1, the intellectual core).** Prefix reuse only fires if the prompt prefix is
**byte-stable across turns** — and Hermes _already_ engineers for exactly this (frozen memory snapshot, tiered
prompt, date-only timestamps). Hermes-XLR's job is not to reinvent that, but to **ride it**: keep the transport
from adding per-turn entropy, and let the engine's block-reuse cache hit. _That_ is "maximize the GPU and Hermes
simultaneously" — they meet here.

## Architecture

Hermes-XLR hooks Hermes at exactly **one seam — the `ProviderTransport`** — and adds nothing the core must know about.

```
   AIAgent  (Hermes sync core — untouched)
     │  build_kwargs()
     ▼
   ┌─────────────── XLRTransport (pure translation; adds NO per-turn entropy) ───────────────┐
   │   capability mapper ──▶ ExecutionPlan ──▶ configures ▼                                   │
   └──────────────────────── engine-backend seam (OpenAI-compatible contract) ───────────────┘
     │                                   ▲ SSE token stream
     ▼  ▶ llama.cpp — native Windows, CUDA │   · INT4 weights        · block reuse (lever 1)
          (default local backend, no WSL2) │   · speculative decode  · CUDA graphs
        ▶ TensorRT-LLM — performance/scale │   · INT8/FP8 KV
          (Linux native; WSL2 on Windows)  │
          MLX / ROCm — seam open, off-scope│
     │                                    │
     ▼  normalize_response()  ──▶ native structured tool_calls (no XML scraping)
```

Three components — that is the whole surface:

- **Capability mapper** — probes the host and emits a typed `ExecutionPlan` (model + quant + KV config + decode
  levers + layer placement + backend), reading _detected_ capabilities, not constants.
- **`XLRTransport`** — a Hermes transport over a **pluggable engine-backend seam**. Two backends in scope:
  **`llama.cpp`** (native-Windows CUDA, no WSL2) — the **default local** backend — and **TensorRT-LLM** (Linux
  native, WSL2 on Windows) — the **performance/scale** path; because the seam _is_ the OpenAI contract, further
  backends (MLX/ROCm) drop in below it unchanged — open, but out of scope.
- **Benchmark harness** — measures TTFT, prefix-cache hit rate, inter-token latency, spec-decode acceptance,
  end-to-end turn latency, and peak VRAM against an honest A/B baseline.

**Invariants we never break** (from Hermes' own design): don't break the prefix cache · tool calls stay native ·
the transport is stateless translation · latency hiding is additive and safe (read-only prefetch, async
persistence, no speculative side effects).

## Scope & non-goals

- **NIM-compatible, not NIM-on-6 GB.** NIM's floor is 8 GB VRAM; it will not run on a 6 GB GPU.
  Hermes-XLR speaks the same OpenAI contract NIM speaks — _develop locally on native Windows (`llama.cpp`), step
  up to TensorRT-LLM / NIM for performance and scale_. We never claim NIM runs on 6 GB.
- **We saturate the decode wall; we don't break it.** "Amplify" means extract the maximum from the silicon
  present — not exceed physics. There is no sub-millisecond agent loop; there is a budget, fully spent.
- **Python above the seam, compiled where it counts.** The hot path is the engine's compiled CUDA (`llama.cpp`
  or TensorRT-LLM). The glue is Python for velocity and portability; native code only ever arrives behind a
  profiler's evidence.
- **Windows + NVIDIA first; portable by architecture, not by scope.** The capability mapper and backend seam are
  OS- and vendor-agnostic by construction, but the only paths we build and measure are Windows + NVIDIA (native
  `llama.cpp` by default, TensorRT-LLM via WSL2 for performance). MLX / ROCm / Linux-datacenter stay reachable
  through the seam — explicitly out of scope until the Windows NVIDIA path is proven end-to-end.

## Hardware & OS support

Scope is **Windows + NVIDIA**, across the full consumer GPU range. The architecture is vendor-agnostic above the
backend seam, but Windows + NVIDIA is what we build, run, and measure.

| | Supported | Notes |
|---|---|---|
| **GPU** | NVIDIA, Ampere → Blackwell | Verified against NVIDIA's [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html): Ampere **SM80/SM86**, Ada SM89, Hopper SM90, Blackwell. **Consumer Ampere (SM86)** cards are explicitly supported. Turing/Volta have dropped off the list. |
| **OS** | Windows 11 (+ Linux) | Two engine paths below. macOS is out of scope (no CUDA). |
| **VRAM** | 6 GB (entry); more is better | 6 GB is tight and **unproven** until bring-up — the matrix sets no minimum, and no numbers are measured yet. |

**Two engine paths on Windows — the capability mapper picks automatically, native-first:**

- **`llama.cpp` (native Windows, CUDA)** — the **default local backend**: GGUF INT4, `llama-server`'s
  OpenAI-compatible `/v1`, slot/prefix reuse, CUDA graphs, n-gram speculative decoding. **No WSL2, no engine
  build.** The genuinely Windows-native path, and the easiest bring-up. The same backend later carries
  Metal / ROCm / CPU portability.
- **TensorRT-LLM — the performance / scale path**, and **Linux-only**: NVIDIA's matrix states it _"requires
  Linux x86_64 or Linux aarch64."_ On Windows it runs through **WSL2** (opt-in, for maximum throughput). Caveat:
  WSL2 _serves_ fine, but **engine builds are slow on the WSL2 filesystem** — build the INT4 engine on native
  Linux (or ship a pre-built engine) and serve under WSL2. At deploy, the same OpenAI contract reaches **NIM**.

> **Native Linux** (Docker Engine + NVIDIA Container Toolkit) is TensorRT-LLM's home and the lowest-overhead path
> for it — direct GPU, ~1 GB more usable VRAM — and the recommended place to _build_ its engines.

## Repository layout

```
src/hermes_nim_xlr/
  mapper/      capability mapper — DETECT host/GPU, PLAN the ExecutionPlan
  transport/   XLRTransport — stateless translation seam over the engine-backend contract
  backends/    pluggable inference-engine backends and their tuned per-backend configs
  harness/     measurement harness — honest A/B benchmarking against a vanilla baseline
```

Each submodule maps directly to a component in [Architecture](#architecture). They start as stubs; later
sprints (see [Status & roadmap](#status--roadmap)) fill them in. Install locally with `pip install -e .`; lint
with `ruff check src/`.

## Status & roadmap

This is **v0.1, a design proposal.** No runtime code has landed. Next, in order:

1. **Capability mapper** — implement `DETECT` + `plan()`; prove it emits the expected `ExecutionPlan` on the
   local GPU and flips correctly for synthetic Ada/Hopper, multi-GPU, and tiny-VRAM offload profiles.
2. **Bring-up** — start with the default native-Windows `llama.cpp` (no WSL2); add the TensorRT-LLM
   performance path (native-Linux Docker, or WSL2 on Windows) afterward. Stand up an OpenAI-compatible endpoint
   with a concrete INT4 model; record first real VRAM + tok/s numbers.
3. **`XLRTransport`** — wire it to Hermes; verify native `tool_calls` round-trip and prefix stability.
4. **Decode levers** — add INT4 → KV-quant → CUDA graphs → speculative decoding, A/B each.
5. **Latency hiding** — async persistence + safe read-only prefetch + constrained tool-arg decoding.

## Built on

- **[hermes-agent](https://github.com/NousResearch/hermes-agent)** (NousResearch) — the agent framework Hermes-XLR accelerates.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) — the default local engine: native-Windows CUDA, `llama-server` OpenAI-compatible endpoint.
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** (NVIDIA) — the performance/scale engine and `trtllm-serve` OpenAI-compatible server (Linux / WSL2).
- **[TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)**, **[nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/)**, **[nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** — quantization, GPU detection, and container GPU passthrough.

## License

[MIT](./LICENSE) © 2026 jeancmaia
