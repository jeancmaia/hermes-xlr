<div align="center">

# Hermes-XLR

**A Windows-native, optimization-first agent runtime for NVIDIA GPUs — maximize a local GPU and Hermes, _simultaneously_.**

[![Status](https://img.shields.io/badge/status-active-green)](#status--roadmap)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#)
[![Local engine](https://img.shields.io/badge/local%20engine-llama.cpp%20native%20Windows-orange)](https://github.com/ggml-org/llama.cpp)
[![Platform](https://img.shields.io/badge/platform-Windows%20%2B%20NVIDIA-76B900)](#hardware--os-support)
[![Built on](https://img.shields.io/badge/built%20on-hermes--agent%200.17-555)](https://github.com/NousResearch/hermes-agent)

</div>

Hermes-XLR is a **Windows-native** acceleration layer for the [`hermes-agent`](https://github.com/NousResearch/hermes-agent)
framework — it runs **natively on Windows + NVIDIA** using `llama.cpp` with CUDA, no WSL2 required. It plugs in at a
single seam — the provider transport — detects whatever NVIDIA GPU it is running on, and emits an execution plan
that **saturates that silicon**, from a 6 GB laptop to a 24 GB desktop — without forking Hermes or touching its
core.

---

## Quick start

Two scripts. That's it.

### 1. Install Hermes

```powershell
git clone https://github.com/jeancmaia/hermes-xlr.git
cd hermes-xlr
.\scripts\install-hermes.ps1
```

This runs the official Hermes Agent installer (Python, Node.js, ripgrep, ffmpeg — all handled automatically).

### 2. Install XLR

```powershell
.\scripts\install-xlr.ps1
```

This does everything else:

- Fetches the CUDA `llama-server.exe` binary
- Downloads a GGUF model from Hugging Face (Llama-3.2-3B-Instruct Q4_K_M by default)
- Installs `hermes-nim-xlr` into the Hermes venv
- Drops a `.pth` hook so Hermes **auto-registers `XLRTransport`** — every API request
  carries plan-derived config (`cache_prompt`, KV fraction, `n_gpu_layers`, etc.)
- Configures Hermes to use `http://127.0.0.1:8080/v1` as its provider

> Pass `-ModelPath C:\path\to\your.gguf` if you already have a model.
> Pass `-ModelRepo` / `-ModelFile` for a different Hugging Face model.
>
> Default model: `bartowski/Llama-3.2-3B-Instruct-GGUF` (open-access,
> 128K context — Hermes-ready). For gated models, set `$env:HF_TOKEN` first.

### 3. Run

```powershell
# Terminal 1 — launch the tuned engine (auto-detects model in models/)
.\scripts\start-xlr-engine.ps1

# Terminal 2 — configure provider (first time only)
hermes model
# -> Custom endpoint (self-hosted / VLLM / etc.)
# -> http://127.0.0.1:8080/v1
# -> (no API key)
# -> (press Enter to auto-detect model)

# Terminal 2 — start chatting
hermes
```

That's it — Hermes Agent running locally on your NVIDIA GPU, tuned by XLR. No cloud API keys,
no per-token cost, fully private.

> For troubleshooting, architecture details, and the full API reference, see the
> [Integration Guide](docs/integration-guide.md).

---

## Why

An agent turn is dominated by GPU inference, and GPU inference is **two** workloads with opposite characteristics:
a _compute-bound prefill_ (process the prompt) and a _memory-bound decode_ (generate the answer). You cannot
out-engineer the decode wall in software — so Hermes-XLR doesn't try to. It wins by **(1)** never recomputing a
cached prefix, **(2)** moving fewer bytes per token, **(3)** emitting more tokens per memory-read, and **(4)**
hiding the non-inference work behind the decode it can't avoid — and it adapts those tactics to whatever
Windows NVIDIA GPU it finds.

## How it works: where the milliseconds are

"Speed up" is Amdahl's law — optimize the dominant term. So, honestly, where does a turn's time go?
(≈150-token reply, illustrative.)

```
  capability mapping   ▏ 0 ms      (runs once at startup, never on the turn)
  transport glue       ▏ ~3 ms     (build request, parse SSE — Python, <0.1%)
  prefill (cache hit)  ██ 20–80 ms (engine, compiled C++/CUDA)
  decode (150 tok)     ████████████████████████  3,000–5,000 ms   ← THE BUDGET (engine, C++/CUDA)
```

The milliseconds live in the GPU, in compiled CUDA we **delegate** to the engine (`llama.cpp` natively on
Windows) — our own code is ~0.1% of a turn. So you beat the budget by **shrinking inference**, not by speeding
up orchestration. That is the whole game, and it is the four levers.

## The four levers

| # | Lever | Attacks | Mechanism | Target |
|---|---|---|---|---|
| 1 | **Prefix-cache reuse** | prefill / TTFT | reuse KV blocks for the unchanged prompt prefix; prefill only the new suffix | skip re-prefill of a multi-thousand-token static prompt _every turn_ |
| 2 | **Quantization (INT4 + KV)** | decode | fewer weight bytes read per token (decode is bandwidth-bound) | ~2–4× the decode ceiling, and the model _fits_ smaller VRAM |
| 3 | **Speculative decoding** | decode | draft proposes K tokens, target verifies all K in one pass | ~1.5–3× decode throughput |
| 4 | **Latency hiding** | the rest | async persistence, safe read-only prefetch, constrained decoding | overlap non-inference work with decode; emit fewer wasted tokens |

> Targets are validated by a real A/B benchmark — see `docs/ab-reports/`.

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
          (default local backend, no WSL2)  │   · speculative decode  · CUDA graphs
                                           │   · INT8/FP8 KV
     │                                    │
     ▼  normalize_response()  ──▶ native structured tool_calls (no XML scraping)
```

Three components — that is the whole surface:

- **Capability mapper** — probes the host and emits a typed `ExecutionPlan` (model + quant + KV config + decode
  levers + layer placement + backend), reading _detected_ capabilities, not constants.
- **`XLRTransport`** — a Hermes transport over a **pluggable engine-backend seam**. The default backend is
  **`llama.cpp`** (native-Windows CUDA, no WSL2). Because the seam _is_ the OpenAI contract, further backends
  (TensorRT-LLM, MLX/ROCm) drop in below it unchanged — open, but out of scope.
- **Benchmark harness** — measures TTFT, prefix-cache hit rate, inter-token latency, spec-decode acceptance,
  end-to-end turn latency, and peak VRAM against an honest A/B baseline.

**Invariants we never break** (from Hermes' own design): don't break the prefix cache · tool calls stay native ·
the transport is stateless translation · latency hiding is additive and safe (read-only prefetch, async
persistence, no speculative side effects).

## What XLR tunes vs. a plain llama-server

| Setting | Manual | XLR |
|---------|--------|-----|
| GPU layer count | Guess `-ngl 99` | Plan-driven: exact count for your VRAM |
| Context size | Guess `-c 65536` | Plan-driven: fits your VRAM budget |
| KV-cache dtype | Manual `--cache-type-k q8_0` | Plan-driven: FP8 on Ada+, INT8 on Ampere |
| CUDA graphs | Manual `--cuda-graphs` | Plan-driven: enabled if GPU supports it |
| Speculative decoding | Manual `--speculative-ngram` | Plan-driven: n-gram when budget allows |
| Prefix cache | Manual `--cache-prompt` | Plan-driven: block reuse always on |
| Model selection | Manual | Catalog-driven: largest model that fits |

## Hardware & OS support

| | Supported | Notes |
|---|---|---|
| **GPU** | NVIDIA, Ampere → Blackwell | Consumer Ampere (SM86) through Blackwell. CUDA 12.4+. |
| **OS** | Windows 11 | Native Windows, no WSL2 required. |
| **VRAM** | 6 GB minimum | 6 GB is tight but works with INT4 models. More is better. |

## Repository layout

```
hermes_nim_xlr/
  mapper/        capability mapper — DETECT host/GPU, PLAN the ExecutionPlan
  transport/     XLRTransport — stateless translation seam over the engine-backend contract
  backends/      pluggable inference-engine backends and their tuned per-backend configs
  harness/       measurement harness — honest A/B benchmarking against a vanilla baseline
  cli.py         `xlr plan` and `xlr benchmark` CLI entry points
  hermes_hook.py auto-registers XLRTransport into Hermes via .pth hook
scripts/
  install-hermes.ps1        install Hermes Agent (runs official installer)
  install-xlr.ps1           full XLR setup: binary + model + package + Hermes hook + config
  download-cuda-engine.ps1  fetch prebuilt CUDA llama-server + runtime DLLs
  start-xlr-engine.ps1      detect → plan → launch tuned engine
docs/
  integration-guide.md      end-to-end Hermes + XLR setup walkthrough
  examples/                  working example script
  ab-reports/                A/B benchmark results
  s2-bring-up.md             CUDA llama.cpp bring-up notes
```

## CLI reference

```powershell
# Install (one-time)
.\scripts\install-hermes.ps1                            # install Hermes Agent
.\scripts\install-xlr.ps1                                # install XLR + model + Hermes hook

# Run
.\scripts\start-xlr-engine.ps1                          # launch tuned engine (no args needed)
hermes                                                   # start chatting

# XLR CLI
uv run xlr plan                                          # probe GPU + emit plan as JSON
uv run xlr benchmark run --endpoint http://127.0.0.1:8080/v1  # A/B benchmark suite

# Diagnostics
hermes doctor                                            # diagnose Hermes issues
```

> `start-xlr-engine.ps1` auto-detects the model in `models/`. Pass
> `-ModelPath <path>` to override. The engine runs in the foreground;
> press Ctrl+C to stop it.

## Status & roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| S0 — Foundations | ✅ Done | Repo skeleton, pinned deps, CI, test scaffolding |
| S0.5 — Bring-up spike | ✅ Done | Hand-tuned engine proof of concept |
| S1 — Capability mapper | ✅ Done | `detect()` → `plan()` → `ExecutionPlan` |
| S2 — Backend seam | ✅ Done | `LlamaCppBackend` + CUDA bring-up |
| S3 — XLRTransport | ✅ Done | Agent integration, native tool calls, prefix stability |
| S4 — Measurement | ✅ Done | Benchmark harness + vanilla baseline |
| S5 — Decode levers | ✅ Done | INT4, KV-quant, CUDA graphs, spec-decode A/B |
| S6 — Latency hiding | ✅ Done | Async persistence, constrained decoding, safe prefetch |
| MX — Integration guide | ✅ Done | This README + `docs/integration-guide.md` |

## Built on

- **[hermes-agent](https://github.com/NousResearch/hermes-agent)** (NousResearch) — the agent framework Hermes-XLR accelerates.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) — the default local engine: native-Windows CUDA, `llama-server` OpenAI-compatible endpoint.
- **[nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/)** — GPU detection via NVML (optional, falls back to `nvidia-smi`).

## License

[MIT](./LICENSE) © 2026 jeancmaia
