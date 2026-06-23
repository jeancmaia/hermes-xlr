# S2 — CUDA llama.cpp Bring-up & Tool-call Validation

**Date:** 2026-06-22
**Engine:** llama-server b9763 (dec5ca557) via CUDA 12.4
**Model:** Llama-3.2-3B-Instruct Q4_K_M GGUF
**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 572.40)
**CUDA Runtime:** 12.4 (cudart64_12.dll)

## CUDA vs Vulkan comparison

| Metric | Vulkan (S0.5, Nemotron-4B) | CUDA (S2, Llama-3.2-3B) |
|---|---|---|
| VRAM (loaded) | 3178 MiB | 4516 MiB |
| VRAM (free) | ~2825 MiB | ~1487 MiB |
| Model buffer | — | 1918 MiB |
| KV cache | 256 MiB (4096 ctx) | 2128 MiB (131072 ctx default) |
| Decode (uncached) | ~45 tok/s | ~26 tok/s |
| Decode (cached) | ~52 tok/s | ~48 tok/s |
| TTFT (uncached) | 288 ms (26 tok) | 644 ms (42 tok) |
| TTFT (cached) | 61 ms (4.7×) | 419 ms (cached) |
| Layers GPU | 33/33 | 29/29 |
| Tool calls | Text (`<toolcall>` XML) | **Native `tool_calls`** ✅ |

## Tool-call validation

- **finish_reason:** `tool_calls` (native OpenAI format)
- **Tool name:** `get_weather`
- **Arguments:** `{"city": "Paris"}` (valid JSON, extracted from structured `function.arguments`)
- **Prompt tokens:** 179, **Completion tokens:** 17
- **No XML/text scraping needed** — invariant #2 satisfied.

## Key findings

1. **CUDA works.** The CUDA build (`b9763`) correctly finds the GPU, offloads all 29/29 layers, and serves inference.

2. **CUDA runtime requirement.** The CUDA build needs `cudart64_12.dll` alongside the binary. The release provides this separately (`cudart-llama-bin-*`). Without it, `llama-server` falls back to CPU with a "no usable GPU found" warning.

3. **Llama-3.2-3B-Instruct has native tool-call support.** Unlike Nemotron-Mini-4B-Instruct (which emits `<toolcall>` XML in text), Llama-3.2 produces structured `tool_calls` with proper `finish_reason: tool_calls`. This satisfies the project invariant that tool calls stay native.

4. **VRAM budget is adequate but tighter with Llama-3.2.** The model uses ~4516 MiB vs 3179 MiB for Nemotron, primarily because the default context window (131k) allocates a large KV cache. The server should be configured with `--ctx-size 4096` or `--no-context-length` to reduce KV memory waste.

5. **Prefix caching works.** Turn 2 decode (48 tok/s) is faster than turn 1 (26 tok/s), showing effective slot-based KV reuse.

## Notes for downstream

- Default `LlamaCppBackend` config should set `--ctx-size 4096` to keep KV cache within budget.
- Llama-3.2-3B-Instruct Q4_K_M is recommended as the new primary reference model over Nemotron for its native tool-call support.
- The `bin/` directory should be checked into git or managed as a download artifact.
