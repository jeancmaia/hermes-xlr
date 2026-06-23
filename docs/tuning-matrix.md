# Engine Tuning Matrix

**Reference platform:**
**Engine:** llama-server (CUDA build)
**Model:** Llama-3.2-3B-Instruct Q4_K_M GGUF
**Date:** 2026-06-23

## Measurement protocol

Each lever is enabled **one at a time** on top of a common baseline. Record:

- **TTFT** (time to first token) — cached vs uncached
- **Decode tok/s** — tokens per second during decode (steady-state, excluding prefill)
- **Acceptance rate** (spec-decode only) — fraction of speculated tokens accepted
- **Peak VRAM** — `nvidia-smi` peak memory consumption

Baseline: all levers OFF, `--ctx-size 4096`, `--n-gpu-layers -1`.

---

## Lever 1: Prefix/slot reuse (KV-cache block reuse)

**Enables:** cached-prompt TTFT improvement by reusing KV blocks across turns.

| Level | Server CLI | API |
|-------|-----------|-----|
| Server | (built-in slot cache) | `"cache_prompt": true` |
| Persistent | `--slot-save-file <path>` | — |

**Implementation in code:**

- `KvCacheConfig.enable_block_reuse` → `extra_body["cache_prompt"]`
- Already on by default in the planner (`enable_block_reuse=True`)

**Measurement template:**

| Scenario | TTFT (ms) | Decode tok/s | Peak VRAM (MiB) |
|----------|-----------|--------------|-----------------|
| Cold (turn 1) | | | |
| Cached (turn 2) | | | |

---

## Lever 2: CUDA graphs

**Enables:** lower launch overhead by recording GPU kernel launch sequences as CUDA graphs and replaying them.

| Level | Server CLI | API |
|-------|-----------|-----|
| Server | `--cuda-graphs` | `"cuda_graphs": true` |

**Implementation in code:**

- `DecodeLevers.cuda_graphs` → `LlamaCppBackend(cuda_graphs=True)` CLI + `extra_body["cuda_graphs"]`
- Planner sets it when `GpuCapabilities.supports_cuda_graphs` is `True`

**Measurement template:**

| Config | TTFT (ms) | Decode tok/s | Delta vs baseline |
|--------|-----------|--------------|-------------------|
| OFF (baseline) | | | — |
| ON | | | |

---

## Lever 3: N-gram speculative decoding

**Enables:** the zero-cost speculation path — the engine generates N candidate tokens via n-gram lookup, validates them against the model, and batches accepted tokens into a single forward pass.

| Level | Server CLI | API |
|-------|-----------|-----|
| Server | `--speculative-ngram N` | `"speculative": {"mode": "ngram"}` |

**Implementation in code:**

- `DecodeLevers.spec_decode == SpecDecode.NGRAM` → `LlamaCppBackend(speculative_ngram=N)` CLI + `extra_body["speculative"] = {"mode": "ngram"}`
- Planner selects NGRAM when VRAM budget can't fit a draft model

**N-gram depth tuning:** start with N=32, measure acceptance rate; try N=16, N=48, N=64.

**Measurement template:**

| Config | Acceptance rate | Decode tok/s | Delta vs baseline |
|--------|----------------|--------------|-------------------|
| OFF (baseline) | — | | — |
| N=16 | | | |
| N=32 | | | |
| N=48 | | | |

---

## Lever 4: All levers combined

Compose all three levers: slot reuse + CUDA graphs + n-gram spec-decode (best N from above).

| Config | TTFT cold | TTFT cached | Decode tok/s | Peak VRAM |
|--------|-----------|-------------|--------------|-----------|
| All OFF | | | | |
| All ON | | | | |

---

## Measurement commands

```powershell
# Cold (turn 1)
curl -X POST http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":50}'

# Cached (turn 2) — same system prompt, different question
curl -X POST http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":50}'
```
