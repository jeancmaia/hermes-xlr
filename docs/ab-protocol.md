# A/B Protocol — Hermes-NIM-XLR Measurement Harness

## Purpose

Standardised procedure for measuring the impact of each decode lever
(Prefix/CUDA graphs / Speculative decoding) against a common all-levers-OFF
baseline on the reference RTX 3050 6 GB laptop GPU.

## Running a baseline

```powershell
# Via the BenchmarkHarness API:
uv run python -c "
from hermes_nim_xlr.contracts import ExecutionPlan
from hermes_nim_xlr.harness.benchmark import BenchmarkHarness
from hermes_nim_xlr.mapper import detect, plan

host = detect.detect()
plan = plan(host)
harness = BenchmarkHarness(plan)
report = harness.run(turns=5)
print(report.to_json())
"
```

A baseline run produces a `BenchmarkReport` JSON with:
- `plan` — the full `ExecutionPlan` (all levers OFF)
- `turns` — per-turn metrics (TTFT, tok/s, tool-call count, peak VRAM)
- `summary` — aggregated statistics

## Toggling levers

Each lever is tested one at a time against the same baseline. The `run_ab()`
method accepts a baseline report, a lever name, and a dict of field overrides:

```python
# CUDA graphs
ab_result = harness.run_ab(
    baseline_report=baseline,
    lever_name="cuda_graphs",
    lever_config={"cuda_graphs": True},
)

# N-gram speculative decoding
ab_result = harness.run_ab(
    baseline_report=baseline,
    lever_name="spec_decode_ngram",
    lever_config={"spec_decode": SpecDecode.NGRAM},
)
```

## Lever matrix

| Lever | `lever_config` | Effect |
|-------|---------------|--------|
| Prefix/slot reuse | `{"enable_block_reuse": True}` | KV-cache block reuse via `cache_prompt` |
| CUDA graphs | `{"cuda_graphs": True}` | Lower launch overhead via CUDA graph replay |
| N-gram spec-decode | `{"spec_decode": SpecDecode.NGRAM}` | Zero-cost n-gram speculation |
| Draft-target spec-decode | `{"spec_decode": SpecDecode.DRAFT_TARGET, "draft_model": "..."}` | Small draft model speculation |

## JSON report structure

```json
{
  "plan": { ... ExecutionPlan as dict ... },
  "turns": [
    {
      "turn": 0,
      "ttft_ms": 644.0,
      "tokens": 42,
      "tok_s": 26.0,
      "tool_calls": 0,
      "peak_vram_mb": 4516,
      "cached": false
    },
    {
      "turn": 1,
      "ttft_ms": 419.0,
      "tokens": 48,
      "tok_s": 48.0,
      "tool_calls": 1,
      "peak_vram_mb": 4516,
      "cached": true
    }
  ],
  "summary": {
    "avg_ttft_first_ms": 644.0,
    "avg_ttft_cached_ms": 419.0,
    "steady_state_tok_s": 48.0,
    "spec_acceptance_rate": 0.0,
    "peak_vram_mb": 4516,
    "end_to_end_ms": 1063.0,
    "config": {
      "cuda_graphs": false,
      "spec_decode": "none",
      "draft_model": null,
      "kv_block_reuse": true,
      "kv_dtype": "int8",
      "ctx_size": 4096
    },
    "timestamp": "2026-06-23T12:00:00+00:00"
  }
}
```

## A/B result structure

```json
{
  "lever_name": "cuda_graphs",
  "delta_ttft_cold_ms": -24.0,
  "delta_ttft_cached_ms": -39.0,
  "delta_tok_s_pct": 8.5,
  "delta_vram_mb": 24,
  "baseline_report": { ... BenchmarkReport ... },
  "treatment_report": { ... BenchmarkReport ... }
}
```

Positive `delta_ttft_*` means *slower* (higher latency). Positive
`delta_tok_s_pct` means *faster* (higher throughput). Positive
`delta_vram_mb` means *more memory used*.