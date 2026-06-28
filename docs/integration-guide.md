# Hermes Agent + XLR Integration Guide

Wire your `hermes-agent` into the Hermes-NIM-XLR runtime for local GPU inference on Windows.

The XLR runtime provides four layers — detect, plan, start the engine, and transport — that
replace the remote API path with a tuned local engine while keeping the same `ProviderTransport`
contract the agent already speaks.

## Prerequisites

- Windows with an NVIDIA GPU (CUDA 12.4+)
- Python 3.11+
- `hermes-nim-xlr` installed (`uv sync`)
- A GGUF model file staged locally (e.g. `Llama-3.2-3B-Instruct Q4_K_M`)
- `llama-server.exe` (CUDA build) on PATH or at `bin/llama-server.exe`

Optional:
- `nvidia-ml-py` for GPU detection via NVML (`pip install nvidia-ml-py`)

## Quick start

The entire lifecycle is about 30 lines of Python:

```python
from hermes_nim_xlr.mapper import detect, plan
from hermes_nim_xlr.backends import create_backend
from hermes_nim_xlr.transport import XLRTransport
from hermes_nim_xlr.contracts import ExecutionPlan
from agent.transports.types import NormalizedResponse

# 1. Detect the host GPU
host = detect()

# 2. Generate an execution plan
plan: ExecutionPlan = plan(host)

# 3. Start the engine backend
backend = create_backend(
    "llama_cpp",
    binary_path="bin/llama-server.exe",
    model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=plan.placement.gpu_layers,
    ctx_size=plan.target_ctx_tokens,
)
backend.start()

# 4. Create the transport
transport = XLRTransport(plan, endpoint_url=backend.serve_endpoint)

# 5. Build API kwargs (plan-derived: KV cache, CUDA graphs, spec decode)
kwargs = transport.build_kwargs(
    model=plan.model.repo,
    messages=[{"role": "user", "content": "Hello!"}],
)

# 6. Call the engine (via openai.OpenAI)
import openai
client = openai.OpenAI(base_url=backend.serve_endpoint, api_key="ignored")
response = client.chat.completions.create(**kwargs)

# 7. Normalize response → native tool_calls, no XML scraping
normalized: NormalizedResponse = transport.normalize_response(response)
print(normalized.content)

# 8. Stop the engine
backend.stop()
```

## Architecture

```
DETECT ──→ HostCapabilities ──→ PLAN ──→ ExecutionPlan ──→ START backend ──→ XLRTransport ──→ Agent loop
   │                                   │
   ├ GPU (NVML / nvidia-smi)           ├ Model selection (largest fitting)
   ├ OS / WSL                          ├ KV-cache dtype (FP8 / INT8 / FP16)
   ├ CPU RAM                           ├ Layer placement (GPU / shard / offload)
   └ Container runtime                 ├ Speculative decoding strategy
                                       └ Estimate bandwidth + VRAM
```

Every layer is independent — you can skip the mapper and construct an `ExecutionPlan` by hand if
you know your hardware profile.

## Layer reference

### 1. DETECT — `detect()`

```python
from hermes_nim_xlr.mapper.detect import detect

host = detect()
# host.gpus[0].name           → "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
# host.gpus[0].arch           → GpuArch.AMPERE
# host.gpus[0].vram_total_mb  → 6144
# host.gpus[0].supports_fp8   → True/False
# host.os                     → "Windows"
# host.is_wsl                 → False
```

Probes GPUs via NVML (primary) or `nvidia-smi` (fallback). Returns an immutable
`HostCapabilities` frozen dataclass. Bandwidth figures come from a static lookup table
by GPU name.

Detection is optional — you can construct `HostCapabilities` manually for testing or
offline planning:

```python
from hermes_nim_xlr.contracts import HostCapabilities, GpuCapabilities, GpuArch

host = HostCapabilities(
    os="Windows",
    is_wsl=False,
    cpu_ram_gb=32.0,
    container_runtime=None,
    has_nvidia_container_toolkit=False,
    gpus=(
        GpuCapabilities(
            index=0,
            name="NVIDIA GeForce RTX 3050 Laptop GPU",
            arch=GpuArch.AMPERE,
            compute_capability=(8, 6),
            vram_total_mb=6144,
            vram_free_mb=5120,
            mem_bandwidth_gbs=192.0,
            pcie_bandwidth_gbs=8.0,
            supports_fp8=False,
            supports_int8=True,
            supports_cuda_graphs=True,
            driver_version="572.40",
        ),
    ),
)
```

### 2. PLAN — `plan()`

```python
from hermes_nim_xlr.mapper import plan

execution_plan = plan(host)

# Key fields:
# execution_plan.model.repo               → "QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF"
# execution_plan.model.weight_quant        → WeightQuant.INT4_AWQ
# execution_plan.kv.dtype                  → KvDtype.INT8
# execution_plan.levers.cuda_graphs        → True
# execution_plan.levers.spec_decode        → SpecDecode.NGRAM
# execution_plan.placement.gpu_layers      → 28 (fully resident)
# execution_plan.backend.kind              → BackendKind.LLAMACPP
# execution_plan.backend.serve_endpoint    → "http://127.0.0.1:8080/v1"
# execution_plan.est_vram_mb               → 2800
# execution_plan.est_decode_tok_s          → (30, 50)
```

The planner is deterministic — identical input always yields the same plan. Every branch
appends a human-readable rationale line.

Two objectives:
- `THROUGHPUT_FIRST` — avoids CPU offload; prefers smaller models fully resident
- `QUALITY_FIRST` — accepts a PCIe-cliff penalty to run a larger model

```python
execution_plan = plan(host, objective=Objective.QUALITY_FIRST)
```

You can also build an `ExecutionPlan` by hand for complete control:

```python
from hermes_nim_xlr.contracts import (
    ExecutionPlan, ModelChoice, WeightQuant, KvCacheConfig, KvDtype,
    DecodeLevers, SpecDecode, LayerPlacement, BackendChoice, BackendKind,
    BringUp, Objective,
)

execution_plan = ExecutionPlan(
    objective=Objective.THROUGHPUT_FIRST,
    model=ModelChoice(
        repo="QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF",
        params_b=3.2,
        weight_quant=WeightQuant.INT4_AWQ,
        est_weight_mb=1800,
        n_layers=28,
        kv_heads=8,
        head_dim=128,
        max_context_tokens=8192,
    ),
    placement=LayerPlacement(
        total_layers=28, gpu_layers=28, cpu_offload_layers=0,
        tensor_parallel=1, pipeline_parallel=1,
        note="fully GPU-resident",
    ),
    kv=KvCacheConfig(
        dtype=KvDtype.INT8, enable_block_reuse=True,
        free_gpu_memory_fraction=0.35, host_cache_size_bytes=0,
    ),
    levers=DecodeLevers(
        cuda_graphs=True, spec_decode=SpecDecode.NGRAM, draft_model=None,
    ),
    backend=BackendChoice(
        kind=BackendKind.LLAMACPP, bring_up=BringUp.NATIVE_WINDOWS,
        serve_endpoint="http://127.0.0.1:8080/v1",
    ),
    target_ctx_tokens=4096,
    est_vram_mb=2800,
    est_decode_tok_s=(30, 50),
    rationale=("INT4 weights for ~5000 MB budget", "fully GPU-resident"),
    warnings=(),
)
```

### 3. START — engine backend

```python
from hermes_nim_xlr.backends import create_backend

backend = create_backend(
    "llama_cpp",
    binary_path="bin/llama-server.exe",
    model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,                       # -1 = all layers
    ctx_size=4096,
    cuda_graphs=True,                       # --cuda-graphs
    speculative_ngram=32,                   # --speculative-ngram 32
    kv_cache_type_k="q8_0",                # --cache-type-k q8_0
    kv_cache_type_v="q8_0",                # --cache-type-v q8_0
)
backend.start()

# endpoint = backend.serve_endpoint  → "http://127.0.0.1:8080/v1"
# info    = backend.engine_info       → {"version": "b9763", "cuda": 1, ...}
```

`create_backend()` looks up the registered class by kind name (`"llama_cpp"`). The
backend manages the full process lifecycle — spawn, health-poll, and graceful stop.

Lifecycle methods:
- `start()` — assert version match, spawn process, poll until healthy
- `stop()` — graceful shutdown with timeout
- `health()` — lightweight check (GET /v1/models)

Available factory functions:
- `register(kind, cls)` — register a custom backend class
- `create_backend(kind, **kwargs)` — instantiate by kind name

```python
from hermes_nim_xlr.backends import register

# Register a custom backend
register("my_engine", MyEngineBackend)
backend = create_backend("my_engine", ...)
```

### 4. TRANSPORT — XLRTransport

`XLRTransport` is a `ProviderTransport` for the `chat_completions` api_mode. It
implements the four-method contract:

| Method | Purpose |
|--------|---------|
| `api_mode` | Returns `"chat_completions"` |
| `convert_messages(messages)` | Strips internal scaffolding keys; pass-through otherwise |
| `convert_tools(tools)` | Identity — tools already in OpenAI format |
| `build_kwargs(model, messages, tools, **params)` | Wires plan-derived config into `extra_body` |
| `normalize_response(response)` | Extracts native `tool_calls`; returns `NormalizedResponse` |

```python
from hermes_nim_xlr.transport import XLRTransport

transport = XLRTransport(
    execution_plan=execution_plan,
    endpoint_url="http://127.0.0.1:8080/v1",
    persist_dir=None,              # optional: async persistence to JSON
)
```

**Critical invariant:** The transport injects zero per-turn entropy. No timestamps,
UUIDs, or random seeds appear in the request body — the first N bytes are stable
across turns, enabling prefix caching.

#### ProviderTransport contract (for AIAgent integration)

To use XLRTransport with `hermes-agent`'s `AIAgent`:

```python
from agent.transports import register_transport

# Register XLRTransport so AIAgent._get_transport() picks it up
register_transport("chat_completions", XLRTransport)
```

**Note:** `get_transport()` instantiates transports with no arguments (`cls()`), but
`XLRTransport.__init__` requires `execution_plan` and `endpoint_url`. For full AIAgent
integration, create a factory or partial that provides these:

```python
from functools import partial
from agent.transports import register_transport

# You would need a wrapper that captures plan + endpoint
# and passes them to XLRTransport on construction.
```

The simplest approach for a `hermes-agent` loop is to use `XLRTransport` directly to
build kwargs and normalize responses, rather than routing through the agent's
auto-discovery path. The example script in `docs/examples/` demonstrates this pattern.

## Running the example

```powershell
# Set paths to your binaries and model
$env:XLR_BINARY_PATH = "C:\tools\llama-server.exe"
$env:XLR_MODEL_PATH = "C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Run the example
uv run python docs/examples/xlr_hermes_integration.py
```

Expected output (approximate):

```
=== Phase 1: DETECT ===
  OS:   Windows
  GPU:  [NVIDIA GeForce RTX 3050 6GB Laptop GPU]

=== Phase 2: PLAN ===
  Model:    QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF
  Backend:  llama_cpp
  Endpoint: http://127.0.0.1:8080/v1
  VRAM est: 2800 MiB
  Levers:   CUDA graphs=True, spec=ngram

=== Phase 3: START ===
  Starting backend (binary=bin/llama-server.exe)...
  Endpoint healthy at http://127.0.0.1:8080/v1

=== Phase 4: TRANSPORT ===
  XLRTransport ready: api_mode=chat_completions

=== Phase 5: RUN ===
  Tokens: 179 prompt + 17 completion
  Tool call: get_weather({"city": "Paris"})
    ...
=== Done — 3 turns completed ===
```

## ProviderTransport contract mapping

`XLRTransport` maps to the hermes-agent `ProviderTransport` ABC as follows:

| ABC method | XLRTransport behaviour | Notes |
|---|---|---|
| `api_mode` | `"chat_completions"` | Matches the engine's OpenAI-compatible API |
| `convert_messages` | Strip internal keys | `_`-prefixed, `tool_name`, `timestamp`, Codex fields |
| `convert_tools` | Identity | Tools already in OpenAI `functions` / `tools` format |
| `build_kwargs` | Wire plan into `extra_body` | KV-cache, CUDA graphs, spec-decode, layers, ctx size |
| `normalize_response` | Extract native `tool_calls` | Structured `function.name` + `function.arguments`, no regex |
| `validate_response` | Not overridden | Default returns `True` |
| `extract_cache_stats` | Not overridden | Default returns `None` |
| `map_finish_reason` | Not overridden | Default returns raw reason |

## Invariants

1. **Prefix stability** — no per-turn entropy in the prompt path (`extra_body` is
   plan-derived, never contains timestamps/UUIDs). The first N bytes of the request
   body are identical across turns with the same system prompt.

2. **Native tool calls** — `normalize_response` reads the provider's structured
   `tool_calls` field. No regex or XML scraping of text content.

3. **Transport is stateless** — caching, retry, credentials, and streaming live on
   the agent, not in the transport.

4. **Latency hiding is additive** — async persistence fires on a background thread
   and does not block `normalize_response` return.

## CLI quick reference

```powershell
# Probe the host and print the execution plan as JSON
uv run xlr plan

# Run the benchmark suite (requires a running engine)
uv run xlr benchmark run --endpoint http://127.0.0.1:8080/v1
```

## See also

- `docs/examples/xlr_hermes_integration.py` — working multi-turn example
- `docs/s2-bring-up.md` — CUDA llama.cpp bring-up notes
- `docs/s0.5-verdict.md` — spike go/no-go and capacity findings
- `docs/ab-protocol.md` — A/B measurement protocol
- `tests/test_transport.py` — prefix-stability and tool-call round-trip tests
- `ref/hermes-architecture-presentation.md` — framework architecture reference