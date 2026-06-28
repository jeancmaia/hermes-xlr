# Hermes Agent + XLR Integration Guide

Run [Hermes Agent](https://hermes-agent.nousresearch.com) locally on your NVIDIA GPU,
tuned to the metal by the Hermes-NIM-XLR runtime — no cloud API keys, no per-token
cost, fully private.

XLR detects your hardware, generates an optimized execution plan, and launches a
tuned `llama.cpp` engine. Hermes Agent then connects to that engine as a
"Custom endpoint" provider and runs as it would against any cloud LLM — with
tools, sessions, and multi-turn conversations.

## What you'll get

By the end of this guide you'll have:

1. Hermes Agent installed and configured
2. An XLR-tuned `llama-server` running on your GPU
3. Hermes chatting through the local engine with tool calls

---

## Step 1 — Install Hermes Agent

Install Hermes Agent the standard way. On Windows (PowerShell):

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Reload your shell and verify:

```powershell
hermes --version
```

> The installer handles Python, Node.js, ripgrep, ffmpeg, and the virtualenv
> automatically. See the [official installation guide][hermes-install] for
> details.

[hermes-install]: https://hermes-agent.nousresearch.com/docs/getting-started/installation

---

## Step 2 — Install Hermes-NIM-XLR

XLR is a Python package that lives alongside Hermes. Clone the repo and install
it in editable mode:

```powershell
git clone https://github.com/jeancmaia/hermes-xlr.git
cd hermes-xlr
uv sync
```

Verify the CLI works:

```powershell
uv run xlr plan
```

You should see a JSON execution plan with your GPU's name, VRAM budget, and
selected model. If `xlr plan` errors with "no GPU detected", make sure
`nvidia-smi` is on your PATH (it ships with the NVIDIA driver).

---

## Step 3 — Stage a model

XLR needs a GGUF model file on local disk. The planner selects the best model
for your VRAM budget, but you need to download it first.

For a smaller GPU (6–8 GB VRAM), a good starting model is
**Llama-3.2-3B-Instruct Q4_K_M** (~2 GB on disk):

```powershell
# Create a models directory
mkdir models

# Download from Hugging Face (pick one)
# Option A: huggingface-cli
huggingface-cli download QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF Meta-Llama-3.2-3B-Instruct.Q4_K_M.gguf --local-dir models

# Option B: direct URL with curl
curl -L -o models/Llama-3.2-3B-Instruct-Q4_K_M.gguf https://huggingface.co/QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF/resolve/main/Meta-Llama-3.2-3B-Instruct.Q4_K_M.gguf
```

> **Context length:** Hermes requires at least 64K tokens of context. XLR
> configures the engine context size automatically from the execution plan —
> you don't need to set `--ctx-size` manually.

---

## Step 4 — Get a CUDA llama-server binary

XLR drives `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp))
as its engine backend. You need a CUDA-enabled build.

**Option A: Download a prebuilt release**

Grab the latest Windows CUDA release from the
[llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases). Look
for `llama-*-bin-win-cuda-cu12*.zip`. Extract `llama-server.exe` and place it
either:

- In `bin/llama-server.exe` inside the hermes-xlr repo, or
- Anywhere on your `PATH`

**Option B: Build from source**

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
# Binary: build/bin/Release/llama-server.exe
```

> **Tool calling requires `--jinja`:** XLR passes this flag automatically. The
> model must also support native tool calling — Llama 3.x, Qwen 2.5, and Hermes
> 2/3 all work. See the [llama.cpp function calling docs][llamacpp-tools] for
> the full list.

[llamacpp-tools]: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md

Verify the binary works:

```powershell
llama-server.exe --version
```

---

## Step 5 — Launch the engine with XLR

This is where XLR earns its keep. Instead of manually guessing flags, XLR
detects your GPU, picks the right quantization, KV-cache dtype, context size,
and decode levers — then launches `llama-server` with the optimal config.

### One-liner via the XLR CLI

```powershell
uv run xlr plan
```

This prints the execution plan as JSON. Review it to see what XLR chose for
your hardware. Example output:

```json
{
  "model": {
    "repo": "QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF",
    "weight_quant": "int4_awq",
    "est_weight_mb": 1800
  },
  "placement": { "gpu_layers": 28, "note": "fully GPU-resident" },
  "kv": { "dtype": "int8", "enable_block_reuse": true },
  "levers": { "cuda_graphs": true, "spec_decode": "ngram" },
  "backend": { "serve_endpoint": "http://127.0.0.1:8080/v1" },
  "target_ctx_tokens": 65536,
  "est_vram_mb": 2800
}
```

### Start the engine

Use the plan to launch the backend. The easiest way is the example script:

```powershell
# Set paths to your binary and model
$env:XLR_BINARY_PATH = "C:\tools\llama-server.exe"
$env:XLR_MODEL_PATH = "C:\path\to\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Launch — XLR detects, plans, and starts the tuned engine
uv run python docs/examples/xlr_hermes_integration.py
```

Or start the backend programmatically:

```python
from hermes_nim_xlr.mapper import detect, plan
from hermes_nim_xlr.backends import create_backend

host = detect()
p = plan(host)

backend = create_backend(
    "llama_cpp",
    binary_path="C:/tools/llama-server.exe",
    model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=p.placement.gpu_layers,
    ctx_size=p.target_ctx_tokens,
    cuda_graphs=p.levers.cuda_graphs,
    kv_cache_type_k=p.kv.cache_type_k,
    kv_cache_type_v=p.kv.cache_type_v,
)
backend.start()

print(f"Engine ready at {backend.serve_endpoint}")
# → http://127.0.0.1:8080/v1
```

The engine is now serving an OpenAI-compatible API at
`http://127.0.0.1:8080/v1`.

---

## Step 6 — Point Hermes at the engine

Now tell Hermes to use the local engine as its LLM provider. Hermes calls this
a "Custom endpoint" — any OpenAI-compatible API works.

### Interactive setup (recommended)

```powershell
hermes model
```

Select **"Custom endpoint (self-hosted / VLLM / etc.)"** and enter:

| Prompt | Value |
|--------|-------|
| API base URL | `http://127.0.0.1:8080/v1` |
| API key | *(leave empty — local server doesn't need one)* |
| Model name | *(press Enter to auto-detect, or type the GGUF name)* |

### Manual config

Alternatively, edit `~/.hermes/config.yaml` directly:

```yaml
model:
  default: Meta-Llama-3.2-3B-Instruct-Q4_K_M
  provider: custom
  base_url: http://127.0.0.1:8080/v1
  api_key: local
```

Or use `hermes config set`:

```powershell
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8080/v1
hermes config set model.default Meta-Llama-3.2-3B-Instruct-Q4_K_M
```

---

## Step 7 — Chat

```powershell
hermes
```

You'll see the Hermes banner with your local model loaded. Try a prompt that
uses a tool:

```
> What files are in the current directory?
```

Hermes will call the terminal tool, read the directory, and respond — all
running through your local GPU with XLR's tuned config.

### Verify tool calls are native

Tool calls should come back as structured `tool_calls` in the response, not as
text. If you see raw JSON in the assistant's message instead of tool execution,
the engine isn't running with `--jinja`. XLR handles this automatically — but
if you launched `llama-server` manually, add `--jinja` to the command.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Hermes Agent                                           │
│  (tools, sessions, skills, messaging)                   │
│         │                                                │
│         │  OpenAI-compatible HTTP                        │
│         ▼                                                │
│  http://127.0.0.1:8080/v1                               │
│         │                                                │
│         │  XLRTransport (ProviderTransport)              │
│         │  • build_kwargs() ← plan-derived config        │
│         │  • normalize_response() ← native tool_calls    │
│         │  • zero per-turn entropy (prefix-cache safe)   │
│         │                                                │
│         ▼                                                │
│  LlamaCppBackend (llama-server.exe)                      │
│  • CUDA, KV-quant, CUDA graphs, n-gram spec-decode       │
│  • All layers GPU-resident (or CPU-offloaded if tight)   │
└─────────────────────────────────────────────────────────┘
```

### What XLR adds vs. a plain llama-server

| Setting | Manual | XLR |
|---------|--------|-----|
| GPU layer count | Guess `-ngl 99` | Plan-driven: exact count for your VRAM |
| Context size | Guess `-c 65536` | Plan-driven: fits your VRAM budget |
| KV-cache dtype | Manual `--cache-type-k q8_0` | Plan-driven: FP8 on Ada+, INT8 on Ampere |
| CUDA graphs | Manual `--cuda-graphs` | Plan-driven: enabled if GPU supports it |
| Speculative decoding | Manual `--speculative-ngram` | Plan-driven: n-gram when budget allows |
| Prefix cache | Manual `--cache-prompt` | Plan-driven: block reuse always on |
| Model selection | Manual | Catalog-driven: largest model that fits |

### Invariants

1. **Prefix stability** — XLRTransport injects zero per-turn entropy into the
   request body. No timestamps, UUIDs, or random seeds. The first N bytes are
   byte-identical across turns with the same system prompt, enabling the
   engine's prefix cache to fire.

2. **Native tool calls** — `normalize_response()` reads the provider's
   structured `tool_calls` field. No regex or XML scraping of text content.

3. **Transport is stateless translation** — caching, retry, credentials, and
   streaming live on the Hermes agent, not in the transport. XLRTransport only
   translates request/response formats.

---

## Troubleshooting

### "Connection refused" at `http://127.0.0.1:8080/v1`

The engine isn't running. Start it with the example script or
`backend.start()`. Verify with:

```powershell
curl http://127.0.0.1:8080/v1/models
```

### "context length too small" error in Hermes

Hermes requires at least 64K tokens. XLR sets this from the plan's
`target_ctx_tokens` — if your VRAM is very tight, the planner may reduce it.
Check `uv run xlr plan` and look for `target_ctx_tokens`. If it's below 65536,
you need a smaller model or more VRAM.

### Tool calls appear as text instead of executing

The engine needs `--jinja` for native tool calling. XLR passes this
automatically; if you launched `llama-server` manually, add `--jinja` to the
command. Verify with:

```powershell
curl http://127.0.0.1:8080/props
```

The `chat_template` field should be present.

### Model loads but produces garbage

Make sure the model supports tool calling. Llama 3.x, Qwen 2.5, and Hermes 2/3
all work. Nemotron-Mini-4B does **not** — it emits `<toolcall>` XML in text
instead of structured `tool_calls`.



---

## Reference

### CLI quick reference

```powershell
# XLR
uv run xlr plan                                          # probe + emit plan as JSON
uv run xlr benchmark run --endpoint http://127.0.0.1:8080/v1  # A/B benchmarks

# Hermes
hermes model          # configure provider (choose "Custom endpoint")
hermes                # start chatting
hermes --tui          # modern TUI
hermes doctor         # diagnose issues
```

### API quick reference

```python
# Detect → Plan → Start → Transport
from hermes_nim_xlr.mapper import detect, plan
from hermes_nim_xlr.backends import create_backend
from hermes_nim_xlr.transport import XLRTransport

host = detect()
p = plan(host)

backend = create_backend("llama_cpp", binary_path=..., model_path=...)
backend.start()

transport = XLRTransport(p, endpoint_url=backend.serve_endpoint)
kwargs = transport.build_kwargs(model=p.model.repo, messages=messages, tools=tools)
# → pass kwargs to openai.OpenAI().chat.completions.create(**kwargs)
```

### Files

- `docs/examples/xlr_hermes_integration.py` — working example script
- `docs/s2-bring-up.md` — CUDA llama.cpp bring-up notes
- `docs/s0.5-verdict.md` — spike go/no-go and capacity findings
- `docs/ab-protocol.md` — A/B measurement protocol
- `tests/test_transport.py` — prefix-stability and tool-call round-trip tests
- `ref/hermes-architecture-presentation.md` — framework architecture reference