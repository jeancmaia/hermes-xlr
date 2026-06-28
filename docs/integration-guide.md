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
3. `XLRTransport` auto-loaded into Hermes — every request carries plan-derived config
4. Hermes chatting through the local engine with tool calls

---

## Step 1 — Install Hermes Agent

```powershell
git clone https://github.com/jeancmaia/hermes-xlr.git
cd hermes-xlr
.\scripts\install-hermes.ps1
```

This runs the official Hermes installer (Python, Node.js, ripgrep, ffmpeg — all automatic).

---

## Step 2 — Install Hermes-NIM-XLR

```powershell
.\scripts\install-xlr.ps1
```

This single script does everything:

- Fetches the CUDA `llama-server.exe` binary
- Downloads a GGUF model from Hugging Face
- Installs `hermes-nim-xlr` into the Hermes venv
- Drops a `.pth` hook so Hermes auto-registers `XLRTransport`
- Configures Hermes to use `http://127.0.0.1:8080/v1` as its provider

> Pass `-ModelPath C:\path\to\your.gguf` if you already have a model.

---

## Step 3 — Launch the engine

```powershell
.\scripts\start-xlr-engine.ps1
```

The script detects your GPU, generates an execution plan, and launches `llama-server` with all the tuned
settings — GPU layers, context size, KV-cache dtype, CUDA graphs, speculative decoding. When the engine is
healthy, it prints the endpoint URL.

---

## Step 4 — Chat

```powershell
hermes
```

`install-xlr.ps1` already configured Hermes to use the local endpoint and
dropped a `.pth` hook that auto-registers `XLRTransport`. You're ready to chat.

Try a prompt that uses a tool:

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