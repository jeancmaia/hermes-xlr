"""Launch an XLR-tuned llama.cpp engine for Hermes Agent.

Usage
-----
    $env:XLR_BINARY_PATH = "C:\tools\llama-server.exe"
    $env:XLR_MODEL_PATH = "C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    uv run python docs/examples/xlr_hermes_integration.py

The script detects your GPU, generates an execution plan, launches a tuned
llama-server, and prints the endpoint URL. Point Hermes at that URL:

    hermes model
    → Custom endpoint
    → http://127.0.0.1:8080/v1
    → (no API key)
    → (auto-detect model)

Then start chatting:  hermes

Press Ctrl+C to stop the engine.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

from hermes_nim_xlr.backends import create_backend
from hermes_nim_xlr.mapper import detect, plan

BINARY_PATH = os.environ.get(
    "XLR_BINARY_PATH",
    str(Path("bin/llama-server.exe").resolve()),
)
MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.environ.get(
    "XLR_MODEL_PATH",
    str(Path.home() / ".cache" / "hermes" / "models" / MODEL_FILE),
)


def main() -> int:
    if not Path(BINARY_PATH).exists():
        print(f"ERROR: llama-server not found at {BINARY_PATH}")
        print("Set XLR_BINARY_PATH to your llama-server.exe location.")
        return 1
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Set XLR_MODEL_PATH to your GGUF model file.")
        return 1

    print("=== DETECT ===")
    host = detect()
    print(f"  OS:   {host.os}")
    print(f"  GPU:  {[g.name for g in host.gpus]}")
    if not host.gpus:
        print("  WARNING: no GPU detected — engine will fall back to CPU")
    print()

    print("=== PLAN ===")
    p = plan(host)
    print(f"  Model:       {p.model.repo}")
    print(f"  Backend:     {p.backend.kind.value}")
    print(f"  VRAM est:    {p.est_vram_mb} MiB")
    print(f"  Context:     {p.target_ctx_tokens} tokens")
    print(f"  KV dtype:    {p.kv.dtype.value}")
    print(f"  CUDA graphs: {p.levers.cuda_graphs}")
    print(f"  Spec decode: {p.levers.spec_decode.value}")
    print(f"  GPU layers:  {p.placement.gpu_layers}/{p.placement.total_layers}")
    for note in p.rationale:
        print(f"    {note}")
    print()

    print("=== START ===")
    backend = create_backend(
        "llama_cpp",
        binary_path=BINARY_PATH,
        model_path=MODEL_PATH,
        n_gpu_layers=p.placement.gpu_layers,
        ctx_size=p.target_ctx_tokens,
        cuda_graphs=p.levers.cuda_graphs,
        kv_cache_type_k=p.kv.cache_type_k,
        kv_cache_type_v=p.kv.cache_type_v,
    )

    print(f"  Launching {BINARY_PATH}...")
    backend.start()
    print(f"  Endpoint: {backend.serve_endpoint}")
    print(f"  Engine:   {json.dumps(backend.engine_info)}")
    print()

    print("=== READY ===")
    print()
    print("  Hermes Agent is ready to connect.")
    print()
    print("  Run in another terminal:")
    print()
    print("    hermes model")
    print("    → Custom endpoint")
    print(f"    → {backend.serve_endpoint}")
    print("    → (no API key)")
    print("    → (auto-detect model)")
    print()
    print("    hermes")
    print()
    print("  Press Ctrl+C to stop the engine.")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n=== SHUTDOWN ===")
        backend.stop()
        print("  Engine stopped.")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    raise SystemExit(main())
