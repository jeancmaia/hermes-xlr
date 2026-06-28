"""Working example: plug hermes-agent into the Hermes-NIM-XLR runtime.

Usage
-----
    uv run python docs/examples/xlr_hermes_integration.py

Prerequisites
-------------
- A CUDA-capable NVIDIA GPU
- ``llama-server.exe`` on PATH or at ``bin/llama-server.exe``
- A GGUF model file (e.g. Llama-3.2-3B-Instruct Q4_K_M)
- ``hermes-nim-xlr`` installed with ``uv sync``

The script walks through the full lifecycle:
    1. DETECT — probe the host GPU
    2. PLAN  — generate an execution plan
    3. START — launch the engine backend
    4. TRANSPORT — wire XLRTransport for the agent loop
    5. RUN   — drive a multi-turn conversation with tool calls
"""

import json
import os
import time
from pathlib import Path

import openai
from hermes_nim_xlr.backends import create_backend
from hermes_nim_xlr.mapper import detect, plan
from hermes_nim_xlr.transport import XLRTransport

# ---------------------------------------------------------------------------
# Configuration — adjust these to your local setup
# ---------------------------------------------------------------------------

# Path to the llama-server CUDA binary
BINARY_PATH = os.environ.get(
    "XLR_BINARY_PATH",
    str(Path("bin/llama-server.exe").resolve()),
)
# Path to a GGUF model file
MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.environ.get(
    "XLR_MODEL_PATH",
    str(Path.home() / ".cache" / "hermes" / "models" / MODEL_FILE),
)

# How many agent turns to run
TURNS = 3


def main() -> int:
    # ------------------------------------------------------------------
    # Phase 1: DETECT — probe GPU, OS, memory
    # ------------------------------------------------------------------
    print("=== Phase 1: DETECT ===")
    host = detect()
    print(f"  OS:   {host.os}")
    print(f"  GPU:  {[g.name for g in host.gpus]}")
    if not host.gpus:
        print("  WARNING: no GPU detected — engine may fall back to CPU")
    print()

    # ------------------------------------------------------------------
    # Phase 2: PLAN — generate the deterministic execution plan
    # ------------------------------------------------------------------
    print("=== Phase 2: PLAN ===")
    execution_plan = plan(host)
    print(f"  Model:    {execution_plan.model.repo}")
    print(f"  Backend:  {execution_plan.backend.kind.value}")
    print(f"  Endpoint: {execution_plan.backend.serve_endpoint}")
    print(f"  VRAM est: {execution_plan.est_vram_mb} MiB")
    print(
        f"  Levers:   CUDA graphs={execution_plan.levers.cuda_graphs}, "
        f"spec={execution_plan.levers.spec_decode.value}"
    )
    for note in execution_plan.rationale:
        print(f"    {note}")
    print()

    # ------------------------------------------------------------------
    # Phase 3: START — launch the engine backend
    # ------------------------------------------------------------------
    print("=== Phase 3: START ===")
    backend = create_backend(
        "llama_cpp",
        binary_path=BINARY_PATH,
        model_path=MODEL_PATH,
        n_gpu_layers=execution_plan.placement.gpu_layers,
        ctx_size=execution_plan.target_ctx_tokens,
        cuda_graphs=execution_plan.levers.cuda_graphs,
        cache_type_k=execution_plan.kv.cache_type_k,
        cache_type_v=execution_plan.kv.cache_type_v,
    )

    try:
        print(f"  Starting backend (binary={BINARY_PATH})...")
        backend.start()
        print(f"  Endpoint healthy at {backend.serve_endpoint}")
        print()

        # ------------------------------------------------------------------
        # Phase 4: TRANSPORT — wire XLRTransport
        # ------------------------------------------------------------------
        print("=== Phase 4: TRANSPORT ===")

        endpoint = execution_plan.backend.serve_endpoint
        transport = XLRTransport(
            execution_plan=execution_plan,
            endpoint_url=endpoint,
        )
        print(f"  XLRTransport ready: api_mode={transport.api_mode}")
        print()

        # Build an OpenAI-compatible client pointed at the engine
        client = openai.OpenAI(
            base_url=endpoint,
            api_key="not-needed",  # local engines ignore the key
        )

        # ------------------------------------------------------------------
        # Phase 5: RUN — multi-turn agent conversation with tool calls
        # ------------------------------------------------------------------
        print("=== Phase 5: RUN ===")

        # Define a weather tool the model can call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name, e.g. Paris",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to weather data. "
                    "Use the get_weather tool when asked about weather."
                ),
            },
        ]

        for turn in range(TURNS):
            print(f"\n--- Turn {turn + 1} ---")

            if turn == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "What is the weather in Paris?",
                    }
                )

            # Build kwargs via XLRTransport (plan-derived config)
            kwargs = transport.build_kwargs(
                model=execution_plan.model.repo,
                messages=messages,
                tools=tools,
            )

            # Send to the engine
            response = client.chat.completions.create(**kwargs)

            # Normalize via XLRTransport (extract native tool_calls)
            normalized = transport.normalize_response(response)
            print(f"  Finish reason: {normalized.finish_reason}")
            if normalized.usage:
                print(
                    f"  Tokens: {normalized.usage.prompt_tokens} prompt + "
                    f"{normalized.usage.completion_tokens} completion"
                )

            if normalized.tool_calls:
                for tc in normalized.tool_calls:
                    print(f"  Tool call: {tc.name}({tc.arguments})")
                    # Simulate tool execution
                    if tc.name == "get_weather":
                        args = json.loads(tc.arguments)
                        city = args.get("city", "Unknown")
                        tool_result = json.dumps(
                            {
                                "temperature": 22,
                                "condition": "sunny",
                                "city": city,
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.name,
                                            "arguments": tc.arguments,
                                        },
                                    }
                                ],
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            }
                        )
            else:
                print(f"  Response: {normalized.content}")
                messages.append(
                    {
                        "role": "assistant",
                        "content": normalized.content,
                    }
                )

            # Add a follow-up question for the next turn
            if turn == 0:
                time.sleep(0.5)

        print(f"\n=== Done — {TURNS} turns completed ===")

    finally:
        # ------------------------------------------------------------------
        # Cleanup: stop the engine backend
        # ------------------------------------------------------------------
        print("\n=== Shutdown ===")
        backend.stop()
        print("  Backend stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
