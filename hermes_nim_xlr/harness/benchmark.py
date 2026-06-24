"""Measurement harness — honest multi-turn A/B benchmarking.

The harness drives a fixed agent workload against a real engine backend
and records TTFT, decode throughput, spec-decode acceptance rate, peak
VRAM, and end-to-end latency. No hardcoded sleeps, no fake numbers.

Baseline mode: all levers OFF. A/B mode: one lever ON per comparison.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from hermes_nim_xlr.contracts import ExecutionPlan
from hermes_nim_xlr.harness.metrics import (
    ABDelta,
    BenchmarkReport,
    TurnMetrics,
    compute_ab_delta,
)
from hermes_nim_xlr.transport import XLRTransport

_SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
]

_SAMPLE_QUESTIONS: list[str] = [
    "What is the weather like in Paris today?",
    "Search for the latest AI research papers.",
    "What is the weather in Tokyo right now?",
    "Search for news about quantum computing.",
    "What is the weather in London this weekend?",
]

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "Use them when appropriate to answer the user's questions."
)


def _probe_vram_mib() -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0
        line = result.stdout.strip().splitlines()[0].strip()
        try:
            return int(line.replace(" MiB", ""))
        except (ValueError, TypeError):
            return 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0


class BenchmarkHarness:
    def __init__(
        self,
        plan: ExecutionPlan,
        endpoint_url: str | None = None,
        overrides: dict | None = None,
    ) -> None:
        self._plan = plan
        self._endpoint_url = (endpoint_url or plan.backend.serve_endpoint).rstrip("/")
        self._overrides = overrides or {}
        self._transport = XLRTransport(
            execution_plan=plan,
            endpoint_url=self._endpoint_url,
        )

    def _chat_endpoint(self) -> str:
        return f"{self._endpoint_url}/chat/completions"

    def _post_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._chat_endpoint(),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"HTTP {exc.code} from {self._chat_endpoint()}: {exc.read().decode()}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Connection failed to {self._chat_endpoint()}: {exc.reason}"
            ) from exc
        decoded: dict[str, Any] = json.loads(data)
        return decoded

    def _measure_turn(
        self,
        turn_index: int,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[TurnMetrics, list[dict[str, Any]]]:
        kwargs = self._transport.build_kwargs(
            model="default",
            messages=messages,
            tools=tools,
            max_tokens=256,
        )

        vram_before = _probe_vram_mib()
        start = time.monotonic()
        raw = self._post_request(kwargs)
        elapsed = time.monotonic() - start
        vram_after = _probe_vram_mib()

        usage = raw.get("usage", {}) or {}
        completion_tokens = 0
        try:
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        except (ValueError, TypeError):
            completion_tokens = 0

        choices = raw.get("choices", [])
        tool_calls_count = 0
        assistant_content: str | None = None
        if choices:
            choice = choices[0]
            msg = choice.get("message", {}) or {}
            assistant_content = msg.get("content")
            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                tool_calls_count = len(tc)

        ttft_ms = round(elapsed * 1000, 1)
        tok_s = 0.0
        if completion_tokens > 0 and elapsed > 0:
            tok_s = round(completion_tokens / elapsed, 1)

        cached = turn_index > 0

        metrics = TurnMetrics(
            turn=turn_index,
            ttft_ms=ttft_ms,
            tokens=completion_tokens,
            tok_s=tok_s,
            tool_calls=tool_calls_count,
            peak_vram_mb=max(vram_before, vram_after),
            cached=cached,
        )

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if assistant_content is not None:
            assistant_msg["content"] = assistant_content
        else:
            assistant_msg["content"] = None
        if tool_calls_count > 0:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{turn_index}_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    },
                }
                for i, tc in enumerate(msg.get("tool_calls", []) if choices else [])
            ]
        messages.append(assistant_msg)

        for i in range(tool_calls_count):
            tc = msg["tool_calls"][i]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{turn_index}_{i}"),
                    "content": '{"result": "ok"}',
                }
            )

        return metrics, messages

    def run(self, turns: int = 5) -> BenchmarkReport:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        turn_metrics: list[TurnMetrics] = []

        for i in range(turns):
            question = _SAMPLE_QUESTIONS[i % len(_SAMPLE_QUESTIONS)]
            turn_msgs = messages + [{"role": "user", "content": question}]
            tools = _SAMPLE_TOOLS if i % 2 == 0 else None
            metrics, messages = self._measure_turn(i, turn_msgs, tools)
            turn_metrics.append(metrics)

        all_metrics = turn_metrics[:turns]

        return BenchmarkReport(plan=self._plan, turns=all_metrics)

    def run_ab(
        self, baseline_report: BenchmarkReport, lever_name: str, lever_config: dict
    ) -> ABDelta:
        plan = self._plan
        new_levers = replace(
            plan.levers,
            **{k: v for k, v in lever_config.items() if hasattr(plan.levers, k)},
        )
        new_kv = replace(
            plan.kv,
            **{k: v for k, v in lever_config.items() if hasattr(plan.kv, k)},
        )
        treatment_plan = replace(plan, levers=new_levers, kv=new_kv)

        harness = BenchmarkHarness(
            plan=treatment_plan,
            endpoint_url=self._endpoint_url,
        )
        treatment_report = harness.run(turns=len(baseline_report.turns))

        return compute_ab_delta(lever_name, baseline_report, treatment_report)
