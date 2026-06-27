"""XLRTransport: the stateless translation seam over the pluggable engine backend.

XLRTransport is a ``ProviderTransport`` (``agent.transports.base``) for the
``chat_completions`` api_mode, targeting a local inference engine
(llama.cpp / TensorRT-LLM / MLX). Messages are already in OpenAI format, so
``convert_messages`` and ``convert_tools`` are near-identity. The transport
wires endpoint URL and inference parameters from the ``ExecutionPlan`` into
the API kwargs.

Critical invariant (HER-18 / AGENTS.md section-1): **zero per-turn entropy**. No
timestamps, no UUIDs, no non-deterministic fields are injected ahead of the
volatile prompt tier — the first N bytes of the request body must be
identical across turns with identical system prompts.
"""

from __future__ import annotations

from typing import Any

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

from hermes_nim_xlr.contracts import ExecutionPlan


class XLRTransport(ProviderTransport):
    """ProviderTransport for a local engine backend.

    Accepts an ``ExecutionPlan`` and endpoint URL at init time. Every inference
    parameter derives from the plan — no per-turn entropy is injected.

    Args:
        execution_plan: Frozen plan from the capability mapper. Controls
            model choice, KV-cache config, decode levers, and placement.
        endpoint_url: Base URL of the engine's OpenAI-compatible endpoint
            (e.g. ``"http://127.0.0.1:8080/v1"``).
    """

    def __init__(
        self,
        execution_plan: ExecutionPlan,
        endpoint_url: str,
    ) -> None:
        self._plan = execution_plan
        self._endpoint_url = endpoint_url.rstrip("/")

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    # ------------------------------------------------------------------
    # Convert methods — near-identity for OpenAI-format messages/tools
    # ------------------------------------------------------------------

    def convert_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Pass-through for OpenAI-format messages.

        Strips internal scaffolding keys that strict OpenAI-compatible
        providers reject (``_``-prefixed markers, ``tool_name``,
        ``timestamp``, Codex fields).
        """
        sanitized: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                sanitized.append(msg)
                continue
            cleaned = {
                k: v
                for k, v in msg.items()
                if not (
                    isinstance(k, str)
                    and (
                        k.startswith("_")
                        or k
                        in {
                            "tool_name",
                            "timestamp",
                            "codex_reasoning_items",
                            "codex_message_items",
                        }
                    )
                )
            }
            sanitized.append(cleaned)
        return sanitized

    def convert_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Tools are already in OpenAI format — identity."""
        return tools

    # ------------------------------------------------------------------
    # Build kwargs — wire execution plan params into the API call
    # ------------------------------------------------------------------

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Build ``chat.completions.create()`` kwargs.

        Wires the execution plan's inference configuration into
        ``extra_body`` so the engine applies the tuned settings:
        KV-cache block reuse, CUDA graphs, speculative decoding,
        GPU layer count, and context length.

        Zero per-turn entropy: no timestamps, UUIDs, random seeds,
        or date strings are injected into the payload.
        """
        sanitized = self.convert_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized,
        }

        # extra_body carries engine-specific parameters from the plan.
        # These are static per ExecutionPlan — they never change turn-to-turn.
        extra_body: dict[str, Any] = {}

        if tools:
            kwargs["tools"] = tools

        max_tokens = params.get("max_tokens")
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        extra_body["cache_prompt"] = self._plan.kv.enable_block_reuse
        extra_body["kv_cache_free_gpu_mem_fraction"] = (
            self._plan.kv.free_gpu_memory_fraction
        )

        if self._plan.levers.cuda_graphs:
            extra_body["cuda_graphs"] = True

        spec = self._plan.levers.spec_decode
        if spec.value != "none":
            extra_body["speculative"] = {"mode": spec.value}
            if self._plan.levers.draft_model is not None:
                extra_body["speculative"]["draft_model"] = self._plan.levers.draft_model

        extra_body["n_gpu_layers"] = self._plan.placement.gpu_layers

        extra_body["n_ctx"] = self._plan.target_ctx_tokens

        if self._plan.kv.cache_type_k:
            extra_body["cache_type_k"] = self._plan.kv.cache_type_k
        if self._plan.kv.cache_type_v:
            extra_body["cache_type_v"] = self._plan.kv.cache_type_v

        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    # ------------------------------------------------------------------
    # Normalize response — extract native tool_calls (no XML scraping)
    # ------------------------------------------------------------------

    def normalize_response(
        self,
        response: Any,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Normalize an OpenAI ChatCompletion to ``NormalizedResponse``.

        Extracts native ``tool_calls`` from the provider's response — NO
        regex or XML scraping. Every ``tool_calls`` entry comes from the
        provider's structured ``tool_calls`` field.

        Raises:
            ValueError: if the response has no choices or the choice has
                no message.
        """
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError("response missing choices")

        choice = response.choices[0]
        if not hasattr(choice, "message") or choice.message is None:
            raise ValueError("response choice missing message")

        msg = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id if hasattr(tc, "id") else None,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        usage = None
        if hasattr(response, "usage") and response.usage is not None:
            u = response.usage
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        return NormalizedResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )
