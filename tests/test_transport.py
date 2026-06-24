"""Tests for XLRTransport (HER-18).

Covers:
  - Round-trip of a multi-turn conversation with tool calls
  - Native tool_calls extraction (no XML scraping)
  - Prefix byte-stability across turns
  - Zero per-turn entropy (no timestamps, UUIDs, non-deterministic fields)
  - Message sanitization (internal scaffolding keys stripped)
  - Response normalization edge cases
"""

from __future__ import annotations

import json
import re
from typing import Any
from unittest import mock

import pytest
from agent.transports.types import NormalizedResponse
from hermes_nim_xlr.contracts import (
    BackendChoice,
    BackendKind,
    BringUp,
    DecodeLevers,
    ExecutionPlan,
    KvCacheConfig,
    KvDtype,
    LayerPlacement,
    ModelChoice,
    Objective,
    SpecDecode,
    WeightQuant,
)
from hermes_nim_xlr.transport import XLRTransport

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def execution_plan() -> ExecutionPlan:
    """A representative ExecutionPlan for a 6 GB laptop GPU (Ampere)."""
    return ExecutionPlan(
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
            total_layers=28,
            gpu_layers=28,
            cpu_offload_layers=0,
            tensor_parallel=1,
            pipeline_parallel=1,
            note="fully GPU-resident",
        ),
        kv=KvCacheConfig(
            dtype=KvDtype.INT8,
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.35,
            host_cache_size_bytes=0,
        ),
        levers=DecodeLevers(
            cuda_graphs=True,
            spec_decode=SpecDecode.NGRAM,
            draft_model=None,
        ),
        backend=BackendChoice(
            kind=BackendKind.LLAMACPP,
            bring_up=BringUp.NATIVE_WINDOWS,
            serve_endpoint="http://127.0.0.1:8080/v1",
        ),
        target_ctx_tokens=4096,
        est_vram_mb=2800,
        est_decode_tok_s=(30, 50),
        rationale=("INT4 weights for ~5000 MB budget", "fully GPU-resident"),
        warnings=(),
    )


@pytest.fixture
def transport(execution_plan: ExecutionPlan) -> XLRTransport:
    return XLRTransport(
        execution_plan=execution_plan,
        endpoint_url=execution_plan.backend.serve_endpoint,
    )


def _make_chat_completion(
    content: str | None = None,
    tool_calls_data: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> mock.MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    msg = mock.MagicMock()
    msg.content = content
    msg.tool_calls = None
    if tool_calls_data:
        msg.tool_calls = []
        for tc_data in tool_calls_data:
            tc = mock.MagicMock()
            tc.id = tc_data.get("id", "call_abc123")
            tc.function.name = tc_data.get("name", "test_tool")
            tc.function.arguments = tc_data.get("arguments", "{}")
            msg.tool_calls.append(tc)

    choice = mock.MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = mock.MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = mock.MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ===========================================================================
# Convert methods
# ===========================================================================


def test_convert_messages_passthrough(transport: XLRTransport):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    result = transport.convert_messages(messages)
    assert result == messages


def test_convert_messages_strips_internal_keys(transport: XLRTransport):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello!",
            "_internal_marker": True,
            "tool_name": "test",
        },
    ]
    result = transport.convert_messages(messages)
    assert len(result) == 2
    assert "_internal_marker" not in result[1]
    assert "tool_name" not in result[1]
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Hello!"


def test_convert_tools_passthrough(transport: XLRTransport):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = transport.convert_tools(tools)
    assert result == tools


# ===========================================================================
# Build kwargs — execution plan wiring
# ===========================================================================


def test_build_kwargs_basic_structure(
    transport: XLRTransport,
    execution_plan: ExecutionPlan,
):
    model = "QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    kwargs = transport.build_kwargs(model=model, messages=messages)

    assert kwargs["model"] == model
    assert kwargs["messages"] == messages
    assert "max_tokens" not in kwargs

    extra = kwargs["extra_body"]
    assert extra["cache_prompt"] is execution_plan.kv.enable_block_reuse
    expected_fraction = execution_plan.kv.free_gpu_memory_fraction
    assert extra["kv_cache_free_gpu_mem_fraction"] == expected_fraction
    assert extra["cuda_graphs"] is True
    assert extra["speculative"] == {"mode": "ngram"}
    assert extra["n_gpu_layers"] == execution_plan.placement.gpu_layers
    assert extra["n_ctx"] == execution_plan.target_ctx_tokens


def test_build_kwargs_with_tools(transport: XLRTransport):
    messages = [{"role": "user", "content": "Search something"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    kwargs = transport.build_kwargs(model="test-model", messages=messages, tools=tools)
    assert kwargs["tools"] == tools


def test_build_kwargs_max_tokens(transport: XLRTransport):
    kwargs = transport.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=512,
    )
    assert kwargs["max_tokens"] == 512


def test_build_kwargs_spec_decode_draft_model(execution_plan: ExecutionPlan):
    from dataclasses import replace

    plan = replace(
        execution_plan,
        levers=DecodeLevers(
            cuda_graphs=False,
            spec_decode=SpecDecode.DRAFT_TARGET,
            draft_model="Qwen/Qwen2-0.5B-Instruct-GGUF",
        ),
    )
    t = XLRTransport(plan, endpoint_url="http://127.0.0.1:8080/v1")
    kwargs = t.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    extra = kwargs["extra_body"]
    assert extra["speculative"]["mode"] == "draft_target"
    assert extra["speculative"]["draft_model"] == "Qwen/Qwen2-0.5B-Instruct-GGUF"


def test_build_kwargs_spec_decode_none(execution_plan: ExecutionPlan):
    from dataclasses import replace

    plan = replace(
        execution_plan,
        levers=DecodeLevers(
            cuda_graphs=False,
            spec_decode=SpecDecode.NONE,
            draft_model=None,
        ),
    )
    t = XLRTransport(plan, endpoint_url="http://127.0.0.1:8080/v1")
    kwargs = t.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    extra = kwargs.get("extra_body", {})
    assert "speculative" not in extra


# ===========================================================================
# Normalize response
# ===========================================================================


def test_normalize_response_text(transport: XLRTransport):
    response = _make_chat_completion(
        content="Hello, world!",
        finish_reason="stop",
        prompt_tokens=50,
        completion_tokens=10,
    )
    result = transport.normalize_response(response)
    assert isinstance(result, NormalizedResponse)
    assert result.content == "Hello, world!"
    assert result.tool_calls is None
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 50
    assert result.usage.completion_tokens == 10


def test_normalize_response_tool_calls_native(transport: XLRTransport):
    """Tool calls come from the provider's structured field, not XML scraping."""
    response = _make_chat_completion(
        content=None,
        tool_calls_data=[
            {
                "id": "call_abc123",
                "name": "web_search",
                "arguments": '{"query": "latest AI news"}',
            },
        ],
        finish_reason="tool_calls",
    )
    result = transport.normalize_response(response)
    assert result.content is None
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_abc123"
    assert tc.name == "web_search"
    assert tc.arguments == '{"query": "latest AI news"}'
    assert result.finish_reason == "tool_calls"


def test_normalize_response_multiple_tool_calls(transport: XLRTransport):
    response = _make_chat_completion(
        content=None,
        tool_calls_data=[
            {"id": "call_1", "name": "web_search", "arguments": '{"q": "a"}'},
            {"id": "call_2", "name": "calculator", "arguments": '{"expr": "2+2"}'},
        ],
        finish_reason="tool_calls",
    )
    result = transport.normalize_response(response)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[1].name == "calculator"


def test_normalize_response_missing_choices_raises(transport: XLRTransport):
    response = mock.MagicMock()
    response.choices = []
    with pytest.raises(ValueError, match="missing choices"):
        transport.normalize_response(response)


def test_normalize_response_missing_message_raises(transport: XLRTransport):
    choice = mock.MagicMock()
    choice.message = None
    response = mock.MagicMock()
    response.choices = [choice]
    with pytest.raises(ValueError, match="missing message"):
        transport.normalize_response(response)


def test_normalize_response_no_usage(transport: XLRTransport):
    response = mock.MagicMock()
    response.choices = [mock.MagicMock()]
    response.choices[0].message.content = "Hello"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = None

    result = transport.normalize_response(response)
    assert result.content == "Hello"
    assert result.usage is None


# ===========================================================================
# Multi-turn round-trip
# ===========================================================================


def test_round_trip_single_turn(transport: XLRTransport):
    """Messages → build_kwargs → mock response → normalize_response."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    kwargs = transport.build_kwargs(model="test-model", messages=messages)
    assert kwargs["messages"] == messages

    response = _make_chat_completion(content="4", finish_reason="stop")
    result = transport.normalize_response(response)
    assert result.content == "4"
    assert result.tool_calls is None


def test_round_trip_with_tool_calls(transport: XLRTransport):
    """Full round-trip: messages → kwargs → response → normalized, with tool calls."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Search for AI news."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    kwargs = transport.build_kwargs(model="test-model", messages=messages, tools=tools)
    assert kwargs["tools"] == tools
    assert "extra_body" in kwargs

    response = _make_chat_completion(
        content=None,
        tool_calls_data=[
            {
                "id": "call_xyz789",
                "name": "web_search",
                "arguments": '{"query": "AI news 2026"}',
            }
        ],
        finish_reason="tool_calls",
    )

    result = transport.normalize_response(response)
    assert result.content is None
    assert result.tool_calls is not None
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == '{"query": "AI news 2026"}'


def test_round_trip_multi_turn(transport: XLRTransport):
    """Simulate a multi-turn conversation with tool calls and follow-up."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]

    # Turn 1: model calls weather_tool
    kwargs_t1 = transport.build_kwargs(model="test-model", messages=messages)
    assert kwargs_t1["messages"] == messages

    result_t1 = transport.normalize_response(
        _make_chat_completion(
            content=None,
            tool_calls_data=[
                {
                    "id": "call_weather",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                }
            ],
            finish_reason="tool_calls",
        )
    )
    assert result_t1.tool_calls is not None

    # Append assistant tool call + tool result
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": '{"temperature": 22, "condition": "sunny"}',
        }
    )

    # Turn 2: model responds with final answer
    kwargs_t2 = transport.build_kwargs(model="test-model", messages=messages)
    assert "tool_calls" in str(kwargs_t2["messages"][-2])

    result_t2 = transport.normalize_response(
        _make_chat_completion(
            content="It is 22°C and sunny in Paris!", finish_reason="stop"
        )
    )
    assert result_t2.content == "It is 22°C and sunny in Paris!"


# ===========================================================================
# Prefix byte-stability (AGENTS.md invariant #1)
# ===========================================================================


def test_prefix_byte_stability(transport: XLRTransport):
    """Two consecutive turns with identical system prompts must produce
    identical first N bytes of the request payload."""
    system_prompt = "You are a helpful AI assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello!"},
    ]

    kwargs_a = transport.build_kwargs(model="test-model", messages=messages)
    kwargs_b = transport.build_kwargs(model="test-model", messages=messages)

    payload_a = json.dumps(kwargs_a, sort_keys=True, ensure_ascii=False)
    payload_b = json.dumps(kwargs_b, sort_keys=True, ensure_ascii=False)

    assert payload_a == payload_b


def test_prefix_byte_stability_across_turns(transport: XLRTransport):
    """The stable prefix (system prompt + first messages) must be byte-identical
    across turns, even as new messages are appended."""
    system_prompt = "You are helpful."
    base_messages = [
        {"role": "system", "content": system_prompt},
    ]

    messages_t1 = base_messages + [{"role": "user", "content": "What is 2+2?"}]
    kwargs_t1 = transport.build_kwargs(model="test-model", messages=messages_t1)

    messages_t2 = base_messages + [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    kwargs_t2 = transport.build_kwargs(model="test-model", messages=messages_t2)

    payload_t1 = json.dumps(kwargs_t1, sort_keys=True, ensure_ascii=False)
    payload_t2 = json.dumps(kwargs_t2, sort_keys=True, ensure_ascii=False)

    prefix = json.dumps({"messages": base_messages}, sort_keys=True)[:-1]
    assert payload_t2.startswith(payload_t1[: len(prefix) + 50])


def test_prefix_stable_with_identical_plans():
    """Two distinct transport instances with identical plans produce
    identical kwargs for the same messages."""
    ep = ExecutionPlan(
        objective=Objective.THROUGHPUT_FIRST,
        model=ModelChoice(
            repo="test/model",
            params_b=3.0,
            weight_quant=WeightQuant.INT4_AWQ,
            est_weight_mb=1800,
            n_layers=28,
            kv_heads=8,
            head_dim=128,
            max_context_tokens=8192,
        ),
        placement=LayerPlacement(
            total_layers=28,
            gpu_layers=28,
            cpu_offload_layers=0,
            tensor_parallel=1,
            pipeline_parallel=1,
            note="",
        ),
        kv=KvCacheConfig(
            dtype=KvDtype.INT8,
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.3,
            host_cache_size_bytes=0,
        ),
        levers=DecodeLevers(
            cuda_graphs=False,
            spec_decode=SpecDecode.NONE,
            draft_model=None,
        ),
        backend=BackendChoice(
            kind=BackendKind.LLAMACPP,
            bring_up=BringUp.NATIVE_WINDOWS,
            serve_endpoint="http://127.0.0.1:8080/v1",
        ),
        target_ctx_tokens=2048,
        est_vram_mb=2000,
        est_decode_tok_s=(20, 40),
        rationale=("test",),
        warnings=(),
    )

    t1 = XLRTransport(ep, endpoint_url=ep.backend.serve_endpoint)
    t2 = XLRTransport(ep, endpoint_url=ep.backend.serve_endpoint)

    messages = [{"role": "user", "content": "Hi"}]

    k1 = t1.build_kwargs(model="test-model", messages=messages)
    k2 = t2.build_kwargs(model="test-model", messages=messages)

    assert json.dumps(k1, sort_keys=True) == json.dumps(k2, sort_keys=True)


# ===========================================================================
# Zero per-turn entropy (AGENTS.md invariant #1)
# ===========================================================================

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
_TIMESTAMP_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
    _UUID_RE,
    re.compile(r"\b\d{10}\b"),
    re.compile(r"\b\d{13}\b"),
]


def test_no_timestamp_in_payload(transport: XLRTransport):
    """The request payload must not contain any ISO-8601 timestamps,
    UUIDs, or unix timestamps."""
    kwargs = transport.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    payload_text = json.dumps(kwargs, ensure_ascii=False)

    for pattern in _TIMESTAMP_PATTERNS:
        matches = pattern.findall(payload_text)
        assert not matches, (
            f"Found timestamp/UUID `{pattern.pattern}` in payload: {matches}"
        )


def test_no_timestamp_repeated_calls(transport: XLRTransport):
    """Multiple calls to build_kwargs must produce identical payloads —
    no hidden counter, no clock read, no non-determinism."""
    messages = [{"role": "user", "content": "Repeatable"}]
    results = []
    for _ in range(5):
        kwargs = transport.build_kwargs(model="test-model", messages=messages)
        results.append(json.dumps(kwargs, sort_keys=True))

    first = results[0]
    for r in results[1:]:
        assert r == first, "Payload differs between calls — entropy detected"


def test_no_extra_body_entropy(transport: XLRTransport):
    """The extra_body dict must contain only plan-derived fields, never
    a timestamp, seed, or request ID."""
    kwargs = transport.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    extra = kwargs.get("extra_body", {})
    allowed_keys = {
        "cache_prompt",
        "kv_cache_free_gpu_mem_fraction",
        "cuda_graphs",
        "speculative",
        "n_gpu_layers",
        "n_ctx",
        "cache_type_k",
        "cache_type_v",
    }
    assert set(extra.keys()).issubset(allowed_keys), (
        f"Unexpected keys in extra_body: {set(extra.keys()) - allowed_keys}"
    )


def test_no_timestamp_across_different_messages(transport: XLRTransport):
    """Different user messages must not cause timestamp injection."""
    kwargs_a = transport.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    kwargs_b = transport.build_kwargs(
        model="test-model",
        messages=[{"role": "user", "content": "World"}],
    )

    payload_a = json.dumps(kwargs_a, ensure_ascii=False)
    payload_b = json.dumps(kwargs_b, ensure_ascii=False)

    for pattern in _TIMESTAMP_PATTERNS:
        msg_a = f"Timestamp `{pattern.pattern}` found in payload A"
        msg_b = f"Timestamp `{pattern.pattern}` found in payload B"
        assert not pattern.findall(payload_a), msg_a
        assert not pattern.findall(payload_b), msg_b


# ===========================================================================
# API mode
# ===========================================================================


def test_api_mode(transport: XLRTransport):
    assert transport.api_mode == "chat_completions"


def test_endpoint_url(transport: XLRTransport, execution_plan: ExecutionPlan):
    assert transport._endpoint_url == execution_plan.backend.serve_endpoint.rstrip("/")


def test_endpoint_url_strips_trailing_slash():
    t = XLRTransport(
        execution_plan=mock.MagicMock(spec=ExecutionPlan),
        endpoint_url="http://127.0.0.1:8080/v1/",
    )
    assert t._endpoint_url == "http://127.0.0.1:8080/v1"
