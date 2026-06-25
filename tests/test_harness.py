"""Tests for the measurement harness (HER-19).

Covers:
  - Metrics dataclass construction and JSON round-trip
  - BenchmarkReport aggregate computation
  - A/B delta math against fixed numbers
  - No hardcoded sleeps in benchmark code
  - Mocked harness lifecycle
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
from dataclasses import replace
from unittest import mock

import pytest
from hermes_nim_xlr.contracts import ExecutionPlan
from hermes_nim_xlr.harness.benchmark import BenchmarkHarness
from hermes_nim_xlr.harness.metrics import (
    ABDelta,
    BenchmarkReport,
    TurnMetrics,
    compute_ab_delta,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def report_2turns(sample_plan: ExecutionPlan) -> BenchmarkReport:
    return BenchmarkReport(
        plan=sample_plan,
        turns=[
            TurnMetrics(
                turn=0,
                ttft_ms=600.0,
                tokens=42,
                tok_s=26.0,
                tool_calls=0,
                peak_vram_mb=3200,
                cached=False,
            ),
            TurnMetrics(
                turn=1,
                ttft_ms=420.0,
                tokens=48,
                tok_s=48.0,
                tool_calls=0,
                peak_vram_mb=3300,
                cached=True,
            ),
        ],
    )


@pytest.fixture
def report_3turns_all_levers_off(sample_plan: ExecutionPlan) -> BenchmarkReport:
    return BenchmarkReport(
        plan=sample_plan,
        turns=[
            TurnMetrics(
                turn=0,
                ttft_ms=650.0,
                tokens=40,
                tok_s=25.0,
                tool_calls=1,
                peak_vram_mb=3200,
                cached=False,
            ),
            TurnMetrics(
                turn=1,
                ttft_ms=430.0,
                tokens=50,
                tok_s=50.0,
                tool_calls=0,
                peak_vram_mb=3300,
                cached=True,
            ),
            TurnMetrics(
                turn=2,
                ttft_ms=410.0,
                tokens=52,
                tok_s=52.0,
                tool_calls=1,
                peak_vram_mb=3300,
                cached=True,
            ),
        ],
    )


@pytest.fixture
def report_3turns_cuda_graphs(sample_plan: ExecutionPlan) -> BenchmarkReport:
    plan = replace(sample_plan, levers=replace(sample_plan.levers, cuda_graphs=True))
    return BenchmarkReport(
        plan=plan,
        turns=[
            TurnMetrics(
                turn=0,
                ttft_ms=620.0,
                tokens=42,
                tok_s=28.0,
                tool_calls=1,
                peak_vram_mb=3250,
                cached=False,
            ),
            TurnMetrics(
                turn=1,
                ttft_ms=380.0,
                tokens=52,
                tok_s=55.0,
                tool_calls=0,
                peak_vram_mb=3320,
                cached=True,
            ),
            TurnMetrics(
                turn=2,
                ttft_ms=360.0,
                tokens=54,
                tok_s=56.0,
                tool_calls=1,
                peak_vram_mb=3320,
                cached=True,
            ),
        ],
    )


def _mock_chat_response(
    content: str | None = "test response",
    finish_reason: str = "stop",
    completion_tokens: int = 50,
    prompt_tokens: int = 200,
    tool_calls: list[dict] | None = None,
) -> bytes:
    choice: dict = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish_reason,
    }
    if tool_calls:
        choice["message"]["tool_calls"] = tool_calls
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return json.dumps(payload).encode("utf-8")


def _mock_urlopen(data: bytes) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.read.return_value = data
    resp.status = 200
    return resp


# ===========================================================================
# Metrics dataclass construction
# ===========================================================================


class TestTurnMetrics:
    def test_constructs_with_all_fields(self):
        m = TurnMetrics(
            turn=0,
            ttft_ms=500.0,
            tokens=100,
            tok_s=45.0,
            tool_calls=2,
            peak_vram_mb=4096,
            cached=False,
        )
        assert m.turn == 0
        assert m.ttft_ms == 500.0
        assert m.tokens == 100
        assert m.tok_s == 45.0
        assert m.tool_calls == 2
        assert m.peak_vram_mb == 4096
        assert m.cached is False

    def test_immutable_fields_via_type_check(self):
        m = TurnMetrics(0, 100.0, 10, 45.0, 0, 3000, True)
        assert isinstance(m.turn, int)
        assert isinstance(m.ttft_ms, float)
        assert isinstance(m.tok_s, float)


# ===========================================================================
# BenchmarkReport aggregate computation
# ===========================================================================


class TestBenchmarkReport:
    def test_avg_ttft_first(self, report_2turns: BenchmarkReport):
        assert report_2turns.avg_ttft_first_ms() == 600.0

    def test_avg_ttft_cached(self, report_2turns: BenchmarkReport):
        assert report_2turns.avg_ttft_cached_ms() == 420.0

    def test_steady_state_tok_s(self, report_2turns: BenchmarkReport):
        assert report_2turns.steady_state_tok_s() == 48.0

    def test_peak_vram(self, report_2turns: BenchmarkReport):
        assert report_2turns.peak_vram_mb() == 3300

    def test_end_to_end_ms(self, report_2turns: BenchmarkReport):
        assert report_2turns.end_to_end_ms() == 1020.0

    def test_empty_report_returns_zero_safely(self, sample_plan: ExecutionPlan):
        r = BenchmarkReport(plan=sample_plan)
        assert r.avg_ttft_first_ms() == 0.0
        assert r.avg_ttft_cached_ms() == 0.0
        assert r.steady_state_tok_s() == 0.0
        assert r.peak_vram_mb() == 0
        assert r.end_to_end_ms() == 0.0

    def test_spec_acceptance_rate_default(self, report_2turns: BenchmarkReport):
        assert report_2turns.spec_acceptance_rate() == 0.0

    def test_config_derived_from_plan(self, report_2turns: BenchmarkReport):
        cfg = report_2turns.config()
        assert cfg["cuda_graphs"] is False
        assert cfg["spec_decode"] == "none"
        assert cfg["kv_block_reuse"] is True

    def test_timestamp_is_iso_format(self, report_2turns: BenchmarkReport):
        ts = report_2turns.timestamp()
        assert "T" in ts
        assert ts.endswith("+00:00") or "+" in ts


# ===========================================================================
# JSON serialization / deserialization
# ===========================================================================


class TestJsonRoundTrip:
    def test_to_json_contains_expected_keys(self, report_2turns: BenchmarkReport):
        raw = report_2turns.to_json()
        data = json.loads(raw)
        assert "plan" in data
        assert "turns" in data
        assert "summary" in data
        assert data["summary"]["avg_ttft_first_ms"] == 600.0
        assert data["summary"]["steady_state_tok_s"] == 48.0

    def test_round_trip_preserves_data(self, report_2turns: BenchmarkReport):
        raw = report_2turns.to_json()
        restored = BenchmarkReport.from_json(raw)
        assert restored.avg_ttft_first_ms() == report_2turns.avg_ttft_first_ms()
        assert restored.avg_ttft_cached_ms() == report_2turns.avg_ttft_cached_ms()
        assert restored.steady_state_tok_s() == report_2turns.steady_state_tok_s()
        assert restored.peak_vram_mb() == report_2turns.peak_vram_mb()
        assert len(restored.turns) == len(report_2turns.turns)

    def test_to_json_with_single_turn(self, sample_plan: ExecutionPlan):
        r = BenchmarkReport(
            plan=sample_plan,
            turns=[
                TurnMetrics(0, 500.0, 30, 40.0, 0, 3000, False),
            ],
        )
        raw = r.to_json()
        data = json.loads(raw)
        assert len(data["turns"]) == 1
        assert data["turns"][0]["ttft_ms"] == 500.0


# ===========================================================================
# A/B delta computation
# ===========================================================================


class TestABDelta:
    def test_compute_ab_delta_correct_math(
        self,
        report_3turns_all_levers_off: BenchmarkReport,
        report_3turns_cuda_graphs: BenchmarkReport,
    ):
        delta = compute_ab_delta(
            "cuda_graphs",
            report_3turns_all_levers_off,
            report_3turns_cuda_graphs,
        )
        assert delta.lever_name == "cuda_graphs"
        assert delta.delta_ttft_cold_ms == pytest.approx(-30.0)
        assert delta.delta_ttft_cached_ms == pytest.approx(-50.0)
        expected_tok_pct = ((55.5 - 51.0) / 51.0) * 100
        assert delta.delta_tok_s_pct == pytest.approx(expected_tok_pct)
        assert delta.delta_vram_mb == 20

    def test_compute_ab_delta_returns_typed_result(
        self,
        report_3turns_all_levers_off: BenchmarkReport,
        report_3turns_cuda_graphs: BenchmarkReport,
    ):
        delta = compute_ab_delta(
            "cuda_graphs",
            report_3turns_all_levers_off,
            report_3turns_cuda_graphs,
        )
        assert isinstance(delta, ABDelta)
        assert isinstance(delta.baseline_report, BenchmarkReport)
        assert isinstance(delta.treatment_report, BenchmarkReport)

    def test_compute_ab_delta_identical_reports(self, report_2turns: BenchmarkReport):
        delta = compute_ab_delta("identity", report_2turns, report_2turns)
        assert delta.delta_ttft_cold_ms == 0.0
        assert delta.delta_ttft_cached_ms == 0.0
        assert delta.delta_tok_s_pct == 0.0
        assert delta.delta_vram_mb == 0


# ===========================================================================
# Harness lifecycle (mocked backend + HTTP)
# ===========================================================================


class TestHarnessLifecycle:
    def test_constructs_with_plan(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(plan=sample_plan)
        assert harness._endpoint_url == "http://127.0.0.1:8080/v1"

    def test_constructs_with_custom_endpoint(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(
            plan=sample_plan, endpoint_url="http://localhost:9090/v1"
        )
        assert harness._endpoint_url == "http://localhost:9090/v1"

    def test_run_returns_benchmark_report(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(
            plan=sample_plan, endpoint_url="http://127.0.0.1:9999/v1"
        )
        response_data = _mock_chat_response(
            content="Test response.",
            finish_reason="stop",
            completion_tokens=10,
            prompt_tokens=50,
        )

        with (
            mock.patch(
                "urllib.request.urlopen", return_value=_mock_urlopen(response_data)
            ),
            mock.patch(
                "hermes_nim_xlr.harness.benchmark._probe_vram_mib",
                return_value=3000,
            ),
            mock.patch("time.monotonic", side_effect=[0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
        ):
            report = harness.run(turns=3)

        assert isinstance(report, BenchmarkReport)
        assert len(report.turns) == 3
        for m in report.turns:
            assert m.ttft_ms > 0
            assert m.tok_s > 0
            assert m.peak_vram_mb == 3000

        assert report.turns[0].cached is False
        assert report.turns[1].cached is True
        assert report.turns[2].cached is True

    def test_run_with_tool_call_turns(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(
            plan=sample_plan, endpoint_url="http://127.0.0.1:9999/v1"
        )

        tool_call_response = _mock_chat_response(
            content=None,
            finish_reason="tool_calls",
            completion_tokens=20,
            prompt_tokens=100,
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        )
        text_response = _mock_chat_response(
            content="It is sunny.",
            finish_reason="stop",
            completion_tokens=15,
            prompt_tokens=120,
        )

        responses = [_mock_urlopen(tool_call_response), _mock_urlopen(text_response)]

        with (
            mock.patch("urllib.request.urlopen", side_effect=responses),
            mock.patch(
                "hermes_nim_xlr.harness.benchmark._probe_vram_mib",
                return_value=3000,
            ),
            mock.patch("time.monotonic", side_effect=[0.0, 0.5, 0.0, 0.5]),
        ):
            report = harness.run(turns=2)

        assert len(report.turns) == 2
        assert report.turns[0].tool_calls == 1
        assert report.turns[1].tool_calls == 0

    def test_run_ab_returns_delta(
        self,
        sample_plan: ExecutionPlan,
        report_3turns_all_levers_off: BenchmarkReport,
    ):
        harness = BenchmarkHarness(
            plan=sample_plan, endpoint_url="http://127.0.0.1:9999/v1"
        )
        response_data = _mock_chat_response(
            content="Response.",
            finish_reason="stop",
            completion_tokens=10,
        )

        with (
            mock.patch(
                "urllib.request.urlopen", return_value=_mock_urlopen(response_data)
            ),
            mock.patch(
                "hermes_nim_xlr.harness.benchmark._probe_vram_mib",
                return_value=3000,
            ),
            mock.patch("time.monotonic", side_effect=[0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
        ):
            delta = harness.run_ab(
                baseline_report=report_3turns_all_levers_off,
                lever_name="cuda_graphs",
                lever_config={"cuda_graphs": True},
            )

        assert isinstance(delta, ABDelta)
        assert delta.lever_name == "cuda_graphs"

    def test_overrides_change_plan(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(
            plan=sample_plan,
            overrides={"cuda_graphs": True},
        )
        assert harness._overrides == {"cuda_graphs": True}

    def test_http_error_propagates(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(
            plan=sample_plan, endpoint_url="http://127.0.0.1:9999/v1"
        )
        error_resp = mock.MagicMock()
        error_resp.read.return_value = b'{"error": "model not loaded"}'
        error_resp.code = 400

        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.HTTPError(
                    url="http://127.0.0.1:9999/v1/chat/completions",
                    code=400,
                    msg="Bad Request",
                    hdrs={},
                    fp=None,
                ),
            ),
            mock.patch(
                "hermes_nim_xlr.harness.benchmark._probe_vram_mib",
                return_value=3000,
            ),
        ):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                harness.run(turns=1)

    def test_connection_error_propagates(self, sample_plan: ExecutionPlan):
        harness = BenchmarkHarness(
            plan=sample_plan, endpoint_url="http://127.0.0.1:9999/v1"
        )

        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError(reason="connection refused"),
            ),
            mock.patch(
                "hermes_nim_xlr.harness.benchmark._probe_vram_mib",
                return_value=3000,
            ),
        ):
            with pytest.raises(RuntimeError, match="Connection failed"):
                harness.run(turns=1)


# ===========================================================================
# No hardcoded sleeps
# ===========================================================================


def test_no_hardcoded_sleeps_in_benchmark():
    with open("hermes_nim_xlr/harness/benchmark.py") as f:
        content = f.read()
    lines_with_sleep = [
        (i + 1, line)
        for i, line in enumerate(content.splitlines())
        if "time.sleep" in line and not line.strip().startswith("#")
    ]
    assert len(lines_with_sleep) == 0, (
        f"Found hardcoded time.sleep calls in benchmark.py: {lines_with_sleep}"
    )


def test_no_hardcoded_sleeps_in_metrics():
    with open("hermes_nim_xlr/harness/metrics.py") as f:
        content = f.read()
    assert "time.sleep" not in content, "Found hardcoded time.sleep call in metrics.py"


# ===========================================================================
# VRAM probe
# ===========================================================================


def test_vram_probe_fallback_on_failure():
    from hermes_nim_xlr.harness.benchmark import _probe_vram_mib

    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert _probe_vram_mib() == 0

    with mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
    ):
        assert _probe_vram_mib() == 0


def test_vram_probe_safe_parse():
    from hermes_nim_xlr.harness.benchmark import _probe_vram_mib

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "  3200 MiB  \n"

    with mock.patch("subprocess.run", return_value=mock_result):
        assert _probe_vram_mib() == 3200

    mock_result.stdout = ""
    with mock.patch("subprocess.run", return_value=mock_result):
        assert _probe_vram_mib() == 0


def test_vram_probe_safe_parse_garbage():
    from hermes_nim_xlr.harness.benchmark import _probe_vram_mib

    mock_result = mock.MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "  N/A  \n"

    with mock.patch("subprocess.run", return_value=mock_result):
        assert _probe_vram_mib() == 0


# ===========================================================================
# Multi-turn async persistence (HER-24)
# ===========================================================================


def test_multi_turn_async_persistence(sample_plan: ExecutionPlan):
    """Files are written for each turn and the harness doesn't block on persistence."""
    import tempfile

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        harness = BenchmarkHarness(
            plan=sample_plan,
            endpoint_url="http://127.0.0.1:9999/v1",
            persist_dir=tmpdir,
        )

        response_data = _mock_chat_response(
            content="Turn response.",
            finish_reason="stop",
            completion_tokens=10,
            prompt_tokens=50,
        )

        with (
            mock.patch(
                "urllib.request.urlopen",
                return_value=_mock_urlopen(response_data),
            ),
            mock.patch(
                "hermes_nim_xlr.harness.benchmark._probe_vram_mib",
                return_value=3000,
            ),
            mock.patch(
                "time.monotonic",
                side_effect=[0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
            ),
        ):
            report = harness.run(turns=3)

        assert len(report.turns) == 3

        entries = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            entries = sorted(f for f in os.listdir(tmpdir) if f.endswith(".json"))
            if len(entries) >= 3:
                break
            time.sleep(0.01)

        assert len(entries) == 3, (
            f"Expected 3 persisted files, found {len(entries)}: {entries}"
        )
        for filename in entries:
            assert filename.startswith("turn_")
            assert filename.endswith(".json")
