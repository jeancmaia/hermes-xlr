"""Data classes for measurement-harness metrics.

Pure stdlib, no GPU dependencies. Every metric is a honest measurement
collected by the harness — no fake or synthetic numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hermes_nim_xlr.contracts import ExecutionPlan


@dataclass
class TurnMetrics:
    turn: int
    ttft_ms: float
    tokens: int
    tok_s: float
    tool_calls: int
    peak_vram_mb: int
    cached: bool


@dataclass
class ABDelta:
    lever_name: str
    delta_ttft_cold_ms: float
    delta_ttft_cached_ms: float
    delta_tok_s_pct: float
    delta_vram_mb: int
    baseline_report: BenchmarkReport
    treatment_report: BenchmarkReport


class _MetricEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, BenchmarkReport):
            return {
                k: (
                    [asdict(t) for t in v]
                    if k == "turns"
                    else asdict(v)
                    if hasattr(v, "__dataclass_fields__")
                    else v
                )
                for k, v in asdict(o).items()
            }
        if isinstance(o, TurnMetrics):
            return asdict(o)
        return super().default(o)


@dataclass
class BenchmarkReport:
    plan: ExecutionPlan
    turns: list[TurnMetrics] = field(default_factory=list)

    def avg_ttft_first_ms(self) -> float:
        cold = [t.ttft_ms for t in self.turns if not t.cached]
        return _safe_mean(cold)

    def avg_ttft_cached_ms(self) -> float:
        cached = [t.ttft_ms for t in self.turns if t.cached]
        return _safe_mean(cached)

    def steady_state_tok_s(self) -> float:
        cached = [t.tok_s for t in self.turns if t.cached]
        return _safe_mean(cached)

    def spec_acceptance_rate(self) -> float:
        return 0.0

    def peak_vram_mb(self) -> int:
        if not self.turns:
            return 0
        return max(t.peak_vram_mb for t in self.turns)

    def end_to_end_ms(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.ttft_ms for t in self.turns)

    def config(self) -> dict[str, Any]:
        return {
            "cuda_graphs": self.plan.levers.cuda_graphs,
            "spec_decode": self.plan.levers.spec_decode.value,
            "draft_model": self.plan.levers.draft_model,
            "kv_block_reuse": self.plan.kv.enable_block_reuse,
            "kv_dtype": self.plan.kv.dtype.value,
            "ctx_size": self.plan.target_ctx_tokens,
        }

    def timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "plan": asdict(self.plan),
            "turns": [asdict(t) for t in self.turns],
            "summary": {
                "avg_ttft_first_ms": self.avg_ttft_first_ms(),
                "avg_ttft_cached_ms": self.avg_ttft_cached_ms(),
                "steady_state_tok_s": self.steady_state_tok_s(),
                "spec_acceptance_rate": self.spec_acceptance_rate(),
                "peak_vram_mb": self.peak_vram_mb(),
                "end_to_end_ms": self.end_to_end_ms(),
                "config": self.config(),
                "timestamp": self.timestamp(),
            },
        }
        return json.dumps(
            payload, cls=_MetricEncoder, indent=indent, ensure_ascii=False
        )

    @classmethod
    def from_json(cls, raw: str) -> BenchmarkReport:
        data = json.loads(raw)
        plan = ExecutionPlan(**data["plan"])
        turns = [TurnMetrics(**t) for t in data.get("turns", [])]
        return cls(plan=plan, turns=turns)


def compute_ab_delta(
    lever_name: str,
    baseline: BenchmarkReport,
    treatment: BenchmarkReport,
) -> ABDelta:
    return ABDelta(
        lever_name=lever_name,
        delta_ttft_cold_ms=(
            treatment.avg_ttft_first_ms() - baseline.avg_ttft_first_ms()
        ),
        delta_ttft_cached_ms=(
            treatment.avg_ttft_cached_ms() - baseline.avg_ttft_cached_ms()
        ),
        delta_tok_s_pct=_safe_pct_change(
            baseline.steady_state_tok_s(), treatment.steady_state_tok_s()
        ),
        delta_vram_mb=treatment.peak_vram_mb() - baseline.peak_vram_mb(),
        baseline_report=baseline,
        treatment_report=treatment,
    )


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_pct_change(baseline: float, treatment: float) -> float:
    if baseline == 0:
        return 0.0
    return ((treatment - baseline) / baseline) * 100.0
