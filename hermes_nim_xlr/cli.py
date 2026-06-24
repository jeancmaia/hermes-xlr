"""Command-line entry points for Hermes-NIM-XLR."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from hermes_nim_xlr import contracts
from hermes_nim_xlr.harness import BenchmarkHarness
from hermes_nim_xlr.harness.metrics import (
    BenchmarkReport,
    compute_ab_delta,
)
from hermes_nim_xlr.mapper import detect, plan


class _PlanEncoder(json.JSONEncoder):
    """JSON encoder that renders enums by value and lets dataclass-asdict
    output round-trip cleanly.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


def _serialize_plan(execution_plan: contracts.ExecutionPlan) -> str:
    payload = dataclasses.asdict(execution_plan)
    return json.dumps(payload, cls=_PlanEncoder, indent=2, ensure_ascii=False)


def _cmd_plan(args: argparse.Namespace) -> int:
    host = detect.detect()
    objective = contracts.Objective(args.objective)
    execution_plan = plan(host, objective=objective)
    print(_serialize_plan(execution_plan))
    return 0


def _save_report(path: Path, report: BenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    print(f"  wrote {path}")


def _load_report(path: Path) -> BenchmarkReport:
    raw = path.read_text(encoding="utf-8")
    return BenchmarkReport.from_json(raw)


def _cmd_benchmark_run(args: argparse.Namespace) -> int:
    """Run the full A/B protocol: baseline + each lever + summary."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    host = detect.detect()
    print(f"Host: {host.os} | GPU: {[g.name for g in host.gpus]}")
    print(f"Objective: {args.objective}")

    baseline_plan = plan(host, objective=contracts.Objective(args.objective))

    turns = args.turns
    endpoint = args.endpoint

    # Build a strict baseline (all levers OFF)
    baseline_levers = dataclasses.replace(
        baseline_plan.levers,
        cuda_graphs=False,
        spec_decode=contracts.SpecDecode.NONE,
        draft_model=None,
    )
    baseline_kv = dataclasses.replace(
        baseline_plan.kv,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    baseline_plan = dataclasses.replace(
        baseline_plan, levers=baseline_levers, kv=baseline_kv
    )

    print("\n=== Baseline (all levers OFF) ===")
    baseline_harness = BenchmarkHarness(baseline_plan, endpoint_url=endpoint)
    baseline_report = baseline_harness.run(turns=turns)
    _save_report(out / "baseline.json", baseline_report)
    _print_baseline(baseline_report)

    lever_configs: list[tuple[str, dict[str, Any]]] = [
        ("cuda-graphs", {"cuda_graphs": True}),
        (  # n-gram spec-decode with depth 32
            "spec-decode-ngram",
            {"spec_decode": contracts.SpecDecode.NGRAM},
        ),
        (  # KV-cache quant: q8_0 for both K and V
            "kv-quant",
            {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        ),
    ]

    ab_results: list[tuple[str, BenchmarkReport]] = []
    for lever_name, lever_cfg in lever_configs:
        print(f"\n=== A/B: {lever_name} ===")
        ab = baseline_harness.run_ab(
            baseline_report=baseline_report,
            lever_name=lever_name,
            lever_config=lever_cfg,
        )
        _save_report(out / f"{lever_name}.json", ab.treatment_report)
        _print_ab_delta(ab)
        ab_results.append((lever_name, ab.treatment_report))

    print("\n=== All levers combined ===")
    all_config: dict[str, Any] = {
        "cuda_graphs": True,
        "spec_decode": contracts.SpecDecode.NGRAM,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
    }
    all_ab = baseline_harness.run_ab(
        baseline_report=baseline_report,
        lever_name="all-combined",
        lever_config=all_config,
    )
    _save_report(out / "all-combined.json", all_ab.treatment_report)
    _print_ab_delta(all_ab)
    ab_results.append(("all-combined", all_ab.treatment_report))

    summary_path = out / "s5-summary.md"
    _write_summary_md(summary_path, baseline_report, ab_results)
    print(f"\n  wrote {summary_path}")

    return 0


def _cmd_benchmark_ab(args: argparse.Namespace) -> int:
    """Run a single A/B lever against a saved baseline."""
    baseline_report = _load_report(Path(args.baseline))
    endpoint = args.endpoint

    host = detect.detect()
    baseline_plan = plan(host, objective=contracts.Objective(args.objective))

    baseline_levers = dataclasses.replace(
        baseline_plan.levers,
        cuda_graphs=False,
        spec_decode=contracts.SpecDecode.NONE,
        draft_model=None,
    )
    baseline_kv = dataclasses.replace(
        baseline_plan.kv,
        cache_type_k="f16",
        cache_type_v="f16",
    )
    baseline_plan = dataclasses.replace(
        baseline_plan, levers=baseline_levers, kv=baseline_kv
    )

    lever_config: dict[str, Any] = {}
    if args.lever == "cuda_graphs":
        lever_config = {"cuda_graphs": True}
    elif args.lever == "spec_decode":
        lever_config = {"spec_decode": contracts.SpecDecode.NGRAM}
    elif args.lever == "kv_quant":
        lever_config = {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}
    elif args.lever == "all":
        lever_config = {
            "cuda_graphs": True,
            "spec_decode": contracts.SpecDecode.NGRAM,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        }
    else:
        print(f"Unknown lever: {args.lever}", file=sys.stderr)
        return 1

    harness = BenchmarkHarness(baseline_plan, endpoint_url=endpoint)
    ab = harness.run_ab(
        baseline_report=baseline_report,
        lever_name=args.lever,
        lever_config=lever_config,
    )

    out = Path(args.output) if args.output else Path(f"{args.lever}.json")
    _save_report(out, ab.treatment_report)
    _print_ab_delta(ab)

    return 0


def _cmd_benchmark_summary(args: argparse.Namespace) -> int:
    """Regenerate the summary table from saved reports."""
    reports_dir = Path(args.reports_dir)
    baseline_report = _load_report(reports_dir / "baseline.json")

    lever_names = ["cuda-graphs", "spec-decode-ngram", "kv-quant", "all-combined"]
    ab_results: list[tuple[str, BenchmarkReport]] = []
    for name in lever_names:
        path = reports_dir / f"{name}.json"
        if path.exists():
            report = _load_report(path)
            ab_results.append((name, report))

    out = Path(args.output) if args.output else reports_dir / "s5-summary.md"
    _write_summary_md(out, baseline_report, ab_results)
    print(f"  wrote {out}")

    return 0


def _print_baseline(report: BenchmarkReport) -> None:
    print(f"  TTFT cold:     {report.avg_ttft_first_ms():.1f} ms")
    print(f"  TTFT cached:   {report.avg_ttft_cached_ms():.1f} ms")
    print(f"  Decode tok/s:  {report.steady_state_tok_s():.1f}")
    print(f"  Peak VRAM:     {report.peak_vram_mb()} MiB")
    print(f"  End-to-end:    {report.end_to_end_ms():.1f} ms")


def _print_ab_delta(ab: Any) -> None:
    print(f"  TTFT cold d:   {ab.delta_ttft_cold_ms:+.1f} ms")
    print(f"  TTFT cached d: {ab.delta_ttft_cached_ms:+.1f} ms")
    print(f"  tok/s d:       {ab.delta_tok_s_pct:+.1f}%")
    print(f"  VRAM d:        {ab.delta_vram_mb:+d} MiB")


def _write_summary_md(
    path: Path,
    baseline: BenchmarkReport,
    results: list[tuple[str, BenchmarkReport]],
) -> None:
    lines: list[str] = []
    lines.append("# S5 — Decode-Lever A/B Summary")
    lines.append("")
    lines.append(
        f"**Date:** {baseline.timestamp()}  \n"
        f"**Model:** {baseline.plan.model.repo}  \n"
        f"**Engine:** llama-server (CUDA)  \n"
        f"**Context:** {baseline.plan.target_ctx_tokens} tokens  \n"
        f"**Turns:** {len(baseline.turns)}  \n"
    )
    lines.append("")

    lines.append("## Baseline (all levers OFF)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| TTFT cold (ms) | {baseline.avg_ttft_first_ms():.1f} |")
    lines.append(f"| TTFT cached (ms) | {baseline.avg_ttft_cached_ms():.1f} |")
    lines.append(f"| Decode tok/s | {baseline.steady_state_tok_s():.1f} |")
    lines.append(f"| Peak VRAM (MiB) | {baseline.peak_vram_mb()} |")
    lines.append(f"| End-to-end (ms) | {baseline.end_to_end_ms():.1f} |")
    lines.append("")
    lines.append("## A/B Deltas vs Baseline")
    lines.append("")
    lines.append(
        "| Lever | TTFT cold d (ms) | TTFT cached d (ms) | tok/s d (%) | VRAM d (MiB) |"
    )
    lines.append(
        "|-------|-----------------|-------------------|-------------|--------------|"
    )

    baseline_deltas = _compute_self_deltas(baseline)
    lines.append(
        f"| Baseline | {baseline_deltas['ttft_cold']:.1f} | "
        f"{baseline_deltas['ttft_cached']:.1f} | "
        f"{baseline_deltas['tok_s']:.1f} | "
        f"{baseline_deltas['vram']:+d} |"
    )

    for lever_name, report in results:
        delta = compute_ab_delta(lever_name, baseline, report)
        lines.append(
            f"| {lever_name} | {delta.delta_ttft_cold_ms:.1f} | "
            f"{delta.delta_ttft_cached_ms:.1f} | "
            f"{delta.delta_tok_s_pct:+.1f} | "
            f"{delta.delta_vram_mb:+d} |"
        )

    lines.append("")
    lines.append("## Per-Lever Reports")
    lines.append("")
    lines.append("| Lever | Report |")
    lines.append("|-------|--------|")
    lines.append("| Baseline | `baseline.json` |")
    for lever_name, report in results:
        lines.append(f"| {lever_name} | `{lever_name}.json` |")

    lines.append("")
    lines.append(
        "_Generated by `xlr benchmark run`. Positive tok/s d = faster. "
        "Positive TTFT d = slower (higher latency)._"
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _compute_self_deltas(report: BenchmarkReport) -> dict[str, float]:
    return {
        "ttft_cold": 0.0,
        "ttft_cached": 0.0,
        "tok_s": 0.0,
        "vram": 0,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``xlr`` console script."""
    parser = argparse.ArgumentParser(prog="xlr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="probe the host and emit the deterministic execution plan as JSON",
    )
    plan_parser.add_argument(
        "--objective",
        choices=[
            contracts.Objective.THROUGHPUT_FIRST.value,
            contracts.Objective.QUALITY_FIRST.value,
        ],
        default=contracts.Objective.THROUGHPUT_FIRST.value,
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="run the A/B measurement protocol against a live engine endpoint",
    )
    bench_sub = benchmark_parser.add_subparsers(dest="bench_command", required=True)

    run_parser = bench_sub.add_parser(
        "run",
        help="run full protocol: baseline + each lever + summary",
    )
    run_parser.add_argument(
        "--endpoint",
        default=None,
        help="engine endpoint URL (default: from ExecutionPlan)",
    )
    run_parser.add_argument(
        "--objective",
        choices=[
            contracts.Objective.THROUGHPUT_FIRST.value,
            contracts.Objective.QUALITY_FIRST.value,
        ],
        default=contracts.Objective.THROUGHPUT_FIRST.value,
    )
    run_parser.add_argument(
        "--turns",
        type=int,
        default=5,
        help="number of turns per benchmark run (default: 5)",
    )
    run_parser.add_argument(
        "--output-dir",
        default="docs/ab-reports",
        help="output directory for reports (default: docs/ab-reports)",
    )

    ab_parser = bench_sub.add_parser(
        "ab",
        help="run a single A/B lever against a saved baseline",
    )
    ab_parser.add_argument(
        "--baseline",
        required=True,
        help="path to saved baseline JSON report",
    )
    ab_parser.add_argument(
        "--lever",
        required=True,
        choices=["cuda_graphs", "spec_decode", "kv_quant", "all"],
        help="which decode lever to test",
    )
    ab_parser.add_argument(
        "--endpoint",
        default=None,
        help="engine endpoint URL (default: from ExecutionPlan)",
    )
    ab_parser.add_argument(
        "--objective",
        choices=[
            contracts.Objective.THROUGHPUT_FIRST.value,
            contracts.Objective.QUALITY_FIRST.value,
        ],
        default=contracts.Objective.THROUGHPUT_FIRST.value,
    )
    ab_parser.add_argument(
        "--output",
        default=None,
        help="output path for the treatment JSON report (default: <lever>.json)",
    )

    summary_parser = bench_sub.add_parser(
        "summary",
        help="regenerate summary table from saved reports",
    )
    summary_parser.add_argument(
        "--reports-dir",
        default="docs/ab-reports",
        help="directory containing saved reports (default: docs/ab-reports)",
    )
    summary_parser.add_argument(
        "--output",
        default=None,
        help="output path for summary markdown (default: <reports-dir>/s5-summary.md)",
    )

    args = parser.parse_args(argv)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "benchmark":
        if args.bench_command == "run":
            return _cmd_benchmark_run(args)
        if args.bench_command == "ab":
            return _cmd_benchmark_ab(args)
        if args.bench_command == "summary":
            return _cmd_benchmark_summary(args)
        benchmark_parser.print_help()
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
