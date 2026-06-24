"""Tests for the CLI entry points."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from hermes_nim_xlr.cli import main
from hermes_nim_xlr.contracts import ExecutionPlan
from hermes_nim_xlr.harness.metrics import BenchmarkReport, TurnMetrics

# ===========================================================================
# Help output
# ===========================================================================


class TestHelp:
    def test_plan_help(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["plan", "--help"])
        assert exc.value.code == 0

    def test_benchmark_help(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["benchmark", "--help"])
        assert exc.value.code == 0

    def test_benchmark_run_help(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["benchmark", "run", "--help"])
        assert exc.value.code == 0

    def test_benchmark_ab_help(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["benchmark", "ab", "--help"])
        assert exc.value.code == 0

    def test_benchmark_summary_help(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["benchmark", "summary", "--help"])
        assert exc.value.code == 0


# ===========================================================================
# benchmark run
# ===========================================================================


class TestBenchmarkRun:
    @mock.patch("hermes_nim_xlr.cli.detect.detect")
    def test_baseline_report_saved(
        self, mock_detect: mock.Mock, sample_plan: ExecutionPlan, tmp_path: Path
    ) -> None:
        mock_detect.return_value = mock.Mock(
            os="Windows",
            gpus=(mock.Mock(name="RTX 3050"),),
        )
        with mock.patch("hermes_nim_xlr.cli.plan") as mock_plan:
            mock_plan.return_value = sample_plan
            with mock.patch(
                "hermes_nim_xlr.harness.benchmark.BenchmarkHarness.run"
            ) as mock_run:
                mock_run.return_value = BenchmarkReport(
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
                            tool_calls=1,
                            peak_vram_mb=3200,
                            cached=True,
                        ),
                    ],
                )

                result = main(
                    [
                        "benchmark",
                        "run",
                        "--output-dir",
                        str(tmp_path),
                        "--turns",
                        "2",
                    ]
                )
                assert result == 0

                baseline_path = tmp_path / "baseline.json"
                assert baseline_path.exists()
                data = json.loads(baseline_path.read_text(encoding="utf-8"))
                assert "plan" in data
                assert "turns" in data
                assert "summary" in data
                assert data["summary"]["avg_ttft_first_ms"] == 600.0

    @mock.patch("hermes_nim_xlr.cli.detect.detect")
    def test_summary_md_generated(
        self, mock_detect: mock.Mock, sample_plan: ExecutionPlan, tmp_path: Path
    ) -> None:
        mock_detect.return_value = mock.Mock(
            os="Windows",
            gpus=(mock.Mock(name="RTX 3050"),),
        )
        with mock.patch("hermes_nim_xlr.cli.plan") as mock_plan:
            mock_plan.return_value = sample_plan
            report = BenchmarkReport(
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
                        tool_calls=1,
                        peak_vram_mb=3200,
                        cached=True,
                    ),
                ],
            )
            with mock.patch(
                "hermes_nim_xlr.harness.benchmark.BenchmarkHarness.run",
                return_value=report,
            ):
                result = main(
                    [
                        "benchmark",
                        "run",
                        "--output-dir",
                        str(tmp_path),
                        "--turns",
                        "2",
                    ]
                )
                assert result == 0

                summary_path = tmp_path / "s5-summary.md"
                assert summary_path.exists()
                content = summary_path.read_text(encoding="utf-8")
                assert "S5" in content
                assert "600.0" in content
                assert "cuda-graphs" in content
                assert "all-combined" in content


# ===========================================================================
# benchmark ab
# ===========================================================================


class TestBenchmarkAb:
    @mock.patch("hermes_nim_xlr.cli.detect.detect")
    def test_single_lever(
        self, mock_detect: mock.Mock, sample_plan: ExecutionPlan, tmp_path: Path
    ) -> None:
        mock_detect.return_value = mock.Mock(
            os="Windows",
            gpus=(mock.Mock(name="RTX 3050"),),
        )

        baseline_report = BenchmarkReport(
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
                    tool_calls=1,
                    peak_vram_mb=3200,
                    cached=True,
                ),
            ],
        )
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(baseline_report.to_json(), encoding="utf-8")

        treatment_report = BenchmarkReport(
            plan=sample_plan,
            turns=[
                TurnMetrics(
                    turn=0,
                    ttft_ms=550.0,
                    tokens=48,
                    tok_s=30.0,
                    tool_calls=0,
                    peak_vram_mb=3224,
                    cached=False,
                ),
                TurnMetrics(
                    turn=1,
                    ttft_ms=380.0,
                    tokens=52,
                    tok_s=52.0,
                    tool_calls=1,
                    peak_vram_mb=3224,
                    cached=True,
                ),
            ],
        )

        with mock.patch("hermes_nim_xlr.cli.plan") as mock_plan:
            mock_plan.return_value = sample_plan
            with mock.patch(
                "hermes_nim_xlr.harness.benchmark.BenchmarkHarness.run",
                return_value=treatment_report,
            ):
                output_path = tmp_path / "cuda-graphs.json"
                result = main(
                    [
                        "benchmark",
                        "ab",
                        "--baseline",
                        str(baseline_path),
                        "--lever",
                        "cuda_graphs",
                        "--output",
                        str(output_path),
                    ]
                )
                assert result == 0
                assert output_path.exists()

                data = json.loads(output_path.read_text(encoding="utf-8"))
                assert "summary" in data

    def test_unknown_lever(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "benchmark",
                    "ab",
                    "--baseline",
                    str(baseline_path),
                    "--lever",
                    "invalid",
                ]
            )
        assert exc.value.code == 2


# ===========================================================================
# benchmark summary
# ===========================================================================


class TestBenchmarkSummary:
    def test_regenerates_summary(
        self, sample_plan: ExecutionPlan, tmp_path: Path
    ) -> None:
        report = BenchmarkReport(
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
                    tool_calls=1,
                    peak_vram_mb=3200,
                    cached=True,
                ),
            ],
        )

        (tmp_path / "baseline.json").write_text(report.to_json(), encoding="utf-8")

        for name in ["cuda-graphs", "spec-decode-ngram", "kv-quant", "all-combined"]:
            (tmp_path / f"{name}.json").write_text(report.to_json(), encoding="utf-8")

        output_path = tmp_path / "s5-summary.md"
        result = main(
            [
                "benchmark",
                "summary",
                "--reports-dir",
                str(tmp_path),
                "--output",
                str(output_path),
            ]
        )
        assert result == 0
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "600.0" in content
        assert "cuda-graphs" in content
        assert "all-combined" in content
