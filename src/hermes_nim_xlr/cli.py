"""Command-line entry points for Hermes-NIM-XLR."""

import argparse
import dataclasses
import json
import sys
from enum import Enum
from typing import Any

from hermes_nim_xlr import contracts
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
    execution_plan = plan(
        host,
        objective=objective,
        prefer_performance=args.prefer_performance,
    )
    print(_serialize_plan(execution_plan))
    return 0


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
    plan_parser.add_argument(
        "--prefer-performance",
        action="store_true",
        help="opt into the TensorRT-LLM / WSL2 performance path on Windows",
    )

    args = parser.parse_args(argv)
    if args.command == "plan":
        return _cmd_plan(args)

    # argparse with required subparsers prevents this, but keep the type
    # checker and future-proofing happy.
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
