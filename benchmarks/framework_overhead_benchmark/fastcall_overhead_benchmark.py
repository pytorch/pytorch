from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.utils.benchmark as benchmark


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    statement: str


CASES = (
    BenchmarkCase("pass", "pass"),
    BenchmarkCase("torch.is_floating_point(x)", "torch.is_floating_point(x)"),
    BenchmarkCase("torch.add(x, y)", "torch.add(x, y)"),
    BenchmarkCase("x + y", "x + y"),
    BenchmarkCase("torch.mul(x, y)", "torch.mul(x, y)"),
    BenchmarkCase("torch.sum(x)", "torch.sum(x)"),
    BenchmarkCase("torch.sum(input=x, dim=0)", "torch.sum(input=x, dim=0)"),
    BenchmarkCase("x.sum()", "x.sum()"),
    BenchmarkCase("x.sum(dim=0)", "x.sum(dim=0)"),
    BenchmarkCase("x.size(0)", "x.size(0)"),
    BenchmarkCase("x.stride(0)", "x.stride(0)"),
    BenchmarkCase("x.is_contiguous()", "x.is_contiguous()"),
    BenchmarkCase("x.contiguous()", "x.contiguous()"),
    BenchmarkCase("x.view(-1)", "x.view(-1)"),
    BenchmarkCase("torch.zeros(())", "torch.zeros(())"),
    BenchmarkCase("torch.empty(())", "torch.empty(())"),
    BenchmarkCase("torch.ops.aten.add.Tensor(x, y)", "torch.ops.aten.add.Tensor(x, y)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Python overhead for representative PyTorch calls."
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in CASES],
        help="Measure only this call. May be specified more than once.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Use a fixed iteration count instead of blocked autorange.",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=1.0,
        help="Minimum seconds spent measuring each call.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    globals_ = {
        "torch": torch,
        "x": torch.ones(1),
        "y": torch.ones(1),
    }

    print(f"{'Call':<42} {'Median (ns)':>14} {'IQR (ns)':>12}")
    print("-" * 70)
    selected_cases = (
        CASES
        if args.case is None
        else tuple(case for case in CASES if case.name in args.case)
    )
    for case in selected_cases:
        timer = benchmark.Timer(
            stmt=case.statement,
            globals=globals_,
        )
        measurement = (
            timer.blocked_autorange(min_run_time=args.min_run_time)
            if args.iterations is None
            else timer.timeit(number=args.iterations)
        )
        print(
            f"{case.name:<42} "
            f"{measurement.median * 1e9:>14.1f} "
            f"{measurement.iqr * 1e9:>12.1f}"
        )


if __name__ == "__main__":
    main()
