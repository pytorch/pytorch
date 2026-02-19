import argparse
import json
import operator
import os
import platform
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.node import Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase


# Simple, deterministic operator support: only `operator.add` is supported.
class _AddOnlySupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: Node) -> bool:  # type: ignore[override]
        return node.op == "call_function" and node.target is operator.add


def _make_large_partition_fn(num_nodes: int, num_unsupported: int, seed: int = 0):
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if num_unsupported < 0 or num_unsupported > num_nodes:
        raise ValueError("num_unsupported must be in [0, num_nodes]")

    rng = random.Random(seed)
    if num_unsupported:
        unsupported_indices = set(rng.sample(range(num_nodes), num_unsupported))
    else:
        unsupported_indices = set()

    def fn(a, b):
        # Two streams with frequent merges to avoid a straight-line graph and
        # to exercise partition-cycle detection across unsupported ops.
        # Each loop iteration emits exactly one op so total ops == num_nodes.
        left = a
        right = b
        for i in range(num_nodes):
            if i in unsupported_indices:
                if rng.random() < 0.5:
                    left = left.relu()
                else:
                    right = right.relu()
            else:
                if rng.random() < 0.5:
                    left = left + right
                else:
                    right = right + left
        return left, right

    return fn


@dataclass
class BenchmarkResult:
    num_nodes: int
    num_unsupported: int
    iters: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _run_case(num_nodes: int, num_unsupported: int, iters: int) -> BenchmarkResult:
    timings: list[float] = []
    for _ in range(iters):
        fn = _make_large_partition_fn(
            num_nodes=num_nodes, num_unsupported=num_unsupported
        )
        gm = symbolic_trace(fn)
        t0 = time.perf_counter()
        partitioner = CapabilityBasedPartitioner(
            gm,
            _AddOnlySupport(),
            allows_single_node_partition=True,
        )

        partitioner.propose_partitions()
        e = time.perf_counter()
        take = (e - t0) * 1000
        timings.append(take)

    timings.sort()
    median = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95) - 1]
    return BenchmarkResult(
        num_nodes=num_nodes,
        num_unsupported=num_unsupported,
        iters=iters,
        median_ms=median,
        p95_ms=p95,
        min_ms=min(timings),
        max_ms=max(timings),
    )


def _parse_size(item: str) -> tuple[int, int]:
    try:
        num_nodes_str, num_unsup_str = item.split(":", 1)
        return int(num_nodes_str), int(num_unsup_str)
    except Exception as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(
            "Sizes must be formatted as NUM_NODES:NUM_UNSUPPORTED"
        ) from exc


def _collect_env() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FX partitioner micro-benchmark")
    parser.add_argument(
        "--sizes",
        type=_parse_size,
        default=[
            (4_000, 40),
            (10_000, 100),
            (40_000, 400),
            (100_000, 1_000),
            (1_000_000, 10_000),
        ],
        help="List of NUM_NODES:NUM_UNSUPPORTED pairs.",
        nargs="+",
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="Timed iterations per case."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write JSON results.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    env = _collect_env()
    print("Environment:", json.dumps(env, indent=2))

    results: list[BenchmarkResult] = []
    for num_nodes, num_unsupported in args.sizes:
        result = _run_case(num_nodes, num_unsupported, args.iters)
        results.append(result)
        print(
            f"nodes={num_nodes:>7} unsupported={num_unsupported:>6} "
            f"median={result.median_ms:7.2f}ms p95={result.p95_ms:7.2f}ms "
            f"min={result.min_ms:7.2f}ms max={result.max_ms:7.2f}ms"
        )

    if args.json:
        payload = {
            "env": env,
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote results to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
