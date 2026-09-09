"""Compare steady-state CPU forward-call overhead using the same clone body.

Empty/tiny tensors emphasize wrapper overhead, while larger sizes include more
tensor work. Registration, warmup, correctness checks, and backward are not timed.
The requires_grad case includes forward autograd graph construction. Results are
medians and IQRs of pooled per-call block timings, not confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import timeit
from collections.abc import Callable

import torch
from torch._library.dispatchless import dispatchless_custom_op
from torch.utils.benchmark import Timer


def clone(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def clone_backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output


def benchmark_case(
    variants: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    size: int,
    mode: str,
    min_run_time: float,
    repeats: int,
) -> list[dict[str, str | int | float]]:
    requires_grad = mode == "requires_grad"
    x = torch.ones(size, dtype=torch.float32, device="cpu", requires_grad=requires_grad)
    samples: dict[str, list[float]] = {name: [] for name in variants}
    with torch.set_grad_enabled(mode != "no_grad"):
        expected = clone(x)
        for fn in variants.values():
            actual = fn(x)
            torch.testing.assert_close(actual, expected)
            if requires_grad:
                (grad,) = torch.autograd.grad(actual.sum(), x)
                torch.testing.assert_close(grad, torch.ones_like(x))
            for _ in range(10):
                fn(x)

        names = list(variants)
        for repeat in range(repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                measurement = Timer(
                    stmt="fn(x)",
                    globals={"fn": variants[name], "x": x},
                    num_threads=1,
                    timer=timeit.default_timer,
                ).blocked_autorange(min_run_time=min_run_time)
                samples[name].extend(measurement.times)

    return summarize(samples, size, mode)


def summarize(
    samples: dict[str, list[float]], size: int, mode: str
) -> list[dict[str, str | int | float]]:
    medians = {name: statistics.median(times) * 1e6 for name, times in samples.items()}
    rows: list[dict[str, str | int | float]] = []
    for name, times in samples.items():
        quartiles = (
            statistics.quantiles(times, n=4, method="inclusive")
            if len(times) > 1
            else [times[0]] * 3
        )
        rows.append(
            {
                "mode": mode,
                "numel": size,
                "variant": name,
                "median_us": medians[name],
                "iqr_us": (quartiles[2] - quartiles[0]) * 1e6,
                "overhead_vs_plain_us": medians[name] - medians["plain"],
                "speedup_vs_custom_op": medians["custom_op"] / medians[name],
                "blocks": len(times),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 1, 1024])
    args = parser.parse_args()
    if args.min_run_time <= 0 or args.repeats < 1 or any(size < 0 for size in args.sizes):
        parser.error("run time and repeats must be positive; sizes must be nonnegative")

    torch.set_num_threads(1)
    custom = torch.library.custom_op(
        "dispatchless_benchmark::clone", clone, mutates_args=()
    )
    custom.register_autograd(clone_backward)
    variants = {
        "plain": clone,
        "custom_op": custom,
        "dispatchless_custom_op": dispatchless_custom_op(clone),
    }
    rows = []
    for mode in ("no_grad", "grad_enabled", "requires_grad"):
        for size in args.sizes:
            rows.extend(
                benchmark_case(variants, size, mode, args.min_run_time, args.repeats)
            )
    report = {
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
            "platform": platform.platform(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "dtype": "float32",
            "min_run_time": args.min_run_time,
            "repeats": args.repeats,
        },
        "results": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
