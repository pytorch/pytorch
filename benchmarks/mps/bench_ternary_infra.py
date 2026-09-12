#!/usr/bin/env python3
"""A/B benchmark for MPS torch.clamp ternary dispatch flavors."""

from __future__ import annotations

import argparse
import os
import platform
import re
import statistics
import subprocess
from contextlib import contextmanager

import torch
from torch.testing import assert_close
from torch.utils.benchmark import Timer


ENV_VAR = "PYTORCH_TERNARY_FORCE_FLAVOR"
TIMER_STMT = """for _ in range(K):
    torch.clamp(x, min=lo, max=hi)
torch.mps.synchronize()"""
TRIALS = 3
INNER_SWEEP = (8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 4096)
INNER_STRIDED_MIN_EXTENT = 0


@contextmanager
def ternary_force(flavor: str):
    old = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = flavor
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = old


def gpu_device_utilization() -> int:
    out = subprocess.check_output(
        ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    if m is None:
        raise RuntimeError("ioreg did not report Device Utilization %")
    return int(m.group(1))


def ioreg_guard(max_util: int = 20) -> None:
    # External GPU contention produces garbage numbers; refuse to bench a
    # busy device.
    if platform.system() != "Darwin":
        return
    util = gpu_device_utilization()
    print(f"gpu_device_utilization={util}%")
    if util > max_util:
        raise RuntimeError(
            f"GPU busy (Device Utilization {util}% > {max_util}%); not benching"
        )


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _rtol_atol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 1.0e-2, 1.0e-2
    return 1.0e-4, 1.0e-5


def _k_iterations(numel: int, quick: bool) -> int:
    base = 16 if quick else 32
    if numel <= 2**20:
        return base * 2
    if numel <= 2**22:
        return base
    return max(1, base // 2)


def assert_case(
    x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, flavor: str
) -> None:
    expected = torch.clamp(x.cpu(), min=lo.cpu(), max=hi.cpu())
    with ternary_force(flavor):
        observed = torch.clamp(x, min=lo, max=hi)
        torch.mps.synchronize()
    rtol, atol = _rtol_atol(expected.dtype)
    assert_close(observed.cpu(), expected, rtol=rtol, atol=atol)


def run_flavor(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    flavor: str,
    quick: bool,
    min_run_time: float,
) -> float:
    globals_dict = {
        "torch": torch,
        "x": x,
        "lo": lo,
        "hi": hi,
        "K": _k_iterations(x.numel(), quick),
    }
    with ternary_force(flavor):
        t = Timer(stmt=TIMER_STMT, globals=globals_dict)
        return t.blocked_autorange(min_run_time=min_run_time).median * 1e3


def run_pair(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    flavor_a: str,
    flavor_b: str,
    quick: bool,
    min_run_time: float,
) -> tuple[float, float, float]:
    assert_case(x, lo, hi, flavor_a)
    assert_case(x, lo, hi, flavor_b)

    a_times: list[float] = []
    b_times: list[float] = []
    for _ in range(TRIALS):
        a_times.append(run_flavor(x, lo, hi, flavor_a, quick, min_run_time))
        b_times.append(run_flavor(x, lo, hi, flavor_b, quick, min_run_time))
    med_a = statistics.median(a_times)
    med_b = statistics.median(b_times)
    speedup = med_a / med_b if med_b else float("nan")
    return med_a, med_b, speedup


def make_bounds(
    x: torch.Tensor, bound_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    xb = x.to(bound_dtype)
    return xb - 0.25, xb + 0.25


def print_markdown(
    title: str, rows: list[tuple[str, str, float, float, float]]
) -> None:
    print()
    print(f"### {title}")
    print("| case | dtype | a_ms | b_ms | speedup_B_over_A |")
    print("| --- | --- | ---:| ---:| ---:|")
    for case_name, dtype_name, a_ms, b_ms, speedup in rows:
        print(
            f"| {case_name} | {dtype_name} | "
            f"{a_ms:0.4f} | {b_ms:0.4f} | {speedup:0.4f} |"
        )


def print_summary(rows: list[tuple[str, str, float, float, float]]) -> None:
    print()
    print("## Summary")
    print("| case | dtype | speedup_B_over_A |")
    print("| --- | --- | ---:|")
    for case_name, dtype_name, _a, _b, speedup in rows:
        print(f"| {case_name} | {dtype_name} | {speedup:0.4f} |")


def run_dense_contig(
    rows: list[tuple[str, str, float, float, float]],
    quick: bool,
    min_run_time: float,
) -> None:
    case_rows: list[tuple[str, str, float, float, float]] = []
    n_values = [2**18, 2**22, 2**24] if not quick else [2**18, 2**22]
    for n in n_values:
        shape_name = f"dense-contig n={n}"
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(n, device="mps", dtype=dtype)
            lo, hi = make_bounds(x, dtype)
            a_ms, b_ms, speedup = run_pair(
                x,
                lo,
                hi,
                "scalar",
                "ilp",
                quick,
                min_run_time,
            )
            entry = (
                shape_name,
                f"{dtype_name(dtype)}/{dtype_name(dtype)}",
                a_ms,
                b_ms,
                speedup,
            )
            rows.append(entry)
            case_rows.append(entry)

        # one mixed-dtype variant requested: half input with float bounds
        if n == 2**18:
            x = torch.randn(n, device="mps", dtype=torch.float16)
            lo = torch.full((n,), -0.25, device="mps", dtype=torch.float32)
            hi = torch.full((n,), 0.25, device="mps", dtype=torch.float32)
            a_ms, b_ms, speedup = run_pair(
                x,
                lo,
                hi,
                "scalar",
                "ilp",
                quick,
                min_run_time,
            )
            entry = (shape_name + " mixed", "float16/float32", a_ms, b_ms, speedup)
            rows.append(entry)
            case_rows.append(entry)

    print_markdown("dense-contig", case_rows)


def run_unit_inner(
    rows: list[tuple[str, str, float, float, float]],
    quick: bool,
    min_run_time: float,
) -> None:
    case_rows: list[tuple[str, str, float, float, float]] = []
    m = 1024 if quick else 4096
    n = 1024 if quick else 4096
    shape_name = f"unit-inner broadcast [{m},{n}]"
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(m, n, device="mps", dtype=dtype)
        lo, hi = make_bounds(torch.randn(n, device="mps", dtype=dtype), dtype)
        a_ms, b_ms, speedup = run_pair(
            x,
            lo,
            hi,
            "strided",
            "inner_contiguous",
            quick,
            min_run_time,
        )
        entry = (
            shape_name,
            f"{dtype_name(dtype)}/{dtype_name(dtype)}",
            a_ms,
            b_ms,
            speedup,
        )
        rows.append(entry)
        case_rows.append(entry)
    print_markdown("unit-inner broadcast", case_rows)


def run_channel_broadcast(
    rows: list[tuple[str, str, float, float, float]],
    quick: bool,
    min_run_time: float,
) -> None:
    case_rows: list[tuple[str, str, float, float, float]] = []
    x_shape = (16, 16, 16, 16) if quick else (64, 64, 56, 56)
    bound_shape = (1, x_shape[1], 1, 1)
    shape_name = f"channel broadcast {list(x_shape)}"
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(*x_shape, device="mps", dtype=dtype)
        lo, hi = make_bounds(
            torch.randn(*bound_shape, device="mps", dtype=dtype),
            dtype,
        )
        a_ms, b_ms, speedup = run_pair(
            x,
            lo,
            hi,
            "strided",
            "inner_strided",
            quick,
            min_run_time,
        )
        entry = (
            shape_name,
            f"{dtype_name(dtype)}/{dtype_name(dtype)}",
            a_ms,
            b_ms,
            speedup,
        )
        rows.append(entry)
        case_rows.append(entry)
    print_markdown("channel broadcast", case_rows)


def run_transposed(
    rows: list[tuple[str, str, float, float, float]],
    quick: bool,
    min_run_time: float,
) -> None:
    case_rows: list[tuple[str, str, float, float, float]] = []
    size = 1024 if quick else 4096
    base = torch.randn(size, size, device="mps", dtype=torch.float32)
    x = base.t()
    lo, hi = make_bounds(
        torch.randn(size, size, device="mps", dtype=torch.float32), torch.float32
    )
    a_ms, b_ms, speedup = run_pair(
        x,
        lo,
        hi,
        "strided",
        "inner_strided",
        quick,
        min_run_time,
    )
    entry = ("transposed base.t()", "float32/float32", a_ms, b_ms, speedup)
    rows.append(entry)
    case_rows.append(entry)

    x16 = x.to(torch.float16)
    lo16, hi16 = make_bounds(
        torch.randn(size, size, device="mps", dtype=torch.float16),
        torch.float16,
    )
    a_ms, b_ms, speedup = run_pair(
        x16,
        lo16,
        hi16,
        "strided",
        "inner_strided",
        quick,
        min_run_time,
    )
    entry = ("transposed base.t()", "float16/float16", a_ms, b_ms, speedup)
    rows.append(entry)
    case_rows.append(entry)

    # cast path: half transposed input vs float bounds. The per-element
    # runtime-type switch dominates cast loads, so inner_strided wins here
    # even without inner locality (gate: cast_needed || unit-or-broadcast).
    xc = torch.randn(size, size, device="mps", dtype=torch.float16).t()
    loc = torch.randn(size, size, device="mps") - 0.25
    hic = torch.randn(size, size, device="mps") + 0.25
    a_ms, b_ms, speedup = run_pair(
        xc, loc, hic, "strided", "inner_strided", quick, min_run_time
    )
    entry = ("transposed cast", "float16/float32", a_ms, b_ms, speedup)
    rows.append(entry)
    case_rows.append(entry)

    print_markdown("transposed", case_rows)


def run_scalar_bounds(
    rows: list[tuple[str, str, float, float, float]],
    quick: bool,
    min_run_time: float,
) -> None:
    case_rows: list[tuple[str, str, float, float, float]] = []
    n = 2**18 if quick else 2**22
    shape_name = f"scalar tensor bounds [{n}]"
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(n, device="mps", dtype=dtype)
        lo = torch.tensor(-0.25, device="mps", dtype=dtype)
        hi = torch.tensor(0.25, device="mps", dtype=dtype)
        a_ms, b_ms, speedup = run_pair(
            x,
            lo,
            hi,
            "strided",
            "inner_strided",
            quick,
            min_run_time,
        )
        entry = (shape_name, f"{dtype_name(dtype)}/scalar", a_ms, b_ms, speedup)
        rows.append(entry)
        case_rows.append(entry)
    print_markdown("scalar tensor bounds", case_rows)


def run_sweep(
    rows: list[tuple[str, str, float, float, float]],
    quick: bool,
    min_run_time: float,
) -> int:
    case_rows: list[tuple[str, str, float, float, float]] = []
    inners = INNER_SWEEP if not quick else INNER_SWEEP[:4]
    global INNER_STRIDED_MIN_EXTENT
    crossover = None

    for inner in inners:
        outer = max(1, (2**22) // inner)
        x = torch.randn(outer, inner, device="mps", dtype=torch.float32)
        lo, hi = make_bounds(
            torch.randn(outer, 1, device="mps", dtype=torch.float32), torch.float32
        )
        a_ms, b_ms, speedup = run_pair(
            x,
            lo,
            hi,
            "strided",
            "inner_strided",
            quick,
            min_run_time,
        )
        case_name = f"sweep inner={inner} (outer={outer})"
        entry = (case_name, "float32/float32", a_ms, b_ms, speedup)
        rows.append(entry)
        case_rows.append(entry)
        if crossover is None and speedup >= 1.0:
            crossover = inner

    if crossover is None:
        crossover = inners[-1]
    INNER_STRIDED_MIN_EXTENT = crossover
    print_markdown("sweep x[outer,inner] bounds[outer,1]", case_rows)
    print(f"INNER_STRIDED_MIN_EXTENT={INNER_STRIDED_MIN_EXTENT}")
    return INNER_STRIDED_MIN_EXTENT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    ioreg_guard()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    print(f"label={args.label} torch={torch.__version__}")
    min_run_time = 0.2 if args.quick else 0.5

    torch.manual_seed(0)
    rows: list[tuple[str, str, float, float, float]] = []

    run_dense_contig(rows, args.quick, min_run_time)
    run_unit_inner(rows, args.quick, min_run_time)
    run_channel_broadcast(rows, args.quick, min_run_time)
    run_transposed(rows, args.quick, min_run_time)
    run_scalar_bounds(rows, args.quick, min_run_time)
    if args.sweep:
        run_sweep(rows, args.quick, min_run_time)

    print_summary(rows)


if __name__ == "__main__":
    main()
