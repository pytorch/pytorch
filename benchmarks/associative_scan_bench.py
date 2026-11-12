# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and performance harness for the native aten::associative_scan.

References:
  * eager: torch.cumsum / torch.cumprod for the add / mul combine modes
  * sequential: an explicit Python loop over the scan dimension (the naive
    O(N) reference, matching jax.lax.associative_scan for builtin combines)
  * mamba: the linear-recurrence reference used by SSM kernels, computed with
    an explicit sequential loop over the (a, b) pairs.

Usage:
    python benchmarks/associative_scan_bench.py [--cpu] [--cuda]
"""

import argparse
import time

import torch


def _scan_ref(x, combine_mode, dim):
    dim = dim if dim >= 0 else dim + x.ndim
    out = torch.empty_like(x)
    acc = None
    for i in range(x.size(dim)):
        sl = [slice(None)] * x.ndim
        sl[dim] = i
        v = x[sl]
        if acc is None:
            acc = v.clone()
        elif combine_mode == "add":
            acc = acc + v
        elif combine_mode == "mul":
            acc = acc * v
        elif combine_mode == "max":
            acc = torch.maximum(acc, v)
        elif combine_mode == "min":
            acc = torch.minimum(acc, v)
        else:
            raise AssertionError(f"unknown combine_mode {combine_mode}")
        out[sl] = acc
    return out


def _linrec_ref(a, b, dim):
    dim = dim if dim >= 0 else dim + a.ndim
    n = a.size(dim)
    A = torch.empty_like(a)
    H = torch.empty_like(b)
    sl0 = [slice(None)] * a.ndim
    sl0[dim] = 0
    A[sl0] = a[sl0].clone()
    H[sl0] = b[sl0].clone()
    for i in range(1, n):
        sl = [slice(None)] * a.ndim
        sl[dim] = i
        prev = [slice(None)] * a.ndim
        prev[dim] = i - 1
        A[sl] = a[sl] * A[prev]
        H[sl] = a[sl] * H[prev] + b[sl]
    return A, H


def _check(name, actual, expected, atol=1e-5, rtol=1e-5):
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    diff = (actual - expected).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  (max abs diff {diff:.3e})")
    return ok


def correctness(device):
    print(f"\n=== Correctness ({device}) ===")
    ok = True
    x = torch.randn(64, 32, device=device)
    for mode, ref in [
        ("add", lambda t: torch.cumsum(t, 0)),
        ("mul", lambda t: torch.cumprod(t, 0)),
    ]:
        ok &= _check(
            f"native {mode} vs eager cum{'sum' if mode == 'add' else 'prod'}",
            torch.associative_scan(x, mode, 0),
            ref(x),
            atol=1e-4,
            rtol=1e-4,
        )

    for mode in ["add", "mul", "max", "min"]:
        ok &= _check(
            f"native {mode} vs sequential reference",
            torch.associative_scan(x, mode, 0),
            _scan_ref(x, mode, 0),
            atol=1e-4,
            rtol=1e-4,
        )

    a = torch.rand(64, 32, device=device) * 0.5 + 0.5
    b = torch.randn(64, 32, device=device)
    ea, eh = _linrec_ref(a, b, 0)
    na, nh = torch.associative_scan([a, b], "linear_recurrence", 0)
    ok &= _check("native linear_recurrence A vs mamba reference", na, ea)
    ok &= _check("native linear_recurrence H vs mamba reference", nh, eh)

    from torch._higher_order_ops.associative_scan import associative_scan as hop_scan

    for combine, mode in [
        (lambda u, v: u + v, "add"),
        (lambda u, v: u * v, "mul"),
        (torch.maximum, "max"),
        (torch.minimum, "min"),
    ]:
        ok &= _check(
            f"HOP {mode} (native routing) vs sequential reference",
            hop_scan(combine, x, dim=0),
            _scan_ref(x, mode, 0),
            atol=1e-4,
            rtol=1e-4,
        )

    def _combine(u, v):
        a1, b1 = u
        a2, b2 = v
        return (a2 * a1, a2 * b1 + b2)

    ha, hb = hop_scan(_combine, (a, b), dim=0)
    ok &= _check("HOP linear_recurrence (generic) vs mamba reference", ha, ea)
    ok &= _check("HOP linear_recurrence H (generic) vs mamba reference", hb, eh)
    return ok


def _bench(fn, iters):
    # warmup
    fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def performance(device):
    print(f"\n=== Performance ({device}) ===")
    sizes = [(2**16, 64), (2**18, 64), (2**20, 64)]
    header = (
        f"{'shape':>14} | {'native add':>10} | {'cumsum':>10} | "
        f"{'native mul':>10} | {'cumprod':>10} | {'native linrec':>14} | {'seq ref':>10}"
    )
    print(header)
    print("-" * len(header))
    for rows, cols in sizes:
        x = torch.randn(rows, cols, device=device)
        a = torch.rand(rows, cols, device=device) * 0.5 + 0.5
        b = torch.randn(rows, cols, device=device)
        t_add = _bench(lambda: torch.associative_scan(x, "add", 0), 5)
        t_cumsum = _bench(lambda: torch.cumsum(x, 0), 5)
        t_mul = _bench(lambda: torch.associative_scan(x, "mul", 0), 5)
        t_cumprod = _bench(lambda: torch.cumprod(x, 0), 5)
        t_linrec = _bench(
            lambda: torch.associative_scan([a, b], "linear_recurrence", 0), 5
        )
        t_seq = _bench(lambda: _linrec_ref(a, b, 0), 1)
        print(
            f"{f'{rows}x{cols}':>14} | {t_add:>10.3f} | {t_cumsum:>10.3f} | "
            f"{t_mul:>10.3f} | {t_cumprod:>10.3f} | {t_linrec:>14.3f} | {t_seq:>10.3f}"
        )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="run CPU checks")
    parser.add_argument("--cuda", action="store_true", help="run CUDA checks")
    parser.add_argument("--fast", action="store_true", help="small shapes only")
    args = parser.parse_args()

    devices = []
    if args.cpu:
        devices.append("cpu")
    if args.cuda:
        if not torch.cuda.is_available():
            print("CUDA requested but unavailable")
            return 1
        devices.append("cuda")
    if not devices:
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    ok = True
    for device in devices:
        ok &= correctness(device)
        if not args.fast:
            ok &= performance(device)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
