#!/usr/bin/env python3
"""Fast plumbing check: torch.compile -> Inductor -> Triton -> GPU.

Deliberately small. This answers "is the stack wired up and producing correct
Triton kernels", not "is Inductor correct" -- the full suites do that.

It also says nothing about whether TLX engaged: TORCHINDUCTOR_TLX_MODE is
printed but never asserted on, so this passes identically on upstream Triton.
`dev.py test` sets that variable and gates TLX separately, by running
doctor first.
"""

from __future__ import annotations

import os
import sys
import time

import triton

import torch
from torch._inductor.utils import run_and_get_code


def _check(name: str, fn) -> bool:
    start = time.perf_counter()
    try:
        fn()
    except Exception as e:  # report every check, do not abort on the first
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return False
    print(f"  ok    {name} ({time.perf_counter() - start:.1f}s)")
    return True


def _eager() -> None:
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    torch.testing.assert_close(a @ b, (a.cpu() @ b.cpu()).cuda(), atol=1e-3, rtol=1e-3)


def _pointwise() -> None:
    def fn(x, y):
        return torch.sigmoid(x) * torch.tanh(y) + x.relu()

    x = torch.randn(1024, 256, device="cuda")
    y = torch.randn(1024, 256, device="cuda")
    ref = fn(x, y)
    out, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    if "@triton.jit" not in code:
        raise AssertionError("Inductor did not emit a Triton kernel")


def _reduction() -> None:
    def fn(x):
        return torch.softmax(x, dim=-1).sum(dim=0)

    x = torch.randn(512, 512, device="cuda")
    ref = fn(x)
    out, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    if "@triton.jit" not in code:
        raise AssertionError("Inductor did not emit a Triton kernel")


def _backward() -> None:
    def fn(x, w):
        return torch.nn.functional.gelu(x @ w).sum()

    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, 128, device="cuda", requires_grad=True)
    torch.compile(fn, fullgraph=True)(x, w).backward()
    if x.grad is None or w.grad is None:
        raise AssertionError("no gradients produced")
    if not (torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()):
        raise AssertionError("non-finite gradients")


def main() -> int:
    if not torch.cuda.is_available():
        print("no GPU available; nothing to check")
        return 1

    print(f"torch {torch.__version__} | triton {triton.__version__}")
    print(f"device {torch.cuda.get_device_name(0)}")
    print(
        f"TORCHINDUCTOR_TLX_MODE={os.environ.get('TORCHINDUCTOR_TLX_MODE', '<unset>')}"
    )

    start = time.perf_counter()
    results = [
        _check("eager matmul", _eager),
        _check("compile pointwise", _pointwise),
        _check("compile reduction", _reduction),
        _check("compile backward", _backward),
    ]
    elapsed = time.perf_counter() - start

    passed = sum(results)
    print(f"{passed}/{len(results)} passed in {elapsed:.1f}s")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
