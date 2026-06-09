"""
PrintHashTrace: A lightweight TorchDispatchMode for cross-backend hash comparison.

When wrapped around test computation blocks, prints the hash of every tensor
input/output via print() (printf-style). This allows comparing CI logs across
CUDA, XPU, MPS, and CPU to identify where numerical divergence begins.

Always active when hash_trace_wrapper() context manager is entered.
The caller controls scope by placing it around specific test blocks.

Usage in a test:
    from torch.testing._internal.hash_trace_utils import hash_trace_wrapper

    def test_conv_large(self, device):
        with hash_trace_wrapper():
            # ... computation to trace ...

Each traced op prints (to stdout):
    [HASH_TRACE] op_name
    [HASH_TRACE]   arg0: shape=... dtype=... hash=...
    [HASH_TRACE]   -> result: shape=... dtype=... hash=...

Output is grep-friendly: grep 'HASH_TRACE' test.log to extract trace.
"""

import hashlib
import os
import sys
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _hash_tensor(t: torch.Tensor) -> str:
    """Compute MD5 hash of tensor content (converted to fp32 if low-precision)."""
    is_low = t.dtype in (torch.float16, torch.bfloat16)
    t_cpu = t.detach().cpu()
    if is_low:
        t_cpu = t_cpu.float()
    return hashlib.md5(t_cpu.numpy().tobytes()).hexdigest()


class PrintHashTrace(TorchDispatchMode):
    """
    TorchDispatchMode that prints input/output tensor hashes for every ATen op.

    All output goes through print() (not files), using the [HASH_TRACE] prefix
    for easy grep extraction from CI logs.
    """

    def __init__(self, max_ops: int = 200):
        super().__init__()
        self._op_count = 0
        self._max_ops = max_ops
        self._active = True

    def _print(self, msg: str) -> None:
        if self._active and self._op_count <= self._max_ops:
            print(f"[HASH_TRACE] {msg}", flush=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self._op_count += 1

        if self._op_count > self._max_ops:
            return func(*args, **kwargs)

        func_name = getattr(func, "__name__", str(func))
        self._print(f"[{self._op_count:03d}] {func_name}")

        # Hash inputs
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                h = _hash_tensor(arg)
                self._print(
                    f"  arg{i}: shape={tuple(arg.shape)} dtype={arg.dtype} "
                    f"hash={h} device={arg.device}"
                )
            elif isinstance(arg, (int, float, bool, str)):
                self._print(f"  arg{i}: {arg}")
            elif arg is None:
                self._print(f"  arg{i}: None")

        # Execute
        result = func(*args, **kwargs)

        # Hash outputs
        if isinstance(result, torch.Tensor):
            h = _hash_tensor(result)
            self._print(
                f"  -> result: shape={tuple(result.shape)} dtype={result.dtype} "
                f"hash={h} device={result.device}"
            )
        elif isinstance(result, (tuple, list)):
            for j, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    h = _hash_tensor(r)
                    self._print(
                        f"  -> result[{j}]: shape={tuple(r.shape)} dtype={r.dtype} "
                        f"hash={h} device={r.device}"
                    )

        return result


def hash_trace_wrapper(max_ops: int = 200):
    """
    Context manager: prints tensor hashes for every ATen op within the block.

    Controlled by env var PYTORCH_TEST_HASH_TRACE:
        - Set to "1" to enable (default in CI via test.sh)
        - Unset → no-op (clean local test output)

    Usage:
        with hash_trace_wrapper():
            # ... computation to trace ...
    """
    from contextlib import nullcontext

    if os.environ.get("PYTORCH_TEST_HASH_TRACE") == "1":
        print("[HASH_TRACE] === Trace start ===", flush=True)
        return PrintHashTrace(max_ops=max_ops)
    return nullcontext()
