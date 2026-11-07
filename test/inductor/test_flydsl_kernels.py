# Owner(s): ["module: inductor"]
"""Tests for user-defined FlyDSL kernels (``@flyc.jit`` host functions) in PT2.

Modeled on the Triton suite (test/inductor/test_triton_kernels.py) and the OSS
CuTeDSL stack's test_cutedsl_kernels.py: direct HOP use, wrap_flydsl eager launches
and tracing (make_fx, non-strict export), the FlyDSL compile helpers, Dynamo,
Inductor (Python wrapper) and AOTInductor (export_to_c objects linked into the model
.so). Needs a ROCm GPU and flydsl >= 0.4, except FlyDSLAOTUnitTests (CPU, no flydsl:
codegen/flydsl/user_kernel_aot.py against synthetic exports and a fake compile).
"""

import contextlib
import ctypes
import dataclasses
import functools
import inspect
import io
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import weakref
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import sympy

import torch
import torch._dynamo.testing
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
)
from torch._dynamo.utils import ReinplaceCounters
from torch._higher_order_ops.flydsl_kernel_wrap import (
    flydsl_kernel_side_table,
    flydsl_kernel_wrapper_functional,
    flydsl_kernel_wrapper_mutation,
    FlyDSLKernel,
    get_flydsl_mutated_tensors,
    get_host_fn_info,
)
from torch._inductor import config as inductor_config
from torch._inductor.codegen.flydsl import user_kernel_aot as fuka
from torch._inductor.runtime import flydsl_user_kernel
from torch._inductor.runtime.flydsl_user_kernel import (
    FLYDSL_COMPILE_LOCK,
    flydsl_version_ok,
)
from torch._inductor.test_case import TestCase
from torch._inductor.utils import IndentedBuffer, run_and_get_code, run_and_get_cpp_code
from torch._library.flydsl import wrap_flydsl
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


HAS_ROCM = torch.cuda.is_available() and torch.version.hip is not None

# The DSL is an optional dependency: define the toy kernels only when it (and a ROCm
# device) is available.
if not HAS_ROCM:
    _ENV_SKIP_REASON = "requires a ROCm GPU"
elif not flydsl_version_ok():
    _ENV_SKIP_REASON = "requires flydsl >= 0.4"
else:
    _ENV_SKIP_REASON = None

HAS_FLYDSL = _ENV_SKIP_REASON is None
requires_flydsl = unittest.skipUnless(HAS_FLYDSL, _ENV_SKIP_REASON or "")

# A module global named like an attribute the kernels use (fx.thread_idx.x): FlyDSL's
# globals-drift check treats it as a global the kernels read. See
# test_wrap_flydsl_globals_drift_does_not_trigger.
x = None

# Module globals kernels read: scale_global_kernel's, which the tests rebind, one a
# kernel of test_globals_key reads, and one declared but not defined (yet).
_GLOBAL_SCALE = 2.0
_OTHER_SCALE = 3.0
_DEFINED_LATER: int

if HAS_FLYDSL:
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    # --- BEGIN TOY KERNELS ---
    # 1-D kernels use 1024 threads per block and guard with the runtime n, like the
    # Triton toys.

    @flyc.kernel
    def _add_k(X, Y, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] + Y[i]

    @flyc.jit
    def add_kernel(X, Y, Out, n: fx.Int32, stream: fx.Stream):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Reads and writes X: x *= 2.
    @flyc.kernel
    def _scale_inplace_k(X, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            X[i] = X[i] * 2.0

    @flyc.jit
    def scale_inplace_kernel(X, n: fx.Int32, stream: fx.Stream):
        _scale_inplace_k(X, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Accumulates, out += x: every launch is visible in the result.
    @flyc.kernel
    def _acc_k(X, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = Out[i] + X[i]

    @flyc.jit
    def acc_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
        _acc_k(X, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # 2-D: out[i, j] = x[i, j] + 1, indexing through each tensor's own layout, so
    # non-contiguous inputs and outputs work. The extent comes from X's (static)
    # shape: there is no scalar argument at all.
    @flyc.kernel
    def _add_one_2d_k(X, Out):
        i = fx.block_idx.x
        j = fx.thread_idx.x
        Out[i, j] = X[i, j] + 1.0

    @flyc.jit
    def add_one_2d_kernel(X, Out, stream: fx.Stream):
        m, n = X.shape.unpack()
        _add_one_2d_k(X, Out).launch(grid=(m, 1, 1), block=(n, 1, 1), stream=stream)

    # Constexpr param: alpha is baked into the compiled kernel.
    @flyc.kernel
    def _mul_const_k(X, Out, n: fx.Int32, alpha: fx.Constexpr[float]):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] * alpha

    @flyc.jit
    def mul_const_kernel(
        X, Out, n: fx.Int32, alpha: fx.Constexpr[float], stream: fx.Stream
    ):
        _mul_const_k(X, Out, n, alpha).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Two outputs, plus a runtime Float32 scalar with a default that sits after the
    # stream param: sum = (x + y) * scale, diff = (x - y) * scale.
    @flyc.kernel
    def _add_sub_k(X, Y, Sum, Diff, n: fx.Int32, scale: fx.Float32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Sum[i] = (X[i] + Y[i]) * scale
            Diff[i] = (X[i] - Y[i]) * scale

    @flyc.jit
    def add_sub_kernel(
        X, Y, Sum, Diff, n: fx.Int32, stream: fx.Stream, scale: fx.Float32 = 1.0
    ):
        _add_sub_k(X, Y, Sum, Diff, n, scale).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Unannotated scalar k: out = x + k. FlyDSL types it from the value (int: Int32,
    # float: Float32, bool: Boolean), so it is a runtime argument.
    @flyc.kernel
    def _add_scalar_k(X, Out, n: fx.Int32, k):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] + k

    @flyc.jit
    def add_scalar_kernel(X, Out, n: fx.Int32, k, stream: fx.Stream):
        _add_scalar_k(X, Out, n, k).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Two different host functions with the same __name__ and __qualname__: the
    # compile caches must key on the function object, not its name.
    @flyc.kernel
    def _sub_k(X, Y, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] - Y[i]

    @flyc.jit
    def binary_kernel(X, Y, Out, n: fx.Int32, stream: fx.Stream):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    same_name_add_kernel = binary_kernel

    @flyc.jit
    def binary_kernel(X, Y, Out, n: fx.Int32, stream: fx.Stream):
        _sub_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    same_name_sub_kernel = binary_kernel

    # Invalid for wrap_flydsl (it needs exactly one fx.Stream param), but valid
    # FlyDSL: without a stream parameter FlyDSL launches on the null stream.
    @flyc.jit
    def no_stream_kernel(X, Y, Out, n: fx.Int32):
        _add_k(X, Y, Out, n).launch(grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1))

    @flyc.jit
    def two_stream_kernel(
        X, Y, Out, n: fx.Int32, stream: fx.Stream, stream2: fx.Stream
    ):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # --- END TOY KERNELS ---

    # --- BEGIN CASE KERNELS ---
    # Kernels for specific cases of the tests below.

    # A runtime Float32 scalar whose default is a FlyDSL Numeric value: out = x + c.
    @flyc.kernel
    def _add_c_k(X, Out, n: fx.Int32, c: fx.Float32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] + c

    @flyc.jit
    def add_c_kernel(
        X, Out, n: fx.Int32, stream: fx.Stream, c: fx.Float32 = fx.Float32(0.25)
    ):
        _add_c_k(X, Out, n, c).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # A Constexpr tuple read on the host: out = x * scales[0].
    @flyc.jit
    def scale_by_first_kernel(
        X, Out, n: fx.Int32, scales: fx.Constexpr[tuple], stream: fx.Stream
    ):
        _mul_const_k(X, Out, n, scales[0]).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # An unannotated flag, which FlyDSL makes a runtime Boolean: out = x * 3 if flag
    # else x * 5.
    @flyc.kernel
    def _select_k(X, Out, n: fx.Int32, flag: fx.Boolean):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            if flag:
                Out[i] = X[i] * 3.0
            else:
                Out[i] = X[i] * 5.0

    @flyc.jit
    def select_kernel(X, Out, n: fx.Int32, flag, stream: fx.Stream):
        _select_k(X, Out, n, flag).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # The same with an annotated flag (a Boolean in the exported C ABI).
    @flyc.jit
    def select_bool_kernel(X, Out, n: fx.Int32, flag: fx.Boolean, stream: fx.Stream):
        _select_k(X, Out, n, flag).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # A runtime Float16 scalar, which export_to_c passes as raw bits.
    @flyc.jit
    def add_half_kernel(X, Out, n: fx.Int32, h: fx.Float16, stream: fx.Stream):
        _add_c_k(X, Out, n, fx.Float32(h)).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Host functions that differ only in a captured list: out = x * c[0] + c[1].
    def make_affine_kernel(c):
        @flyc.kernel
        def _affine_k(X, Out, n: fx.Int32):
            i = fx.block_idx.x * 1024 + fx.thread_idx.x
            if i < n:
                Out[i] = X[i] * c[0] + c[1]

        @flyc.jit
        def affine_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
            _affine_k(X, Out, n).launch(
                grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
            )

        return affine_kernel

    # Only for test_wrap_flydsl_globals_drift_does_not_trigger (it rebinds the module
    # global x, which this kernel "uses" through fx.thread_idx.x).
    @flyc.jit
    def drift_kernel(X, Y, Out, n: fx.Int32, stream: fx.Stream):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # out = x * _GLOBAL_SCALE, a module global FlyDSL bakes into the kernel.
    @flyc.kernel
    def _scale_global_k(X, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] * _GLOBAL_SCALE

    @flyc.jit
    def scale_global_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
        _scale_global_k(X, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # A Constexpr lambda: out = op(x).
    @flyc.kernel
    def _map_k(X, Out, n: fx.Int32, op: fx.Constexpr[Callable]):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = op(X[i])

    @flyc.jit
    def map_kernel(X, Out, n: fx.Int32, op: fx.Constexpr[Callable], stream: fx.Stream):
        _map_k(X, Out, n, op).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # A lambda inside a Constexpr tuple: out = ops[0](x).
    @flyc.jit
    def map_first_kernel(
        X, Out, n: fx.Int32, ops: fx.Constexpr[tuple], stream: fx.Stream
    ):
        _map_k(X, Out, n, ops[0]).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Keyword-only parameters (after *): the stream, a scalar, a tensor, a Constexpr
    # with a default.
    @flyc.jit
    def kwonly_stream_kernel(X, Y, Out, n: fx.Int32, *, stream: fx.Stream):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    @flyc.jit
    def kwonly_n_kernel(X, Y, Out, *, n: fx.Int32, stream: fx.Stream):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    @flyc.jit
    def kwonly_out_kernel(X, Y, n: fx.Int32, stream: fx.Stream, *, Out):
        _add_k(X, Y, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    @flyc.jit
    def kwonly_alpha_kernel(
        X, Out, n: fx.Int32, stream: fx.Stream, *, alpha: fx.Constexpr[float] = 2.0
    ):
        _mul_const_k(X, Out, n, alpha).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # Unannotated parameters whose defaults are FlyDSL Numeric values, which FlyDSL
    # compiles as their own types: out = x + k * k (int64), out = x + c (float64).
    @flyc.kernel
    def _add_square_k(X, Out, n: fx.Int32, k):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] + k * k

    @flyc.jit
    def int64_default_kernel(
        X, Out, n: fx.Int32, stream: fx.Stream, k=fx.Int64(100000)
    ):
        _add_square_k(X, Out, n, k).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    @flyc.jit
    def big_int64_default_kernel(
        X, Out, n: fx.Int32, stream: fx.Stream, k=fx.Int64(3_000_000_000)
    ):
        _add_square_k(X, Out, n, k).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    @flyc.jit
    def float64_default_kernel(
        X, Out, n: fx.Int32, stream: fx.Stream, c=fx.Float64(0.1)
    ):
        _add_scalar_k(X, Out, n, c).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # --- END CASE KERNELS ---


DEVICE = "cuda"
# Error types acceptable for user errors (bad arguments to wrap_flydsl / a wrapper).
USER_ERRORS = (TypeError, ValueError, RuntimeError)

if HAS_FLYDSL:
    # Module-level (global) wrappers, reused across tests, between which
    # torch._dynamo.reset() runs (it resets the Triton kernel side table; the FlyDSL
    # records these wrappers hold must keep resolving).
    add_w = wrap_flydsl(add_kernel, mutates_args=["Out"])
    scale_inplace_w = wrap_flydsl(scale_inplace_kernel, mutates_args=["X"])
    acc_w = wrap_flydsl(acc_kernel, mutates_args=["Out"])
    add_one_2d_w = wrap_flydsl(add_one_2d_kernel, mutates_args=["Out"])
    mul_const_w = wrap_flydsl(mul_const_kernel, mutates_args=["Out"])
    add_sub_w = wrap_flydsl(add_sub_kernel, mutates_args=["Sum", "Diff"])
    add_scalar_w = wrap_flydsl(add_scalar_kernel, mutates_args=["Out"])
    same_name_add_w = wrap_flydsl(same_name_add_kernel, mutates_args=["Out"])
    same_name_sub_w = wrap_flydsl(same_name_sub_kernel, mutates_args=["Out"])


def _rand(*shape, dtype=torch.float32, requires_grad=False):
    return torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=requires_grad)


def _raw_stream():
    return torch.cuda.current_stream().cuda_stream


def _hop_nodes(gm, hop):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is hop]


def _is_tensor_node(v):
    return isinstance(v, torch.fx.Node) and isinstance(v.meta.get("val"), torch.Tensor)


def _scalar_args(node):
    """The non-tensor kernel args of a FlyDSL HOP node, wherever the capture put
    them: scalars inside the node's ``kwargs`` dict (FX nodes when symbolic) plus the
    side-table constant args."""
    args = {k: v for k, v in node.kwargs["kwargs"].items() if not _is_tensor_node(v)}
    args.update(
        flydsl_kernel_side_table.get_constant_args(node.kwargs["constant_args_idx"])
    )
    return args


def _normalize_indices(code):
    """Side-table indices depend on what ran before; blank them for expected code."""
    code = re.sub(r"\bkernel_idx = \d+", "kernel_idx = K", code)
    return re.sub(r"\bconstant_args_idx = \d+", "constant_args_idx = C", code)


def _strip_comments(code):
    """Inductor output code carries `Original ATen:` / source comments that would
    make positive substring checks pass vacuously."""
    lines = (ln for ln in code.splitlines() if not ln.lstrip().startswith(("#", "//")))
    return "\n".join(lines)


def _flydsl_launches(code, host_name):
    """`<launcher>.run(...)` calls of the user kernel ``host_name`` in Python wrapper
    code, as (launcher name, args) pairs. The launcher is named after the host
    function (`flydsl_add_kernel_0.run(...)`)."""
    pattern = rf"\b(flydsl_{re.escape(host_name)}_\d+)\.run\(([^\n]*)\)"
    return re.findall(pattern, _strip_comments(code))


def _pointer_alignment(t):
    ptr = t.data_ptr()
    return next(a for a in (16, 8, 4, 2, 1) if ptr % a == 0)


def _reference_launch(host_fn, *args, **kwargs):
    """Launch ``host_fn`` with plain FlyDSL, no PT2 involved: a direct call of the
    JitFunction with static-layout tensor arguments (from_torch_tensor, with the
    alignment their data pointer really has) and the current stream. Positional
    ``args`` bind to the non-stream params in order."""
    params = list(inspect.signature(host_fn.func).parameters.values())
    stream_name = next(p.name for p in params if p.annotation is fx.Stream)
    bound = dict(zip((p.name for p in params if p.name != stream_name), args))
    bound.update(kwargs)
    bound[stream_name] = _raw_stream()
    for name, v in bound.items():
        if isinstance(v, torch.Tensor):
            bound[name] = flyc.from_torch_tensor(v, assumed_align=_pointer_alignment(v))
    host_fn(**bound)


def _add_kernel_to_table(fn, mutates_args):
    """Side-table index of the FlyDSLKernel record for ``fn``."""
    if mutates_args != "unknown":
        mutates_args = tuple(mutates_args)
    return flydsl_kernel_side_table.add_kernel(FlyDSLKernel(fn, mutates_args))


def _add_kwargs(x, y, out, n=None):
    """HOP ``kwargs`` for add_kernel: tensors and the runtime scalar n."""
    return {"X": x, "Y": y, "Out": out, "n": out.numel() if n is None else n}


def _record_constant_args_indices(indices):
    """A patch of flydsl_kernel_side_table.add_constant_args that appends the index
    of every call's constant args to ``indices``. The table is process-global and
    never reset, so a test compares these indices instead of counting its entries."""
    add_constant_args = flydsl_kernel_side_table.add_constant_args

    def spy(args):
        indices.append(add_constant_args(args))
        return indices[-1]

    return mock.patch.object(flydsl_kernel_side_table, "add_constant_args", spy)


def _lock_held_elsewhere():
    """Whether another thread holds FLYDSL_COMPILE_LOCK (an RLock, which the calling
    thread could re-acquire)."""
    result = []

    def probe():
        acquired = FLYDSL_COMPILE_LOCK.acquire(blocking=False)
        if acquired:
            FLYDSL_COMPILE_LOCK.release()
        result.append(not acquired)

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return result[0]


@requires_flydsl
class FlyDSLToyKernelTests(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_toy_kernels_plain_flydsl(self):
        # The toy kernels themselves, through plain FlyDSL (no PT2): if this fails
        # the environment is broken, not the PT2 integration.
        x, y = _rand(2048), _rand(2048)
        out = torch.zeros_like(x)
        _reference_launch(add_kernel, x, y, out, 2048)
        self.assertEqual(out, x + y)
        x2 = _rand(4, 8)
        out2 = torch.zeros(4, 8, device=DEVICE)
        _reference_launch(add_one_2d_kernel, x2, out2)
        self.assertEqual(out2, x2 + 1)
        s, d = torch.zeros_like(x), torch.zeros_like(x)
        _reference_launch(add_sub_kernel, x, y, s, d, 2048, scale=0.5)
        self.assertEqual(s, (x + y) * 0.5)
        self.assertEqual(d, (x - y) * 0.5)


@requires_flydsl
class FlyDSLHOPTests(TestCase):
    """Direct use of the HOPs and the side table (no wrap_flydsl)."""

    hw_classification = HardwareClassification.CUDA

    def test_side_table_dedupes_constant_args(self):
        # Eager wrapper calls go through the HOP and this table is never reset, so
        # equal constants share one entry; equal values of other types do not.
        table = flydsl_kernel_side_table
        idx = table.add_constant_args({"alpha": 2, "k": None})
        self.assertEqual(table.add_constant_args({"k": None, "alpha": 2}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": 2.0, "k": None}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": True, "k": None}), idx)
        # FlyDSL Numeric values by type and value.
        num = table.add_constant_args({"c": fx.Float32(2.0)})
        self.assertEqual(table.add_constant_args({"c": fx.Float32(2.0)}), num)
        self.assertNotEqual(table.add_constant_args({"c": fx.Float32(3.0)}), num)
        self.assertNotEqual(table.add_constant_args({"c": fx.Float64(2.0)}), num)

    def test_runtime_scalars_reach_the_dsl_as_python_values(self):
        # FlyDSL converts only exact bool/int/float: an IntEnum for an unannotated
        # param, and a FlyDSL Numeric default, are passed as plain Python values.
        import enum

        class K(enum.IntEnum):
            FIVE = 5

        x = _rand(1000)
        out = torch.empty_like(x)
        flydsl_kernel_wrapper_mutation(
            kernel_idx=_add_kernel_to_table(add_scalar_kernel, ["Out"]),
            constant_args_idx=flydsl_kernel_side_table.add_constant_args({}),
            kwargs={"X": x, "Out": out, "n": 1000, "k": K.FIVE},
        )
        self.assertEqual(out, x + 5)

        default = get_host_fn_info(add_c_kernel).defaults["c"]
        self.assertIs(type(default), float)
        self.assertEqual(default, 0.25)
        out = torch.empty_like(x)
        flydsl_kernel_wrapper_mutation(
            kernel_idx=_add_kernel_to_table(add_c_kernel, ["Out"]),
            constant_args_idx=flydsl_kernel_side_table.add_constant_args({}),
            kwargs={"X": x, "Out": out, "n": 1000},
        )
        self.assertEqual(out, x + 0.25)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPUs")
    def test_launch_on_two_devices(self):
        # A compiled FlyDSL kernel is bound to the device of its first launch, so the
        # same specialization is compiled once per device. The launch switches to
        # the tensors' device; tensors on two devices are refused.
        torch.empty(1, device="cuda:0")  # a context on the current device
        idx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})
        n = 1237  # a size no other test uses: a compile-cache miss
        for device in ("cuda:1", "cuda:0", "cuda:1"):
            x, y = _rand(n).to(device), _rand(n).to(device)
            out = torch.zeros_like(x)
            with torch.cuda.device(0):
                flydsl_kernel_wrapper_mutation(
                    kernel_idx=idx,
                    constant_args_idx=cidx,
                    kwargs=_add_kwargs(x, y, out),
                )
            self.assertEqual(out, x + y)
        with self.assertRaisesRegex(ValueError, "one CUDA device"):
            flydsl_kernel_wrapper_mutation(
                kernel_idx=idx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(x, y.to("cuda:0"), out),
            )

    def test_side_table_records(self):
        import dataclasses

        r1 = FlyDSLKernel(add_kernel, ("Out",))
        r2 = FlyDSLKernel(add_kernel, ("Out",))
        self.assertEqual(r1, r2)
        self.assertEqual(hash(r1), hash(r2))
        self.assertEqual(r1.name, "add_kernel")
        untyped_record: Any = r1
        with self.assertRaises(dataclasses.FrozenInstanceError):
            untyped_record.mutates_args = "unknown"
        idx = flydsl_kernel_side_table.add_kernel(r1)
        # Equal records share an index; a different write set is another kernel.
        self.assertEqual(flydsl_kernel_side_table.add_kernel(r2), idx)
        self.assertIs(flydsl_kernel_side_table.get_kernel(idx).fn, add_kernel)
        self.assertNotEqual(_add_kernel_to_table(add_kernel, "unknown"), idx)

    def test_flydsl_hop_eager(self):
        # Mirrors test_triton_kernel_higher_order_func.
        kidx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})
        x, y = _rand(1000), _rand(1000)

        out = torch.zeros_like(x)
        ret = flydsl_kernel_wrapper_mutation(
            kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
        )
        self.assertIsNone(ret)
        self.assertEqual(out, x + y)

        # The functional HOP launches on a clone and leaves its input alone.
        out = torch.zeros_like(x)
        out_dict = flydsl_kernel_wrapper_functional(
            kernel_idx=kidx,
            constant_args_idx=cidx,
            kwargs=_add_kwargs(x, y, out),
            tensors_to_clone=["Out"],
        )
        self.assertEqual(set(out_dict), {"Out"})
        self.assertEqual(out_dict["Out"], x + y)
        self.assertEqual(out, torch.zeros_like(x))

    def test_flydsl_hop_functional_clone_keeps_unwritten_elements(self):
        # n < numel: the kernel writes only the head, so the clone must carry the
        # input's tail (a clone, not a fresh allocation).
        kidx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})
        x, y = _rand(1000), _rand(1000)
        out = torch.full_like(x, 7.0)
        res = flydsl_kernel_wrapper_functional(
            kernel_idx=kidx,
            constant_args_idx=cidx,
            kwargs=_add_kwargs(x, y, out, n=500),
            tensors_to_clone=["Out"],
        )["Out"]
        self.assertEqual(res[:500], x[:500] + y[:500])
        self.assertEqual(res[500:], torch.full_like(x[500:], 7.0))
        self.assertEqual(out, torch.full_like(x, 7.0))

    def test_flydsl_hop_eager_inplace_and_2d(self):
        kidx = _add_kernel_to_table(scale_inplace_kernel, ["X"])
        cidx = flydsl_kernel_side_table.add_constant_args({})
        x = torch.arange(1000, device=DEVICE, dtype=torch.float32)
        flydsl_kernel_wrapper_mutation(
            kernel_idx=kidx, constant_args_idx=cidx, kwargs={"X": x, "n": 1000}
        )
        self.assertEqual(x, 2 * torch.arange(1000, device=DEVICE, dtype=torch.float32))

        kidx = _add_kernel_to_table(add_one_2d_kernel, ["Out"])
        xt = _rand(8, 4).t()  # strides (1, 4)
        out = torch.zeros(4, 8, device=DEVICE)
        flydsl_kernel_wrapper_mutation(
            kernel_idx=kidx, constant_args_idx=cidx, kwargs={"X": xt, "Out": out}
        )
        self.assertEqual(out, xt + 1)

    def test_flydsl_hop_constant_args(self):
        # Constexpr values travel in the side table; each distinct constant is a
        # distinct compiled kernel.
        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_const_kernel, ["Out"])
        for alpha in (3.0, 2.5):
            out = torch.zeros_like(x)
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=flydsl_kernel_side_table.add_constant_args(
                    {"alpha": alpha}
                ),
                kwargs={"X": x, "Out": out, "n": 1000},
            )
            self.assertEqual(out, x * alpha)

    def test_flydsl_hop_fake_mode(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        kidx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})
        with FakeTensorMode():
            x, y, out = (torch.empty(1000, device=DEVICE) for _ in range(3))
            self.assertIsNone(
                flydsl_kernel_wrapper_mutation(
                    kernel_idx=kidx,
                    constant_args_idx=cidx,
                    kwargs=_add_kwargs(x, y, out),
                )
            )
            res = flydsl_kernel_wrapper_functional(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(x, y, out),
                tensors_to_clone=["Out"],
            )
            self.assertEqual(res["Out"].shape, out.shape)
            self.assertEqual(res["Out"].stride(), out.stride())

    @parametrize("tracing_mode", ["fake", "symbolic"])
    def test_flydsl_hop_make_fx(self, tracing_mode):
        kidx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            out = torch.zeros_like(x)
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return out

        gm = make_fx(f, tracing_mode=tracing_mode)(_rand(1000), _rand(1000))
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
        self.assertEqual(node.name, "flydsl_kernel_wrapper_mutation_proxy")
        self.assertEqual(node.kwargs["kernel_idx"], kidx)
        self.assertEqual(node.kwargs["constant_args_idx"], cidx)
        hop_kwargs = node.kwargs["kwargs"]
        self.assertEqual(list(hop_kwargs), ["X", "Y", "Out", "n"])
        self.assertEqual(hop_kwargs["X"].op, "placeholder")
        if tracing_mode == "symbolic":
            self.assertIsInstance(hop_kwargs["n"], torch.fx.Node)
        else:
            self.assertEqual(hop_kwargs["n"], 1000)
            self.assertExpectedInline(
                _normalize_indices(gm.code.strip()),
                """\
def forward(self, x_1, y_1):
    zeros_like = torch.ops.aten.zeros_like.default(x_1, pin_memory = False)
    flydsl_kernel_wrapper_mutation_proxy = torch.ops.higher_order.flydsl_kernel_wrapper_mutation(kernel_idx = K, constant_args_idx = C, kwargs = {'X': x_1, 'Y': y_1, 'Out': zeros_like, 'n': 1000});  x_1 = y_1 = flydsl_kernel_wrapper_mutation_proxy = None
    return zeros_like""",
            )
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(gm(x, y), x + y)
        if tracing_mode == "symbolic":
            x, y = _rand(300), _rand(300)
            self.assertEqual(gm(x, y), x + y)

    def test_flydsl_hop_functionalize(self):
        # Mirrors test_triton_kernel_functionalize: functionalization rewrites the
        # mutation HOP into the functional HOP cloning exactly mutates_args.
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            out = torch.zeros_like(x)
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return out

        x, y = _rand(1000), _rand(1000)
        with FunctionalTensorMode():
            gm_py = make_fx(PythonFunctionalizeAPI().functionalize(f))(x, y)
        gm_cpp = make_fx(CppFunctionalizeAPI().functionalize(f))(x, y)
        gm_func = make_fx(torch.func.functionalize(f))(x, y)
        for gm in (gm_py, gm_cpp, gm_func):
            self.assertEqual(_hop_nodes(gm, flydsl_kernel_wrapper_mutation), [])
            (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_functional)
            self.assertEqual(node.kwargs["tensors_to_clone"], ["Out"])
            self.assertNotIn("tensor_bases", node.kwargs)
            self.assertEqual(gm(x, y), x + y)

        gm = make_fx(torch.func.functionalize(f), tracing_mode="fake")(x, y)
        self.assertExpectedInline(
            _normalize_indices(gm.code.strip()),
            """\
def forward(self, x_1, y_1):
    zeros_like = torch.ops.aten.zeros_like.default(x_1, pin_memory = False)
    flydsl_kernel_wrapper_functional_proxy = torch.ops.higher_order.flydsl_kernel_wrapper_functional(kernel_idx = K, constant_args_idx = C, kwargs = {'X': x_1, 'Y': y_1, 'Out': zeros_like, 'n': 1000}, tensors_to_clone = ['Out']);  x_1 = y_1 = zeros_like = None
    getitem = flydsl_kernel_wrapper_functional_proxy['Out'];  flydsl_kernel_wrapper_functional_proxy = None
    return getitem""",
        )

    def test_flydsl_hop_functionalize_inplace(self):
        # A kernel that reads and writes the same argument: only it is cloned, and
        # the mutation of the graph input is written back.
        kidx = _add_kernel_to_table(scale_inplace_kernel, ["X"])
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def f(x):
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"X": x, "n": x.numel()},
            )
            return x

        x = _rand(1000)
        gm = make_fx(torch.func.functionalize(f))(x.clone())
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["X"])
        x_in = x.clone()
        self.assertEqual(gm(x_in), 2 * x)
        self.assertEqual(x_in, 2 * x)

    @parametrize("view", ["offset", "gap"])
    def test_flydsl_hop_functionalize_view_arg(self, view):
        # A mutated arg that is a view is cloned through its base: the functional
        # HOP gets tensor_bases. "gap" makes the view non-dense (cf.
        # test_triton_kernel_mutated_offset_view_not_miscompiled).
        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(scale_inplace_kernel, ["X"])
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def take_view(base):
            return base[4:] if view == "offset" else base[::2]

        def f(base):
            v = take_view(base)
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"X": v, "n": v.numel()},
            )
            return base

        base = torch.arange(1, 1001, device=DEVICE, dtype=torch.float32)
        expected = base.clone()
        take_view(expected).mul_(2)

        # Python functionalization (what AOTAutograd runs): the view's base is a
        # tracked value, so it is passed as tensor_bases.
        with FunctionalTensorMode():
            gm = make_fx(PythonFunctionalizeAPI().functionalize(f))(base.clone())
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["X"])
        self.assertEqual(list(node.kwargs["tensor_bases"]), ["X"])
        self.assertEqual(gm(base.clone()), expected)

        # torch.func.functionalize: no tensor_bases (as for Triton); the result and
        # the write-back into the input must still be right.
        gm = make_fx(torch.func.functionalize(f))(base.clone())
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["X"])
        b = base.clone()
        self.assertEqual(gm(b), expected)
        self.assertEqual(b, expected)

    @parametrize("api", ["python", "cpp"])
    def test_flydsl_hop_functionalize_refuses_aliased_writes(self, api):
        # Each written argument is cloned and written back on its own, so two written
        # arguments that may overlap (here X and Out, both in mutates_args) would lose
        # a write: functionalization refuses the same tensor twice and overlapping
        # views. Two rows of one base, proven disjoint, are written back correctly.
        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(add_kernel, ["X", "Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def trace(f, *args):
            if api == "cpp":
                return make_fx(torch.func.functionalize(f))(*args)
            with FunctionalTensorMode():
                return make_fx(PythonFunctionalizeAPI().functionalize(f))(*args)

        def in_place(x, y):
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, x)
            )
            return x

        def shifted(x, y):
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(x[1:], y[1:], x[:-1]),
            )
            return x

        def views(base, y):
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(base[0], y, base[1]),
            )
            return base

        x, y = _rand(1000), _rand(1000)
        for f in (in_place, shifted):
            with self.assertRaisesRegex(
                RuntimeError,
                r"^FlyDSL kernel add_kernel: 'X' and 'Out' are the same tensor or views "
                r"that may overlap, and both count as written "
                r"\(mutates_args=\('X', 'Out'\)\)\. PT2 clones each written argument "
                r"and writes it back on its own, so one write could be lost\. Pass "
                r"tensors that do not overlap\.$",
            ):
                trace(f, x.clone(), y)
        base = torch.stack([x, torch.zeros_like(x)])
        gm = trace(views, base.clone(), y)
        self.assertEqual(gm(base.clone(), y), torch.stack([x, x + y]))

    def test_flydsl_hop_unknown_mutates_args_clones_every_tensor(self):
        kidx = _add_kernel_to_table(add_kernel, "unknown")
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            out = torch.zeros_like(x)
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return out

        x, y = _rand(1000), _rand(1000)
        gm = make_fx(torch.func.functionalize(f))(x, y)
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_functional)
        self.assertEqual(sorted(node.kwargs["tensors_to_clone"]), ["Out", "X", "Y"])
        x_in, y_in = x.clone(), y.clone()
        self.assertEqual(gm(x_in, y_in), x + y)
        self.assertEqual((x_in, y_in), (x, y))

    @parametrize("case", ["e8m0", "complex", "meta", "sparse", "fp8"])
    def test_flydsl_hop_functional_decomposed_before_type_fallback(self, case):
        # The post-grad decomposition turns the functional HOP into the mutation HOP
        # whatever its tensors' types (the lowering checks them afterwards), instead
        # of skipping the node and failing its "was not removed" check.
        import contextlib

        import torch._inductor.lowering as lowering
        from torch._inductor.fx_passes.post_grad import (
            decompose_user_kernel_wrapper_functional,
        )

        kidx = _add_kernel_to_table(add_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({"n": 16})

        def f(src, output):
            out = flydsl_kernel_wrapper_functional(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"X": src, "Y": src, "Out": output},
                tensors_to_clone=["Out"],
            )
            return out["Out"]

        dtype = {
            "e8m0": torch.float8_e8m0fnu,
            "complex": torch.complex64,
            "fp8": torch.float8_e4m3fn,
        }.get(case, torch.float32)
        device = {"meta": "meta", "sparse": "cpu"}.get(case, DEVICE)
        src = torch.empty(16, dtype=dtype, device=device)
        output = torch.empty(16, device=device)
        if case == "sparse":
            src = torch.sparse_coo_tensor([[0]], [1.0], (16,))
        fp8_unsupported = (
            mock.patch.object(
                lowering, "is_triton_fp8_dtype_supported", return_value=False
            )
            if case == "fp8"
            else contextlib.nullcontext()
        )
        with fp8_unsupported:
            gm = make_fx(f, tracing_mode="fake")(src, output)
            self.assertEqual(len(_hop_nodes(gm, flydsl_kernel_wrapper_functional)), 1)
            decompose_user_kernel_wrapper_functional(gm.graph)
            self.assertEqual(len(_hop_nodes(gm, flydsl_kernel_wrapper_mutation)), 1)

    def test_get_flydsl_mutated_tensors(self):
        x, y, s, d = (_rand(16) for _ in range(4))
        kwargs = {"X": x, "Y": y, "Sum": s, "Diff": d, "n": 16, "scale": 1.0}
        cidx = flydsl_kernel_side_table.add_constant_args({})
        kidx = _add_kernel_to_table(add_sub_kernel, ["Sum", "Diff"])
        self.assertEqual(
            sorted(get_flydsl_mutated_tensors(kidx, cidx, kwargs)), ["Diff", "Sum"]
        )
        # "unknown": every tensor argument, never the scalars.
        kidx = _add_kernel_to_table(add_sub_kernel, "unknown")
        self.assertEqual(
            sorted(get_flydsl_mutated_tensors(kidx, cidx, kwargs)),
            ["Diff", "Sum", "X", "Y"],
        )

    def test_flydsl_hop_mutation_hidden_from_autograd(self):
        # Mirrors test_triton_kernel_mutation_type.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        kidx = _add_kernel_to_table(scale_inplace_kernel, ["X"])
        cidx = flydsl_kernel_side_table.add_constant_args({})

        def prep():
            x = torch.ones(4, device=DEVICE, requires_grad=True)
            with FunctionalTensorMode():
                x_func = FunctionalTensor.to_functional(x)
            self.assertTrue(torch._is_functional_tensor(x_func.elem))
            return x_func

        def launch(x_func):
            flydsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs={"X": x_func, "n": 4}
            )

        # kernel mutation only
        with FakeTensorMode():
            x_func = prep()
            with FunctionalTensorMode():
                launch(x_func)
            self.assertTrue(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

        # normal mutation + kernel mutation
        with FakeTensorMode():
            x_func = prep()
            with FunctionalTensorMode():
                x_func.mul_(2)
                launch(x_func)
            self.assertFalse(
                torch._functionalize_are_all_mutations_hidden_from_autograd(x_func.elem)
            )

    def test_flydsl_side_table_is_separate(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        self.assertIsNot(flydsl_kernel_side_table, kernel_side_table)
        record = FlyDSLKernel(add_kernel, ("Out",))
        idx = flydsl_kernel_side_table.add_kernel(record)
        # torch._dynamo.reset() resets Triton's table, not this one.
        torch._dynamo.reset()
        self.assertEqual(flydsl_kernel_side_table.get_kernel(idx), record)

    def test_flydsl_constant_rules(self):
        # FlyDSL's rules key a Numeric by its type and value, and the side table holds
        # a copy of it; any other FlyDSL value, such as a Basis (fx.E(0), which has
        # modes besides its value), is keyed as itself.
        from torch._higher_order_ops.flydsl_kernel_wrap import (
            _constant_key,
            FLYDSL_CONSTANT_RULES,
        )

        rules = FLYDSL_CONSTANT_RULES
        self.assertEqual(rules.scalar_value(fx.Int32(5)), 5)
        self.assertEqual(_constant_key(fx.Int32(5)), _constant_key(fx.Int32(5)))
        self.assertNotEqual(_constant_key(fx.Int32(5)), _constant_key(fx.Int32(6)))
        self.assertNotEqual(_constant_key(fx.Int32(5)), _constant_key(fx.Int64(5)))
        self.assertIsNone(rules.scalar_value(fx.E(0)))
        self.assertEqual(_constant_key(fx.E(0)), _constant_key(fx.E(0)))
        self.assertNotEqual(_constant_key(fx.E(0)), _constant_key(fx.E(1)))
        numeric = fx.Float32(0.5)
        idx = flydsl_kernel_side_table.add_constant_args({"c": numeric})
        numeric.value = 2.0
        self.assertEqual(
            flydsl_kernel_side_table.get_constant_args(idx)["c"].value, 0.5
        )

    def test_flydsl_hop_info(self):
        # The pair, its side table and its HOPifier are registered with the shared
        # user-kernel HOP registry, which the DSL-agnostic passes consult.
        from torch._higher_order_ops import flydsl_kernel_wrap
        from torch._higher_order_ops.reinplace_trigger import ReInplaceTrigger
        from torch._higher_order_ops.user_kernel_hop import get_user_kernel_hop_info

        info = flydsl_kernel_wrap.flydsl_kernel_hop_info
        self.assertEqual((info.dsl, info.display_name), ("flydsl", "FlyDSL"))
        self.assertIs(info.side_table, flydsl_kernel_side_table)
        self.assertIs(info.reinplace_trigger, ReInplaceTrigger.FLYDSL_OPS)
        self.assertFalse(info.serializable)
        self.assertEqual(info.get_layout_constraint(), "needs_fixed_stride_order")
        self.assertIs(flydsl_kernel_wrap.FlyDSLHOPifier.hop_info, info)
        for hop in (flydsl_kernel_wrapper_mutation, flydsl_kernel_wrapper_functional):
            self.assertIs(get_user_kernel_hop_info(hop), info)
        idx = flydsl_kernel_side_table.add_kernel(FlyDSLKernel(add_kernel, ("Out",)))
        self.assertEqual(info.get_kernel_name({"kernel_idx": idx}), "add_kernel")
        # The HOPs take their arguments by keyword only.
        untyped_hop: Any = flydsl_kernel_wrapper_mutation
        with self.assertRaises(TypeError):
            untyped_hop(idx, flydsl_kernel_side_table.add_constant_args({}), {})


@requires_flydsl
class FlyDSLWrapTests(TestCase):
    """wrap_flydsl: eager launches, make_fx / non-strict export capture, errors."""

    hw_classification = HardwareClassification.CUDA

    def test_wrap_flydsl_mutates_args_iterator(self):
        # A one-shot iterable is read once, not consumed by its validation.
        for names in ((n for n in ["Out"]), iter(["Out"]), map(str, ["Out"])):
            w = wrap_flydsl(add_kernel, mutates_args=names)
            self.assertEqual(w.mutates_args, ("Out",))

    def test_wrap_flydsl_requires_mutates_args(self):
        # FlyDSL write sets cannot be inferred (yet): mutates_args is required and
        # "infer" is refused.
        untyped_wrap: Any = wrap_flydsl
        with self.assertRaises(TypeError):
            untyped_wrap(add_kernel)
        with self.assertRaisesRegex(USER_ERRORS, "mutates_args"):
            wrap_flydsl(add_kernel, mutates_args="infer")

    def test_wrap_flydsl_make_fx_nested_symint_specialized(self):
        # A SymInt inside a Constexpr tuple is specialized like a bare one, so the
        # side table only holds concrete values.
        w = wrap_flydsl(scale_by_first_kernel, mutates_args=["Out"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), (x.shape[0] // 250,))
            return out

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
        scales = _scalar_args(node)["scales"]
        self.assertEqual(scales, (4,))
        self.assertIs(type(scales[0]), int)
        self.assertEqual(gm(x), 4 * x)

    def test_wrap_flydsl_error_tensor_in_container(self):
        # Only top-level tensors become graph arguments.
        w = wrap_flydsl(scale_by_first_kernel, mutates_args=["Out"])
        x = _rand(1000)
        with self.assertRaisesRegex(RuntimeError, "inside a container"):
            w(x, torch.empty_like(x), 1000, (x,))

    def test_wrap_flydsl_numeric_values_as_constants(self):
        # FlyDSL Numeric values are ordinary constants (FlyDSL makes this one a
        # runtime Float32); only FlyDSL tensors and streams are refused.
        x = _rand(1000)
        out = torch.empty_like(x)
        add_scalar_w(x, out, 1000, fx.Float32(2.0))
        self.assertEqual(out, x + 2)

        def f(x):
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), fx.Float32(3.0))
            return out

        self.assertEqual(make_fx(f)(x)(x), x + 3)

    def test_wrap_flydsl_returns_traceable_wrapper(self):
        w = wrap_flydsl(add_kernel, mutates_args=["Out"])
        self.assertEqual(type(w).__name__, "TraceableFlyDSLKernelWrapper")
        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        self.assertIsNone(w(x, y, out, 1000))
        self.assertEqual(out, x + y)

    def test_wrap_flydsl_eager_kwargs_and_partial_write(self):
        x, y = _rand(1000), _rand(1000)
        out = torch.full_like(x, 7.0)
        add_w(X=x, Y=y, Out=out, n=500)
        self.assertEqual(out[:500], x[:500] + y[:500])
        self.assertEqual(out[500:], torch.full_like(x[500:], 7.0))

    def test_wrap_flydsl_eager_inplace(self):
        x = _rand(1000)
        ref = 2 * x
        scale_inplace_w(x, x.numel())
        self.assertEqual(x, ref)

    @parametrize(
        "layout", ["contiguous", "transposed", "row_slice", "gap_out", "t_out"]
    )
    def test_wrap_flydsl_eager_2d_layouts(self, layout):
        # The kernel indexes through each tensor's own layout: strides and storage
        # offsets of views must reach it unchanged.
        x = _rand(4, 8)
        out = torch.zeros(4, 8, device=DEVICE)
        if layout == "transposed":
            x = _rand(8, 4).t()
        elif layout == "row_slice":
            x = _rand(6, 8)[2:]
        elif layout == "gap_out":
            big = torch.zeros(4, 16, device=DEVICE)
            out = big[:, ::2]
        elif layout == "t_out":
            out = torch.zeros(8, 4, device=DEVICE).t()
        add_one_2d_w(x, out)
        self.assertEqual(out, x + 1)
        if layout == "gap_out":
            self.assertEqual(big[:, 1::2], torch.zeros(4, 8, device=DEVICE))

    def test_wrap_flydsl_eager_unaligned_view(self):
        # data_ptr is 4 bytes past a 16-byte boundary: the launch compiles for the
        # alignment the pointer really has.
        base = _rand(4, 9)
        x = base[:, 1:]
        self.assertNotEqual(x.data_ptr() % 16, 0)
        out = torch.zeros(4, 8, device=DEVICE)
        add_one_2d_w(x, out)
        self.assertEqual(out, x + 1)
        y = _rand(1001)[1:]
        out1 = torch.zeros(1000, device=DEVICE)
        add_w(y, y, out1, 1000)
        self.assertEqual(out1, 2 * y)

    def test_wrap_flydsl_eager_compile_cache_keys_on_layout(self):
        # Layouts are static in the compiled kernel and a CompiledFunction called
        # with another layout silently computes garbage, so the eager cache must key
        # on shape and stride (and dtype).
        for shape, transpose in (((4, 8), False), ((16, 32), False), ((4, 8), True)):
            x = _rand(*shape[::-1]).t() if transpose else _rand(*shape)
            out = torch.zeros(*shape, device=DEVICE)
            add_one_2d_w(x, out)
            self.assertEqual(out, x + 1)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x, y = _rand(300, dtype=dtype), _rand(300, dtype=dtype)
            out = torch.zeros_like(x)
            add_w(x, y, out, 300)
            self.assertEqual(out, x + y)

    def test_wrap_flydsl_eager_constexpr(self):
        x = _rand(1000)
        for alpha in (3.0, 2.5, 3.0):
            out = torch.zeros_like(x)
            mul_const_w(x, out, 1000, alpha)
            self.assertEqual(out, x * alpha)

    def test_wrap_flydsl_eager_unannotated_scalar(self):
        # FlyDSL types an unannotated scalar from its value, so the Python type is
        # part of the specialization: an int and a float are two compiled kernels.
        x = _rand(1000)
        for k in (5, 7, 2.5, True):
            out = torch.zeros_like(x)
            add_scalar_w(x, out, 1000, k)
            self.assertEqual(out, x + k)

    def test_wrap_flydsl_eager_two_outputs_and_default(self):
        x, y = _rand(1000), _rand(1000)
        s, d = torch.zeros_like(x), torch.zeros_like(x)
        add_sub_w(x, y, s, d, 1000)  # scale defaults to 1.0
        self.assertEqual((s, d), (x + y, x - y))
        add_sub_w(x, y, s, d, 1000, scale=0.5)
        self.assertEqual((s, d), ((x + y) * 0.5, (x - y) * 0.5))

    def test_wrap_flydsl_eager_same_name_kernels(self):
        # Two host functions with the same __name__/__qualname__ must not share a
        # compiled kernel.
        x, y = _rand(1000), _rand(1000)
        o1, o2 = torch.zeros_like(x), torch.zeros_like(x)
        same_name_add_w(x, y, o1, 1000)
        same_name_sub_w(x, y, o2, 1000)
        self.assertEqual(o1, x + y)
        self.assertEqual(o2, x - y)

    def test_wrap_flydsl_eager_requires_grad_inputs(self):
        x, y = _rand(1000, requires_grad=True), _rand(1000, requires_grad=True)
        out = torch.zeros(1000, device=DEVICE)
        add_w(x, y, out, 1000)
        self.assertEqual(out, (x + y).detach())

    def test_wrap_flydsl_eager_launches_on_current_stream(self):
        # Capturing a graph only works if the launch goes to the capturing (current)
        # stream; a launch on any other stream breaks the capture.
        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        add_w(x, y, out, 1000)  # compile + load outside of the capture
        torch.cuda.synchronize()
        out.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            add_w(x, y, out, 1000)
        torch.cuda.synchronize()
        self.assertEqual(out, torch.zeros_like(x))
        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(out, x + y)
        # And on a side stream, ordered by the usual stream semantics.
        s = torch.cuda.Stream()
        out2 = torch.zeros_like(x)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            add_w(x, y, out2, 1000)
        torch.cuda.current_stream().wait_stream(s)
        self.assertEqual(out2, x + y)

    def test_wrap_flydsl_eager_matches_plain_flydsl(self):
        # The wrapper's eager launch is exactly the plain FlyDSL launch.
        x, y = _rand(3000), _rand(3000)
        ref, out = torch.zeros_like(x), torch.zeros_like(x)
        _reference_launch(add_kernel, x, y, ref, 3000)
        add_w(x, y, out, 3000)
        self.assertEqual(out, ref)
        ref_s, ref_d, s, d = (torch.zeros_like(x) for _ in range(4))
        _reference_launch(add_sub_kernel, x, y, ref_s, ref_d, 3000, scale=0.5)
        add_sub_w(x, y, s, d, 3000, scale=0.5)
        self.assertEqual((s, d), (ref_s, ref_d))

    def test_wrap_flydsl_launches_exactly_once(self):
        # FlyDSL compiles AND launches when it compiles from device tensors. PT2
        # compiles from meta placeholders, so the first launch of a new
        # specialization applies a non-idempotent kernel exactly once, through the
        # wrapper, the mutation HOP and the functional HOP alike.
        kidx = _add_kernel_to_table(acc_kernel, ["Out"])
        cidx = flydsl_kernel_side_table.add_constant_args({})
        x = _rand(1301)  # sizes no other test uses: compile-cache misses
        out = torch.zeros_like(x)
        acc_w(x, out, x.numel())
        self.assertEqual(out, x)
        acc_w(x, out, x.numel())
        self.assertEqual(out, 2 * x)

        x = _rand(1302)
        out = torch.zeros_like(x)
        flydsl_kernel_wrapper_mutation(
            kernel_idx=kidx,
            constant_args_idx=cidx,
            kwargs={"X": x, "Out": out, "n": x.numel()},
        )
        self.assertEqual(out, x)

        x = _rand(1303)
        out = torch.zeros_like(x)
        res = flydsl_kernel_wrapper_functional(
            kernel_idx=kidx,
            constant_args_idx=cidx,
            kwargs={"X": x, "Out": out, "n": x.numel()},
            tensors_to_clone=["Out"],
        )["Out"]
        self.assertEqual(res, x)
        self.assertEqual(out, torch.zeros_like(x))

    def test_wrap_flydsl_compiles_meta_placeholders_through_private_jits(self):
        # Each compile goes through a new private JitFunction for the user's host
        # function (never the user's own, whose caches stay empty), from meta
        # placeholders and a null stream.
        compiled = []
        orig_compile = flyc.compile

        def spy_compile(fn, *args, **kwargs):
            compiled.append((fn, args))
            return orig_compile(fn, *args, **kwargs)

        x, y = _rand(1304), _rand(1304)
        mem_cache = dict(add_kernel._mem_cache)
        # A fresh (process-wide) compile cache: both launches compile.
        with (
            mock.patch.object(
                flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
            ),
            mock.patch.object(flyc, "compile", spy_compile),
        ):
            for dtype in (torch.float32, torch.float16):
                a, b = x.to(dtype), y.to(dtype)
                out = torch.zeros_like(a)
                add_w(a, b, out, 1304)
                self.assertEqual(out, a + b)
        self.assertEqual(add_kernel._mem_cache, mem_cache)
        self.assertEqual(len(compiled), 2)
        (jf1, args1), (jf2, _) = compiled
        self.assertIsNot(jf1, jf2)
        for jf in (jf1, jf2):
            self.assertEqual(type(jf).__name__, "JitFunction")
            self.assertIsNot(jf, add_kernel)
            self.assertEqual(jf.func.__name__, "add_kernel")
            self.assertEqual(jf.func.__annotations__, add_kernel.func.__annotations__)
        tensors = [a.torch_tensor for a in args1 if hasattr(a, "torch_tensor")]
        self.assertEqual([t.device.type for t in tensors], ["meta"] * 3)
        self.assertEqual(args1[3], 1304)
        self.assertIsInstance(args1[4], fx.Stream)
        self.assertIsNone(args1[4].value)

    def test_wrap_flydsl_globals_drift_does_not_trigger(self):
        # FlyDSL raises when a module global a JitFunction uses changes after its
        # first compile, and counts attribute names (x in fx.thread_idx.x) as such
        # globals. PT2 compiles every specialization through a fresh private
        # JitFunction, so rebinding x between two compiles does not raise.
        global x
        saved = x
        a, b = _rand(1000), _rand(1000)
        w = wrap_flydsl(drift_kernel, mutates_args=["Out"])

        def meta_args(t):
            return [flyc.from_torch_tensor(t.to("meta")) for _ in range(3)]

        try:
            out = torch.zeros_like(a)
            w(a, b, out, 1000)
            self.assertEqual(out, a + b)
            # The user's JitFunction itself does raise (compiles that launch nothing).
            with (
                FLYDSL_COMPILE_LOCK,
                flydsl_user_kernel.flydsl_env(flydsl_user_kernel.device_arch()),
            ):
                flyc.compile(drift_kernel, *meta_args(a), 1000, None)
                x = torch.randn(8)
                with self.assertRaisesRegex(RuntimeError, "changed since first"):
                    flyc.compile(drift_kernel, *meta_args(a.half()), 1000, None)
            a16, b16 = a.half(), b.half()
            out16 = torch.zeros_like(a16)
            w(a16, b16, out16, 1000)  # a new specialization, compiled after the rebind
            self.assertEqual(out16, a16 + b16)
        finally:
            x = saved

    def test_wrap_flydsl_global_rebind_recompiles(self):
        # The values of the module globals a kernel reads are part of the compile
        # key: after a rebind every launch uses the new value (where FlyDSL itself
        # raises), cached specializations included. A module global the kernels use
        # only as an attribute name (x in fx.thread_idx.x) is not one of them.
        module = sys.modules[__name__]
        self.addCleanup(setattr, module, "_GLOBAL_SCALE", _GLOBAL_SCALE)
        self.addCleanup(setattr, module, "x", x)
        patcher = mock.patch.object(
            flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        w = wrap_flydsl(scale_global_kernel, mutates_args=["Out"])
        a, a2 = _rand(1000), _rand(2000)

        def run(t):
            out = torch.zeros_like(t)
            w(t, out, t.numel())
            return out

        self.assertEqual(run(a), 2 * a)
        module._GLOBAL_SCALE = 3.0
        self.assertEqual(run(a), 3 * a)
        self.assertEqual(run(a2), 3 * a2)
        module._GLOBAL_SCALE = 2.0
        self.assertEqual(run(a), 2 * a)
        compiled = flydsl_user_kernel._compiled[scale_global_kernel]
        self.assertEqual(len(compiled), 3)
        module.x = torch.randn(8)
        self.assertEqual(run(a), 2 * a)
        self.assertEqual(len(compiled), 3)

    def test_wrap_flydsl_compile_hints_and_env_in_key(self):
        # FlyDSL's compile hints (the host function's own and the calling thread's)
        # and the environment variables that change its compiles are part of the
        # compile key too.
        from flydsl.compiler.kernel_function import CompilationContext

        a, b = _rand(1000), _rand(1000)

        def launch_key():
            # The compile-cache key of a launch; the compile itself is skipped.
            with (
                mock.patch.object(
                    flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
                ),
                mock.patch.object(
                    flydsl_user_kernel,
                    "compile_host_fn",
                    return_value=lambda *args: None,
                ),
            ):
                add_w(a, b, torch.zeros_like(a), 1000)
                (signature,) = flydsl_user_kernel._compiled[add_kernel]
            return signature

        key = launch_key()
        self.assertEqual(launch_key(), key)
        with mock.patch.dict(os.environ, {"FLYDSL_COMPILE_OPT_LEVEL": "1"}):
            self.assertNotEqual(launch_key(), key)
        with CompilationContext.compile_hints({"fast_fp_math": True}):
            self.assertNotEqual(launch_key(), key)
        with mock.patch.dict(add_kernel.compile_hints, {"fast_fp_math": True}):
            self.assertNotEqual(launch_key(), key)
            out = torch.zeros_like(a)
            with mock.patch.object(
                flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
            ):
                add_w(a, b, out, 1000)
                (signature,) = flydsl_user_kernel._compiled[add_kernel]
            self.assertEqual(out, a + b)
            self.assertIn(
                ("fast_fp_math", "builtins", "bool", "True"), signature[-1][1]
            )

    def test_wrap_flydsl_constexpr_lambdas_keyed_by_code(self):
        # FlyDSL keys a Constexpr lambda by its code: equal lambdas made per call
        # share one compile and one side-table entry; other code compiles anew.
        w = wrap_flydsl(map_kernel, mutates_args=["Out"])
        a = _rand(1000)
        indices = []
        with (
            _record_constant_args_indices(indices),
            mock.patch.object(
                flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
            ),
        ):
            for _ in range(3):
                out = torch.zeros_like(a)
                w(a, out, 1000, lambda v: v * 2.0)
                self.assertEqual(out, 2 * a)
            self.assertEqual(len(flydsl_user_kernel._compiled[map_kernel]), 1)
            out = torch.zeros_like(a)
            w(a, out, 1000, lambda v: v * 3.0)
            self.assertEqual(out, 3 * a)
            self.assertEqual(len(flydsl_user_kernel._compiled[map_kernel]), 2)
        self.assertEqual(len(indices), 4)
        self.assertEqual(set(indices[:3]), {indices[0]})
        self.assertNotEqual(indices[3], indices[0])

    def test_wrap_flydsl_constexpr_lambdas_in_tuple_keyed_by_code(self):
        # The same for a lambda inside a Constexpr tuple, which FlyDSL keys item by
        # item.
        w = wrap_flydsl(map_first_kernel, mutates_args=["Out"])
        a = _rand(1000)
        indices = []
        with (
            _record_constant_args_indices(indices),
            mock.patch.object(
                flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
            ),
        ):
            for _ in range(3):
                out = torch.zeros_like(a)
                w(a, out, 1000, (lambda v: v * 2.0,))
                self.assertEqual(out, 2 * a)
            self.assertEqual(len(flydsl_user_kernel._compiled[map_first_kernel]), 1)
            out = torch.zeros_like(a)
            w(a, out, 1000, (lambda v: v * 3.0,))
            self.assertEqual(out, 3 * a)
            self.assertEqual(len(flydsl_user_kernel._compiled[map_first_kernel]), 2)
        self.assertEqual(len(indices), 4)
        self.assertEqual(set(indices[:3]), {indices[0]})
        self.assertNotEqual(indices[3], indices[0])

    def test_wrap_flydsl_constexpr_float_sign(self):
        # 0.0 and -0.0 are different constants (they compare equal).
        ones = torch.ones(1000, device=DEVICE)
        for alphas in ((0.0, -0.0), (-0.0, 0.0)):
            with mock.patch.object(
                flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
            ):
                for alpha in alphas:
                    out = torch.ones_like(ones)
                    mul_const_w(ones, out, 1000, alpha)
                    self.assertEqual(
                        torch.signbit(out).all().item(), math.copysign(1.0, alpha) < 0
                    )

    def test_wrap_flydsl_none_mutates_args(self):
        with self.assertRaisesRegex(RuntimeError, r"mutates_args must be .*got None"):
            # pyrefly: ignore [bad-argument-type]
            wrap_flydsl(add_kernel, mutates_args=None)

    def test_wrap_flydsl_methods_refused(self):
        # FlyDSL binds a first parameter called self to the object of a method,
        # which PT2 cannot pass.
        class Holder:
            @flyc.jit
            def launch(self, X, Y, Out, n: fx.Int32, stream: fx.Stream):
                _add_k(X, Y, Out, n).launch(
                    grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
                )

        pattern = (
            r"@flyc.jit methods are not supported \(FlyDSL binds the first parameter, "
            r"'self', to the object\)"
        )
        for host_fn in (Holder.launch, Holder().launch):
            with self.assertRaisesRegex(RuntimeError, pattern):
                wrap_flydsl(host_fn, mutates_args=["Out"])

    def test_wrap_flydsl_positional_only_refused(self):
        # PT2 binds and compiles host functions by parameter name.
        @flyc.jit
        def posonly_kernel(X, /, Y, Out, n: fx.Int32, stream: fx.Stream):
            _add_k(X, Y, Out, n).launch(
                grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"posonly_kernel: positional-only parameters are not supported \('X'\)",
        ):
            wrap_flydsl(posonly_kernel, mutates_args=["Out"])

    def test_wrap_flydsl_tensor_default_refused(self):
        # PT2 passes each tensor as a graph argument; a tensor default would be a
        # constant of the compiled kernel, which constant keys refuse: refused when
        # the host function is wrapped, so that eager calls and PT2 agree.
        weight = _rand(1000)

        @flyc.jit
        def tensor_default_kernel(X, Out, n: fx.Int32, stream: fx.Stream, Y=weight):
            _add_k(X, Y, Out, n).launch(
                grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"tensor_default_kernel: the default of 'Y' is a tensor; pass each tensor "
            "as its own argument",
        ):
            wrap_flydsl(tensor_default_kernel, mutates_args=["Out"])

    def test_wrap_flydsl_tensorless_host_fn_refused(self):
        # FlyDSL launches a host function without tensor arguments when it compiles
        # it: refused before compiling (it would run twice).
        @flyc.jit
        def no_tensor_kernel(n: fx.Int32, stream: fx.Stream):
            pass

        w = wrap_flydsl(no_tensor_kernel, mutates_args=[])
        with (
            mock.patch.object(flyc, "compile", side_effect=AssertionError) as compile,
            self.assertRaisesRegex(
                ValueError,
                r"FlyDSL kernel no_tensor_kernel: host functions without a tensor "
                r"argument are not supported",
            ),
        ):
            w(7)
        compile.assert_not_called()

    def test_wrap_flydsl_keyword_only_params(self):
        # Keyword-only parameters are passed by name, to the wrapper (as Python binds
        # the call) and to FlyDSL's compile.
        a, b = _rand(1000), _rand(1000)
        for host_fn, call, expected in (
            (kwonly_stream_kernel, lambda w, out: w(a, b, out, 1000), a + b),
            (kwonly_n_kernel, lambda w, out: w(a, b, out, n=1000), a + b),
            (kwonly_out_kernel, lambda w, out: w(a, b, 1000, Out=out), a + b),
            (kwonly_alpha_kernel, lambda w, out: w(a, out, 1000), 2 * a),
            (kwonly_alpha_kernel, lambda w, out: w(a, out, 1000, alpha=3.0), 3 * a),
        ):
            with self.subTest(host_fn=host_fn.func.__name__):
                out = torch.zeros_like(a)
                call(wrap_flydsl(host_fn, mutates_args=["Out"]), out)
                self.assertEqual(out, expected)
        w = wrap_flydsl(kwonly_n_kernel, mutates_args=["Out"])
        with self.assertRaisesRegex(
            RuntimeError,
            r"kwonly_n_kernel\(\) takes 3 positional arguments without the stream but "
            r"4 were given",
        ):
            w(a, b, torch.zeros_like(a), 1000)

    def test_wrap_flydsl_unannotated_numeric_defaults(self):
        # FlyDSL compiles a Numeric default of an unannotated parameter as its own
        # type (a Python value would be an Int32 or a Float32), as a direct call does.
        xi = torch.arange(1000, device=DEVICE, dtype=torch.int64)
        xd = _rand(1000, dtype=torch.float64)
        for host_fn, t, expected in (
            # k * k = 10**10 does not fit an Int32, nor does k = 3 * 10**9.
            (int64_default_kernel, xi, xi + 10**10),
            (big_int64_default_kernel, xi, xi + 3_000_000_000**2),
            (float64_default_kernel, xd, xd + 0.1),  # 0.1 is not a Float32
        ):
            with self.subTest(host_fn=host_fn.func.__name__):
                direct = torch.zeros_like(t)
                _reference_launch(host_fn, t, direct, 1000)
                self.assertEqual(direct, expected, atol=0, rtol=0)
                out = torch.zeros_like(t)
                wrap_flydsl(host_fn, mutates_args=["Out"])(t, out, 1000)
                self.assertEqual(out, expected, atol=0, rtol=0)

    def test_wrap_flydsl_int_range_check(self):
        # FlyDSL wraps an int that does not fit its Int32 parameter like a C cast
        # (an unannotated int is an Int32 too): the launch refuses it instead.
        x = _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(ValueError, r"n=2147483648 does not fit Int32"):
            add_w(x, x, out, 2**31)
        with self.assertRaisesRegex(ValueError, r"k=-2147483649 does not fit Int32"):
            add_scalar_w(x, out, 16, -(2**31) - 1)
        self.assertEqual(out, torch.zeros_like(x))
        add_scalar_w(x, out, 16, 2**31 - 1)  # the largest Int32 is fine
        self.assertEqual(out, x + (2**31 - 1))

    def test_wrap_flydsl_error_stream_passed(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(USER_ERRORS, "stream"):
            add_w(x, y, out, 16, stream=_raw_stream())
        with self.assertRaisesRegex(USER_ERRORS, "stream"):
            add_w(x, y, out, 16, _raw_stream())

    def test_wrap_flydsl_error_stream_params(self):
        # Exactly one fx.Stream parameter is required.
        for fn in (no_stream_kernel, two_stream_kernel):
            with self.subTest(fn=fn.func.__name__):
                with self.assertRaisesRegex(USER_ERRORS, r"fx\.Stream"):
                    wrap_flydsl(fn, mutates_args=["Out"])

    def test_wrap_flydsl_error_bad_mutates_args(self):
        for bad, pattern in (
            (["not_a_param"], "not_a_param"),
            (["n"], "'n'"),  # a scalar parameter cannot be written
            (["stream"], "stream"),
            ("Out", "mutates_args"),  # a bare string other than "unknown"
        ):
            with self.subTest(mutates_args=bad):
                with self.assertRaisesRegex(USER_ERRORS, pattern):
                    wrap_flydsl(add_kernel, mutates_args=bad)

    def test_wrap_flydsl_error_not_flyc_jit(self):
        def plain(X, Y, Out, n, stream: fx.Stream):
            pass

        class Holder:
            kernel = add_kernel

        # A plain function, a @flyc.kernel device kernel, and a JitFunction bound
        # as a method (a functools.partial) are not host functions.
        for fn in (plain, _add_k, Holder().kernel):
            with self.subTest(fn=fn):
                with self.assertRaisesRegex(USER_ERRORS, r"@flyc\.jit"):
                    wrap_flydsl(fn, mutates_args=["Out"])

    def test_wrap_flydsl_error_flydsl_tensor_arg(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(USER_ERRORS, "torch.Tensor"):
            add_w(flyc.from_torch_tensor(x), y, out, 16)

    def test_wrap_flydsl_error_bad_call(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(USER_ERRORS, "(?i)missing"):
            add_w(x, y, out)
        with self.assertRaisesRegex(USER_ERRORS, "not_a_param"):
            add_w(x, y, out, 16, not_a_param=1)
        with self.assertRaisesRegex(USER_ERRORS, "(?i)multiple values"):
            add_w(x, y, out, 16, n=16)
        with self.assertRaisesRegex(USER_ERRORS, "'n'"):
            add_w(x, y, out, x)  # a tensor for a scalar parameter

    def test_wrap_flydsl_error_old_flydsl(self):
        import flydsl

        torch._library.flydsl._flydsl_version.cache_clear()
        try:
            with mock.patch.object(flydsl, "__version__", "0.3.2"):
                self.assertFalse(flydsl_version_ok())
                with self.assertRaisesRegex(
                    RuntimeError, r"flydsl >= 0\.4\.0, found flydsl 0\.3\.2"
                ):
                    wrap_flydsl(add_kernel, mutates_args=["Out"])
        finally:
            torch._library.flydsl._flydsl_version.cache_clear()
        self.assertTrue(flydsl_version_ok())

    @parametrize("tracing_mode", ["fake", "symbolic"])
    def test_wrap_flydsl_make_fx(self, tracing_mode):
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        gm = make_fx(f, tracing_mode=tracing_mode)(_rand(1000), _rand(1000))
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
        self.assertEqual(node.name, "flydsl_kernel_wrapper_mutation_proxy")
        record = flydsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertIs(record.fn, add_kernel)
        self.assertEqual(tuple(record.mutates_args), ("Out",))
        hop_kwargs = node.kwargs["kwargs"]
        self.assertEqual(
            [k for k, v in hop_kwargs.items() if _is_tensor_node(v)],
            ["X", "Y", "Out"],
        )
        self.assertEqual(hop_kwargs["X"].op, "placeholder")
        scalars = _scalar_args(node)
        self.assertNotIn("stream", scalars)
        self.assertNotIn("stream", hop_kwargs)
        if tracing_mode == "symbolic":
            # A runtime (fx.Int32-annotated) scalar stays symbolic.
            self.assertIsInstance(hop_kwargs["n"], torch.fx.Node)
        else:
            self.assertEqual(scalars["n"], 1000)
            self.assertExpectedInline(
                _normalize_indices(gm.code.strip()),
                """\
def forward(self, x_1, y_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False)
    flydsl_kernel_wrapper_mutation_proxy = torch.ops.higher_order.flydsl_kernel_wrapper_mutation(kernel_idx = K, constant_args_idx = C, kwargs = {'X': x_1, 'Y': y_1, 'Out': empty_like, 'n': 1000});  x_1 = y_1 = flydsl_kernel_wrapper_mutation_proxy = None
    return empty_like""",
            )
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(gm(x, y), x + y)
        if tracing_mode == "symbolic":
            x, y = _rand(77), _rand(77)
            self.assertEqual(gm(x, y), x + y)

    def test_wrap_flydsl_make_fx_constexpr_specialized(self):
        # A Constexpr param is baked into the compiled kernel: a symbolic value is
        # specialized to a Python int, while the runtime scalar n stays symbolic.
        def f(x):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), x.shape[0] // 250)
            return out

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
        alpha = _scalar_args(node)["alpha"]
        self.assertIs(type(alpha), int)
        self.assertEqual(alpha, 4)
        self.assertIsInstance(node.kwargs["kwargs"]["n"], torch.fx.Node)
        self.assertEqual(gm(x), 4 * x)

    def test_wrap_flydsl_make_fx_unannotated_scalar_stays_symbolic(self):
        # FlyDSL makes a Python int for an unannotated parameter a runtime Int32, so
        # a symbolic value stays a kernel argument (no specialization).
        def f(x):
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), x.shape[0] // 100)
            return out

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
        self.assertIsInstance(_scalar_args(node)["k"], torch.fx.Node)
        self.assertEqual(gm(x), x + 10)
        x2 = _rand(2000)
        self.assertEqual(gm(x2), x2 + 20)

    def test_wrap_flydsl_make_fx_defaults_resolved_at_launch(self):
        # As for Triton, a parameter the call leaves out is not captured: its
        # default applies when the kernel is launched.
        def f(x, y, scale=None):
            s, d = torch.empty_like(x), torch.empty_like(x)
            if scale is None:
                add_sub_w(x, y, s, d, x.numel())
            else:
                add_sub_w(x, y, s, d, x.numel(), scale=scale)
            return s, d

        x, y = _rand(1000), _rand(1000)
        for scale, expect in ((None, 1.0), (0.5, 0.5)):
            gm = make_fx(functools.partial(f, scale=scale))(x, y)
            (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
            self.assertEqual(_scalar_args(node).get("scale"), scale)
            s, d = gm(x, y)
            self.assertEqual((s, d), ((x + y) * expect, (x - y) * expect))

    @parametrize("dynamic", [False, True])
    def test_wrap_flydsl_non_strict_export(self, dynamic):
        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        dynamic_shapes = None
        if dynamic:
            dim = torch.export.Dim("n", min=2, max=8192)
            dynamic_shapes = {"x": {0: dim}, "y": {0: dim}}
        ep = torch.export.export(
            M(), (x, y), dynamic_shapes=dynamic_shapes, strict=False
        )
        self.assertIn(
            "torch.ops.higher_order.flydsl_kernel_wrapper_", ep.graph_module.code
        )
        decomposed = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.higher_order.flydsl_kernel_wrapper_functional", 1, exactly=True
        ).run(decomposed.graph_module.code)
        self.assertEqual(ep.module()(x, y), x + y)
        self.assertEqual(decomposed.module()(x, y), x + y)
        if dynamic:
            x2, y2 = _rand(77), _rand(77)
            self.assertEqual(ep.module()(x2, y2), x2 + y2)
            self.assertEqual(decomposed.module()(x2, y2), x2 + y2)

    def test_wrap_flydsl_non_strict_export_input_mutation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                scale_inplace_w(x, x.numel())
                return x + 1

        x = _rand(1000)
        ep = torch.export.export(M(), (x.clone(),), strict=False)
        x_in = x.clone()
        self.assertEqual(ep.module()(x_in), 2 * x + 1)
        self.assertEqual(x_in, 2 * x)

    def test_wrap_flydsl_non_strict_export_views_and_constants(self):
        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty(x.shape[1], x.shape[0], device=x.device)
                add_one_2d_w(x.t(), out)  # non-contiguous input view
                flat = out.reshape(-1)
                o2 = torch.empty_like(flat)
                mul_const_w(flat, o2, flat.numel(), 3.0)
                o3 = torch.empty_like(o2)
                add_scalar_w(o2, o3, o2.numel(), 5)
                return o3

        x = _rand(8, 4)
        expected = (x.t() + 1).reshape(-1) * 3 + 5
        ep = torch.export.export(M(), (x,), strict=False)
        self.assertEqual(ep.module()(x), expected)
        self.assertEqual(ep.run_decompositions().module()(x), expected)

    def test_wrap_flydsl_non_strict_export_refuses_aliased_writes(self):
        # The same tensor, or overlapping views, passed to two written parameters (Sum
        # and Diff, both in mutates_args; X and Out under "unknown") would be cloned
        # once per parameter and only one clone written back: exporting refuses it
        # when it functionalizes the graph. An eager call is not functionalized, and
        # two rows of one base are fine.
        unknown_add_w = wrap_flydsl(add_kernel, mutates_args="unknown")

        class SameOutput(torch.nn.Module):
            def forward(self, x, y, out):
                add_sub_w(x, y, out, out, x.numel())
                return out

        class InPlace(torch.nn.Module):
            def forward(self, x, y):
                unknown_add_w(x, y, x, x.numel())
                return x

        class Shifted(torch.nn.Module):
            def forward(self, x, y, buf):
                add_sub_w(x, y, buf[1:], buf[:-1], x.numel())
                return buf

        class Views(torch.nn.Module):
            def forward(self, x, y, base):
                add_sub_w(x, y, base[0], base[1], x.numel())
                return base

        x, y = _rand(1000), _rand(1000)
        out = torch.empty_like(x)
        buf = torch.empty(1001, device=DEVICE)
        for module, args, pair in (
            (SameOutput(), (x, y, out), "add_sub_kernel: 'Sum' and 'Diff'"),
            (InPlace(), (x.clone(), y), "add_kernel: 'X' and 'Out'"),
            (Shifted(), (x, y, buf), "add_sub_kernel: 'Sum' and 'Diff'"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                f"FlyDSL kernel {pair} are the same tensor or views that may overlap",
            ):
                torch.export.export(module, args, strict=False).run_decompositions()
        x_in = x.clone()
        InPlace()(x_in, y)
        self.assertEqual(x_in, x + y)
        base = torch.zeros(2, 1000, device=DEVICE)
        ep = torch.export.export(Views(), (x, y, base.clone()), strict=False)
        self.assertEqual(
            ep.run_decompositions().module()(x, y, base.clone()),
            torch.stack([x + y, x - y]),
        )

    def test_wrap_flydsl_compiles_under_compile_lock_and_pinned_env(self):
        # Every PT2 compile holds FLYDSL_COMPILE_LOCK, with the target pinned to the
        # launch device's arch, the disk cache off and COMPILE_ONLY unset; the
        # environment is restored afterwards.
        seen = []
        orig_compile = flyc.compile

        def spy_compile(*args, **kwargs):
            env = os.environ
            seen.append(
                (
                    _lock_held_elsewhere(),
                    env.get("ARCH"),
                    env.get("FLYDSL_GPU_ARCH"),
                    env.get("FLYDSL_RUNTIME_ENABLE_CACHE"),
                    "COMPILE_ONLY" in env,
                )
            )
            return orig_compile(*args, **kwargs)

        arch = flydsl_user_kernel.device_arch()
        x, y = _rand(1305), _rand(1305)
        out = torch.zeros_like(x)
        user_env = {"ARCH": "gfx000", "COMPILE_ONLY": "1"}
        with mock.patch.dict(os.environ, user_env):
            # A fresh (process-wide) compile cache: the launch compiles.
            with (
                mock.patch.object(
                    flydsl_user_kernel, "_compiled", weakref.WeakKeyDictionary()
                ),
                mock.patch.object(flyc, "compile", spy_compile),
            ):
                add_w(x, y, out, 1305)
            self.assertEqual(os.environ["ARCH"], "gfx000")
            self.assertEqual(os.environ["COMPILE_ONLY"], "1")
        self.assertEqual(out, x + y)
        self.assertEqual(seen, [(True, arch, arch, "0", False)])

    def test_wrap_flydsl_export_not_serializable(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(M(), (x, y), strict=False)
        # The node indexes a process-local side table: no serialized form.
        with self.assertRaisesRegex(Exception, "flydsl user kernels"):
            torch.export.save(ep, io.BytesIO())


@requires_flydsl
class FlyDSLCompileHelperTests(TestCase):
    """The FlyDSL compile helpers every PT2 compile uses (eager launches here; the
    AOTInductor export later)."""

    hw_classification = HardwareClassification.CUDA

    def test_globals_key(self):
        # The module globals a host function and the kernels it calls load (not the
        # attribute names they use, x in fx.thread_idx.x): simple constants by value,
        # other objects by identity. A name that is not a global yet is one too.
        globals_key = flydsl_user_kernel._globals_key
        module = sys.modules[__name__]
        key = globals_key(scale_global_kernel)
        self.assertEqual(
            [name for name, _ in key], ["_scale_global_k", "fx", "_GLOBAL_SCALE"]
        )
        self.assertEqual(dict(key)["_GLOBAL_SCALE"], (float, (2.0).hex()))
        with mock.patch.object(module, "_GLOBAL_SCALE", 2.0):
            self.assertEqual(globals_key(scale_global_kernel), key)
        with mock.patch.object(module, "_GLOBAL_SCALE", 3.0):
            self.assertNotEqual(globals_key(scale_global_kernel), key)

        # A kernel rebound to one that reads other globals: those are tracked too.
        @flyc.kernel
        def other_scale_k(X, Out, n: fx.Int32):
            i = fx.block_idx.x * 1024 + fx.thread_idx.x
            if i < n:
                Out[i] = X[i] * _OTHER_SCALE

        with mock.patch.object(module, "_scale_global_k", other_scale_k):
            other_key = globals_key(scale_global_kernel)
            self.assertNotEqual(other_key, key)
            self.assertIn("_OTHER_SCALE", [name for name, _ in other_key])
        with mock.patch.object(module, "x", torch.randn(8)):
            self.assertEqual(globals_key(scale_global_kernel), key)
        self.assertEqual(globals_key(scale_global_kernel), key)

        @flyc.jit
        def later_global_kernel(X, stream: fx.Stream):
            if _DEFINED_LATER:  # noqa: F821 (declared, not yet defined)
                pass

        key = globals_key(later_global_kernel)
        self.assertIn(("_DEFINED_LATER", (type(None), None)), key)
        with mock.patch.object(module, "_DEFINED_LATER", 1, create=True):
            self.assertNotEqual(globals_key(later_global_kernel), key)

    def test_device_arch(self):
        arch = flydsl_user_kernel.device_arch()
        gcn = torch.cuda.get_device_properties(0).gcnArchName
        self.assertEqual(arch, gcn.split(":")[0])
        self.assertTrue(arch.startswith("gfx"), arch)
        self.assertEqual(flydsl_user_kernel.device_arch(torch.device("cuda:0")), arch)

    def test_flydsl_env_pins_and_restores(self):
        user_env = {
            "ARCH": "gfx000",
            "COMPILE_ONLY": "1",
            "FLYDSL_RUNTIME_RUN_ONLY": "1",
            "ROCM_PATH": "/opt/rocm-test",
            "PATH": "/usr/bin",
        }
        with mock.patch.dict(os.environ, user_env):
            os.environ.pop("FLYDSL_GPU_ARCH", None)
            with FLYDSL_COMPILE_LOCK, flydsl_user_kernel.flydsl_env("gfx950"):
                self.assertEqual(os.environ["ARCH"], "gfx950")
                self.assertEqual(os.environ["FLYDSL_GPU_ARCH"], "gfx950")
                self.assertEqual(os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"], "0")
                self.assertNotIn("COMPILE_ONLY", os.environ)
                self.assertNotIn("FLYDSL_RUNTIME_RUN_ONLY", os.environ)
                self.assertEqual(os.environ["PATH"], "/usr/bin")
            self.assertEqual(os.environ["ARCH"], "gfx000")
            self.assertEqual(os.environ["COMPILE_ONLY"], "1")
            self.assertEqual(os.environ["FLYDSL_RUNTIME_RUN_ONLY"], "1")
            self.assertNotIn("FLYDSL_GPU_ARCH", os.environ)
            # An export also needs the ROCm LLVM linker and objcopy on PATH.
            with (
                FLYDSL_COMPILE_LOCK,
                flydsl_user_kernel.flydsl_env("gfx950", for_export=True),
            ):
                self.assertEqual(
                    os.environ["PATH"],
                    "/opt/rocm-test/llvm/bin" + os.pathsep + "/usr/bin",
                )
            self.assertEqual(os.environ["PATH"], "/usr/bin")
            # Restored on errors too.
            with self.assertRaises(KeyError):
                with flydsl_user_kernel.flydsl_env("gfx950"):
                    raise KeyError
            self.assertEqual(os.environ["ARCH"], "gfx000")

    def test_fresh_private_jit(self):
        code = add_kernel.func.__code__
        pj1 = flydsl_user_kernel.fresh_private_jit(add_kernel)
        pj2 = flydsl_user_kernel.fresh_private_jit(add_kernel)
        self.assertIsNot(pj1, pj2)
        for pj in (pj1, pj2):
            self.assertIsNot(pj, add_kernel)
            self.assertIsNot(pj.func, add_kernel.func)
            self.assertEqual(pj.func.__name__, "add_kernel")
            self.assertEqual(pj.func.__qualname__, add_kernel.func.__qualname__)
            self.assertEqual(pj.func.__annotations__, add_kernel.func.__annotations__)
        # The user's function object is not rewritten again.
        self.assertIs(add_kernel.func.__code__, code)

    def test_compile_host_fn_from_placeholders_launches_nothing(self):
        # A compile from meta placeholders launches nothing; the CompiledFunction is
        # called with every parameter positionally (torch tensors, the raw stream).
        # A size marked dynamic is a runtime value of the compiled kernel.
        n = 2048
        placeholders = {
            name: flydsl_user_kernel.meta_placeholder(
                torch.float32, (n,), (1,), 16, dynamic_sizes=(0,)
            )
            for name in ("X", "Out")
        }
        compiled = flydsl_user_kernel.compile_host_fn(
            acc_kernel,
            {**placeholders, "n": n, "stream": fx.Stream(None)},
            flydsl_user_kernel.device_arch(),
        )
        x = _rand(n)
        out = torch.zeros_like(x)
        torch.cuda.synchronize()
        self.assertEqual(out, torch.zeros_like(x))
        compiled(x, out, n, _raw_stream())
        self.assertEqual(out, x)
        x2 = _rand(1000)
        out2 = torch.zeros_like(x2)
        compiled(x2, out2, 1000, _raw_stream())
        self.assertEqual(out2, x2)


@requires_flydsl
class FlyDSLDynamoTests(TestCase):
    """torch.compile front end: the wrapper (global or created inline) becomes one
    HOP node; raw @flyc.jit calls graph-break; recompiles follow the arg classes."""

    hw_classification = HardwareClassification.CUDA

    def test_wrap_of_a_wrapper_graph_breaks(self):
        # Eager refuses to wrap a wrapper (it is not a @flyc.jit function), so Dynamo
        # must not silently replace its write set.
        def f(x, y):
            out = torch.empty_like(x)
            wrap_flydsl(add_w, mutates_args=["Out"])(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        with self.assertRaises(RuntimeError):
            f(x, y)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)

    def test_wrap_without_mutates_args_graph_breaks(self):
        # wrap_flydsl requires mutates_args (FlyDSL write sets are not inferred).
        untyped_wrap: Any = wrap_flydsl

        def f(x, y):
            out = torch.empty_like(x)
            untyped_wrap(add_kernel)(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "wrap_flydsl"):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)

    def test_wrap_with_none_mutates_args_graph_breaks(self):
        # None is an invalid write set (as in eager), not an unwrapped host function.
        def f(x, y):
            out = torch.empty_like(x)
            # pyrefly: ignore [bad-argument-type]
            wrap_flydsl(add_kernel, mutates_args=None)(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, r"mutates_args must be .*got None"
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)

    def test_refused_constant_graph_breaks(self):
        # A constant FlyDSL's side table refuses (an unhashable value that is no
        # container, here a set in the Constexpr tuple, or an object that refers to
        # itself) is a graph break naming the kernel, not an internal error; eager
        # refuses it with the same TypeError or ValueError.
        w = wrap_flydsl(scale_by_first_kernel, mutates_args=["Out"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), (2.0, {1}))
            return out

        x = _rand(1000)
        msg = "a constant kernel argument of type set is unhashable"
        with self.assertRaisesRegex(TypeError, msg):
            f(x)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, rf"scale_by_first_kernel\(\): {msg}"
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x)

        class Cfg:
            def __init__(self) -> None:
                self.me = self

        cfg = Cfg()

        def g(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        msg = r"a constant kernel argument of type Cfg refers to itself through \.me"
        with self.assertRaisesRegex(ValueError, msg):
            g(x)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, rf"scale_by_first_kernel\(\): {msg}"
        ):
            torch.compile(g, backend="eager", fullgraph=True)(x)

    @parametrize("backend", ("eager", "aot_eager"))
    def test_constexpr_lambda_defaults_are_guarded(self, backend):
        # A lambda passed in is guarded on its code and, when it has defaults (FlyDSL
        # bakes them in), on their values: another default recompiles instead of
        # reusing the first lambda.
        w = wrap_flydsl(map_kernel, mutates_args=["Out"])

        def f(t, op):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), op)
            return out

        def scale_by(k):
            return lambda v, k=k: v * k

        cf = torch.compile(f, backend=backend, fullgraph=True)
        a = _rand(1000)
        self.assertEqual(cf(a, scale_by(5.0)), 5 * a)
        self.assertEqual(cf(a, scale_by(6.0)), 6 * a)
        self.assertEqual(cf(a, lambda v: v * 2.0), 2 * a)
        self.assertEqual(cf(a, lambda v: v * 2.0), 2 * a)

    @parametrize("backend", ("eager", "aot_eager"))
    def test_constexpr_lambda_made_in_the_frame(self, backend):
        # Dynamo names a function made in the frame by its qualified name; FlyDSL
        # takes a Constexpr lambda only when it is named "<lambda>", as in eager.
        w = wrap_flydsl(map_kernel, mutates_args=["Out"])

        def f(t):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), lambda v: v * 2.0)
            return out

        a = _rand(1000)
        self.assertEqual(torch.compile(f, backend=backend, fullgraph=True)(a), 2 * a)

    @parametrize("backend", ("eager", "aot_eager"))
    @parametrize("in_tuple", (False, True))
    def test_constexpr_lambda_defaults_guarded_by_value(self, backend, in_tuple):
        # Passed in or inside a Constexpr tuple, a lambda with defaults is guarded on
        # its code and the values of its defaults (their types included): another
        # default recompiles instead of reusing the first lambda, and an equal lambda
        # made per call reuses the graph.
        kernel = map_first_kernel if in_tuple else map_kernel
        w = wrap_flydsl(kernel, mutates_args=["Out"])

        def f(t, op):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), op)
            return out

        def scale_by(k):
            op = lambda v, k=k: v * k  # noqa: E731
            return (op,) if in_tuple else op

        cnt = CompileCounterWithBackend(backend)
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        a = _rand(1000)
        for k in (5.0, 6.0, 6.0, 6):
            self.assertEqual(cf(a, scale_by(k)), k * a)
        self.assertEqual(cnt.frame_count, 3)

    @parametrize("backend", ("eager", "aot_eager"))
    def test_constexpr_lambda_in_tuple_made_in_the_frame(self, backend):
        # The same for a lambda made in the frame inside a Constexpr tuple.
        w = wrap_flydsl(map_first_kernel, mutates_args=["Out"])

        def f(t):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), (lambda v: v * 2.0,))
            return out

        a = _rand(1000)
        self.assertEqual(torch.compile(f, backend=backend, fullgraph=True)(a), 2 * a)

    @parametrize("backend", ("eager", "aot_eager"))
    def test_constexpr_lambda_symbolic_default(self, backend):
        # A lambda made in the frame takes frame values through its defaults (FlyDSL
        # refuses closures). A symbolic one (a float or int input once it is
        # dynamic) is specialized, as in a Constexpr argument.
        w = wrap_flydsl(map_kernel, mutates_args=["Out"])
        w_first = wrap_flydsl(map_first_kernel, mutates_args=["Out"])

        def f(t, k):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), lambda v, k=k: v * k)
            return out

        def g(t, k):
            out = torch.zeros_like(t)
            w_first(t, out, t.numel(), (lambda v, k=k: v * k,))
            return out

        a = _rand(1000)
        cases = (
            (f, (5.0, 6.0), None),
            (f, (5, 6), None),
            (f, (5.0,), True),
            (f, (5,), True),
            (g, (5.0,), True),
        )
        for fn, ks, dynamic in cases:
            torch._dynamo.reset()
            cf = torch.compile(fn, backend=backend, fullgraph=True, dynamic=dynamic)
            for k in ks:
                self.assertEqual(cf(a, k), k * a)

    def test_inline_wrapper_live_across_graph_break(self):
        # A wrapper made in the frame is rebuilt at the graph break, so both parts
        # compile instead of the frame falling back to eager.
        def f(x, y):
            w = wrap_flydsl(add_kernel, mutates_args=["Out"])
            out = torch.zeros_like(x)
            torch._dynamo.graph_break()
            w(x, y, out, x.numel())
            return out

        cnt = CompileCounter()
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, backend=cnt)(x, y), x + y)
        self.assertEqual(cnt.frame_count, 2)

    def test_wrapper_type_inside_compile(self):
        from torch._higher_order_ops.flydsl_kernel_wrap import (
            TraceableFlyDSLKernelWrapper,
        )

        def f(x):
            return x + int(isinstance(add_w, TraceableFlyDSLKernelWrapper))

        x = _rand(4)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), x + 1)

    def test_numeric_constant_fullgraph(self):
        # A constant that is not a Python constant (a FlyDSL Numeric value) is passed
        # as the object, guarded on its type and value, by which the side table keys
        # and copies it: an equal Numeric reuses the graph, a changed value (also of
        # the same object) or another type recompiles.
        def f(x, alpha):
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), alpha)
            return out

        x = _rand(1000)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        alpha = fx.Float32(2.0)
        self.assertEqual(cf(x, alpha), x + 2)
        self.assertEqual(cf(x, fx.Float32(2.0)), x + 2)
        self.assertEqual(cnt.frame_count, 1)
        alpha.value = 3.0
        self.assertEqual(cf(x, alpha), x + 3)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cf(x, fx.Int32(3)), x + 3)
        self.assertEqual(cnt.frame_count, 3)

    def test_constant_modified_in_region_graph_breaks(self):
        # Dynamo applies a write made earlier in the region only after the graph runs,
        # so a constant written before the launch is a graph break (fullgraph raises),
        # and the call then runs eagerly, seeing the write.
        def f(x, alpha):
            alpha.value = 3.0
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), alpha)
            return out

        x = _rand(1000)
        cf = torch.compile(f, backend="aot_eager", fullgraph=True)
        with self.assertRaisesRegex(
            Exception,
            r"add_scalar_kernel\(\): L\['alpha'\], reached through 'k', is modified",
        ):
            cf(x, fx.Float32(2.0))
        out = torch.compile(f, backend="aot_eager")(x, fx.Float32(2.0))
        self.assertEqual(out, x + 3)

    def test_nested_symint_in_constexpr_tuple(self):
        # Specialized like a bare SymInt: one graph per value.
        w = wrap_flydsl(scale_by_first_kernel, mutates_args=["Out"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), (x.shape[0] // 250,))
            return out

        cf = torch.compile(f, backend="aot_eager", dynamic=True, fullgraph=True)
        for n in (1000, 2000):
            x = _rand(n)
            self.assertEqual(cf(x), (n // 250) * x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_global_wrapper_fullgraph(self, backend):
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(2048), _rand(2048)
        self.assertEqual(f(x, y), (x + y).relu())
        self.assertEqual(cnt.frame_count, 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_inline_wrap_flydsl_fullgraph(self, backend):
        def f(x, y):
            out = torch.empty_like(x)
            wrap_flydsl(add_kernel, mutates_args=["Out"])(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        compiled = torch.compile(f, backend=backend, fullgraph=True)
        self.assertEqual(compiled(x, y), x + y)

    def test_dynamo_graph_has_one_hop_node(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(x, y), x + y)
        (gm,) = backend.graphs
        (node,) = _hop_nodes(gm, flydsl_kernel_wrapper_mutation)
        record = flydsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertIs(record.fn, add_kernel)
        self.assertEqual(record.mutates_args, ("Out",))
        self.assertNotIn("stream", node.kwargs["kwargs"])

    def test_aot_eager_graph_is_functional(self):
        backend = AotEagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(x, y), x + y)
        (fw,) = backend.fw_graphs
        self.assertEqual(_hop_nodes(fw, flydsl_kernel_wrapper_mutation), [])
        (node,) = _hop_nodes(fw, flydsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["Out"])

    @parametrize("view", ["offset", "gap"])
    def test_aot_eager_mutated_view_gets_tensor_bases(self, view):
        # The production functionalization path (AOTAutograd) hands the base of a
        # mutated view to the functional HOP.
        backend = AotEagerAndRecordGraphs()

        def take_view(base):
            return base[4:] if view == "offset" else base[::2]

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            base = x * 1
            v = take_view(base)
            scale_inplace_w(v, v.numel())
            return base

        x = torch.arange(1, 1001, device=DEVICE, dtype=torch.float32)
        expected = x.clone()
        take_view(expected).mul_(2)
        self.assertEqual(f(x), expected)
        (node,) = _hop_nodes(backend.fw_graphs[0], flydsl_kernel_wrapper_functional)
        self.assertEqual(list(node.kwargs["tensor_bases"]), ["X"])

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("grad_fn", [torch.no_grad, torch.enable_grad])
    def test_wrapper_with_grad_option(self, backend, grad_fn):
        # Mirrors test_triton_kernel_with_grad_option.
        def f(x):
            with grad_fn():
                out = torch.empty_like(x)
                mul_const_w(x, out, x.numel(), 2.0)
                return out

        x = _rand(1000)
        self.assertEqual(torch.compile(f, backend=backend, fullgraph=True)(x), 2 * x)

    def test_runtime_scalar_stays_dynamic(self):
        # n is annotated fx.Int32: a runtime arg, so a new size does not recompile
        # under dynamic=True.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        for n in (1000, 3000, 77):
            x, y = _rand(n), _rand(n)
            self.assertEqual(f(x, y), x + y)
        self.assertEqual(cnt.frame_count, 1)

    def test_constexpr_specializes_and_recompiles(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x, alpha_from_shape):
            out = torch.empty_like(x)
            alpha = x.shape[0] if alpha_from_shape else 3
            mul_const_w(x, out, x.numel(), alpha)
            return out

        # A constant Constexpr value: shape changes stay dynamic.
        for n in (8, 16):
            x = _rand(n)
            self.assertEqual(f(x, False), 3 * x)
        self.assertEqual(cnt.frame_count, 1)
        # A symbolic value for a Constexpr param is guarded on: a new value
        # recompiles, a seen value does not.
        for n in (8, 16, 8):
            x = _rand(n)
            self.assertEqual(f(x, True), n * x)
        self.assertEqual(cnt.frame_count, 3)

    def test_unannotated_scalar_stays_dynamic(self):
        # A Python int for an unannotated parameter is a runtime Int32 in FlyDSL: a
        # symbolic value stays a graph input and new values do not recompile.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x):
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), x.shape[0])
            return out

        for n in (8, 16, 8):
            x = _rand(n)
            self.assertEqual(f(x), x + n)
        self.assertEqual(cnt.frame_count, 1)

    def test_float_scalar_default_and_explicit(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, y):
            s1, d1 = torch.empty_like(x), torch.empty_like(x)
            add_sub_w(x, y, s1, d1, x.numel())
            s2, d2 = torch.empty_like(x), torch.empty_like(x)
            add_sub_w(x, y, s2, d2, x.numel(), scale=0.25)
            return s1, d1, s2, d2

        x, y = _rand(1000), _rand(1000)
        s1, d1, s2, d2 = f(x, y)
        self.assertEqual((s1, d1), (x + y, x - y))
        self.assertEqual((s2, d2), ((x + y) * 0.25, (x - y) * 0.25))

    def test_wrapper_guards_on_kernel_and_write_set(self):
        # A wrapper passed in is guarded on its host function and write set, not on
        # its identity: an equal wrapper reuses the graph, another one recompiles.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def f(w, x, y):
            out = torch.zeros_like(x)
            w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(add_w, x, y), x + y)
        self.assertEqual(f(wrap_flydsl(add_kernel, mutates_args=["Out"]), x, y), x + y)
        self.assertEqual(cnt.frame_count, 1)
        unknown_w = wrap_flydsl(add_kernel, mutates_args="unknown")
        self.assertEqual(f(unknown_w, x, y), x + y)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(f(same_name_sub_w, x, y), x - y)
        self.assertEqual(cnt.frame_count, 3)

    def test_raw_flyc_jit_call_graph_breaks(self):
        def f(x, y):
            out = torch.empty_like(x)
            add_kernel(x, y, out, x.numel())
            return out

        x, y = _rand(16), _rand(16)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "wrap_flydsl"):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)

    def test_raw_flyc_jit_call_runs_eagerly_after_graph_break(self):
        # Without fullgraph, a correct raw FlyDSL call (torch tensors and a raw
        # stream) graph-breaks with the wrap_flydsl hint and then runs eagerly.
        def f(x, a, b, out, n, stream):
            y = x + 1
            add_kernel(a, b, out, n, stream)
            return y * 2

        x, a, b = _rand(1000), _rand(1000), _rand(1000)
        out = torch.zeros_like(a)
        torch._dynamo.utils.counters.clear()
        res = torch.compile(f, backend="eager")(x, a, b, out, 1000, _raw_stream())
        self.assertEqual(res, (x + 1) * 2)
        self.assertEqual(out, a + b)
        reasons = list(torch._dynamo.utils.counters["graph_break"])
        self.assertTrue(any("wrap_flydsl" in r for r in reasons), reasons)

    def test_strict_export(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(M(), (x, y), strict=True)
        self.assertIn(
            "torch.ops.higher_order.flydsl_kernel_wrapper_", ep.graph_module.code
        )
        self.assertEqual(ep.module()(x, y), x + y)
        self.assertEqual(ep.run_decompositions().module()(x, y), x + y)

    @parametrize("strict", [False, True])
    def test_later_strict_export_keeps_earlier_programs_valid(self, strict):
        # Strict export runs Dynamo, which resets per-process tables (the Triton
        # kernel side table is reset by torch._dynamo.reset()). A program exported
        # earlier must keep launching its own kernel, not whatever now sits at its
        # kernel_idx.
        class Add(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        class Scale(torch.nn.Module):
            def forward(self, x):
                y = x.clone()
                scale_inplace_w(y, y.numel())
                return y

        x, y = _rand(1000), _rand(1000)
        ep1 = torch.export.export(Add(), (x, y), strict=strict)
        torch._dynamo.reset()
        ep2 = torch.export.export(Scale(), (x,), strict=True)
        self.assertEqual(ep1.module()(x, y), x + y)
        self.assertEqual(ep1.run_decompositions().module()(x, y), x + y)
        self.assertEqual(ep2.module()(x), 2 * x)

    def test_global_wrapper_survives_dynamo_reset(self):
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        gm = make_fx(f)(x, y)
        self.assertEqual(
            torch.compile(f, backend="aot_eager", fullgraph=True)(x, y), x + y
        )
        torch._dynamo.reset()
        # Another kernel takes the next side-table slot after the reset.
        z = _rand(1000)
        scale_inplace_w(z.clone(), 1000)
        self.assertEqual(
            torch.compile(f, backend="aot_eager", fullgraph=True)(x, y), x + y
        )
        self.assertEqual(gm(x, y), x + y)
        self.assertEqual(make_fx(f)(x, y)(x, y), x + y)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_kernel_output_in_autograd_graph(self, backend):
        # The launch has no derivative: its output enters the autograd graph as a
        # constant, exactly as in eager.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x.detach(), y.detach(), out, x.numel())
            return (out * x).sum()

        x, y = _rand(1000, requires_grad=True), _rand(1000)
        f(x, y).backward()
        eager_grad = x.grad.clone()
        x.grad = None
        torch.compile(f, backend=backend, fullgraph=True)(x, y).backward()
        self.assertEqual(x.grad, eager_grad)
        self.assertEqual(eager_grad, (x + y).detach())


@requires_flydsl
class FlyDSLInductorTests(TestCase):
    """Inductor: the HOP lowers to ir.UserDefinedFlyDSLKernel, launched from the
    Python wrapper as `<launcher>.run(<args>, stream=raw_streamN)`."""

    hw_classification = HardwareClassification.CUDA

    def test_global_rebind_reaches_compiled_graphs(self):
        # The launcher keys its compiles on the module globals the kernel reads, so
        # a compiled graph (and a recompile after torch._dynamo.reset()) uses a
        # rebound global's new value.
        module = sys.modules[__name__]
        self.addCleanup(setattr, module, "_GLOBAL_SCALE", _GLOBAL_SCALE)
        w = wrap_flydsl(scale_global_kernel, mutates_args=["Out"])

        def f(t):
            out = torch.zeros_like(t)
            w(t, out, t.numel())
            return out + 1

        a = _rand(1000)
        cf = torch.compile(f, backend="inductor", fullgraph=True)
        self.assertEqual(cf(a), 2 * a + 1)
        module._GLOBAL_SCALE = 3.0
        self.assertEqual(cf(a), 3 * a + 1)
        torch._dynamo.reset()
        self.assertEqual(torch.compile(f, backend="inductor")(a), 3 * a + 1)

    def test_constexpr_lambdas(self):
        # Constexpr lambdas through the Python wrapper: one made in the frame (named
        # "<lambda>" for FlyDSL) and ones passed in with other defaults (guarded).
        w = wrap_flydsl(map_kernel, mutates_args=["Out"])

        def inline(t):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), lambda v: v * 2.0)
            return out

        def passed(t, op):
            out = torch.zeros_like(t)
            w(t, out, t.numel(), op)
            return out

        def scale_by(k):
            return lambda v, k=k: v * k

        a = _rand(1000)
        self.assertEqual(torch.compile(inline, fullgraph=True)(a), 2 * a)
        cf = torch.compile(passed, fullgraph=True)
        self.assertEqual(cf(a, scale_by(5.0)), 5 * a)
        self.assertEqual(cf(a, scale_by(6.0)), 6 * a)

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_keyword_only_params(self, backend):
        a, b = _rand(1000), _rand(1000)
        kwonly_n_w = wrap_flydsl(kwonly_n_kernel, mutates_args=["Out"])
        kwonly_out_w = wrap_flydsl(kwonly_out_kernel, mutates_args=["Out"])
        kwonly_alpha_w = wrap_flydsl(kwonly_alpha_kernel, mutates_args=["Out"])

        def f(t, u):
            o1, o2, o3 = (torch.zeros_like(t) for _ in range(3))
            kwonly_n_w(t, u, o1, n=t.numel())
            kwonly_out_w(t, u, t.numel(), Out=o2)
            kwonly_alpha_w(t, o3, t.numel(), alpha=3.0)
            return o1, o2, o3

        res = torch.compile(f, backend=backend, fullgraph=True)(a, b)
        self.assertEqual(res, (a + b, a + b, 3 * a))

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_unannotated_numeric_defaults(self, backend):
        # A Numeric default of an unannotated parameter is a constant of FlyDSL's
        # own type: an Int64 that does not fit an Int32, a Float64 that is not a
        # Float32.
        xi = torch.arange(1000, device=DEVICE, dtype=torch.int64)
        xd = _rand(1000, dtype=torch.float64)
        for host_fn, t, expected in (
            (int64_default_kernel, xi, xi + 10**10),
            (big_int64_default_kernel, xi, xi + 3_000_000_000**2),
            (float64_default_kernel, xd, xd + 0.1),
        ):
            w = wrap_flydsl(host_fn, mutates_args=["Out"])

            def f(t):
                out = torch.zeros_like(t)
                w(t, out, t.numel())
                return out

            with self.subTest(host_fn=host_fn.func.__name__):
                torch._dynamo.reset()
                res = torch.compile(f, backend=backend, fullgraph=True)(t)
                self.assertEqual(res, expected, atol=0, rtol=0)

    def test_aoti_keyword_only_and_unannotated_numeric_default(self):
        # AOTInductor: keyword-only parameters export as the others do. A Numeric
        # default of an unannotated parameter is a constant to PT2 but a run-time
        # argument of FlyDSL's C ABI: refused, with a hint (as for a Numeric value
        # passed to one).
        def package(model, args):
            ep = torch.export.export(model, args, strict=False)
            return torch._inductor.aoti_compile_and_package(ep)

        kwonly_w = wrap_flydsl(kwonly_n_kernel, mutates_args=["Out"])
        default_w = wrap_flydsl(int64_default_kernel, mutates_args=["Out"])

        class KwOnly(torch.nn.Module):
            def forward(self, t, u):
                out = torch.empty_like(t)
                kwonly_w(t, u, out, n=t.numel())
                return out

        class Default(torch.nn.Module):
            def forward(self, t):
                out = torch.empty_like(t)
                default_w(t, out, t.numel())
                return out

        a, b = _rand(1000), _rand(1000)
        runner = torch._inductor.aoti_load_package(package(KwOnly(), (a, b)))
        self.assertEqual(runner(a, b), a + b)
        xi = torch.arange(1000, device=DEVICE, dtype=torch.int64)
        with self.assertRaisesRegex(
            Exception, r"FlyDSL passes \['k'\] at run time: annotate them"
        ):
            package(Default(), (xi,))

    def test_flydsl_kernel_inductor_basic(self):
        from torch._inductor import ir

        codegen_calls = []
        orig_codegen = ir.UserDefinedFlyDSLKernel.codegen

        def spy_codegen(self, wrapper):
            codegen_calls.append(self)
            return orig_codegen(self, wrapper)

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(ir.UserDefinedFlyDSLKernel, "codegen", spy_codegen):
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y).relu())
        self.assertEqual(len(codegen_calls), 1)
        self.assertIs(codegen_calls[0].kernel.fn, add_kernel)
        launches = _flydsl_launches(code, "add_kernel")
        self.assertEqual(launches, [("flydsl_add_kernel_0", mock.ANY)], code)
        # Launched on the stream Inductor fetched for the device (named
        # `raw_stream0` or `stream0` depending on the version), never a constant.
        self.assertRegex(launches[0][1], r"stream=(raw_)?stream\d+$")
        # The header comment names the user's function (JitFunction.func).
        FileCheck().check(
            "# flydsl_add_kernel_0: FlyDSL host function "
            f"{add_kernel.func.__module__}.add_kernel"
        ).run(code)

    def test_flydsl_kernel_symbolic_bool_scalar(self):
        # A symbolic bool for an unannotated param is a runtime scalar (a FlyDSL
        # Boolean), evaluated at each launch like a symbolic int.
        w = wrap_flydsl(select_kernel, mutates_args=["Out"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), x.shape[0] > 1500)
            return out

        cf = torch.compile(f, dynamic=True, fullgraph=True)
        for n in (1000, 2000):
            x = _rand(n)
            self.assertEqual(cf(x), (3 if n > 1500 else 5) * x)

    def test_flydsl_kernel_numeric_scalar_values(self):
        # A FlyDSL Numeric default, and a Numeric value passed for a Numeric param,
        # reach the generated code as Python floats.
        w = wrap_flydsl(add_c_kernel, mutates_args=["Out"])
        c = fx.Float32(0.75)

        def f(x):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel())
            w(x, b, x.numel(), c=c)
            return a, b

        x = _rand(1000)
        (a, b), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual(a, x + 0.25)
        self.assertEqual(b, x + 0.75)
        args = [a for _, a in _flydsl_launches(code, "add_c_kernel")]
        self.assertEqual(len(args), 2, code)
        self.assertTrue(any(", 0.25, stream=" in a for a in args), args)
        self.assertTrue(any(", 0.75, stream=" in a for a in args), args)

    def test_flydsl_kernel_graph_not_serializable(self):
        # The generated module binds launchers of this process's host functions, so
        # serializing the compiled graph (aot_compile) fails loudly.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(Exception, "cannot be serialized"):
            torch.compile(f, fullgraph=True).aot_compile(((x, y), {}))

    @parametrize("grad", [False, True])
    @parametrize("dynamic", [False, True])
    @inductor_config.patch("implicit_fallbacks", False)
    def test_flydsl_kernel_no_clones(self, grad, dynamic):
        # Mirrors test_triton_kernel_no_clones: with the right mutates_args the
        # kernel writes the graph input in place, no clone or copy-back.
        def call_flydsl(x, y, output):
            n_elements = output.numel()
            tmp = torch.add(x, 1)
            add_w(x, y, output, n_elements)
            return output, tmp

        t1 = _rand(5, requires_grad=grad)
        t2 = _rand(5, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)
        torch_add = call_flydsl(t1, t2, o1)
        o2 = torch.zeros_like(t1, requires_grad=grad)
        test, (code,) = run_and_get_code(
            torch.compile(call_flydsl, dynamic=dynamic), t1, t2, o2
        )
        self.assertEqual(torch_add, test)
        # Like the Triton test, these rely on `Original ATen:` comments, so a clone
        # fused into some kernel is caught too.
        self.assertNotIn("aten.copy", code)
        self.assertNotIn("aten.clone", code)
        self.assertEqual(len(_flydsl_launches(code, "add_kernel")), 1, code)
        if not grad:
            self.assertIn("return (buf0, )", code)

    def test_flydsl_kernel_reinplaces_intermediate(self):
        def f(x, y):
            out = torch.zeros_like(x)
            add_w(x, y, out, x.numel())
            s, d = torch.empty_like(x), torch.empty_like(x)
            add_sub_w(out, y, s, d, x.numel(), scale=0.5)
            return s + d

        x, y = _rand(1000), _rand(1000)
        ReinplaceCounters.clear()
        res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y) * 0.5 * 2)
        self.assertEqual(ReinplaceCounters.get_total_missed(), 0)
        self.assertNotIn("aten.clone", code)
        self.assertNotIn("aten.copy", code)

    def test_flydsl_kernel_mutates_graph_input(self):
        def f(x):
            scale_inplace_w(x, x.numel())
            return x.sum()

        x = _rand(1000)
        x_ref = x.clone()
        res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual(x, 2 * x_ref)
        self.assertEqual(res, (2 * x_ref).sum())
        self.assertNotIn("aten.clone", code)
        self.assertNotIn("aten.copy", code)

    @parametrize("view", ["offset", "gap"])
    def test_flydsl_kernel_mutates_graph_input_view(self, view):
        def take_view(t):
            return t[4:] if view == "offset" else t[::2]

        def f(x):
            v = take_view(x)
            scale_inplace_w(v, v.numel())
            return x + 1

        x = torch.arange(1, 1001, device=DEVICE, dtype=torch.float32)
        expected = x.clone()
        take_view(expected).mul_(2)
        res = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(x, expected)
        self.assertEqual(res, expected + 1)

    @parametrize("dynamic", [False, True])
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_flydsl_kernel_with_views(self, dynamic, backend):
        # Mirrors test_triton_kernel_with_views.
        def call_take_view(x):
            output = torch.zeros_like(x)
            mul_const_w(x, output, output.numel(), 2.0)
            return output

        def call_return_view(x):
            output = torch.zeros_like(x)
            mul_const_w(x.view(-1), output.view(-1), output.numel(), 2.0)
            return output.view(4, 4)

        t = _rand(4, 4)
        t_view = t.view(16)
        compiled = torch.compile(
            call_take_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled(t_view))
        self.assertEqual(2 * t, compiled(t_view).view(4, 4))
        compiled = torch.compile(
            call_return_view, backend=backend, fullgraph=True, dynamic=dynamic
        )
        self.assertEqual(2 * t_view, compiled(t).view(16))
        self.assertEqual(2 * t, compiled(t))

    @parametrize("which", ["left", "right", "slice_view", "transposed"])
    def test_flydsl_kernel_strided_input(self, which):
        # Mirrors test_triton_kernel_strided_input{,_nonzero_offset} and
        # test_triton_kernel_slice_and_view_input with a layout-indexing kernel.
        def f(inp):
            if which in ("left", "right"):
                left, right = torch.split(inp, [128, 128], dim=1)
                src = left if which == "left" else right  # strides (256, 1)
            elif which == "slice_view":
                src = inp[:, :128].reshape(64, 128)[:, ::2]
            else:
                src = inp.t()
            out = torch.empty(src.shape, device=src.device)
            add_one_2d_w(src, out)
            return out + src

        inp = _rand(64, 256)
        self.assertEqual(torch.compile(f, fullgraph=True)(inp), f(inp))

    @parametrize("gap", [1, 2])
    def test_flydsl_kernel_mutated_offset_view_not_miscompiled(self, gap):
        # Mirrors test_triton_kernel_mutated_offset_view_not_miscompiled: the kernel
        # writes a (possibly gapped) row view of an intermediate at a non-zero
        # storage offset; elements it does not write must keep their values.
        n_cols, n_rows, skipped_row = 64, 8, 7
        width = n_cols * gap

        def make_base():
            return torch.arange(
                1, n_rows * width + 1, dtype=torch.float32, device=DEVICE
            ).reshape(n_rows, width)

        def f(src):
            base = make_base()
            for i in range(n_rows):
                if i != skipped_row:
                    add_one_2d_w(src, base[i : i + 1, ::gap])
            return base

        src = _rand(1, n_cols)
        expected = make_base()
        for i in range(n_rows):
            if i != skipped_row:
                expected[i : i + 1, ::gap] = src + 1
        self.assertEqual(f(src), expected)
        self.assertEqual(torch.compile(f, fullgraph=True)(src), expected)

    @parametrize("dynamic", [False, True])
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_flydsl_kernel_multiple_outputs(self, dynamic, backend):
        # Mirrors test_triton_kernel_multiple_outputs (incl. a training graph).
        @torch.compile(fullgraph=True, backend=backend, dynamic=dynamic)
        def f(x, y, z):
            s, d = torch.empty_like(x), torch.empty_like(x)
            add_sub_w(x, y, s, d, x.numel())
            return s, d, z**2

        x, y, z = (_rand(300, requires_grad=True) for _ in range(3))
        s, d, z2 = f(x, y, z)
        self.assertEqual(s, x + y)
        self.assertEqual(d, x - y)
        self.assertEqual(z2, z**2)
        z2.sum().backward()
        self.assertEqual(z.grad, 2 * z)

    def test_flydsl_kernel_dependencies(self):
        # Mirrors test_triton_kernel_dependancies: kernel -> kernel -> aten op.
        def f(x, y):
            out = torch.zeros_like(x)
            add_w(x, y, out, x.numel())
            out2 = torch.zeros_like(out)
            add_w(out, y, out2, out.numel())
            return torch.add(out2, 1)

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), f(x, y))

    def test_flydsl_kernel_reinplace_inplaceable_pass(self):
        # Mirrors test_triton_kernel_reinplace_inplaceable_pass: the second launch
        # reads and writes the same buffer through two parameters (X is Out).
        def f(x, y):
            out = torch.zeros_like(x)
            add_w(x, y, out, x.numel())
            add_w(out, x, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), 2 * x + y)

    def test_flydsl_kernel_caching(self):
        # Mirrors test_triton_kernel_caching: one kernel launched in a loop is
        # defined once and launched four times.
        def f(x, y):
            for _ in range(4):
                out = torch.zeros_like(x)
                add_w(x, y, out, x.numel())
                x = out
            return x

        x, y = torch.ones(1000, device=DEVICE), torch.ones(1000, device=DEVICE)
        res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, 5 * torch.ones(1000, device=DEVICE))
        launches = _flydsl_launches(code, "add_kernel")
        self.assertEqual(len(launches), 4, code)
        self.assertEqual(len({name for name, _ in launches}), 1, launches)

    def test_flydsl_kernel_same_name_kernels(self):
        # Mirrors test_triton_kernel_caching_duplicate: two different host
        # functions with one name get two launchers.
        def f(x, y):
            o1, o2 = torch.empty_like(x), torch.empty_like(x)
            same_name_add_w(x, y, o1, x.numel())
            same_name_sub_w(x, y, o2, x.numel())
            return o1, o2

        x, y = _rand(1000), _rand(1000)
        (o1, o2), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(o1, x + y)
        self.assertEqual(o2, x - y)
        launches = _flydsl_launches(code, "binary_kernel")
        self.assertEqual(len({name for name, _ in launches}), 2, launches)

    @parametrize("case", ["triton_first", "flydsl_first", "triton_in_cond"])
    def test_flydsl_kernel_launcher_name_taken_by_a_triton_kernel(self, case):
        # User Triton kernels and launchers are numbered apart, so a Triton kernel
        # named flydsl_add_kernel would be flydsl_add_kernel_0 too, as add_kernel's
        # launcher is, and loading the module would bind the launcher over it. With
        # one set of names, the kernel defined second takes the next free index, also
        # when a torch.cond branch defines it.
        import triton
        import triton.language as tl

        @triton.jit
        def flydsl_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + i, mask=i < n)
            y = tl.load(y_ptr + i, mask=i < n)
            tl.store(out_ptr + i, x - y, mask=i < n)

        def triton_sub(x, y):
            out, n = torch.empty_like(x), x.numel()
            flydsl_add_kernel[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
            return out

        def flydsl_add(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        def f(x, y):
            # The second kernel reads what the first wrote: they run in this order.
            if case == "triton_first":
                first = triton_sub(x, y)
                return first, flydsl_add(first, y)
            first = flydsl_add(x, y)
            if case == "flydsl_first":
                return first, triton_sub(first, y)
            sub = torch.cond(x.sum() > 0, triton_sub, lambda a, b: a - b, (first, y))
            return first, sub

        x, y = _rand(1000).abs(), _rand(1000)
        cf = torch.compile(f, fullgraph=True)
        (first, second), (code,) = run_and_get_code(cf, x, y)
        self.assertEqual(first, x - y if case == "triton_first" else x + y)
        self.assertEqual(second, x)
        (launcher,) = re.findall(r"(?m)^# (\w+): FlyDSL host function ", code)
        self.assertEqual(launcher, f"flydsl_add_kernel_{int(case == 'triton_first')}")
        run = r"\b(flydsl_add_kernel_\d+)\.run\("
        launched = set(re.findall(run, _strip_comments(code)))
        self.assertEqual(launched, {"flydsl_add_kernel_0", "flydsl_add_kernel_1"})

    def test_flydsl_kernel_module_attributes_bound_once(self):
        # Loading the generated module binds its launchers, constants and the classes
        # of opaque values as globals, by name: a launcher named as one of those
        # classes is refused instead of replacing it.
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        from torch._inductor.virtualized import V

        define = PythonWrapperCodegen.define_user_defined_flydsl_kernel

        def define_as_a_class(self, *args, **kwargs):
            name = define(self, *args, **kwargs)
            V.graph.opaque_value_type_classes[name] = type(name, (), {})
            return name

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        name = "define_user_defined_flydsl_kernel"
        with (
            mock.patch.object(PythonWrapperCodegen, name, define_as_a_class),
            self.assertRaisesRegex(
                Exception,
                r"module attributes bound twice: \['flydsl_add_kernel_\d+'\]",
            ),
        ):
            torch.compile(f, fullgraph=True)(x, y)

    def test_flydsl_kernel_in_cond_branches(self):
        # The branches become subgraph functions of the root module. Their launches
        # resolve to launchers the root module binds, one per kernel and constants,
        # shared with the root graph's launches and each defined once.
        def true_fn(x):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), 2)
            return out

        def false_fn(x):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), 5)
            return out

        def f(x):
            a = torch.empty_like(x)
            mul_const_w(x, a, x.numel(), 2)
            return a, torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

        cf = torch.compile(f, fullgraph=True)
        x = _rand(1000).abs()
        (a, b), (code,) = run_and_get_code(cf, x)
        self.assertEqual((a, b), (2 * x, 2 * x))
        self.assertEqual(cf(-x), (-2 * x, -5 * x))
        launches = _flydsl_launches(code, "mul_const_kernel")
        self.assertEqual(len(launches), 3, code)
        fns = re.findall(r"(?m)^def (?!call\()\w+\(args\):\n(?:[ \t].*\n|\n)*", code)
        in_fns = _flydsl_launches("".join(fns), "mul_const_kernel")
        self.assertEqual(len(in_fns), 2, code)
        names = {name for name, _ in launches}
        self.assertEqual(len(names), 2, code)
        for name in names:
            self.assertEqual(len(re.findall(rf"(?m)^# {name}: ", code)), 1, code)

    @parametrize("triton_layout", ["needs_fixed_stride_order", "flexible_layout"])
    def test_flydsl_kernel_layout_constraint_is_its_own(self, triton_layout):
        # FlyDSL's mutation HOP keeps the strides its tensors were traced with
        # (needs_fixed_stride_order, FlyDSL's own constraint) whatever Triton's
        # triton_kernel_default_layout_constraint says: when the HOP is lowered, x is
        # already fixed to its traced strides (1, 8), where a flexible layout would
        # still show Inductor's default, contiguous (4, 1).
        from torch._inductor import ir
        from torch._inductor.lowering import lowerings

        strides = []
        lowering = lowerings[flydsl_kernel_wrapper_mutation]

        def lower(**hop_kwargs):
            x = ir.ExternKernel.realize_input(hop_kwargs["kwargs"]["X"])
            strides.append(tuple(x.get_stride()))
            return lowering(**hop_kwargs)

        def f(inp):
            x = inp.permute(1, 0) * 2
            out = torch.empty_like(x)
            add_one_2d_w(x, out)
            return out

        inp = _rand(4, 8)
        with (
            mock.patch.dict(lowerings, {flydsl_kernel_wrapper_mutation: lower}),
            inductor_config.patch(
                force_disable_caches=True,
                triton_kernel_default_layout_constraint=triton_layout,
            ),
        ):
            out = torch.compile(f, fullgraph=True)(inp)
        self.assertEqual(strides, [(1, 8)])
        self.assertEqual(out, inp.T * 2 + 1)

    def test_flydsl_kernel_constants(self):
        # Constexpr values are baked in per launch site; an unannotated int is a
        # runtime Int32 argument of the launch.
        def f(x):
            o1, o2, o3 = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
            mul_const_w(x, o1, x.numel(), 3)
            mul_const_w(x, o2, x.numel(), 2.5)
            add_scalar_w(x, o3, x.numel(), 5)
            return o1, o2, o3

        x = _rand(1000)
        (o1, o2, o3), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual((o1, o2, o3), (3 * x, 2.5 * x, x + 5))
        self.assertEqual(
            len({n for n, _ in _flydsl_launches(code, "mul_const_kernel")}), 2
        )
        ((_, args),) = _flydsl_launches(code, "add_scalar_kernel")
        self.assertRegex(args, r", 1000, 5, stream=")

    def test_flydsl_kernel_dynamic_shapes(self):
        # One dynamic graph serves every size: the launcher compiles each new layout
        # (FlyDSL kernels are compiled for static layouts).
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x, y, m):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            out2 = torch.empty_like(m)
            add_one_2d_w(m, out2)
            return out, out2

        for n, shape in ((1000, (4, 8)), (3000, (16, 32)), (77, (3, 300))):
            x, y, m = _rand(n), _rand(n), _rand(*shape)
            out, out2 = f(x, y, m)
            self.assertEqual(out, x + y)
            self.assertEqual(out2, m + 1)
        self.assertEqual(cnt.frame_count, 1)

    def test_flydsl_kernel_dynamic_shapes_layout_change(self):
        # A transposed input must not reuse the row-major kernel.
        @torch.compile(fullgraph=True, dynamic=True)
        def f(m):
            out = torch.empty(m.shape, device=m.device)
            add_one_2d_w(m, out)
            return out

        for m in (_rand(4, 8), _rand(16, 32), _rand(8, 4).t(), _rand(32, 16).t()):
            self.assertEqual(f(m), m + 1)

    def test_flydsl_kernel_unaligned_view(self):
        def f(base, v):
            x = base[:, 1:]  # data_ptr 4 bytes past a 16-byte boundary
            out = torch.empty(x.shape, device=x.device)
            add_one_2d_w(x, out)
            out1 = torch.empty_like(v)
            add_w(v, v, out1, v.numel())
            return out, out1

        base, v = _rand(4, 9), _rand(1001)[1:]
        out, out1 = torch.compile(f, fullgraph=True)(base, v)
        self.assertEqual(out, base[:, 1:] + 1)
        self.assertEqual(out1, 2 * v)

    def test_flydsl_kernel_epilogue_and_prologue(self):
        # Inductor ops around the user kernel (it is an extern launch: correctness
        # only, no fusion expected).
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x.sin(), y * 2, out, x.numel())
            return out.relu() + 1

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), f(x, y))

    def test_flydsl_kernel_reduce_overhead(self):
        # CUDA graphs (cudagraph trees, HIP graphs here) capture the launch on the
        # capture stream.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
        counters = torch._dynamo.utils.counters
        counters.clear()
        for _ in range(3):
            x, y = _rand(1000), _rand(1000)
            self.assertEqual(compiled(x, y), (x + y) * 2)
        # The graph was captured, not skipped.
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    def test_flydsl_kernel_inference_mode(self):
        # inference_mode disables view tracking (cf. MutationTests'
        # test_triton_kernel_inference_mode and the inference-mode offset-view test).
        def f(x):
            base = x * 1
            v = base[4:]
            scale_inplace_w(v, v.numel())
            return base

        x = _rand(1000)
        expected = x.clone()
        expected[4:] *= 2
        with torch.inference_mode():
            self.assertEqual(torch.compile(f, fullgraph=True)(x), expected)

    def test_flydsl_kernel_specialized_constants_dynamic(self):
        # A Constexpr value computed from a dynamic size is specialized (guarded):
        # the generated code must bake each value in. An unannotated int computed
        # from it stays a runtime scalar.
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x):
            o1, o2 = torch.empty_like(x), torch.empty_like(x)
            mul_const_w(x, o1, x.numel(), x.shape[0] // 8)
            add_scalar_w(x, o2, x.numel(), x.shape[0])
            return o1, o2

        for n in (8, 16, 8):
            x = _rand(n)
            o1, o2 = f(x)
            self.assertEqual(o1, (n // 8) * x)
            self.assertEqual(o2, x + n)

    @inductor_config.patch(fx_graph_cache=True, fx_graph_remote_cache=False)
    def test_flydsl_kernel_bypasses_fx_graph_cache(self):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with fresh_cache():
            counters.clear()
            self.assertEqual(torch.compile(f)(x, y), (x + y).relu())
            torch._dynamo.reset()
            self.assertEqual(torch.compile(f)(x, y), (x + y).relu())
        # The generated module binds the launcher in-process: never cached.
        self.assertGreater(counters["inductor"]["fxgraph_cache_bypass"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    def test_flydsl_kernel_launcher_caches_compiled_functions(self):
        # The Python wrapper's launcher compiles each specialization once (a fresh
        # private JitFunction per compile) and then calls the CompiledFunction.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        compiled = torch.compile(f, fullgraph=True)
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(compiled(x, y), x + y)
        with mock.patch.object(
            flydsl_user_kernel,
            "compile_host_fn",
            wraps=flydsl_user_kernel.compile_host_fn,
        ) as compile_host_fn:
            for _ in range(3):
                x, y = _rand(1000), _rand(1000)
                self.assertEqual(compiled(x, y), x + y)
        self.assertEqual(compile_host_fn.call_count, 0)


class _AddModule(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_w(x, y, out, x.numel())
        return out


class _ScaleModule(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        scale_inplace_w(y, y.numel())
        return y


@contextlib.contextmanager
def _captured_stderr_fd():
    """What is written to file descriptor 2 inside the block (C++ std::cerr
    included), as the only element of the yielded list once the block exits."""
    captured = []
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield captured
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            captured.append(tmp.read().decode(errors="replace"))


def _package_so(pkg, out_dir):
    """The model .so inside AOTInductor package ``pkg``, extracted to ``out_dir``."""
    with zipfile.ZipFile(pkg) as z:
        (name,) = [n for n in z.namelist() if n.endswith(".so")]
        return z.extract(name, out_dir)


# A module global _bias_k reads (test_aoti_reexport_after_global_rebind rebinds it).
_AOTI_BIAS = 2.0

# Module globals outside the kernel prefix, which test_aoti_reexport_after_rebind_
# outside_prefix rebinds: a tuple _tuple_scale_k reads, and a float read by the plain
# helper _helper_scale_k calls.
_AOTI_SCALE = (2.0,)
_AOTI_FACTOR = 2.0


def _aoti_scaled(v):
    return v * _AOTI_FACTOR


if HAS_FLYDSL:
    # --- BEGIN AOTI CASE KERNELS ---
    # out = x + _AOTI_BIAS: the global's value is baked into the compiled kernel.
    @flyc.kernel
    def _bias_k(X, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] + _AOTI_BIAS

    @flyc.jit
    def bias_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
        _bias_k(X, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # out = x * _AOTI_SCALE[0] and out = _aoti_scaled(x) = x * _AOTI_FACTOR.
    @flyc.kernel
    def _tuple_scale_k(X, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] * _AOTI_SCALE[0]

    @flyc.jit
    def tuple_scale_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
        _tuple_scale_k(X, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    @flyc.kernel
    def _helper_scale_k(X, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = _aoti_scaled(X[i])

    @flyc.jit
    def helper_scale_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
        _helper_scale_k(X, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # out = x + k for an Int64 scalar k.
    @flyc.kernel
    def _add_i64_k(X, Out, n: fx.Int32, k: fx.Int64):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            Out[i] = X[i] + k

    @flyc.jit
    def add_i64_kernel(X, Out, n: fx.Int32, k: fx.Int64, stream: fx.Stream):
        _add_i64_k(X, Out, n, k).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # out[0] = the high 32 bits of a Uint64 scalar.
    @flyc.kernel
    def _u64_hi_k(Out, s: fx.Uint64):
        if fx.thread_idx.x < 1:
            Out[0] = fx.Int32(s >> 32)

    @flyc.jit
    def u64_hi_kernel(Out, s: fx.Uint64, stream: fx.Stream):
        _u64_hi_k(Out, s).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    # Parameters named like C++ keywords (new, default), a C macro (errno) and the
    # generated launcher's stream parameter: errno = old + new * default,
    # stream = old - new.
    @flyc.kernel
    def _kw_blend_k(old, new, errno, stream, n: fx.Int32, default: fx.Float32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            errno[i] = old[i] + new[i] * default
            stream[i] = old[i] - new[i]

    @flyc.jit
    def kw_blend_kernel(
        old, new, errno, stream, n: fx.Int32, default: fx.Float32, s: fx.Stream
    ):
        _kw_blend_k(old, new, errno, stream, n, default).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=s
        )

    # add_one_2d_kernel with an exact block size, and with a user maximum flat
    # work-group size of 256 threads.
    @flyc.kernel(known_block_size=[64, 1, 1])
    def _add_one_2d_kbs_k(X, Out):
        i = fx.block_idx.x
        j = fx.thread_idx.x
        Out[i, j] = X[i, j] + 1.0

    @flyc.jit
    def add_one_2d_kbs_kernel(X, Out, stream: fx.Stream):
        m, n = X.shape.unpack()
        _add_one_2d_kbs_k(X, Out).launch(grid=(m, 1, 1), block=(n, 1, 1), stream=stream)

    @flyc.jit
    def add_one_2d_fwg256_kernel(X, Out, stream: fx.Stream):
        m, n = X.shape.unpack()
        _add_one_2d_k(
            X, Out, value_attrs={"rocdl.flat_work_group_size": "1,256"}
        ).launch(grid=(m, 1, 1), block=(n, 1, 1), stream=stream)

    # out = x + 1 through side-effecting inline asm on a loaded value: the write-set
    # analysis cannot tell what the asm writes, so "infer" gives every tensor.
    from flydsl._mlir import ir as _mlir_ir
    from flydsl._mlir.dialects import llvm as _mlir_llvm

    @flyc.kernel
    def _asm_add_one_k(X, Out, n: fx.Int32):
        i = fx.block_idx.x * 1024 + fx.thread_idx.x
        if i < n:
            v = _mlir_llvm.inline_asm(
                _mlir_ir.F32Type.get(),
                [X[i].ir_value()],
                "v_mov_b32 $0, $1",
                "=v,v",
                has_side_effects=True,
            )
            Out[i] = fx.Float32(v) + 1.0

    @flyc.jit
    def asm_add_one_kernel(X, Out, n: fx.Int32, stream: fx.Stream):
        _asm_add_one_k(X, Out, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    # --- END AOTI CASE KERNELS ---


@requires_flydsl
class FlyDSLAOTITests(TestCase):
    """AOTInductor: the kernel is compiled ahead of time (export_to_c), its object is
    linked into the model .so, and the generated C++ launches it through the
    header's <prefix>_call after <prefix>__module_load(-1)."""

    hw_classification = HardwareClassification.CUDA

    def _package(self, model, args, dynamic_shapes=None, strict=False):
        ep = torch.export.export(
            model, args, dynamic_shapes=dynamic_shapes, strict=strict
        )
        return torch._inductor.aoti_compile_and_package(ep)

    def _package_and_code(self, model, args, **kwargs):
        # run_and_get_cpp_code starts with torch._dynamo.reset(), so the export has
        # to happen inside it (see test_aoti_program_exported_before_reset).
        return run_and_get_cpp_code(self._package, model, args, **kwargs)

    def test_aoti_add(self):
        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(_AddModule(), (x, y))
        code = _strip_comments(code)
        # One launcher per export, which loads the module on the current device and
        # calls the typed entry point of the spliced header.
        FileCheck().check("static inline void call_flydsl_add_kernel_gfx").check(
            "__module_load(-1)"
        ).check("_call(").run(code)
        self.assertRegex(code, r"call_flydsl_add_kernel_gfx\w+\(\w+\.data_ptr\(\)")
        runner = torch._inductor.aoti_load_package(pkg)
        self.assertEqual(runner(x, y), x + y)
        x2, y2 = _rand(1000), _rand(1000)
        self.assertEqual(runner(x2, y2), x2 + y2)

    def test_aoti_flydsl_symbols_stay_local(self):
        # The version script keeps the five symbols of each export out of the model
        # .so's dynamic symbol table (two model .so files could otherwise bind each
        # other's kernels).
        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(_AddModule(), (x, y))
        prefix = re.findall(r"\bcall_(flydsl_add_kernel_gfx\w+)\(", code)[0]
        with tempfile.TemporaryDirectory() as d:
            lib = ctypes.CDLL(_package_so(pkg, d))
            self.assertTrue(hasattr(lib, "AOTInductorModelContainerCreateWithDevice"))
            for suffix in ("", "__module_init", "__module_load", "__metadata"):
                self.assertFalse(hasattr(lib, prefix + suffix), prefix + suffix)
        self.assertEqual(torch._inductor.aoti_load_package(pkg)(x, y), x + y)

    @parametrize("strict", [False, True])
    def test_aoti_dynamic_shapes(self, strict):
        dim = torch.export.Dim("n", min=2, max=8192)
        pkg = self._package(
            _AddModule(),
            (_rand(1000), _rand(1000)),
            dynamic_shapes={"x": {0: dim}, "y": {0: dim}},
            strict=strict,
        )
        runner = torch._inductor.aoti_load_package(pkg)
        for n in (1000, 77, 8192):
            x, y = _rand(n), _rand(n)
            self.assertEqual(runner(x, y), x + y)

    def test_aoti_2d_view_dynamic(self):
        # Dynamic sizes and strides of a transposed view: the packed layout structs
        # carry int32 sizes and int64 strides.
        class M(torch.nn.Module):
            def forward(self, x):
                xt = x.t()  # strides (1, x.shape[1]): a non-contiguous view
                out = torch.empty(xt.shape, device=x.device)
                add_one_2d_w(xt, out)
                return out

        a, b = (
            torch.export.Dim("a", min=2, max=512),
            torch.export.Dim("b", min=2, max=512),
        )
        pkg, code = self._package_and_code(
            M(), (_rand(8, 4),), dynamic_shapes={"x": {0: a, 1: b}}
        )
        # x.t() has strides (1, b): int32 sizes and one int64 stride.
        self.assertIn(
            "{static_cast<int32_t>(flydsl_arg0_size0), "
            "static_cast<int32_t>(flydsl_arg0_size1)}, "
            "{static_cast<int64_t>(a0_stride0)}}",
            code,
        )
        runner = torch._inductor.aoti_load_package(pkg)
        for shape in ((8, 4), (33, 17), (4, 300)):
            x = _rand(*shape)
            self.assertEqual(runner(x), x.t() + 1)

    def test_aoti_constexpr(self):
        class M(torch.nn.Module):
            def forward(self, x):
                o1, o2 = torch.empty_like(x), torch.empty_like(x)
                mul_const_w(x, o1, x.numel(), 3)
                mul_const_w(x, o2, x.numel(), 2.5)
                return o1, o2

        x = _rand(1000)
        pkg, code = self._package_and_code(M(), (x,))
        # One export per Constexpr value.
        self.assertEqual(code.count("#define FLYDSL_DECLARED_"), 2)
        o1, o2 = torch._inductor.aoti_load_package(pkg)(x)
        self.assertEqual((o1, o2), (3 * x, 2.5 * x))

    def test_aoti_boolean_and_float_scalars(self):
        # A Boolean takes the truthiness of its value (x.shape[0] % 3 can be 2), as
        # in eager; a defaulted Float32 is passed as a float.
        w = wrap_flydsl(select_bool_kernel, mutates_args=["Out"])
        add_c_w = wrap_flydsl(add_c_kernel, mutates_args=["Out"])

        class M(torch.nn.Module):
            def forward(self, x):
                a, b = torch.empty_like(x), torch.empty_like(x)
                w(x, a, x.numel(), x.shape[0] % 3)
                add_c_w(x, b, x.numel())
                return a, b

        n = torch.export.Dim("n", min=2, max=4096)
        pkg = self._package(M(), (_rand(1000),), dynamic_shapes={"x": {0: n}})
        run = torch._inductor.aoti_load_package(pkg)
        for size in (1000, 1001, 1002):  # size % 3 = 1, 2, 0
            x = _rand(size)
            a, b = run(x)
            self.assertEqual(a, (3 if size % 3 else 5) * x)
            self.assertEqual(b, x + 0.25)

    def test_aoti_unannotated_scalar(self):
        # FlyDSL types an unannotated scalar from each launch's Python value; the
        # exported C ABI needs its type fixed, so AOTInductor asks for an annotation.
        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                add_scalar_w(x, out, x.numel(), 5)
                return out

        x = _rand(1000)
        # The Python wrapper launches it fine ...
        self.assertEqual(torch.compile(M(), fullgraph=True)(x), x + 5)
        # ... AOTInductor cannot export it.
        with self.assertRaisesRegex(Exception, "annotated with a FlyDSL Numeric"):
            self._package(M(), (x,))

    def test_aoti_half_scalar_refused(self):
        # export_to_c passes Float16 / BFloat16 scalars as raw bits, which the C++
        # launcher does not compute.
        w = wrap_flydsl(add_half_kernel, mutates_args=["Out"])

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                w(x, out, x.numel(), 0.5)
                return out

        with self.assertRaisesRegex(Exception, "Float16/BFloat16 scalars as raw bits"):
            self._package(M(), (_rand(1000),))

    def test_aoti_int32_scalar_range_checked(self):
        # The exported ABI takes n as an int32_t: the launcher refuses a value that
        # does not fit instead of letting it wrap (as FlyDSL would).
        class M(torch.nn.Module):
            def forward(self, x, y, m):
                out = torch.zeros_like(x)
                add_w(x, y, out, m.shape[0] * 256)
                return out

        x, y = _rand(1000), _rand(1000)
        dim = torch.export.Dim("m", min=2, max=2**23)
        pkg = self._package(
            M(), (x, y, _rand(2)), dynamic_shapes={"x": None, "y": None, "m": {0: dim}}
        )
        runner = torch._inductor.aoti_load_package(pkg)
        out = runner(x, y, _rand(2))  # n = 512
        self.assertEqual(out[:512], (x + y)[:512])
        self.assertEqual(out[512:], torch.zeros_like(x[512:]))
        with self.assertRaisesRegex(RuntimeError, "scalar n does not fit in int32_t"):
            runner(x, y, torch.empty(2**23, device=DEVICE))  # n = 2**31

    def test_aoti_numeric_constant_for_unannotated_param(self):
        # A FlyDSL Numeric value for an unannotated parameter is a constant to PT2
        # (guarded on, baked into the generated code) but a runtime argument of the
        # exported C ABI: refused, with a hint.
        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                add_scalar_w(x, out, x.numel(), fx.Float32(2.0))
                return out

        with self.assertRaisesRegex(
            Exception, r"FlyDSL passes \['k'\] at run time: annotate them"
        ):
            self._package(M(), (_rand(1000),))

    def test_export_tools_missing(self):
        # export_to_c shells out to a linker and objcopy; say which one is missing
        # instead of FlyDSL's bare FileNotFoundError.
        from torch._inductor.codegen.flydsl import user_kernel_aot

        with mock.patch.object(user_kernel_aot.shutil, "which", return_value=None):
            with self.assertRaisesRegex(
                RuntimeError, r"needs a relocatable linker \(ld.lld or ld\)"
            ):
                user_kernel_aot._check_export_tools()

    def test_aoti_misaligned_input_guard(self):
        # Exported with aligned inputs, the kernel assumes 16-byte alignment and
        # the launch checks it.
        base, base2 = _rand(1001), _rand(1001)
        pkg = self._package(_AddModule(), (base[:1000], base2[:1000]))
        runner = torch._inductor.aoti_load_package(pkg)
        x, y = base[1:], base2[1:]
        if inductor_config.is_fbcode():
            # fbcode AOTI does not copy misaligned inputs, so the guard fires.
            with _captured_stderr_fd() as err, self.assertRaises(RuntimeError) as cm:
                runner(x, y)
            # The C++ exception is in the error (or only on stderr, for runners that
            # do not forward it).
            self.assertIn(
                "argument X is not 16-byte aligned", str(cm.exception) + err[0]
            )
        else:
            self.assertEqual(runner(x, y), x + y)

    def test_aoti_unaligned_view(self):
        # A view the export cannot prove 16-byte aligned is compiled for its
        # element alignment instead (no check).
        class M(torch.nn.Module):
            def forward(self, base):
                x = base[:, 1:]  # data_ptr 4 bytes past a 16-byte boundary
                out = torch.empty(x.shape, device=x.device)
                add_one_2d_w(x, out)
                return out

        base = _rand(4, 9)
        runner = torch._inductor.aoti_load_package(self._package(M(), (base,)))
        self.assertEqual(runner(base), base[:, 1:] + 1)

    def test_aoti_two_outputs_and_default_scale(self):
        # The host function takes `scale: fx.Float32 = 1.0` after the stream: a
        # defaulted runtime scalar must still be passed.
        class M(torch.nn.Module):
            def forward(self, x, y):
                s1, d1 = torch.empty_like(x), torch.empty_like(x)
                add_sub_w(x, y, s1, d1, x.numel())
                s2, d2 = torch.empty_like(x), torch.empty_like(x)
                add_sub_w(x, y, s2, d2, x.numel(), scale=0.25)
                return s1, d1, s2, d2

        x, y = _rand(1000), _rand(1000)
        runner = torch._inductor.aoti_load_package(self._package(M(), (x, y)))
        s1, d1, s2, d2 = runner(x, y)
        self.assertEqual((s1, d1), (x + y, x - y))
        self.assertEqual((s2, d2), ((x + y) * 0.25, (x - y) * 0.25))

    def test_aoti_inplace_kernel_and_view(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x * 1
                scale_inplace_w(y, y.numel())
                v = y[4:]
                scale_inplace_w(v, v.numel())
                return y

        x = _rand(1000)
        expected = 2 * x
        expected[4:] *= 2
        runner = torch._inductor.aoti_load_package(self._package(M(), (x,)))
        self.assertEqual(runner(x), expected)

    def test_aoti_same_name_kernels(self):
        # Two host functions named binary_kernel: their exported symbols must not
        # collide in the model .so.
        class M(torch.nn.Module):
            def forward(self, x, y):
                o1, o2 = torch.empty_like(x), torch.empty_like(x)
                same_name_add_w(x, y, o1, x.numel())
                same_name_sub_w(x, y, o2, x.numel())
                return o1, o2

        x, y = _rand(1000), _rand(1000)
        runner = torch._inductor.aoti_load_package(self._package(M(), (x, y)))
        o1, o2 = runner(x, y)
        self.assertEqual(o1, x + y)
        self.assertEqual(o2, x - y)

    def test_aoti_kernels_differing_in_closure_values(self):
        # Two host functions from one factory get the same prefix (it cannot see a
        # captured list); each still gets its own export, under a unique symbol.
        ka = wrap_flydsl(make_affine_kernel([2.0, 0.0]), mutates_args=["Out"])
        kb = wrap_flydsl(make_affine_kernel([5.0, 1.0]), mutates_args=["Out"])

        class M(torch.nn.Module):
            def forward(self, x):
                a, b = torch.empty_like(x), torch.empty_like(x)
                ka(x, a, x.numel())
                kb(x, b, x.numel())
                return a, b

        x = torch.arange(1000, device=DEVICE, dtype=torch.float32)
        a, b = torch._inductor.aoti_load_package(self._package(M(), (x,)))(x)
        self.assertEqual(a, 2 * x)
        self.assertEqual(b, 5 * x + 1)

    @parametrize("strict", [False, True])
    def test_aoti_program_exported_before_reset(self, strict):
        # A program exported before torch._dynamo.reset() and another strict export
        # must still compile its own kernel (the Triton side table is reset there).
        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(_AddModule(), (x, y), strict=strict)
        torch._dynamo.reset()
        torch.export.export(_ScaleModule(), (x,), strict=True)
        runner = torch._inductor.aoti_load_package(
            torch._inductor.aoti_compile_and_package(ep)
        )
        self.assertEqual(runner(x, y), x + y)

    def test_aoti_call_sites_share_one_export(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                a, b = torch.empty_like(x), torch.empty_like(x)
                add_w(x, y, a, x.numel())
                add_w(a, y, b, x.numel())
                return b

        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(M(), (x, y))
        self.assertEqual(code.count("#define FLYDSL_DECLARED_"), 1)
        calls = re.findall(r"\bcall_flydsl_add_kernel_\w+\(", _strip_comments(code))
        self.assertEqual(len(calls), 3, calls)  # the definition and two launches
        self.assertEqual(torch._inductor.aoti_load_package(pkg)(x, y), x + 2 * y)

    def test_aoti_user_stream_region_uses_region_stream(self):
        # In a user stream region every launch uses the region's stream, which its
        # aten fallbacks run on (as in torch.compile's wrapper), so the kernel stays
        # ordered with the aten mm and the Triton kernels around it. Outside the
        # region, the model's stream.
        class M(torch.nn.Module):
            def forward(self, x, w):
                s = torch.cuda.Stream()
                e = torch.cuda.Event()
                with torch.cuda.stream(s):
                    a = torch.mm(x, w)
                    b = torch.empty_like(a)
                    add_one_2d_w(a, b)
                    c = torch.empty_like(a)
                    add_one_2d_w(b * 3, c)
                    z = c * 2
                    e.record()
                torch.cuda.current_stream().wait_event(e)
                out = torch.empty_like(z)
                add_one_2d_w(z, out)
                return out

        # add_one_2d_kernel runs one block per row, with a thread per column.
        x, w = _rand(4096, 1024), _rand(1024, 1024)
        pkg, code = self._package_and_code(M(), (x, w), strict=True)
        code = _strip_comments(code)
        # The region's stream, named as the region's entry names it (not by a local
        # variable, which a later block could not see).
        (aux,) = set(
            re.findall(
                r"_aoti_current_stream_guard = std::make_unique<[\w:]+>\((.*), "
                r"this->device_idx_\);",
                code,
            )
        )
        calls = re.findall(r"\bcall_flydsl_add_one_2d_kernel\w*\((.*)\);", code)
        streams = [
            "aux" if call.endswith(f", {aux}") else call.rsplit(", ", 1)[-1]
            for call in calls
        ]
        self.assertEqual(streams, ["aux", "aux", "stream"])
        # A Triton launch ends with the device index, stream, kernels and cubin dir.
        calls = re.findall(r"\bcall_triton_\w+\((.*)\);", code)
        self.assertGreaterEqual(len(calls), 2)
        tail = f", this->device_idx_, {aux}, kernels, this->cubin_dir_"
        self.assertTrue(all(call.endswith(tail) for call in calls), calls)
        run = torch._inductor.aoti_load_package(pkg)
        expected = ((x @ w + 1) * 3 + 1) * 2 + 1
        self.assertEqual(run(x, w), expected, atol=1e-2, rtol=1e-3)

    def test_aoti_exports_not_reused_from_disk(self):
        # With the in-process export cache empty (as in a new process) the kernel is
        # exported again, into a fresh directory: nothing on disk is reused.
        from torch._inductor.codegen.flydsl import user_kernel_aot

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(
            user_kernel_aot,
            "export_flydsl_kernel",
            wraps=user_kernel_aot.export_flydsl_kernel,
        ) as export:
            for _ in range(2):
                user_kernel_aot._exports.clear()
                self._package(_AddModule(), (x, y))
        dirs = [c.kwargs["out_dir"] for c in export.call_args_list]
        self.assertEqual(len(dirs), 2)
        self.assertNotEqual(dirs[0], dirs[1])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPUs")
    def test_aoti_two_devices(self):
        # The exported module is loaded per device (__module_load(-1) under
        # AOTInductor's device guard), so one package runs on either GPU, in one
        # process, in any order.
        x, y = _rand(1000), _rand(1000)
        pkg = self._package(_AddModule(), (x, y))
        run1 = torch._inductor.aoti_load_package(pkg, device_index=1)
        run0 = torch._inductor.aoti_load_package(pkg, device_index=0)
        x1, y1 = x.to("cuda:1"), y.to("cuda:1")
        self.assertEqual(run1(x1, y1), x1 + y1)
        self.assertEqual(run0(x, y), x + y)
        self.assertEqual(run1(2 * x1, y1), 2 * x1 + y1)

    def test_flydsl_kernel_jit_cpp_wrapper(self):
        # torch.compile with the C++ wrapper links the exported objects into the JIT
        # module the same way.
        x, y = _rand(1000), _rand(1000)
        with inductor_config.patch(cpp_wrapper=True):
            res, (code,) = run_and_get_code(
                torch.compile(_AddModule(), fullgraph=True), x, y
            )
        self.assertEqual(res, x + y)
        FileCheck().check("call_flydsl_add_kernel_").run(code)

    def test_aoti_export_for_another_arch_refused(self):
        # The export checks the target recorded in the object's metadata: FlyDSL
        # takes it from the environment, which must not leak in.
        real_compile = flydsl_user_kernel.compile_host_fn

        def compile_for_gfx942(fn, args, arch):
            return real_compile(fn, args, "gfx942")

        with mock.patch.object(
            flydsl_user_kernel, "compile_host_fn", compile_for_gfx942
        ):
            with self.assertRaisesRegex(
                Exception,
                "FlyDSLExportError: .* compiled for 'gfx942', not the target 'gfx9",
            ):
                self._package(_AddModule(), (_rand(1000), _rand(1000)))

    def test_aoti_module_for_another_arch_fails_cleanly(self):
        # A package exported for another GPU fails at its first launch with an error
        # naming the kernel and the HIP error, and leaves no HIP error pending.
        from torch._inductor.codegen.flydsl import user_kernel_aot

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(user_kernel_aot, "target_arch", return_value="gfx942"):
            pkg = self._package(_AddModule(), (x, y))
        runner = torch._inductor.aoti_load_package(pkg)
        with _captured_stderr_fd() as err, self.assertRaises(RuntimeError) as cm:
            runner(x, y)
        self.assertRegex(
            str(cm.exception) + err[0],
            r"FlyDSL kernel flydsl_add_kernel_gfx942_\w+: \w+__module_load on the "
            r"current device failed: status 209 \(hipErrorNoBinaryForGpu",
        )
        # The next HIP call is not hit by the failed load.
        self.assertEqual(x + y, torch.add(x, y))
        torch.cuda.synchronize()

    def test_aoti_reexport_after_global_rebind(self):
        # A plain-constant module global the kernel reads is part of its prefix:
        # once it is rebound, a new package exports the kernel again (under another
        # prefix) instead of linking the stale export, and the first package keeps
        # its own kernel. Other globals: test_aoti_reexport_after_rebind_outside_
        # prefix.
        bias_w = wrap_flydsl(bias_kernel, mutates_args=["Out"])

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                bias_w(x, out, x.numel())
                return out

        x = _rand(1000)
        pkg1, code1 = self._package_and_code(M(), (x,))
        run1 = torch._inductor.aoti_load_package(pkg1)
        self.assertEqual(run1(x), x + 2.0)
        with mock.patch.dict(globals(), {"_AOTI_BIAS": 5.0}):
            torch._dynamo.reset()
            pkg2, code2 = self._package_and_code(M(), (x,))
        run2 = torch._inductor.aoti_load_package(pkg2)
        self.assertEqual(run2(x), x + 5.0)
        self.assertEqual(run1(x), x + 2.0)
        prefixes = [
            re.findall(r"\bcall_(flydsl_bias_kernel_gfx\w+)\(", c)[0]
            for c in (code1, code2)
        ]
        self.assertNotEqual(prefixes[0], prefixes[1])

    @parametrize("case", ["tuple", "helper_global"])
    def test_aoti_reexport_after_rebind_outside_prefix(self, case):
        # The prefix records only the plain constants a kernel reads itself, not a
        # tuple global or the globals of a Python helper it calls. The export cache
        # holds the values of all of them, as the eager launch key does: after a
        # rebind, a new package and the JIT C++ wrapper export the kernel again
        # (under the same prefix) and agree with eager; the first package keeps its
        # own kernel.
        kernel, name, value = {
            "tuple": (tuple_scale_kernel, "_AOTI_SCALE", (3.0,)),
            "helper_global": (helper_scale_kernel, "_AOTI_FACTOR", 3.0),
        }[case]
        w = wrap_flydsl(kernel, mutates_args=["Out"])

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                w(x, out, x.numel())
                return out

        x = _rand(1000)
        run1 = torch._inductor.aoti_load_package(self._package(M(), (x,)))
        self.assertEqual(run1(x), x * 2.0)
        with mock.patch.dict(globals(), {name: value}):
            self.assertEqual(M()(x), x * 3.0)
            torch._dynamo.reset()
            run2 = torch._inductor.aoti_load_package(self._package(M(), (x,)))
            self.assertEqual(run2(x), x * 3.0)
            with inductor_config.patch(cpp_wrapper=True):
                torch._dynamo.reset()
                self.assertEqual(torch.compile(M())(x), x * 3.0)
        self.assertEqual(run1(x), x * 2.0)

    def test_aoti_int_constant_out_of_range_refused(self):
        # A Python int constant for an integer scalar is checked against the
        # scalar's C type when the C++ is generated, as eager refuses a value FlyDSL
        # would wrap: Int64 values outside int64, and an Int32 value beyond 64 bits
        # (which no C++ literal holds). The int64 extremes reach the kernel.
        i64_w = wrap_flydsl(add_i64_kernel, mutates_args=["Out"])

        def i64_model(*ks):
            class M(torch.nn.Module):
                def forward(self, x):
                    outs = []
                    for k in ks:
                        out = torch.empty_like(x)
                        i64_w(x, out, x.numel(), k)
                        outs.append(out)
                    return torch.stack(outs)

            return M()

        class Int32Model(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                add_w(x, x, out, 2**64 + 5)
                return out

        x = torch.zeros(16, dtype=torch.int64, device=DEVICE)
        for model, arg, message in (
            (i64_model(2**63 + 5), x, "k=9223372036854775813 does not fit"),
            (i64_model(-(2**63) - 1), x, "k=-9223372036854775809 does not fit"),
            (Int32Model(), _rand(16), "n=18446744073709551621 does not fit"),
        ):
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    model(arg)
                with self.assertRaisesRegex(Exception, message):
                    self._package(model, (arg,))
                torch._dynamo.reset()
                with (
                    inductor_config.patch(cpp_wrapper=True),
                    self.assertRaisesRegex(Exception, message),
                ):
                    torch.compile(model)(arg)
        ks = (2**63 - 1, -(2**63))
        runner = torch._inductor.aoti_load_package(self._package(i64_model(*ks), (x,)))
        expected = torch.tensor(ks, device=DEVICE)[:, None].expand(2, 16)
        self.assertEqual(runner(x), expected)

    def test_aoti_uint64_scalar_above_int64_max(self):
        # A Uint64 scalar reaches the kernel as a uint64_t: values from 2**63 up
        # pass, constants checked when the code is generated (as ULL literals), a
        # symbolic value when the launch runs.
        u64_hi_w = wrap_flydsl(u64_hi_kernel, mutates_args=["Out"])
        ks = (2**63, 0x9E3779B97F4A7C15, 2**64 - 1)

        def model(*constants):
            class M(torch.nn.Module):
                def forward(self, x):
                    outs = []
                    for k in (*constants, x.shape[0] * 2**33):
                        out = torch.zeros(1, dtype=torch.int32, device=x.device)
                        u64_hi_w(out, k)
                        outs.append(out)
                    return torch.cat(outs)

            return M()

        def high_bits(*values):  # as the kernel's Int32
            return torch.tensor(
                [((k >> 32) + 2**31) % 2**32 - 2**31 for k in values],
                dtype=torch.int32,
                device=DEVICE,
            )

        x = _rand(3)
        expected = high_bits(*ks, 3 * 2**33)
        self.assertEqual(model(*ks)(x), expected)  # eager
        dim = torch.export.Dim("n", min=2, max=64)
        pkg, code = self._package_and_code(
            model(*ks), (x,), dynamic_shapes={"x": {0: dim}}
        )
        runner = torch._inductor.aoti_load_package(pkg)
        self.assertEqual(runner(x), expected)
        self.assertEqual(runner(_rand(5)), high_bits(*ks, 5 * 2**33))
        self.assertIn("(buf0.data_ptr(), 9223372036854775808ULL, stream)", code)
        self.assertRegex(
            code,
            r"AOTI_RUNTIME_CHECK\(\(.+\) >= 0, \"FlyDSL kernel flydsl_u64_hi_kernel_"
            r"\w+: scalar s does not fit in uint64_t\"\);",
        )
        with inductor_config.patch(cpp_wrapper=True):
            torch._dynamo.reset()
            self.assertEqual(torch.compile(model(*ks))(x), expected)
        # Out of range: refused when the code is generated.
        for k in (-1, 2**64):
            with self.assertRaisesRegex(
                Exception, rf"scalar s={k} does not fit in uint64_t"
            ):
                self._package(model(k), (x,))

    def test_aoti_cpp_keyword_param_names(self):
        # The generated launcher names its parameters a0, a1, ...: host-function
        # parameters named like C++ keywords (new, default), a C macro (errno) or
        # the launcher's stream parameter compile; comments keep the names.
        blend_w = wrap_flydsl(kw_blend_kernel, mutates_args=["errno", "stream"])

        class M(torch.nn.Module):
            def forward(self, old, new):
                errno, stream = torch.empty_like(old), torch.empty_like(old)
                blend_w(old, new, errno, stream, old.numel(), 0.5)
                return errno, stream

        old, new = _rand(1000), _rand(1000)
        expected = (old + new * 0.5, old - new)
        self.assertEqual(M()(old, new), expected)
        pkg, code = self._package_and_code(M(), (old, new))
        self.assertIn("void* flydsl_arg1_data = a1;  // new", code)
        self.assertIn("const float flydsl_arg5 = static_cast<float>(a5);", code)
        self.assertEqual(torch._inductor.aoti_load_package(pkg)(old, new), expected)
        with inductor_config.patch(cpp_wrapper=True):
            torch._dynamo.reset()
            self.assertEqual(torch.compile(M())(old, new), expected)

    def test_aoti_dual_wrapper_refused(self):
        # AOTInductor without compile-time autotuning generates dual-wrapper code,
        # which FlyDSL kernels do not support: the refusal says how to get out.
        with inductor_config.patch({"triton.autotune_at_compile_time": False}):
            with self.assertRaisesRegex(
                Exception,
                r"User-defined FlyDSL kernels are not supported in dual wrapper mode "
                r"\(AOTInductor with torch\._inductor\.config\.triton\."
                r"autotune_at_compile_time=False\); set torch\._inductor\.config\."
                r"triton\.autotune_at_compile_time=True, or unset "
                r"TORCHINDUCTOR_AUTOTUNE_AT_COMPILE_TIME",
            ):
                self._package(_AddModule(), (_rand(1000), _rand(1000)))

    def test_aoti_2d_view_dynamic_large_blocks(self):
        # test_aoti_2d_view_dynamic's model at shapes inside its declared range
        # whose transposed view launches blocks of 300 and 512 threads: the block
        # derives from a dynamic size, which the export compiles for up to 1024.
        class M(torch.nn.Module):
            def forward(self, x):
                xt = x.t()
                out = torch.empty(xt.shape, device=x.device)
                add_one_2d_w(xt, out)
                return out

        a, b = (
            torch.export.Dim("a", min=2, max=512),
            torch.export.Dim("b", min=2, max=512),
        )
        runner = torch._inductor.aoti_load_package(
            self._package(M(), (_rand(8, 4),), dynamic_shapes={"x": {0: a, 1: b}})
        )
        for shape in ((300, 4), (512, 3), (8, 4)):
            x = _rand(*shape)
            self.assertEqual(runner(x), x.t() + 1)

    def test_aoti_dynamic_block_over_256(self):
        # Without a known block size the ROCDL lowering would cap the kernel at 256
        # threads per block; the export gives it HIP's default, 1024.
        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                add_one_2d_w(x, out)
                return out

        dim = torch.export.Dim("n", min=2, max=1024)
        runner = torch._inductor.aoti_load_package(
            self._package(M(), (_rand(32, 64),), dynamic_shapes={"x": {1: dim}})
        )
        for n in (64, 256, 257, 1024):
            x = _rand(32, n)
            self.assertEqual(runner(x), x + 1)

    def test_flydsl_kernel_jit_cpp_wrapper_dynamic_block(self):
        # The same for the JIT C++ wrapper with dynamic shapes: one dynamic kernel
        # serves blocks of 64 to 1024 threads.
        def fn(x):
            out = torch.empty_like(x)
            add_one_2d_w(x, out)
            return out

        counter = CompileCounterWithBackend("inductor")
        with inductor_config.patch(cpp_wrapper=True):
            compiled = torch.compile(fn, dynamic=True, backend=counter)
            for shape in ((32, 64), (8, 300), (4, 1024)):
                x = _rand(*shape)
                self.assertEqual(compiled(x), x + 1)
        self.assertEqual(counter.frame_count, 1)

    def test_aoti_flat_work_group_size_only_where_unset(self):
        # Only exports with dynamic layouts set a maximum flat work-group size, and
        # only on kernels without a known block size or one of their own.
        import flydsl.compiler.backends.rocm as rocm

        from torch._inductor.codegen.flydsl import user_kernel_aot as fly_aot
        from torch._inductor.codegen.flydsl.user_kernel_aot import (
            StreamParamSpec,
            TensorParamSpec,
        )

        def specs(dynamic):
            n = None if dynamic else 64
            return (
                *(
                    TensorParamSpec(t, torch.float32, (8, n), (n, 1), 16)
                    for t in ("X", "Out")
                ),
                StreamParamSpec("stream"),
            )

        changed = []
        real = fly_aot._set_default_flat_work_group_size

        def spy(module, max_size):
            changed.append(real(module, max_size))
            return changed[-1]

        hook = rocm.RocmBackend.__dict__["lower_compile_hints"]
        arch = flydsl_user_kernel.device_arch()
        cases = (
            (add_one_2d_kernel, False),
            (add_one_2d_kernel, True),
            (add_one_2d_kbs_kernel, True),
            (add_one_2d_fwg256_kernel, True),
        )
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(fly_aot, "_set_default_flat_work_group_size", spy):
                for i, (fn, dynamic) in enumerate(cases):
                    prefix = f"flydsl_fwg{i}"
                    fly_aot.export_flydsl_kernel(
                        fn, specs(dynamic), out_dir=d, prefix=prefix, arch=arch
                    )
        self.assertEqual(len(changed), 3)  # not called for the static layouts
        self.assertEqual(len(changed[0]), 1)
        self.assertTrue(changed[0][0].startswith("_add_one_2d_k"), changed)
        self.assertEqual(changed[1:], [[], []])
        self.assertIs(rocm.RocmBackend.__dict__["lower_compile_hints"], hook)

    def test_aoti_dynamic_block_over_kernel_limit_fails_with_hint(self):
        # A kernel's own maximum flat work-group size stays: a larger launch block
        # fails, with a hint naming the arguments whose layout is dynamic. The
        # failure is not sticky.
        w = wrap_flydsl(add_one_2d_fwg256_kernel, mutates_args=["Out"])

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                w(x, out)
                return out

        dim = torch.export.Dim("n", min=2, max=1024)
        runner = torch._inductor.aoti_load_package(
            self._package(M(), (_rand(32, 64),), dynamic_shapes={"x": {1: dim}})
        )
        x = _rand(32, 256)
        self.assertEqual(runner(x), x + 1)
        hint = (
            r"launch failed: status 719 \(hipErrorLaunchFailure: .*\); this export has "
            r"dynamic sizes or strides for X, Out: a launch block derived from them "
            r"must fit the kernel's maximum flat work-group size"
        )
        with _captured_stderr_fd(), self.assertRaisesRegex(RuntimeError, hint):
            runner(_rand(32, 300))
        x = _rand(32, 128)
        self.assertEqual(runner(x), x + 1)


###############################################################################
# Helpers of FlyDSLAOTUnitTests (CPU: no flydsl needed)


def _elf_object(symbols, data=b""):
    """A minimal ELF64 little-endian relocatable object, as an export_to_c object
    without code: section 1 is a .data section holding ``data``; ``symbols`` are
    (name, binding, section, value, size) symbol table entries (binding 0 local,
    1 global, 2 weak; section 0 undefined)."""
    strtab = b"\0"
    entries = [bytes(24)]
    for name, binding, section, value, size in symbols:
        info = binding << 4 | 1  # STT_OBJECT
        entries.append(
            struct.pack("<IBBHQQ", len(strtab), info, 0, section, value, size)
        )
        strtab += name.encode() + b"\0"
    shstrtab = b"\0.data\0.symtab\0.strtab\0.shstrtab\0"
    parts = [data, b"".join(entries), strtab, shstrtab]
    offsets, pos = [], 64
    for part in parts:
        offsets.append(pos)
        pos += len(part)
    shoff = (pos + 7) // 8 * 8
    sections = [
        bytes(64),
        struct.pack("<IIQQQQIIQQ", 1, 1, 3, 0, offsets[0], len(data), 0, 0, 1, 0),
        struct.pack("<IIQQQQIIQQ", 7, 2, 0, 0, offsets[1], len(parts[1]), 3, 1, 8, 24),
        struct.pack("<IIQQQQIIQQ", 15, 3, 0, 0, offsets[2], len(strtab), 0, 0, 1, 0),
        struct.pack("<IIQQQQIIQQ", 23, 3, 0, 0, offsets[3], len(shstrtab), 0, 0, 1, 0),
    ]
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01",
        1,  # ET_REL
        62,  # EM_X86_64
        1,
        0,
        0,
        shoff,
        0,
        64,
        0,
        0,
        64,
        len(sections),
        4,
    )
    body = header + b"".join(parts)
    return body + bytes(shoff - len(body)) + b"".join(sections)


def _fake_flydsl_type(name):
    # What fuka reads of a FlyDSL Numeric annotation: its class name and module.
    return type(name, (), {"__module__": "flydsl.expr.numeric"})


FakeInt8, FakeInt32, FakeInt64, FakeUint64, FakeBoolean, FakeFloat32, FakeFloat16 = (
    _fake_flydsl_type(n)
    for n in ("Int8", "Int32", "Int64", "Uint64", "Boolean", "Float32", "Float16")
)


class _FakeJitFunction:
    """Stands in for a @flyc.jit JitFunction (fuka recognizes one by its class name
    and module and reads the user's function from _original_func; the export key
    reads its compile hints)."""

    def __init__(self, func):
        self.func = func
        self._original_func = func
        self.compile_hints = {}

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _effective_compile_hints(self):
        return dict(self.compile_hints)


_FakeJitFunction.__name__ = _FakeJitFunction.__qualname__ = "JitFunction"
_FakeJitFunction.__module__ = "flydsl.compiler.jit_function"


@_FakeJitFunction
def _aot_add_kernel(X, Y, Out, n, stream):
    Out.copy_(X + Y)


@_FakeJitFunction
def _aot_scale_kernel(X, Out, scale, stream, alpha=1.0, bias=None):
    Out.copy_(X * scale * alpha)


AOT_BLOCK = 128


@_FakeJitFunction
def _aot_inner_kernel(X, Out):
    Out.copy_(X)


@_FakeJitFunction
def _aot_outer_kernel(X, Out, stream):
    _aot_inner_kernel(X, Out)
    Out.mul_(AOT_BLOCK)


AOT_SIGN = 1.0


@_FakeJitFunction
def _aot_signed_kernel(X, Out, stream):
    Out.copy_(X * AOT_SIGN)


# A tuple global: not part of the kernel prefix, which records plain constants only.
AOT_TILE = (2.0,)


@_FakeJitFunction
def _aot_tile_kernel(X, Out, stream):
    Out.copy_(X * AOT_TILE[0])


@dataclasses.dataclass(frozen=True)
class _AotConfig:
    block: int


def _synthetic_export(specs, prefix, arch, *, bits=32):
    """The metadata JSON and header export_to_c writes for ``specs`` (flydsl 0.4,
    schema 2): a slot per tensor data pointer, per tensor layout with dynamic dims
    (``bits``-bit sizes and strides), per runtime scalar and for the stream, and the
    typed ``<prefix>_call`` taking them in order."""
    abi, params = [], []
    for s in specs:
        if isinstance(s, fuka.TensorParamSpec):
            abi.append({"kind": "tensor_data", "arg_name": s.name, "ctype": "pointer"})
            params.append(f"void *{s.name}_data")
            shape = [i for i, v in enumerate(s.sizes) if v is None]
            stride = [i for i, v in enumerate(s.strides) if v is None]
            if shape or stride:
                slot = {
                    "kind": "tensor_layout",
                    "arg_name": s.name,
                    "ctype": "bytes",
                    "shape_dims": shape,
                    "stride_dims": stride,
                    "shape_bits": bits,
                }
                if stride:
                    slot["stride_bits"] = bits
                abi.append(slot)
                params.append(f"const {prefix}_{s.name}_layout_t *{s.name}_layout")
        elif isinstance(s, fuka.ScalarParamSpec):
            ctype = fuka.scalar_ctype(s.annotation).removesuffix("_t")
            abi.append({"kind": "scalar", "arg_name": s.name, "ctype": ctype})
            params.append(f"{fuka._slot_ctype(ctype)} {s.name}")
        elif isinstance(s, fuka.StreamParamSpec):
            abi.append({"kind": "stream", "arg_name": s.name, "ctype": "pointer"})
            params.append(f"void *{s.name}")
    metadata = {
        "schema_version": 2,
        "symbol": prefix,
        "metadata_symbol": f"{prefix}__metadata",
        **{
            f"module_{n}_symbol": f"{prefix}__module_{n}"
            for n in ("init", "load", "unload")
        },
        "arch": arch,
        "return_abi": {"ctype": "int32", "kind": "status", "success": 0},
        "abi": abi,
        "linker_flags": ["-lamdhip64", "-pthread", "-ldl"],
        "flydsl_version": "0.4.0",
    }
    header = (
        "#pragma once\n"
        f"int32_t {prefix}(void **args);\n"
        f"static inline int32_t {prefix}_call({', '.join(params)}) {{\n"
        "  return 0;\n"
        "}\n"
    )
    return metadata, header


def _write_export(out_dir, prefix, metadata, header, extra_symbols=()):
    # The object defines the five export symbols plus ``extra_symbols``; a local and
    # an undefined symbol show which ones the symbol check leaves alone.
    meta = json.dumps(metadata).encode() + b"\0"
    symbols = [
        (prefix, 1, 1, 0, 0),
        (f"{prefix}__metadata", 1, 1, 0, len(meta)),
        *((f"{prefix}__module_{n}", 1, 1, 0, 0) for n in ("init", "load", "unload")),
        ("flydsl_aot_runtime_state", 0, 1, 0, 0),
        ("hipModuleLaunchKernel", 1, 0, 0, 0),
        *((name, 1, 1, 0, 0) for name in extra_symbols),
    ]
    Path(out_dir, f"{prefix}.o").write_bytes(_elf_object(symbols, meta))
    Path(out_dir, f"{prefix}.h").write_text(header)


class _FakeExports:
    """What _fake_flydsl_export records and lets a test change between exports."""

    def __init__(self) -> None:
        self.prefixes: list[str] = []  # what export_to_c ran for
        self.arch_override: str | None = None  # compile for this arch instead
        self.extra_symbols: list[str] = []  # stray globals of the next objects
        self.env: tuple[object, ...] = ()  # the FlyDSL environment variables' values


@contextlib.contextmanager
def _fake_flydsl_export():
    """Exports without flydsl: _compile_args passes the specs through (by name), a
    fake flydsl_user_kernel.compile_host_fn returns an object whose export_to_c
    writes the synthetic export of those specs, and the export tools check and the
    FlyDSL version are faked; fuka's export cache starts empty. The export key reads
    the globals the kernel itself loads (flydsl_user_kernel._global_refs, which also
    follows its callees, needs flydsl), the environment from ``fake.env`` and no
    compile hints of the thread."""
    fake = _FakeExports()

    def global_refs(fn):
        func = fn._original_func
        names = dict.fromkeys(flydsl_user_kernel._loaded_globals(func.__code__))
        return tuple((name, func.__globals__) for name in names)

    def compile_host_fn(fn, args, arch):
        specs = tuple(args.values())

        class Compiled:
            def export_to_c(self, out_dir, prefix, *, function_prefix):
                if function_prefix != prefix:
                    raise AssertionError((function_prefix, prefix))
                fake.prefixes.append(prefix)
                metadata, header = _synthetic_export(
                    specs, prefix, fake.arch_override or arch
                )
                _write_export(out_dir, prefix, metadata, header, fake.extra_symbols)

        return Compiled()

    with contextlib.ExitStack() as stack:
        for target, name, value in (
            (fuka, "_compile_args", lambda specs: {s.name: s for s in specs}),
            (fuka, "_check_export_tools", lambda: None),
            (fuka, "version_tag", lambda: "flydsl=0.4.0"),
            (fuka, "_exports", weakref.WeakKeyDictionary()),
            (flydsl_user_kernel, "compile_host_fn", compile_host_fn),
            (flydsl_user_kernel, "_global_refs", global_refs),
            (flydsl_user_kernel, "_globals_snapshots", weakref.WeakKeyDictionary()),
            (flydsl_user_kernel, "_key_readers", lambda: (lambda: fake.env, dict)),
        ):
            stack.enter_context(mock.patch.object(target, name, value))
        yield fake


@contextlib.contextmanager
def _flydsl_hidden():
    """Inside the block flydsl is not installed, as far as imports go: importing any
    of its modules raises ModuleNotFoundError, also those imported already (they are
    put back on exit). flydsl_user_kernel's cached readers are dropped, before and
    after."""

    def is_flydsl(name):
        return name == "flydsl" or name.startswith("flydsl.")

    loaded = {name: m for name, m in sys.modules.items() if is_flydsl(name)}
    flydsl_user_kernel._key_readers.cache_clear()
    hidden: Any = dict.fromkeys(["flydsl", *loaded])  # None stops an import
    sys.modules.update(hidden)
    try:
        yield
    finally:
        for name in [name for name in sys.modules if is_flydsl(name)]:
            del sys.modules[name]
        sys.modules.update(loaded)
        flydsl_user_kernel._key_readers.cache_clear()


class FlyDSLAOTUnitTests(TestCase):
    """codegen/flydsl/user_kernel_aot.py without a GPU or flydsl: ELF reading, symbol
    checks, the exported ABI, kernel prefixes, the version script and link flags,
    exports (through a fake compile) and their cache, and the generated C++."""

    hw_classification = HardwareClassification.GENERIC

    def _tmp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return d.name

    def _add_specs(self, n=None, n_type=FakeInt32, hint=1000):
        return (
            *(fuka.TensorParamSpec(t, torch.float32, (n,), (1,), 16) for t in "XY"),
            fuka.TensorParamSpec("Out", torch.float32, (n,), (1,), 16),
            fuka.ScalarParamSpec("n", n_type, hint),
            fuka.StreamParamSpec("stream"),
        )

    def test_runs_without_flydsl(self):
        # This class has no skip decorator, so it also runs where flydsl is not
        # installed (the caffe2/test/inductor target, OSS CI): every other test of
        # it passes with flydsl hidden, checked on hosts that have flydsl too.
        names = unittest.defaultTestLoader.getTestCaseNames(type(self))
        result = unittest.TestResult()
        with _flydsl_hidden():
            for name in names:
                if name != self._testMethodName:
                    type(self)(name).run(result)
        self.assertEqual(result.testsRun, len(names) - 1)
        failed = [
            f"{test.id().rsplit('.', 1)[-1]}: {trace.strip().splitlines()[-1]}"
            for test, trace in result.errors + result.failures
        ]
        self.assertFalse(failed, "\n".join(failed))

    def test_elf_reader(self):
        meta = b'{"abi": []}\0'
        path = os.path.join(self._tmp(), "k.o")
        symbols = [
            ("k", 1, 1, 0, 0),
            ("k__metadata", 1, 1, 0, len(meta)),
            ("k_local", 0, 1, 0, 0),
            ("hipLaunchKernel", 1, 0, 0, 0),
            ("k_weak", 2, 1, 0, 0),
        ]
        Path(path).write_bytes(_elf_object(symbols, meta))
        self.assertEqual(
            [(s.name, s.binding, s.section) for s in fuka.elf_symbols(path)],
            [("", 0, 0), *((n, b, s) for n, b, s, _, _ in symbols)],
        )
        self.assertEqual(fuka.elf_global_symbols(path), ["k", "k__metadata", "k_weak"])
        self.assertEqual(fuka.elf_symbol_data(path, "k__metadata"), meta)
        for name in ("hipLaunchKernel", "missing"):
            with self.assertRaisesRegex(
                fuka.FlyDSLExportError, f"does not define one data symbol {name}"
            ):
                fuka.elf_symbol_data(path, name)
        bad = os.path.join(self._tmp(), "bad.o")
        Path(bad).write_bytes(b"!<arch>\n" + bytes(64))
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError, "is not an ELF64 little-endian object"
        ):
            fuka.elf_global_symbols(bad)

    @unittest.skipUnless(
        sys.platform == "linux" and shutil.which("cc"),
        "needs a C compiler that writes ELF objects",
    )
    def test_elf_reader_compiled_object(self):
        d = self._tmp()
        src = os.path.join(d, "k.c")
        Path(src).write_text(
            "static int k_local;\n"
            "int k(void) { return k_local; }\n"
            'const char k__metadata[] = "{\\"arch\\": \\"gfx950\\"}";\n'
        )
        subprocess.check_call(["cc", "-c", "-fPIC", "-o", f"{d}/k.o", src])
        self.assertEqual(
            sorted(fuka.elf_global_symbols(f"{d}/k.o")), ["k", "k__metadata"]
        )
        self.assertEqual(fuka._read_metadata(f"{d}/k.o", "k"), {"arch": "gfx950"})

    def test_check_object_symbols(self):
        # An export defines exactly its five symbols (export_to_c localizes the rest,
        # its embedded runtime included); local and undefined symbols do not count.
        d = self._tmp()
        _write_export(d, "k", {}, "")
        fuka._check_object_symbols(f"{d}/k.o", "k")
        _write_export(d, "k", {}, "", extra_symbols=["k_weak"])
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError,
            r"defines global symbols outside k k__metadata k__module_init "
            r"k__module_load k__module_unload: \['k_weak'\]",
        ):
            fuka._check_object_symbols(f"{d}/k.o", "k")
        Path(f"{d}/k.o").write_bytes(
            _elf_object([("k", 1, 1, 0, 0), ("k__metadata", 1, 1, 0, 0)])
        )
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError,
            r"k.o does not define \['k__module_init', 'k__module_load', "
            r"'k__module_unload'\]",
        ):
            fuka._check_object_symbols(f"{d}/k.o", "k")

    def test_scalar_ctype(self):
        self.assertEqual(fuka.scalar_ctype(FakeInt32), "int32_t")
        self.assertEqual(fuka.scalar_ctype(FakeBoolean), "bool")
        self.assertEqual(fuka.scalar_ctype(FakeFloat32), "float")
        # Float16/BFloat16 scalars are exported as raw bits; other types as nothing.
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError, "Float16 is not supported by AOTInductor"
        ):
            fuka.scalar_ctype(FakeFloat16)
        for annotation in (int, type("Int32", (), {"__module__": "cutlass"})):
            with self.assertRaisesRegex(
                fuka.FlyDSLExportError, "is not supported by FlyDSL export_to_c"
            ):
                fuka.scalar_ctype(annotation)

    def test_parse_exported_abi(self):
        # The metadata and header of an export, checked against the specs and target.
        specs = (
            fuka.TensorParamSpec("X", torch.float32, (None, 4), (None, 1), 16),
            fuka.TensorParamSpec("Out", torch.float32, (8,), (1,), 16),
            fuka.ScalarParamSpec("n", FakeInt32, 7),
            fuka.ConstantParamSpec("block", 256),
            fuka.StreamParamSpec("stream"),
        )
        metadata, header = _synthetic_export(specs, "k", "gfx950")
        abi = fuka.parse_exported_abi(
            metadata, header, specs, prefix="k", arch="gfx950"
        )
        self.assertEqual(
            [(s.kind, s.name, s.cpp_type) for s in abi.slots],
            [
                ("tensor_data", "X", "void *"),
                ("tensor_layout", "X", "const k_X_layout_t *"),
                ("tensor_data", "Out", "void *"),
                ("scalar", "n", "int32_t"),
                ("stream", "stream", "void *"),
            ],
        )
        layout = abi.layout("X")
        if layout is None:
            self.fail("the ABI has no layout for X")
        self.assertEqual((layout.shape_dims, layout.stride_dims), ((0,), (0,)))
        self.assertEqual((layout.shape_bits, layout.stride_bits), (32, 32))
        self.assertIsNone(abi.layout("Out"))
        self.assertEqual(abi.linker_flags, ("-lamdhip64", "-pthread", "-ldl"))
        exported = fuka.FlyDSLExportedKernel("k", "k.h", "k.o", header, abi, specs)
        self.assertEqual(
            [(p.kind, p.name, p.launcher_ctype) for p in fuka.launch_params(exported)],
            [
                ("tensor", "X", "int64_t"),
                ("tensor", "Out", "int64_t"),
                ("scalar", "n", "int64_t"),
                ("stream", "stream", "int64_t"),
            ],
        )

        def parse(metadata=metadata, header=header, arch="gfx950"):
            return fuka.parse_exported_abi(
                metadata, header, specs, prefix="k", arch=arch
            )

        for change, message in (
            ({"schema_version": 1}, "metadata schema 1, expected 2"),
            ({"arch": "gfx942"}, "compiled for 'gfx942', not the target 'gfx950'"),
            ({"module_load_symbol": "x"}, "unexpected lifecycle symbol names"),
            ({"return_abi": {}}, "unexpected return ABI"),
            ({"linker_flags": "-ldl"}, "unexpected linker_flags"),
        ):
            with self.assertRaisesRegex(fuka.FlyDSLExportError, message):
                parse({**metadata, **change})
        # A FlyDSL Numeric constant for an unannotated parameter is a runtime
        # argument to FlyDSL: the mismatch names it.
        runtime_block = {
            **metadata,
            "abi": [
                *metadata["abi"][:-1],
                {"kind": "scalar", "arg_name": "block", "ctype": "int32"},
                metadata["abi"][-1],
            ],
        }
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError,
            r"FlyDSL passes \['block'\] at run time: annotate them with fx.Constexpr",
        ):
            parse(runtime_block)
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError, "k_call takes 'n' as 'int64_t', not int32"
        ):
            parse(header=header.replace("int32_t n", "int64_t n"))
        with self.assertRaisesRegex(fuka.FlyDSLExportError, "header lacks k_call"):
            parse(header="int32_t k(void **args);\n")

    def test_target_arch(self):
        with mock.patch.object(
            flydsl_user_kernel, "device_arch", return_value="gfx942"
        ) as device_arch:
            self.assertEqual(fuka.target_arch(torch.device("cuda", 1)), "gfx942")
        device_arch.assert_called_once_with(torch.device("cuda", 1))

    def test_prefix(self):
        def prefix(specs, fn=_aot_add_kernel, arch="gfx950"):
            return fuka.flydsl_kernel_prefix(fn, specs, arch)

        version = mock.patch.object(fuka, "version_tag", return_value="flydsl=0.4.0")
        version.start()
        self.addCleanup(version.stop)
        specs = self._add_specs()
        p = prefix(specs)
        self.assertRegex(p, r"^flydsl_aot_add_kernel_gfx950_[0-9a-f]{16}$")
        self.assertEqual(prefix(specs), p)
        # The scalar hint is not baked into the kernel; everything else is.
        self.assertEqual(prefix(self._add_specs(hint=7)), p)
        x16 = fuka.TensorParamSpec("X", torch.float16, (None,), (1,), 16)
        x4 = fuka.TensorParamSpec("X", torch.float32, (None,), (1,), 4)
        for other in (
            self._add_specs(2048),
            self._add_specs(n_type=FakeInt8),
            (x16, *specs[1:]),
            (x4, *specs[1:]),
        ):
            self.assertNotEqual(prefix(other), p)
        self.assertNotEqual(prefix(specs, arch="gfx942"), p)
        with mock.patch.object(fuka, "version_tag", return_value="flydsl=0.4.1"):
            self.assertNotEqual(prefix(specs), p)
        with mock.patch.object(fuka, "_FORMAT_VERSION", 2):
            self.assertNotEqual(prefix(specs), p)
        with self.assertRaisesRegex(ValueError, "arch must look like 'gfx950'"):
            prefix(specs, arch="sm_90a")

        # Constants count by value (no object addresses).
        def scale_specs(scale):
            return (
                fuka.TensorParamSpec("X", torch.float32, (8,), (1,), 16),
                fuka.TensorParamSpec("Out", torch.float32, (8,), (1,), 16),
                fuka.ConstantParamSpec("scale", scale),
                fuka.StreamParamSpec("stream"),
                fuka.ConstantParamSpec("alpha", 1.0),
                fuka.ConstantParamSpec("bias", None),
            )

        for a, b, c in (
            ([2, 5], [2, 5], [2, 6]),
            (_AotConfig(2), _AotConfig(2), _AotConfig(3)),
        ):
            self.assertEqual(
                prefix(scale_specs(a), _aot_scale_kernel),
                prefix(scale_specs(b), _aot_scale_kernel),
            )
            self.assertNotEqual(
                prefix(scale_specs(a), _aot_scale_kernel),
                prefix(scale_specs(c), _aot_scale_kernel),
            )
        self.assertNotIn("0x", fuka._stable_repr(_AotConfig(2)))

    def test_prefix_closure(self):
        # The host function's source, the FlyDSL functions it calls and the simple
        # constants they read, transitively.
        parts = fuka._dsl_function_closure(_aot_outer_kernel)
        self.assertEqual(
            [part.split("\n")[0] for part in parts],
            [
                f"{__name__}._aot_outer_kernel",
                "AOT_BLOCK=128",
                f"{__name__}._aot_inner_kernel",
            ],
        )
        self.assertIn("Out.mul_(AOT_BLOCK)", parts[0])
        # FlyDSL's own functions are covered by its version.
        with mock.patch.object(fuka, "_DSL_PACKAGE", __name__):
            parts = fuka._dsl_function_closure(_aot_outer_kernel)
        self.assertEqual(len(parts), 2)

    def test_link_inputs_and_flags(self):
        d = os.path.join(self._tmp(), "out dir")
        abi = fuka.FlyDSLExportedABI((), "gfx950", "0.4.0", ("-lamdhip64", "-ldl"))
        abi2 = dataclasses.replace(abi, linker_flags=("-ldl", "-pthread"))
        k1, k2 = (
            fuka.FlyDSLExportedKernel(f"flydsl_{n}", f"/x/{n}.h", f"/x/{n}.o", "", a)
            for n, a in (("b", abi), ("a", abi2))
        )
        # Every object once, in order (each embeds FlyDSL's AOT runtime).
        self.assertEqual(fuka.flydsl_link_inputs([k1, k2, k1]), ["/x/b.o", "/x/a.o"])
        flags = fuka.flydsl_link_flags(d, [k1, k2, k1])
        # The version script (its path a token of its own), then the metadata's
        # link flags once each.
        self.assertEqual(flags[1:], ["-lamdhip64", "-ldl", "-pthread"])
        tokens = shlex.split(flags[0])
        self.assertEqual(tokens[:3], ["-Xlinker", "--version-script", "-Xlinker"])
        path = tokens[3]
        self.assertEqual(os.path.dirname(path), d)
        self.assertRegex(os.path.basename(path), r"^flydsl_[0-9a-f]{16}\.ver$")
        self.assertEqual(
            Path(path).read_text(),
            "/* FlyDSL user kernels: export_to_c symbols stay in this library. */\n"
            "{\n  local:\n    flydsl_a;\n    flydsl_a__*;\n"
            "    flydsl_b;\n    flydsl_b__*;\n};\n",
        )
        # Written once per content (the prefixes are sorted; the metadata flags keep
        # the order of first appearance).
        mtime = os.stat(path).st_mtime_ns
        self.assertEqual(
            fuka.flydsl_link_flags(d, [k2, k1]),
            [flags[0], "-ldl", "-pthread", "-lamdhip64"],
        )
        self.assertEqual(os.stat(path).st_mtime_ns, mtime)
        self.assertNotEqual(fuka.flydsl_link_flags(d, [k1])[0], flags[0])
        self.assertEqual(len(os.listdir(d)), 2)
        self.assertEqual(fuka.flydsl_link_flags(d, []), [])
        bad = dataclasses.replace(k1, prefix="a-b")
        with self.assertRaisesRegex(ValueError, "'a-b' is not a C identifier"):
            fuka.flydsl_link_flags(d, [k1, bad])

    def _cpp_kernel(self, *specs):
        # The export of ``specs``, by default every launcher parameter kind: a tensor
        # with a 32-bit dynamic layout, int8, uint64, Boolean and Float32 scalars, the
        # stream (not last), a static tensor and a constant (no slot).
        specs = specs or (
            fuka.TensorParamSpec("X", torch.float32, (None, 4), (None, 1), 16),
            fuka.ScalarParamSpec("i8", FakeInt8, -3),
            fuka.ScalarParamSpec("u64", FakeUint64, 7),
            fuka.ScalarParamSpec("flag", FakeBoolean, True),
            fuka.ScalarParamSpec("f", FakeFloat32, 0.5),
            fuka.StreamParamSpec("stream"),
            fuka.TensorParamSpec("Out", torch.float32, (8,), (1,), 16),
            fuka.ConstantParamSpec("alpha", 2),
        )
        metadata, header = _synthetic_export(specs, "flydsl_k", "gfx950")
        abi = fuka.parse_exported_abi(
            metadata, header, specs, prefix="flydsl_k", arch="gfx950"
        )
        return fuka.FlyDSLExportedKernel(
            "flydsl_k", "/x/flydsl_k.h", "/x/flydsl_k.o", header, abi, specs
        )

    def _launch(self, exported, **overrides):
        kwargs: dict[str, Any] = {
            "tensor_args": {
                "X": ("x_ptr", ["x_size0"], ["x_stride0"]),
                "Out": ("out_ptr", [], []),
            },
            "scalar_args": {n: f"{n}_value" for n in ("i8", "u64", "flag", "f")},
            "stream_expr": "s",
            "check_align": {"X": 16},
            **overrides,
        }
        return fuka.emit_cpp_launch(exported, **kwargs)

    def test_emit_cpp(self):
        exported = self._cpp_kernel()
        decls = fuka.emit_cpp_declarations(exported)
        prelude = [
            *fuka._CPP_PRELUDE.splitlines(),
            *fuka._STATUS_HELPERS.splitlines(),
        ]
        self.assertEqual(decls[: len(prelude)], prelude)
        call = exported.header_text.splitlines()[2]
        self.assertTrue(call.startswith("static inline int32_t flydsl_k_call("), call)
        self.assertEqual(
            decls[len(prelude) :],
            [
                "// FlyDSL user kernel flydsl_k: export_to_c header flydsl_k.h",
                "#ifndef FLYDSL_DECLARED_flydsl_k",
                "#define FLYDSL_DECLARED_flydsl_k",
                "int32_t flydsl_k(void **args);",
                call,
                "  return 0;",
                "}",
                "static std::once_flag flydsl_k_init_once;",
                "static inline void flydsl_k_init_module_once() {",
                "  std::call_once(flydsl_k_init_once, [] {",
                "    const int32_t status = flydsl_k__module_init();",
                '    flydsl_user_kernel_check(status, "FlyDSL kernel flydsl_k: '
                'flydsl_k__module_init failed");',
                "  });",
                "}",
                "#endif  // FLYDSL_DECLARED_flydsl_k",
            ],
        )
        self.assertEqual(
            "\n".join(self._launch(exported)),
            """\
{
  // FlyDSL user kernel flydsl_k
  flydsl_k_init_module_once();
  flydsl_user_kernel_check(flydsl_k__module_load(-1), "FlyDSL kernel flydsl_k: flydsl_k__module_load on the current device failed");
  void* flydsl_arg0_data = x_ptr;  // X
  AOTI_RUNTIME_CHECK(reinterpret_cast<std::uintptr_t>(flydsl_arg0_data) % 16 == 0, "FlyDSL kernel flydsl_k: argument X is not 16-byte aligned");
  const int64_t flydsl_arg0_size0 = static_cast<int64_t>(x_size0);
  AOTI_RUNTIME_CHECK(flydsl_arg0_size0 >= 0 && flydsl_arg0_size0 <= 2147483647LL, "FlyDSL kernel flydsl_k: dynamic size 0 of argument X does not fit in int32");
  const int64_t flydsl_arg0_stride0 = static_cast<int64_t>(x_stride0);
  AOTI_RUNTIME_CHECK(flydsl_arg0_stride0 >= -2147483648LL && flydsl_arg0_stride0 <= 2147483647LL, "FlyDSL kernel flydsl_k: dynamic stride 0 of argument X does not fit in int32");
  const flydsl_k_X_layout_t flydsl_arg0_layout{{static_cast<int32_t>(flydsl_arg0_size0)}, {static_cast<int32_t>(flydsl_arg0_stride0)}};
  const int64_t flydsl_arg1_value = static_cast<int64_t>(i8_value);  // i8
  AOTI_RUNTIME_CHECK(flydsl_arg1_value >= -128LL && flydsl_arg1_value <= 127LL, "FlyDSL kernel flydsl_k: scalar i8 does not fit in int8_t");
  const int8_t flydsl_arg1 = static_cast<int8_t>(flydsl_arg1_value);
  const uint64_t flydsl_arg2 = static_cast<uint64_t>(u64_value);  // u64
  const bool flydsl_arg3 = static_cast<bool>((flag_value) != 0);  // flag
  const float flydsl_arg4 = static_cast<float>(f_value);  // f
  void* flydsl_arg6_data = out_ptr;  // Out
  flydsl_user_kernel_check(flydsl_k_call(flydsl_arg0_data, &flydsl_arg0_layout, flydsl_arg1, flydsl_arg2, flydsl_arg3, flydsl_arg4, static_cast<void*>(s), flydsl_arg6_data), "FlyDSL kernel flydsl_k: launch failed", "; this export has dynamic sizes or strides for X: a launch block derived from them must fit the kernel's maximum flat work-group size (1024 threads unless the kernel sets known_block_size or value_attrs={'rocdl.flat_work_group_size': '<min>,<max>'})");
}""",
        )

    def test_emit_cpp_errors(self):
        exported = self._cpp_kernel()
        for overrides, message in (
            ({"tensor_args": {"X": ("p", ["s"], ["t"])}}, r"tensor_args \['X'\] != "),
            (
                {"scalar_args": {}},
                r"scalar_args \[\] != scalars \['f', 'flag', 'i8', 'u64'\]",
            ),
            (
                {"check_align": {"i8": 16}},
                r"check_align names \['i8'\] are not tensors",
            ),
            ({"check_align": {"X": 12}}, r"check_align\[X\]=12 is not a power of two"),
            (
                {"tensor_args": {"X": ("p", [], ["t"]), "Out": ("q", [], [])}},
                "X needs 1 dynamic sizes and 1 dynamic strides, got 0 and 1",
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                self._launch(exported, **overrides)
        # A scalar C type the launcher cannot convert to (parse_exported_abi refuses
        # such an ABI; a hand-made one reaches the C++ emitter).
        slots = tuple(
            dataclasses.replace(s, ctype="int128") if s.name == "i8" else s
            for s in exported.abi.slots
        )
        odd = dataclasses.replace(
            exported, abi=dataclasses.replace(exported.abi, slots=slots)
        )
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError,
            "flydsl_k: unsupported scalar C type int128_t for i8",
        ):
            self._launch(odd)
        # A uint64 launcher parameter for a scalar the export takes as int32_t.
        with self.assertRaisesRegex(
            fuka.FlyDSLExportError,
            "flydsl_k: u64 is passed as a uint64_t, not as int32_t",
        ):
            fuka._cpp_scalar(
                exported,
                fuka.ExportedParam("scalar", "u64", "int32_t", uint64=True),
                "v",
                "e",
            )

    def test_call_wrapper(self):
        # The launcher's parameters are named a<i> (a FlyDSL parameter may be named
        # like a C++ keyword or macro, or like the stream) and a uint64_t scalar is
        # taken as a uint64_t (values above INT64_MAX); messages keep the host
        # function's parameter names.
        wrapper: Any = types.SimpleNamespace(prefix=IndentedBuffer())
        exported = self._cpp_kernel()
        fuka.FlyDSLCallWrapper("call_flydsl_k", exported, {"X": 16}).generate(wrapper)
        lines = wrapper.prefix.getvalue().splitlines()
        self.assertEqual(
            lines[0],
            "static inline void call_flydsl_k(void* a0, int64_t a0_size0, "
            "int64_t a0_stride0, int64_t a1, uint64_t a2, int64_t a3, double a4, "
            "void* a6, hipStream_t stream)",
        )
        launch = self._launch(
            exported,
            tensor_args={
                "X": ("a0", ["a0_size0"], ["a0_stride0"]),
                "Out": ("a6", [], []),
            },
            scalar_args={"i8": "a1", "u64": "a2", "flag": "a3", "f": "a4"},
            stream_expr="stream",
        )
        self.assertEqual(lines[1:], launch)
        text = "\n".join(launch)
        # Any value of the launcher's uint64_t fits: no check in the launcher.
        self.assertIn(
            "  const uint64_t flydsl_arg2 = static_cast<uint64_t>(a2);  // u64", launch
        )
        self.assertNotIn("scalar u64 does not fit", text)
        self.assertIn("argument X is not 16-byte aligned", text)
        self.assertIn("scalar i8 does not fit in int8_t", text)

    def test_cpp_uint64_arg(self):
        # A call site passing a uint64_t launcher parameter: Python ints are checked
        # when the code is generated, values computed at run time when it runs.
        exported = self._cpp_kernel()
        for value, literal in (
            (0, "0ULL"),
            (2**63, "9223372036854775808ULL"),
            (2**64 - 1, "18446744073709551615ULL"),
        ):
            self.assertEqual(
                fuka.cpp_uint64_arg(exported, "u64", value, str(value)), ([], literal)
            )
        for value in (-1, 2**64):
            with self.assertRaisesRegex(
                ValueError,
                rf"FlyDSL kernel flydsl_k: scalar u64={value} does not fit in uint64_t",
            ):
                fuka.cpp_uint64_arg(exported, "u64", value, str(value))
        self.assertEqual(
            fuka.cpp_uint64_arg(exported, "u64", object(), "s0 * 4"),
            (
                [
                    'AOTI_RUNTIME_CHECK((s0 * 4) >= 0, "FlyDSL kernel flydsl_k: '
                    'scalar u64 does not fit in uint64_t");'
                ],
                "s0 * 4",
            ),
        )

    def test_call_args_scalars(self):
        # generate_user_defined_flydsl_kernel_call's arguments for runtime scalars:
        # Python values as literals, symbolic ones through the wrapper; a scalar the
        # launcher takes as a uint64_t is checked now (a Python int) or at run time.
        exported = self._cpp_kernel()
        lines: list[str] = []
        wrapper: Any = types.SimpleNamespace(
            _flydsl_call_wrappers={
                "call_flydsl_k": fuka.FlyDSLCallWrapper("call_flydsl_k", exported, {})
            },
            writeline=lines.append,
            codegen_cpp_sizevar=lambda e: f"cpp({e})",
            generate_float_value=lambda v: f"{v}f",
        )

        def call_args(**runtime_args):
            return fuka.flydsl_call_args(wrapper, "call_flydsl_k", runtime_args)

        self.assertEqual(
            call_args(i8=-3, u64=2**63, flag=True, f=0.5),
            ["-3", "9223372036854775808ULL", "true", "0.5f"],
        )
        self.assertEqual(lines, [])
        s0 = sympy.Symbol("s0", integer=True)
        self.assertEqual(
            call_args(i8=s0, u64=4 * s0, flag=False, f=0.5),
            ["cpp(s0)", "cpp(4*s0)", "false", "0.5f"],
        )
        self.assertEqual(
            lines,
            [
                'AOTI_RUNTIME_CHECK((cpp(4*s0)) >= 0, "FlyDSL kernel flydsl_k: '
                'scalar u64 does not fit in uint64_t");'
            ],
        )
        with self.assertRaisesRegex(ValueError, "scalar u64=-1 does not fit"):
            call_args(i8=0, u64=-1)

    def test_call_args_int_constants_checked(self):
        # A Python int for an integer scalar is checked against the scalar's C type
        # when the code is generated, as eager refuses a value FlyDSL would wrap: an
        # int64_t too (the launcher takes it as it is), and beyond 64 bits (no C++
        # literal holds it). The smallest int64 is spelled so that C++ reads it as
        # one. A Boolean or Float32 scalar takes any int.
        exported = self._cpp_kernel(
            fuka.ScalarParamSpec("i8", FakeInt8, 1),
            fuka.ScalarParamSpec("k", FakeInt64, 1),
            fuka.ScalarParamSpec("flag", FakeBoolean, True),
            fuka.ScalarParamSpec("f", FakeFloat32, 0.5),
            fuka.StreamParamSpec("stream"),
        )
        wrapper: Any = types.SimpleNamespace(
            _flydsl_call_wrappers={
                "call_flydsl_k": fuka.FlyDSLCallWrapper("call_flydsl_k", exported, {})
            }
        )

        def call_args(i8=0, k=0):
            return fuka.flydsl_call_args(
                wrapper, "call_flydsl_k", {"i8": i8, "k": k, "flag": 3, "f": 2}
            )

        self.assertEqual(
            call_args(-128, 2**63 - 1), ["-128", "9223372036854775807", "3", "2"]
        )
        self.assertEqual(
            call_args(127, -(2**63)), ["127", "(-9223372036854775807LL - 1)", "3", "2"]
        )
        i64 = "int64_t [-9223372036854775808, 9223372036854775807]"
        for name, value, bounds in (
            ("i8", 128, "int8_t [-128, 127]"),
            ("i8", -129, "int8_t [-128, 127]"),
            ("k", 2**63 + 5, i64),
            ("k", -(2**63) - 1, i64),
            ("k", 2**64 + 5, i64),
        ):
            with self.assertRaisesRegex(
                ValueError,
                re.escape(
                    f"FlyDSL kernel flydsl_k: scalar {name}={value} does not fit in "
                    f"{bounds}, and FlyDSL would wrap it"
                ),
            ):
                call_args(**{name: value})

    def test_define_flydsl_launcher(self):
        # CppWrapperGpu.define_user_defined_flydsl_kernel's body: the export of the
        # call's specs for the device's arch; once per prefix, its declarations go
        # into the wrapper's header and its launcher into _flydsl_call_wrappers.
        exported = self._cpp_kernel()
        specs = list(exported.specs)
        wrapper: Any = types.SimpleNamespace(
            flydsl_exported_kernels={},
            _flydsl_call_wrappers={},
            header=IndentedBuffer(),
        )
        kernel = FlyDSLKernel(_aot_add_kernel, ("Out",))
        device = torch.device("cuda", 0)
        with contextlib.ExitStack() as stack:
            param_specs, target_arch, export_for_link = (
                stack.enter_context(mock.patch.object(fuka, name, return_value=value))
                for name, value in (
                    ("flydsl_param_specs", (specs, {"X": 16})),
                    ("target_arch", "gfx950"),
                    ("export_for_link", exported),
                )
            )
            for _ in range(2):
                self.assertEqual(
                    fuka.define_flydsl_launcher(
                        wrapper, kernel, {"alpha": 2}, {"X": "x"}, device
                    ),
                    "call_flydsl_k",
                )
        param_specs.assert_called_with(kernel, {"X": "x"}, {"alpha": 2})
        target_arch.assert_called_with(device)
        args, kwargs = export_for_link.call_args
        self.assertEqual(args, (_aot_add_kernel, specs, "gfx950"))
        self.assertIs(kwargs["linked"], wrapper.flydsl_exported_kernels)
        self.assertEqual(os.path.basename(kwargs["out_root"]), "flydsl_user_kernels")
        self.assertEqual(wrapper.flydsl_exported_kernels, {"flydsl_k": exported})
        launcher = wrapper._flydsl_call_wrappers["call_flydsl_k"]
        self.assertEqual(
            (launcher.exported, launcher.check_align), (exported, {"X": 16})
        )
        self.assertEqual(
            wrapper.header.getvalue().count("#define FLYDSL_DECLARED_flydsl_k"), 1
        )

    def test_export_for_link(self):
        root = self._tmp()
        # Static sizes, as in every export test here: an export with dynamic ones
        # hooks FlyDSL's compiler (_default_flat_work_group_size), which needs flydsl.
        specs = self._add_specs(1000)

        def export(linked):
            return fuka.export_for_link(
                _aot_add_kernel, specs, "gfx950", linked=linked, out_root=root
            )

        with _fake_flydsl_export() as fake:
            e1 = export({})
            self.assertRegex(e1.prefix, r"^flydsl_aot_add_kernel_gfx950_[0-9a-f]{16}$")
            self.assertEqual(
                e1.prefix, fuka.flydsl_kernel_prefix(_aot_add_kernel, specs, "gfx950")
            )
            self.assertEqual(e1.specs, specs)
            self.assertEqual(
                [(s.kind, s.name) for s in e1.abi.slots],
                [
                    ("tensor_data", "X"),
                    ("tensor_data", "Y"),
                    ("tensor_data", "Out"),
                    ("scalar", "n"),
                    ("stream", "stream"),
                ],
            )
            out_dir = os.path.dirname(e1.object_path)
            self.assertEqual(os.path.dirname(out_dir), root)
            self.assertEqual(
                sorted(os.listdir(out_dir)), [f"{e1.prefix}.h", f"{e1.prefix}.o"]
            )
            self.assertEqual(e1.header_path, os.path.join(out_dir, f"{e1.prefix}.h"))
            self.assertEqual(Path(e1.header_path).read_text(), e1.header_text)
            # Reused within the process, also when the library links it already.
            self.assertIs(export({}), e1)
            self.assertIs(export({e1.prefix: e1}), e1)
            self.assertEqual(fake.prefixes, [e1.prefix])
            self.assertEqual(len(fuka._exports[_aot_add_kernel]), 1)
            # An export whose directory is gone is exported again; nothing is read
            # back from disk.
            shutil.rmtree(out_dir)
            e2 = export({})
            self.assertEqual(e2.prefix, e1.prefix)
            self.assertNotEqual(e2.object_path, e1.object_path)
            # Another kernel of the library has the prefix: a new one.
            other = dataclasses.replace(e2, header_path="other.h")
            e3 = export({e2.prefix: other})
            self.assertEqual(e3.prefix, f"{e1.prefix}_1")
            self.assertEqual(fake.prefixes, [e1.prefix, e1.prefix, e3.prefix])

    def test_export_errors(self):
        out_dir = self._tmp()
        specs = self._add_specs(1000)

        def export(specs=specs, prefix="flydsl_k", arch="gfx950"):
            return fuka.export_flydsl_kernel(
                _aot_add_kernel, specs, out_dir=out_dir, prefix=prefix, arch=arch
            )

        with _fake_flydsl_export() as fake:
            for kwargs, error, message in (
                (
                    {"specs": specs[:4]},
                    ValueError,
                    r"must name every parameter of _aot_add_kernel",
                ),
                ({"prefix": "a-b"}, ValueError, "prefix 'a-b' is not a C identifier"),
                ({"arch": "sm_90a"}, ValueError, "arch must look like 'gfx950'"),
                (
                    {"specs": self._add_specs(1000, n_type=FakeFloat16)},
                    fuka.FlyDSLExportError,
                    "Float16 is not supported by AOTInductor",
                ),
            ):
                with self.assertRaisesRegex(error, message):
                    export(**kwargs)
            self.assertEqual(fake.prefixes, [])
            # The target recorded in the metadata must be the one asked for (FlyDSL
            # takes it from the environment); nothing reaches out_dir.
            fake.arch_override = "gfx942"
            with self.assertRaisesRegex(
                fuka.FlyDSLExportError,
                "FlyDSL export flydsl_k: compiled for 'gfx942', not the target "
                "'gfx950'",
            ):
                export()
            self.assertEqual(os.listdir(out_dir), [])
            fake.arch_override = None
            # A stray global symbol: validated before anything reaches out_dir.
            fake.extra_symbols = ["flydsl_rt_helper"]
            with self.assertRaisesRegex(
                fuka.FlyDSLExportError,
                r"defines global symbols outside flydsl_k flydsl_k__metadata "
                r".*: \['flydsl_rt_helper'\]",
            ):
                export()
            self.assertEqual(os.listdir(out_dir), [])
            fake.extra_symbols = []
            exported = export()
        self.assertEqual(exported.object_path, os.path.join(out_dir, "flydsl_k.o"))
        self.assertEqual(sorted(os.listdir(out_dir)), ["flydsl_k.h", "flydsl_k.o"])

    def test_export_cache_keyed_like_a_launch(self):
        # Besides the specs, target and prefix, an export is keyed with what the
        # eager launch key adds: the values of the module globals the kernel reads (a
        # tuple is not part of the prefix, which records plain constants only), the
        # FlyDSL environment variables and the compile hints. Each change exports
        # again, under the same prefix; once undone, the first export is reused.
        root = self._tmp()
        specs = (
            fuka.TensorParamSpec("X", torch.float32, (8,), (1,), 16),
            fuka.TensorParamSpec("Out", torch.float32, (8,), (1,), 16),
            fuka.StreamParamSpec("stream"),
        )

        def export():
            return fuka.export_for_link(
                _aot_tile_kernel, specs, "gfx950", linked={}, out_root=root
            )

        with _fake_flydsl_export() as fake:
            e1 = export()
            changed = []
            with mock.patch.dict(globals(), {"AOT_TILE": (3.0,)}):
                changed.append(export())
            fake.env = ("3",)
            changed.append(export())
            fake.env = ()
            hints = {"fast_fp_math": True}
            with mock.patch.dict(_aot_tile_kernel.compile_hints, hints):
                changed.append(export())
            self.assertIs(export(), e1)
        self.assertEqual(len({id(e) for e in (e1, *changed)}), 4)
        self.assertEqual(fake.prefixes, [e1.prefix] * 4)

    def test_export_cache_follows_rebound_globals(self):
        # A rebound plain-constant global the kernel reads changes its prefix: the
        # export misses, so the kernel is exported again; with the global back, the
        # first export is reused (D-aoti-1: the cache key includes the prefix).
        root = self._tmp()
        specs = (
            fuka.TensorParamSpec("X", torch.float32, (8,), (1,), 16),
            fuka.TensorParamSpec("Out", torch.float32, (8,), (1,), 16),
            fuka.StreamParamSpec("stream"),
        )

        def export():
            return fuka.export_for_link(
                _aot_signed_kernel, specs, "gfx950", linked={}, out_root=root
            )

        with _fake_flydsl_export() as fake:
            e1 = export()
            with mock.patch.dict(globals(), {"AOT_SIGN": -1.0}):
                e2 = export()
                self.assertNotEqual(e2.prefix, e1.prefix)
                self.assertIs(export(), e2)
            self.assertIs(export(), e1)
            self.assertEqual(fake.prefixes, [e1.prefix, e2.prefix])


instantiate_parametrized_tests(FlyDSLHOPTests)
instantiate_parametrized_tests(FlyDSLWrapTests)
instantiate_parametrized_tests(FlyDSLDynamoTests)
instantiate_parametrized_tests(FlyDSLInductorTests)
instantiate_parametrized_tests(FlyDSLAOTITests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
