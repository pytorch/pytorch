# Owner(s): ["module: inductor"]
"""Tests for user-defined FlyDSL kernels (``@flyc.jit`` host functions) in PT2.

Modeled on the Triton suite (test/inductor/test_triton_kernels.py) and the OSS
CuTeDSL stack's test_cutedsl_kernels.py: direct HOP use, wrap_flydsl eager launches
and tracing (make_fx, non-strict export), the FlyDSL compile helpers, and Dynamo.
Needs a ROCm GPU and flydsl >= 0.4.
"""

import functools
import inspect
import io
import math
import os
import re
import sys
import threading
import unittest
import weakref
from collections.abc import Callable
from typing import Any
from unittest import mock

import torch
import torch._dynamo.testing
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
)
from torch._higher_order_ops.flydsl_kernel_wrap import (
    flydsl_kernel_side_table,
    flydsl_kernel_wrapper_functional,
    flydsl_kernel_wrapper_mutation,
    FlyDSLKernel,
    get_flydsl_mutated_tensors,
    get_host_fn_info,
)
from torch._inductor.runtime import flydsl_user_kernel
from torch._inductor.runtime.flydsl_user_kernel import (
    FLYDSL_COMPILE_LOCK,
    flydsl_version_ok,
)
from torch._inductor.test_case import TestCase
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
                r"\(mutates_args=\('X', 'Out'\)\), which PT2 cannot functionalize",
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
        # container, here a set in the Constexpr tuple) is a graph break naming the
        # kernel, not an internal error; eager refuses it with the same TypeError.
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

    def test_kernel_output_in_autograd_graph(self):
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
        torch.compile(f, backend="aot_eager", fullgraph=True)(x, y).backward()
        self.assertEqual(x.grad, eager_grad)
        self.assertEqual(eager_grad, (x + y).detach())


instantiate_parametrized_tests(FlyDSLHOPTests)
instantiate_parametrized_tests(FlyDSLWrapTests)
instantiate_parametrized_tests(FlyDSLDynamoTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
