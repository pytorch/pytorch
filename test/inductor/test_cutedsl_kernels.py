# Owner(s): ["module: inductor"]
"""Tests for user-defined CuTeDSL kernels (``@cute.jit`` host functions) in PT2.

Mirrors test/inductor/test_triton_kernels.py (the reference suite for user-defined
Triton kernels): direct HOP use, the eager launch, wrap_cutedsl tracing (make_fx,
non-strict export), Dynamo and Inductor (mutation without clones, reinplacing,
views, dynamic shapes, CUDA graphs).
"""

import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import importlib.util
import inspect
import io
import logging
import os
import re
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import torch
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
)
from torch._dynamo.utils import ReinplaceCounters
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    CUTEDSL_CONSTANT_RULES,
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_functional,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLKernel,
    get_cutedsl_mutated_tensors,
    TraceableCuteDSLKernelWrapper,
)
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._library.cutedsl import wrap_cutedsl
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


HAS_CUDA = torch.cuda.is_available() and torch.version.hip is None

# The DSL is an optional dependency: guard the imports and define the toy kernels
# only when it (and a CUDA device) is available.
_CUTLASS_IMPORT_ERROR = None
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTLASS = True
except Exception as e:  # ImportError, or a driver/library load failure
    HAS_CUTLASS = False
    _CUTLASS_IMPORT_ERROR = e

HAS_CUTEDSL = HAS_CUDA and HAS_CUTLASS

if not HAS_CUDA:
    _ENV_SKIP_REASON = "requires a CUDA device"
elif not HAS_CUTLASS:
    _ENV_SKIP_REASON = f"requires nvidia-cutlass-dsl ({_CUTLASS_IMPORT_ERROR!r})"
else:
    _ENV_SKIP_REASON = None

requires_cutedsl = unittest.skipUnless(HAS_CUTEDSL, _ENV_SKIP_REASON or "")


if HAS_CUTEDSL:
    # 1-D kernels use 128 threads per block and guard with the runtime n.

    @cute.kernel
    def _add_k(gX: cute.Tensor, gY: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gOut[i] = gX[i] + gY[i]

    @cute.jit
    def add_kernel(
        mX: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _add_k(mX, mY, mOut, n).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Reads and writes mX: x *= 2.
    @cute.kernel
    def _scale_inplace_k(gX: cute.Tensor, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gX[i] = gX[i] * 2.0

    @cute.jit
    def scale_inplace_kernel(mX: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        _scale_inplace_k(mX, n).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # 2-D: out[i, j] = x[i, j] + 1, indexing through each tensor's own layout, so
    # non-contiguous inputs and outputs (transposes, slices) work. The extent comes
    # from mX's shape: there is no scalar argument at all.
    @cute.kernel
    def _add_one_2d_k(gX: cute.Tensor, gOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        m, n = gX.shape
        idx = bidx * 128 + tidx
        if idx < m * n:
            gOut[idx // n, idx % n] = gX[idx // n, idx % n] + 1.0

    @cute.jit
    def add_one_2d_kernel(mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        m, n = mX.shape
        _add_one_2d_k(mX, mOut).launch(
            grid=[(m * n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Constexpr param: alpha is baked into the compiled kernel.
    @cute.kernel
    def _mul_const_k(
        gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, alpha: cutlass.Constexpr
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gOut[i] = gX[i] * alpha

    @cute.jit
    def mul_const_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        alpha: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _mul_const_k(mX, mOut, n, alpha).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Two outputs, plus a runtime Float32 scalar with a default that sits after the
    # stream param (the quack style): sum = (x + y) * scale, diff = (x - y) * scale.
    @cute.kernel
    def _add_sub_k(
        gX: cute.Tensor,
        gY: cute.Tensor,
        gSum: cute.Tensor,
        gDiff: cute.Tensor,
        n: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gSum[i] = (gX[i] + gY[i]) * scale
            gDiff[i] = (gX[i] - gY[i]) * scale

    @cute.jit
    def add_sub_kernel(
        mX: cute.Tensor,
        mY: cute.Tensor,
        mSum: cute.Tensor,
        mDiff: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        scale: cutlass.Float32 = 1.0,
    ):
        _add_sub_k(mX, mY, mSum, mDiff, n, scale).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Unannotated param k: out = x + k. PT2, like the DSL, which turns an unannotated
    # int into a dynamic Int32 arg (cutlass/base_dsl/runtime/jit_arg_adapters.py),
    # passes an int for k as a runtime scalar, at call time.
    @cute.kernel
    def _add_scalar_k(
        gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, k: cutlass.Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gOut[i] = gX[i] + cutlass.Float32(k)

    @cute.jit
    def add_scalar_kernel(
        mX: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32, k, stream: cuda.CUstream
    ):
        _add_scalar_k(mX, mOut, n, k).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Invalid for PT2 (get_host_fn_info needs exactly one cuda.CUstream param), but
    # valid CuTeDSL: launch() without a stream uses the default stream.
    @cute.jit
    def no_stream_kernel(
        mX: cute.Tensor, mY: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32
    ):
        _add_k(mX, mY, mOut, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1])

    @cute.jit
    def two_stream_kernel(
        mX: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        stream2: cuda.CUstream,
    ):
        _add_k(mX, mY, mOut, n).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # cfg (an object read on the host) and unused (None) are unannotated, so not
    # Constexpr: the DSL takes no runtime value from them, but the compiled function
    # still takes them at call time. out = x * cfg.alpha.
    @cute.jit
    def mul_cfg_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        cfg,
        stream: cuda.CUstream,
        unused=None,
    ):
        _mul_const_k(mX, mOut, n, cfg.alpha).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Like mul_cfg_kernel, but the host code caches a layout on cfg as it compiles.
    @cute.jit
    def mul_caching_cfg_kernel(
        mX: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32, cfg, stream: cuda.CUstream
    ):
        cfg.layout = cute.make_layout((128,))
        _mul_const_k(mX, mOut, n, cfg.alpha).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # A regular dataclass (mutable, so unhashable) as the default of a Constexpr
    # param: out = x * cfg.alpha.
    @dataclasses.dataclass
    class MulCfg:
        alpha: float = 2.0

    @cute.jit
    def mul_default_cfg_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        cfg: cutlass.Constexpr = MulCfg(),
    ):
        _mul_const_k(mX, mOut, n, cfg.alpha).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # A Constexpr param whose default is an IntEnum member, read as one on the host:
    # out = x * factor.value.
    class Factor(enum.IntEnum):
        TWO = 2
        THREE = 3

    @cute.jit
    def mul_enum_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        factor: cutlass.Constexpr = Factor.THREE,
    ):
        _mul_const_k(mX, mOut, n, factor.value).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # A runtime Float32 scalar whose default is a cutlass Numeric value: out = x + c.
    @cute.kernel
    def _add_c_k(
        gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, c: cutlass.Float32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gOut[i] = gX[i] + c

    @cute.jit
    def add_c_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        c: cutlass.Float32 = cutlass.Float32(0.25),
    ):
        _add_c_k(mX, mOut, n, c).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Like add_c_kernel, but c defaults to None, which the DSL refuses for a cutlass
    # Numeric parameter when it compiles.
    @cute.jit
    def add_opt_c_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        c: cutlass.Float32 = None,
    ):
        _add_c_k(mX, mOut, n, c).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Unannotated runtime scalars whose defaults are numpy scalars:
    # out = x + a + b + c + d.
    @cute.jit
    def add_numpy_defaults_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
        a=np.float32(0.5),
        b=np.float16(0.25),
        c=np.int64(2),
        d=np.bool_(True),
    ):
        total = cutlass.Float32(a) + cutlass.Float32(b)
        total += cutlass.Float32(c) + cutlass.Float32(d)
        _add_c_k(mX, mOut, n, total).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # A kernel object whose @cute.jit __call__ is its host code, as in the CUTLASS
    # examples, and a host function that takes one as a Constexpr and calls it:
    # out = x * alpha.
    class MulKernel:
        def __init__(self, alpha):
            self.alpha = alpha

        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mOut: cute.Tensor,
            n: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            _mul_const_k(mX, mOut, n, self.alpha).launch(
                grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
            )

    # A kernel object whose __call__ sets attributes on it, as CUTLASS kernel objects
    # do (self.a_dtype = mA.element_type, self.tiled_mma = ...): out = x * alpha.
    class CachingMulKernel(MulKernel):
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mOut: cute.Tensor,
            n: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.dtype = mX.element_type
            self.layout = cute.make_layout((128,))
            _mul_const_k(mX, mOut, n, self.alpha).launch(
                grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
            )

    @cute.jit
    def call_mul_kernel(
        kernel: cutlass.Constexpr,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        kernel(mX, mOut, n, stream)

    # A Constexpr container read on the host: out = x * scales[0].
    @cute.jit
    def scale_by_first_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        scales: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _mul_const_k(mX, mOut, n, int(scales[0])).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    def _spec_scale(spec):
        """spec if it is a number, its alpha if it has one, else the product of the
        _spec_scale of the values of a dict or of the items of another container."""
        if isinstance(spec, (int, float)):
            return spec
        if hasattr(spec, "alpha"):
            return spec.alpha
        items = spec.values() if isinstance(spec, dict) else spec
        return functools.reduce(lambda a, b: a * b, map(_spec_scale, items), 1)

    # A Constexpr container read on the host: out = x * _spec_scale(spec).
    @cute.jit
    def spec_scale_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        spec: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _mul_const_k(mX, mOut, n, _spec_scale(spec)).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # A Constexpr config object read on the host: out = x * cfg.alpha.
    @cute.jit
    def cfg_scale_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        cfg: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _mul_const_k(mX, mOut, n, cfg.alpha).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # An unannotated flag, which the DSL makes a runtime Boolean: out = x * 3 if
    # flag else x * 5.
    @cute.kernel
    def _select_k(
        gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, flag: cutlass.Boolean
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            if flag:
                gOut[i] = gX[i] * 3.0
            else:
                gOut[i] = gX[i] * 5.0

    @cute.jit
    def select_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        flag,
        stream: cuda.CUstream,
    ):
        _select_k(mX, mOut, n, cutlass.Boolean(flag)).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )


DEVICE = "cuda"

if HAS_CUTEDSL:
    # Two different host functions with the same __name__ and __qualname__: the
    # compile caches must key on the function object, not its name.
    @cute.kernel
    def _sub_k(gX: cute.Tensor, gY: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gOut[i] = gX[i] - gY[i]

    @cute.jit
    def binary_kernel(
        mX: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _add_k(mX, mY, mOut, n).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    same_name_add_kernel = binary_kernel

    @cute.jit
    def binary_kernel(
        mX: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        _sub_k(mX, mY, mOut, n).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    same_name_sub_kernel = binary_kernel
    del binary_kernel


def _wrap(kernel, mutates_args):
    """wrap_cutedsl(kernel, mutates_args=mutates_args) for the wrappers that most tests
    share as module globals: Dynamo must handle a global TraceableCuteDSLKernelWrapper,
    and the wrappers are reused across tests, between which torch._dynamo.reset() runs
    (it resets the Triton kernel side table; the CuTeDSL records these wrappers hold
    must keep resolving). They are built at import, so if wrap_cutedsl raises, the
    result raises its error when called instead: only the tests that use a wrapper
    fail, not every test of this file."""
    try:
        return wrap_cutedsl(kernel, mutates_args=mutates_args)
    except Exception as e:
        error = e

        def failed(*args, **kwargs):
            raise error

        return failed


if HAS_CUTEDSL:
    add_w = _wrap(add_kernel, ["mOut"])
    scale_inplace_w = _wrap(scale_inplace_kernel, ["mX"])
    add_one_2d_w = _wrap(add_one_2d_kernel, ["mOut"])
    mul_const_w = _wrap(mul_const_kernel, ["mOut"])
    add_sub_w = _wrap(add_sub_kernel, ["mSum", "mDiff"])
    add_scalar_w = _wrap(add_scalar_kernel, ["mOut"])
    same_name_add_w = _wrap(same_name_add_kernel, ["mOut"])
    same_name_sub_w = _wrap(same_name_sub_kernel, ["mOut"])


if HAS_CUTEDSL:
    # The toy: out = relu(x + y), the add launched as in eager CuTeDSL code (cute
    # tensors from from_dlpack, the current stream) with no wrap_cutedsl. Dynamo
    # captures the raw call (torch.compile, strict export); make_fx cannot see it.
    @cute.kernel
    def toy_add_device(
        gX: cute.Tensor, gY: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 1024 + tidx
        if i < n:
            gOut[i] = gX[i] + gY[i]

    @cute.jit
    def toy_add_kernel(
        mX: cute.Tensor,
        mY: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        toy_add_device(mX, mY, mOut, n).launch(
            grid=((n + 1023) // 1024, 1, 1), block=(1024, 1, 1), stream=stream
        )

    def toy_launch_add(x, y, out):
        n = out.numel()
        toy_add_kernel(
            from_dlpack(x),
            from_dlpack(y),
            from_dlpack(out),
            n,
            cutlass_torch.current_stream(),
        )

    def toy(x, y):
        out = torch.empty_like(x)
        toy_launch_add(x, y, out)
        return out.relu()


def _rand(*shape, dtype=torch.float32, requires_grad=False):
    return torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=requires_grad)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _hop_nodes(gm, hop):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is hop]


def _normalize_indices(code):
    """Side-table indices depend on what ran before; blank them for expected code."""
    code = re.sub(r"\bkernel_idx = \d+", "kernel_idx = K", code)
    return re.sub(r"\bconstant_args_idx = \d+", "constant_args_idx = C", code)


def _pointer_alignment(t):
    ptr = t.data_ptr()
    return next(a for a in (16, 8, 4, 2, 1) if ptr % a == 0)


@functools.cache
def _compile_lock():
    from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

    return CUTEDSL_COMPILE_LOCK


@contextlib.contextmanager
def _counting_compiles():
    """A spy on cute.compile, with the eager compile caches (the shared one and the
    eager launchers) emptied for the duration, so that its call count is the number
    of compiles eager launches did."""
    from torch._native.cutedsl import user_kernel as cutedsl_user_kernel

    with (
        mock.patch.dict(cutedsl_user_kernel._compiled, clear=True),
        mock.patch.dict(cutedsl_user_kernel._eager_launchers, clear=True),
        mock.patch.object(cute, "compile", wraps=cute.compile) as compile_spy,
    ):
        yield compile_spy


def _is_constexpr_param(p):
    return isinstance(p.annotation, type) and issubclass(
        p.annotation, cutlass.Constexpr
    )


def _reference_launch(host_fn, *args, **kwargs):
    """Launch ``host_fn`` with plain CuTeDSL, no PT2 involved: torch tensors go
    through from_dlpack (with the alignment their data pointer really has), the
    current stream fills the CUstream param, defaults apply. Positional ``args`` bind
    to the non-stream params in order."""
    params = list(inspect.signature(inspect.unwrap(host_fn)).parameters.values())
    stream_name = next(p.name for p in params if p.annotation is cuda.CUstream)
    bound = dict(zip((p.name for p in params if p.name != stream_name), args))
    bound.update(kwargs)
    bound[stream_name] = _stream()

    def values():
        out = []
        for p in params:
            v = bound.get(p.name, p.default)
            if isinstance(v, torch.Tensor):
                v = from_dlpack(v.detach(), assumed_align=_pointer_alignment(v))
            out.append(v)
        return out

    with _compile_lock():
        compiled = cute.compile(host_fn, *values())
    compiled(*(v for p, v in zip(params, values()) if not _is_constexpr_param(p)))


def _add_kernel_to_table(fn, mutates_args):
    """Side-table index of the CuteDSLKernel record for ``fn``."""
    return cutedsl_kernel_side_table.add_kernel(CuteDSLKernel(fn, mutates_args))


def _add_kwargs(x, y, out, n=None):
    """HOP ``kwargs`` for add_kernel: tensors and the runtime scalar n."""
    return {"mX": x, "mY": y, "mOut": out, "n": out.numel() if n is None else n}


def _make_cfg(alpha):
    """A function with an alpha attribute, made by the same def on every call."""

    def cfg():
        pass

    cfg.alpha = alpha
    return cfg


@requires_cutedsl
class CuteDSLToyKernelTests(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_toy_kernels_plain_cutedsl(self):
        # The toy kernels themselves, through plain CuTeDSL (no PT2): if this fails
        # the environment is broken, not the PT2 integration.
        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        _reference_launch(add_kernel, x, y, out, 1000)
        self.assertEqual(out, x + y)
        xt = _rand(8, 4).t()
        out2 = torch.zeros(4, 8, device=DEVICE)
        _reference_launch(add_one_2d_kernel, xt, out2)
        self.assertEqual(out2, xt + 1)
        s, d = torch.zeros_like(x), torch.zeros_like(x)
        _reference_launch(add_sub_kernel, x, y, s, d, 1000, scale=0.5)
        self.assertEqual(s, (x + y) * 0.5)
        self.assertEqual(d, (x - y) * 0.5)


@requires_cutedsl
class CuteDSLHOPTests(TestCase):
    """Direct use of the HOPs, the side table and the eager launch."""

    hw_classification = HardwareClassification.CUDA

    def test_side_table_dedupes_cutlass_numerics(self):
        # A cutlass Numeric constant is keyed by its type and value, signed zeros
        # apart, so the entry holds a copy: changing the caller's value later leaves
        # the entry and its key as they were. test/test_hop_infra.py tests the rest
        # of the dedupe, which needs no DSL.
        table = cutedsl_kernel_side_table

        def add(c):
            return table.add_constant_args({"c": c})

        numeric = cutlass.Float32(2.0)
        idx = add(numeric)
        numeric.value = 5.0
        self.assertEqual(add(cutlass.Float32(2.0)), idx)
        self.assertEqual(table.get_constant_args(idx)["c"].value, 2.0)
        self.assertNotEqual(add(cutlass.Int32(2)), idx)
        self.assertNotEqual(add(cutlass.Float32(0.0)), add(cutlass.Float32(-0.0)))

    def test_constants_from_cutlass_named_modules_are_keyed_by_content(self):
        # The eager compile cache keys an object of a class from a module merely
        # named like the cutlass package by its attributes, not by a numeric .value
        # like a cutlass Numeric's (test/test_hop_infra.py has the side table's part).
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        class Cfg:
            __module__ = "cutlass_configs"

            def __init__(self, **attrs):
                vars(self).update(attrs)

        x = _rand(1000)
        for alpha in (2.0, 3.0):
            out = torch.empty_like(x)
            args = {"mX": x, "mOut": out, "n": 1000, "cfg": Cfg(value=1, alpha=alpha)}
            launch_cutedsl_kernel(mul_cfg_kernel, args)
            self.assertEqual(out, x * alpha)

    def test_constants_from_the_cutlass_package_are_keyed_as_themselves(self):
        # Equal values of a class from the cutlass package get their own eager
        # compiles (test/test_hop_infra.py has the side table's part).
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        class DslCfg:
            __module__ = "cutlass.cute.fake"

            def __init__(self, alpha):
                self.alpha = alpha

        x = _rand(1000)
        with _counting_compiles() as compile_spy:
            for cfg in (DslCfg(2.0), DslCfg(2.0)):
                out = torch.empty_like(x)
                args = {"mX": x, "mOut": out, "n": 1000, "cfg": cfg}
                launch_cutedsl_kernel(mul_cfg_kernel, args)
                self.assertEqual(out, x * 2.0)
        self.assertEqual(compile_spy.call_count, 2)

    def test_signed_zero_constants(self):
        # Float constants are keyed by their bits in the eager compile cache too, as
        # in the side table (test/test_hop_infra.py): -0.0 and 0.0 are different
        # constants (x * alpha has the sign of alpha for x > 0).
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        x = torch.rand(1000, device=DEVICE) + 1
        for alpha in (0.0, -0.0):
            out = torch.empty_like(x)
            launch_cutedsl_kernel(
                mul_const_kernel, {"mX": x, "mOut": out, "n": 1000, "alpha": alpha}
            )
            self.assertEqual(torch.signbit(out), torch.signbit(x * alpha))

    def test_cyclic_constants_are_refused(self):
        # The eager launch refuses a constant that refers to itself, as constant_key
        # does (test/test_hop_infra.py), naming the kernel and the parameter.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        class Cfg:
            def __init__(self, alpha, **attrs):
                self.alpha = alpha
                vars(self).update(attrs)

        cfg = Cfg(2.0, parent=Cfg(1.0))
        cfg.parent.child = cfg
        x = _rand(1000)
        args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, "cfg": cfg}
        msg = (
            r"^CuTeDSL kernel mul_cfg_kernel: parameter 'cfg': a constant kernel "
            r"argument of type Cfg refers to itself through \.parent\.child$"
        )
        with self.assertRaisesRegex(ValueError, msg):
            launch_cutedsl_kernel(mul_cfg_kernel, args)

    def test_map_constant_walks_what_the_side_table_keys(self):
        # map_constant, on which the wrapper's leaves and specialize_nested are
        # built, walks the containers constant_key keys by content and copies only
        # those whose contents fn changed. It refuses what constant_key refuses as or
        # in a constant, with the same errors, saying where: a tensor (also of a
        # subclass that does not hash, which is no plain object), a DSL runtime
        # object and a cycle.
        from torch._higher_order_ops.user_kernel_hop import constant_key, map_constant

        class UnhashableTensor(torch.Tensor):
            __hash__ = None

        rules = CUTEDSL_CONSTANT_RULES
        value = (MulCfg([1.0, "a"]), {"k": 2.0})
        leaves = []

        def collect(leaf):
            leaves.append(leaf)
            return leaf

        self.assertIs(map_constant(value, rules, collect), value)
        self.assertEqual(leaves, [1.0, "a", "k", 2.0])

        def double(leaf):
            return 2 * leaf if isinstance(leaf, float) else leaf

        doubled = map_constant(value, rules, double)
        self.assertEqual((doubled[0].alpha, doubled[1]), ([2.0, "a"], {"k": 4.0}))
        self.assertEqual(value[0].alpha[0], 1.0)

        tensor_fix = "pass each tensor as its own argument"
        dsl_fix = (
            "pass torch.Tensor and Python values; PT2 converts tensors and supplies "
            "the stream"
        )
        walks = {
            "map_constant": functools.partial(map_constant, fn=collect),
            "constant_key": constant_key,
        }
        for held, name, fix in (
            (_rand(4).as_subclass(UnhashableTensor), "tensor", tensor_fix),
            (from_dlpack(_rand(4), assumed_align=16), "_Tensor", dsl_fix),
            (_stream(), "CUstream", dsl_fix),
        ):
            for value, what in (
                ((MulCfg([1.0, held]),), rf"holds a {name} at \[0\]\.alpha\[1\]"),
                (held, f"is a {name}"),
            ):
                msg = f"^a constant kernel argument {what}: {re.escape(fix)}$"
                for walk_name, walk in walks.items():
                    with self.subTest(held=name, walk=walk_name, top=value is held):
                        with self.assertRaisesRegex(TypeError, msg):
                            walk(value, rules)
        items = [1.0]
        items.append(items)
        msg = r"^a constant kernel argument of type list refers to itself through "
        with self.assertRaisesRegex(ValueError, msg + r"\[1\]$"):
            map_constant(items, rules, collect)

    def test_kernel_object_keyed_by_content_only_under_cutedsl_rules(self):
        # Keying a callable object by its attributes is up to the DSL
        # (ConstantRules.keyed_by_content): under the default rules a kernel object
        # is keyed and stored as itself, like any other callable, while CuTeDSL's
        # rules key it by its attributes and store a copy.
        from torch._higher_order_ops.user_kernel_hop import (
            _snapshot,
            constant_key,
            ConstantRules,
            KernelSideTable,
        )

        kernels = [MulKernel(3.0), MulKernel(3.0)]
        for rules, by_content in (
            (ConstantRules(), False),
            (CUTEDSL_CONSTANT_RULES, True),
        ):
            with self.subTest(rules=type(rules).__name__):
                keys = [constant_key(k, rules) for k in kernels]
                self.assertEqual(keys[0] == keys[1], by_content)
                snapshot = _snapshot(kernels[0], rules)
                self.assertEqual(snapshot is kernels[0], not by_content)
                table = KernelSideTable(constant_rules=rules)
                idx = [table.add_constant_args({"kernel": k}) for k in kernels]
                self.assertEqual(idx[0] == idx[1], by_content)

    def test_kernel_object_check_is_cached_per_class(self):
        # Whether a callable object is a kernel object, which keyed_by_content asks
        # for each one a walk meets, depends only on its class, whose __call__ is
        # inspected once.
        from torch._higher_order_ops.user_kernel_hop import constant_key
        from torch._library import cutedsl as library_cutedsl

        class Kernel(MulKernel):
            pass

        class Functor:
            def __call__(self):
                pass

        rules = CUTEDSL_CONSTANT_RULES
        check = mock.patch.object(
            library_cutedsl,
            "is_cutedsl_jit_function",
            wraps=library_cutedsl.is_cutedsl_jit_function,
        )
        with check as is_cutedsl_jit_function:
            answers = [rules.keyed_by_content(k) for k in (Kernel(1.0), Functor())]
            for _ in range(2):
                constant_key([Kernel(2.0), Functor()], rules)
        self.assertEqual(answers, [True, False])
        self.assertEqual(is_cutedsl_jit_function.call_count, 2)

    def test_constants_holding_tensors_are_refused(self):
        # The eager launch refuses a constant that holds a tensor, as constant_key
        # does (test/test_hop_infra.py), naming the kernel and the parameter.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        @dataclasses.dataclass
        class Cfg:
            alpha: float
            weight: object = None

        x = _rand(1000)
        args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, "cfg": Cfg(2.0, x)}
        msg = (
            r"^CuTeDSL kernel mul_cfg_kernel: parameter 'cfg': a constant kernel "
            r"argument holds a tensor at \.weight: pass each tensor as its own "
            "argument$"
        )
        with self.assertRaisesRegex(TypeError, msg):
            launch_cutedsl_kernel(mul_cfg_kernel, args)

    def test_launch_checks_argument_names(self):
        # The eager launch binds every parameter but the stream by name: missing ones
        # without a default, a name that is not a parameter (here a typo of the
        # defaulted scale, which would launch with scale=1.0), the stream and a
        # tensor for a scalar parameter are refused with a TypeError, and a parameter
        # left to a tensor default with a ValueError, naming the kernel and the
        # parameters.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        x, y, s, d = (_rand(1000) for _ in range(4))
        args = {"mX": x, "mY": y, "mSum": s, "mDiff": d, "n": 1000}
        missing = {k: v for k, v in args.items() if k not in ("mX", "n")}
        typo, stream = {**args, "scael": 0.5}, {**args, "stream": _stream()}
        mul_args = {"mX": x, "mOut": s, "n": 1000, "alpha": x}
        for kernel, bad_args, msg in (
            (add_sub_kernel, missing, r"missing arguments \['mX', 'n'\]"),
            (add_sub_kernel, typo, "got an unexpected argument 'scael'"),
            (add_sub_kernel, stream, "got a value for its stream parameter 'stream'"),
            (add_sub_kernel, {**args, "scale": x}, "parameter 'scale' takes a scalar"),
            (mul_const_kernel, mul_args, "parameter 'alpha' takes a scalar"),
        ):
            name = kernel.__name__
            with self.assertRaisesRegex(TypeError, f"^CuTeDSL kernel {name}: {msg}"):
                launch_cutedsl_kernel(kernel, bad_args)
        launch_cutedsl_kernel(add_sub_kernel, {**args, "scale": 0.5})
        self.assertEqual((s, d), ((x + y) * 0.5, (x - y) * 0.5))

        def tensor_default(mX: cute.Tensor, stream: cuda.CUstream, mY=y):
            pass

        msg = "^CuTeDSL kernel tensor_default: parameter 'mY' defaults to a tensor; "
        with self.assertRaisesRegex(ValueError, msg + "pass the tensor$"):
            launch_cutedsl_kernel(tensor_default, {"mX": x})

    def test_get_host_fn_info_rejects_unlaunchable_signatures(self):
        from torch._higher_order_ops.cutedsl_kernel_wrap import get_host_fn_info

        def positional_only(mX: cute.Tensor, /, stream: cuda.CUstream):
            pass

        def keyword_only(mX: cute.Tensor, stream: cuda.CUstream, *, n: cutlass.Int32):
            pass

        def var_args(mX: cute.Tensor, stream: cuda.CUstream, *rest):
            pass

        def var_kwargs(mX: cute.Tensor, stream: cuda.CUstream, **rest):
            pass

        def string_annotation(mX, stream: cuda.CUstream):
            pass

        for fn, param, kind in (
            (positional_only, "mX", "positional-only"),
            (keyword_only, "n", "keyword-only"),
            (var_args, "rest", "variadic positional"),
            (var_kwargs, "rest", "variadic keyword"),
        ):
            msg = (
                f"^CuTeDSL kernel {fn.__name__}: parameter '{param}' is {kind}, but "
                "PT2 needs every parameter to accept both positional and keyword "
                "arguments$"
            )
            with self.assertRaisesRegex(ValueError, msg):
                get_host_fn_info(fn)
        # What `from __future__ import annotations` would leave.
        string_annotation.__annotations__["mX"] = "cute.Tensor"
        one_stream = r"expected exactly one parameter annotated cuda\.CUstream"
        for fn, msg in (
            (no_stream_kernel, rf"{one_stream} .*, found \[\]$"),
            (two_stream_kernel, rf"{one_stream} .*, found \['stream', 'stream2'\]$"),
            (string_annotation, "parameter 'mX' has a string annotation"),
        ):
            msg = f"^CuTeDSL kernel {fn.__name__}: {msg}"
            with self.assertRaisesRegex(ValueError, msg):
                get_host_fn_info(fn)

    def test_launch_compiles_once_per_argument_signature(self):
        # A compiled CuTeDSL kernel checks nothing at launch, so the eager compile
        # cache keys on the dtype, shape, stride and alignment of every tensor: a
        # change in any of them compiles again, and every result must be right.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        def add(x, y):
            out = torch.empty_like(x)
            launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))
            self.assertEqual(out, x + y)

        def add_one_2d(x):
            out = torch.empty(x.shape, device=DEVICE)
            launch_cutedsl_kernel(add_one_2d_kernel, {"mX": x, "mOut": out})
            self.assertEqual(out, x + 1)

        x, y = _rand(1001), _rand(1001)
        self.assertEqual((_pointer_alignment(x), _pointer_alignment(x[1:])), (16, 4))
        with _counting_compiles() as compile_spy:
            add(x[:1000], y[:1000])
            add(x[:1000].clone(), y[:1000].clone())
            self.assertEqual(compile_spy.call_count, 1)
            add(x[1:], y[1:])
            self.assertEqual(compile_spy.call_count, 2)
            add(x[:1000].half(), y[:1000].half())
            self.assertEqual(compile_spy.call_count, 3)
            add_one_2d(_rand(8, 4))
            add_one_2d(_rand(4, 8).t())
            self.assertEqual(compile_spy.call_count, 5)

    def test_launch_on_the_given_stream(self):
        # stream= is the raw stream to launch on: queued behind a sleep on a side
        # stream, the kernel has not run yet when the current stream reads its output.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel
        from torch.testing._internal.common_utils import get_cycles_per_ms

        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        args = _add_kwargs(x, y, out)
        launch_cutedsl_kernel(add_kernel, args)  # compiles outside the timed part
        out.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(int(200 * get_cycles_per_ms()))
        launch_cutedsl_kernel(add_kernel, args, stream=side.cuda_stream)
        before = out.clone()
        torch.cuda.current_stream().wait_stream(side)
        self.assertEqual(before, torch.zeros_like(x))
        self.assertEqual(out, x + y)

    def test_cutedsl_hop_object_and_none_constants_for_unannotated_params(self):
        # The launch passes the constants of parameters that are not Constexpr (an
        # object, None) at call time too. An object is keyed by its contents in the
        # side table and the compile cache alike: equal objects share an entry and a
        # compile, and one mutated between launches is compiled again.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_cfg_kernel, ["mOut"])

        def launch(**constants):
            out = torch.empty_like(x)
            cidx = cutedsl_kernel_side_table.add_constant_args(constants)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"mX": x, "mOut": out, "n": 1000},
            )
            self.assertEqual(out, x * constants["cfg"].alpha)
            return cidx

        with _counting_compiles() as compile_spy:
            cfg = Cfg(3.0)
            cidx = launch(cfg=cfg)
            self.assertEqual(launch(cfg=Cfg(3.0)), cidx)
            out = torch.empty_like(x)
            args = {"mX": x, "mOut": out, "n": 1000, "cfg": Cfg(3.0)}
            launch_cutedsl_kernel(mul_cfg_kernel, args)  # no side table
            self.assertEqual(out, x * 3.0)
            self.assertEqual(compile_spy.call_count, 1)
            cfg.alpha = 0.5
            self.assertNotEqual(launch(cfg=cfg), cidx)
            launch(cfg=cfg, unused=None)
            self.assertEqual(compile_spy.call_count, 2)

    def test_launch_compiles_copies_of_the_constants(self):
        # The host code that a compile runs can change a constant (here it caches a
        # layout on cfg). The compile gets copies, so the side table's snapshot and
        # the caller's object keep the contents they were keyed by, and later
        # launches through either reuse the compile.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        x, cfg = _rand(1000), Cfg(3.0)
        kidx = _add_kernel_to_table(mul_caching_cfg_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({"cfg": cfg})
        with _counting_compiles() as compile_spy:
            for _ in range(2):
                out = torch.empty_like(x)
                cutedsl_kernel_wrapper_mutation(
                    kernel_idx=kidx,
                    constant_args_idx=cidx,
                    kwargs={"mX": x, "mOut": out, "n": 1000},
                )
                self.assertEqual(out, x * 3.0)
                args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, "cfg": cfg}
                launch_cutedsl_kernel(mul_caching_cfg_kernel, args)
                self.assertEqual(args["mOut"], x * 3.0)
            self.assertEqual(compile_spy.call_count, 1)
        snapshot = cutedsl_kernel_side_table.get_constant_args(cidx)["cfg"]
        self.assertEqual((vars(snapshot), vars(cfg)), ({"alpha": 3.0},) * 2)

    def test_cutedsl_hop_unhashable_default_of_a_constexpr_param(self):
        # A default the call leaves out is keyed like a passed constant, so a regular
        # (unhashable) dataclass default is keyed by its fields.
        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_default_cfg_kernel, ["mOut"])
        for constants, alpha in (({}, 2.0), ({"cfg": MulCfg(3.0)}, 3.0)):
            out = torch.empty_like(x)
            cidx = cutedsl_kernel_side_table.add_constant_args(constants)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"mX": x, "mOut": out, "n": 1000},
            )
            self.assertEqual(out, x * alpha)

    def test_runtime_scalars_reach_the_dsl_as_python_values(self):
        # The DSL converts only exact bool/int/float: an IntEnum for an unannotated
        # param, and a cutlass Numeric default, are passed as plain Python values.
        class K(enum.IntEnum):
            FIVE = 5

        x = _rand(1000)
        out = torch.empty_like(x)
        cutedsl_kernel_wrapper_mutation(
            kernel_idx=_add_kernel_to_table(add_scalar_kernel, ["mOut"]),
            constant_args_idx=cutedsl_kernel_side_table.add_constant_args({}),
            kwargs={"mX": x, "mOut": out, "n": 1000, "k": K.FIVE},
        )
        self.assertEqual(out, x + 5)

        out = torch.empty_like(x)
        cutedsl_kernel_wrapper_mutation(
            kernel_idx=_add_kernel_to_table(add_c_kernel, ["mOut"]),
            constant_args_idx=cutedsl_kernel_side_table.add_constant_args({}),
            kwargs={"mX": x, "mOut": out, "n": 1000},
        )
        self.assertEqual(out, x + 0.25)

        # An int for a float parameter is passed as a float, even one beyond int64.
        y, total, diff = _rand(1000), torch.empty_like(x), torch.empty_like(x)
        cutedsl_kernel_wrapper_mutation(
            kernel_idx=_add_kernel_to_table(add_sub_kernel, ["mSum", "mDiff"]),
            constant_args_idx=cutedsl_kernel_side_table.add_constant_args({}),
            kwargs={
                "mX": x,
                "mY": y,
                "mSum": total,
                "mDiff": diff,
                "n": 1000,
                "scale": 2**64,
            },
        )
        self.assertEqual(total, (x + y) * 2.0**64)

    def test_constexpr_default_reaches_the_dsl_as_written(self):
        # Only runtime scalars are converted: the IntEnum default of a Constexpr param
        # reaches cute.compile as the enum member, as in a plain CuTeDSL call.
        from torch._higher_order_ops.cutedsl_kernel_wrap import get_host_fn_info

        defaults = get_host_fn_info(mul_enum_kernel).defaults
        self.assertIs(defaults["factor"], Factor.THREE)
        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_enum_kernel, ["mOut"])
        for constants, expected in (({}, 3 * x), ({"factor": Factor.TWO}, 2 * x)):
            out = torch.empty_like(x)
            cidx = cutedsl_kernel_side_table.add_constant_args(constants)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"mX": x, "mOut": out, "n": 1000},
            )
            self.assertEqual(out, expected)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_launch_on_a_device_that_is_not_current(self):
        # DLPack only exports tensors of the current device, so the launch switches
        # to the tensors' device first; tensors on two devices are refused.
        torch.empty(1, device="cuda:0")  # a context on the current device
        x, y = _rand(1000).to("cuda:1"), _rand(1000).to("cuda:1")
        out = torch.empty_like(x)
        idx = _add_kernel_to_table(add_kernel, ["mOut"])
        with torch.cuda.device(0):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=idx,
                constant_args_idx=cutedsl_kernel_side_table.add_constant_args({}),
                kwargs=_add_kwargs(x, y, out),
            )
            self.assertEqual(out, x + y)
            with self.assertRaisesRegex(ValueError, "one CUDA device"):
                cutedsl_kernel_wrapper_mutation(
                    kernel_idx=idx,
                    constant_args_idx=cutedsl_kernel_side_table.add_constant_args({}),
                    kwargs=_add_kwargs(x, y.to("cuda:0"), out),
                )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_launch_shares_compiles_across_devices_of_one_capability(self):
        # A compiled function's kernels are resolved on every device (see the
        # docstring of torch/_native/cutedsl/user_kernel.py), so a signature
        # compiled under cuda:0 launches on cuda:1 without another compile.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
            self.skipTest("requires two CUDA devices of one compute capability")
        with _counting_compiles() as compile_spy:
            for device in ("cuda:0", "cuda:1"):
                x, y = _rand(1000).to(device), _rand(1000).to(device)
                out = torch.empty_like(x)
                launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))
                self.assertEqual(out, x + y)
        self.assertEqual(compile_spy.call_count, 1)

    def test_launch_keys_dataclass_constants_by_all_their_attributes(self):
        # A dataclass constant is keyed by its attributes outside its fields too, in
        # the eager compile cache and the side table (test/test_hop_infra.py): two
        # configs that differ only in one set in __post_init__ or ad hoc are
        # compiled apart, through the HOP or not.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        @dataclasses.dataclass
        class Cfg:
            name: str

            def __post_init__(self):
                self.alpha = 2.0

        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_cfg_kernel, ["mOut"])
        with _counting_compiles() as compile_spy:
            for alpha in (2.0, 3.0):
                cfg = Cfg("mul")
                cfg.alpha = alpha
                cidx = cutedsl_kernel_side_table.add_constant_args({"cfg": cfg})
                out = torch.empty_like(x)
                cutedsl_kernel_wrapper_mutation(
                    kernel_idx=kidx,
                    constant_args_idx=cidx,
                    kwargs={"mX": x, "mOut": out, "n": 1000},
                )
                self.assertEqual(out, x * alpha)
                args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, "cfg": cfg}
                launch_cutedsl_kernel(mul_cfg_kernel, args)
                self.assertEqual(args["mOut"], x * alpha)
        self.assertEqual(compile_spy.call_count, 2)

    def test_numpy_scalars_reach_the_dsl_as_python_values(self):
        # A numpy bool, integer or floating scalar for an unannotated parameter is a
        # runtime scalar, passed or as a default, so other values of the same types
        # share a compile; one for a cutlass float parameter reaches it as a float.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        x = _rand(1000)
        passed = {
            "a": np.float32(1.5),
            "b": np.float16(-0.5),
            "c": np.int64(-3),
            "d": np.bool_(False),
        }
        with _counting_compiles() as compile_spy:
            for scalars, total in (({}, 3.75), (passed, -2.0)):
                args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, **scalars}
                launch_cutedsl_kernel(add_numpy_defaults_kernel, args)
                self.assertEqual(args["mOut"], x + total)
            self.assertEqual(compile_spy.call_count, 1)
            for c in (np.float16(0.5), np.int64(2)):
                args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, "c": c}
                launch_cutedsl_kernel(add_c_kernel, args)
                self.assertEqual(args["mOut"], x + float(c))
            self.assertEqual(compile_spy.call_count, 2)

    def test_eager_compiles_are_logged_to_native_dsl_compile(self):
        # Eager compiles go through plan_cache.cached_plan, as the native ops' do, so
        # TORCH_LOGS=+native_dsl_compile logs each one under the kernel's name; a
        # launch that reuses a compile logs nothing.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel
        from torch.testing._internal.logging_utils import log_settings

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        name = "torch._native.instrumentation.__native_dsl_compile"
        logger = logging.getLogger(name)
        x, y = _rand(1000), _rand(1000)
        with log_settings("+native_dsl_compile"), _counting_compiles():
            logger.addHandler(handler)
            try:
                for _ in range(2):
                    out = torch.empty_like(x)
                    launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))
            finally:
                logger.removeHandler(handler)
        messages = [record.getMessage() for record in records]
        self.assertEqual(len(messages), 1, messages)
        self.assertRegex(messages[0], r"^add_kernel \[cutedsl\] compiled in ")

    def test_hop_launch_reuses_the_side_table_keys(self):
        # The side table has keyed the constants it stores, so a launch through the
        # HOP reuses their keys instead of walking them again, and it shares its
        # compile with a launch of equal constants that keys them itself.
        from torch._higher_order_ops import user_kernel_hop
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_const_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({"alpha": 3.0})
        kwargs = {"mX": x, "mOut": torch.empty_like(x), "n": 1000}
        key_spy = mock.patch.object(
            user_kernel_hop, "constant_key", wraps=user_kernel_hop.constant_key
        )
        with _counting_compiles() as compile_spy, key_spy as constant_key:
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=kwargs
            )
            self.assertEqual(constant_key.call_count, 0)
            launch_cutedsl_kernel(mul_const_kernel, {**kwargs, "alpha": 3.0})
            self.assertEqual(constant_key.call_count, 1)
        self.assertEqual(compile_spy.call_count, 1)
        self.assertEqual(kwargs["mOut"], x * 3.0)

    def test_side_table_records(self):
        r1 = CuteDSLKernel(add_kernel, ("mOut",))
        r2 = CuteDSLKernel(add_kernel, ["mOut", "mOut"])  # stored as ("mOut",)
        self.assertEqual(r1, r2)
        self.assertEqual(hash(r1), hash(r2))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r1.mutates_args = "unknown"
        idx = cutedsl_kernel_side_table.add_kernel(r1)
        # Equal records share an index; a different write set is another kernel.
        self.assertEqual(cutedsl_kernel_side_table.add_kernel(r2), idx)
        self.assertEqual(cutedsl_kernel_side_table.get_kernel(idx), r1)
        self.assertIs(cutedsl_kernel_side_table.get_kernel(idx).fn, add_kernel)
        idx_unknown = _add_kernel_to_table(add_kernel, "unknown")
        self.assertNotEqual(idx_unknown, idx)
        cidx = cutedsl_kernel_side_table.add_constant_args({"alpha": 3})
        self.assertEqual(
            cutedsl_kernel_side_table.get_constant_args(cidx), {"alpha": 3}
        )

    def test_side_table_record_checks_mutates_args(self):
        # The record refuses a name that is not a parameter (a typo, whose argument
        # would never be cloned and written back, so a write would be lost) or names
        # one the kernel cannot write (the stream, a Constexpr or a cutlass Numeric
        # parameter) with a ValueError, and anything but "unknown" or an iterable of
        # str with a TypeError, as wrap_cutedsl does (normalize_mutates_args): None,
        # a bare string, a non-str name, an array of names (numpy strings), and an
        # object equal to everything, compared with "unknown" only if it is a str.
        # An empty write set does not parse the signature.
        msg = (
            r"^CuTeDSL kernel {}: mutates_args names '{}', but the parameters it can "
            r"write \(all but the stream, Constexpr and cutlass Numeric ones\) are "
            r"\({}\)$"
        )
        add_params = "'mX', 'mY', 'mOut'"
        for fn, names, bad, params in (
            (add_kernel, ("mout",), "mout", add_params),
            (add_kernel, ["mOut", "stream"], "stream", add_params),
            (add_kernel, ["n"], "n", add_params),
            (mul_const_kernel, ["mOut", "alpha"], "alpha", "'mX', 'mOut'"),
        ):
            error = msg.format(fn.__name__, bad, params)
            with self.assertRaisesRegex(ValueError, error):
                CuteDSLKernel(fn, names)

        class EqualToAll:
            def __eq__(self, other):
                return True

        iterable = (
            'CuTeDSL kernel add_kernel: mutates_args must be "unknown" or an iterable '
            "of parameter names, each exactly a str (not a numpy string or another str "
            "subclass), got "
        )
        names = np.array(["mX", "mOut"])
        for bad in ("mOut", None, 5, [1], ["mOut", 1], names, EqualToAll()):
            error = "^" + re.escape(f"{iterable}{bad!r}") + "$"
            with self.assertRaisesRegex(TypeError, error):
                CuteDSLKernel(add_kernel, bad)
        self.assertEqual(CuteDSLKernel(no_stream_kernel, ()).mutates_args, ())

    def test_cutedsl_hop_eager(self):
        # Mirrors test_triton_kernel_higher_order_func.
        kidx = _add_kernel_to_table(add_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})
        x, y = _rand(1000), _rand(1000)

        out = torch.zeros_like(x)
        ret = cutedsl_kernel_wrapper_mutation(
            kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
        )
        self.assertIsNone(ret)
        self.assertEqual(out, x + y)

        # The functional HOP launches on a clone and leaves its input alone.
        out = torch.zeros_like(x)
        out_dict = cutedsl_kernel_wrapper_functional(
            kernel_idx=kidx,
            constant_args_idx=cidx,
            kwargs=_add_kwargs(x, y, out),
            tensors_to_clone=["mOut"],
        )
        self.assertEqual(set(out_dict), {"mOut"})
        self.assertEqual(out_dict["mOut"], x + y)
        self.assertEqual(out, torch.zeros_like(x))

    def test_cutedsl_hop_functional_clone_keeps_unwritten_elements(self):
        # n < numel: the kernel writes only the head, so the clone must carry the
        # input's tail (a clone, not a fresh allocation).
        kidx = _add_kernel_to_table(add_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})
        x, y = _rand(1000), _rand(1000)
        out = torch.full_like(x, 7.0)
        res = cutedsl_kernel_wrapper_functional(
            kernel_idx=kidx,
            constant_args_idx=cidx,
            kwargs=_add_kwargs(x, y, out, n=500),
            tensors_to_clone=["mOut"],
        )["mOut"]
        self.assertEqual(res[:500], x[:500] + y[:500])
        self.assertEqual(res[500:], torch.full_like(x[500:], 7.0))
        self.assertEqual(out, torch.full_like(x, 7.0))

    def test_cutedsl_hop_eager_inplace_and_2d(self):
        kidx = _add_kernel_to_table(scale_inplace_kernel, ["mX"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})
        x = torch.arange(1000, device=DEVICE, dtype=torch.float32)
        cutedsl_kernel_wrapper_mutation(
            kernel_idx=kidx, constant_args_idx=cidx, kwargs={"mX": x, "n": 1000}
        )
        self.assertEqual(x, 2 * torch.arange(1000, device=DEVICE, dtype=torch.float32))

        kidx = _add_kernel_to_table(add_one_2d_kernel, ["mOut"])
        xt = _rand(8, 4).t()  # strides (1, 4)
        out = torch.zeros(4, 8, device=DEVICE)
        cutedsl_kernel_wrapper_mutation(
            kernel_idx=kidx, constant_args_idx=cidx, kwargs={"mX": xt, "mOut": out}
        )
        self.assertEqual(out, xt + 1)

    def test_eager_launch_imports_neither_dynamo_nor_inductor(self):
        # This process has imported both, so the launch runs in a fresh one, from a
        # file since cute.jit reads the host function's source.
        import subprocess
        import sys
        import tempfile
        import textwrap

        program = textwrap.dedent(
            """\
            import sys

            import cuda.bindings.driver as cuda
            import cutlass
            import cutlass.cute as cute
            import torch
            from torch._higher_order_ops.cutedsl_kernel_wrap import (
                cutedsl_kernel_side_table as table,
                cutedsl_kernel_wrapper_mutation,
                CuteDSLKernel,
            )


            @cute.kernel
            def scale_k(gX: cute.Tensor, n: cutlass.Int32):
                tidx, _, _ = cute.arch.thread_idx()
                bidx, _, _ = cute.arch.block_idx()
                i = bidx * 128 + tidx
                if i < n:
                    gX[i] = gX[i] * 2.0


            @cute.jit
            def scale(mX: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
                grid = [(n + 127) // 128, 1, 1]
                scale_k(mX, n).launch(grid=grid, block=[128, 1, 1], stream=stream)


            x = torch.ones(1000, device="cuda")
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=table.add_kernel(CuteDSLKernel(scale, ["mX"])),
                constant_args_idx=table.add_constant_args({}),
                kwargs={"mX": x, "n": 1000},
            )
            loaded = sorted({"torch._dynamo", "torch._inductor"} & set(sys.modules))
            print(x.sum().item(), loaded)
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(program)
            f.flush()
            proc = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "2000.0 []")

    def test_cutedsl_hop_constant_args(self):
        # Values from the side table. The Constexpr alpha is baked into the compiled
        # kernel, so each value is a compile; an int for the unannotated k is a
        # runtime scalar, as in the DSL, so k=5 and k=7 share one compile.
        x = _rand(1000)
        with _counting_compiles() as compile_spy:
            kidx = _add_kernel_to_table(mul_const_kernel, ["mOut"])
            for alpha in (3, 2.5):
                out = torch.zeros_like(x)
                cidx = cutedsl_kernel_side_table.add_constant_args({"alpha": alpha})
                cutedsl_kernel_wrapper_mutation(
                    kernel_idx=kidx,
                    constant_args_idx=cidx,
                    kwargs={"mX": x, "mOut": out, "n": 1000},
                )
                self.assertEqual(out, x * alpha)
            self.assertEqual(compile_spy.call_count, 2)
            kidx = _add_kernel_to_table(add_scalar_kernel, ["mOut"])
            for k in (5, 7):
                out = torch.zeros_like(x)
                cidx = cutedsl_kernel_side_table.add_constant_args({"k": k})
                cutedsl_kernel_wrapper_mutation(
                    kernel_idx=kidx,
                    constant_args_idx=cidx,
                    kwargs={"mX": x, "mOut": out, "n": 1000},
                )
                self.assertEqual(out, x + k)
            self.assertEqual(compile_spy.call_count, 3)

    def test_cutedsl_hop_fake_mode(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        kidx = _add_kernel_to_table(add_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})
        with FakeTensorMode():
            x, y, out = (torch.empty(1000, device=DEVICE) for _ in range(3))
            self.assertIsNone(
                cutedsl_kernel_wrapper_mutation(
                    kernel_idx=kidx,
                    constant_args_idx=cidx,
                    kwargs=_add_kwargs(x, y, out),
                )
            )
            res = cutedsl_kernel_wrapper_functional(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(x, y, out),
                tensors_to_clone=["mOut"],
            )
            self.assertEqual(res["mOut"].shape, out.shape)
            self.assertEqual(res["mOut"].stride(), out.stride())

    @parametrize("tracing_mode", ["fake", "symbolic"])
    def test_cutedsl_hop_make_fx(self, tracing_mode):
        kidx = _add_kernel_to_table(add_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            out = torch.zeros_like(x)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return out

        gm = make_fx(f, tracing_mode=tracing_mode)(_rand(1000), _rand(1000))
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        self.assertEqual(node.name, "cutedsl_kernel_wrapper_mutation_proxy")
        self.assertEqual(node.kwargs["kernel_idx"], kidx)
        self.assertEqual(node.kwargs["constant_args_idx"], cidx)
        hop_kwargs = node.kwargs["kwargs"]
        self.assertEqual(list(hop_kwargs), ["mX", "mY", "mOut", "n"])
        self.assertEqual(hop_kwargs["mX"].op, "placeholder")
        if tracing_mode == "symbolic":
            self.assertIsInstance(hop_kwargs["n"], torch.fx.Node)
        else:
            self.assertEqual(hop_kwargs["n"], 1000)
            self.assertExpectedInline(
                _normalize_indices(gm.code.strip()),
                """\
def forward(self, x_1, y_1):
    zeros_like = torch.ops.aten.zeros_like.default(x_1, pin_memory = False)
    cutedsl_kernel_wrapper_mutation_proxy = torch.ops.higher_order.cutedsl_kernel_wrapper_mutation(kernel_idx = K, constant_args_idx = C, kwargs = {'mX': x_1, 'mY': y_1, 'mOut': zeros_like, 'n': 1000});  x_1 = y_1 = cutedsl_kernel_wrapper_mutation_proxy = None
    return zeros_like""",
            )
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(gm(x, y), x + y)
        if tracing_mode == "symbolic":
            x, y = _rand(300), _rand(300)
            self.assertEqual(gm(x, y), x + y)

    def test_cutedsl_hop_functionalize(self):
        # Mirrors test_triton_kernel_functionalize: functionalization rewrites the
        # mutation HOP into the functional HOP cloning exactly mutates_args.
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(add_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            out = torch.zeros_like(x)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return out

        x, y = _rand(1000), _rand(1000)
        with FunctionalTensorMode():
            gm_py = make_fx(PythonFunctionalizeAPI().functionalize(f))(x, y)
        gm_cpp = make_fx(CppFunctionalizeAPI().functionalize(f))(x, y)
        gm_func = make_fx(torch.func.functionalize(f))(x, y)
        for gm in (gm_py, gm_cpp, gm_func):
            self.assertEqual(_hop_nodes(gm, cutedsl_kernel_wrapper_mutation), [])
            (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_functional)
            self.assertEqual(node.kwargs["tensors_to_clone"], ["mOut"])
            self.assertNotIn("tensor_bases", node.kwargs)
            self.assertEqual(gm(x, y), x + y)

        gm = make_fx(torch.func.functionalize(f), tracing_mode="fake")(x, y)
        self.assertExpectedInline(
            _normalize_indices(gm.code.strip()),
            """\
def forward(self, x_1, y_1):
    zeros_like = torch.ops.aten.zeros_like.default(x_1, pin_memory = False)
    cutedsl_kernel_wrapper_functional_proxy = torch.ops.higher_order.cutedsl_kernel_wrapper_functional(kernel_idx = K, constant_args_idx = C, kwargs = {'mX': x_1, 'mY': y_1, 'mOut': zeros_like, 'n': 1000}, tensors_to_clone = ['mOut']);  x_1 = y_1 = zeros_like = None
    getitem = cutedsl_kernel_wrapper_functional_proxy['mOut'];  cutedsl_kernel_wrapper_functional_proxy = None
    return getitem""",
        )

    def test_cutedsl_hop_functionalize_inplace(self):
        # A kernel that reads and writes the same argument: only it is cloned, and
        # the mutation of the graph input is written back.
        kidx = _add_kernel_to_table(scale_inplace_kernel, ["mX"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def f(x):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"mX": x, "n": x.numel()},
            )
            return x

        x = _rand(1000)
        gm = make_fx(torch.func.functionalize(f))(x.clone())
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["mX"])
        x_in = x.clone()
        self.assertEqual(gm(x_in), 2 * x)
        self.assertEqual(x_in, 2 * x)

    @parametrize("view", ["offset", "gap"])
    def test_cutedsl_hop_functionalize_view_arg(self, view):
        # A mutated arg that is a view is cloned through its base: the functional
        # HOP gets tensor_bases. "gap" makes the view non-dense (cf.
        # test_triton_kernel_mutated_offset_view_not_miscompiled).
        kidx = _add_kernel_to_table(scale_inplace_kernel, ["mX"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def take_view(base):
            return base[4:] if view == "offset" else base[::2]

        def f(base):
            v = take_view(base)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"mX": v, "n": v.numel()},
            )
            return base

        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        base = torch.arange(1, 1001, device=DEVICE, dtype=torch.float32)
        expected = base.clone()
        take_view(expected).mul_(2)

        # Python functionalization (what AOTAutograd runs): the view's base is a
        # tracked value, so it is passed as tensor_bases.
        with FunctionalTensorMode():
            gm = make_fx(PythonFunctionalizeAPI().functionalize(f))(base.clone())
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["mX"])
        self.assertEqual(list(node.kwargs["tensor_bases"]), ["mX"])
        self.assertEqual(gm(base.clone()), expected)

        # torch.func.functionalize: the C++ functional wrappers carry no _base, so
        # there are no tensor_bases (as for Triton); the result and the write-back
        # into the input must still be right.
        gm = make_fx(torch.func.functionalize(f))(base.clone())
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["mX"])
        b = base.clone()
        self.assertEqual(gm(b), expected)
        self.assertEqual(b, expected)

    def test_cutedsl_hop_unknown_mutates_args_clones_every_tensor(self):
        kidx = _add_kernel_to_table(add_kernel, "unknown")
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            out = torch.zeros_like(x)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return out

        x, y = _rand(1000), _rand(1000)
        gm = make_fx(torch.func.functionalize(f))(x, y)
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_functional)
        self.assertEqual(sorted(node.kwargs["tensors_to_clone"]), ["mOut", "mX", "mY"])
        x_in, y_in = x.clone(), y.clone()
        self.assertEqual(gm(x_in, y_in), x + y)
        self.assertEqual((x_in, y_in), (x, y))

    @parametrize("api", ["python", "cpp", "cpp_view_copies"])
    def test_cutedsl_hop_functionalize_refuses_aliased_writes(self, api):
        # test/test_hop_infra.py checks which written arguments functionalization
        # refuses as possibly overlapping. Here the views of one base that it accepts,
        # interleaved ones included (the q, k and v columns of a fused buffer), are
        # written back correctly, unless functionalization copies views
        # (remove="mutations_and_views"); and with mutates_args naming the written
        # parameters, a refusal does not suggest naming fewer.
        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        unknown_kidx = _add_kernel_to_table(add_kernel, "unknown")
        named_kidx = _add_kernel_to_table(add_kernel, ["mX", "mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def launch(kidx, x, y, out):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )

        def trace(f, *args, tracing_mode="real"):
            if api == "python":
                with FunctionalTensorMode():
                    f_func = PythonFunctionalizeAPI().functionalize(f)
                    return make_fx(f_func, tracing_mode=tracing_mode)(*args)
            remove = "mutations_and_views" if api == "cpp_view_copies" else "mutations"
            f_func = torch.func.functionalize(f, remove=remove)
            return make_fx(f_func, tracing_mode=tracing_mode)(*args)

        def in_place(x, y):
            launch(named_kidx, x, y, x)
            return x

        def views(base, y):
            launch(unknown_kidx, base[0], y, base[1])
            return base

        def named_views(base, y):
            launch(named_kidx, base[0], y, base[1])
            return base

        def columns(qkv):
            d = qkv.shape[1] // 3
            launch(unknown_kidx, *(qkv.narrow(1, i * d, d) for i in range(3)))
            return qkv

        def empty(x, y):
            launch(unknown_kidx, x, y, x[500:500])  # an empty view overlaps nothing
            return x

        written = (
            ", and both count as written (mutates_args=('mX', 'mOut')). PT2 clones "
            "each written argument and writes it back on its own, so one write could "
            "be lost. "
        )
        x, y = _rand(1000), _rand(1000)
        base, qkv = torch.stack([x, torch.zeros_like(x)]), _rand(40, 75)
        overlap = "are the same tensor or views that may overlap"
        fix = "Pass tensors that do not overlap."
        refusals = [(in_place, (x.clone(), y), f"{overlap}{written}{fix}")]
        if api == "cpp_view_copies":
            copied = (
                "are views of one tensor, which functionalization with "
                'remove="mutations_and_views" copied, so PT2 cannot tell whether they '
                f'overlap{written}Functionalize with remove="mutations" (the default), '
                "which keeps views, or pass tensors that are not views of one tensor."
            )
            refusals.append((named_views, (base, y), copied))
        for f, args, msg in refusals:
            msg = f"CuTeDSL kernel add_kernel: 'mX' and 'mOut' {msg}"
            with self.assertRaisesRegex(RuntimeError, re.escape(msg) + "$"):
                trace(f, *args)
        if api != "cpp_view_copies":
            q, k = qkv[:, :25], qkv[:, 25:50]
            gm = trace(views, base.clone(), y)
            self.assertEqual(gm(base.clone(), y), torch.stack([x, x + y]))
            gm = trace(columns, qkv.clone())
            self.assertEqual(gm(qkv.clone()), torch.cat([q, k, q + k], dim=1))
            # Traced with fake tensors: CUDA refuses a launch of zero blocks (n = 0).
            trace(empty, x.clone(), y, tracing_mode="fake")

    @parametrize("api", ["python", "cpp"])
    def test_cutedsl_hop_functionalize_columns_symbolic(self, api):
        # With symbolic sizes too, the q, k and v columns of a fused buffer are
        # proven disjoint. compute_overlapping_tensors guards on the sizes it
        # compares, and its guards hold for other sizes too.
        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(add_kernel, "unknown")
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def f(x, y):
            n, d = x.shape
            qkv = x.new_zeros(n, 3 * d)
            q, k, v = (qkv.narrow(1, i * d, d) for i in range(3))
            q.copy_(x)
            k.copy_(y)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(q, k, v)
            )
            return qkv

        if api == "python":
            with FunctionalTensorMode():
                f_func = PythonFunctionalizeAPI().functionalize(f)
                gm = make_fx(f_func, tracing_mode="symbolic")(_rand(8, 4), _rand(8, 4))
        else:
            f_func = torch.func.functionalize(f)
            gm = make_fx(f_func, tracing_mode="symbolic")(_rand(8, 4), _rand(8, 4))
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]
        for shape in ((8, 4), (5, 7)):
            x, y = _rand(*shape), _rand(*shape)
            self.assertTrue(gm.shape_env.evaluate_guards_for_args(inputs, [x, y]))
            self.assertEqual(gm(x, y), torch.cat([x, y, x + y], dim=1))

    @parametrize("case", ["e8m0", "complex", "meta", "sparse", "fp8"])
    def test_cutedsl_hop_functional_decomposed_before_type_fallback(self, case):
        # Mirrors test_triton_kernel_functional_decomposition_before_type_fallback: the
        # post-grad decomposition turns the functional HOP into the mutation HOP
        # whatever its tensors' types (the lowering checks them afterwards), instead
        # of skipping the node and failing its "was not removed" check.
        import torch._inductor.lowering as lowering
        from torch._inductor.fx_passes.post_grad import (
            decompose_user_kernel_wrapper_functional,
        )

        kidx = _add_kernel_to_table(add_kernel, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({"n": 16})

        def f(src, output):
            out = cutedsl_kernel_wrapper_functional(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs={"mX": src, "mY": src, "mOut": output},
                tensors_to_clone=["mOut"],
            )
            return out["mOut"]

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
            self.assertEqual(len(_hop_nodes(gm, cutedsl_kernel_wrapper_functional)), 1)
            decompose_user_kernel_wrapper_functional(gm.graph)
            self.assertEqual(len(_hop_nodes(gm, cutedsl_kernel_wrapper_mutation)), 1)

    def test_get_cutedsl_mutated_tensors(self):
        x, y, s, d = (_rand(16) for _ in range(4))
        kwargs = {"mX": x, "mY": y, "mSum": s, "mDiff": d, "n": 16, "scale": 1.0}
        cidx = cutedsl_kernel_side_table.add_constant_args({})
        kidx = _add_kernel_to_table(add_sub_kernel, ["mSum", "mDiff"])
        self.assertEqual(
            sorted(get_cutedsl_mutated_tensors(kidx, cidx, kwargs)), ["mDiff", "mSum"]
        )
        # "unknown": every tensor argument, never the scalars.
        kidx = _add_kernel_to_table(add_sub_kernel, "unknown")
        self.assertEqual(
            sorted(get_cutedsl_mutated_tensors(kidx, cidx, kwargs)),
            ["mDiff", "mSum", "mX", "mY"],
        )

    def test_cutedsl_hop_mutation_hidden_from_autograd(self):
        # Mirrors test_triton_kernel_mutation_type.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        kidx = _add_kernel_to_table(scale_inplace_kernel, ["mX"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def prep():
            x = torch.ones(4, device=DEVICE, requires_grad=True)
            with FunctionalTensorMode():
                x_func = FunctionalTensor.to_functional(x)
            self.assertTrue(torch._is_functional_tensor(x_func.elem))
            return x_func

        def launch(x_func):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs={"mX": x_func, "n": 4}
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

    def test_cutedsl_side_table_is_separate(self):
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        self.assertIsNot(cutedsl_kernel_side_table, kernel_side_table)
        record = CuteDSLKernel(add_kernel, ("mOut",))
        idx = cutedsl_kernel_side_table.add_kernel(record)
        # The write set is part of the record's identity.
        self.assertEqual(
            cutedsl_kernel_side_table.add_kernel(CuteDSLKernel(add_kernel, ("mOut",))),
            idx,
        )
        self.assertNotEqual(
            cutedsl_kernel_side_table.add_kernel(CuteDSLKernel(add_kernel, "unknown")),
            idx,
        )
        # torch._dynamo.reset() resets Triton's table, not this one.
        torch._dynamo.reset()
        self.assertEqual(cutedsl_kernel_side_table.get_kernel(idx), record)

    def test_cutedsl_layout_constraint_ignores_triton_config(self):
        # Inductor keeps the traced strides of a CuTeDSL kernel's tensors, whatever
        # Triton's layout config says.
        from torch._higher_order_ops.user_kernel_hop import get_user_kernel_hop_info
        from torch._inductor import config

        info = get_user_kernel_hop_info(cutedsl_kernel_wrapper_mutation)
        for layout in ("needs_fixed_stride_order", "flexible_layout"):
            with config.patch(triton_kernel_default_layout_constraint=layout):
                constraint = info.get_layout_constraint()
            self.assertEqual(constraint, "needs_fixed_stride_order")


def _is_tensor_node(v):
    return isinstance(v, torch.fx.Node) and isinstance(v.meta.get("val"), torch.Tensor)


def _scalar_args(node):
    """The non-tensor kernel args of a CuTeDSL HOP node, wherever the capture put
    them: scalars inside the node's ``kwargs`` dict (FX nodes when symbolic) plus the
    side-table constant args."""
    args = {k: v for k, v in node.kwargs["kwargs"].items() if not _is_tensor_node(v)}
    args.update(
        cutedsl_kernel_side_table.get_constant_args(node.kwargs["constant_args_idx"])
    )
    return args


@requires_cutedsl
class CuteDSLWrapTests(TestCase):
    """wrap_cutedsl: eager launches, make_fx / non-strict export capture, errors."""

    hw_classification = HardwareClassification.CUDA

    def test_wrap_cutedsl_mutates_args_iterator(self):
        # A one-shot iterable is read once, not consumed by its validation.
        for names in ((n for n in ["mOut"]), iter(["mOut"]), map(str, ["mOut"])):
            w = wrap_cutedsl(add_kernel, mutates_args=names)
            self.assertEqual(w.mutates_args, ("mOut",))

    def test_wrap_cutedsl_make_fx_nested_symint_specialized(self):
        # A SymInt inside a Constexpr container is specialized like a bare one, so the
        # side table only holds concrete values. Containers are the ones the side
        # table walks (see constant_contents), subclasses of tuple (such as
        # torch.Size, and a namedtuple, rebuilt with _make), list and dict included.
        # One that cannot be rebuilt with the new items raises its own error.
        class Scales(list):
            pass

        class ScaleMap(dict):
            pass

        class Pair(tuple):
            __slots__ = ()

            def __new__(cls, a, b):
                return super().__new__(cls, (a, b))

        Tile = collections.namedtuple("Tile", ["m", "n"])
        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

        def f(x, make_scales):
            out = torch.empty_like(x)
            w(x, out, x.numel(), make_scales(x.shape[0] // 250))
            return out

        x = _rand(1000)
        cases = (
            (lambda s: (s,), (4,)),
            (lambda s: torch.Size([s]), torch.Size([4])),
            (lambda s: Tile(s, 1), Tile(4, 1)),
            (lambda s: Scales([s]), Scales([4])),
            (lambda s: ScaleMap({0: s}), ScaleMap({0: 4})),
        )
        for make_scales, expected in cases:
            fn = functools.partial(f, make_scales=make_scales)
            with self.subTest(container=type(expected).__name__):
                gm = make_fx(fn, tracing_mode="symbolic")(x)
                (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
                scales = _scalar_args(node)["scales"]
                self.assertIs(type(scales), type(expected))
                self.assertEqual(scales, expected)
                self.assertIs(type(scales[0]), int)
                self.assertEqual(gm(x), 4 * x)
        fn = functools.partial(f, make_scales=lambda s: Pair(s, 1))
        with self.assertRaisesRegex(TypeError, r"__new__\(\) missing 1 required"):
            make_fx(fn, tracing_mode="symbolic")(x)

    def test_wrap_cutedsl_make_fx_symint_in_object_specialized(self):
        # The side table keys a dataclass instance or plain object by its contents
        # (see _object_state), so a SymInt inside one, given to a Constexpr or an
        # unannotated parameter, is specialized with a guard as inside a tuple, in a
        # copy of the object: in a field of a frozen dataclass too, and in an
        # attribute that is not a field (set in __post_init__).
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        @dataclasses.dataclass(frozen=True)
        class FrozenCfg:
            alpha: int

        @dataclasses.dataclass
        class DerivedCfg:
            n: int

            def __post_init__(self):
                self.alpha = self.n // 2

        mul_default_cfg_w = wrap_cutedsl(mul_default_cfg_kernel, mutates_args=["mOut"])
        mul_cfg_w = wrap_cutedsl(mul_cfg_kernel, mutates_args=["mOut"])
        cfgs = []

        def f(x):
            n = x.shape[0]
            cfgs[:] = [MulCfg(n // 250), Cfg(n // 500), FrozenCfg(n // 125)]
            cfgs.append(DerivedCfg(n // 100))
            outs = [torch.empty_like(x) for _ in cfgs]
            mul_default_cfg_w(x, outs[0], x.numel(), cfg=cfgs[0])
            mul_cfg_w(x, outs[1], x.numel(), cfgs[1])
            mul_default_cfg_w(x, outs[2], x.numel(), cfg=cfgs[2])
            mul_cfg_w(x, outs[3], x.numel(), cfgs[3])
            return outs

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        nodes = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        captured = [_scalar_args(node)["cfg"] for node in nodes]
        classes = [MulCfg, Cfg, FrozenCfg, DerivedCfg]
        self.assertEqual([type(cfg) for cfg in captured], classes)
        self.assertEqual([cfg.alpha for cfg in captured], [4, 2, 8, 5])
        self.assertEqual([type(cfg.alpha) for cfg in captured], [int] * 4)
        self.assertEqual((captured[3].n, type(captured[3].n)), (10, int))
        self.assertEqual([type(cfg.alpha) for cfg in cfgs], [torch.SymInt] * 4)
        size = gm.graph.find_nodes(op="placeholder")[0].meta["val"].shape[0]
        guards = [str(guard.expr) for guard in gm.shape_env.guards]
        for divisor, value in ((250, 4), (500, 2), (125, 8), (100, 10)):
            self.assertIn(f"Eq(({size}//{divisor}), {value})", guards)
        self.assertEqual(gm(x), [4 * x, 2 * x, 8 * x, 5 * x])

    def test_wrap_cutedsl_error_tensor_in_container(self):
        # Only top-level tensors become graph arguments; one inside a container would
        # be a trace-time constant. Containers are the ones the side table walks (see
        # constant_contents): subclasses of list too, and dicts, keys included. The
        # error is the side table's, naming the kernel, the parameter and where the
        # tensor sits, under make_fx and eagerly, and the launch's own, the same, on
        # the fast path.
        from torch._library.triton import set_wrap_triton_enabled

        class Scales(list):
            pass

        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

        def f(x, make_scales):
            out = torch.empty_like(x)
            w(x, out, x.numel(), make_scales(x))
            return out

        x = _rand(1000)
        error = (
            r"^CuTeDSL kernel scale_by_first_kernel: parameter 'scales': a constant "
            r"kernel argument holds a tensor at {}: pass each tensor as its own "
            r"argument$"
        )
        cases = {
            "tuple": (lambda x: (x,), r"\[0\]"),
            "list subclass": (lambda x: Scales([x]), r"\[0\]"),
            "dict key": (lambda x: {x: 1}, r"\.keys\(\)\[0\]"),
        }
        for case, (make_scales, where) in cases.items():
            fn = functools.partial(f, make_scales=make_scales)
            with self.subTest(case=case):
                with self.assertRaisesRegex(TypeError, error.format(where)):
                    make_fx(fn, tracing_mode="fake")(x)
                with self.assertRaisesRegex(TypeError, error.format(where)):
                    fn(x)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(TypeError, error.format(where)):
                        fn(x)

    def test_wrap_cutedsl_error_tensor_in_object(self):
        # A dataclass instance or plain object is a container too: a tensor inside
        # one, in a field or in an attribute that is not a field, would be baked into
        # a make_fx graph as a fake tensor constant, and a cute tensor or stream
        # inside one would reach the DSL. A tensor whose class does not hash is still
        # a tensor, not a plain object. Each is refused with the side table's error,
        # saying where it sits, under make_fx, eagerly and on the fast path.
        from torch._library.triton import set_wrap_triton_enabled

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        class UnhashableTensor(torch.Tensor):
            __hash__ = None

        mul_default_cfg_w = wrap_cutedsl(mul_default_cfg_kernel, mutates_args=["mOut"])

        def f(x, make_cfg):
            out = torch.empty_like(x)
            mul_default_cfg_w(x, out, x.numel(), cfg=make_cfg(x))
            return out

        def with_extra(x):
            cfg = MulCfg(2.0)
            cfg.extra = x
            return cfg

        x = _rand(1000)
        unhashable_x = x.as_subclass(UnhashableTensor)
        cute_x, stream = from_dlpack(x, assumed_align=16), _stream()
        tensor_fix = "pass each tensor as its own argument"
        dsl_fix = (
            "pass torch.Tensor and Python values; PT2 converts tensors and supplies "
            "the stream"
        )
        cases = (
            (MulCfg, "tensor", r"\.alpha", tensor_fix),
            (lambda x: Cfg([x]), "tensor", r"\.alpha\[0\]", tensor_fix),
            (lambda x: MulCfg(unhashable_x), "tensor", r"\.alpha", tensor_fix),
            (with_extra, "tensor", r"\.extra", tensor_fix),
            (lambda x: MulCfg(cute_x), "_Tensor", r"\.alpha", dsl_fix),
            (lambda x: Cfg(stream), "CUstream", r"\.alpha", dsl_fix),
        )
        for i, (make_cfg, held, where, fix) in enumerate(cases):
            error = (
                r"^CuTeDSL kernel mul_default_cfg_kernel: parameter 'cfg': a constant "
                rf"kernel argument holds a {held} at {where}: {re.escape(fix)}$"
            )
            fn = functools.partial(f, make_cfg=make_cfg)
            with self.subTest(case=i):
                with self.assertRaisesRegex(TypeError, error):
                    make_fx(fn, tracing_mode="fake")(x)
                with self.assertRaisesRegex(TypeError, error):
                    fn(x)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(TypeError, error):
                        fn(x)

    def test_wrap_cutedsl_error_cyclic_constant(self):
        # A constant that refers to itself (a config object with a back-pointer, one
        # that is a key of a dict it holds, one holding a logging.Logger, whose
        # manager's loggerDict holds the logger, or a list holding itself) is refused
        # instead of walked until a RecursionError: under make_fx, before a SymInt in
        # it is specialized in a copy, and eagerly, also on the fast path, with the
        # side table's ValueError naming the kernel, the parameter and the cycle. The
        # logger is one of its own manager's: the cycle the walk meets first in a
        # logger of the process depends on the process's loggers and handlers.
        import traceback

        from torch._library.triton import set_wrap_triton_enabled

        log = logging.Manager(logging.RootLogger(logging.WARNING)).getLogger("cfg")

        class Parent:
            def __init__(self, child):
                self.child = child

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha
                self.parent = Parent(self)

        class KeyCfg:
            def __init__(self, alpha):
                self.alpha = alpha
                self.lookup = {self: 1}

        class LoggingCfg:
            def __init__(self, alpha):
                self.alpha = alpha
                self.log = log

        def self_list(alpha):
            cfg = [alpha]
            cfg.append(cfg)
            return cfg

        mul_cfg_w = wrap_cutedsl(mul_cfg_kernel, mutates_args=["mOut"])

        def f(x, make_cfg):
            out = torch.empty_like(x)
            mul_cfg_w(x, out, x.numel(), make_cfg(x.shape[0] // 500))
            return out

        x = _rand(1000)
        cases = (
            (Cfg, r"Cfg refers to itself through \.parent\.child"),
            (KeyCfg, r"KeyCfg refers to itself through \.lookup\.keys\(\)\[0\]"),
            (
                LoggingCfg,
                r"Logger refers to itself through \.manager\.loggerDict\['cfg'\]",
            ),
            (self_list, r"list refers to itself through \[1\]"),
        )
        for make_cfg, cycle in cases:
            fn = functools.partial(f, make_cfg=make_cfg)
            error = (
                "^CuTeDSL kernel mul_cfg_kernel: parameter 'cfg': a constant kernel "
                f"argument of type {cycle}$"
            )
            with self.subTest(cfg=make_cfg.__name__):
                with self.assertRaisesRegex(ValueError, error):
                    make_fx(fn, tracing_mode="symbolic")(x)
                with self.assertRaisesRegex(ValueError, error) as caught:
                    fn(x)
                # Not chained to the walk's ValueError, which says the same.
                printed = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn("During handling", printed)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(ValueError, error):
                        fn(x)

    def test_wrap_cutedsl_cutlass_values_as_constants(self):
        # cutlass Numeric values (like dtype classes and enums) are ordinary
        # constants; only DSL tensors and streams are refused.
        x = _rand(1000)
        out = torch.empty_like(x)
        mul_const_w(x, out, 1000, cutlass.Float32(2.0))
        self.assertEqual(out, 2 * x)

        def f(x):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), cutlass.Float32(3.0))
            return out

        self.assertEqual(make_fx(f)(x)(x), 3 * x)

    def test_wrap_cutedsl_returns_traceable_wrapper(self):
        w = wrap_cutedsl(add_kernel, mutates_args=["mOut"])
        self.assertIsInstance(w, TraceableCuteDSLKernelWrapper)
        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        self.assertIsNone(w(x, y, out, 1000))
        self.assertEqual(out, x + y)

    def test_wrap_cutedsl_builds_one_side_table_record(self):
        # Error messages name the host function without building a CuteDSLKernel
        # record: wrap_cutedsl builds the one it adds to the side table, and calls of
        # the wrapper build none.
        def f(x, y, out):
            w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        with mock.patch(
            "torch._higher_order_ops.cutedsl_kernel_wrap.CuteDSLKernel",
            wraps=CuteDSLKernel,
        ) as record_spy:
            w = wrap_cutedsl(add_kernel, mutates_args=["mOut"])
            f(x, y, out)
            make_fx(f)(x, y, torch.empty_like(x))
            with self.assertRaisesRegex(TypeError, "kernel add_kernel: missing"):
                w(x, y, out)
        self.assertEqual(record_spy.call_count, 1)
        self.assertEqual(out, x + y)

    def test_wrap_cutedsl_eager_kwargs_and_partial_write(self):
        x, y = _rand(1000), _rand(1000)
        out = torch.full_like(x, 7.0)
        add_w(mX=x, mY=y, mOut=out, n=500)
        self.assertEqual(out[:500], x[:500] + y[:500])
        self.assertEqual(out[500:], torch.full_like(x[500:], 7.0))

    def test_wrap_cutedsl_eager_inplace(self):
        x = _rand(1000)
        ref = 2 * x
        scale_inplace_w(x, x.numel())
        self.assertEqual(x, ref)

    @parametrize(
        "layout", ["contiguous", "transposed", "row_slice", "gap_out", "t_out"]
    )
    def test_wrap_cutedsl_eager_2d_layouts(self, layout):
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

    def test_wrap_cutedsl_eager_unaligned_view(self):
        # data_ptr is 4 bytes past a 16-byte boundary: from_dlpack(assumed_align=16)
        # rejects it, so the launch must use the alignment the pointer really has.
        base = _rand(4, 9)
        x = base[:, 1:]
        self.assertNotEqual(x.data_ptr() % 16, 0)
        out = torch.zeros(4, 8, device=DEVICE)
        add_one_2d_w(x, out)
        self.assertEqual(out, x + 1)
        y = _rand(1001)[1:]
        out1 = torch.zeros(1000, device=DEVICE)[0:]
        add_w(y, y, out1, 1000)
        self.assertEqual(out1, 2 * y)

    def test_wrap_cutedsl_eager_compile_cache_keys_on_layout(self):
        # Layouts are static in the compiled kernel and a compiled function called
        # with another shape silently computes garbage, so the eager cache must key
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

    def test_wrap_cutedsl_eager_constexpr(self):
        x = _rand(1000)
        for alpha in (3, 2.5, 3):
            out = torch.zeros_like(x)
            mul_const_w(x, out, 1000, alpha)
            self.assertEqual(out, x * alpha)

    def test_wrap_cutedsl_eager_unannotated_scalar(self):
        x = _rand(1000)
        for k in (5, 7):
            out = torch.zeros_like(x)
            add_scalar_w(x, out, 1000, k)
            self.assertEqual(out, x + k)

    def test_wrap_cutedsl_eager_two_outputs_and_default(self):
        x, y = _rand(1000), _rand(1000)
        s, d = torch.zeros_like(x), torch.zeros_like(x)
        add_sub_w(x, y, s, d, 1000)  # scale defaults to 1.0
        self.assertEqual((s, d), (x + y, x - y))
        add_sub_w(x, y, s, d, 1000, scale=0.5)
        self.assertEqual((s, d), ((x + y) * 0.5, (x - y) * 0.5))

    def test_wrap_cutedsl_eager_same_name_kernels(self):
        # Two host functions with the same __name__/__qualname__ must not share a
        # compiled kernel.
        x, y = _rand(1000), _rand(1000)
        o1, o2 = torch.zeros_like(x), torch.zeros_like(x)
        same_name_add_w(x, y, o1, 1000)
        same_name_sub_w(x, y, o2, 1000)
        self.assertEqual(o1, x + y)
        self.assertEqual(o2, x - y)

    def test_wrap_cutedsl_eager_requires_grad_inputs(self):
        # from_dlpack refuses tensors that require grad ("use tensor.detach()").
        x, y = _rand(1000, requires_grad=True), _rand(1000, requires_grad=True)
        out = torch.zeros(1000, device=DEVICE)
        add_w(x, y, out, 1000)
        self.assertEqual(out, (x + y).detach())

    def test_wrap_cutedsl_eager_launches_on_current_stream(self):
        # Capturing a CUDA graph only works if the launch goes to the capturing
        # (current) stream; a launch on any other stream breaks the capture.
        from torch._library.triton import set_wrap_triton_enabled
        from torch.testing._internal.common_utils import get_cycles_per_ms

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
        # On a side stream, through the HOP or not, the launch is queued behind a
        # sleep there, so the stream current outside reads the output before the
        # kernel has run.
        side = torch.cuda.Stream()
        for enabled in (True, False):
            out.zero_()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), set_wrap_triton_enabled(enabled):
                torch.cuda._sleep(int(200 * get_cycles_per_ms()))
                add_w(x, y, out, 1000)
            before = out.clone()
            torch.cuda.current_stream().wait_stream(side)
            self.assertEqual(before, torch.zeros_like(x))
            self.assertEqual(out, x + y)

    def test_wrap_after_plain_cute_compile(self):
        # The DSL deletes _dsl_cls from the user function at its first compile
        # (only the functools.wraps wrapper keeps a copy), so wrapping a host
        # function that was already compiled must still be accepted.
        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        _reference_launch(add_kernel, x, y, out, 1000)
        self.assertEqual(out, x + y)
        w = wrap_cutedsl(add_kernel, mutates_args=["mOut"])
        out2 = torch.zeros_like(x)
        w(x, y, out2, 1000)
        self.assertEqual(out2, x + y)

    def test_wrap_cutedsl_eager_matches_plain_cutedsl(self):
        # The wrapper's eager launch is exactly the plain CuTeDSL launch.
        x = _rand(64, 300)
        ref, out = torch.zeros_like(x), torch.zeros_like(x)
        _reference_launch(add_one_2d_kernel, x, ref)
        add_one_2d_w(x, out)
        self.assertEqual(out, ref)

    def test_wrap_cutedsl_eager_fast_path_skips_hop(self):
        # With wrap_triton disabled, as in a triton_op's eager backend, the wrapper
        # launches directly instead of dispatching the HOP, and still checks the call.
        # It neither walks nor specializes an argument itself: the launch walks each
        # constant once, to key it, and refuses a tensor in one as the HOP path does.
        # The kernel name comes from the parsed signature (CuteDSLHostFnInfo.name),
        # not from unwrapping the host function again.
        from torch._higher_order_ops import cutedsl_kernel_wrap, user_kernel_hop
        from torch._library.triton import set_wrap_triton_enabled

        hop_calls = []
        hop_type = type(cutedsl_kernel_wrapper_mutation)
        hop_call = hop_type.__call__

        def spy_hop_call(hop, *args, **kwargs):
            hop_calls.append(kwargs["kernel_idx"])
            return hop_call(hop, *args, **kwargs)

        def spy(module, name):
            return mock.patch.object(module, name, wraps=getattr(module, name))

        scale_by_first_w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])
        x, y = _rand(1000), _rand(1000)
        out, out2, out3 = (torch.zeros_like(x) for _ in range(3))
        with mock.patch.object(hop_type, "__call__", spy_hop_call):
            add_w(x, y, out, 1000)
            self.assertEqual(hop_calls, [add_w.kernel_idx])  # the spy sees the HOP
            with set_wrap_triton_enabled(False):
                add_w(x, y, out2, 1000)
                walk = spy(cutedsl_kernel_wrap, "map_constant")
                key = spy(user_kernel_hop, "constant_key")
                name = spy(cutedsl_kernel_wrap, "cutedsl_kernel_name")
                with walk as map_constant, key as constant_key, name as kernel_name:
                    scale_by_first_w(x, out3, 1000, (2,))
                self.assertEqual(map_constant.call_count, 0)
                self.assertEqual(constant_key.call_count, 1)
                self.assertEqual(kernel_name.call_count, 0)
                with self.assertRaisesRegex(TypeError, "its stream parameter"):
                    add_w(x, y, out2, 1000, stream=_stream())
                error = r"'scales': a constant kernel argument holds a tensor at \[0\]"
                with self.assertRaisesRegex(TypeError, error):
                    scale_by_first_w(x, out2, 1000, (x,))
        self.assertEqual(hop_calls, [add_w.kernel_idx])
        self.assertEqual(out, x + y)
        self.assertEqual(out2, x + y)
        self.assertEqual(out3, 2 * x)

    def test_wrap_cutedsl_error_stream_passed(self):
        # A call that passes the stream does not bind: a TypeError (by keyword, the
        # launch's own, from CuteDSLHostFnInfo.check_call), through the HOP or not,
        # and under make_fx with fake tensors, where the launch does not run.
        from torch._library.triton import set_wrap_triton_enabled

        x = _rand(16)
        out = torch.zeros_like(x)
        w = add_w
        keyword = (
            "^CuTeDSL kernel add_kernel: got a value for its stream parameter "
            "'stream', which the launch supplies$"
        )
        positional = (
            r"^CuTeDSL kernel add_kernel: takes 4 positional arguments \(the "
            r"parameters before its stream parameter 'stream', which the launch "
            r"supplies\) but got 5$"
        )
        cases = (
            ("keyword", lambda x, out: w(x, x, out, 16, stream=_stream()), keyword),
            ("positional", lambda x, out: w(x, x, out, 16, _stream()), positional),
        )
        for case, call, error in cases:
            with self.subTest(case=case):
                for enabled in (True, False):
                    with set_wrap_triton_enabled(enabled):
                        with self.assertRaisesRegex(TypeError, error):
                            call(x, out)
                with self.assertRaisesRegex(TypeError, error):
                    make_fx(call, tracing_mode="fake")(x, out)
        self.assertEqual(out, torch.zeros_like(x))

    def test_wrap_cutedsl_error_positional_argument_past_the_stream(self):
        # Callers of a wrapper never pass the stream, so a positional argument at or
        # past its position is refused, naming the parameter to pass by keyword,
        # rather than bound to the parameter after the stream: here a scale, a
        # CUstream (which would be refused as a value for scale) and a raw stream
        # handle (which would launch with scale set to it).
        from torch._library.triton import set_wrap_triton_enabled

        x, y = _rand(1000), _rand(1000)
        s, d = torch.zeros_like(x), torch.zeros_like(x)
        error = (
            r"^CuTeDSL kernel add_sub_kernel: takes 5 positional arguments \(the "
            r"parameters before its stream parameter 'stream', which the launch "
            r"supplies\) but got 6; pass 'scale' by keyword$"
        )

        def f(x, value):
            add_sub_w(x, x, torch.empty_like(x), torch.empty_like(x), 1000, value)

        handle = torch.cuda.current_stream().cuda_stream
        for value in (0.5, _stream(), handle):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(TypeError, error):
                    add_sub_w(x, y, s, d, 1000, value)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(TypeError, error):
                        add_sub_w(x, y, s, d, 1000, value)
                with self.assertRaisesRegex(TypeError, error):
                    make_fx(functools.partial(f, value=value))(x)
        self.assertEqual((s, d), (torch.zeros_like(x), torch.zeros_like(x)))
        add_sub_w(x, y, s, d, 1000, scale=0.5)
        self.assertEqual((s, d), ((x + y) * 0.5, (x - y) * 0.5))

    def test_wrap_cutedsl_error_stream_params(self):
        # Exactly one cuda.CUstream parameter is required: wrap_cutedsl refuses any
        # other host function itself, whatever mutates_args, with the error of the
        # side-table record.
        one_stream = (
            r"expected exactly one parameter annotated cuda\.CUstream \(PT2 launches "
            r"the kernel on the current CUDA stream\), found "
        )
        for fn, found in (
            (no_stream_kernel, r"\[\]"),
            (two_stream_kernel, r"\['stream', 'stream2'\]"),
        ):
            error = f"^CuTeDSL kernel {fn.__name__}: {one_stream}{found}$"
            with self.subTest(fn=fn.__name__):
                for mutates_args in (["mOut"], "unknown"):
                    with self.assertRaisesRegex(ValueError, error):
                        wrap_cutedsl(fn, mutates_args=mutates_args)
                with self.assertRaisesRegex(ValueError, error):
                    CuteDSLKernel(fn, ["mOut"])

    def test_wrap_cutedsl_error_bad_mutates_args(self):
        # Refused by wrap_cutedsl itself, not at the first call of the wrapper, with
        # the side-table record's errors (normalize_mutates_args): a ValueError for a
        # name of a parameter the kernel cannot write, a TypeError for anything but
        # "unknown" or an iterable of str, such as an array of names (numpy strings)
        # or an object equal to everything, which is not taken for "unknown".
        class EqualToAll:
            def __eq__(self, other):
                return True

        def names(bad, params="'mX', 'mY', 'mOut'"):
            return (
                f"mutates_args names {bad!r}, but the parameters it can write (all but "
                f"the stream, Constexpr and cutlass Numeric ones) are ({params})"
            )

        def iterable(bad):
            return (
                'mutates_args must be "unknown" or an iterable of parameter names, '
                "each exactly a str (not a numpy string or another str subclass), got "
                f"{bad!r}"
            )

        cases = [
            (add_kernel, ["not_a_param"], ValueError, names("not_a_param")),
            (add_kernel, ["n"], ValueError, names("n")),  # a cutlass Numeric parameter
            (add_kernel, ["stream"], ValueError, names("stream")),
            (mul_const_kernel, ["alpha"], ValueError, names("alpha", "'mX', 'mOut'")),
        ]
        for bad in ("mOut", None, [1], np.array(["mX", "mOut"]), EqualToAll()):
            cases.append((add_kernel, bad, TypeError, iterable(bad)))
        for fn, bad, error_type, message in cases:
            error = f"^{re.escape(f'CuTeDSL kernel {fn.__name__}: {message}')}$"
            with self.subTest(mutates_args=repr(bad)):
                with self.assertRaisesRegex(error_type, error):
                    wrap_cutedsl(fn, mutates_args=bad)
                with self.assertRaisesRegex(error_type, error):
                    CuteDSLKernel(fn, bad)

    def test_wrap_cutedsl_error_scalar_for_a_written_parameter(self):
        # An unannotated parameter may take a tensor, so mutates_args can name it,
        # but a call that gives it a scalar is refused with a TypeError, and one that
        # leaves it to a default other than None with a ValueError, eagerly, on the
        # fast path and under make_fx (with fake tensors: the call's own checks
        # refuse it, not the launch). None stands for an optional tensor that the
        # call leaves out.
        from torch._library.triton import set_wrap_triton_enabled

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        # The DSL re-runs this def in the module's globals, so defaults name no locals.
        @cute.jit
        def add_default_k_kernel(
            mX: cute.Tensor,
            mOut: cute.Tensor,
            n: cutlass.Int32,
            stream: cuda.CUstream,
            k=5,
        ):
            _add_scalar_k(mX, mOut, n, k).launch(
                grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
            )

        x = _rand(1000)
        out = torch.zeros_like(x)

        def assert_refused(w, error_type, error, *args):
            with self.assertRaisesRegex(error_type, error):
                w(x, out, 1000, *args)
            with set_wrap_triton_enabled(False):
                with self.assertRaisesRegex(error_type, error):
                    w(x, out, 1000, *args)
            with self.assertRaisesRegex(error_type, error):
                make_fx(
                    lambda x: w(x, torch.empty_like(x), x.numel(), *args),
                    tracing_mode="fake",
                )(x)

        names = (
            "^CuTeDSL kernel {}: mutates_args names parameter 'k', which takes a "
            "tensor or None, "
        )
        w = wrap_cutedsl(add_scalar_kernel, mutates_args=["mOut", "k"])
        assert_refused(w, TypeError, names.format("add_scalar_kernel") + "not int$", 5)
        w = wrap_cutedsl(add_default_k_kernel, mutates_args=["mOut", "k"])
        left_to = "but the call leaves it to its default, of type int$"
        assert_refused(w, ValueError, names.format("add_default_k_kernel") + left_to)
        self.assertEqual(out, torch.zeros_like(x))
        wrap_cutedsl(add_default_k_kernel, mutates_args=["mOut"])(x, out, 1000)
        self.assertEqual(out, x + 5)

        w = wrap_cutedsl(mul_cfg_kernel, mutates_args=["mOut", "unused"])
        cfg = Cfg(3.0)
        for kwargs in ({"unused": None}, {}):
            out.zero_()
            w(x, out, 1000, cfg, **kwargs)
            self.assertEqual(out, 3 * x)
        error = (
            "^CuTeDSL kernel mul_cfg_kernel: mutates_args names parameter 'unused', "
            "which takes a tensor or None, not float$"
        )
        with self.assertRaisesRegex(TypeError, error):
            w(x, out, 1000, cfg, unused=0.5)

    def test_wrap_cutedsl_error_tensor_default(self):
        # A parameter left to a tensor default would be launched with a tensor PT2
        # never saw, so it could not functionalize a write to it: the call is
        # refused, written parameter or not ("unknown" counts every tensor argument
        # as written), with the launch's ValueError, eagerly, on the fast path and
        # under make_fx with fake tensors, where the launch does not run. A call that
        # passes the tensor launches.
        from torch._library.triton import set_wrap_triton_enabled

        # The DSL re-runs this def in the module's globals, so defaults name no locals.
        @cute.jit
        def add_aux_kernel(
            mX: cute.Tensor,
            mOut: cute.Tensor,
            n: cutlass.Int32,
            stream: cuda.CUstream,
            mAux=torch.zeros(4, device=DEVICE),
        ):
            _add_scalar_k(mX, mOut, n, 1).launch(
                grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
            )

        x = _rand(1000)
        error = (
            "^CuTeDSL kernel add_aux_kernel: parameter 'mAux' defaults to a tensor; "
            "pass the tensor$"
        )
        for mutates_args in (["mOut", "mAux"], ["mOut"], "unknown"):
            w = wrap_cutedsl(add_aux_kernel, mutates_args=mutates_args)
            out = torch.zeros_like(x)
            with self.subTest(mutates_args=mutates_args):
                with self.assertRaisesRegex(ValueError, error):
                    w(x, out, 1000)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(ValueError, error):
                        w(x, out, 1000)
                with self.assertRaisesRegex(ValueError, error):
                    make_fx(
                        lambda x: w(x, torch.empty_like(x), x.numel()),
                        tracing_mode="fake",
                    )(x)
                self.assertEqual(out, torch.zeros_like(x))
                w(x, out, 1000, mAux=torch.zeros(4, device=DEVICE))
                self.assertEqual(out, x + 1)

    def test_wrap_cutedsl_error_not_cute_jit(self):
        # Only @cute.jit host functions are wrapped: not a plain function, a
        # @cute.kernel device function, or a functools.wraps wrapper of a host
        # function (a decorator, or torch.compiler.disable), although that copies the
        # host function's _dsl_cls. Telling one needs CuTeDSL's class, imported from
        # where the vendored quack code imports it: a cutlass that moved it raises
        # its ImportError instead of refusing every host function, and without
        # cutlass imported no function is one.
        from torch._library.cutedsl import is_cutedsl_jit_function

        def plain(mX, mY, mOut, n, stream: cuda.CUstream):
            pass

        @functools.wraps(add_kernel)
        def logged(*args, **kwargs):
            return add_kernel(*args, **kwargs)

        error = (
            "^CuTeDSL kernel {}: wrap_cutedsl only works on host functions decorated "
            r"with @cute\.jit$"
        )
        # A wrapper's name is that of the host function it wraps.
        for fn, name in ((plain, "plain"), (_add_k, "_add_k"), (logged, "add_kernel")):
            with self.subTest(fn=fn.__qualname__):
                with self.assertRaisesRegex(TypeError, error.format(name)):
                    wrap_cutedsl(fn, mutates_args=["mOut"])
        self.assertTrue(is_cutedsl_jit_function(add_kernel))
        for fn in (logged, torch.compiler.disable(add_kernel)):
            self.assertIn("_dsl_cls", vars(fn))
            self.assertFalse(is_cutedsl_jit_function(fn))
        moved = {
            name: types.ModuleType(name)
            for name in ("cutlass.cutlass_dsl", "cutlass.cutlass_dsl.cutlass")
        }
        with mock.patch.dict(sys.modules, moved):
            with self.assertRaisesRegex(ImportError, "cannot import name 'CuTeDSL'"):
                wrap_cutedsl(add_kernel, mutates_args=["mOut"])
        with mock.patch.dict(sys.modules, {"cutlass": None}):
            self.assertFalse(is_cutedsl_jit_function(add_kernel))

    def test_wrap_cutedsl_error_bound_method(self):
        # A bound @cute.jit method, or a kernel object standing for its bound
        # __call__ as in cute.compile, is refused with an error that names the method
        # and a workaround, which works: a host function that takes the object as a
        # Constexpr and calls it. An object whose __call__ is not @cute.jit is not
        # a kernel object.
        kernel = MulKernel(3.0)
        error = (
            r"^CuTeDSL kernel __call__: wrap_cutedsl does not support bound methods or "
            r"kernel objects \(got MulKernel\.__call__\): instead, wrap a @cute\.jit "
            r"function that takes the object as a cutlass\.Constexpr parameter and "
            r"calls the method on it, and pass the object to that wrapper, which keys "
            r"a kernel object by its attributes: equal objects share one compile, and "
            r"one mutated between calls is compiled again$"
        )
        for host_fn in (kernel.__call__, kernel):
            with self.subTest(host_fn=type(host_fn).__name__):
                with self.assertRaisesRegex(TypeError, error):
                    wrap_cutedsl(host_fn, mutates_args=["mOut"])
        error = (
            r"^CuTeDSL kernel cutedsl_kernel: wrap_cutedsl only works on host "
            r"functions decorated with @cute\.jit$"
        )
        with self.assertRaisesRegex(TypeError, error):
            wrap_cutedsl(add_w, mutates_args=["mOut"])

        w = wrap_cutedsl(call_mul_kernel, mutates_args=["mOut"])

        def f(x):
            out = torch.empty_like(x)
            w(kernel, x, out, x.numel())
            return out

        x = _rand(1000)
        self.assertEqual(f(x), 3 * x)
        self.assertEqual(make_fx(f)(x)(x), 3 * x)

    def test_wrap_cutedsl_kernel_object_keyed_by_attributes(self):
        # A kernel object given to a host function that calls it is keyed by its
        # attributes (see _object_state), in the eager compile cache and the side
        # table alike: equal objects, such as one built for each call, share one
        # compile and one side-table entry, which holds a snapshot, and an object
        # mutated between calls is compiled again. Any other callable object is keyed
        # by its identity.
        from torch._library.triton import set_wrap_triton_enabled

        w = wrap_cutedsl(call_mul_kernel, mutates_args=["mOut"])

        def f(x, kernel):
            out = torch.empty_like(x)
            w(kernel, x, out, x.numel())
            return out

        x = _rand(1000)
        with _counting_compiles() as compile_spy:
            self.assertEqual(f(x, MulKernel(3.0)), 3 * x)
            self.assertEqual(f(x, MulKernel(3.0)), 3 * x)
            with set_wrap_triton_enabled(False):
                self.assertEqual(f(x, MulKernel(3.0)), 3 * x)
            self.assertEqual(compile_spy.call_count, 1)
            kernel = MulKernel(3.0)
            f(x, kernel)
            kernel.alpha = 0.5
            self.assertEqual(f(x, kernel), 0.5 * x)
            self.assertEqual(compile_spy.call_count, 2)

        kernels = [MulKernel(4.0), MulKernel(4.0)]
        gms = [make_fx(functools.partial(f, kernel=k))(x) for k in kernels]
        nodes = [_hop_nodes(gm, cutedsl_kernel_wrapper_mutation)[0] for gm in gms]
        idx = nodes[0].kwargs["constant_args_idx"]
        self.assertEqual(nodes[1].kwargs["constant_args_idx"], idx)
        stored = cutedsl_kernel_side_table.get_constant_args(idx)["kernel"]
        self.assertIs(type(stored), MulKernel)
        self.assertFalse(any(stored is k for k in kernels))
        kernels[0].alpha = 1.0
        self.assertEqual(stored.alpha, 4.0)
        self.assertEqual([gm(x) for gm in gms], [4 * x, 4 * x])

        class Functor:
            def __call__(self):
                pass

        table = cutedsl_kernel_side_table
        self.assertNotEqual(
            table.add_constant_args({"c": Functor()}),
            table.add_constant_args({"c": Functor()}),
        )

    def test_wrap_cutedsl_kernel_object_that_sets_attributes(self):
        # The compile runs a kernel object's __call__, which may set attributes on it
        # (here a cute layout, which does not hash): launch_cutedsl_kernel compiles
        # with a copy, so neither the caller's object nor the side table's snapshot
        # changes, and both keep matching the key of the kernel compiled for them.
        from torch._library.triton import set_wrap_triton_enabled

        w = wrap_cutedsl(call_mul_kernel, mutates_args=["mOut"])

        def f(x, kernel):
            out = torch.empty_like(x)
            w(kernel, x, out, x.numel())
            return out

        x = _rand(1000)
        kernel, fast_kernel = CachingMulKernel(3.0), CachingMulKernel(2.0)
        with _counting_compiles() as compile_spy:
            for k in (kernel, kernel, CachingMulKernel(3.0)):
                self.assertEqual(f(x, k), 3 * x)
            gm = make_fx(functools.partial(f, kernel=CachingMulKernel(3.0)))(x)
            self.assertEqual(gm(x), 3 * x)
            self.assertEqual(gm(x), 3 * x)
            with set_wrap_triton_enabled(False):
                self.assertEqual(f(x, fast_kernel), 2 * x)
                self.assertEqual(f(x, fast_kernel), 2 * x)
        self.assertEqual(compile_spy.call_count, 2)
        self.assertEqual(vars(fast_kernel), {"alpha": 2.0})

    def test_wrap_cutedsl_error_in_kernel_object(self):
        # A kernel object is a container like a plain object (see _object_state): one
        # holding a tensor, a reference cycle (a logging.Logger of its own manager,
        # whose loggerDict holds it) or an unhashable value (a set or a numpy array,
        # which a kernel object keyed by identity could hold) is refused, under
        # make_fx, eagerly and on the fast path, with the side table's error naming
        # the kernel and the parameter.
        from torch._library.triton import set_wrap_triton_enabled

        w = wrap_cutedsl(call_mul_kernel, mutates_args=["mOut"])

        def f(x, extra):
            kernel = MulKernel(3.0)
            kernel.extra = extra
            out = torch.empty_like(x)
            w(kernel, x, out, x.numel())
            return out

        x = _rand(1000)
        log = logging.Manager(logging.RootLogger(logging.WARNING)).getLogger("cfg")
        unhashable = (
            "is unhashable: pass a hashable value, or a list, tuple, dict, dataclass "
            "or plain object of them"
        )
        tensor = r"holds a tensor at \.extra: pass each tensor as its own argument$"
        cycle = r"refers to itself through \.manager\.loggerDict\['cfg'\]$"
        cases = (
            (x, TypeError, tensor),
            (log, ValueError, f"of type Logger {cycle}"),
            ({1, 2}, TypeError, f"of type set {unhashable}$"),
            (np.arange(3), TypeError, f"of type ndarray {unhashable}$"),
        )
        prefix = (
            "^CuTeDSL kernel call_mul_kernel: parameter 'kernel': a constant kernel "
            "argument "
        )
        for extra, error_type, error in cases:
            fn = functools.partial(f, extra=extra)
            with self.subTest(extra=type(extra).__name__):
                with self.assertRaisesRegex(error_type, prefix + error):
                    make_fx(fn, tracing_mode="fake")(x)
                with self.assertRaisesRegex(error_type, prefix + error):
                    fn(x)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(error_type, prefix + error):
                        fn(x)

    def test_wrap_cutedsl_error_unwrapped_host_fn(self):
        # inspect.unwrap(host_fn) is the user's undecorated function. Whether it
        # still carries _dsl_cls depends on whether it was compiled before (the DSL
        # deletes it then); either way it is not a @cute.jit function.
        _reference_launch(
            add_kernel, _rand(8), _rand(8), torch.zeros(8, device=DEVICE), 8
        )
        error = (
            "^CuTeDSL kernel add_kernel: wrap_cutedsl only works on host functions "
            r"decorated with @cute\.jit$"
        )
        with self.assertRaisesRegex(TypeError, error):
            wrap_cutedsl(inspect.unwrap(add_kernel), mutates_args=["mOut"])

    def test_wrap_cutedsl_error_dsl_runtime_object_arg(self):
        # PT2 makes the DSL's runtime objects itself: a cute tensor, a pointer
        # (cuda.CUdeviceptr) or a stream given as an argument is refused, through the
        # HOP or not, as one inside a constant is
        # (test_wrap_cutedsl_error_tensor_in_object).
        from torch._library.triton import set_wrap_triton_enabled

        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        error = (
            r"^CuTeDSL kernel {}: parameter '{}' got a {}: pass torch\.Tensor and "
            r"Python values; PT2 converts tensors and supplies the stream$"
        )
        cute_x, pointer = from_dlpack(x, assumed_align=16), cuda.CUdeviceptr(0)
        cases = (
            (add_w, (cute_x, y, out, 16), "mX", "_Tensor"),
            (add_scalar_w, (x, out, 16, pointer), "k", "CUdeviceptr"),
            (add_w, (x, y, out, _stream()), "n", "CUstream"),
        )
        for w, args, param, held in cases:
            message = error.format(w.kernel.__name__, param, held)
            with self.subTest(arg=held):
                for enabled in (True, False):
                    with set_wrap_triton_enabled(enabled):
                        with self.assertRaisesRegex(TypeError, message):
                            w(*args)
                # Under make_fx with fake tensors, where the launch does not run.
                with self.assertRaisesRegex(TypeError, message):
                    make_fx(lambda x: w(*args), tracing_mode="fake")(x)
        self.assertEqual(out, torch.zeros_like(x))

    def test_wrap_cutedsl_error_bad_call(self):
        # A call that does not bind, or that gives a tensor to a cutlass Numeric or a
        # Constexpr parameter, is refused with the launch's TypeError
        # (CuteDSLHostFnInfo.check_call and split_args), printed once rather than
        # chained to the check's, through the HOP or not, and under make_fx with fake
        # tensors, where the launch does not run.
        import traceback

        from torch._library.triton import set_wrap_triton_enabled

        x = _rand(16)
        out = torch.zeros_like(x)
        w, m = add_w, mul_const_w
        missing = r"add_kernel: missing arguments \['n'\]$"
        unexpected = (
            r"add_kernel: got an unexpected argument 'not_a_param'; its parameters are "
            r"\('mX', 'mY', 'mOut', 'n', 'stream'\)$"
        )
        multiple = "add_kernel: got multiple values for argument 'n'$"
        scalar = (
            r"{}: parameter '{}' takes a scalar \(it is Constexpr or annotated with a "
            r"cutlass Numeric type\), not a tensor$"
        )
        numeric = scalar.format("add_kernel", "n")
        constexpr = scalar.format("mul_const_kernel", "alpha")
        cases = (
            ("missing", lambda x, out: w(x, x, out), missing),
            ("unexpected", lambda x, out: w(x, x, out, 16, not_a_param=1), unexpected),
            ("multiple", lambda x, out: w(x, x, out, 16, n=16), multiple),
            ("tensor for n", lambda x, out: w(x, x, out, x), numeric),
            ("tensor for alpha", lambda x, out: m(x, out, 16, x), constexpr),
        )
        for case, call, error in cases:
            message = "^CuTeDSL kernel " + error
            with self.subTest(case=case):
                for enabled in (True, False):
                    with set_wrap_triton_enabled(enabled):
                        with self.assertRaisesRegex(TypeError, message) as caught:
                            call(x, out)
                    printed = "".join(traceback.format_exception(caught.exception))
                    self.assertNotIn("During handling", printed)
                with self.assertRaisesRegex(TypeError, message):
                    make_fx(call, tracing_mode="fake")(x, out)
        self.assertEqual(out, torch.zeros_like(x))

    def test_wrap_cutedsl_user_errors_go_through_raise_user_error(self):
        # Each refusal of a write set, host function or call goes through
        # CuteDSLHOPifier.raise_user_error, which the Dynamo HOPifier turns into a
        # graph break, and the exception raised is the one it is given: from
        # wrap_cutedsl (normalize_mutates_args, get_host_fn_info), from binding a
        # call (check_call and bind_call's own checks), also on the fast path, and
        # from the walk of a constant (bind_args).
        from torch._higher_order_ops.cutedsl_kernel_wrap import TracingCuteDSLHOPifier
        from torch._library.triton import set_wrap_triton_enabled

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha
                self.me = self

        def on_fast_path(call):
            with set_wrap_triton_enabled(False):
                call()

        mul_cfg_w = wrap_cutedsl(mul_cfg_kernel, mutates_args=["mOut"])
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        missing = r"^CuTeDSL kernel add_kernel: missing arguments \['n'\]$"
        cases = {
            "mutates_args None": (
                lambda: wrap_cutedsl(add_kernel, mutates_args=None),
                TypeError,
                '^CuTeDSL kernel add_kernel: mutates_args must be "unknown" or an ',
            ),
            "mutates_args ['n']": (
                lambda: wrap_cutedsl(add_kernel, mutates_args=["n"]),
                ValueError,
                "^CuTeDSL kernel add_kernel: mutates_args names 'n', but ",
            ),
            "no stream": (
                lambda: wrap_cutedsl(no_stream_kernel, mutates_args="unknown"),
                ValueError,
                "^CuTeDSL kernel no_stream_kernel: expected exactly one parameter ",
            ),
            "missing argument": (lambda: add_w(x, y, out), TypeError, missing),
            "missing argument, fast path": (
                lambda: on_fast_path(lambda: add_w(x, y, out)),
                TypeError,
                missing,
            ),
            "past the stream": (
                lambda: add_w(x, y, out, 16, _stream()),
                TypeError,
                "^CuTeDSL kernel add_kernel: takes 4 positional arguments ",
            ),
            "cyclic constant": (
                lambda: mul_cfg_w(x, out, 16, Cfg(2.0)),
                ValueError,
                r"^CuTeDSL kernel mul_cfg_kernel: parameter 'cfg': a constant kernel "
                r"argument of type Cfg refers to itself through \.me$",
            ),
        }
        real = TracingCuteDSLHOPifier.raise_user_error
        spy = mock.patch.object(
            TracingCuteDSLHOPifier, "raise_user_error", autospec=True, side_effect=real
        )
        with spy as raise_user_error:
            for case, (call, error_type, error) in cases.items():
                with self.subTest(case=case):
                    raise_user_error.reset_mock()
                    with self.assertRaisesRegex(error_type, error) as caught:
                        call()
                    raise_user_error.assert_called_once()
                    self.assertIs(raise_user_error.call_args.args[1], caught.exception)
        self.assertEqual(out, torch.zeros_like(x))

    @parametrize("tracing_mode", ["fake", "symbolic"])
    def test_wrap_cutedsl_make_fx(self, tracing_mode):
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        gm = make_fx(f, tracing_mode=tracing_mode)(_rand(1000), _rand(1000))
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        self.assertEqual(node.name, "cutedsl_kernel_wrapper_mutation_proxy")
        record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertIs(record.fn, add_kernel)
        self.assertEqual(tuple(record.mutates_args), ("mOut",))
        hop_kwargs = node.kwargs["kwargs"]
        self.assertEqual(
            [k for k, v in hop_kwargs.items() if _is_tensor_node(v)],
            ["mX", "mY", "mOut"],
        )
        self.assertEqual(hop_kwargs["mX"].op, "placeholder")
        scalars = _scalar_args(node)
        self.assertNotIn("stream", scalars)
        self.assertNotIn("stream", hop_kwargs)
        if tracing_mode == "symbolic":
            # A runtime (cutlass.Int32-annotated) scalar stays symbolic.
            self.assertIsInstance(hop_kwargs["n"], torch.fx.Node)
        else:
            self.assertEqual(scalars["n"], 1000)
            self.assertExpectedInline(
                _normalize_indices(gm.code.strip()),
                """\
def forward(self, x_1, y_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False)
    cutedsl_kernel_wrapper_mutation_proxy = torch.ops.higher_order.cutedsl_kernel_wrapper_mutation(kernel_idx = K, constant_args_idx = C, kwargs = {'mX': x_1, 'mY': y_1, 'mOut': empty_like, 'n': 1000});  x_1 = y_1 = cutedsl_kernel_wrapper_mutation_proxy = None
    return empty_like""",
            )
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(gm(x, y), x + y)
        if tracing_mode == "symbolic":
            x, y = _rand(77), _rand(77)
            self.assertEqual(gm(x, y), x + y)

    def test_wrap_cutedsl_make_fx_constexpr_specialized(self):
        # A Constexpr param is baked into the compiled kernel: a symbolic value is
        # specialized to a Python int, while the runtime scalar n stays symbolic.
        def f(x):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), x.shape[0] // 250)
            return out

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        alpha = _scalar_args(node)["alpha"]
        self.assertIs(type(alpha), int)
        self.assertEqual(alpha, 4)
        self.assertIsInstance(node.kwargs["kwargs"]["n"], torch.fx.Node)
        self.assertEqual(gm(x), 4 * x)

    def test_wrap_cutedsl_make_fx_unannotated_scalar_stays_symbolic(self):
        # The DSL converts a Python int for an unannotated parameter to a runtime
        # Int32, so a symbolic value stays a kernel argument (no specialization).
        def f(x):
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), x.shape[0] // 100)
            return out

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        self.assertIsInstance(_scalar_args(node)["k"], torch.fx.Node)
        self.assertEqual(gm(x), x + 10)
        x2 = _rand(2000)
        self.assertEqual(gm(x2), x2 + 20)

    def test_wrap_cutedsl_make_fx_defaults_resolved_at_launch(self):
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
            (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
            if scale is None:
                self.assertNotIn("scale", _scalar_args(node))
            else:
                self.assertEqual(_scalar_args(node)["scale"], scale)
            s, d = gm(x, y)
            self.assertEqual((s, d), ((x + y) * expect, (x - y) * expect))

    def test_wrap_cutedsl_make_fx_runtime_scalars_as_python_values(self):
        # A runtime scalar is captured as the Python value the launch passes
        # (python_scalar): a numpy scalar, which is not graphable, as its .item(), an
        # IntEnum member as an int, and an int for a cutlass float parameter as a
        # float. The node holds what the launch compiles for, and a numpy scalar is
        # an argument of the node, not a side-table constant.
        add_c_w = wrap_cutedsl(add_c_kernel, mutates_args=["mOut"])
        calls = (
            (add_scalar_w, "k", np.int64(5), 5),
            (add_scalar_w, "k", Factor.TWO, 2),
            (add_c_w, "c", np.float16(0.5), 0.5),
            (add_c_w, "c", 3, 3.0),
        )

        def f(x):
            outs = [torch.empty_like(x) for _ in calls]
            for out, (w, param, value, _) in zip(outs, calls):
                w(x, out, x.numel(), **{param: value})
            return outs

        x = _rand(1000)
        gm = make_fx(f)(x)
        nodes = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        for node, (_, param, _, expected) in zip(nodes, calls, strict=True):
            captured = node.kwargs["kwargs"][param]
            self.assertEqual((captured, type(captured)), (expected, type(expected)))
            idx = node.kwargs["constant_args_idx"]
            self.assertEqual(cutedsl_kernel_side_table.get_constant_args(idx), {})
        self.assertEqual(gm(x), [x + expected for *_, expected in calls])

    @parametrize("dynamic", [False, True])
    def test_wrap_cutedsl_non_strict_export(self, dynamic):
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
            "torch.ops.higher_order.cutedsl_kernel_wrapper_", ep.graph_module.code
        )
        decomposed = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.higher_order.cutedsl_kernel_wrapper_functional", 1, exactly=True
        ).run(decomposed.graph_module.code)
        self.assertEqual(ep.module()(x, y), x + y)
        self.assertEqual(decomposed.module()(x, y), x + y)
        if dynamic:
            x2, y2 = _rand(77), _rand(77)
            self.assertEqual(ep.module()(x2, y2), x2 + y2)
            self.assertEqual(decomposed.module()(x2, y2), x2 + y2)

    def test_wrap_cutedsl_non_strict_export_input_mutation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                scale_inplace_w(x, x.numel())
                return x + 1

        x = _rand(1000)
        ep = torch.export.export(M(), (x.clone(),), strict=False)
        x_in = x.clone()
        self.assertEqual(ep.module()(x_in), 2 * x + 1)
        self.assertEqual(x_in, 2 * x)

    def test_wrap_cutedsl_non_strict_export_unknown_mutates_args(self):
        # "unknown" reaches the side table as is and makes every tensor argument
        # written: each is cloned, and every user input becomes a mutated input.
        w = wrap_cutedsl(add_kernel, mutates_args="unknown")
        self.assertEqual(w.mutates_args, "unknown")

        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(M(), (x, y), strict=False).run_decompositions()
        (node,) = _hop_nodes(ep.graph_module, cutedsl_kernel_wrapper_functional)
        record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertEqual(record, CuteDSLKernel(add_kernel, "unknown"))
        self.assertEqual(sorted(node.kwargs["tensors_to_clone"]), ["mOut", "mX", "mY"])
        sig = ep.graph_signature
        mutated = sorted(sig.user_inputs_to_mutate.values())
        self.assertEqual(mutated, sorted(sig.user_inputs))
        x_in, y_in = x.clone(), y.clone()
        self.assertEqual(ep.module()(x_in, y_in), x + y)
        self.assertEqual((x_in, y_in), (x, y))

    def test_wrap_cutedsl_non_strict_export_views_and_constants(self):
        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty(x.shape[1], x.shape[0], device=x.device)
                add_one_2d_w(x.t(), out)  # non-contiguous input view
                flat = out.reshape(-1)
                o2 = torch.empty_like(flat)
                mul_const_w(flat, o2, flat.numel(), 3)
                o3 = torch.empty_like(o2)
                add_scalar_w(o2, o3, o2.numel(), 5)
                return o3

        x = _rand(8, 4)
        expected = (x.t() + 1).reshape(-1) * 3 + 5
        ep = torch.export.export(M(), (x,), strict=False)
        self.assertEqual(ep.module()(x), expected)
        self.assertEqual(ep.run_decompositions().module()(x), expected)

    def test_wrap_cutedsl_compiles_under_compile_lock(self):
        # CuTeDSL's preprocessor and library init are not thread-safe: every
        # in-process cute.compile holds CUTEDSL_COMPILE_LOCK.
        held = []

        def record_lock(*args, **kwargs):
            held.append(_compile_lock().locked())
            return mock.DEFAULT  # the spy then calls the real cute.compile

        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        with _counting_compiles() as compile_spy:
            compile_spy.side_effect = record_lock
            add_w(x, y, out, 1000)
        self.assertEqual(out, x + y)
        self.assertEqual(held, [True])

    def test_wrap_cutedsl_export_not_serializable(self):
        import io

        from torch._export.serde.serialize import SerializeError

        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(M(), (x, y), strict=False)
        # The node indexes a process-local side table: no serialized form.
        with self.assertRaises(SerializeError) as cm:
            torch.export.save(ep, io.BytesIO())
        # serialize_graph wraps any error of a node in a SerializeError of its own.
        error = cm.exception
        while error.__cause__ is not None:
            error = error.__cause__
        self.assertIsInstance(error, SerializeError)
        self.assertRegex(str(error), r"nodes \(cutedsl user kernels\) is not supported")


# Written through a global in
# CuteDSLDynamoTests.test_object_constant_modified_through_another_path_graph_breaks.
global_inner = types.SimpleNamespace(alpha=3.0)


@requires_cutedsl
class CuteDSLDynamoTests(TestCase):
    """torch.compile front end: the wrapper (global or created inline) becomes one
    HOP node, and so does a raw @cute.jit call written as for eager (from_dlpack,
    cutlass.torch.current_stream()); recompiles follow the arg classes and the
    contents of object constants."""

    hw_classification = HardwareClassification.CUDA

    def test_bad_wrap_cutedsl_calls_graph_break(self):
        # A call that eager refuses too (of a wrapper, a functools.wraps wrapper of a
        # host function, a @cute.kernel function, a kernel object or a bound @cute.jit
        # method, which are not @cute.jit functions, or with missing or extra
        # arguments) graph-breaks as a user error, whose explanation gives eager's
        # remedy for a kernel object or a bound method, and the eager call then
        # raises; a non-constant mutates_args, which eager takes, is the one call
        # Dynamo could support.
        from torch._dynamo import graph_break_hints

        user_error = graph_break_hints.USER_ERROR[0]
        supportable = graph_break_hints.SUPPORTABLE[0]

        def f(x, y, args, kwargs):
            out = torch.empty_like(x)
            wrap_cutedsl(*args, **kwargs)(x, y, out, x.numel())
            return out

        def generator(x, y):
            out = torch.empty_like(x)
            names = (name for name in ["mOut"])
            wrap_cutedsl(add_kernel, mutates_args=names)(x, y, out, x.numel())
            return out

        @functools.wraps(add_kernel)
        def decorated(*args, **kwargs):
            return add_kernel(*args, **kwargs)

        x, y = _rand(1000), _rand(1000)
        kernel_object = MulKernel(2.0)
        not_jit = "wrap_cutedsl: host_fn is not a @cute.jit function"
        improper = "wrap_cutedsl: improper arguments"
        non_constant = "wrap_cutedsl: non-constant mutates_args"
        only_jit = "only works on host functions decorated with @cute.jit"
        no_methods = "does not support bound methods or kernel objects"
        remedy = "takes the object as a cutlass.Constexpr parameter and calls"
        missing = "missing 1 required keyword-only argument: 'mutates_args'"
        two = "takes 1 positional argument but 2 were given"
        extra = "got an unexpected keyword argument 'x'"
        out_only = {"mutates_args": ["mOut"]}
        for args, kwargs, message, gb_type in (
            ((add_w,), out_only, only_jit, not_jit),
            ((decorated,), out_only, only_jit, not_jit),
            ((_add_k,), {"mutates_args": ["gOut"]}, only_jit, not_jit),
            ((kernel_object,), out_only, no_methods, not_jit),
            ((kernel_object.__call__,), out_only, no_methods, not_jit),
            ((add_kernel,), {}, missing, improper),
            ((add_kernel, ["mOut"]), {}, two, improper),
            ((add_kernel,), {**out_only, "x": 1}, extra, improper),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                torch._dynamo.reset()
                with self.assertRaisesRegex(Exception, re.escape(message)) as eager:
                    f(x, y, args, kwargs)
                with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
                    cf = torch.compile(f, backend="eager", fullgraph=True)
                    cf(x, y, args, kwargs)
                self.assertEqual(cm.exception.gb_type, gb_type)
                self.assertIn(user_error, str(cm.exception))
                self.assertNotIn(supportable, str(cm.exception))
                if gb_type == not_jit:
                    self.assertIn(remedy, " ".join(str(cm.exception).split()))
                # Without fullgraph, eager's own error.
                with self.assertRaises(type(eager.exception)) as cm:
                    torch.compile(f, backend="eager")(x, y, args, kwargs)
                self.assertEqual(str(cm.exception), str(eager.exception))
        self.assertEqual(generator(x, y), x + y)
        with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
            torch.compile(generator, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(cm.exception.gb_type, non_constant)
        self.assertIn(supportable, str(cm.exception))
        self.assertIn("Pass mutates_args as 'unknown'", str(cm.exception))
        self.assertNotIn(user_error, str(cm.exception))
        self.assertEqual(torch.compile(generator, backend="eager")(x, y), x + y)

    def test_inline_wrap_cutedsl_bad_mutates_args(self):
        # Validated as in eager, None included (it does not stand for the write set
        # of a raw call): under fullgraph the call graph-breaks as a user error with
        # eager's message, otherwise the eager wrap_cutedsl raises it: a TypeError
        # for a value that is not "unknown" or an iterable of names, a ValueError
        # for a name of a parameter that the kernel cannot write.
        from torch._dynamo import graph_break_hints

        def f(x, y, mutates_args):
            out = torch.empty_like(x)
            wrap_cutedsl(add_kernel, mutates_args=mutates_args)(x, y, out, x.numel())
            return out

        x, y = _rand(16), _rand(16)
        not_names = r"CuTeDSL kernel add_kernel: mutates_args must be .*, got "
        cannot_write = r"CuTeDSL kernel add_kernel: mutates_args names "
        for bad, error, pattern in (
            (None, TypeError, not_names + "None"),
            ("mOut", TypeError, not_names + "'mOut'"),
            (["not_a_param"], ValueError, cannot_write + "'not_a_param'"),
            (["n"], ValueError, cannot_write + "'n'"),
        ):
            with self.subTest(mutates_args=bad):
                torch._dynamo.reset()
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, pattern
                ) as cm:
                    torch.compile(f, backend="eager", fullgraph=True)(x, y, bad)
                self.assertEqual(cm.exception.gb_type, "CuTeDSL kernel call error")
                self.assertIn(graph_break_hints.USER_ERROR[0], str(cm.exception))
                with self.assertRaisesRegex(error, pattern) as cm:
                    torch.compile(f, backend="eager")(x, y, bad)
                self.assertNotIsInstance(
                    cm.exception, torch._dynamo.exc.TorchDynamoException
                )

    def test_inline_wrapper_live_across_graph_break(self):
        # A wrapper made in the frame is rebuilt at the graph break with its write
        # set, so both parts compile instead of the frame falling back to eager, and
        # the resumed graph launches the record of that write set.
        def make(mutates_args):
            def f(x, y):
                w = wrap_cutedsl(add_kernel, mutates_args=mutates_args)
                out = torch.zeros_like(x)
                torch._dynamo.graph_break()
                w(x, y, out, x.numel())
                return out

            return f

        x, y = _rand(1000), _rand(1000)
        for mutates_args, written in (
            (["mOut"], ("mOut",)),
            (("mOut",), ("mOut",)),
            ("unknown", "unknown"),
        ):
            with self.subTest(mutates_args=mutates_args):
                backend = EagerAndRecordGraphs()
                cnt = CompileCounterWithBackend(backend)
                cf = torch.compile(make(mutates_args), backend=cnt)
                self.assertEqual(cf(x, y), x + y)
                self.assertEqual(cnt.frame_count, 2)
                _, resumed = backend.graphs
                (node,) = _hop_nodes(resumed, cutedsl_kernel_wrapper_mutation)
                record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
                self.assertEqual(record, CuteDSLKernel(add_kernel, written))

    def test_wrapper_type_inside_compile(self):
        # A wrapper has its own type in the region: TraceableCuteDSLKernelWrapper, or
        # the subclass passed in. A subclass that overrides __call__ is not taken for
        # a wrapper, so its override runs.
        from torch._higher_order_ops.cutedsl_kernel_wrap import (
            TraceableCuteDSLKernelWrapper,
        )

        class SubWrapper(TraceableCuteDSLKernelWrapper):
            pass

        class Doubling(TraceableCuteDSLKernelWrapper):
            def __call__(self, x, y, out, n):
                super().__call__(x, y, out, n)
                out.mul_(2)

        def f(x):
            return x + int(isinstance(add_w, TraceableCuteDSLKernelWrapper))

        def g(x, w):
            return x + int(isinstance(w, SubWrapper) and type(w) is SubWrapper)

        def launch(w, x, y):
            out = torch.empty_like(x)
            w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), x + 1)
        cg = torch.compile(g, backend="eager", fullgraph=True)
        self.assertEqual(cg(x, SubWrapper(add_kernel, ["mOut"])), x + 1)
        doubling = Doubling(add_kernel, ["mOut"])
        self.assertEqual(launch(doubling, x, y), 2 * (x + y))
        cl = torch.compile(launch, backend="aot_eager")
        self.assertEqual(cl(doubling, x, y), 2 * (x + y))

    def test_object_constants_fullgraph(self):
        # Constants that are not Python constants are passed through the side table:
        # a config object as a snapshot, guarded on its contents, a cutlass Numeric
        # value as itself, guarded on its type and value, by which the side table
        # keys it, bare or in a config object (where Dynamo's own guards do not
        # check its type): an equal new one reuses the graph, and another value or
        # type recompiles.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def bare(x, alpha):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), alpha)
            return out

        def in_a_cfg(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        x = _rand(1000)
        for f, wrap in ((bare, lambda alpha: alpha), (in_a_cfg, Cfg)):
            with self.subTest(f=f.__name__):
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(f, backend=cnt, fullgraph=True)
                for alpha, frame_count in (
                    (cutlass.Float32(2.0), 1),
                    (cutlass.Float32(2.0), 1),
                    (cutlass.Float32(0.5), 2),
                    (cutlass.Float16(0.5), 3),
                ):
                    self.assertEqual(cf(x, wrap(alpha)), alpha.value * x)
                    self.assertEqual(cnt.frame_count, frame_count)

    def test_object_constant_modified_outside_the_region(self):
        # The side table keeps a snapshot of the object, whose contents are guarded
        # down through attributes (their names too, even ones that are not Python
        # names, and those of a dataclass that are not fields), lists and dicts (their
        # keys too): a change made between calls recompiles, so the kernel sees it as
        # in eager, while the same contents, in this object or a copy, reuse the
        # graph.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        class Override:  # a change adds an attribute
            def __init__(self, alpha):
                self.base = alpha

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return getattr(self, "override", self.base)

            @alpha.setter
            def alpha(self, alpha):
                self.override = alpha

        @dataclasses.dataclass
        class DataclassOverride:  # a change adds an attribute that is not a field
            base: float

            alpha = Override.alpha

        class Nested:
            def __init__(self, alpha):
                self.parts = [{"cfg": Cfg(alpha)}]

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return self.parts[0]["cfg"].alpha

            @alpha.setter
            def alpha(self, alpha):
                self.parts[0]["cfg"].alpha = alpha

        class Keyed:
            def __init__(self, alpha):
                self.table = {Cfg(alpha): None}

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return next(iter(self.table)).alpha

            @alpha.setter
            def alpha(self, alpha):
                next(iter(self.table)).alpha = alpha

        def namespace(alpha):  # a keyword and a dotted name, as argparse dests
            return types.SimpleNamespace(**{"alpha": alpha, "from": 0, "a.b": 0})

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        x = _rand(1000)
        makes = (Cfg, MulCfg, Override, DataclassOverride, Nested, Keyed, namespace)
        for make in makes:
            with self.subTest(make=make.__name__):
                torch._dynamo.reset()
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(f, backend=cnt, fullgraph=True)
                cfg = make(3.0)
                self.assertEqual(cf(x, cfg), 3 * x)
                self.assertEqual(cnt.frame_count, 1)
                cfg.alpha = 5.0
                self.assertEqual(cf(x, cfg), 5 * x)
                self.assertEqual(cnt.frame_count, 2)
                self.assertEqual(cf(x, cfg), 5 * x)
                if make is not Keyed:  # a copied dict key fails Dynamo's equality guard
                    self.assertEqual(cf(x, copy.deepcopy(cfg)), 5 * x)
                self.assertEqual(cnt.frame_count, 2)

    def test_unhashable_constant_graph_breaks(self):
        class Opaque:
            __slots__ = ("alpha",)
            __hash__ = None

            def __init__(self, alpha):
                self.alpha = alpha

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "unhashable"):
            torch.compile(f, backend="eager", fullgraph=True)(_rand(1000), Opaque(2.0))

    def test_object_constant_attribute_name_not_a_str_graph_breaks(self):
        # Dynamo guards an attribute only under an exact str name; eager also takes
        # a str subclass, so the call graph-breaks and runs eagerly.
        class Name(str):
            __slots__ = ()

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        x, cfg = _rand(1000), Cfg(3.0)
        vars(cfg)[Name("tag")] = 0
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "not an exact str"):
            torch.compile(f, backend="eager", fullgraph=True)(x, cfg)
        self.assertEqual(torch.compile(f, backend="aot_eager")(x, cfg), 3 * x)

    def test_object_constant_modified_in_the_region_graph_breaks(self):
        # Dynamo applies writes after the graph runs, so a kernel given an object (or
        # a class) written earlier in the region, itself or through an attribute, a
        # dict key, its class or a list its class holds, would see the old value: the
        # call graph-breaks and runs eagerly, after the write.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        def cfg_class(alpha):
            return type("CfgClass", (), {"alpha": alpha})

        def class_attribute(alpha):
            return cfg_class(alpha)()

        def class_list(alpha):  # alpha is in a list the class holds
            alpha_property = property(lambda self: self.parts[0])
            return type("ListCfg", (), {"parts": [alpha], "alpha": alpha_property})()

        class Holder:
            def __init__(self, alpha):
                self.cfg = Cfg(alpha)

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return self.cfg.alpha

        class Keyed:
            def __init__(self, alpha):
                self.table = {Cfg(alpha): None}

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return next(iter(self.table)).alpha

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def launch(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        def direct(x, cfg):
            cfg.alpha = 5.0
            return launch(x, cfg)

        def nested(x, holder):
            holder.cfg.alpha = 5.0
            return launch(x, holder)

        def through_key(x, keyed):
            next(iter(keyed.table)).alpha = 5.0
            return launch(x, keyed)

        def through_class(x, cfg):
            type(cfg).alpha = 5.0
            return launch(x, cfg)

        def through_class_list(x, cfg):
            cfg.parts[0] = 5.0
            return launch(x, cfg)

        x = _rand(1000)
        cases = (
            (direct, Cfg),
            (direct, cfg_class),
            (nested, Holder),
            (through_key, Keyed),
            (through_class, class_attribute),
            (through_class_list, class_list),
        )
        for f, make in cases:
            with self.subTest(f=f.__name__, make=make.__name__):
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported,
                    "modified earlier in the compiled region",
                ):
                    torch.compile(f, backend="eager", fullgraph=True)(x, make(3.0))
                cfg = make(3.0)
                self.assertEqual(torch.compile(f, backend="aot_eager")(x, cfg), 5 * x)
                self.assertEqual(cfg.alpha, 5.0)

    def test_object_constant_modified_through_another_path_graph_breaks(self):
        # As above, with what the kernel reads (an object the config object holds,
        # as an attribute or a dict key, or the config object's class) written
        # through another argument, a closure or a global, whichever path the region
        # reads first.
        class Inner:
            def __init__(self, alpha):
                self.alpha = alpha

        class Cfg:
            def __init__(self, inner):
                self.inner = inner

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return self.inner.alpha

        class Keyed:
            def __init__(self, inner):
                self.table = {inner: None}

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return next(iter(self.table)).alpha

        class ClassCfg:
            alpha = 3.0

        def instance(cls):
            return cls()

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])
        shared = Inner(3.0)

        def launch(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        def through_argument(x, cfg, inner):
            inner.alpha += 2.0
            return launch(x, cfg)

        def through_argument_after_cfg(x, cfg, inner):
            inner.alpha = cfg.inner.alpha + 2.0
            return launch(x, cfg)

        def through_closure(x, cfg, inner):
            shared.alpha += 2.0
            return launch(x, cfg)

        def through_global(x, cfg, inner):
            global_inner.alpha += 2.0
            return launch(x, cfg)

        x = _rand(1000)
        cases = (
            (through_argument, Cfg, shared),
            (through_argument_after_cfg, Cfg, shared),
            (through_closure, Cfg, shared),
            (through_global, Cfg, global_inner),
            (through_argument, Keyed, shared),
            (through_argument, instance, ClassCfg),
        )
        for f, make, inner in cases:
            with self.subTest(f=f.__name__, make=make.__name__):
                inner.alpha = 3.0
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported,
                    "modified earlier in the compiled region",
                ):
                    cf = torch.compile(f, backend="eager", fullgraph=True)
                    cf(x, make(inner), inner)
                cf = torch.compile(f, backend="aot_eager")
                self.assertEqual(cf(x, make(inner), inner), 5 * x)
                self.assertEqual(inner.alpha, 5.0)

    def _assert_call_error(self, f, args, message):
        """``f(*args)`` graph-breaks under fullgraph as a user error whose developer
        debug context is ``message``."""
        with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
            torch.compile(f, backend="eager", fullgraph=True)(*args)
        self.assertEqual(cm.exception.gb_type, "CuTeDSL kernel call error")
        self.assertIn(f"Developer debug context: {message}\n", str(cm.exception))

    def _walk_error(self, value):
        """The message of the error the side table's walk raises for ``value``."""
        from torch._higher_order_ops.cutedsl_kernel_wrap import CUTEDSL_CONSTANT_RULES
        from torch._higher_order_ops.user_kernel_hop import constant_key

        with self.assertRaises((TypeError, ValueError)) as cm:
            constant_key(value, CUTEDSL_CONSTANT_RULES)
        return str(cm.exception)

    def test_object_constant_holding_a_tensor_graph_breaks(self):
        # As in a tuple or a list, a tensor or a DSL object inside an object
        # constant (a dict key included) would be baked into the side table, which
        # is never reset and so also keeps it alive: the call graph-breaks, with the
        # error of the side table's walk, and a raw call runs eagerly. So does a
        # tensor or a DSL object in a tuple made in the region.
        class Cfg:
            def __init__(self, alpha, held):
                self.alpha = alpha
                self.held = held

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])
        w_spec = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def wrapped(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        def raw(x, cfg):
            out = torch.empty_like(x)
            cfg_scale_kernel(
                from_dlpack(x),
                from_dlpack(out),
                x.numel(),
                cfg,
                cutlass_torch.current_stream(),
            )
            return out

        x = _rand(1000)
        cfg = "CuTeDSL kernel cfg_scale_kernel: parameter 'cfg': "
        nested = (
            "a cute tensor, pointer or stream cannot be nested inside a constant "
            "argument"
        )
        dsl_object = "a constant kernel argument holds a {} at {}: " + nested
        for held, message in (
            (x, cfg + self._walk_error(Cfg(3.0, x))),
            ({x: None}, cfg + self._walk_error(Cfg(3.0, {x: None}))),
            (from_dlpack(x), cfg + dsl_object.format("_Tensor", ".held")),
            (_stream(), cfg + dsl_object.format("CUstream", ".held")),
        ):
            for f in (wrapped, raw):
                with self.subTest(held=type(held).__name__, f=f.__name__):
                    self._assert_call_error(f, (x, Cfg(3.0, held)), message)
        self.assertEqual(torch.compile(raw, backend="eager")(x, Cfg(3.0, x)), 3 * x)

        def in_a_tuple(x, held):
            out = torch.empty_like(x)
            w_spec(x, out, x.numel(), (3.0, held(x)))
            return out

        spec = "CuTeDSL kernel spec_scale_kernel: parameter 'spec': "
        for held, message in (
            (lambda x: x, spec + self._walk_error((3.0, x))),
            (lambda x: x + 1, spec + self._walk_error((3.0, x))),
            (from_dlpack, spec + dsl_object.format("_Tensor", "[1]")),
            (
                lambda x: cutlass_torch.current_stream(),
                spec + dsl_object.format("CUstream", "[1]"),
            ),
        ):
            with self.subTest(message=message):
                self._assert_call_error(in_a_tuple, (x, held), message)

    def test_constant_that_refers_to_itself_graph_breaks(self):
        # A constant that holds itself cannot be guarded on its contents (nor keyed
        # by them in the side table): a config object with a back-pointer, one
        # holding a logging.Logger (whose manager's loggerDict holds the logger), a
        # list or a dict holding itself, made outside the region or in it. The call
        # graph-breaks instead of recursing forever, with the error of the side
        # table's walk, and a raw call runs eagerly.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        back_pointer, with_logger = Cfg(3.0), Cfg(3.0)
        back_pointer.parent = types.SimpleNamespace(child=back_pointer)
        # A manager of its own, so that what is walked before the cycle does not
        # depend on the process's other loggers and handlers.
        manager = logging.Manager(logging.RootLogger(logging.WARNING))
        with_logger.log = manager.getLogger("cfg")
        scales = [3]
        scales.append(scales)
        spec = {"alpha": 3.0}
        spec["self"] = spec
        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])
        w_scales = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])
        w_spec = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def wrapped(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        def raw(x, cfg):
            out = torch.empty_like(x)
            cfg_scale_kernel(
                from_dlpack(x),
                from_dlpack(out),
                x.numel(),
                cfg,
                cutlass_torch.current_stream(),
            )
            return out

        def list_argument(x, scales):
            out = torch.empty_like(x)
            w_scales(x, out, x.numel(), scales)
            return out

        def list_made_in_the_region(x, scales):
            scales = [scales[0]]
            scales.append(scales)
            return list_argument(x, scales)

        def dict_argument(x, spec):
            out = torch.empty_like(x)
            w_spec(x, out, x.numel(), spec)
            return out

        def dict_made_in_the_region(x, alpha):
            spec = {"alpha": alpha}
            spec["self"] = spec
            return dict_argument(x, spec)

        x = _rand(1000)
        cfg = "CuTeDSL kernel cfg_scale_kernel: parameter 'cfg': "
        back_pointer_message = cfg + self._walk_error(back_pointer)
        self.assertTrue(back_pointer_message.endswith(".parent.child"))
        logger_message = cfg + self._walk_error(with_logger)
        list_message = "CuTeDSL kernel scale_by_first_kernel: parameter 'scales': "
        list_message += self._walk_error(scales)
        self.assertTrue(list_message.endswith("list refers to itself through [1]"))
        dict_message = "CuTeDSL kernel spec_scale_kernel: parameter 'spec': "
        dict_message += self._walk_error(spec)
        self.assertTrue(dict_message.endswith("dict refers to itself through ['self']"))
        for f, arg, message in (
            (wrapped, back_pointer, back_pointer_message),
            (raw, back_pointer, back_pointer_message),
            (wrapped, with_logger, logger_message),
            (raw, with_logger, logger_message),
            (list_argument, scales, list_message),
            (list_made_in_the_region, scales, list_message),
            (dict_argument, spec, dict_message),
            (dict_made_in_the_region, 3.0, dict_message),
        ):
            with self.subTest(f=f.__name__, arg=type(arg).__name__):
                self._assert_call_error(f, (x, arg), message)
        for cfg in (back_pointer, with_logger):
            self.assertEqual(torch.compile(raw, backend="eager")(x, cfg), 3 * x)

    def test_object_constants_inside_a_tuple(self):
        # An object inside a tuple is passed like a bare one: as a snapshot, guarded
        # on its contents. An object made in the region only exists after the graph
        # runs, so passing one graph-breaks.
        class Scale:
            def __init__(self, v):
                self.v = v

            def __int__(self):
                return self.v

        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

        def f(x, scale):
            out = torch.empty_like(x)
            w(x, out, x.numel(), (scale,))
            return out

        def g(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), (Scale(2),))
            return out

        x = _rand(1000)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        three = Scale(3)
        cases = ((three, 1), (three, 1), (Scale(4), 2), (Scale(3), 2))
        for scale, frame_count in cases:
            self.assertEqual(cf(x, scale), scale.v * x)
            self.assertEqual(cnt.frame_count, frame_count)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "parameter 'scales': cannot pass the Scale"
        ):
            torch.compile(g, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(g, backend="eager")(x), 2 * x)

    def test_function_constants(self):
        # A function from outside the region is passed as itself and guarded on its
        # identity: its own guard (on its code) also passes for another function
        # made by the same def. A @cute.jit function (e.g. a callback) is passed the
        # same way. Dynamo has no object for a bound method (one is made on each
        # access), so passing one graph-breaks.
        @cute.jit
        def jit_cfg():
            pass

        jit_cfg.alpha = 4.0

        class Holder:
            def method(self):
                pass

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        def g(x, holder):
            return f(x, holder.method)

        x = _rand(1000)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        three = _make_cfg(3.0)
        for cfg, frame_count in (
            (three, 1),
            (three, 1),
            (_make_cfg(5.0), 2),
            (jit_cfg, 3),
        ):
            self.assertEqual(cf(x, cfg), cfg.alpha * x)
            self.assertEqual(cnt.frame_count, frame_count)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "parameter 'cfg': cannot pass the method"
        ):
            torch.compile(g, backend="eager", fullgraph=True)(x, Holder())

    def test_dict_constant_changed_outside_the_region(self):
        # A dict passed in, bare or in a tuple or list argument, is guarded on its
        # keys, their order and each value, as a dict inside an object constant is,
        # with a number in it specialized, as in a tuple, once Dynamo makes it
        # symbolic on the recompile: a change of its keys, of their order or of a
        # number, or a function in it replaced by another made by the same def,
        # recompiles, so the kernel sees the new dict as in eager, while an equal
        # dict reuses the graph. A dict modified earlier in the region graph-breaks,
        # as an object does.
        def bare(spec):
            return spec

        def in_a_tuple(spec):
            return (spec,)

        def in_a_list(spec):
            return [spec]

        w = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def f(x, spec):
            out = torch.empty_like(x)
            w(x, out, x.numel(), spec)
            return out

        def modified(x, spec):
            spec["alpha"] = 5.0
            return f(x, spec)

        x, three = _rand(1000), _make_cfg(3.0)
        cases = (
            ("keys", {"alpha": 3.0}, {"alpha": 3.0, "beta": 2.0}, 6.0),
            ("key order", {"alpha": 3, "beta": 1}, {"beta": 1, "alpha": 3}, 3),
            ("number", {"alpha": 3}, {"alpha": 5}, 5),
            ("function", {"alpha": three}, {"alpha": _make_cfg(5.0)}, 5.0),
        )
        for wrap in (bare, in_a_tuple, in_a_list):
            for case, first, second, scale in cases:
                with self.subTest(wrap=wrap.__name__, case=case):
                    torch._dynamo.reset()
                    cnt = CompileCounterWithBackend("aot_eager")
                    cf = torch.compile(f, backend=cnt, fullgraph=True)
                    self.assertEqual(cf(x, wrap(first)), 3 * x)
                    self.assertEqual(cf(x, wrap(dict(first))), 3 * x)
                    self.assertEqual(cnt.frame_count, 1)
                    self.assertEqual(cf(x, wrap(second)), scale * x)
                    self.assertEqual(cnt.frame_count, 2)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "modified earlier in the compiled region"
        ):
            torch.compile(modified, backend="eager", fullgraph=True)(x, {"alpha": 3.0})
        cf = torch.compile(modified, backend="aot_eager")
        self.assertEqual(cf(x, {"alpha": 3.0}), 5 * x)

    def test_dict_constant_made_in_the_region(self):
        # A dict or an OrderedDict made in the region has no object to guard on, so
        # it is walked like a tuple or list: a value from outside the region is
        # guarded on what the side table keys it by (a function on its identity, a
        # config object on its attributes), a symbolic value is specialized, and a
        # tensor graph-breaks.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        w = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def holding(x, value, make):
            out = torch.empty_like(x)
            w(x, out, x.numel(), make(alpha=value))
            return out

        def holding_a_size(x, make):
            return holding(x, x.shape[0] // 250, make)

        for make in (dict, collections.OrderedDict):
            with self.subTest(make=make.__name__):
                torch._dynamo.reset()
                x, three, cfg = _rand(1000), _make_cfg(3.0), Cfg(3.0)
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(holding, backend=cnt, fullgraph=True)
                for fn, frame_count in ((three, 1), (three, 1), (_make_cfg(5.0), 2)):
                    self.assertEqual(cf(x, fn, make), fn.alpha * x)
                    self.assertEqual(cnt.frame_count, frame_count)
                self.assertEqual(cf(x, cfg, make), 3 * x)
                cfg.alpha = 5.0
                self.assertEqual(cf(x, cfg, make), 5 * x)
                cf = torch.compile(
                    holding_a_size, backend="aot_eager", dynamic=True, fullgraph=True
                )
                for n in (1000, 2000):
                    x = _rand(n)
                    self.assertEqual(cf(x, make), (n // 250) * x)
                message = "CuTeDSL kernel spec_scale_kernel: parameter 'spec': "
                message += self._walk_error({"alpha": x})
                self._assert_call_error(holding, (x, x, make), message)

    def test_namedtuple_constant(self):
        # A namedtuple is keyed like a tuple. One passed in reuses the graph when
        # equal, and recompiles when a number in it changes, specialized as in a
        # tuple once Dynamo makes it symbolic on the recompile. One made in the
        # region is guarded through each value from outside the region in it, so a
        # function in it replaced by another made by the same def recompiles.
        Spec = collections.namedtuple("Spec", ["a", "b"])
        w = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def f(x, spec):
            out = torch.empty_like(x)
            w(x, out, x.numel(), spec)
            return out

        def made_in_the_region(x, a):
            return f(x, Spec(a, 1))

        x = _rand(1000)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        for spec, frame_count in ((Spec(3, 1), 1), (Spec(3, 1), 1), (Spec(5, 1), 2)):
            self.assertEqual(cf(x, spec), spec.a * x)
            self.assertEqual(cnt.frame_count, frame_count)
        cf = torch.compile(made_in_the_region, backend="aot_eager", fullgraph=True)
        for alpha in (3.0, 5.0):
            self.assertEqual(cf(x, _make_cfg(alpha)), alpha * x)

    def test_frozenset_and_range_constants(self):
        # A frozenset or a range passed in is guarded on its value, by which the
        # side table keys it: an equal new one reuses the graph, and another
        # recompiles. A frozenset made in the region is guarded through each value
        # from outside the region in it, as a namedtuple is.
        w = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def f(x, spec):
            out = torch.empty_like(x)
            w(x, out, x.numel(), spec)
            return out

        def made_in_the_region(x, fn):
            return f(x, frozenset([fn]))

        x = _rand(1000)
        for first, equal, other in (
            (frozenset([3]), frozenset([3]), frozenset([5])),
            (range(3, 4), range(3, 4), range(5, 6)),
        ):
            with self.subTest(type=type(first).__name__):
                torch._dynamo.reset()
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(f, backend=cnt, fullgraph=True)
                self.assertEqual(cf(x, first), 3 * x)
                self.assertEqual(cf(x, equal), 3 * x)
                self.assertEqual(cnt.frame_count, 1)
                self.assertEqual(cf(x, other), 5 * x)
                self.assertEqual(cnt.frame_count, 2)
        cf = torch.compile(made_in_the_region, backend="aot_eager", fullgraph=True)
        for alpha in (3.0, 5.0):
            self.assertEqual(cf(x, _make_cfg(alpha)), alpha * x)

    def test_kernel_object_constant(self):
        # A kernel object (an instance whose class's __call__ is a @cute.jit host
        # function) passed as a Constexpr is guarded on its contents, as a config
        # object is: an equal new object reuses the graph, and one changed between
        # calls recompiles, as eager recompiles the kernel for it.
        w = wrap_cutedsl(call_mul_kernel, mutates_args=["mOut"])

        def f(x, kernel):
            out = torch.empty_like(x)
            w(kernel, x, out, x.numel())
            return out

        x, kernel = _rand(1000), MulKernel(3.0)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        self.assertEqual(cf(x, kernel), 3 * x)
        self.assertEqual(cf(x, MulKernel(3.0)), 3 * x)
        self.assertEqual(cnt.frame_count, 1)
        kernel.alpha = 5.0
        self.assertEqual(cf(x, kernel), 5 * x)
        self.assertEqual(cnt.frame_count, 2)

    def test_container_constants_are_guarded_as_the_side_table_reads_them(self):
        # Dynamo's guards read a dict, list or tuple through its storage, and the side
        # table reads it through its methods (see constant_contents): a class that
        # overrides one of those graph-breaks, and the call then runs eagerly, with
        # eager's result before and after a change that only that method sees. An
        # OrderedDict is guarded on its own order, so moving a key recompiles.
        class Capped(dict):  # items() caps each value at its cap
            def items(self):
                return [(k, min(v, self.cap)) for k, v in dict.items(self)]

        class First:  # reads the first value of its table
            def __init__(self, table):
                self.table = table

            @property
            def alpha(self):  # what cfg_scale_kernel reads
                return next(iter(self.table.values()))

        def launcher(host_fn):
            w = wrap_cutedsl(host_fn, mutates_args=["mOut"])

            def launch(x, arg):
                out = torch.empty_like(x)
                w(x, out, x.numel(), arg)
                return out

            return launch

        x, spec = _rand(1000), Capped(alpha=5.0)
        spec.cap = 3.0
        f = launcher(spec_scale_kernel)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "its class Capped overrides items, which"
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x, spec)
        cf = torch.compile(f, backend="aot_eager")
        self.assertEqual(cf(x, spec), 3 * x)
        spec.cap = 10.0
        self.assertEqual(cf(x, spec), 5 * x)
        cfg = First(collections.OrderedDict(a=3.0, b=5.0))
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(launcher(cfg_scale_kernel), backend=cnt, fullgraph=True)
        self.assertEqual(cf(x, cfg), 3 * x)
        cfg.table.move_to_end("a")
        self.assertEqual(cf(x, cfg), 5 * x)
        self.assertEqual(cnt.frame_count, 2)

    def test_constants_keyed_by_equality(self):
        # An int, float or str subclass is keyed by equality, as an int, a float or a
        # str is, and guarded on its type and value: an equal new one, such as one
        # built for each call, reuses the graph, while another value or another type
        # recompiles.
        class Alpha(int):
            __slots__ = ()

        class OtherAlpha(int):
            __slots__ = ()

        class Scale(float):
            __slots__ = ()

        class Name(str):
            __slots__ = ()

        w = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def scaled(x, alpha):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), alpha)
            return out

        def keyed(x, spec):
            out = torch.empty_like(x)
            w(x, out, x.numel(), spec)
            return out

        x = _rand(1000)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(scaled, backend=cnt, fullgraph=True)
        for alpha, frame_count in (
            (Alpha(3), 1),
            (Alpha(3), 1),
            (Alpha(5), 2),
            (OtherAlpha(5), 3),
            (Scale(5.0), 4),
            (Scale(5.0), 4),
        ):
            self.assertEqual(cf(x, alpha), float(alpha) * x)
            self.assertEqual(cnt.frame_count, frame_count)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(keyed, backend=cnt, fullgraph=True)
        for name, frame_count in (("alpha", 1), ("alpha", 1), ("beta", 2)):
            self.assertEqual(cf(x, {Name(name): 3.0}), 3 * x)
            self.assertEqual(cnt.frame_count, frame_count)

    def test_constants_guarded_by_identity_stay_alive(self):
        # A Fraction or a Decimal is keyed by equality, which no Dynamo guard
        # checks for it, so it is guarded on its identity, and an equal new one
        # recompiles. It cannot be weakly referenced, and the side table may hold an
        # equal one instead of it: the compiled code keeps it alive, so that its id
        # cannot be taken by a new value of another value, which would pass the
        # guard and launch the old snapshot.
        import decimal
        import fractions

        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

        def scaled(x, scales):
            out = torch.empty_like(x)
            w(x, out, x.numel(), scales)
            return out

        x = _rand(1000)
        for make in (fractions.Fraction, decimal.Decimal):
            with self.subTest(make=make.__name__):
                torch._dynamo.reset()
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(scaled, backend=cnt, fullgraph=True)
                values = (3, 3, 5, 5, 7, 3, 9)
                for v in values:
                    self.assertEqual(cf(x, (make(v),)), v * x)
                self.assertEqual(cnt.frame_count, len(values))

    def test_numpy_scalar_arguments(self):
        # Dynamo traces a numpy scalar as a 0-d array backed by a tensor. One made
        # outside the region is passed as itself, as in eager (the launch converts
        # one for a runtime scalar parameter), and guarded on its type and value: an
        # equal new one, such as one built for each call, reuses the graph, while
        # another value or another type recompiles (for a runtime scalar parameter
        # too, where eager compiles once), in a wrapper or a raw call, bare, in a
        # config object or in a tuple. A numpy scalar made in the region
        # graph-breaks.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])
        w_scales = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

        def constexpr(x, alpha):
            out = torch.empty_like(x)
            mul_const_w(x, out, x.numel(), alpha)
            return out

        def raw_constexpr(x, alpha):
            out = torch.empty_like(x)
            mul_const_kernel(
                from_dlpack(x),
                from_dlpack(out),
                x.numel(),
                alpha,
                cutlass_torch.current_stream(),
            )
            return out

        def runtime(x, k):
            out = torch.empty_like(x)
            add_scalar_w(x, out, x.numel(), k)
            return out

        def in_cfg(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        def in_tuple(x, scales):
            out = torch.empty_like(x)
            w_scales(x, out, x.numel(), scales)
            return out

        def made_in_the_region(x):
            return constexpr(x, np.float32(2.0))

        x = _rand(1000)
        calls = ((np.int64(3), 1), (np.int64(3), 1), (np.int64(5), 2), (np.int32(5), 3))
        for f, fullgraph in (
            (constexpr, True),
            (raw_constexpr, True),
            (raw_constexpr, False),
            (runtime, True),
        ):
            with self.subTest(f=f.__name__, fullgraph=fullgraph):
                torch._dynamo.reset()
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(f, backend=cnt, fullgraph=fullgraph)
                for value, frame_count in calls:
                    expected = x + float(value) if f is runtime else float(value) * x
                    self.assertEqual(cf(x, value), expected)
                    self.assertEqual(cnt.frame_count, frame_count)
        for f, make in (
            (in_cfg, lambda v: Cfg(np.float32(v))),
            (in_tuple, lambda v: (np.int64(v),)),
        ):
            with self.subTest(f=f.__name__):
                cnt = CompileCounterWithBackend("aot_eager")
                cf = torch.compile(f, backend=cnt, fullgraph=True)
                for v in (3, 3, 5, 5, 7, 3, 9):
                    self.assertEqual(cf(x, make(v)), v * x)
                self.assertEqual(cnt.frame_count, 4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"parameter 'alpha': cannot pass a numpy array, or a numpy scalar made in "
            r"the compiled region",
        ):
            torch.compile(made_in_the_region, backend="eager", fullgraph=True)(x)

    def test_constant_made_in_the_region_guards_each_value_once(self):
        # A Python constant made in the region is guarded through each value from
        # outside the region in it, together with what that value holds: a tuple
        # argument in a namedtuple guards the function in it once, not again through
        # the tuple.
        from torch._dynamo.variables.cutedsl import DynamoCuteDSLHOPifier

        Spec = collections.namedtuple("Spec", ["fns", "b"])
        w = wrap_cutedsl(spec_scale_kernel, mutates_args=["mOut"])

        def f(x, fns):
            out = torch.empty_like(x)
            w(x, out, x.numel(), Spec(fns, 1))
            return out

        x = _rand(1000)
        guard_contents = DynamoCuteDSLHOPifier.guard_contents
        with mock.patch.object(
            DynamoCuteDSLHOPifier,
            "guard_contents",
            autospec=True,
            side_effect=guard_contents,
        ) as spy:
            cf = torch.compile(f, backend="aot_eager", fullgraph=True)
            self.assertEqual(cf(x, (_make_cfg(3.0),)), 3 * x)
        sources = [call.args[1].name for call in spy.call_args_list]
        self.assertEqual(sources.count("L['fns'][0]"), 1, sources)

    def test_nested_symint_in_constexpr_tuple(self):
        # Specialized like a bare SymInt: one graph per value.
        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

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

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(x, y), (x + y).relu())
        self.assertEqual(cnt.frame_count, 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_inline_wrap_cutedsl_fullgraph(self, backend):
        def f(x, y):
            out = torch.empty_like(x)
            wrap_cutedsl(add_kernel, mutates_args=["mOut"])(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        compiled = torch.compile(f, backend=backend, fullgraph=True)
        self.assertEqual(compiled(x, y), x + y)

    def test_inline_wrap_cutedsl_of_a_kernel_read_without_a_source(self):
        # A @cute.jit function that Dynamo reads without a source, here a staticmethod
        # of a module made in the region, is taken by wrap_cutedsl as in eager.
        class M(torch.nn.Module):
            kernel = staticmethod(add_kernel)

            def forward(self, x, y):
                out = torch.empty_like(x)
                wrap_cutedsl(self.kernel, mutates_args=["mOut"])(x, y, out, x.numel())
                return out

        def f(x, y):
            return M()(x, y)

        def live_across_graph_break(x, y):
            w = wrap_cutedsl(M().kernel, mutates_args=["mOut"])
            out = torch.empty_like(x)
            torch._dynamo.graph_break()
            w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        backend = EagerAndRecordGraphs()
        self.assertEqual(torch.compile(f, backend=backend, fullgraph=True)(x, y), x + y)
        (gm,) = backend.graphs
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertIs(record.fn, add_kernel)
        self.assertEqual(record.mutates_args, ("mOut",))
        # Dynamo cannot rebuild such a function at a graph break, nor the wrapper, so
        # the frame runs eagerly.
        cf = torch.compile(live_across_graph_break, backend="eager")
        self.assertEqual(cf(x, y), x + y)

    def test_raw_call_of_functions_read_without_a_source(self):
        # A raw call is captured when Dynamo reads the @cute.jit function,
        # from_dlpack and current_stream without a source, here as staticmethods of
        # a module made in the region; a @cute.jit function read that way and passed
        # as a constant argument is passed as itself.
        @cute.jit
        def jit_cfg():
            pass

        jit_cfg.alpha = 4.0

        class M(torch.nn.Module):
            kernel = staticmethod(add_kernel)
            to_cute = staticmethod(from_dlpack)
            stream = staticmethod(cutlass_torch.current_stream)
            cfg = staticmethod(jit_cfg)

            def forward(self, x, y):
                out, scaled = torch.empty_like(x), torch.empty_like(x)
                mx, n = self.to_cute(x), x.numel()
                self.kernel(mx, self.to_cute(y), self.to_cute(out), n, self.stream())
                cfg_scale_kernel(mx, self.to_cute(scaled), n, self.cfg, self.stream())
                return out, scaled

        def f(x, y):
            return M()(x, y)

        x, y = _rand(1000), _rand(1000)
        backend = EagerAndRecordGraphs()
        cf = torch.compile(f, backend=backend, fullgraph=True)
        self.assertEqual(cf(x, y), (x + y, 4 * x))
        (gm,) = backend.graphs
        add_node, scale_node = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        record = cutedsl_kernel_side_table.get_kernel(add_node.kwargs["kernel_idx"])
        self.assertEqual(record, CuteDSLKernel(add_kernel, "unknown"))
        constant_args_idx = scale_node.kwargs["constant_args_idx"]
        constant_args = cutedsl_kernel_side_table.get_constant_args(constant_args_idx)
        self.assertIs(constant_args["cfg"], jit_cfg)

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
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertIs(record.fn, add_kernel)
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
        self.assertEqual(_hop_nodes(fw, cutedsl_kernel_wrapper_mutation), [])
        (node,) = _hop_nodes(fw, cutedsl_kernel_wrapper_functional)
        self.assertEqual(node.kwargs["tensors_to_clone"], ["mOut"])

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
        (node,) = _hop_nodes(backend.fw_graphs[0], cutedsl_kernel_wrapper_functional)
        self.assertEqual(list(node.kwargs["tensor_bases"]), ["mX"])

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("grad_fn", [torch.no_grad, torch.enable_grad])
    def test_wrapper_with_grad_option(self, backend, grad_fn):
        # Mirrors test_triton_kernel_with_grad_option.
        def f(x):
            with grad_fn():
                out = torch.empty_like(x)
                mul_const_w(x, out, x.numel(), 2)
                return out

        x = _rand(1000)
        self.assertEqual(torch.compile(f, backend=backend, fullgraph=True)(x), 2 * x)

    def test_wrapper_argument_guards(self):
        # A wrapper passed in is guarded on its type, kernel and write set, not on
        # its identity: an equal wrapper reuses the compiled code, while another
        # write set, another kernel of the same name, or another wrapper type
        # recompiles into a node for it.
        from torch._higher_order_ops.cutedsl_kernel_wrap import (
            TraceableCuteDSLKernelWrapper,
        )

        class SubWrapper(TraceableCuteDSLKernelWrapper):
            pass

        backend = AotEagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(w, x, y):
            out = torch.zeros_like(x)
            w(x, y, out, x.numel())
            return out

        add_out = wrap_cutedsl(same_name_add_kernel, mutates_args=["mOut"])
        add_out_again = wrap_cutedsl(same_name_add_kernel, mutates_args=("mOut",))
        add_unknown = wrap_cutedsl(same_name_add_kernel, mutates_args="unknown")
        sub_out = wrap_cutedsl(same_name_sub_kernel, mutates_args=["mOut"])
        add_out_sub = SubWrapper(same_name_add_kernel, ["mOut"])
        x, y = _rand(1000), _rand(1000)
        for w, expected, frame_count in (
            (add_out, x + y, 1),
            (add_out_again, x + y, 1),
            (add_unknown, x + y, 2),
            (sub_out, x - y, 3),
            (add_out_sub, x + y, 4),
        ):
            self.assertEqual(f(w, x, y), expected)
            self.assertEqual(cnt.frame_count, frame_count)
        self.assertEqual(len(backend.graphs), 4)
        self.assertEqual(len(backend.fw_graphs), 4)
        for gm, fw, w, cloned in zip(
            backend.graphs,
            backend.fw_graphs,
            (add_out, add_unknown, sub_out, add_out_sub),
            (["mOut"], ["mOut", "mX", "mY"], ["mOut"], ["mOut"]),
        ):
            (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
            self.assertEqual(node.kwargs["kernel_idx"], w.kernel_idx)
            (fn_node,) = _hop_nodes(fw, cutedsl_kernel_wrapper_functional)
            self.assertEqual(sorted(fn_node.kwargs["tensors_to_clone"]), cloned)

    def test_runtime_scalar_stays_dynamic(self):
        # n is annotated cutlass.Int32: a runtime arg, so a new size does not
        # recompile under dynamic=True.
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
        # A Python int for an unannotated parameter is a runtime Int32 in the DSL:
        # a symbolic value stays a graph input and new values do not recompile.
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

    def test_toy_raw_call(self):
        # The toy: Dynamo captures the raw call as one HOP node, as the call of
        # wrap_cutedsl(toy_add_kernel) with the write set of a raw call. A new size
        # does not recompile: n is a runtime Int32.
        x, y = _rand(2048), _rand(2048)
        self.assertEqual(toy(x, y), (x + y).relu())
        backend = AotEagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        cf = torch.compile(toy, backend=cnt, fullgraph=True, dynamic=True)
        for n in (2048, 1000):
            x, y = _rand(n), _rand(n)
            self.assertEqual(cf(x, y), (x + y).relu())
        self.assertEqual(cnt.frame_count, 1)
        (node,) = _hop_nodes(backend.graphs[0], cutedsl_kernel_wrapper_mutation)
        self.assertEqual(sorted(node.kwargs["kwargs"]), ["mOut", "mX", "mY", "n"])
        kernel = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertIs(kernel.fn, toy_add_kernel)
        self.assertEqual(kernel.mutates_args, "unknown")
        (fn_node,) = _hop_nodes(backend.fw_graphs[0], cutedsl_kernel_wrapper_functional)
        self.assertEqual(
            sorted(fn_node.kwargs["tensors_to_clone"]), ["mOut", "mX", "mY"]
        )

    def test_raw_call_guards_the_host_function(self):
        # The @cute.jit function of a raw call, passed in, is guarded on its
        # identity: another host function of the same name recompiles into a node
        # for it instead of launching the first one.
        def f(kernel, x, y):
            out = torch.empty_like(x)
            stream = cutlass_torch.current_stream()
            kernel(from_dlpack(x), from_dlpack(y), from_dlpack(out), x.numel(), stream)
            return out

        x, y = _rand(1000), _rand(1000)
        cnt = CompileCounterWithBackend("aot_eager")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        for kernel, expected, frame_count in (
            (same_name_add_kernel, x + y, 1),
            (same_name_sub_kernel, x - y, 2),
            (same_name_add_kernel, x + y, 2),
        ):
            self.assertEqual(cf(kernel, x, y), expected)
            self.assertEqual(cnt.frame_count, frame_count)

    def test_raw_call_keywords_defaults_and_constexpr(self):
        # A raw call binds like a wrapper call: keywords, a default after the stream,
        # a Constexpr baked into the compiled kernel.
        stream = cutlass_torch.current_stream

        def f(x, y):
            s1, d1, s2, d2, m = [torch.empty_like(x) for _ in range(5)]
            n = x.numel()
            add_sub_kernel(
                from_dlpack(x),
                from_dlpack(y),
                from_dlpack(s1),
                from_dlpack(d1),
                n,
                stream(),
            )
            add_sub_kernel(
                mX=from_dlpack(x),
                mY=from_dlpack(y),
                mSum=from_dlpack(s2),
                mDiff=from_dlpack(d2),
                n=n,
                stream=stream(),
                scale=0.25,
            )
            mul_const_kernel(from_dlpack(x), from_dlpack(m), n, 3.0, stream())
            return s1, d1, s2, d2, m

        x, y = _rand(1000), _rand(1000)
        expected = (x + y, x - y, (x + y) * 0.25, (x - y) * 0.25, 3 * x)
        self.assertEqual(f(x, y), expected)
        cf = torch.compile(f, backend="aot_eager", fullgraph=True)
        self.assertEqual(cf(x, y), expected)

    def test_raw_call_from_dlpack_options(self):
        # assumed_align is a hint PT2 replaces with the alignment it compiles for (each
        # pointer's own, at most 16 bytes), and has to be a constant; an option that
        # changes the compiled kernel graph-breaks.
        def f(x, y):
            out = torch.empty_like(x)
            add_kernel(
                from_dlpack(x, assumed_align=16),
                from_dlpack(y, 4),
                from_dlpack(out, use_32bit_stride=False),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        def g(x, y):
            out = torch.empty_like(x)
            add_kernel(
                from_dlpack(x, use_32bit_stride=True),
                from_dlpack(y),
                from_dlpack(out),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        def h(x, y, align):
            out = torch.empty_like(x)
            add_kernel(
                from_dlpack(x, assumed_align=align),
                from_dlpack(y),
                from_dlpack(out),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        x, y = _rand(1000), _rand(1000)
        cf = torch.compile(f, backend="aot_eager", fullgraph=True)
        self.assertEqual(cf(x, y), x + y)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "use_32bit_stride"):
            torch.compile(g, backend="eager", fullgraph=True)(x, y)
        # A dynamic int, so a symbolic value in the region.
        ch = torch.compile(h, backend="eager", fullgraph=True, dynamic=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "assumed_align must be a constant"
        ):
            ch(x, y, 16)

    def test_raw_call_stream_made_outside_graph_breaks(self):
        # Only cutlass.torch.current_stream() called in the region can be the stream:
        # the captured kernel is launched on the current stream.
        def f(x, y, stream):
            out = torch.empty_like(x)
            add_kernel(
                from_dlpack(x), from_dlpack(y), from_dlpack(out), x.numel(), stream
            )
            return out

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, r"pass cutlass\.torch\.current_stream\(\)"
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x, y, _stream())

    def test_raw_call_torch_tensor_graph_breaks(self):
        # Eager CuTeDSL refuses a torch.Tensor for a cute.Tensor param, and so does
        # the capture of a raw call, as a user error.
        from cutlass.base_dsl.common import DSLBaseError

        def f(x, y):
            out = torch.empty_like(x)
            add_kernel(x, y, out, x.numel(), cutlass_torch.current_stream())
            return out

        x, y = _rand(1000), _rand(1000)
        eager = r"expects argument #1 \(mX\) to be .*, but got .*torch\.Tensor"
        with self.assertRaisesRegex(DSLBaseError, eager):
            f(x, y)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"parameter 'mX': pass from_dlpack\(tensor\), not a torch\.Tensor",
        ) as cm:
            torch.compile(f, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(cm.exception.gb_type, "CuTeDSL kernel call error")

    def test_raw_call_graph_break_messages(self):
        # Each refusal of a raw call graph-breaks with its own message: as a user
        # error where eager CuTeDSL refuses the call too (with the error given), and
        # as an unsupported feature where only Dynamo cannot capture it (an
        # assumed_align that eager takes, a value other than a tensor for
        # from_dlpack, any read of a cute tensor or a stream made in the region).
        from cutlass.base_dsl.common import DSLBaseError
        from torch._dynamo import graph_break_hints

        stream = cutlass_torch.current_stream

        def cute(x, y):
            return from_dlpack(x), from_dlpack(y), from_dlpack(torch.empty_like(x))

        def aligned(x, y, align):
            return from_dlpack(x, assumed_align=align), *cute(x, y)[1:]

        n = 1000
        no_stream = (
            "Required argument `stream` is missing"
            "|Missing required argument in `add_kernel`: 'stream'"
        )
        no_n = (
            "Required argument `n` is missing"
            "|Missing required argument in `add_kernel`: 'n'"
        )
        cute_read = "a cute tensor made in the compiled region cannot be read there"
        stream_read = "the stream cannot be read in the compiled region"
        cases = (
            (
                lambda x, y: add_kernel(*cute(x, y), n, stream(), 1),
                (DSLBaseError, "too many positional arguments"),
                "CuTeDSL kernel add_kernel: takes 5 positional arguments but got 6",
            ),
            (
                lambda x, y: add_kernel(*cute(x, y), n, stream(), bogus=1),
                (DSLBaseError, "unexpected keyword argument 'bogus'"),
                "CuTeDSL kernel add_kernel: got an unexpected keyword argument 'bogus'",
            ),
            (
                lambda x, y: add_kernel(*cute(x, y), n, stream(), mX=from_dlpack(x)),
                (DSLBaseError, "multiple values for argument 'mX'"),
                "CuTeDSL kernel add_kernel: got multiple values for argument 'mX'",
            ),
            (
                lambda x, y: add_kernel(*cute(x, y), n),
                (DSLBaseError, no_stream),
                "CuTeDSL kernel add_kernel: missing arguments ['stream']",
            ),
            (
                lambda x, y: add_kernel(*cute(x, y)),
                (DSLBaseError, no_n),
                "CuTeDSL kernel add_kernel: missing arguments ['n', 'stream']",
            ),
            (
                lambda x, y: add_kernel(*cute(x, y), n, stream(1)),
                (TypeError, r"takes 0 positional arguments but 1 was given"),
                "cutlass.torch.current_stream() takes no arguments",
            ),
            (
                lambda x, y: from_dlpack(),
                (TypeError, "missing 1 required positional argument: 'tensor_dlpack'"),
                "from_dlpack(): missing a required argument: 'tensor_dlpack'",
            ),
            (
                lambda x, y: from_dlpack(x, bogus=1),
                (TypeError, "unexpected keyword argument 'bogus'"),
                "from_dlpack(): got an unexpected keyword argument 'bogus'",
            ),
            (
                lambda x, y: add_kernel(*aligned(x, y, 16.0), n, stream()),
                (TypeError, r"build_memref_desc\(\): incompatible function arguments"),
                "from_dlpack(): assumed_align must be an int, got 16.0",
            ),
            (
                lambda x, y: add_kernel(*aligned(x, y, 32), n, stream()),
                None,
                "from_dlpack(): assumed_align must be a power of two up to 16",
            ),
            (
                lambda x, y: add_kernel(*aligned(x, y, 3), n, stream()),
                None,
                "from_dlpack(): assumed_align must be a power of two up to 16",
            ),
            (
                lambda x, y: from_dlpack(3),
                None,
                "from_dlpack() of a value of type int: only a torch.Tensor is captured",
            ),
            (
                lambda x, y: from_dlpack(x).element_type,
                None,
                f"from_dlpack(...).element_type: {cute_read}",
            ),
            (
                lambda x, y: from_dlpack(x).mark_layout_dynamic(),
                None,
                f"from_dlpack(...).mark_layout_dynamic: {cute_read}",
            ),
            (
                lambda x, y: from_dlpack(x) == 1,
                None,
                f"from_dlpack(...).__eq__(): {cute_read}",
            ),
            (
                lambda x, y: hasattr(from_dlpack(x), "shape"),
                None,
                f"hasattr(from_dlpack(...), 'shape'): {cute_read}",
            ),
            (
                lambda x, y: stream().getPtr,
                None,
                f"cutlass.torch.current_stream().getPtr: {stream_read}",
            ),
            (
                lambda x, y: stream().getPtr(),
                None,
                f"cutlass.torch.current_stream().getPtr: {stream_read}",
            ),
            (
                lambda x, y: stream() == 1,
                None,
                f"cutlass.torch.current_stream().__eq__(): {stream_read}",
            ),
            (
                lambda x, y: hasattr(stream(), "getPtr"),
                None,
                f"hasattr(cutlass.torch.current_stream(), 'getPtr'): {stream_read}",
            ),
        )
        x, y = _rand(n), _rand(n)
        user_error = graph_break_hints.USER_ERROR[0]
        for f, eager, message in cases:
            with self.subTest(message=message):
                torch._dynamo.reset()
                if eager is not None:
                    with self.assertRaisesRegex(*eager):
                        f(x, y)
                with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
                    torch.compile(f, backend="eager", fullgraph=True)(x, y)
                if eager is None:
                    gb_type = "CuTeDSL kernel unsupported feature"
                else:
                    gb_type = "CuTeDSL kernel call error"
                self.assertEqual(cm.exception.gb_type, gb_type)
                self.assertIn(f"Developer debug context: {message}", str(cm.exception))
                self.assertEqual(user_error in str(cm.exception), eager is not None)

    def test_raw_call_arguments_live_across_graph_break(self):
        # from_dlpack(t) and current_stream() made before a graph break are made
        # again after it. The resumed frame then sees DSL objects made outside its
        # region, so the call graph-breaks there and runs eagerly. The reason is the
        # stream's, which is checked first, and only comes from tracing the rebuilt
        # stream.
        def f(x, y):
            out = torch.zeros_like(x)
            args = (from_dlpack(x), from_dlpack(y, assumed_align=16), from_dlpack(out))
            stream = cutlass_torch.current_stream()
            torch._dynamo.graph_break()
            add_kernel(*args, x.numel(), stream)
            return out, args[1]

        x, y = _rand(1000), _rand(1000)
        torch._dynamo.utils.counters.clear()
        out, ty = torch.compile(f, backend="eager")(x, y)
        self.assertEqual(out, x + y)
        self.assertEqual(ty._assumed_align, 16)
        reasons = list(torch._dynamo.utils.counters["graph_break"])
        stream_reason = "pass cutlass.torch.current_stream(), read on the stream"
        self.assertTrue(any(stream_reason in r for r in reasons), reasons)
        failures = list(torch._dynamo.utils.counters["unimplemented"])
        self.assertFalse(any("Reconstruction failure" in r for r in failures), failures)

    def test_raw_call_stream_is_the_one_read(self):
        # cutlass.torch.current_stream() stands for the stream current where it is
        # called. A raw call on another stream graph-breaks (the captured kernel would
        # launch on the call's stream), and a value that leaves the graph is the stream
        # it was read on, not the one current at the graph break.
        side = torch.cuda.Stream()

        def read_outside_call_inside(x, y):
            out = torch.empty_like(x)
            stream = cutlass_torch.current_stream()
            with torch.cuda.stream(side):
                add_kernel(
                    from_dlpack(x), from_dlpack(y), from_dlpack(out), x.numel(), stream
                )
            return out

        def read_inside(x):
            with torch.cuda.stream(side):
                stream = cutlass_torch.current_stream()
            return x + 1, stream

        x, y = _rand(1000), _rand(1000)
        cf = torch.compile(read_outside_call_inside, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, r"read on the stream current at this call"
        ):
            cf(x, y)
        _, stream = torch.compile(read_inside, backend="eager", fullgraph=True)(x)
        self.assertEqual(int(stream), side.cuda_stream)

    @requires_cuda_and_triton
    @parametrize("backend", ["eager", "aot_eager"])
    def test_raw_call_on_a_side_stream(self, backend):
        # With the stream read and the call inside torch.cuda.stream(side), the raw
        # call is captured under fullgraph, and its node carries the region's stream
        # annotation, as the node of a user Triton kernel launched there does. A
        # backend that runs each node on its annotated stream, as Inductor does,
        # launches the kernel on side. The eager and aot_eager backends ignore the
        # annotations and run every node, aten ops and Triton kernels too, on the
        # current stream, so this does not check which stream the kernel runs on.
        import triton

        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional as triton_functional,
            triton_kernel_wrapper_mutation as triton_mutation,
        )
        from torch.testing._internal.triton_utils import add_kernel as triton_add

        side = torch.cuda.Stream()

        def f(x, y):
            out, ref = torch.empty_like(x), torch.empty_like(x)
            n = x.numel()
            with torch.cuda.stream(side):
                add_kernel(
                    from_dlpack(x),
                    from_dlpack(y),
                    from_dlpack(out),
                    n,
                    cutlass_torch.current_stream(),
                )
                triton_add[(triton.cdiv(n, 128),)](x, y, ref, n, BLOCK_SIZE=128)
            torch.cuda.current_stream().wait_stream(side)
            return out, ref

        if backend == "eager":
            record = EagerAndRecordGraphs()
        else:
            record = AotEagerAndRecordGraphs()
        x, y = _rand(1000), _rand(1000)
        torch.cuda.synchronize()  # the region reads x and y without waiting for them
        out, ref = torch.compile(f, backend=record, fullgraph=True)(x, y)
        self.assertEqual((out, ref), (x + y, x + y))
        (gm,) = record.graphs
        hops = [(gm, cutedsl_kernel_wrapper_mutation, triton_mutation)]
        if backend == "aot_eager":
            (fw,) = record.fw_graphs
            hops.append((fw, cutedsl_kernel_wrapper_functional, triton_functional))
        for gm, hop, triton_hop in hops:
            (node,) = _hop_nodes(gm, hop)
            (triton_node,) = _hop_nodes(gm, triton_hop)
            self.assertIn("stream", triton_node.meta["custom"])
            self.assertEqual(node.meta.get("custom"), triton_node.meta["custom"])

    def test_raw_call_cute_tensor_attributes_graph_break(self):
        # A from_dlpack(t) made in the region cannot be read there: reading the cute
        # tensor (its element type, shape, ...) graph-breaks and runs eagerly.
        def f(x, y):
            out = torch.zeros_like(x)
            mx = from_dlpack(x)
            if mx.element_type is cutlass.Float32 and mx.shape == (1000,):
                add_kernel(
                    mx,
                    from_dlpack(y),
                    from_dlpack(out),
                    x.numel(),
                    cutlass_torch.current_stream(),
                )
            else:
                out.fill_(-1.0)
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(x, y), x + y)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"from_dlpack\(\.\.\.\)\.element_type: a cute",
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)
        torch._dynamo.reset()  # the compiled code is cached on f's code object
        self.assertEqual(torch.compile(f, backend="eager")(x, y), x + y)

    def test_raw_call_values_have_no_repr_in_the_region(self):
        # repr() and str() of a from_dlpack(t) or current_stream() value made in the
        # region graph-break like any other read, and then run on the real object, as
        # in eager. An f-string of one is formatted after the graph, from the real
        # object made there.
        def f(x):
            return x + 1, str(from_dlpack(x)), str(cutlass_torch.current_stream())

        def formatted(x):
            return x + 1, f"{from_dlpack(x)}", f"{cutlass_torch.current_stream()}"

        x = _rand(8)
        for g in (
            lambda x: str(from_dlpack(x)),
            lambda x: repr(cutlass_torch.current_stream()),
        ):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "cannot be read"
            ):
                torch.compile(g, backend="eager", fullgraph=True)(x)
        for g, fullgraph in ((f, False), (formatted, True)):
            with self.subTest(g=g.__name__):
                cg = torch.compile(g, backend="eager", fullgraph=fullgraph)
                _, tensor_text, stream_text = cg(x)
                self.assertEqual(tensor_text, str(from_dlpack(x)))
                self.assertTrue(stream_text.startswith("<CUstream"), stream_text)

    def test_raw_call_from_dlpack_requires_grad_graph_breaks(self):
        # Eager from_dlpack refuses a tensor that requires grad, and so does the
        # capture; from_dlpack(t.detach()) is captured.
        def f(x, y):
            out = torch.empty_like(x)
            add_kernel(
                from_dlpack(x),
                from_dlpack(y),
                from_dlpack(out),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        def g(x, y):
            out = torch.empty_like(x)
            add_kernel(
                from_dlpack(x.detach()),
                from_dlpack(y),
                from_dlpack(out),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        x, y = _rand(1000, requires_grad=True), _rand(1000)
        with self.assertRaisesRegex(BufferError, "Can't export tensors that require"):
            f(x, y)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"from_dlpack\(\) of a tensor that requires grad",
        ) as cm:
            torch.compile(f, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(cm.exception.gb_type, "CuTeDSL kernel call error")
        cg = torch.compile(g, backend="aot_eager", fullgraph=True)
        self.assertEqual(cg(x, y), (x + y).detach())

    def test_raw_call_replaced_from_dlpack_is_traced(self):
        # A functools.wraps wrapper of from_dlpack (such as one stored under its
        # name) has its __module__ and __qualname__ but is not taken for it: Dynamo
        # inlines it like any function, so its body runs (it records each call), and
        # captures the from_dlpack it calls.
        from torch._library.cutedsl import is_cutedsl_from_dlpack

        calls = []

        @functools.wraps(from_dlpack)
        def logged(tensor, **kwargs):
            calls.append(tensor.shape)
            return from_dlpack(tensor, **kwargs)

        self.assertTrue(is_cutedsl_from_dlpack(from_dlpack))
        self.assertFalse(is_cutedsl_from_dlpack(logged))

        def f(x, y):
            out = torch.zeros_like(x)
            add_kernel(
                logged(x),
                logged(y),
                logged(out),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        x, y = _rand(1000), _rand(1000)
        backend = EagerAndRecordGraphs()
        cf = torch.compile(f, backend=backend, fullgraph=True)
        self.assertEqual(cf(x, y), x + y)
        self.assertEqual(calls, [x.shape] * 3)
        (gm,) = backend.graphs
        self.assertEqual(len(_hop_nodes(gm, cutedsl_kernel_wrapper_mutation)), 1)

    def test_raw_call_through_a_decorator_runs_its_body(self):
        # A functools.wraps decorator around a host function (here one that rewrites
        # an argument) copies its _dsl_cls, but is not a @cute.jit function: Dynamo
        # inlines it, read with or without a source, so its body runs, and captures
        # the raw call of the host function in it.
        def first_half(fn):
            @functools.wraps(fn)
            def wrapper(mX, mY, mOut, n, stream):
                return fn(mX, mY, mOut, n // 2, stream)

            return wrapper

        add_first_half = first_half(add_kernel)

        class M(torch.nn.Module):
            kernel = staticmethod(add_first_half)

        def launch(kernel, x, y):
            out = torch.zeros_like(x)
            stream = cutlass_torch.current_stream()
            kernel(from_dlpack(x), from_dlpack(y), from_dlpack(out), x.numel(), stream)
            return out

        def with_a_source(x, y):
            return launch(add_first_half, x, y)

        def without_a_source(x, y):
            return launch(M().kernel, x, y)

        x, y = _rand(1000), _rand(1000)
        expected = torch.cat([(x + y)[:500], torch.zeros_like(x[500:])])
        self.assertEqual(with_a_source(x, y), expected)
        for f in (with_a_source, without_a_source):
            with self.subTest(f=f.__name__):
                backend = EagerAndRecordGraphs()
                cf = torch.compile(f, backend=backend, fullgraph=True)
                self.assertEqual(cf(x, y), expected)
                (gm,) = backend.graphs
                (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
                record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
                self.assertEqual(record, CuteDSLKernel(add_kernel, "unknown"))

    def test_raw_call_of_a_disabled_host_function_runs_eagerly(self):
        # torch.compiler.disable(host_fn), a functools.wraps wrapper too, is
        # respected, read with or without a source: the call is not captured, and
        # runs eagerly.
        disabled_add = torch.compiler.disable(add_kernel)

        class M(torch.nn.Module):
            kernel = staticmethod(disabled_add)

        def launch(kernel, x, y):
            out = torch.zeros_like(x)
            stream = cutlass_torch.current_stream()
            kernel(from_dlpack(x), from_dlpack(y), from_dlpack(out), x.numel(), stream)
            return out

        def with_a_source(x, y):
            return launch(disabled_add, x, y)

        def without_a_source(x, y):
            return launch(M().kernel, x, y)

        x, y = _rand(1000), _rand(1000)
        for f in (with_a_source, without_a_source):
            with self.subTest(f=f.__name__):
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, r"torch\.compiler\.disable"
                ):
                    torch.compile(f, backend="eager", fullgraph=True)(x, y)
                backend = EagerAndRecordGraphs()
                self.assertEqual(torch.compile(f, backend=backend)(x, y), x + y)
                self.assertGreaterEqual(len(backend.graphs), 1)
                for gm in backend.graphs:
                    hops = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
                    self.assertEqual(hops, [])

    def test_raw_call_after_a_layout_change_graph_breaks(self):
        # A cute tensor stands for its tensor with the layout it had at from_dlpack:
        # a call after an in-place view op changed the size, stride or storage
        # offset graph-breaks, and runs eagerly on a cute tensor made from the tensor
        # as it is then, which for t_() is what eager CuTeDSL launches with too (it
        # reads the shape and stride through the DLPack capsule at the launch). A
        # layout restored before the call is captured, and so is one whose symbolic
        # size a guard specialized in between.
        def transpose(t):
            t.t_()

        def unsqueeze(t):
            t.unsqueeze_(0)

        def transpose_twice(t):
            t.t_().t_()

        def specialize(t):
            if t.shape[0] != 16:
                raise AssertionError(t.shape)

        def f(x, view_op):
            out = torch.zeros_like(x)
            mo = from_dlpack(out)
            view_op(out)
            add_one_2d_kernel(from_dlpack(x), mo, cutlass_torch.current_stream())
            return out

        x = _rand(16, 16)
        for view_op in (transpose, unsqueeze):
            with self.subTest(view_op=view_op.__name__):
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported,
                    "parameter 'mOut': the tensor passed to from_dlpack changed its "
                    "size, stride or storage offset in place after from_dlpack",
                ):
                    torch.compile(f, backend="eager", fullgraph=True)(x, view_op)
        cf = torch.compile(f, backend="aot_eager")
        self.assertEqual(cf(x, transpose), f(x, transpose))
        for view_op, dynamic in ((transpose_twice, False), (specialize, True)):
            with self.subTest(view_op=view_op.__name__):
                backend = EagerAndRecordGraphs()
                cf = torch.compile(f, backend=backend, fullgraph=True, dynamic=dynamic)
                self.assertEqual(cf(x, view_op), x + 1)
                (gm,) = backend.graphs
                hops = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
                self.assertEqual(len(hops), 1)

    def test_raw_call_after_a_storage_change_graph_breaks(self):
        # A cute tensor stands for its tensor with the storage it had at from_dlpack:
        # after set_() (on a tensor made in the region) or an assignment to .data (on
        # a graph input), eager would launch on the data pointer from_dlpack
        # recorded, which the captured kernel would not write, so the call
        # graph-breaks.
        def through_set(t, other):
            t = t.clone()
            mt = from_dlpack(t)
            t.set_(other)
            scale_inplace_kernel(mt, t.numel(), cutlass_torch.current_stream())
            return t

        def through_data(t, other):
            mt = from_dlpack(t)
            t.data = other
            scale_inplace_kernel(mt, t.numel(), cutlass_torch.current_stream())
            return t

        swapped = (
            "parameter 'mX': the tensor passed to from_dlpack was given another "
            "storage in place after from_dlpack"
        )
        for f in (through_set, through_data):
            with self.subTest(f=f.__name__):
                cf = torch.compile(f, backend="eager", fullgraph=True)
                with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, swapped):
                    cf(_rand(8), _rand(8))

    @parametrize("view", [False, True])
    def test_raw_call_in_place(self, view):
        # x = x + y in place: the same tensor (or an identical view of it) for mX and
        # mOut. Every tensor of a raw call counts as written, and two written
        # arguments that are the same memory cannot be functionalized, so the call
        # graph-breaks and runs eagerly instead of failing in the backend.
        def f(x, y):
            x = x.clone()
            add_kernel(
                from_dlpack(x),
                from_dlpack(y),
                from_dlpack(x.view(-1) if view else x),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return x

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(x, y), x + y)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "'mX' and 'mOut' are the same tensor .* a raw call counts every tensor",
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(torch.compile(f, backend="aot_eager")(x, y), x + y)

    def test_wrapper_call_with_overlapping_written_args(self):
        # As for a raw call: two written arguments that are the same memory, under
        # mutates_args="unknown" or for two named parameters, cannot be
        # functionalized, so the call graph-breaks and runs eagerly instead of
        # failing in the backend.
        add_unknown = wrap_cutedsl(add_kernel, mutates_args="unknown")

        def in_place(x, y):
            x = x.clone()
            add_unknown(x, y, x, x.numel())
            return x

        def same_outputs(x, y):
            out = torch.empty_like(x)
            add_sub_w(x, y, out, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        for f, pattern in (
            (in_place, r"'mX' and 'mOut' .* written \(mutates_args='unknown'\)"),
            (same_outputs, r"'mSum' and 'mDiff' .* written \(mutates_args=\('mSum',"),
        ):
            with self.subTest(f=f.__name__):
                expected = f(x, y)
                with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, pattern):
                    torch.compile(f, backend="aot_eager", fullgraph=True)(x, y)
                self.assertEqual(torch.compile(f, backend="aot_eager")(x, y), expected)

    def test_raw_call_counts_read_only_tensors_as_written(self):
        # As the wrap_cutedsl docstring says, a raw call counts every tensor argument
        # as written, so the same tensor, or views that may overlap, passed for two
        # parameters that the kernel only reads graph-break, and the call runs
        # eagerly; a call through wrap_cutedsl naming the precise write set is
        # captured.
        def raw(x, y):
            out = torch.empty_like(y)
            add_kernel(
                from_dlpack(x),
                from_dlpack(y),
                from_dlpack(out),
                y.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        def wrapped(x, y):
            out = torch.empty_like(y)
            add_w(x, y, out, y.numel())
            return out

        base = _rand(1001)
        for x, y in ((base, base), (base[1:], base[:-1])):
            with self.subTest(views=x is not y):
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, "'mX' and 'mY' are the same tensor"
                ):
                    torch.compile(raw, backend="aot_eager", fullgraph=True)(x, y)
                self.assertEqual(torch.compile(raw, backend="aot_eager")(x, y), x + y)
                cf = torch.compile(wrapped, backend="aot_eager", fullgraph=True)
                self.assertEqual(cf(x, y), x + y)

    def test_raw_cute_jit_call_runs_eagerly_after_graph_break(self):
        # Without fullgraph, a raw call of cute tensors made outside the region
        # graph-breaks with its own reason, though it reads the stream in the region;
        # with a stream made outside too, the reason is the stream's, which is
        # checked first. The call then runs eagerly.
        stream = _stream()

        def tensors_made_outside(x, tx, ty, tout, n):
            y = x + 1
            add_kernel(tx, ty, tout, n, cutlass_torch.current_stream())
            return y * 2

        def stream_made_outside(x, tx, ty, tout, n):
            y = x + 1
            add_kernel(tx, ty, tout, n, stream)
            return y * 2

        x, a, b = _rand(1000), _rand(1000), _rand(1000)
        tensor_reason = "'mX' is a _Tensor made outside the compiled region"
        stream_reason = "pass cutlass.torch.current_stream(), read on the stream"
        for f, reason in (
            (tensors_made_outside, tensor_reason),
            (stream_made_outside, stream_reason),
        ):
            with self.subTest(f=f.__name__):
                out = torch.zeros_like(a)
                tx, ty, tout = (from_dlpack(t, assumed_align=16) for t in (a, b, out))
                torch._dynamo.utils.counters.clear()
                cf = torch.compile(f, backend="eager")
                self.assertEqual(cf(x, tx, ty, tout, 1000), (x + 1) * 2)
                self.assertEqual(out, a + b)
                reasons = list(torch._dynamo.utils.counters["graph_break"])
                self.assertTrue(any(reason in r for r in reasons), reasons)

    def test_strict_export(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(M(), (x, y), strict=True)
        self.assertExpectedInline(
            _normalize_indices(ep.graph_module.code.strip()),
            """\
def forward(self, x, y):
    empty_like_default = torch.ops.aten.empty_like.default(x, pin_memory = False)
    cutedsl_kernel_wrapper_mutation = torch.ops.higher_order.cutedsl_kernel_wrapper_mutation(kernel_idx = K, constant_args_idx = C, kwargs = {'mX': x, 'mY': y, 'mOut': empty_like_default});  x = y = cutedsl_kernel_wrapper_mutation = None
    return (empty_like_default,)""",
        )
        # Dynamo passes the static size as a constant of the launch.
        (node,) = _hop_nodes(ep.graph_module, cutedsl_kernel_wrapper_mutation)
        constant_args_idx = node.kwargs["constant_args_idx"]
        constant_args = cutedsl_kernel_side_table.get_constant_args(constant_args_idx)
        self.assertEqual(constant_args, {"n": 1000})
        self.assertEqual(ep.module()(x, y), x + y)
        decomposed = ep.run_decompositions()
        self.assertEqual(decomposed.module()(x, y), x + y)
        # Unlike a raw call (test_strict_export_raw_call), x and y are not written.
        self.assertEqual(decomposed.graph_signature.user_inputs_to_mutate, {})

    def test_strict_export_raw_call(self):
        # The toy's raw call has the write set "unknown": once functionalized, every
        # tensor argument is cloned and every user input is a mutated input.
        class Toy(torch.nn.Module):
            def forward(self, x, y):
                return toy(x, y)

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(Toy(), (x, y), strict=True)
        (node,) = _hop_nodes(ep.graph_module, cutedsl_kernel_wrapper_mutation)
        record = cutedsl_kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        self.assertEqual(record, CuteDSLKernel(toy_add_kernel, "unknown"))
        decomposed = ep.run_decompositions()
        self.assertExpectedInline(
            _normalize_indices(decomposed.graph_module.code.strip()),
            """\
def forward(self, x, y):
    empty_permuted_default = torch.ops.aten.empty_permuted.default([1000], [0], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    cutedsl_kernel_wrapper_functional = torch.ops.higher_order.cutedsl_kernel_wrapper_functional(kernel_idx = K, constant_args_idx = C, kwargs = {'mX': x, 'mY': y, 'mOut': empty_permuted_default}, tensors_to_clone = ['mX', 'mY', 'mOut']);  x = y = empty_permuted_default = None
    getitem = cutedsl_kernel_wrapper_functional['mOut']
    getitem_1 = cutedsl_kernel_wrapper_functional['mX']
    getitem_2 = cutedsl_kernel_wrapper_functional['mY'];  cutedsl_kernel_wrapper_functional = None
    relu_default = torch.ops.aten.relu.default(getitem);  getitem = None
    return (getitem_1, getitem_2, relu_default)""",
        )
        self.assertEqual(
            decomposed.graph_signature.user_inputs_to_mutate,
            {"getitem_1": "x", "getitem_2": "y"},
        )
        for program in (ep, decomposed):
            x_in, y_in = x.clone(), y.clone()
            self.assertEqual(program.module()(x_in, y_in), (x + y).relu())
            self.assertEqual((x_in, y_in), (x, y))

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

    def test_kernel_output_in_autograd_graph(self):
        # The launch has no derivative: its output enters the autograd graph as a
        # constant, exactly as in eager, whether the wrapper is given a tensor that
        # requires grad or a detached one.
        def detached(x, y):
            out = torch.empty_like(x)
            add_w(x.detach(), y.detach(), out, x.numel())
            return (out * x).sum()

        def requiring_grad(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return (out * x).sum()

        for f in (detached, requiring_grad):
            with self.subTest(f=f.__name__):
                x, y = _rand(1000, requires_grad=True), _rand(1000)
                f(x, y).backward()
                eager_grad = x.grad.clone()
                x.grad = None
                torch.compile(f, backend="aot_eager", fullgraph=True)(x, y).backward()
                self.assertEqual(x.grad, eager_grad)
                self.assertEqual(eager_grad, (x + y).detach())


def _strip_comments(code):
    """Inductor output code carries `Original ATen:` / source comments that would
    make positive substring checks pass vacuously."""
    lines = (ln for ln in code.splitlines() if not ln.lstrip().startswith(("#", "//")))
    return "\n".join(lines)


def _cutedsl_launches(code, host_name):
    """`<launcher>.run(...)` calls of the user kernel ``host_name`` in Python wrapper
    code, as (launcher name, args) pairs. The launcher is named after the host
    function (`cutedsl_add_kernel_0.run(...)`); adjust here if that changes."""
    pattern = rf"\b(\w*{re.escape(host_name)}\w*)\.run\(([^\n]*)\)"
    return re.findall(pattern, _strip_comments(code))


# A module of its own with a host function, out = x * cfg[0], and classes of objects
# for cfg to hold.
_SCALE_MODULE_SOURCE = """\
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

PICKLED = []  # the Cfg objects pickled


class Cfg:
    def __init__(self, alpha):
        self.alpha = alpha

    def __copy__(self):  # else copy.copy, as in snapshots, calls __reduce__
        return type(self)(self.alpha)

    def __reduce__(self):
        PICKLED.append(self)
        return type(self), (self.alpha,)


class _FalseClass(type):
    def __len__(cls):
        return 0


class FalseCfg(Cfg, metaclass=_FalseClass):  # bool(FalseCfg) is False
    pass


@cute.kernel
def _scale_k(
    gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, alpha: cutlass.Constexpr
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < n:
        gOut[i] = gX[i] * alpha


@cute.jit
def scale_kernel(
    mX: cute.Tensor,
    mOut: cute.Tensor,
    n: cutlass.Int32,
    cfg: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _scale_k(mX, mOut, n, cfg[0]).launch(
        grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )
"""


def _scale_module(d, name):
    """_SCALE_MODULE_SOURCE, written to a file in directory ``d`` and executed as the
    module ``name``."""
    path = os.path.join(d, "scale_module.py")
    with open(path, "w") as fh:
        fh.write(_SCALE_MODULE_SOURCE)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_as_script(path, code):
    """Write ``code`` to ``path`` and run it as `python <path>` would, but as a module
    not named __main__, so that its benchmark harness, which times 100 calls, sets
    the global seed and resets the CUDA memory stats, stays out of the test process:
    the caller calls its get_args() and call(). Returns its globals."""
    with open(path, "w") as fh:
        fh.write(code)
    return runpy.run_path(path, run_name="cutedsl_output_code")


@requires_cuda_and_triton
@requires_cutedsl
class CuteDSLInductorTests(TestCase):
    """Inductor: the HOP lowers to ir.UserDefinedCuteDSLKernel, launched from the
    Python wrapper as `<launcher>.run(<args>, stream=raw_streamN)`."""

    hw_classification = HardwareClassification.CUDA

    def test_cutedsl_kernel_symbolic_bool_scalar(self):
        # A symbolic bool for an unannotated param is a runtime scalar, evaluated at
        # each launch like a symbolic int: both sizes run one graph.
        w = wrap_cutedsl(select_kernel, mutates_args=["mOut"])
        cnt = CompileCounterWithBackend("inductor")

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), x.shape[0] > 1500)
            return out

        cf = torch.compile(f, backend=cnt, dynamic=True, fullgraph=True)
        for n in (1000, 2000):
            x = _rand(n)
            self.assertEqual(cf(x), (3 if n > 1500 else 5) * x)
        self.assertEqual(cnt.frame_count, 1)

    def test_cutedsl_kernel_numeric_scalar_values(self):
        # A cutlass Numeric default, and a Numeric value passed for a Numeric param,
        # are launch arguments, Python floats in the generated code: the two launches
        # share a launcher.
        w = wrap_cutedsl(add_c_kernel, mutates_args=["mOut"])
        c = cutlass.Float32(0.75)

        def f(x):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel())
            w(x, b, x.numel(), c=c)
            return a, b

        x = _rand(1000)
        (a, b), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual(a, x + 0.25)
        self.assertEqual(b, x + 0.75)
        launches = _cutedsl_launches(code, "add_c_kernel")
        self.assertEqual(len({name for name, _ in launches}), 1, code)
        scalars = []
        for _, args in launches:
            self.assertRegex(args, r"^arg\d+_\d+, buf\d+, 1000, \S+, stream=\w+$")
            scalars.append(args.split(", ")[3])
        self.assertEqual(sorted(scalars), ["0.25", "0.75"])

    def test_cutedsl_kernel_tensor_default_refused_as_in_eager(self):
        # Inductor refuses a parameter left to a tensor default as the eager launch
        # does: both split a call with CuteDSLHostFnInfo.split_args. The graph calls
        # the HOP itself, so that the call reaches Inductor whatever the capture of
        # a wrapper's call checks.
        from torch._inductor.exc import InductorError
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        x, y = _rand(1000), _rand(1000)

        def add_default_y(mX, mOut, stream: cuda.CUstream, mY=y):
            pass

        msg = (
            "CuTeDSL kernel add_default_y: parameter 'mY' defaults to a tensor; pass "
            "the tensor"
        )
        with self.assertRaisesRegex(ValueError, f"^{re.escape(msg)}$"):
            launch_cutedsl_kernel(add_default_y, {"mX": x, "mOut": torch.empty_like(x)})
        kidx = _add_kernel_to_table(add_default_y, ["mOut"])
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def f(x):
            out = torch.empty_like(x)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs={"mX": x, "mOut": out}
            )
            return out

        gm = make_fx(f, tracing_mode="fake")(x)
        with self.assertRaises(InductorError) as cm:
            torch._inductor.compile(gm, [x])
        self.assertIs(type(cm.exception.inner_exception), ValueError)
        self.assertEqual(str(cm.exception.inner_exception), msg)

    @parametrize("passed", [False, True])
    def test_cutedsl_kernel_none_for_a_numeric_param_is_a_constant(self, passed):
        # None for a cutlass Numeric parameter, passed or as its default, is a
        # constant on both paths: the eager launch's launcher and the generated
        # code's bind it instead of passing it, and both launches fail alike in the
        # DSL, which refuses None for a Numeric when it compiles.
        from torch._inductor.graph import GraphLowering
        from torch._native.cutedsl import user_kernel as cutedsl_user_kernel

        w = wrap_cutedsl(add_opt_c_kernel, mutates_args=["mOut"])
        c = {"c": None} if passed else {}

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), **c)
            return out

        x = _rand(1000)
        codes = []
        with _counting_compiles():
            args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, **c}
            with self.assertRaises(cutlass.DSLRuntimeError) as eager:
                cutedsl_user_kernel.launch_cutedsl_kernel(add_opt_c_kernel, args)
            launchers = cutedsl_user_kernel._eager_launchers[add_opt_c_kernel]
            (launcher,) = launchers.values()
            self.assertEqual(launcher.arg_names, ("mX", "mOut", "n"))
            self.assertEqual(launcher.constant_args, {"c": None})
            with (
                mock.patch.object(GraphLowering, "save_output_code", codes.append),
                self.assertRaises(cutlass.DSLRuntimeError) as compiled,
            ):
                torch.compile(f, fullgraph=True)(x)
        self.assertEqual(str(compiled.exception), str(eager.exception))
        (code,) = codes
        header = "run(*['mX', 'mOut', 'n'], stream) with constants {c: NoneType}"
        self.assertIn(header, code)
        ((_, launch_args),) = _cutedsl_launches(code, "add_opt_c_kernel")
        self.assertRegex(launch_args, r"^arg\d+_\d+, buf\d+, 1000, stream=\w+$")

    def test_cutedsl_kernel_graph_not_serializable(self):
        # The generated module binds launchers of this process's host functions, so
        # serializing the compiled graph (aot_compile) fails loudly. Compiling in a
        # subprocess never serializes it: a graph with user-kernel HOPs falls back to
        # compiling in process.
        from torch._inductor import compile_fx
        from torch._inductor.compile_fx_subproc import _SubprocessFxCompile

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            RuntimeError,
            r"This compiled graph launches user CuTeDSL kernels "
            r"\(cutedsl_add_kernel_\d+\) through objects bound in this process, so "
            r"it cannot be serialized \(aot_compile, precompile\)\. Compile it in "
            r"the process that runs it\.",
        ):
            torch.compile(f, fullgraph=True).aot_compile(((x, y), {}))
        mode = compile_fx.FxCompileMode.SUBPROCESS
        sent = AssertionError("sent to a compile subprocess")
        with (
            mock.patch.object(compile_fx, "fx_compile_mode", mode),
            mock.patch.object(_SubprocessFxCompile, "process_pool", side_effect=sent),
        ):
            self.assertEqual(torch.compile(f, fullgraph=True)(x, y), (x + y) * 2)

    @inductor_config.patch(cpp_wrapper=True)
    def test_cutedsl_kernel_cpp_wrapper_refused(self):
        # The C++ wrappers have no launcher for the kernel yet: they refuse it.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            Exception,
            r"NotImplementedError: User-defined CuTeDSL kernels are not supported by "
            r"the C\+\+ wrapper",
        ):
            torch.compile(f, fullgraph=True)(x, y)

    @inductor_config.patch(fx_wrapper=True)
    def test_cutedsl_kernel_fx_wrapper_refused(self):
        # The FX wrapper has no Wrapper IR line for the launch: it refuses the kernel.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            Exception,
            r"NotImplementedError: User-defined CuTeDSL kernels are not supported by "
            r"the FX wrapper",
        ):
            torch.compile(f, fullgraph=True)(x, y)

    def test_cutedsl_kernel_wrapper_codegen(self):
        # The Python wrapper comments each launcher with its host function, the
        # arguments run() takes (tensors and runtime scalars, in signature order) and
        # the constants it binds, by name and type. A launch passes only those
        # arguments, a Numeric value as a Python float, and then the stream.
        half = cutlass.Float32(0.5)

        def f(x, y):
            a, s, d = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
            mul_const_w(x, a, x.numel(), 3)
            add_sub_w(x, y, s, d, x.numel(), scale=half)
            return a, s, d

        x, y = _rand(1000), _rand(1000)
        (a, s, d), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual((a, s, d), (3 * x, (x + y) * 0.5, (x - y) * 0.5))
        ((mul, mul_args),) = _cutedsl_launches(code, "mul_const_kernel")
        ((add_sub, add_sub_args),) = _cutedsl_launches(code, "add_sub_kernel")
        host = f"CuTeDSL host function {__name__}"
        loaded = "bound when this module is loaded"
        self.assertIn(
            f"\n# {mul}: {host}.mul_const_kernel, run(*['mX', 'mOut', 'n'], "
            f"stream) with constants {{alpha: int}}, {loaded}\n",
            code,
        )
        self.assertIn(
            f"\n# {add_sub}: {host}.add_sub_kernel, run(*['mX', 'mY', 'mSum', "
            f"'mDiff', 'n', 'scale'], stream) with constants {{}}, {loaded}\n",
            code,
        )
        tensor, buf = r"arg\d+_\d+", r"buf\d+"
        self.assertRegex(mul_args, rf"^{tensor}, {buf}, 1000, stream=\w+$")
        self.assertRegex(
            add_sub_args,
            rf"^{tensor}, {tensor}, {buf}, {buf}, 1000, 0\.5, stream=\w+$",
        )

    @inductor_config.patch({"aot_inductor.debug_intermediate_value_printer": "2"})
    def test_cutedsl_kernel_debug_printer(self):
        # As around a user Triton kernel's launch, the intermediate value printer
        # prints each tensor the launch takes, before and after it.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y).relu())
        ((name, args),) = _cutedsl_launches(code, "add_kernel")
        tensors = args.split(", ")[:3]  # mX, mY and mOut
        check = FileCheck()
        for line in [
            *(f'"inductor: before_launch - {name} - {t}", {t})' for t in tensors),
            f"{name}.run(",
            *(f'"inductor: after_launch - {name} - {t}", {t})' for t in tensors),
        ]:
            check.check(line)
        check.run(code)
        printed = stdout.getvalue()
        self.assertIn(f"inductor: after_launch - {name} - {tensors[2]}\n", printed)

    def test_cutedsl_kernel_toy_raw_call(self):
        # The toy through Inductor: one launch of the captured raw call, from the
        # Python wrapper.
        x, y = _rand(2048), _rand(2048)
        res, (code,) = run_and_get_code(torch.compile(toy, fullgraph=True), x, y)
        self.assertEqual(res, (x + y).relu())
        self.assertEqual(len(_cutedsl_launches(code, "toy_add_kernel")), 1, code)

    def test_cutedsl_kernel_inductor_basic(self):
        from torch._inductor import ir

        codegen_calls = []
        orig_codegen = ir.UserDefinedCuteDSLKernel.codegen

        def spy_codegen(self, wrapper):
            codegen_calls.append(self)
            return orig_codegen(self, wrapper)

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(ir.UserDefinedCuteDSLKernel, "codegen", spy_codegen):
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y).relu())
        self.assertEqual(len(codegen_calls), 1)
        launches = _cutedsl_launches(code, "add_kernel")
        self.assertEqual(len(launches), 1, code)
        name, args = launches[0]
        # Launched on the stream Inductor fetched for the device (named
        # `raw_stream0` or `stream0` depending on the version), never a constant.
        self.assertRegex(args, r"stream=(raw_)?stream\d+$")
        # The relu after the launch stays an Inductor (Triton) kernel.
        relu_launch = r"triton_\w*relu\w*\.run\("
        FileCheck().check(f"{name}.run(").check_regex(relu_launch).run(code)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_cutedsl_kernel_on_a_device_that_is_not_current(self):
        # On cuda:1 while cuda:0 is current, the launch gets the stream of cuda:1. It
        # reuses the kernel compiled for the cuda:0 call if the devices have one
        # capability: the DSL loads a compiled kernel on every device, so compiles
        # are cached per capability.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        cf = torch.compile(f, fullgraph=True)
        x, y = _rand(1000), _rand(1000)
        with _counting_compiles() as compile_spy:
            self.assertEqual(cf(x, y), (x + y).relu())
            self.assertEqual(compile_spy.call_count, 1)
            x, y = x.to("cuda:1"), y.to("cuda:1")
            with torch.cuda.device(0):
                res, (code,) = run_and_get_code(cf, x, y)
        self.assertEqual(res, (x + y).relu())
        capabilities = {torch.cuda.get_device_capability(i) for i in (0, 1)}
        self.assertEqual(compile_spy.call_count, len(capabilities))
        ((name, args),) = _cutedsl_launches(code, "add_kernel")
        stream = args.rsplit("stream=", 1)[1]
        FileCheck().check(f"{stream} = get_raw_stream(1)").check(name).run(code)

    def test_cutedsl_kernel_user_stream(self):
        # In a user stream region the launch reads the stream current at runtime
        # (the region's), as Inductor's own kernel calls there do.
        def f(x, y):
            s = torch.cuda.Stream()
            e = torch.cuda.Event()
            with torch.cuda.stream(s):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                z = out * 2
                e.record()
            torch.cuda.current_stream().wait_event(e)
            return z + 1

        x, y = _rand(1000), _rand(1000)
        torch.cuda.synchronize()  # the region reads x and y without waiting for them
        res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y) * 2 + 1)
        ((name, args),) = _cutedsl_launches(code, "add_kernel")
        self.assertRegex(args, r", stream=raw_stream$")
        stream_read = "raw_stream = get_raw_stream(0)"
        FileCheck().check("with stream1:").check(stream_read).check(name).run(code)

    @parametrize("compile_on_one_rank", [False, True])
    def test_cutedsl_kernel_stream_picked_as_for_inductor_kernels(
        self, compile_on_one_rank
    ):
        # The launch reads its stream as the Inductor kernel after it does, also
        # under compile-on-one-rank, where both read the stream of the device current
        # at runtime, so that the wrapper is the same on every rank.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with torch.compiler.config.patch(compile_on_one_rank=compile_on_one_rank):
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y).relu())
        # A launch's stream read (the line before it) and stream argument.
        launch = r"(?m)^ *(\w+ = get_raw_stream\(\w+\))\n *{}\.run\(.*stream=(\w+)\)$"
        code = _strip_comments(code)
        (cutedsl,) = re.findall(launch.format(r"cutedsl_add_kernel_\d+"), code)
        (relu,) = re.findall(launch.format(r"triton_\w*relu\w*"), code)
        self.assertEqual(cutedsl, relu)
        if compile_on_one_rank:
            self.assertNotIn("get_raw_stream(0)", code)

    def test_cutedsl_kernel_raw_call_user_stream(self):
        # The stream read and the raw call inside torch.cuda.stream(side): the kernel
        # runs on side, as the user Triton kernel launched next to it does, so both
        # see the fill queued on side (behind a sleep) before the call.
        import triton

        from torch.testing._internal.common_utils import get_cycles_per_ms
        from torch.testing._internal.triton_utils import add_kernel as triton_add

        side = torch.cuda.Stream()

        def f(x, y):
            out, ref = torch.empty_like(x), torch.empty_like(x)
            n = x.numel()
            with torch.cuda.stream(side):
                toy_add_kernel(
                    from_dlpack(x),
                    from_dlpack(y),
                    from_dlpack(out),
                    n,
                    cutlass_torch.current_stream(),
                )
                triton_add[(triton.cdiv(n, 128),)](x, y, ref, n, BLOCK_SIZE=128)
            torch.cuda.current_stream().wait_stream(side)
            return out, ref

        x, y = torch.zeros(1000, device=DEVICE), _rand(1000)
        cf = torch.compile(f, fullgraph=True)
        torch.cuda.synchronize()
        _, (code,) = run_and_get_code(cf, x, y)  # compiles outside the timed part
        ((name, _),) = _cutedsl_launches(code, "toy_add_kernel")
        FileCheck().check("with stream1:").check(f"{name}.run(").run(code)
        with torch.cuda.stream(side):
            torch.cuda._sleep(int(100 * get_cycles_per_ms()))
            x.fill_(1.0)
        out, ref = cf(x, y)
        torch.cuda.synchronize()
        self.assertEqual(ref, y + 1)
        self.assertEqual(out, y + 1)

    @parametrize("grad", [False, True])
    @parametrize("dynamic", [False, True])
    @inductor_config.patch("implicit_fallbacks", False)
    def test_cutedsl_kernel_no_clones(self, grad, dynamic):
        # Mirrors test_triton_kernel_no_clones: with the right mutates_args the
        # kernel writes the graph input in place, no clone or copy-back.
        def call_cutedsl(x, y, output):
            n_elements = output.numel()
            tmp = torch.add(x, 1)
            add_w(x, y, output, n_elements)
            return output, tmp

        t1 = _rand(5, requires_grad=grad)
        t2 = _rand(5, requires_grad=grad)
        o1 = torch.zeros_like(t1, requires_grad=grad)
        torch_add = call_cutedsl(t1, t2, o1)
        o2 = torch.zeros_like(t1, requires_grad=grad)
        test, (code,) = run_and_get_code(
            torch.compile(call_cutedsl, dynamic=dynamic), t1, t2, o2
        )
        self.assertEqual(torch_add, test)
        # Like the Triton test, these rely on `Original ATen:` comments, so a clone
        # fused into some kernel is caught too.
        self.assertNotIn("aten.copy", code)
        self.assertNotIn("aten.clone", code)
        self.assertEqual(len(_cutedsl_launches(code, "add_kernel")), 1, code)
        if not grad:
            self.assertIn("return (buf0, )", code)

    def test_cutedsl_kernel_reinplaces_intermediate(self):
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

    def test_cutedsl_kernel_mutates_graph_input(self):
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
    def test_cutedsl_kernel_mutates_graph_input_view(self, view):
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
    def test_cutedsl_kernel_with_views(self, dynamic, backend):
        # Mirrors test_triton_kernel_with_views.
        def call_take_view(x):
            output = torch.zeros_like(x)
            mul_const_w(x, output, output.numel(), 2)
            return output

        def call_return_view(x):
            output = torch.zeros_like(x)
            mul_const_w(x.view(-1), output.view(-1), output.numel(), 2)
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
    def test_cutedsl_kernel_strided_input(self, which):
        # Mirrors test_triton_kernel_strided_input{,_nonzero_offset} and
        # test_triton_kernel_slice_and_view_input with a layout-indexing kernel. The
        # expected value comes from torch ops: an eager run would launch through the
        # same launcher, so a layout bug would corrupt it as well.
        def source(inp):
            if which in ("left", "right"):
                left, right = torch.split(inp, [128, 128], dim=1)
                return left if which == "left" else right  # strides (256, 1)
            if which == "slice_view":
                return inp[:, :128].reshape(64, 128)[:, ::2]
            return inp.t()

        def f(inp):
            src = source(inp)
            out = torch.empty(src.shape, device=src.device)
            add_one_2d_w(src, out)
            return out + src

        inp = _rand(64, 256)
        src = source(inp)
        self.assertEqual(torch.compile(f, fullgraph=True)(inp), src + 1 + src)

    @parametrize("gap", [1, 2])
    def test_cutedsl_kernel_mutated_offset_view_not_miscompiled(self, gap):
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
    def test_cutedsl_kernel_multiple_outputs(self, dynamic, backend):
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

    def test_cutedsl_kernel_output_in_autograd_graph(self):
        # As CuteDSLDynamoTests.test_kernel_output_in_autograd_graph with Inductor:
        # the launch has no derivative, so its output enters the autograd graph as a
        # constant, exactly as in eager.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x.detach(), y.detach(), out, x.numel())
            return (out * x).sum()

        x, y = _rand(1000, requires_grad=True), _rand(1000)
        f(x, y).backward()
        eager_grad = x.grad.clone()
        x.grad = None
        torch.compile(f, fullgraph=True)(x, y).backward()
        self.assertEqual(x.grad, eager_grad)
        self.assertEqual(eager_grad, (x + y).detach())

    def test_cutedsl_kernel_dependencies(self):
        # Mirrors test_triton_kernel_dependancies: kernel -> kernel -> aten op.
        def f(x, y):
            out = torch.zeros_like(x)
            add_w(x, y, out, x.numel())
            out2 = torch.zeros_like(out)
            add_w(out, y, out2, out.numel())
            return torch.add(out2, 1)

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), x + 2 * y + 1)

    def test_cutedsl_kernel_reinplace_inplaceable_pass(self):
        # Mirrors test_triton_kernel_reinplace_inplaceable_pass: the second launch
        # reads and writes the same buffer through two parameters (mX is mOut).
        def f(x, y):
            out = torch.zeros_like(x)
            add_w(x, y, out, x.numel())
            add_w(out, x, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), 2 * x + y)

    def test_cutedsl_kernel_caching(self):
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
        launches = _cutedsl_launches(code, "add_kernel")
        self.assertEqual(len(launches), 4, code)
        self.assertEqual(len({name for name, _ in launches}), 1, launches)

    def test_cutedsl_kernel_launcher_keys_constants_once(self):
        # A launcher checks its argument names, reads the host function's signature
        # and keys its constants when it is made: a launch keys only its arguments,
        # so its cost does not grow with the constants. It shares its compile with
        # an eager launch of the same call.
        from torch._higher_order_ops import cutedsl_kernel_wrap, user_kernel_hop
        from torch._native.cutedsl.user_kernel import (
            CuteDSLKernelLauncher,
            launch_cutedsl_kernel,
        )

        def spy(module, name):
            return mock.patch.object(module, name, wraps=getattr(module, name))

        names = ("mX", "mOut", "n")
        with self.assertRaisesRegex(TypeError, "bogus"):
            CuteDSLKernelLauncher(cfg_scale_kernel, names, {"cfg": None, "bogus": 1})
        cfg = types.SimpleNamespace(alpha=3.0, table=list(range(1000)))
        x = _rand(1000)
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream().cuda_stream
        with _counting_compiles() as compile_spy:
            args = {"mX": x, "mOut": out, "n": x.numel(), "cfg": cfg}
            launch_cutedsl_kernel(cfg_scale_kernel, args)
            launcher = CuteDSLKernelLauncher(cfg_scale_kernel, names, {"cfg": cfg})
            with (
                spy(user_kernel_hop, "constant_key") as key_spy,
                spy(cutedsl_kernel_wrap, "get_host_fn_info") as info_spy,
            ):
                for _ in range(3):
                    out.zero_()
                    launcher.run(x, out, x.numel(), stream=stream)
                    self.assertEqual(out, 3 * x)
            self.assertEqual(compile_spy.call_count, 1)
        key_spy.assert_not_called()
        info_spy.assert_not_called()

    def test_cutedsl_kernel_launcher_definition_keys_constants_once(self):
        # Defining a launcher in generated code keys each constant once, for the
        # cache of launchers, and hands the keys to the launcher, which would
        # otherwise walk the constants again.
        from torch._higher_order_ops import user_kernel_hop
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen

        define = PythonWrapperCodegen.define_user_defined_cutedsl_kernel
        keyed = []

        def define_and_record_keys(wrapper, kernel, **kwargs):
            key = user_kernel_hop.constant_key
            with mock.patch.object(user_kernel_hop, "constant_key", wraps=key) as spy:
                name = define(wrapper, kernel, **kwargs)
            keyed.extend(call.args[0] for call in spy.call_args_list)
            return name

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])
        cfg = types.SimpleNamespace(alpha=3.0, table=list(range(1000)))

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        x = _rand(1000)
        with mock.patch.object(
            PythonWrapperCodegen,
            "define_user_defined_cutedsl_kernel",
            define_and_record_keys,
        ):
            self.assertEqual(torch.compile(f, fullgraph=True)(x), 3 * x)
        self.assertEqual([type(value) for value in keyed], [types.SimpleNamespace])

    def test_cutedsl_kernel_eager_launches_reuse_a_launcher(self):
        # Eager launches of one call reuse a launcher, which has their compile. The
        # constants are keyed on every launch: one changed in place gets a launcher
        # of its own, and changing it back finds the first launcher again.
        from torch._native.cutedsl.user_kernel import (
            CuteDSLKernelLauncher,
            launch_cutedsl_kernel,
        )

        get_compiled = CuteDSLKernelLauncher._get_compiled
        cfg = types.SimpleNamespace(alpha=3.0)
        x = _rand(1000)
        out = torch.empty_like(x)
        args = {"mX": x, "mOut": out, "n": x.numel(), "cfg": cfg}
        with (
            _counting_compiles() as compile_spy,
            mock.patch.object(
                CuteDSLKernelLauncher,
                "_get_compiled",
                autospec=True,
                side_effect=get_compiled,
            ) as miss_spy,
        ):
            for alpha in (3.0, 3.0, 0.5, 3.0):
                cfg.alpha = alpha
                launch_cutedsl_kernel(cfg_scale_kernel, args)
                self.assertEqual(out, alpha * x)
        self.assertEqual(miss_spy.call_count, 2)
        self.assertEqual(compile_spy.call_count, 2)

    def test_cutedsl_kernel_launcher_compile_logged_to_native_dsl_compile(self):
        # The generated code's launcher compiles at its first launch through
        # plan_cache.cached_plan, as an eager launch does, so
        # TORCH_LOGS=+native_dsl_compile logs that compile under the kernel's name.
        from torch.testing._internal.logging_utils import log_settings

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        records = []
        handler = logging.Handler()
        handler.emit = records.append
        name = "torch._native.instrumentation.__native_dsl_compile"
        logger = logging.getLogger(name)
        cf = torch.compile(f, fullgraph=True)
        x, y = _rand(1000), _rand(1000)
        with log_settings("+native_dsl_compile"), _counting_compiles() as compile_spy:
            logger.addHandler(handler)
            try:
                for _ in range(2):
                    self.assertEqual(cf(x, y), (x + y).relu())
            finally:
                logger.removeHandler(handler)
        self.assertEqual(compile_spy.call_count, 1)
        messages = [record.getMessage() for record in records]
        self.assertEqual(len(messages), 1, messages)
        self.assertRegex(messages[0], r"^add_kernel \[cutedsl\] compiled in ")

    def test_cutedsl_kernel_launch_profiler_record(self):
        # While the profiler runs, a launch is a record function named after its
        # launcher, as a user Triton kernel's launch is named after the kernel; an
        # eager launch's is named after the host function. Otherwise, no record.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        cf = torch.compile(f, fullgraph=True)
        _, (code,) = run_and_get_code(cf, x, y)
        ((name, _),) = _cutedsl_launches(code, "add_kernel")
        out = torch.empty_like(x)
        with mock.patch.object(torch._C._profiler, "_RecordFunctionFast") as record:
            cf(x, y)
            launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))
        record.assert_not_called()
        with torch.profiler.profile() as prof:
            self.assertEqual(cf(x, y), (x + y).relu())
            launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))
        self.assertEqual(out, x + y)
        names = [event.name for event in prof.events()]
        self.assertEqual((names.count(name), names.count("add_kernel")), (1, 1))

    @parametrize("device", ["current", "other"])
    def test_cutedsl_kernel_launch_enters_the_device_only_if_not_current(self, device):
        # A launch makes the tensors' device current, for DLPack export and the
        # launch, only if it is not current already: in generated code, the device
        # guard has made it current.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        if device == "other" and torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices")
        x, y = _rand(1000), _rand(1000)
        if device == "other":
            x, y = x.to("cuda:1"), y.to("cuda:1")
        out = torch.empty_like(x)
        launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))  # compiles
        out.zero_()
        enter = torch.cuda.device.__enter__
        with (
            torch.cuda.device(0),
            mock.patch.object(
                torch.cuda.device, "__enter__", autospec=True, side_effect=enter
            ) as enter_spy,
        ):
            launch_cutedsl_kernel(add_kernel, _add_kwargs(x, y, out))
        self.assertEqual(out, x + y)
        entered = [call.args[0].idx for call in enter_spy.call_args_list]
        self.assertEqual(entered, [1] if device == "other" else [])

    def test_cutedsl_kernel_same_name_kernels(self):
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
        launches = _cutedsl_launches(code, "binary_kernel")
        self.assertEqual(len({name for name, _ in launches}), 2, launches)

    @parametrize("case", ["triton_first", "cutedsl_first", "triton_in_cond"])
    def test_cutedsl_kernel_launcher_name_taken_by_a_triton_kernel(self, case):
        # User Triton kernels and launchers are numbered apart, so a Triton kernel
        # named cutedsl_add_kernel would be cutedsl_add_kernel_0 too, as add_kernel's
        # launcher is, and loading the module would bind the launcher over it. With
        # one set of names, the kernel defined second takes the next free index, also
        # when a torch.cond branch defines it.
        import triton
        import triton.language as tl

        @triton.jit
        def cutedsl_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + i, mask=i < n)
            y = tl.load(y_ptr + i, mask=i < n)
            tl.store(out_ptr + i, x - y, mask=i < n)

        def triton_sub(x, y):
            out, n = torch.empty_like(x), x.numel()
            cutedsl_add_kernel[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
            return out

        def cutedsl_add(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        def f(x, y):
            # The second kernel reads what the first wrote: they run in this order.
            if case == "triton_first":
                first = triton_sub(x, y)
                return first, cutedsl_add(first, y)
            first = cutedsl_add(x, y)
            if case == "cutedsl_first":
                return first, triton_sub(first, y)
            sub = torch.cond(x.sum() > 0, triton_sub, lambda a, b: a - b, (first, y))
            return first, sub

        x, y = _rand(1000).abs(), _rand(1000)
        cf = torch.compile(f, fullgraph=True)
        (first, second), (code,) = run_and_get_code(cf, x, y)
        self.assertEqual(first, x - y if case == "triton_first" else x + y)
        self.assertEqual(second, x)
        (launcher,) = re.findall(r"(?m)^# (\w+): CuTeDSL host function ", code)
        self.assertEqual(launcher, f"cutedsl_add_kernel_{int(case == 'triton_first')}")
        run = r"\b(cutedsl_add_kernel_\d+)\.run\("
        launched = set(re.findall(run, _strip_comments(code)))
        self.assertEqual(launched, {"cutedsl_add_kernel_0", "cutedsl_add_kernel_1"})

    def test_cutedsl_kernel_module_attributes_bound_once(self):
        # Loading the generated module binds its launchers, constants and the classes
        # of opaque values as globals, by name: a launcher named as one of those
        # classes is refused instead of replacing it.
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        from torch._inductor.virtualized import V

        define = PythonWrapperCodegen.define_user_defined_cutedsl_kernel

        def define_as_a_class(self, *args, **kwargs):
            name = define(self, *args, **kwargs)
            V.graph.opaque_value_type_classes[name] = type(name, (), {})
            return name

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        name = "define_user_defined_cutedsl_kernel"
        with (
            mock.patch.object(PythonWrapperCodegen, name, define_as_a_class),
            self.assertRaisesRegex(
                Exception,
                r"module attributes bound twice: \['cutedsl_add_kernel_\d+'\]",
            ),
        ):
            torch.compile(f, fullgraph=True)(x, y)

    def test_cutedsl_kernel_object_that_sets_attributes(self):
        # Generated code binds the side table's own snapshots, and compiling a kernel
        # object runs its __call__, which here caches a layout on itself: the launcher
        # compiles a copy, so a later eager launch of the same entry (from an
        # aot_eager graph, which an equal kernel object maps to it) keys the snapshot
        # as before and reuses the compile.
        w = wrap_cutedsl(call_mul_kernel, mutates_args=["mOut"])

        def f(x, kernel):
            out = torch.empty_like(x)
            w(kernel, x, out, x.numel())
            return out

        x = _rand(1000)
        with _counting_compiles() as compile_spy:
            for backend in ("inductor", "aot_eager"):
                torch._dynamo.reset()
                cf = torch.compile(f, backend=backend, fullgraph=True)
                self.assertEqual(cf(x, CachingMulKernel(2.0)), 2 * x)
        self.assertEqual(compile_spy.call_count, 1)

    def test_cutedsl_kernel_in_cond_branches(self):
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
        launches = _cutedsl_launches(code, "mul_const_kernel")
        self.assertEqual(len(launches), 3, code)
        fns = re.findall(r"(?m)^def (?!call\()\w+\(args\):\n(?:[ \t].*\n|\n)*", code)
        in_fns = _cutedsl_launches("".join(fns), "mul_const_kernel")
        self.assertEqual(len(in_fns), 2, code)
        names = {name for name, _ in launches}
        self.assertEqual(len(names), 2, code)
        for name in names:
            self.assertEqual(len(re.findall(rf"(?m)^# {name}: ", code)), 1, code)

    def test_cutedsl_kernel_constants(self):
        # Each Constexpr value gets a launcher that binds it, while the int for the
        # unannotated k is a runtime scalar, passed at launch.
        def f(x):
            o1, o2, o3 = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
            mul_const_w(x, o1, x.numel(), 3)
            mul_const_w(x, o2, x.numel(), 2.5)
            add_scalar_w(x, o3, x.numel(), 5)
            return o1, o2, o3

        x = _rand(1000)
        (o1, o2, o3), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual((o1, o2, o3), (3 * x, 2.5 * x, x + 5))
        mul_launches = _cutedsl_launches(code, "mul_const_kernel")
        self.assertEqual(len({name for name, _ in mul_launches}), 2, code)
        ((_, add_args),) = _cutedsl_launches(code, "add_scalar_kernel")
        self.assertRegex(add_args, r", 1000, 5, stream=\w+$")

    def test_cutedsl_kernel_constants_with_equal_reprs(self):
        # Constants that print alike but differ get a launcher each: they are told
        # apart as the side table tells them apart, not by repr.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

            def __repr__(self):
                return "Cfg()"

        c3, c5 = Cfg(3.0), Cfg(5.0)
        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel(), c3)
            w(x, b, x.numel(), c5)
            return a, b

        x = _rand(1000)
        (a, b), (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual((a, b), (3 * x, 5 * x))
        launches = _cutedsl_launches(code, "cfg_scale_kernel")
        self.assertEqual(len({name for name, _ in launches}), 2, launches)

    def test_cutedsl_kernel_constant_with_multiline_repr(self):
        # The generated module's comment for a launcher names its constants and
        # their types, never their repr, which can span lines and break the module.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

            def __repr__(self):
                return f"Cfg(\n    alpha={self.alpha},\n)"

        cfg = Cfg(3.0)
        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        x = _rand(1000)
        res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual(res, 3 * x)
        (header,) = re.findall(r"(?m)^# cutedsl_cfg_scale_kernel_\d+: .*$", code)
        self.assertIn(".cfg_scale_kernel, ", header)
        self.assertIn("with constants {cfg: Cfg}", header)
        self.assertNotIn("alpha=", code)

    @parametrize("constant", ["plain_data", "object", "object_of_false_class"])
    def test_cutedsl_kernel_output_code_as_script(self, constant):
        # Run as a script (`python <output_code>.py`, e.g. a TORCH_COMPILE_DEBUG or
        # tlparse artifact), the generated module is not loaded with its launchers:
        # get_args() rebuilds each one whose constants are plain data, importing the
        # host function by module and qualname and unpickling the constants.
        # Pickling an object could run user code (Cfg.__reduce__ records each call)
        # and would write the object, and any tensor it holds, into the module: with
        # one, of a class that is false or not, compiling pickles nothing and the run
        # fails instead, naming the launcher and the constant.
        with tempfile.TemporaryDirectory() as d:
            module = _scale_module(d, "cutedsl_test_scale_module")
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__)
            w = wrap_cutedsl(module.scale_kernel, mutates_args=["mOut"])
            if constant == "plain_data":
                extra = {"tile": (128, None), "tags": frozenset("ab"), "raw": b"x"}
                cfg = (cutlass.Float32(3.0), extra)
            elif constant == "object":
                cfg = (3.0, module.Cfg(5.0))
            else:
                cfg = (3.0, module.FalseCfg(5.0))

            def f(x):
                out = torch.empty_like(x)
                w(x, out, x.numel(), cfg)
                return out

            x = _rand(1000)
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
            self.assertEqual(res, 3 * x)
            path = os.path.join(d, "output_code.py")
            if constant == "plain_data":
                ((name, _),) = _cutedsl_launches(code, "scale_kernel")
                script = _load_as_script(path, code)
                (x,) = script["get_args"]()
                # Rebuilt among the globals call() reads, the launcher keeps its name,
                # which names its profiler record.
                self.assertEqual(script["call"].__globals__[name].name, name)
                (out,) = script["call"]([x])
                self.assertEqual(out, 3 * x)
            else:
                self.assertEqual(module.PICKLED, [])
                self.assertNotIn("pickle.loads", code)
                held = type(cfg[1]).__name__
                msg = (
                    "a run of this module as a script cannot rebuild the launcher of "
                    f"this CuTeDSL kernel (its constant 'cfg' is or holds a {held}); "
                    "its host function must be importable by module and qualname, not "
                    "defined in __main__ or inside a function, and its constants plain "
                    "data: None, bool, int, float, str, bytes, cutlass Numeric values, "
                    "and tuples, lists, dicts and frozensets of these"
                )
                pattern = rf"^cutedsl_scale_kernel_\d+: {re.escape(msg)}$"
                script = _load_as_script(path, code)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    script["get_args"]()

    def test_cutedsl_kernel_output_code_as_script_host_fn_in_main(self):
        # A host function of the compiling script cannot be imported by a run of the
        # generated module as a script, whose __main__ is that module: the run fails,
        # naming the launcher. The compiling process, which calls get_args() itself
        # under profile_bandwidth_output, keeps the launcher it loaded the module with.
        with tempfile.TemporaryDirectory() as d:
            main = _scale_module(d, "__main__")
            w = wrap_cutedsl(main.scale_kernel, mutates_args=["mOut"])
            cfg = (3.0,)

            def f(x):
                out = torch.empty_like(x)
                w(x, out, x.numel(), cfg)
                return out

            x = _rand(1000)
            cf = torch.compile(f, fullgraph=True)
            bandwidth_output = os.path.join(d, "bandwidth.txt")
            real_main = sys.modules["__main__"]
            sys.modules["__main__"] = main  # as when the script itself runs
            try:
                with inductor_config.patch(profile_bandwidth_output=bandwidth_output):
                    with contextlib.redirect_stdout(io.StringIO()):
                        res, (code,) = run_and_get_code(cf, x)
            finally:
                sys.modules["__main__"] = real_main
            self.assertEqual(res, 3 * x)
            script = _load_as_script(os.path.join(d, "output_code.py"), code)
            with self.assertRaisesRegex(
                RuntimeError,
                r"cutedsl_scale_kernel_\d+: a run of this module as a script cannot "
                r"rebuild the launcher of this CuTeDSL kernel \(PicklingError: "
                r"scale_kernel is defined in __main__\)",
            ):
                script["get_args"]()

    def test_cutedsl_kernel_dynamic_shapes(self):
        # One dynamic graph serves every size: the kernels must be compiled with
        # dynamic layouts, not specialized to the first shape seen.
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

    def test_cutedsl_kernel_dynamic_shapes_layout_change(self):
        # Under dynamic shapes a compiled CuTeDSL layout still fixes which dim has
        # stride 1: a transposed input must not reuse the row-major kernel.
        @torch.compile(fullgraph=True, dynamic=True)
        def f(m):
            out = torch.empty(m.shape, device=m.device)
            add_one_2d_w(m, out)
            return out

        for m in (_rand(4, 8), _rand(16, 32), _rand(8, 4).t(), _rand(32, 16).t()):
            self.assertEqual(f(m), m + 1)

    def test_cutedsl_kernel_unaligned_view(self):
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

    def test_cutedsl_kernel_epilogue_and_prologue(self):
        # Inductor ops around the user kernel (it is an extern launch: correctness
        # only, no fusion expected).
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x.sin(), y * 2, out, x.numel())
            return out.relu() + 1

        x, y = _rand(1000), _rand(1000)
        expected = (x.sin() + 2 * y).relu() + 1
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), expected)

    def test_cutedsl_kernel_reduce_overhead(self):
        # CUDA graphs (cudagraph trees) capture the launch on the capture stream:
        # nothing skips them, and their replays launch with each call's inputs.
        from torch._dynamo.utils import counters

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
        counters.clear()
        for _ in range(3):
            x, y = _rand(1000), _rand(1000)
            self.assertEqual(compiled(x, y), (x + y) * 2)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    @parametrize("cudagraphs", [False, True])
    def test_cutedsl_kernel_object_constant_changed_between_calls(self, cudagraphs):
        # Object constants are snapshots guarded on their contents: launch sites with
        # equal objects share a launcher, and changing an object recompiles. So a
        # CUDA graph replay, which runs no Python, never launches an outdated value.
        from torch._dynamo.utils import counters

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x, c1, c2):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel(), c1)
            w(x, b, x.numel(), c2)
            return a, b

        mode = "reduce-overhead" if cudagraphs else None
        cf = torch.compile(f, mode=mode, fullgraph=True)
        x, c1, c2 = _rand(1000), Cfg(3.0), Cfg(3.0)
        (a, b), (code,) = run_and_get_code(cf, x, c1, c2)
        self.assertEqual((a, b), (3 * x, 3 * x))
        launches = _cutedsl_launches(code, "cfg_scale_kernel")
        self.assertEqual(len(launches), 2, code)
        self.assertEqual(len({name for name, _ in launches}), 1, code)
        counters.clear()
        # With CUDA graphs: record and replay, then warmup, record and replay of the
        # graph recompiled for the new value. A new x each time, so that a launch
        # left out of the recorded graph would show.
        for alpha in (3.0, 3.0, 5.0, 5.0, 5.0):
            c2.alpha = alpha
            x = _rand(1000)
            a, b = cf(x, c1, c2)
            self.assertEqual((a, b), (3 * x, alpha * x))
        self.assertEqual(counters["stats"]["unique_graphs"], 1)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    def test_cutedsl_kernel_default_object_constant_is_copied(self):
        # A launch that leaves out a Constexpr param gets a copy of its default, so
        # it can share a launcher with a launch passing an equal object, and keeps its
        # value when the default object changes later (defaults are not guarded: no
        # recompile), also where a new size compiles the kernel again.
        w = wrap_cutedsl(mul_default_cfg_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel())
            w(x, b, x.numel(), cfg=cfg)
            return a, b

        params = inspect.signature(inspect.unwrap(mul_default_cfg_kernel)).parameters
        default = params["cfg"].default
        cnt = CompileCounterWithBackend("inductor")
        cf = torch.compile(f, backend=cnt, fullgraph=True, dynamic=True)
        x, cfg = _rand(1000), MulCfg(2.0)
        with _counting_compiles() as compile_spy:
            (a, b), (code,) = run_and_get_code(cf, x, cfg)
            self.assertEqual((a, b), (2 * x, 2 * x))
            launches = _cutedsl_launches(code, "mul_default_cfg_kernel")
            self.assertEqual(len(launches), 2, code)
            self.assertEqual(len({name for name, _ in launches}), 1, code)
            try:
                default.alpha = 4.0
                x = _rand(2000)
                a, b = cf(x, cfg)
            finally:
                default.alpha = 2.0
            self.assertEqual(compile_spy.call_count, 2)
        self.assertEqual((a, b), (2 * x, 2 * x))
        self.assertEqual(cnt.frame_count, 1)

    def test_cutedsl_kernel_inference_mode(self):
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

    def test_cutedsl_kernel_specialized_constants_dynamic(self):
        # A Constexpr value computed from a dynamic size is specialized (guarded) and
        # bound by the launcher, while the size passed for the unannotated k stays a
        # runtime scalar: n=16 recompiles, and n=9, with n=8's n // 8, reuses n=8's
        # graph.
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x):
            o1, o2 = torch.empty_like(x), torch.empty_like(x)
            mul_const_w(x, o1, x.numel(), x.shape[0] // 8)
            add_scalar_w(x, o2, x.numel(), x.shape[0])
            return o1, o2

        for n in (8, 16, 8, 9):
            x = _rand(n)
            o1, o2 = f(x)
            self.assertEqual(o1, (n // 8) * x)
            self.assertEqual(o2, x + n)
        self.assertEqual(cnt.frame_count, 2)

    @inductor_config.patch(fx_graph_cache=True, fx_graph_remote_cache=False)
    def test_cutedsl_kernel_bypasses_fx_graph_cache(self):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with (
            fresh_cache(),
            self.assertLogs("torch._inductor.codecache", logging.INFO) as logs,
        ):
            counters.clear()
            self.assertEqual(torch.compile(f)(x, y), (x + y).relu())
            torch._dynamo.reset()
            self.assertEqual(torch.compile(f)(x, y), (x + y).relu())
        # The generated module binds the launcher in-process: the HOP is not
        # cacheable, so each compile bypasses the cache.
        reason = "'Can't cache HigherOrderOperator: cutedsl_kernel_wrapper_functional'"
        messages = [r.getMessage() for r in logs.records]
        bypasses = [m for m in messages if m.startswith("Bypassing FX Graph Cache")]
        self.assertEqual(bypasses, [f"Bypassing FX Graph Cache because {reason}"] * 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)


instantiate_parametrized_tests(CuteDSLHOPTests)
instantiate_parametrized_tests(CuteDSLWrapTests)
instantiate_parametrized_tests(CuteDSLDynamoTests)
instantiate_parametrized_tests(CuteDSLInductorTests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
