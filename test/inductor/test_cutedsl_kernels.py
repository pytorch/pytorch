# Owner(s): ["module: inductor"]
"""Tests for user-defined CuTeDSL kernels (``@cute.jit`` host functions) in PT2.

Mirrors test/inductor/test_triton_kernels.py (the reference suite for user-defined
Triton kernels): direct HOP use, the eager launch, wrap_cutedsl tracing (make_fx,
non-strict export), Dynamo, Inductor (mutation without clones, reinplacing, views,
dynamic shapes, CUDA graphs) and the C++ wrapper (AOTInductor, and torch.compile with
cpp_wrapper).
"""

import contextlib
import copy
import ctypes
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
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock

import torch
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
)
from torch._dynamo.utils import ReinplaceCounters
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_functional,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLKernel,
    get_cutedsl_mutated_tensors,
    TraceableCuteDSLKernelWrapper,
)
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code, run_and_get_cpp_code
from torch._library.cutedsl import wrap_cutedsl
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal import inductor_utils
from torch.testing._internal.common_utils import (
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

    # A Constexpr tuple read on the host: out = x * scales[0].
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

    @cute.jit
    def select_bool_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        flag: cutlass.Boolean,
        stream: cuda.CUstream,
    ):
        _select_k(mX, mOut, n, flag).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # A runtime Float16 scalar: out = x + h.
    @cute.kernel
    def _add_half_k(
        gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, h: cutlass.Float16
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gOut[i] = gX[i] + cutlass.Float32(h)

    @cute.jit
    def add_half_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        n: cutlass.Int32,
        h: cutlass.Float16,
        stream: cuda.CUstream,
    ):
        _add_half_k(mX, mOut, n, h).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # Host functions that differ only in a captured list: out = x * c[0] + c[1].
    def make_affine_kernel(c):
        @cute.kernel
        def _affine_k(gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 128 + tidx
            if i < n:
                gOut[i] = gX[i] * c[0] + c[1]

        @cute.jit
        def affine_kernel(
            mX: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream
        ):
            _affine_k(mX, mOut, n).launch(
                grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
            )

        return affine_kernel

    # A runtime scalar of each integer type export_to_c takes, and a Float64:
    # out = [a, b, c, d, e, f, g] (g's bits), outf = [h].
    @cute.kernel
    def _store_scalars_k(
        gOut: cute.Tensor,
        gOutF: cute.Tensor,
        a: cutlass.Int8,
        b: cutlass.Int16,
        c: cutlass.Uint8,
        d: cutlass.Uint16,
        e: cutlass.Uint32,
        f: cutlass.Int64,
        g: cutlass.Uint64,
        h: cutlass.Float64,
    ):
        gOut[0] = cutlass.Int64(a)
        gOut[1] = cutlass.Int64(b)
        gOut[2] = cutlass.Int64(c)
        gOut[3] = cutlass.Int64(d)
        gOut[4] = cutlass.Int64(e)
        gOut[5] = f
        gOut[6] = cutlass.Int64(g)
        gOutF[0] = h

    @cute.jit
    def store_scalars_kernel(
        mOut: cute.Tensor,
        mOutF: cute.Tensor,
        a: cutlass.Int8,
        b: cutlass.Int16,
        c: cutlass.Uint8,
        d: cutlass.Uint16,
        e: cutlass.Uint32,
        f: cutlass.Int64,
        g: cutlass.Uint64,
        h: cutlass.Float64,
        stream: cuda.CUstream,
    ):
        _store_scalars_k(mOut, mOutF, a, b, c, d, e, f, g, h).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )

    # Reads one element, so mX can be a huge expanded tensor: out[0] = x[0].
    @cute.kernel
    def _copy_first_k(gX: cute.Tensor, gOut: cute.Tensor):
        gOut[0] = gX[0]

    @cute.jit
    def copy_first_kernel(mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        _copy_first_k(mX, mOut).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

    # Exported with _abi_specs under the prefix cutedsl_fixture, it gives
    # _EXPORTED_HEADER: a tensor with dynamic sizes and a dynamic stride, a static
    # one, a scalar of each cutlass Numeric type export_to_c supports, a Constexpr
    # and the stream.
    @cute.jit
    def abi_kernel(
        mX: cute.Tensor,
        mOut: cute.Tensor,
        i8: cutlass.Int8,
        i16: cutlass.Int16,
        i32: cutlass.Int32,
        i64: cutlass.Int64,
        u8: cutlass.Uint8,
        u16: cutlass.Uint16,
        u32: cutlass.Uint32,
        u64: cutlass.Uint64,
        flag: cutlass.Boolean,
        f32: cutlass.Float32,
        tf32: cutlass.TFloat32,
        f64: cutlass.Float64,
        f16: cutlass.Float16,
        alpha: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        _copy_first_k(mX, mOut).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

    # Parameters named like the identifiers a C++ launcher named after the host
    # function's parameters would declare (its stream, a tensor's dynamic size, the
    # launch block's locals): out = x + k.
    @cute.jit
    def colliding_names_kernel(
        stream: cute.Tensor,
        cutedsl_arg0: cute.Tensor,
        stream_size0: cutlass.Int32,
        cutedsl_status: cutlass.Int32,
        s: cuda.CUstream,
    ):
        _add_scalar_k(stream, cutedsl_arg0, stream_size0, cutedsl_status).launch(
            grid=[(stream_size0 + 127) // 128, 1, 1], block=[128, 1, 1], stream=s
        )


DEVICE = "cuda"

if HAS_CUTEDSL:
    # Module-level (global) wrappers: Dynamo must handle a global
    # TraceableCuteDSLKernelWrapper, and they are reused across tests, between which
    # torch._dynamo.reset() runs (it resets the Triton kernel side table; the CuTeDSL
    # records these wrappers hold must keep resolving).
    add_w = wrap_cutedsl(add_kernel, mutates_args=["mOut"])
    scale_inplace_w = wrap_cutedsl(scale_inplace_kernel, mutates_args=["mX"])
    add_one_2d_w = wrap_cutedsl(add_one_2d_kernel, mutates_args=["mOut"])
    mul_const_w = wrap_cutedsl(mul_const_kernel, mutates_args=["mOut"])
    add_sub_w = wrap_cutedsl(add_sub_kernel, mutates_args=["mSum", "mDiff"])
    add_scalar_w = wrap_cutedsl(add_scalar_kernel, mutates_args=["mOut"])

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

    same_name_add_w = wrap_cutedsl(same_name_add_kernel, mutates_args=["mOut"])
    same_name_sub_w = wrap_cutedsl(same_name_sub_kernel, mutates_args=["mOut"])


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
    from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

    return CUTEDSL_COMPILE_LOCK


@contextlib.contextmanager
def _counting_compiles():
    """A spy on cute.compile, with the eager compile caches (the shared one and the
    eager launchers) emptied for the duration, so that its call count is the number
    of compiles eager launches did."""
    from unittest import mock

    from torch._inductor.runtime import cutedsl_user_kernel

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


@requires_cutedsl
class CuteDSLToyKernelTests(TestCase):
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

    def test_side_table_dedupes_constant_args(self):
        # The table is never reset and every eager launch through the HOP adds its
        # constants, so interchangeable constants share one entry (see constant_key),
        # which holds a snapshot of the constants it was added with.
        import collections

        Tile = collections.namedtuple("Tile", ["m", "n"])

        @dataclasses.dataclass(frozen=True)
        class FrozenCfg:
            alpha: float

        class Cfg:  # a plain config object: compared and hashed by identity
            def __init__(self, alpha):
                self.alpha = alpha

        table = cutedsl_kernel_side_table
        idx = table.add_constant_args({"alpha": 2, "k": None})
        self.assertEqual(table.add_constant_args({"k": None, "alpha": 2}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": 2.0, "k": None}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": True, "k": None}), idx)

        def add(c):
            return table.add_constant_args({"c": c})

        # By type and content all the way down: inside containers and dataclasses
        # too, equal values of other types, and floats that differ only in sign, are
        # other constants.
        for a, b in (
            ([1, 2], (1, 2)),
            (Tile(1, 2), Tile(1.0, 2)),
            (Tile(1, 2), Tile(True, 2)),
            (Tile(0.0, 1), Tile(-0.0, 1)),
            (FrozenCfg(0.0), FrozenCfg(-0.0)),
        ):
            self.assertNotEqual(add(a), add(b))
        # Mutable values too (a namedtuple holding a list, a plain object), stored as
        # they were when added: mutating the caller's value changes neither the entry
        # nor the entry its old contents map to.
        tile, cfg = Tile([128], 64), Cfg(1.0)
        tiles, cfgs = add(tile), add(cfg)
        self.assertEqual((add(Tile([128], 64)), add(Cfg(1.0))), (tiles, cfgs))
        tile.m.append(32)
        cfg.alpha = 2.0
        self.assertEqual(table.get_constant_args(tiles)["c"], Tile([128], 64))
        self.assertEqual(table.get_constant_args(cfgs)["c"].alpha, 1.0)
        self.assertEqual((add(Tile([128], 64)), add(Cfg(1.0))), (tiles, cfgs))
        self.assertNotEqual(add(tile), tiles)
        self.assertNotEqual(add(cfg), cfgs)

        # An object with __slots__ attributes, even one with a __dict__ too, is keyed
        # as itself: keyed by its __dict__, objects differing only in a slot would
        # share one entry.
        class SlotBase:
            __slots__ = ("alpha",)

        class SlotCfg(SlotBase):
            def __init__(self, alpha):
                self.alpha = alpha

        slot_cfgs = [table.get_constant_args(add(SlotCfg(a)))["c"] for a in (1.0, 2.0)]
        self.assertEqual([c.alpha for c in slot_cfgs], [1.0, 2.0])
        with self.assertRaisesRegex(TypeError, "type set is unhashable"):
            add({1, 2})

    def test_signed_zero_and_nan_constants(self):
        # Float constants are keyed by their bits, in the side table and the eager
        # compile cache alike: -0.0 and 0.0 are different constants (x * alpha has
        # the sign of alpha for x > 0), and every NaN is the same constant.
        import math

        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

        table = cutedsl_kernel_side_table
        pos, neg = (table.add_constant_args({"alpha": a}) for a in (0.0, -0.0))
        self.assertNotEqual(pos, neg)
        self.assertEqual(math.copysign(1, table.get_constant_args(neg)["alpha"]), -1)
        self.assertNotEqual(
            table.add_constant_args({"c": cutlass.Float32(0.0)}),
            table.add_constant_args({"c": cutlass.Float32(-0.0)}),
        )
        nan = table.add_constant_args({"alpha": float("nan")})
        self.assertEqual(table.add_constant_args({"alpha": float("nan")}), nan)

        x = torch.rand(1000, device=DEVICE) + 1
        for alpha in (0.0, -0.0):
            out = torch.empty_like(x)
            launch_cutedsl_kernel(
                mul_const_kernel, {"mX": x, "mOut": out, "n": 1000, "alpha": alpha}
            )
            self.assertEqual(torch.signbit(out), torch.signbit(x * alpha))

    def test_cyclic_constants_are_refused(self):
        # constant_key and _snapshot walk a constant's contents, so a constant that
        # refers to itself is refused with the cycle named, in the side table and the
        # eager launch alike, instead of recursing until a RecursionError. An object,
        # dict or list reached twice on different paths is no cycle.
        import logging

        from torch._higher_order_ops.user_kernel_hop import _snapshot, constant_key
        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

        class Cfg:
            def __init__(self, alpha, **attrs):
                self.alpha = alpha
                vars(self).update(attrs)

        @dataclasses.dataclass
        class Node:
            next: object = None

        cfg = Cfg(2.0, parent=Cfg(1.0))
        cfg.parent.child = cfg
        node = Node()
        node.next = Node(node)
        items = [1.0]
        items.append(items)
        table = {}
        table["self"] = table
        keyed = Cfg(1.0)
        keyed.lookup = {keyed: 1}
        # A logger of its own manager: its manager's loggerDict holds it.
        log = logging.Manager(logging.RootLogger(logging.WARNING)).getLogger("cfg")
        cycles = (
            (cfg, "Cfg", r"\.parent\.child"),
            (node, "Node", r"\.next\.next"),
            ((0, Cfg(1.0, items=items)), "list", r"\[1\]"),
            (table, "dict", r"\['self'\]"),
            (keyed, "Cfg", r"\.lookup\.keys\(\)\[0\]"),
            (Cfg(1.0, log=log), "Logger", r"\.manager\.loggerDict\['cfg'\]"),
        )
        for value, name, path in cycles:
            msg = f"^a constant kernel argument of type {name} refers to itself "
            msg += f"through {path}$"
            for walk in (constant_key, _snapshot):
                with self.assertRaisesRegex(TypeError, msg):
                    walk(value)
            with self.assertRaisesRegex(TypeError, msg):
                cutedsl_kernel_side_table.add_constant_args({"cfg": value})
        # A logger of the process: what its walk meets first depends on the process.
        with self.assertRaisesRegex(TypeError, "^a constant kernel argument of type"):
            constant_key(Cfg(1.0, log=logging.getLogger(__name__)))

        shared = Cfg(1.0, lookup={"k": [0.5]})
        twice = Cfg(2.0, a=shared, b=[shared], c=shared.lookup)
        snapshot = _snapshot(twice)
        self.assertEqual(constant_key(snapshot), constant_key(twice))
        self.assertTrue(all(c is not shared for c in (snapshot.a, snapshot.b[0])))
        self.assertIsNot(snapshot.c, shared.lookup)

        x = _rand(1000)
        args = {"mX": x, "mOut": torch.empty_like(x), "n": 1000, "cfg": cfg}
        with self.assertRaisesRegex(TypeError, r"type Cfg refers to itself"):
            launch_cutedsl_kernel(mul_cfg_kernel, args)

    def test_launch_checks_argument_names(self):
        # The eager launch binds every parameter but the stream by name: missing ones
        # without a default, a name that is not a parameter (here a typo of the
        # defaulted scale, which would launch with scale=1.0) and the stream are
        # refused, with the kernel and the parameters named.
        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

        x, y, s, d = (_rand(1000) for _ in range(4))
        args = {"mX": x, "mY": y, "mSum": s, "mDiff": d, "n": 1000}
        missing = {k: v for k, v in args.items() if k not in ("mX", "n")}
        for bad_args, msg in (
            (missing, r"missing arguments \['mX', 'n'\]"),
            ({**args, "scael": 0.5}, r"got an unexpected argument 'scael'"),
            ({**args, "stream": _stream()}, r"got a value for its stream parameter"),
        ):
            with self.assertRaisesRegex(TypeError, f"kernel add_sub_kernel: {msg}"):
                launch_cutedsl_kernel(add_sub_kernel, bad_args)
        launch_cutedsl_kernel(add_sub_kernel, {**args, "scale": 0.5})
        self.assertEqual((s, d), ((x + y) * 0.5, (x - y) * 0.5))

    def test_get_host_fn_info_rejects_unlaunchable_signatures(self):
        from torch._higher_order_ops.cutedsl_kernel_wrap import get_host_fn_info

        def var_args(mX: cute.Tensor, stream: cuda.CUstream, *rest):
            pass

        def var_kwargs(mX: cute.Tensor, stream: cuda.CUstream, **rest):
            pass

        def string_annotation(mX, stream: cuda.CUstream):
            pass

        # What `from __future__ import annotations` would leave.
        string_annotation.__annotations__["mX"] = "cute.Tensor"
        varargs_msg = r"\*args and \*\*kwargs parameters are not supported"
        for fn, msg in (
            (no_stream_kernel, r"exactly one parameter annotated cuda\.CUstream"),
            (two_stream_kernel, r"found \['stream', 'stream2'\]"),
            (var_args, varargs_msg),
            (var_kwargs, varargs_msg),
            (string_annotation, "annotation of 'mX' is a string"),
        ):
            with self.assertRaisesRegex(ValueError, msg):
                get_host_fn_info(fn)

    def test_launch_compiles_once_per_argument_signature(self):
        # A compiled CuTeDSL kernel checks nothing at launch, so the eager compile
        # cache keys on the dtype, shape, stride and alignment of every tensor: a
        # change in any of them compiles again, and every result must be right.
        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

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
        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel
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
        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

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
        # A name that is not a parameter (a typo) or is the stream would never be
        # cloned and written back, so a write would be lost: the record refuses it.
        # An empty write set does not parse the signature.
        msg = (
            r"add_kernel: mutates_args names '{}', but its parameters other than the "
            r"stream 'stream' are \('mX', 'mY', 'mOut', 'n'\)"
        )
        for names, bad in ((("mout",), "mout"), (["mOut", "stream"], "stream")):
            with self.assertRaisesRegex(ValueError, msg.format(bad)):
                CuteDSLKernel(add_kernel, names)
        with self.assertRaisesRegex(ValueError, 'must be "unknown" or a sequence'):
            CuteDSLKernel(add_kernel, "mOut")
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
        # Each written argument is cloned and written back on its own, so two written
        # arguments that may overlap would lose a write: functionalization refuses the
        # same tensor twice, and views that overlap whatever their shapes, offsets and
        # dtypes. Views of one base that do not overlap, interleaved ones included
        # (the q, k and v columns of a fused buffer), are written back correctly,
        # unless functionalization copies views (remove="mutations_and_views"): the
        # copies do not show where the views sit, so any two views are refused.
        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(add_kernel, "unknown")
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def trace(f, *args, tracing_mode="real"):
            if api == "python":
                with FunctionalTensorMode():
                    f_func = PythonFunctionalizeAPI().functionalize(f)
                    return make_fx(f_func, tracing_mode=tracing_mode)(*args)
            remove = "mutations_and_views" if api == "cpp_view_copies" else "mutations"
            f_func = torch.func.functionalize(f, remove=remove)
            return make_fx(f_func, tracing_mode=tracing_mode)(*args)

        def in_place(x, y):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, x)
            )
            return x

        def reshaped(x, y):
            out = x.view(10, 100)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return x

        def shifted(x, y):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(x[1:], y[1:], x[:-1]),
            )
            return x

        def retyped(x, y):
            out = x.view(torch.float16)[1000:]  # the bytes of x[500:]
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, out)
            )
            return x

        def overlapping_columns(x, y):
            # Rows 1 to 3 of column 2 are in both: each row of out starts inside a
            # row of the other, though not inside the first one.
            base = x.view(10, 100)
            out = base[1:5, 2:5]
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(base[:4, :3], y[:12].view(4, 3), out),
            )
            return x

        def views(base, y):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(base[0], y, base[1]),
            )
            return base

        def columns(qkv):
            d = qkv.shape[1] // 3
            q, k, v = (qkv.narrow(1, i * d, d) for i in range(3))
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(q, k, v)
            )
            return qkv

        def empty(x, y):
            # An empty view overlaps nothing, wherever it sits.
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(x, y, x[500:500]),
            )
            return x

        x, y = _rand(1000), _rand(1000)
        msg = "'mX' and 'mOut' are the same tensor or views that may overlap"
        for f in (in_place, reshaped, shifted, retyped, overlapping_columns):
            with self.assertRaisesRegex(RuntimeError, msg):
                trace(f, x.clone(), y)
        base = torch.stack([x, torch.zeros_like(x)])
        qkv = _rand(40, 75)
        q, k = qkv[:, :25], qkv[:, 25:50]
        if api == "cpp_view_copies":
            with self.assertRaisesRegex(RuntimeError, msg):
                trace(views, base.clone(), y)
            with self.assertRaisesRegex(RuntimeError, "'mX' and 'mY'"):
                trace(columns, qkv.clone())
        else:
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

    def test_overlap_check_on_symbolic_sizes(self):
        # Views of one element size are compared by compute_overlapping_tensors,
        # which decides by the size hints and guards on the decision: x[:n] and x[m:]
        # (n = 10, m = 20) are disjoint while n <= m. Views of different element sizes
        # are compared by their byte ranges, without guards: x[:n] ends where the
        # float16 view from element 2n starts, and may overlap the one from 2m. An
        # unbacked size, which cannot be guarded on, counts as an overlap, even in
        # x[:u] and x[u:].
        from torch._higher_order_ops.cutedsl_kernel_wrap import _may_overlap
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        results = []

        def f(x, y, z):
            n, m = y.shape[0], z.shape[0]
            halves = x.view(torch.float16)
            head, tail = x[:n], x[m:]
            halves_n, halves_m = halves[2 * n :], halves[2 * m :]
            shape_env = n.node.shape_env
            guards = len(shape_env.guards)
            results.extend(_may_overlap(head, h) for h in (halves_n, halves_m))
            self.assertEqual(len(shape_env.guards), guards)
            results.append(_may_overlap(head, tail))
            return x

        gm = make_fx(f, tracing_mode="symbolic")(_rand(100), _rand(10), _rand(20))
        self.assertEqual(results, [False, True, False])
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]
        holds = gm.shape_env.evaluate_guards_for_args
        self.assertTrue(holds(inputs, [_rand(100), _rand(10), _rand(20)]))
        self.assertFalse(holds(inputs, [_rand(100), _rand(20), _rand(10)]))

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u = shape_env.create_unbacked_symint()
            x = torch.empty(100, device=DEVICE)
            self.assertTrue(_may_overlap(x[:u], x[u:]))

    @parametrize("case", ["e8m0", "complex", "meta", "sparse", "fp8"])
    def test_cutedsl_hop_functional_decomposed_before_type_fallback(self, case):
        # Mirrors test_triton_kernel_functional_decomposition_before_type_fallback: the
        # post-grad decomposition turns the functional HOP into the mutation HOP
        # whatever its tensors' types (the lowering checks them afterwards), instead
        # of skipping the node and failing its "was not removed" check.
        from unittest import mock

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

    def test_wrap_cutedsl_mutates_args_iterator(self):
        # A one-shot iterable is read once, not consumed by its validation.
        for names in ((n for n in ["mOut"]), iter(["mOut"]), map(str, ["mOut"])):
            w = wrap_cutedsl(add_kernel, mutates_args=names)
            self.assertEqual(w.mutates_args, ("mOut",))

    def test_wrap_cutedsl_make_fx_nested_symint_specialized(self):
        # A SymInt inside a Constexpr tuple is specialized like a bare one, so the
        # side table only holds concrete values.
        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), (x.shape[0] // 250,))
            return out

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        (node,) = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        scales = _scalar_args(node)["scales"]
        self.assertEqual(scales, (4,))
        self.assertIs(type(scales[0]), int)
        self.assertEqual(gm(x), 4 * x)

    def test_wrap_cutedsl_make_fx_symint_in_object_specialized(self):
        # The side table keys a dataclass instance or plain object by its contents
        # (see _object_state), so a SymInt inside one, given to a Constexpr or an
        # unannotated parameter, is specialized with a guard as inside a tuple, in a
        # copy of the object.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        mul_default_cfg_w = wrap_cutedsl(mul_default_cfg_kernel, mutates_args=["mOut"])
        mul_cfg_w = wrap_cutedsl(mul_cfg_kernel, mutates_args=["mOut"])
        cfgs = []

        def f(x):
            cfgs[:] = [MulCfg(x.shape[0] // 250), Cfg(x.shape[0] // 500)]
            out, out2 = torch.empty_like(x), torch.empty_like(x)
            mul_default_cfg_w(x, out, x.numel(), cfg=cfgs[0])
            mul_cfg_w(x, out2, x.numel(), cfgs[1])
            return out, out2

        x = _rand(1000)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        nodes = _hop_nodes(gm, cutedsl_kernel_wrapper_mutation)
        captured = [_scalar_args(node)["cfg"] for node in nodes]
        self.assertEqual([type(cfg) for cfg in captured], [MulCfg, Cfg])
        self.assertEqual([cfg.alpha for cfg in captured], [4, 2])
        self.assertEqual([type(cfg.alpha) for cfg in captured], [int, int])
        self.assertEqual([type(cfg.alpha) for cfg in cfgs], [torch.SymInt] * 2)
        size = gm.graph.find_nodes(op="placeholder")[0].meta["val"].shape[0]
        guards = [str(guard.expr) for guard in gm.shape_env.guards]
        self.assertIn(f"Eq(({size}//250), 4)", guards)
        self.assertIn(f"Eq(({size}//500), 2)", guards)
        self.assertEqual(gm(x), (4 * x, 2 * x))

    def test_wrap_cutedsl_error_tensor_in_container(self):
        # Only top-level tensors become graph arguments.
        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])
        x = _rand(1000)
        with self.assertRaisesRegex(RuntimeError, "inside a container"):
            w(x, torch.empty_like(x), 1000, (x,))

    def test_wrap_cutedsl_error_tensor_in_object(self):
        # A dataclass instance or plain object is a container too: a tensor inside
        # one would be baked into a make_fx graph as a fake tensor constant, and a
        # cute tensor or stream inside one would reach the DSL. A tensor whose class
        # does not hash is still a tensor, not a plain object.
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

        x = _rand(1000)
        unhashable_x = x.as_subclass(UnhashableTensor)
        cute_x, stream = from_dlpack(x, assumed_align=16), _stream()
        in_container = "'cfg' holds a tensor inside a container"
        cases = (
            (MulCfg, in_container),
            (lambda x: Cfg([x]), in_container),
            (lambda x: MulCfg(unhashable_x), in_container),
            (lambda x: MulCfg(cute_x), r"not _Tensor \('cfg'\)"),
            (lambda x: Cfg(stream), r"not CUstream \('cfg'\)"),
        )
        for i, (make_cfg, error) in enumerate(cases):
            fn = functools.partial(f, make_cfg=make_cfg)
            with self.subTest(case=i):
                with self.assertRaisesRegex(RuntimeError, error):
                    make_fx(fn, tracing_mode="fake")(x)
                with self.assertRaisesRegex(RuntimeError, error):
                    fn(x)

    def test_wrap_cutedsl_error_cyclic_constant(self):
        # A constant that refers to itself (a config object with a back-pointer, one
        # holding a logging.Logger, whose manager leads back to it, or a list holding
        # itself) is refused instead of walked until a RecursionError: under make_fx,
        # before a SymInt in it is specialized in a copy, and eagerly, also on the
        # fast path.
        from torch._library.triton import set_wrap_triton_enabled

        class Parent:
            def __init__(self, child):
                self.child = child

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha
                self.parent = Parent(self)

        class LoggingCfg:
            def __init__(self, alpha):
                self.alpha = alpha
                self.log = logging.getLogger(__name__)

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
            (Cfg, "Cfg -> Parent -> Cfg"),
            (LoggingCfg, ""),
            (self_list, "list -> list"),
        )
        for make_cfg, cycle in cases:
            fn = functools.partial(f, make_cfg=make_cfg)
            error = f"contains a reference cycle: {cycle}"
            with self.subTest(cfg=make_cfg.__name__):
                with self.assertRaisesRegex(RuntimeError, error):
                    make_fx(fn, tracing_mode="symbolic")(x)
                with self.assertRaisesRegex(RuntimeError, error):
                    fn(x)
                with set_wrap_triton_enabled(False):
                    with self.assertRaisesRegex(RuntimeError, error):
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
            with self.assertRaisesRegex(RuntimeError, r"add_kernel\(\) missing"):
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
        from torch._library.triton import set_wrap_triton_enabled

        hop_calls = []
        hop_type = type(cutedsl_kernel_wrapper_mutation)
        hop_call = hop_type.__call__

        def spy_hop_call(hop, *args, **kwargs):
            hop_calls.append(kwargs["kernel_idx"])
            return hop_call(hop, *args, **kwargs)

        scale_by_first_w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])
        x, y = _rand(1000), _rand(1000)
        out, out2 = torch.zeros_like(x), torch.zeros_like(x)
        with mock.patch.object(hop_type, "__call__", spy_hop_call):
            add_w(x, y, out, 1000)
            self.assertEqual(hop_calls, [add_w.kernel_idx])  # the spy sees the HOP
            with set_wrap_triton_enabled(False):
                add_w(x, y, out2, 1000)
                with self.assertRaisesRegex(RuntimeError, "do not pass 'stream'"):
                    add_w(x, y, out2, 1000, stream=_stream())
                with self.assertRaisesRegex(RuntimeError, "inside a container"):
                    scale_by_first_w(x, out2, 1000, (x,))
        self.assertEqual(hop_calls, [add_w.kernel_idx])
        self.assertEqual(out, x + y)
        self.assertEqual(out2, x + y)

    def test_wrap_cutedsl_error_stream_passed(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(RuntimeError, "stream"):
            add_w(x, y, out, 16, stream=_stream())
        with self.assertRaisesRegex(RuntimeError, "stream"):
            add_w(x, y, out, 16, _stream())

    def test_wrap_cutedsl_error_stream_params(self):
        # Exactly one cuda.CUstream parameter is required.
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        for fn in (no_stream_kernel, two_stream_kernel):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(RuntimeError, "(?i)stream"):
                    wrap_cutedsl(fn, mutates_args=["mOut"])(x, y, out, 16)

    def test_wrap_cutedsl_error_bad_mutates_args(self):
        # Rejected by wrap_cutedsl itself, not at the first call of the wrapper.
        for bad, pattern in (
            (["not_a_param"], "not_a_param"),
            (["n"], "'n'"),  # a scalar parameter cannot be written
            (["stream"], "stream"),
            ("mOut", "mutates_args"),  # a bare string other than "unknown"
            (None, "mutates_args must be a sequence"),
        ):
            with self.subTest(mutates_args=bad):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    wrap_cutedsl(add_kernel, mutates_args=bad)

    def test_wrap_cutedsl_error_not_cute_jit(self):
        def plain(mX, mY, mOut, n, stream: cuda.CUstream):
            pass

        with self.assertRaisesRegex(RuntimeError, r"(?i)cute\.jit"):
            wrap_cutedsl(plain, mutates_args=["mOut"])
        # A @cute.kernel device function is not a host function either.
        with self.assertRaisesRegex(RuntimeError, r"(?i)cute\.jit"):
            wrap_cutedsl(_add_k, mutates_args=["gOut"])

    def test_wrap_cutedsl_error_unwrapped_host_fn(self):
        # inspect.unwrap(host_fn) is the user's undecorated function. Whether it
        # still carries _dsl_cls depends on whether it was compiled before (the DSL
        # deletes it then); either way it is not a @cute.jit function.
        _reference_launch(
            add_kernel, _rand(8), _rand(8), torch.zeros(8, device=DEVICE), 8
        )
        with self.assertRaisesRegex(RuntimeError, r"(?i)cute\.jit"):
            wrap_cutedsl(inspect.unwrap(add_kernel), mutates_args=["mOut"])

    def test_wrap_cutedsl_error_cute_tensor_arg(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(RuntimeError, "torch.Tensor"):
            add_w(from_dlpack(x, assumed_align=16), y, out, 16)

    def test_wrap_cutedsl_error_bad_call(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(RuntimeError, "(?i)missing"):
            add_w(x, y, out)
        with self.assertRaisesRegex(RuntimeError, "not_a_param"):
            add_w(x, y, out, 16, not_a_param=1)
        with self.assertRaisesRegex(RuntimeError, "(?i)multiple values"):
            add_w(x, y, out, 16, n=16)
        with self.assertRaisesRegex(RuntimeError, "'n'"):
            add_w(x, y, out, x)  # a tensor for a scalar parameter

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
            self.assertEqual(_scalar_args(node).get("scale"), scale)
            s, d = gm(x, y)
            self.assertEqual((s, d), ((x + y) * expect, (x - y) * expect))

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

    def test_bad_wrap_cutedsl_calls_graph_break(self):
        # A call that eager refuses too (of a wrapper or a @cute.kernel function, which
        # are not @cute.jit functions, or with missing or extra arguments) graph-breaks
        # as a user error, and the eager call then raises; a non-constant
        # mutates_args, which eager takes, is the one call Dynamo could support.
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

        x, y = _rand(1000), _rand(1000)
        not_jit = "wrap_cutedsl: host_fn is not a @cute.jit function"
        improper = "wrap_cutedsl: improper arguments"
        non_constant = "wrap_cutedsl: non-constant mutates_args"
        for args, kwargs, error, gb_type in (
            ((add_w,), {"mutates_args": ["mOut"]}, RuntimeError, not_jit),
            ((_add_k,), {"mutates_args": ["gOut"]}, RuntimeError, not_jit),
            ((add_kernel,), {}, TypeError, improper),
            ((add_kernel, ["mOut"]), {}, TypeError, improper),
            ((add_kernel,), {"mutates_args": ["mOut"], "x": 1}, TypeError, improper),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                torch._dynamo.reset()
                with self.assertRaises(error):
                    f(x, y, args, kwargs)
                with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
                    cf = torch.compile(f, backend="eager", fullgraph=True)
                    cf(x, y, args, kwargs)
                self.assertEqual(cm.exception.gb_type, gb_type)
                self.assertIn(user_error, str(cm.exception))
                self.assertNotIn(supportable, str(cm.exception))
                with self.assertRaises(error) as cm:
                    torch.compile(f, backend="eager")(x, y, args, kwargs)
                self.assertNotIsInstance(
                    cm.exception, torch._dynamo.exc.TorchDynamoException
                )
        self.assertEqual(generator(x, y), x + y)
        with self.assertRaises(torch._dynamo.exc.Unsupported) as cm:
            torch.compile(generator, backend="eager", fullgraph=True)(x, y)
        self.assertEqual(cm.exception.gb_type, non_constant)
        self.assertIn(supportable, str(cm.exception))
        self.assertNotIn(user_error, str(cm.exception))
        self.assertEqual(torch.compile(generator, backend="eager")(x, y), x + y)

    def test_inline_wrap_cutedsl_bad_mutates_args(self):
        # Validated as in eager, None included (it does not stand for the write set
        # of a raw call): under fullgraph the call graph-breaks with eager's reason,
        # otherwise the eager wrap_cutedsl raises it.
        def f(x, y, mutates_args):
            out = torch.empty_like(x)
            wrap_cutedsl(add_kernel, mutates_args=mutates_args)(x, y, out, x.numel())
            return out

        x, y = _rand(16), _rand(16)
        for bad, pattern in (
            (None, "mutates_args must be a sequence"),
            ("mOut", "mutates_args must be a sequence"),
            (["not_a_param"], "not_a_param"),
            (["n"], "'n'"),
        ):
            with self.subTest(mutates_args=bad):
                torch._dynamo.reset()
                with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, pattern):
                    torch.compile(f, backend="eager", fullgraph=True)(x, y, bad)
                with self.assertRaisesRegex(RuntimeError, pattern) as cm:
                    torch.compile(f, backend="eager")(x, y, bad)
                self.assertNotIsInstance(
                    cm.exception, torch._dynamo.exc.TorchDynamoException
                )

    def test_inline_wrapper_live_across_graph_break(self):
        # A wrapper made in the frame is rebuilt at the graph break, so both parts
        # compile instead of the frame falling back to eager.
        def f(x, y):
            w = wrap_cutedsl(add_kernel, mutates_args=["mOut"])
            out = torch.zeros_like(x)
            torch._dynamo.graph_break()
            w(x, y, out, x.numel())
            return out

        cnt = CompileCounter()
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, backend=cnt)(x, y), x + y)
        self.assertEqual(cnt.frame_count, 2)

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
        # value as itself, guarded on its identity.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        cfg, alpha = Cfg(3.0), cutlass.Float32(2.0)
        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel(), cfg)
            mul_const_w(x, b, x.numel(), alpha)
            return a, b

        x = _rand(1000)
        a, b = torch.compile(f, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(a, 3 * x)
        self.assertEqual(b, 2 * x)

    def test_object_constant_modified_outside_the_region(self):
        # The side table keeps a snapshot of the object, whose contents are guarded
        # down through attributes (their names too, even ones that are not Python
        # names), lists and dicts (their keys too): a change made between calls
        # recompiles, so the kernel sees it as in eager, while the same contents, in
        # this object or a copy, reuse the graph.
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
        for make in (Cfg, MulCfg, Override, Nested, Keyed, namespace):
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
            pass

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

    def test_object_constant_holding_a_tensor_graph_breaks(self):
        # As in a tuple or a list, a tensor or a DSL object inside an object
        # constant (a dict key included) would be baked into the side table, which
        # is never reset and so also keeps it alive: the call graph-breaks, and a raw
        # call runs eagerly.
        class Cfg:
            def __init__(self, alpha, held):
                self.alpha = alpha
                self.held = held

        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

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
        for held, pattern in (
            (x, "'cfg' holds a tensor inside a container"),
            ({x: None}, "'cfg' holds a tensor inside a container"),
            (from_dlpack(x), "not _Tensor"),
            (_stream(), "not CUstream"),
        ):
            for f in (wrapped, raw):
                with self.subTest(held=type(held).__name__, f=f.__name__):
                    with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, pattern):
                        cf = torch.compile(f, backend="eager", fullgraph=True)
                        cf(x, Cfg(3.0, held))
        self.assertEqual(torch.compile(raw, backend="eager")(x, Cfg(3.0, x)), 3 * x)

    def test_constant_that_refers_to_itself_graph_breaks(self):
        # A constant that holds itself cannot be guarded on its contents (nor keyed
        # by them in the side table): a config object with a back-pointer, one
        # holding a logging.Logger (whose manager's loggerDict holds the logger), a
        # list holding itself, made outside the region or in it. The call
        # graph-breaks instead of recursing forever, and a raw call runs eagerly.
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
        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])
        w_scales = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])

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

        x = _rand(1000)
        back_pointer_pattern = (
            r"the constant in 'cfg' refers to itself: "
            r"L\['cfg'\]\.__dict__\['parent'\]\.__dict__\['child'\] is L\['cfg'\]"
        )
        logger_pattern = "the constant in 'cfg' refers to itself"
        list_pattern = r"of type list refers to itself through \[1\]"
        for f, arg, pattern in (
            (wrapped, back_pointer, back_pointer_pattern),
            (raw, back_pointer, back_pointer_pattern),
            (wrapped, with_logger, logger_pattern),
            (raw, with_logger, logger_pattern),
            (list_argument, scales, list_pattern),
            (list_made_in_the_region, scales, list_pattern),
        ):
            with self.subTest(f=f.__name__, arg=type(arg).__name__):
                with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, pattern):
                    torch.compile(f, backend="eager", fullgraph=True)(x, arg)
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
            torch._dynamo.exc.Unsupported, "cannot pass the Scale in 'scales'"
        ):
            torch.compile(g, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(g, backend="eager")(x), 2 * x)

    def test_function_constants(self):
        # A function from outside the region is passed as itself and guarded on its
        # identity: its own guard (on its code) also passes for another function
        # made by the same def. A @cute.jit function (e.g. a callback) is passed the
        # same way. Dynamo has no object for a bound method (one is made on each
        # access), so passing one graph-breaks.
        def make_cfg(alpha):
            def cfg():
                pass

            cfg.alpha = alpha
            return cfg

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
        three = make_cfg(3.0)
        for cfg, frame_count in (
            (three, 1),
            (three, 1),
            (make_cfg(5.0), 2),
            (jit_cfg, 3),
        ):
            self.assertEqual(cf(x, cfg), cfg.alpha * x)
            self.assertEqual(cnt.frame_count, frame_count)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "cannot pass the method in 'cfg'"
        ):
            torch.compile(g, backend="eager", fullgraph=True)(x, Holder())

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
        # pointer's own, at most 16 bytes); an option that changes the compiled kernel
        # graph-breaks.
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

        x, y = _rand(1000), _rand(1000)
        cf = torch.compile(f, backend="aot_eager", fullgraph=True)
        self.assertEqual(cf(x, y), x + y)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "use_32bit_stride"):
            torch.compile(g, backend="eager", fullgraph=True)(x, y)

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
        # the capture of a raw call.
        def f(x, y):
            out = torch.empty_like(x)
            add_kernel(x, y, out, x.numel(), cutlass_torch.current_stream())
            return out

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, r"pass from_dlpack\(tensor\) for 'mX'"
        ):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)

    def test_raw_call_arguments_live_across_graph_break(self):
        # from_dlpack(t) and current_stream() made before a graph break are made
        # again after it. The resumed frame then sees DSL objects made outside its
        # region, so the call graph-breaks there (the reason checked below only comes
        # from tracing the rebuilt objects) and runs eagerly.
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
        self.assertTrue(any("current_stream()" in r for r in reasons), reasons)
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
        # repr(), str() and f-strings of a from_dlpack(t) or current_stream() value made
        # in the region graph-break like any other read, and then run on the real
        # object, as in eager.
        def f(x):
            return x + 1, f"{from_dlpack(x)}", str(cutlass_torch.current_stream())

        x = _rand(8)
        for g in (
            lambda x: str(from_dlpack(x)),
            lambda x: repr(cutlass_torch.current_stream()),
        ):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "cannot be read"
            ):
                torch.compile(g, backend="eager", fullgraph=True)(x)
        _, tensor_text, stream_text = torch.compile(f, backend="eager")(x)
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
        with self.assertRaises(BufferError):
            f(x, y)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "requires grad"):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)
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
            torch._dynamo.exc.Unsupported, "'mX' and 'mOut' are the same tensor"
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
            (in_place, "'mX' and 'mOut' are the same tensor"),
            (same_outputs, "'mSum' and 'mDiff' are the same tensor"),
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
        # Without fullgraph, cute tensors and a stream made outside the region
        # graph-break with the capture hint, and the call then runs eagerly.
        def f(x, tx, ty, tout, n, stream):
            y = x + 1
            add_kernel(tx, ty, tout, n, stream)
            return y * 2

        x, a, b = _rand(1000), _rand(1000), _rand(1000)
        out = torch.zeros_like(a)
        tx, ty, tout = (from_dlpack(t, assumed_align=16) for t in (a, b, out))
        torch._dynamo.utils.counters.clear()
        res = torch.compile(f, backend="eager")(x, tx, ty, tout, 1000, _stream())
        self.assertEqual(res, (x + 1) * 2)
        self.assertEqual(out, a + b)
        reasons = list(torch._dynamo.utils.counters["graph_break"])
        self.assertTrue(any("current_stream()" in r for r in reasons), reasons)

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


# Inductor generates Triton for the code around the user kernel.
requires_triton = unittest.skipUnless(
    inductor_utils.HAS_CUDA_AND_TRITON, "requires Triton"
)


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


# A module of its own with a host function, out = x * cfg.alpha, and the class of its
# constant cfg.
_SCALE_MODULE_SOURCE = """\
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class Cfg:
    def __init__(self, alpha):
        self.alpha = alpha


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
    _scale_k(mX, mOut, n, cfg.alpha).launch(
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


def _run_as_script(path, code):
    """Write ``code`` to ``path`` and run it as `python <path>` does, as __main__
    with no arguments. Returns its globals."""
    with open(path, "w") as fh:
        fh.write(code)
    with mock.patch.object(sys, "argv", [path]):
        with contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(path, run_name="__main__")


@requires_triton
@requires_cutedsl
class CuteDSLInductorTests(TestCase):
    """Inductor: the HOP lowers to ir.UserDefinedCuteDSLKernel, launched from the
    Python wrapper as `<launcher>.run(<args>, stream=raw_streamN)`."""

    def test_cutedsl_kernel_symbolic_bool_scalar(self):
        # A symbolic bool for an unannotated param is a runtime scalar, evaluated at
        # each launch like a symbolic int.
        w = wrap_cutedsl(select_kernel, mutates_args=["mOut"])

        def f(x):
            out = torch.empty_like(x)
            w(x, out, x.numel(), x.shape[0] > 1500)
            return out

        cf = torch.compile(f, dynamic=True, fullgraph=True)
        for n in (1000, 2000):
            x = _rand(n)
            self.assertEqual(cf(x), (3 if n > 1500 else 5) * x)

    def test_cutedsl_kernel_numeric_scalar_values(self):
        # A cutlass Numeric default, and a Numeric value passed for a Numeric param,
        # reach the generated code as Python floats.
        w = wrap_cutedsl(add_c_kernel, mutates_args=["mOut"])
        c = cutlass.Float32(0.75)

        def f(x):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel())
            w(x, b, x.numel(), c=c)
            return a, b

        x = _rand(1000)
        a, b = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(a, x + 0.25)
        self.assertEqual(b, x + 0.75)

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
            r"\(cutedsl_add_kernel_\d+\) through objects bound in this process or "
            r"object files it links by path, so it cannot be serialized "
            r"\(aot_compile, precompile\)\. Compile it in the process that runs it\.",
        ):
            torch.compile(f, fullgraph=True).aot_compile(((x, y), {}))
        mode = compile_fx.FxCompileMode.SUBPROCESS
        sent = AssertionError("sent to a compile subprocess")
        with (
            mock.patch.object(compile_fx, "fx_compile_mode", mode),
            mock.patch.object(_SubprocessFxCompile, "process_pool", side_effect=sent),
        ):
            self.assertEqual(torch.compile(f, fullgraph=True)(x, y), (x + y) * 2)

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

    def test_cutedsl_kernel_wrapper_codegen_signatures(self):
        # Every wrapper implements the two codegen hooks with the same parameters,
        # each of which some wrapper uses (see the Python wrapper's docstrings).
        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen
        from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen

        def params(fn):
            return list(inspect.signature(fn).parameters)

        for cls in PythonWrapperCodegen, CppWrapperCpu, CppWrapperGpu, WrapperFxCodegen:
            self.assertEqual(
                params(cls.define_user_defined_cutedsl_kernel),
                ["self", "kernel", "constant_args", "runtime_args"],
            )
            self.assertEqual(
                params(cls.generate_user_defined_cutedsl_kernel_call),
                ["self", "kernel_name", "kernel", "runtime_args", "device"],
            )

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
        # reuses the kernel compiled for the cuda:0 call: the DSL loads a compiled
        # kernel on every device, so compiles are cached per capability.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        cf = torch.compile(f, fullgraph=True)
        x, y = _rand(1000), _rand(1000)
        self.assertEqual(cf(x, y), (x + y).relu())
        x, y = x.to("cuda:1"), y.to("cuda:1")
        with torch.cuda.device(0):
            with mock.patch.object(cute, "compile", wraps=cute.compile) as spy:
                res, (code,) = run_and_get_code(cf, x, y)
        self.assertEqual(res, (x + y).relu())
        if torch.cuda.get_device_capability(0) == torch.cuda.get_device_capability(1):
            spy.assert_not_called()
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

    def test_cutedsl_kernel_stream_picked_as_for_inductor_kernels(self):
        # The launch and Inductor's own kernel calls get their stream from one helper,
        # so a change to user-stream or compile-on-one-rank handling reaches both.
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen

        name = "_get_kernel_call_stream_name"
        pick = getattr(PythonWrapperCodegen, name)
        picked = []

        def spy(self, *args):
            picked.append(pick(self, *args))
            return picked[-1]

        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out.relu()

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(PythonWrapperCodegen, name, spy):
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x, y)
        self.assertEqual(res, (x + y).relu())
        ((_, args),) = _cutedsl_launches(code, "add_kernel")
        self.assertEqual(len(picked), 2, code)  # the launch and the relu kernel
        self.assertIn(args.rsplit("stream=", 1)[1], picked)

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

    def test_cutedsl_kernel_dependencies(self):
        # Mirrors test_triton_kernel_dependancies: kernel -> kernel -> aten op.
        def f(x, y):
            out = torch.zeros_like(x)
            add_w(x, y, out, x.numel())
            out2 = torch.zeros_like(out)
            add_w(out, y, out2, out.numel())
            return torch.add(out2, 1)

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), f(x, y))

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
        from torch._inductor.runtime.cutedsl_user_kernel import (
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

    def test_cutedsl_kernel_eager_launches_reuse_a_launcher(self):
        # Eager launches of one call reuse a launcher, which has their compile. The
        # constants are keyed on every launch: one changed in place gets a launcher
        # of its own, and changing it back finds the first launcher again.
        from torch._inductor.runtime.cutedsl_user_kernel import (
            CuteDSLKernelLauncher,
            launch_cutedsl_kernel,
        )

        compile_ = CuteDSLKernelLauncher._compile
        cfg = types.SimpleNamespace(alpha=3.0)
        x = _rand(1000)
        out = torch.empty_like(x)
        args = {"mX": x, "mOut": out, "n": x.numel(), "cfg": cfg}
        with (
            _counting_compiles() as compile_spy,
            mock.patch.object(
                CuteDSLKernelLauncher, "_compile", autospec=True, side_effect=compile_
            ) as miss_spy,
        ):
            for alpha in (3.0, 3.0, 0.5, 3.0):
                cfg.alpha = alpha
                launch_cutedsl_kernel(cfg_scale_kernel, args)
                self.assertEqual(out, alpha * x)
        self.assertEqual(miss_spy.call_count, 2)
        self.assertEqual(compile_spy.call_count, 2)

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
        # Constexpr and unannotated-param values are baked in per launch site.
        def f(x):
            o1, o2, o3 = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
            mul_const_w(x, o1, x.numel(), 3)
            mul_const_w(x, o2, x.numel(), 2.5)
            add_scalar_w(x, o3, x.numel(), 5)
            return o1, o2, o3

        x = _rand(1000)
        o1, o2, o3 = torch.compile(f, fullgraph=True)(x)
        self.assertEqual((o1, o2, o3), (3 * x, 2.5 * x, x + 5))

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

    def test_cutedsl_kernel_output_code_as_script(self):
        # Run as a script (`python <output_code>.py`, e.g. a TORCH_COMPILE_DEBUG or
        # tlparse artifact), the generated module is not loaded with its launchers:
        # get_args() rebuilds each one, importing the host function by module and
        # qualname and unpickling its constants.
        with tempfile.TemporaryDirectory() as d:
            module = _scale_module(d, "cutedsl_test_scale_module")
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__)
            w = wrap_cutedsl(module.scale_kernel, mutates_args=["mOut"])
            cfg = module.Cfg(3.0)

            def f(x):
                out = torch.empty_like(x)
                w(x, out, x.numel(), cfg)
                return out

            x = _rand(1000)
            res, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
            self.assertEqual(res, 3 * x)
            script = _run_as_script(os.path.join(d, "output_code.py"), code)
            (x,) = script["args"]
            (out,) = script["call"]([x])
        self.assertEqual(out, 3 * x)

    def test_cutedsl_kernel_output_code_as_script_host_fn_in_main(self):
        # A host function of the compiling script cannot be imported by a run of the
        # generated module as a script, whose __main__ is that module: the run fails,
        # naming the launcher. The compiling process, which calls get_args() itself
        # under profile_bandwidth_output, keeps the launcher it loaded the module with.
        with tempfile.TemporaryDirectory() as d:
            main = _scale_module(d, "__main__")
            w = wrap_cutedsl(main.scale_kernel, mutates_args=["mOut"])
            cfg = main.Cfg(3.0)

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
            with self.assertRaisesRegex(
                RuntimeError,
                r"cutedsl_scale_kernel_\d+: a run of this module as a script cannot "
                r"rebuild the launcher of this CuTeDSL kernel \(PicklingError: "
                r"scale_kernel is defined in __main__\)",
            ):
                _run_as_script(os.path.join(d, "output_code.py"), code)

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
        self.assertEqual(torch.compile(f, fullgraph=True)(x, y), f(x, y))

    def test_cutedsl_kernel_reduce_overhead(self):
        # CUDA graphs (cudagraph trees) capture the launch on the capture stream.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
        for _ in range(3):
            x, y = _rand(1000), _rand(1000)
            self.assertEqual(compiled(x, y), (x + y) * 2)

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
        # it can share a launcher with a launch passing an equal object, even if the
        # default object changes later (defaults are not guarded: no recompile).
        w = wrap_cutedsl(mul_default_cfg_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            a, b = torch.empty_like(x), torch.empty_like(x)
            w(x, a, x.numel())
            w(x, b, x.numel(), cfg=cfg)
            return a, b

        params = inspect.signature(inspect.unwrap(mul_default_cfg_kernel)).parameters
        default = params["cfg"].default
        cf = torch.compile(f, fullgraph=True)
        x, cfg = _rand(1000), MulCfg(2.0)
        (a, b), (code,) = run_and_get_code(cf, x, cfg)
        self.assertEqual((a, b), (2 * x, 2 * x))
        launches = _cutedsl_launches(code, "mul_default_cfg_kernel")
        self.assertEqual(len(launches), 2, code)
        self.assertEqual(len({name for name, _ in launches}), 1, code)
        try:
            default.alpha = 4.0
            a, b = cf(x, cfg)
        finally:
            default.alpha = 2.0
        self.assertEqual((a, b), (2 * x, 2 * x))

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
        # Constexpr / unannotated values computed from a dynamic size are
        # specialized (guarded); the generated code must bake each value in.
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
    def test_cutedsl_kernel_bypasses_fx_graph_cache(self):
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


# A 1-D clone whose real output starts 4 bytes into its allocation while its fake
# output is a fresh tensor: Inductor assumes an alignment the output lacks.
@torch.library.custom_op("cutedsl_test::misaligned_clone", mutates_args=())
def misaligned_clone(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.new_empty(1), x])[1:]


@misaligned_clone.register_fake
def _misaligned_clone_fake(x):
    return torch.empty_like(x)


class Toy(torch.nn.Module):
    def forward(self, x, y):
        return toy(x, y)


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


@requires_triton
@requires_cutedsl
class CuteDSLAOTITests(TestCase):
    """The C++ wrapper (AOTInductor, and torch.compile with cpp_wrapper): the kernel
    is compiled ahead of time (export_to_c) and launched from the generated C++
    through its cute_dsl_<prefix>_wrapper."""

    def _package(self, model, args, dynamic_shapes=None, strict=False):
        ep = torch.export.export(
            model, args, dynamic_shapes=dynamic_shapes, strict=strict
        )
        return torch._inductor.aoti_compile_and_package(ep)

    def _package_and_code(self, model, args, **kwargs):
        # run_and_get_cpp_code starts with torch._dynamo.reset(), so the export has
        # to happen inside it (see test_aoti_program_exported_before_reset).
        return run_and_get_cpp_code(self._package, model, args, **kwargs)

    def test_aoti_kernels_differing_in_closure_values(self):
        # Two host functions from one factory get the same prefix (it cannot see a
        # captured list); each still gets its own export.
        ka = wrap_cutedsl(make_affine_kernel([2.0, 0.0]), mutates_args=["mOut"])
        kb = wrap_cutedsl(make_affine_kernel([5.0, 1.0]), mutates_args=["mOut"])

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

    def test_aoti_exports_stored_by_content(self):
        # With the in-process registry empty (as in a new process) the kernel is
        # exported again, into the directory named by the bytes of its files: the
        # same paths and no new files. The version script is stored there as well.
        # An entry that lost its files (a stray one left) is replaced.
        from torch._inductor.codegen.cutedsl import user_kernel_aot
        from torch._inductor.utils import fresh_cache

        real_export_for_link = user_kernel_aot.export_for_link
        exports, listings = [], []

        def export_for_link(*args, **kwargs):
            exports.append(real_export_for_link(*args, **kwargs))
            return exports[-1]

        x, y = _rand(1000), _rand(1000)
        with (
            fresh_cache(),
            mock.patch.object(user_kernel_aot, "export_for_link", export_for_link),
            mock.patch.object(
                user_kernel_aot,
                "export_cutedsl_kernel",
                wraps=user_kernel_aot.export_cutedsl_kernel,
            ) as export,
        ):
            root = user_kernel_aot.cutedsl_cache_dir()
            for _ in range(2):
                user_kernel_aot._exports.clear()
                self._package(_AddModule(), (x, y))
                listings.append(sorted(os.listdir(root)))
            entry = os.path.dirname(exports[0].object_path)
            files = sorted(os.listdir(entry))
            for name in files:
                os.remove(os.path.join(entry, name))
            open(os.path.join(entry, "stray"), "w").close()
            user_kernel_aot._exports.clear()
            pkg = self._package(_AddModule(), (x, y))
            self.assertEqual(sorted(os.listdir(entry)), files)
            self.assertEqual(torch._inductor.aoti_load_package(pkg)(x, y), x + y)
        self.assertEqual(export.call_count, 3)
        obj = exports[0].object_path
        self.assertEqual([e.object_path for e in exports], [obj] * 3)
        self.assertEqual(os.path.dirname(entry), root)
        self.assertEqual(files, [f"{exports[0].prefix}.{ext}" for ext in "ho"])
        self.assertEqual(listings[0], listings[1])
        self.assertEqual(len([f for f in listings[0] if f.endswith(".ver")]), 1)

    def test_aoti_call_sites_share_one_export(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                a, b = torch.empty_like(x), torch.empty_like(x)
                add_w(x, y, a, x.numel())
                add_w(a, y, b, x.numel())
                return b

        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(M(), (x, y))
        self.assertEqual(code.count("#define CUTEDSL_DECLARED_"), 1)
        self.assertEqual(torch._inductor.aoti_load_package(pkg)(x, y), x + 2 * y)

    def test_aoti_module_held_per_model_instance(self):
        # Each model instance holds a reference to the kernel's CUDA library (a
        # member of AOTInductorModelKernels): the first loads it and the last
        # unloads it, even when dlclose leaves the .so mapped. Two runners of one
        # .so share the load, so dropping the later one keeps it, and a runner
        # created after the last one was dropped loads it again.
        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(_AddModule(), (x, y))
        p = re.search(r"\bcall_(cutedsl_add_kernel\w*)\(", code).group(1)
        FileCheck().check("class AOTInductorModelKernels").check(
            f"{p}_module_ref {p}_ref;"
        ).check_next("};").run(code)
        FileCheck().check(f"if ({p}_module_refs == 0)").check(
            f"{p}_Kernel_Module_Load(&{p}_loaded_module);"
        ).check(f"if (--{p}_module_refs == 0)").check(
            f"{p}_Kernel_Module_Unload(&{p}_loaded_module);"
        ).run(code)
        self.assertNotIn(f"static {p}_module_ref", code)
        with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(pkg) as z:
            so = z.extract(next(n for n in z.namelist() if n.endswith(".so")), d)
            first, second = (
                torch._C._aoti.AOTIModelContainerRunnerCuda(so, 1, DEVICE)
                for _ in range(2)
            )
            del second
            self.assertEqual(first.run([x, y]), [x + y])
            del first
            third = torch._C._aoti.AOTIModelContainerRunnerCuda(so, 1, DEVICE)
            self.assertEqual(third.run([x, y]), [x + y])

    def test_aoti_two_packages_same_prefix(self):
        # Two packages of kernels that differ only in a captured list: both link it
        # under one prefix. Its symbols and the DSL runtime's stay out of each .so's
        # dynamic symbol table, so neither can bind to the other's, and both run in
        # one process.
        def module(c):
            k = wrap_cutedsl(make_affine_kernel(c), mutates_args=["mOut"])

            class M(torch.nn.Module):
                def forward(self, x):
                    out = torch.empty_like(x)
                    k(x, out, x.numel())
                    return out

            return M()

        x = torch.arange(1000, device=DEVICE, dtype=torch.float32)
        packages, prefixes = [], set()
        for c in ([2.0, 0.0], [5.0, 1.0]):
            pkg, code = self._package_and_code(module(c), (x,))
            packages.append(pkg)
            prefixes.update(re.findall(r"\bcall_(cutedsl_affine_kernel\w*)\(", code))
        (prefix,) = prefixes
        # The object's entry points, and runtime archive shims it calls.
        private = (
            f"_mlir_{prefix}_cuda_init",
            "_cudaLibraryLoadData",
            "_cudaLibraryGetKernel",
            "_cudaLaunchKernelEx",
        )
        for pkg in packages:
            with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(pkg) as z:
                so = z.extract(next(n for n in z.namelist() if n.endswith(".so")), d)
                lib = ctypes.CDLL(so)
            self.assertTrue(hasattr(lib, "AOTInductorModelContainerCreateWithDevice"))
            for sym in private:
                self.assertFalse(hasattr(lib, sym), sym)
        run_a, run_b = (torch._inductor.aoti_load_package(p) for p in packages)
        self.assertEqual(run_a(x), 2 * x)
        self.assertEqual(run_b(x), 5 * x + 1)

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

        x, w = _rand(2048, 2048), _rand(2048, 2048)
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
        calls = re.findall(r"\bcall_cutedsl_add_one_2d_kernel\w*\((.*)\);", code)
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

    def test_aoti_boolean_and_half_scalars(self):
        # A Boolean takes the truthiness of its value (x.shape[0] % 3 can be 2), as
        # in eager; a Float16 scalar compiles and is converted from double.
        w = wrap_cutedsl(select_bool_kernel, mutates_args=["mOut"])
        h = wrap_cutedsl(add_half_kernel, mutates_args=["mOut"])

        class M(torch.nn.Module):
            def forward(self, x):
                a, b = torch.empty_like(x), torch.empty_like(x)
                w(x, a, x.numel(), x.shape[0] % 3)
                h(x, b, x.numel(), 0.5)
                return a, b

        n = torch.export.Dim("n", min=2, max=4096)
        pkg = self._package(M(), (_rand(1000),), dynamic_shapes={"x": {0: n}})
        run = torch._inductor.aoti_load_package(pkg)
        for size in (1000, 1001, 1002):  # size % 3 = 1, 2, 0
            x = _rand(size)
            a, b = run(x)
            self.assertEqual(a, (3 if size % 3 else 5) * x)
            self.assertEqual(b, x + 0.5)

    def test_aoti_integer_and_float64_scalars(self):
        # Every integer type export_to_c takes, at the ends of its range (Uint64
        # past 2**63), and a Float64 that a float would round. The launch refuses a
        # symbolic Int8 outside int8_t instead of truncating it.
        w = wrap_cutedsl(store_scalars_kernel, mutates_args=["mOut", "mOutF"])
        h = 1 + 2**-30
        static = [-(2**15), 2**8 - 1, 2**16 - 1, 2**32 - 1, -(2**63), 2**63 + 5]

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty(7, dtype=torch.int64, device=x.device)
                outf = torch.empty(1, dtype=torch.float64, device=x.device)
                w(out, outf, x.shape[0] - 1129, *static, h)
                return out, outf

        dims = {"x": {0: torch.export.Dim("n", min=2, max=4096)}}
        pkg, code = self._package_and_code(M(), (_rand(1001),), dynamic_shapes=dims)
        c_types = ("int8_t", "int16_t", "uint8_t", "uint16_t", "uint32_t")
        for name, ctype in zip("abcde", c_types):
            FileCheck().check(f"scalar {name} does not fit in {ctype}").run(code)
        self.assertNotIn("scalar f does not fit", code)
        self.assertNotIn("scalar g does not fit", code)
        run = torch._inductor.aoti_load_package(pkg)
        for size in (1001, 1256):  # a = -128, 127
            out, outf = run(_rand(size))
            self.assertEqual(out.tolist(), [size - 1129, *static[:5], -(2**63) + 5])
            self.assertEqual(outf.item(), h, atol=0, rtol=0)
        for size in (1000, 1257):  # a = -129, 128
            with self.assertRaisesRegex(RuntimeError, "a does not fit in int8_t"):
                run(_rand(size))

    def test_aoti_float_for_integer_and_boolean_scalars(self):
        # A float for an integer or Boolean parameter is converted as the DSL
        # converts it in eager: truncated toward zero, or its truthiness (0.5 is
        # true). A static one when compiling; a symbolic one at launch, by its C++
        # type, which passes SymInt bitwise ops and a symbolic bool as they are. A
        # symbolic float that int64_t (uint64_t for a Uint64) cannot hold is refused
        # at launch. AOTInductor and the JIT cpp_wrapper alike.
        sel = wrap_cutedsl(select_bool_kernel, mutates_args=["mOut"])
        store = wrap_cutedsl(store_scalars_kernel, mutates_args=["mOut", "mOutF"])

        def fn(x):
            n = x.shape[0]
            flags = [torch.empty_like(x) for _ in range(3)]
            for out, flag in zip(flags, (0.5, n > 1000, (n - 1000) / 2000)):
                sel(x, out, x.numel(), flag)
            ints = [x.new_empty(7, dtype=torch.int64) for _ in range(2)]
            floats = [x.new_empty(1, dtype=torch.float64) for _ in range(2)]
            store(ints[0], floats[0], -2.5, 2.9, 3.5, 4.5, 5.5, -6.5, 7.5, 0.25)
            # Parameter f is beyond int64 from n = 1100, g is negative below n = 500.
            symbolic = (-n / 8, n & 7, n / 400, n | 1, n * 1.5, n // 1100 * 1e19)
            store(ints[1], floats[1], *symbolic, (n - 500) / 3, n / 3)
            return *flags, *ints, *floats

        class M(torch.nn.Module):
            def forward(self, x):
                return fn(x)

        dims = {"x": {0: torch.export.Dim("n", min=2, max=4096)}}
        aoti = torch._inductor.aoti_load_package(
            self._package(M(), (_rand(1000),), dynamic_shapes=dims)
        )
        torch._dynamo.reset()
        jit = torch.compile(fn, dynamic=True, fullgraph=True)
        with inductor_config.patch(cpp_wrapper=True):
            for run in (aoti, jit):
                for size, ints in (
                    (1000, [-125, 0, 2, 1001, 1500, 0, 166]),
                    (1001, [-125, 1, 2, 1001, 1501, 0, 167]),
                ):
                    x = _rand(size)
                    result = tuple(run(x))
                    self.assertEqual(result, fn(x))
                    k = 3 if size > 1000 else 5
                    self.assertEqual(result[:3], (3 * x, k * x, k * x))
                    self.assertEqual(result[3].tolist(), [-2, 2, 3, 4, 5, -6, 7])
                    self.assertEqual(result[4].tolist(), ints)
                for size, msg in (
                    (400, "scalar g does not fit in uint64_t"),
                    (1100, "scalar f does not fit in int64_t"),
                ):
                    with self.assertRaisesRegex(RuntimeError, msg):
                        run(_rand(size))

    def test_aoti_static_scalar_out_of_range(self):
        # A static value its integer parameter cannot hold is refused when
        # compiling, naming the kernel and the parameter, where eager wraps it (or
        # raises, for NaN and infinity).
        store = wrap_cutedsl(store_scalars_kernel, mutates_args=["mOut", "mOutF"])

        class M(torch.nn.Module):
            def __init__(self, scalars):
                super().__init__()
                self.scalars = scalars

            def forward(self, x):
                out = torch.empty(7, dtype=torch.int64, device=x.device)
                outf = torch.empty(1, dtype=torch.float64, device=x.device)
                store(out, outf, *self.scalars)
                return out

        for name, value, ctype in (
            ("a", float("inf"), "int8_t"),
            ("b", float("nan"), "int16_t"),
            ("c", 256.5, "uint8_t"),
            ("f", 1e20, "int64_t"),
            ("g", -1.5, "uint64_t"),
            ("g", 2**64, "uint64_t"),
        ):
            scalars = [1, 2, 3, 4, 5, 6, 7, 0.5]
            scalars["abcdefg".index(name)] = value
            msg = f"store_scalars_kernel: {value} for '{name}' does not fit in {ctype}"
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(Exception, re.escape(msg)):
                    self._package(M(scalars), (_rand(8),))

    def test_aoti_dynamic_size_beyond_int32(self):
        # A dynamic size fills an int32 slot of the exported tensor struct: a larger
        # one is refused at launch instead of being truncated.
        w = wrap_cutedsl(copy_first_kernel, mutates_args=["mOut"])

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty(1, device=x.device)
                w(x, out)
                return out

        # Expanded inputs: n elements, one of storage.
        def expanded(n):
            return torch.full((1,), 3.0, device=DEVICE).expand(n)

        dims = {"x": {0: torch.export.Dim("n", min=2)}}
        pkg, code = self._package_and_code(M(), (expanded(1000),), dynamic_shapes=dims)
        msg = "dynamic size 0 of argument mX does not fit in int32"
        FileCheck().check(msg).run(code)
        run = torch._inductor.aoti_load_package(pkg)
        self.assertEqual(run(expanded(5000)), expanded(1))
        with self.assertRaisesRegex(RuntimeError, msg):
            run(expanded(2**31))

    def test_aoti_toy_raw_call(self):
        # The toy through strict export, where Dynamo captures the raw call, and
        # AOTInductor.
        x, y = _rand(2048), _rand(2048)
        pkg, code = self._package_and_code(Toy(), (x, y), strict=True)
        self.assertRegex(code, r"cute_dsl_\w+_wrapper\(")
        runner = torch._inductor.aoti_load_package(pkg)
        self.assertEqual(runner(x, y), (x + y).relu())
        x2, y2 = _rand(2048), _rand(2048)
        self.assertEqual(runner(x2, y2), (x2 + y2).relu())

    def test_aoti_add(self):
        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(_AddModule(), (x, y))
        self.assertRegex(code, r"cute_dsl_\w+_wrapper\(")
        runner = torch._inductor.aoti_load_package(pkg)
        self.assertEqual(runner(x, y), x + y)
        x2, y2 = _rand(1000), _rand(1000)
        self.assertEqual(runner(x2, y2), x2 + y2)

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
        pkg = self._package(M(), (_rand(8, 4),), dynamic_shapes={"x": {0: a, 1: b}})
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
        runner = torch._inductor.aoti_load_package(self._package(M(), (x,)))
        o1, o2 = runner(x)
        self.assertEqual((o1, o2), (3 * x, 2.5 * x))

    def test_aoti_unannotated_scalar(self):
        # export_to_c takes a scalar's C type from its annotation ("Unsupported
        # argument ... with type None" in cutlass 4.4.2), so the C++ wrapper asks for
        # one, for AOTInductor and the JIT cpp_wrapper alike.
        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                add_scalar_w(x, out, x.numel(), 5)
                return out

        x = _rand(1000)
        # The Python wrapper launches it fine ...
        self.assertEqual(torch.compile(M(), fullgraph=True)(x), x + 5)
        # ... the C++ wrapper cannot export it.
        msg = r"C\+\+ wrapper \(AOTInductor or cpp_wrapper\) needs scalar parameter 'k'"
        with self.assertRaisesRegex(Exception, msg):
            self._package(M(), (x,))
        torch._dynamo.reset()
        with inductor_config.patch(cpp_wrapper=True):
            with self.assertRaisesRegex(Exception, msg):
                torch.compile(M(), fullgraph=True)(x)

    @parametrize(
        "option,value,message",
        [
            ("triton.autotune_at_compile_time", False, "dual wrapper mode"),
            ("aot_inductor.package_cpp_only", True, "package_cpp_only"),
            ("aot_inductor.emit_multi_arch_kernel", True, "emit_multi_arch_kernel"),
            ("aot_inductor.cross_target_platform", "windows", "for Windows"),
        ],
        name_fn=lambda option, *_: option.split(".")[-1],
    )
    def test_aoti_refused_config(self, option, value, message):
        # Configurations the exported objects cannot be linked in are refused
        # before anything is exported.
        from torch._inductor.codegen.cutedsl import user_kernel_aot

        # Cross-compiling for Windows needs the constants out of the .so to get here.
        patches = {option: value, "aot_inductor.package_constants_in_so": False}
        x, y = _rand(1000), _rand(1000)
        with (
            inductor_config.patch(patches),
            mock.patch.object(user_kernel_aot, "export_for_link") as export,
            self.assertRaisesRegex(Exception, f"not supported with .*{message}"),
        ):
            self._package(_AddModule(), (x, y))
        export.assert_not_called()

    def test_aoti_misaligned_input_guard(self):
        # Exported with aligned inputs, the kernel assumes 16-byte alignment and
        # the launch checks it for every tensor.
        base, base2 = _rand(1001), _rand(1001)
        pkg, code = self._package_and_code(_AddModule(), (base[:1000], base2[:1000]))
        for name in ("mX", "mY", "mOut"):
            FileCheck().check(f"argument {name} is not 16-byte aligned").run(code)
        runner = torch._inductor.aoti_load_package(pkg)
        x, y = base[1:], base2[1:]
        if inductor_config.is_fbcode():
            # fbcode AOTI does not copy misaligned inputs, so the guard fires.
            with self.assertRaisesRegex(RuntimeError, "mX is not 16-byte aligned"):
                runner(x, y)
        else:
            self.assertEqual(runner(x, y), x + y)

    def test_aoti_misaligned_fallback_output_guard(self):
        # The real output of misaligned_clone is 4 bytes off while its fake output
        # is aligned, so the kernel is compiled to assume 16-byte alignment and the
        # guard fires, also outside fbcode (where misaligned inputs are copied).
        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(misaligned_clone(x), y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(misaligned_clone(x).data_ptr() % 16, 4)
        pkg, code = self._package_and_code(M(), (x, y))
        FileCheck().check("argument mX is not 16-byte aligned").run(code)
        runner = torch._inductor.aoti_load_package(pkg)
        with self.assertRaisesRegex(RuntimeError, "mX is not 16-byte aligned"):
            runner(x, y)

    def test_aoti_two_outputs_and_default_scale(self):
        # The C wrapper takes `float scale` after the stream: a defaulted runtime
        # scalar must still be passed.
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
        # collide in the package. The C++ the two exports share (standard headers,
        # and the CUTE_DSL_CUDA_ERROR_CHECK definition, which must precede the
        # export_to_c headers) is emitted once; AOTI_RUNTIME_CHECK, which it uses,
        # comes from the AOTI headers.
        class M(torch.nn.Module):
            def forward(self, x, y):
                o1, o2 = torch.empty_like(x), torch.empty_like(x)
                same_name_add_w(x, y, o1, x.numel())
                same_name_sub_w(x, y, o2, x.numel())
                return o1, o2

        x, y = _rand(1000), _rand(1000)
        pkg, code = self._package_and_code(M(), (x, y))
        self.assertEqual(code.count("#define CUTEDSL_DECLARED_"), 2)
        # Found by its body: each export_to_c header has an #ifndef-guarded default.
        error_check = 'std::string("CuTeDSL kernel module: ")'
        for text in (error_check, "#include <mutex>"):
            self.assertEqual(code.count(text), 1, text)
        FileCheck().check(error_check).check("#ifndef CUTE_DSL_CUDA_ERROR_CHECK").run(
            code
        )
        self.assertNotIn("#ifndef AOTI_RUNTIME_CHECK", code)
        runner = torch._inductor.aoti_load_package(pkg)
        o1, o2 = runner(x, y)
        self.assertEqual(o1, x + y)
        self.assertEqual(o2, x - y)

    def test_aoti_debug_printer_and_kernel_profile(self):
        # The launch is wrapped like a Triton launch for the intermediate value
        # printer (its tensors before and after) and like a CUDA C++ kernel's for
        # kernel profiling (a record in a block of its own).
        x, y = _rand(1000), _rand(1000)
        patches = {
            "aot_inductor.debug_intermediate_value_printer": "2",
            "cpp.enable_kernel_profile": True,
        }
        with inductor_config.patch(patches):
            pkg, code = self._package_and_code(_AddModule(), (x, y))
        p = re.search(r"\bcall_(cutedsl_add_kernel\w*)\(", code).group(1)
        record = f'RAIIAtenRecordFunctionHandle record_{p}_("{p}", nullptr);'
        (
            FileCheck()
            .check_count(f'"before_launch - {p} - ', 3)
            .check("{")
            .check_next(record)
            .check(f"call_{p}(")
            .check_next("}")
            .check_count(f'"after_launch - {p} - ', 3)
            .run(code)
        )
        self.assertEqual(torch._inductor.aoti_load_package(pkg)(x, y), x + y)

    def test_aoti_launcher_parameter_names(self):
        # Parameters named like the launcher's own identifiers would be if it named
        # its parameters after the host function's (a tensor "stream", a scalar
        # "stream_size0" next to that tensor's dynamic size, "cutedsl_arg0" and
        # "cutedsl_status"): the launcher names its parameters by position.
        w = wrap_cutedsl(colliding_names_kernel, mutates_args=["cutedsl_arg0"])

        class M(torch.nn.Module):
            def forward(self, x):
                out = torch.empty_like(x)
                w(x, out, x.numel(), 5)
                return out

        n = torch.export.Dim("n", min=2, max=4096)
        pkg = self._package(M(), (_rand(1000),), dynamic_shapes={"x": {0: n}})
        run = torch._inductor.aoti_load_package(pkg)
        for size in (1000, 77):
            x = _rand(size)
            self.assertEqual(run(x), x + 5)

    def test_cutedsl_kernel_link_args_from_wrapper(self):
        # The model .so and the JIT module link what the wrapper's cutedsl_link_args
        # returns: the export's object, the DSL's runtime archive and the symbol
        # flags, plus cudart for the JIT module.
        from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
        from torch._inductor.codegen.cutedsl import user_kernel_aot

        link_args = []
        real = CppWrapperGpu.cutedsl_link_args

        def cutedsl_link_args(wrapper):
            link_args.append(real(wrapper))
            return link_args[-1]

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(CppWrapperGpu, "cutedsl_link_args", cutedsl_link_args):
            run = torch._inductor.aoti_load_package(self._package(_AddModule(), (x, y)))
            self.assertEqual(run(x, y), x + y)
            torch._dynamo.reset()
            with inductor_config.patch(cpp_wrapper=True):
                self.assertEqual(
                    torch.compile(_AddModule(), fullgraph=True)(x, y), x + y
                )
        aot, jit = link_args
        archive = user_kernel_aot.cutedsl_runtime_archive()
        self.assertEqual(aot[1], archive)
        self.assertTrue(aot[0].endswith(".o"), aot)
        self.assertIn(f"-Wl,--exclude-libs,{os.path.basename(archive)}", aot)
        self.assertEqual(jit[:-1], aot)
        self.assertIn("cudart", jit[-1])

    def test_aoti_unaligned_view(self):
        class M(torch.nn.Module):
            def forward(self, base):
                x = base[:, 1:]  # data_ptr 4 bytes past a 16-byte boundary
                out = torch.empty(x.shape, device=x.device)
                add_one_2d_w(x, out)
                return out

        base = _rand(4, 9)
        pkg, code = self._package_and_code(M(), (base,))
        # Compiled for the view's 4-byte alignment, mX needs no guard; mOut does.
        FileCheck().check_not("argument mX is not").run(code)
        FileCheck().check("argument mOut is not 16-byte aligned").run(code)
        runner = torch._inductor.aoti_load_package(pkg)
        self.assertEqual(runner(base), base[:, 1:] + 1)

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

    def test_cutedsl_kernel_jit_cpp_wrapper(self):
        # torch.compile with the C++ wrapper links the exported objects into the JIT
        # module the same way.
        x, y = _rand(1000), _rand(1000)
        with inductor_config.patch(cpp_wrapper=True):
            res, (code,) = run_and_get_code(
                torch.compile(_AddModule(), fullgraph=True), x, y
            )
        self.assertEqual(res, x + y)
        FileCheck().check("call_cutedsl_add_kernel_").run(code)

    def test_cutedsl_kernel_jit_cpp_wrapper_object_constant_modified(self):
        # The export compiles the object's contents in; Dynamo guards them, so a
        # change made between calls recompiles and exports again, as eager sees it.
        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        cfg = Cfg(3.0)
        w = wrap_cutedsl(cfg_scale_kernel, mutates_args=["mOut"])

        def f(x, cfg):
            out = torch.empty_like(x)
            w(x, out, x.numel(), cfg)
            return out

        x = _rand(1000)
        cnt = CompileCounterWithBackend("inductor")
        cf = torch.compile(f, backend=cnt, fullgraph=True)
        with inductor_config.patch(cpp_wrapper=True):
            res, (code,) = run_and_get_code(cf, x, cfg)
            self.assertEqual(res, 3 * x)
            self.assertEqual(cnt.frame_count, 1)
            cfg.alpha = 5.0
            self.assertEqual(cf(x, cfg), 5 * x)
            self.assertEqual(cnt.frame_count, 2)
            self.assertEqual(cf(x, cfg), 5 * x)
            self.assertEqual(cnt.frame_count, 2)
        FileCheck().check("call_cutedsl_cfg_scale_kernel_").run(code)

    def test_cutedsl_kernel_jit_cpp_wrapper_link_args_stable(self):
        # Exported again (registry cleared, as in a new process), the kernel links
        # from the same paths, so the JIT module's link command, part of its cache
        # key, is unchanged.
        from torch._inductor.codegen.cutedsl import user_kernel_aot

        x, y = _rand(1000), _rand(1000)
        flags = []
        for _ in range(2):
            user_kernel_aot._exports.clear()
            torch._dynamo.reset()
            with inductor_config.patch(cpp_wrapper=True):
                _, (code,) = run_and_get_code(
                    torch.compile(_AddModule(), fullgraph=True), x, y
                )
            flags.append(re.search(r"extra_flags=(.*)", code).group(1))
        self.assertEqual(flags[0], flags[1])
        self.assertIn(user_kernel_aot.cutedsl_cache_dir(), flags[0])
        self.assertIn(".ver", flags[0])

    def test_cutedsl_kernel_jit_cpp_wrapper_precompiled_header_reused(self):
        # The JIT module links the exports through extra_flags, which its precompiled
        # header is built and keyed without, so a second, different set of exports
        # reuses the header instead of precompiling (and storing) a new one.
        from torch._inductor import codecache
        from torch._inductor.codegen.cutedsl import user_kernel_aot

        x, y = _rand(1000), _rand(1000)
        with (
            tempfile.TemporaryDirectory() as headers,
            mock.patch.object(codecache, "_HEADER_DIR", headers),
            mock.patch.object(
                codecache, "_HEADER_LOCK_DIR", os.path.join(headers, "locks")
            ),
            mock.patch.object(
                codecache,
                "_precompile_header",
                wraps=functools.cache(codecache._precompile_header.__wrapped__),
            ) as precompile,
            inductor_config.patch(cpp_wrapper=True, cpp_cache_precompile_headers=True),
        ):
            for module, args, expected in (
                (_AddModule(), (x, y), x + y),
                (_ScaleModule(), (x,), 2 * x),
            ):
                torch._dynamo.reset()
                self.assertEqual(torch.compile(module, fullgraph=True)(*args), expected)
            stored = [f for f in os.listdir(headers) if f.endswith(".h")]
        first, second = precompile.call_args_list
        self.assertEqual(first, second)
        self.assertNotIn(user_kernel_aot.cutedsl_cache_dir(), str(first))
        self.assertEqual(len(stored), 1)

    def test_cutedsl_kernel_jit_cpp_wrapper_graph_not_serializable(self):
        # The JIT module links the exports by their paths in the local cache
        # directory, so serializing the compiled graph (aot_compile) fails loudly, as
        # for the Python wrapper's launchers.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        x, y = _rand(1000), _rand(1000)
        # The export's prefix (cutedsl_<name>_<arch>_<hash>), not a launcher's name.
        msg = r"\(cutedsl_add_kernel_sm\w+\).*cannot be serialized"
        with inductor_config.patch(cpp_wrapper=True):
            with self.assertRaisesRegex(Exception, msg):
                torch.compile(f, fullgraph=True).aot_compile(((x, y), {}))

    def test_cutedsl_kernel_prefix_same_for_equal_constants(self):
        # Equal constants give the same prefix in every process, so a kernel exported
        # again reuses its files: objects are compared by content, sets in any order,
        # and object addresses are ignored.
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            ConstantParamSpec,
            cutedsl_kernel_prefix,
        )

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

        class Slots:
            __slots__ = ()

        def prefix(cfg):
            spec = ConstantParamSpec("cfg", cfg)
            return cutedsl_kernel_prefix(cfg_scale_kernel, [spec], "sm_90a")

        for a, b in [
            (Cfg(3.0), Cfg(3.0)),
            (frozenset([0, 8]), frozenset([8, 0])),
            (Slots(), Slots()),
        ]:
            self.assertEqual(prefix(a), prefix(b))
        self.assertNotEqual(prefix(Cfg(3.0)), prefix(Cfg(5.0)))

    def test_cutedsl_kernel_prefix_refuses_cyclic_constants(self):
        # constant_key keys a frozenset by its hash without walking it, so a cycle
        # through one passes the side table; the prefix's walk refuses it instead of
        # recursing until a RecursionError.
        from torch._higher_order_ops.user_kernel_hop import constant_key
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            ConstantParamSpec,
            cutedsl_kernel_prefix,
        )

        class Cfg:
            pass

        cfg = Cfg()
        cfg.items = frozenset([cfg])
        constant_key(cfg)
        spec = ConstantParamSpec("cfg", cfg)
        with self.assertRaisesRegex(TypeError, "type Cfg refers to itself$"):
            cutedsl_kernel_prefix(cfg_scale_kernel, [spec], "sm_90a")


# The header export_to_c (cutlass 4.4.2) writes for abi_kernel exported with
# _abi_specs under the prefix cutedsl_fixture.
_EXPORTED_HEADER = r"""
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} cutedsl_fixture_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_cutedsl_fixture_cuda_init(void **);
void _mlir_cutedsl_fixture_cuda_load_to_device(void **);
static inline void cutedsl_fixture_Kernel_Module_Load(cutedsl_fixture_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_cutedsl_fixture_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_cutedsl_fixture_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void cutedsl_fixture_Kernel_Module_Unload(cutedsl_fixture_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
} cutedsl_fixture_Tensor_mX_t;


typedef struct {
    void *data;
} cutedsl_fixture_Tensor_mOut_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_cutedsl_fixture__mlir_ciface_cutlass_abi_kernel_FakeTensorFloat32div1div11i64div1_FakeTensorFloat328441______________3_FakeStream(void **args, int32_t num_args);

static inline int32_t cute_dsl_cutedsl_fixture_wrapper(cutedsl_fixture_Kernel_Module_t *module, cutedsl_fixture_Tensor_mX_t *mX, cutedsl_fixture_Tensor_mOut_t *mOut, int8_t i8, int16_t i16, int32_t i32, int64_t i64, uint8_t u8, uint16_t u16, uint32_t u32, uint64_t u64, uint8_t flag, float f32, float tf32, double f64, __half_raw f16, cudaStream_t stream) {
    int32_t ret;
    void *args[17] = {
        mX, mOut, &i8, &i16, &i32, &i64, &u8, &u16, &u32, &u64, &flag, &f32, &tf32, &f64, &f16, &stream,
        &ret
    };
    _mlir_cutedsl_fixture__mlir_ciface_cutlass_abi_kernel_FakeTensorFloat32div1div11i64div1_FakeTensorFloat328441______________3_FakeStream(args, 17);
    return ret;
}
"""  # noqa: B950


def _numeric(name):
    """Stands in for the cutlass Numeric type ``name``: scalar_ctype reads only the
    name and the module."""
    return type(name, (), {"__module__": "cutlass"})


def _abi_specs(numeric=_numeric):
    """The specs abi_kernel is exported with to produce _EXPORTED_HEADER."""
    from torch._inductor.codegen.cutedsl.user_kernel_aot import (
        ConstantParamSpec,
        ScalarParamSpec,
        StreamParamSpec,
        TensorParamSpec,
    )

    scalars = {
        "i8": "Int8",
        "i16": "Int16",
        "i32": "Int32",
        "i64": "Int64",
        "u8": "Uint8",
        "u16": "Uint16",
        "u32": "Uint32",
        "u64": "Uint64",
        "flag": "Boolean",
        "f32": "Float32",
        "tf32": "TFloat32",
        "f64": "Float64",
        "f16": "Float16",
    }
    return [
        TensorParamSpec("mX", torch.float32, (None, None), (1, None), 16),
        TensorParamSpec("mOut", torch.float32, (8, 4), (4, 1), 4),
        *(ScalarParamSpec(name, numeric(t), 1) for name, t in scalars.items()),
        ConstantParamSpec("alpha", 3),
        StreamParamSpec("stream"),
    ]


class CuteDSLExportTests(TestCase):
    """What the C++ wrapper does with an export_to_c header and object (parse and
    validate the header, check the object's symbols, emit the launch, link flags),
    on a header export_to_c wrote and objects the C++ compiler builds: no GPU or
    cutlass needed, but for test_header_fixture_is_current."""

    def _exported(self, header=_EXPORTED_HEADER):
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            CuteDSLExportedKernel,
            parse_exported_header,
        )

        abi = parse_exported_header(header, "cutedsl_fixture")
        return CuteDSLExportedKernel(
            "cutedsl_fixture", "cutedsl_fixture.h", "cutedsl_fixture.o", header, abi
        )

    def _compile_object(self, out_dir, name, source):
        from torch._inductor.cpp_builder import _compiler_command, get_cpp_compiler
        from torch._inductor.exc import InvalidCxxCompiler

        if sys.platform != "linux":
            self.skipTest("ELF objects")
        src, obj = (os.path.join(out_dir, f"{name}.{ext}") for ext in ("cpp", "o"))
        with open(src, "w") as f:
            f.write(source)
        try:
            compiler = get_cpp_compiler()
        except InvalidCxxCompiler as e:
            self.skipTest(str(e))
        subprocess.run(
            _compiler_command(compiler, "-c", "-fPIC", src, "-o", obj),
            check=True,
            capture_output=True,
        )
        return obj

    @requires_cutedsl
    def test_header_fixture_is_current(self):
        # The installed DSL still writes _EXPORTED_HEADER. If a DSL upgrade breaks
        # this, capture the new header here and fix what the other tests catch.
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            export_cutedsl_kernel,
        )

        specs = _abi_specs(lambda name: getattr(cutlass, name))
        with tempfile.TemporaryDirectory() as d:
            exported = export_cutedsl_kernel(
                abi_kernel, specs, out_dir=d, prefix="cutedsl_fixture", arch="sm_90a"
            )
        self.assertEqual(exported.header_text, _EXPORTED_HEADER)

    def test_parse_exported_header(self):
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            ExportedTensorStruct,
            validate_exported_abi,
        )

        abi = self._exported().abi
        p = "cutedsl_fixture"
        self.assertEqual(
            (abi.module_ctype, abi.load_fn, abi.unload_fn, abi.wrapper_fn),
            (
                f"{p}_Kernel_Module_t",
                f"{p}_Kernel_Module_Load",
                f"{p}_Kernel_Module_Unload",
                f"cute_dsl_{p}_wrapper",
            ),
        )
        self.assertEqual(abi.wrapper_return_ctype, "int32_t")
        self.assertEqual(
            abi.tensor_structs,
            (
                ExportedTensorStruct(
                    f"{p}_Tensor_mX_t", "mX", "int32_t", 2, "int64_t", 1
                ),
                ExportedTensorStruct(f"{p}_Tensor_mOut_t", "mOut", None, 0, None, 0),
            ),
        )
        self.assertEqual(
            abi.params,
            (
                (f"{p}_Tensor_mX_t*", "mX"),
                (f"{p}_Tensor_mOut_t*", "mOut"),
                ("int8_t", "i8"),
                ("int16_t", "i16"),
                ("int32_t", "i32"),
                ("int64_t", "i64"),
                ("uint8_t", "u8"),
                ("uint16_t", "u16"),
                ("uint32_t", "u32"),
                ("uint64_t", "u64"),
                ("uint8_t", "flag"),
                ("float", "f32"),
                ("float", "tf32"),
                ("double", "f64"),
                ("__half_raw", "f16"),
                ("cudaStream_t", "stream"),
            ),
        )
        validate_exported_abi(abi, _abi_specs())

    def test_parse_exported_header_errors(self):
        from torch._inductor.codegen.cutedsl.user_kernel_aot import CuteDSLExportError

        p = "cutedsl_fixture"
        module_arg = f"{p}_Kernel_Module_t *module, "
        for old, new, msg in [
            ("cudaLibrary_t module;", "void *module;", "no module struct"),
            (f"{p}_Kernel_Module_Load(", f"{p}_Load(", "no module load function"),
            (f"{p}_Kernel_Module_Unload(", f"{p}_Unload(", "no module unload function"),
            (
                "void *data;\n    int32_t dynamic_shapes",
                "int32_t dynamic_shapes",
                f"{p}_Tensor_mX_t: first field is not 'void \\*data'",
            ),
            (
                "int64_t dynamic_strides[1];",
                "int64_t dynamic_strides[1];\n    float scale;",
                f"{p}_Tensor_mX_t: unexpected field 'float scale'",
            ),
            (
                "int64_t dynamic_strides[1];",
                "int32_t dynamic_shapes[1];",
                r"unexpected field 'int32_t dynamic_shapes\[1\]'",
            ),
            (
                f"cute_dsl_{p}_wrapper(",
                "cute_dsl_other_wrapper(",
                "no wrapper function",
            ),
            ("int8_t i8,", "int8_t i8[2],", r"cannot parse parameter 'int8_t i8\[2\]'"),
            (module_arg, "", f"first parameter is not a {p}_Kernel_Module_t pointer"),
        ]:
            self.assertEqual(_EXPORTED_HEADER.count(old), 1, old)
            header = _EXPORTED_HEADER.replace(old, new)
            with self.subTest(msg), self.assertRaisesRegex(CuteDSLExportError, msg):
                self._exported(header)

    def test_validate_exported_abi_errors(self):
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            CuteDSLExportError,
            ScalarParamSpec,
            StreamParamSpec,
            TensorParamSpec,
            validate_exported_abi,
        )

        abi = self._exported().abi
        specs = _abi_specs()

        def replaced(spec, name=None):
            return [spec if s.name == (name or spec.name) else s for s in specs]

        int32 = _numeric("Int32")
        mx_sizes = dataclasses.replace(specs[0], sizes=(4, None))
        for case_abi, case_specs, msg in [
            (
                dataclasses.replace(abi, wrapper_return_ctype="void"),
                specs,
                "returns void, expected the int32_t launch status",
            ),
            (
                abi,
                replaced(ScalarParamSpec("m", int32, 1), "i32"),
                r"parameters \[.*\] != ",
            ),
            (
                abi,
                replaced(ScalarParamSpec("mOut", int32, 1)),
                r"unexpected tensor structs \['cutedsl_fixture_Tensor_mOut_t'\]",
            ),
            (
                abi,
                replaced(TensorParamSpec("i32", torch.int32, (4,), (1,), 4)),
                "no tensor struct for i32",
            ),
            (
                abi,
                replaced(mx_sizes),
                r"tensor mX has .* = \('cutedsl_fixture_Tensor_mX_t\*', 'int32_t', "
                r"2, 'int64_t', 1\), expected .*'int32_t', 1, 'int64_t', 1\)",
            ),
            (
                abi,
                replaced(ScalarParamSpec("i8", _numeric("Int16"), 1)),
                "scalar i8 is int8_t, expected int16_t",
            ),
            (
                abi,
                replaced(ScalarParamSpec("i8", _numeric("BFloat16"), 1)),
                "BFloat16.* is not supported by CuTeDSL export_to_c",
            ),
            (
                abi,
                replaced(StreamParamSpec("flag")),
                "stream flag is uint8_t, expected cudaStream_t",
            ),
        ]:
            with self.subTest(msg), self.assertRaisesRegex(CuteDSLExportError, msg):
                validate_exported_abi(case_abi, case_specs)

    def test_unexportable_parameter_names(self):
        # The header names the C parameters after the host function's, next to the
        # names it declares itself: those and C++ keywords are refused before
        # compiling. A constant is not a C parameter.
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            ConstantParamSpec,
            export_cutedsl_kernel,
            ScalarParamSpec,
            StreamParamSpec,
            TensorParamSpec,
        )

        def host(args, ret, module, new, x, s):
            pass

        specs = [
            TensorParamSpec("args", torch.float32, (4,), (1,), 4),
            ScalarParamSpec("ret", _numeric("Int32"), 1),
            ConstantParamSpec("module", 1),
            ScalarParamSpec("new", _numeric("Float32"), 1.0),
            TensorParamSpec("x", torch.float32, (4,), (1,), 4),
            StreamParamSpec("s"),
        ]
        msg = r"host: rename parameters \['args', 'ret', 'new'\]: export_to_c"
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, msg):
                export_cutedsl_kernel(host, specs, out_dir=d, prefix="p", arch="sm_90a")
            self.assertEqual(os.listdir(d), [])

    def test_object_symbols(self):
        # Only global and weak symbols the object defines count: an object passes if
        # they all carry the prefix.
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            _check_object_symbols,
            _elf_global_symbols,
            CuteDSLExportError,
        )

        source = """
            extern "C" {
            int _cudaLaunchKernelEx(void);
            static int local_helper(void) { return 1; }
            int cutedsl_fixture_launch(void) {
              return local_helper() + _cudaLaunchKernelEx();
            }
            int _mlir_cutedsl_fixture_cuda_init(void) { return 0; }
            __attribute__((weak)) int cutedsl_fixture_version = 1;
            }
        """
        stray = """
            extern "C" {
            int helper(void) { return 0; }
            __attribute__((weak)) int weak_helper = 1;
            }
        """
        with tempfile.TemporaryDirectory() as d:
            ok = self._compile_object(d, "ok", source)
            bad = self._compile_object(d, "bad", source + stray)
            self.assertEqual(
                sorted(_elf_global_symbols(ok)),
                [
                    "_mlir_cutedsl_fixture_cuda_init",
                    "cutedsl_fixture_launch",
                    "cutedsl_fixture_version",
                ],
            )
            _check_object_symbols(ok, "cutedsl_fixture")
            with self.assertRaisesRegex(
                CuteDSLExportError, "outside cutedsl_fixture_"
            ) as cm:
                _check_object_symbols(bad, "cutedsl_fixture")
            self.assertIn("'helper'", str(cm.exception))
            self.assertIn("'weak_helper'", str(cm.exception))
            self.assertNotIn("cutedsl_fixture_launch", str(cm.exception))
            with self.assertRaisesRegex(CuteDSLExportError, "outside other_"):
                _check_object_symbols(ok, "other")
            archive = os.path.join(d, "lib.a")
            with open(archive, "wb") as f:
                f.write(b"!<arch>\n")
            with self.assertRaisesRegex(CuteDSLExportError, "not an ELF64"):
                _elf_global_symbols(archive)

    def test_launcher(self):
        # The C++ launcher: parameters named by position (never after the host
        # function's, which could collide), alignment and range checks, the struct
        # fills, each scalar type's conversion and the status check.
        from torch._inductor.codegen.cpp_wrapper_gpu import CuteDSLCallWrapper
        from torch._inductor.utils import IndentedBuffer

        exported = dataclasses.replace(self._exported(), specs=tuple(_abi_specs()))
        wrapper = types.SimpleNamespace(prefix=IndentedBuffer())
        CuteDSLCallWrapper("call_k", exported, {"mX": 16}).generate(wrapper)
        self.assertExpectedInline(
            wrapper.prefix.getvalue(),
            """\
static inline void call_k(void* p0, int64_t p0_size0, int64_t p0_size1, int64_t p0_stride0, void* p1, int64_t p2, int64_t p3, int64_t p4, int64_t p5, int64_t p6, int64_t p7, int64_t p8, uint64_t p9, int64_t p10, double p11, double p12, double p13, double p14, cudaStream_t cutedsl_stream)
{
  // CuTeDSL user kernel cutedsl_fixture
  void* cutedsl_arg0_data = p0;  // mX
  AOTI_RUNTIME_CHECK(reinterpret_cast<std::uintptr_t>(cutedsl_arg0_data) % 16 == 0, "CuTeDSL kernel cutedsl_fixture: argument mX is not 16-byte aligned");
  const int64_t cutedsl_arg0_size0 = static_cast<int64_t>(p0_size0);
  AOTI_RUNTIME_CHECK(cutedsl_arg0_size0 >= 0 && cutedsl_arg0_size0 <= 2147483647LL, "CuTeDSL kernel cutedsl_fixture: dynamic size 0 of argument mX does not fit in int32");
  const int64_t cutedsl_arg0_size1 = static_cast<int64_t>(p0_size1);
  AOTI_RUNTIME_CHECK(cutedsl_arg0_size1 >= 0 && cutedsl_arg0_size1 <= 2147483647LL, "CuTeDSL kernel cutedsl_fixture: dynamic size 1 of argument mX does not fit in int32");
  cutedsl_fixture_Tensor_mX_t cutedsl_arg0{cutedsl_arg0_data, {static_cast<int32_t>(cutedsl_arg0_size0), static_cast<int32_t>(cutedsl_arg0_size1)}, {static_cast<int64_t>(p0_stride0)}};
  void* cutedsl_arg1_data = p1;  // mOut
  cutedsl_fixture_Tensor_mOut_t cutedsl_arg1{cutedsl_arg1_data};
  const int64_t cutedsl_arg2_value = static_cast<int64_t>(p2);  // i8
  AOTI_RUNTIME_CHECK(cutedsl_arg2_value >= -128LL && cutedsl_arg2_value <= 127LL, "CuTeDSL kernel cutedsl_fixture: scalar i8 does not fit in int8_t");
  const int8_t cutedsl_arg2 = static_cast<int8_t>(cutedsl_arg2_value);
  const int64_t cutedsl_arg3_value = static_cast<int64_t>(p3);  // i16
  AOTI_RUNTIME_CHECK(cutedsl_arg3_value >= -32768LL && cutedsl_arg3_value <= 32767LL, "CuTeDSL kernel cutedsl_fixture: scalar i16 does not fit in int16_t");
  const int16_t cutedsl_arg3 = static_cast<int16_t>(cutedsl_arg3_value);
  const int64_t cutedsl_arg4_value = static_cast<int64_t>(p4);  // i32
  AOTI_RUNTIME_CHECK(cutedsl_arg4_value >= -2147483648LL && cutedsl_arg4_value <= 2147483647LL, "CuTeDSL kernel cutedsl_fixture: scalar i32 does not fit in int32_t");
  const int32_t cutedsl_arg4 = static_cast<int32_t>(cutedsl_arg4_value);
  const int64_t cutedsl_arg5_value = static_cast<int64_t>(p5);  // i64
  const int64_t cutedsl_arg5 = static_cast<int64_t>(cutedsl_arg5_value);
  const int64_t cutedsl_arg6_value = static_cast<int64_t>(p6);  // u8
  AOTI_RUNTIME_CHECK(cutedsl_arg6_value >= 0LL && cutedsl_arg6_value <= 255LL, "CuTeDSL kernel cutedsl_fixture: scalar u8 does not fit in uint8_t");
  const uint8_t cutedsl_arg6 = static_cast<uint8_t>(cutedsl_arg6_value);
  const int64_t cutedsl_arg7_value = static_cast<int64_t>(p7);  // u16
  AOTI_RUNTIME_CHECK(cutedsl_arg7_value >= 0LL && cutedsl_arg7_value <= 65535LL, "CuTeDSL kernel cutedsl_fixture: scalar u16 does not fit in uint16_t");
  const uint16_t cutedsl_arg7 = static_cast<uint16_t>(cutedsl_arg7_value);
  const int64_t cutedsl_arg8_value = static_cast<int64_t>(p8);  // u32
  AOTI_RUNTIME_CHECK(cutedsl_arg8_value >= 0LL && cutedsl_arg8_value <= 4294967295LL, "CuTeDSL kernel cutedsl_fixture: scalar u32 does not fit in uint32_t");
  const uint32_t cutedsl_arg8 = static_cast<uint32_t>(cutedsl_arg8_value);
  const uint64_t cutedsl_arg9 = static_cast<uint64_t>(p9);  // u64
  const uint8_t cutedsl_arg10 = static_cast<uint8_t>((p10) != 0);  // flag
  const float cutedsl_arg11 = static_cast<float>(p11);  // f32
  const float cutedsl_arg12 = static_cast<float>(p12);  // tf32
  const double cutedsl_arg13 = static_cast<double>(p13);  // f64
  const __half_raw cutedsl_arg14 = __double2half(static_cast<double>(p14));  // f16
  const int32_t cutedsl_status = cute_dsl_cutedsl_fixture_wrapper(cutedsl_fixture_module(), &cutedsl_arg0, &cutedsl_arg1, cutedsl_arg2, cutedsl_arg3, cutedsl_arg4, cutedsl_arg5, cutedsl_arg6, cutedsl_arg7, cutedsl_arg8, cutedsl_arg9, cutedsl_arg10, cutedsl_arg11, cutedsl_arg12, cutedsl_arg13, cutedsl_arg14, static_cast<cudaStream_t>(cutedsl_stream));
  AOTI_RUNTIME_CHECK(cutedsl_status == 0, std::string("CuTeDSL kernel cutedsl_fixture: launch failed: ") + cudaGetErrorName(static_cast<cudaError_t>(cutedsl_status)) + ": " + cudaGetErrorString(static_cast<cudaError_t>(cutedsl_status)));
}
""",  # noqa: B950
        )

    def test_emit_cpp_launch_errors(self):
        from torch._inductor.codegen.cutedsl.user_kernel_aot import (
            CuteDSLExportError,
            emit_cpp_launch,
            ScalarParamSpec,
        )

        exported = dataclasses.replace(self._exported(), specs=tuple(_abi_specs()))
        tensors = {"mX": ("x", ["s0", "s1"], ["t0"]), "mOut": ("out", [], [])}
        scalars = {
            s.name: s.name for s in exported.specs if isinstance(s, ScalarParamSpec)
        }

        def launch(exported=exported, **kwargs):
            args = {
                "tensor_args": tensors,
                "scalar_args": scalars,
                "stream_expr": "s",
                "check_align": {},
            }
            return emit_cpp_launch(exported, **{**args, **kwargs})

        launch()
        params = [
            ("char" if name == "i8" else ctype, name)
            for ctype, name in exported.abi.params
        ]
        char_abi = dataclasses.replace(exported.abi, params=tuple(params))
        for kwargs, error, msg in [
            (
                {"tensor_args": {"mX": tensors["mX"]}},
                ValueError,
                r"tensor_args \['mX'\] != tensors \['mOut', 'mX'\]",
            ),
            (
                {"scalar_args": {**scalars, "stream": "s"}},
                ValueError,
                r"scalar_args \[.*'stream'.*\] != scalars",
            ),
            (
                {"check_align": {"i32": 16}},
                ValueError,
                r"check_align names \['i32'\] are not tensors",
            ),
            (
                {"check_align": {"mX": 12}},
                ValueError,
                r"check_align\[mX\]=12 is not a power of two",
            ),
            (
                {"tensor_args": {**tensors, "mX": ("x", ["s0"], ["t0"])}},
                ValueError,
                "mX needs 2 dynamic sizes and 1 dynamic strides, got 1 and 1",
            ),
            (
                {"exported": dataclasses.replace(exported, abi=char_abi)},
                CuteDSLExportError,
                "unsupported scalar C type char for i8",
            ),
        ]:
            with self.subTest(msg), self.assertRaisesRegex(error, msg):
                launch(**kwargs)

    def test_link_inputs_and_flags(self):
        # The runtime archive follows the objects (each once); the version script,
        # named by its content, keeps each prefix's symbols local.
        from torch._inductor.codegen.cutedsl import user_kernel_aot

        archive = "/dsl/lib/libcuda_dialect_runtime_static.a"
        exported = [
            dataclasses.replace(self._exported(), object_path=path)
            for path in ("/a/k1.o", "/b/k2.o", "/a/k1.o")
        ]
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(
                user_kernel_aot, "cutedsl_runtime_archive", return_value=archive
            ),
        ):
            self.assertEqual(
                user_kernel_aot.cutedsl_link_inputs(exported),
                ["/a/k1.o", "/b/k2.o", archive],
            )
            self.assertEqual(user_kernel_aot.cutedsl_link_inputs([]), [])
            flags = user_kernel_aot.cutedsl_link_flags(d, ["k_b", "k_a", "k_b"])
            self.assertEqual(
                flags[0], "-Wl,--exclude-libs,libcuda_dialect_runtime_static.a"
            )
            *opts, script = shlex.split(flags[1])
            self.assertEqual(opts, ["-Xlinker", "--version-script", "-Xlinker"])
            self.assertEqual(os.path.dirname(script), d)
            with open(script) as f:
                self.assertExpectedInline(
                    f.read(),
                    """\
/* CuTeDSL user kernels: export_to_c symbols stay local to this library. */
{
  local:
    k_a_*;
    _mlir_k_a_*;
    k_b_*;
    _mlir_k_b_*;
};
""",
                )
            self.assertEqual(
                user_kernel_aot.cutedsl_link_flags(d, ["k_a", "k_b"]), flags
            )
            self.assertNotEqual(user_kernel_aot.cutedsl_link_flags(d, ["k_a"]), flags)
            self.assertEqual(len(os.listdir(d)), 2)
            self.assertEqual(user_kernel_aot.cutedsl_link_flags(d, []), [])
            with self.assertRaisesRegex(
                ValueError, "prefix 'k-a' is not a C identifier"
            ):
                user_kernel_aot.cutedsl_link_flags(d, ["k-a"])


instantiate_parametrized_tests(CuteDSLHOPTests)
instantiate_parametrized_tests(CuteDSLWrapTests)
instantiate_parametrized_tests(CuteDSLDynamoTests)
instantiate_parametrized_tests(CuteDSLInductorTests)
instantiate_parametrized_tests(CuteDSLAOTITests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
