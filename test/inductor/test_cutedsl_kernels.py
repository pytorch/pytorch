# Owner(s): ["module: inductor"]
"""Tests for user-defined CuTeDSL kernels (``@cute.jit`` host functions) in PT2.

Mirrors test/inductor/test_triton_kernels.py (the reference suite for user-defined
Triton kernels): direct HOP use, the eager launch and wrap_cutedsl tracing (make_fx,
non-strict export).
"""

import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import re
import sys
import types
import unittest
from unittest import mock

import numpy as np

import torch
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    CUTEDSL_CONSTANT_RULES,
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_functional,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLKernel,
    get_cutedsl_mutated_tensors,
    TraceableCuteDSLKernelWrapper,
)
from torch._inductor.test_case import TestCase
from torch._library.cutedsl import wrap_cutedsl
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


HAS_CUDA = torch.cuda.is_available() and torch.version.hip is None

# The DSL is an optional dependency: guard the imports and define the toy kernels
# only when it (and a CUDA device) is available.
_CUTLASS_IMPORT_ERROR = None
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
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
    share as module globals. They are built at import, so if wrap_cutedsl raises, the
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
    """A spy on cute.compile, with the eager compile cache emptied for the duration,
    so that its call count is the number of compiles eager launches did."""
    from torch._native.cutedsl import user_kernel as cutedsl_user_kernel

    with mock.patch.dict(cutedsl_user_kernel._compiled, clear=True):
        with mock.patch.object(cute, "compile", wraps=cute.compile) as compile_spy:
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


instantiate_parametrized_tests(CuteDSLHOPTests)
instantiate_parametrized_tests(CuteDSLWrapTests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
