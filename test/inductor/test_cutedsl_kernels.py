# Owner(s): ["module: inductor"]
"""Tests for user-defined CuTeDSL kernels (``@cute.jit`` host functions) in PT2.

Mirrors test/inductor/test_triton_kernels.py (the reference suite for user-defined
Triton kernels): the HOPs used directly (side table, eager launch, fake mode, make_fx,
functionalization).
"""

import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import re
import unittest
from unittest import mock

import numpy as np

import torch
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_functional,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLKernel,
    get_cutedsl_mutated_tensors,
)
from torch._inductor.test_case import TestCase
from torch.fx.experimental.proxy_tensor import make_fx
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


DEVICE = "cuda"


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
        # A name that is not a parameter (a typo) or is the stream would never be
        # cloned and written back, so a write would be lost: the record refuses it
        # with a ValueError, and a write set that is not "unknown" or an iterable of
        # names with a TypeError. An empty write set does not parse the signature.
        msg = (
            r"^CuTeDSL kernel add_kernel: mutates_args names '{}', but its parameters "
            r"other than the stream 'stream' are \('mX', 'mY', 'mOut', 'n'\)$"
        )
        for names, bad in ((("mout",), "mout"), (["mOut", "stream"], "stream")):
            with self.assertRaisesRegex(ValueError, msg.format(bad)):
                CuteDSLKernel(add_kernel, names)
        msg = 'mutates_args must be "unknown" or an iterable of parameter names, got '
        for bad in ("mOut", 5, ["mOut", 1]):
            with self.assertRaisesRegex(TypeError, re.escape(f"{msg}{bad!r}") + "$"):
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


instantiate_parametrized_tests(CuteDSLHOPTests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
