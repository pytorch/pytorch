# Owner(s): ["module: inductor"]
"""Tests for user-defined CuTeDSL kernels (``@cute.jit`` host functions) in PT2."""

import contextlib
import dataclasses
import functools
import gc
import inspect
import logging
import unittest
import weakref
from unittest import mock

import numpy as np

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import HardwareClassification


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


def _add_kwargs(x, y, out, n=None):
    """``kwargs`` for add_kernel: tensors and the runtime scalar n."""
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
    """Direct use of the eager launch."""

    hw_classification = HardwareClassification.CUDA

    def test_constants_from_cutlass_named_modules_are_keyed_by_content(self):
        # The eager compile cache keys an object of a class from a module merely
        # named like the cutlass package by its attributes, not by a numeric .value
        # like a cutlass Numeric's (test/test_hop_infra.py has the side table's part).
        # Equal objects share a compile; objects with equal .values do not.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        class Cfg:
            __module__ = "cutlass_configs"

            def __init__(self, **attrs):
                vars(self).update(attrs)

        x = _rand(1000)
        with _counting_compiles() as compile_spy:
            for alpha in (2.0, 3.0, 2.0):
                out = torch.empty_like(x)
                cfg = Cfg(value=1, alpha=alpha)
                args = {"mX": x, "mOut": out, "n": 1000, "cfg": cfg}
                launch_cutedsl_kernel(mul_cfg_kernel, args)
                self.assertEqual(out, x * alpha)
        self.assertEqual(compile_spy.call_count, 2)

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
        # Float constants, np.float32 ones too, are keyed by their bits in the eager
        # compile cache, as in the side table (test/test_hop_infra.py): -0.0 and 0.0
        # are different constants (x * alpha has the sign of alpha for x > 0).
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        x = torch.rand(1000, device=DEVICE) + 1
        for alpha in (0.0, -0.0, np.float32(0.0), np.float32(-0.0)):
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
            # The shape alone: the kernel reads its extent from mX's shape, so the
            # compile for 2 rows would leave rows 2 and 3 of a 4-row output unwritten.
            short, tall = _rand(2, 8), _rand(4, 8)
            self.assertEqual(short.stride(), tall.stride())
            add_one_2d(short)
            add_one_2d(tall)
            self.assertEqual(compile_spy.call_count, 7)

    def test_launch_keeps_host_functions_made_for_a_launch(self):
        # Freeing a compile unloads its CUDA library, whose kernel a CUDA graph that
        # captured the launch still runs: the cache keeps a host function made for
        # one launch, and so its compiles, after the caller drops it.
        from torch._native.cutedsl import user_kernel as cutedsl_user_kernel
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        def make_host_fn():
            @cute.jit
            def add_one(mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
                add_one_2d_kernel(mX, mOut, stream)

            return add_one

        x = _rand(4, 8)
        out = torch.empty_like(x)
        fn = make_host_fn()
        fn_ref = weakref.ref(fn)
        # Not _counting_compiles: its spy would keep fn alive through its call_args.
        with mock.patch.dict(cutedsl_user_kernel._compiled, clear=True):
            launch_cutedsl_kernel(fn, {"mX": x, "mOut": out})
            del fn
            gc.collect()
            self.assertIsNotNone(fn_ref())
        self.assertEqual(out, x + 1)

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

    def test_launch_on_the_current_stream(self):
        # Without stream=, the launch is on the current stream: under
        # torch.cuda.stream(side), queued behind a sleep and a write of its input on
        # side, the kernel reads the new input.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel
        from torch.testing._internal.common_utils import get_cycles_per_ms

        x, y, new_x = _rand(1000), _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        args = _add_kwargs(x, y, out)
        launch_cutedsl_kernel(add_kernel, args)  # compiles outside the timed part
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(int(200 * get_cycles_per_ms()))
            x.copy_(new_x)
            launch_cutedsl_kernel(add_kernel, args)
        torch.cuda.synchronize()
        self.assertEqual(out, new_x + y)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_launch_shares_compiles_across_devices_of_one_capability(self):
        # A compiled function's kernels are resolved on every device (see the
        # docstring of torch/_native/cutedsl/user_kernel.py), so a signature
        # compiled under cuda:0 launches on cuda:1 without another compile.
        # TODO: run this in a CI job that has two GPUs and installs nvidia-cutlass-dsl;
        # the smoke jobs that run this file skip it unless their runner has two.
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

    def test_numpy_scalars_reach_the_dsl_as_python_values(self):
        # A numpy bool, integer or floating scalar for an unannotated parameter is a
        # runtime scalar, passed or as a default, so other values of the same types
        # share a compile; one for a cutlass float parameter reaches it as a float,
        # and so does an int, even 2**64, which nvidia-cutlass-dsl 4.8 refuses to
        # compile as an int: it comes first, so the compile sees it.
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
            for c in (2**64, np.float16(0.5), np.int64(2)):
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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
