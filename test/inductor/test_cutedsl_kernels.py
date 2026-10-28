# Owner(s): ["module: inductor"]
"""Tests for user-defined CuTeDSL kernels (``@cute.jit`` host functions) in PT2.

Mirrors test/inductor/test_triton_kernels.py (the reference suite for user-defined
Triton kernels): direct HOP use, wrap_cutedsl tracing (make_fx, non-strict export),
Dynamo, Inductor (mutation without clones, reinplacing, views, dynamic shapes) and
AOTInductor.
"""

import contextlib
import functools
import inspect
import re
import unittest
from unittest import mock

import torch
import torch._dynamo.testing
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
)
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._library.cutedsl import wrap_cutedsl
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


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
# Inductor generates Triton for the code around the user kernel.
requires_triton = unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires Triton")


if HAS_CUTEDSL:
    # --- BEGIN TOY KERNELS ---
    # Every kernel below is compiled and checked with plain CuTeDSL (from_dlpack +
    # cute.compile) by probes/toy_kernels_probe.py, which execs exactly this block.
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

    # Unannotated python-int param k: out = x + k. PT2 treats k as a compile-time
    # constant; the DSL itself turns an unannotated int into a dynamic Int32 arg
    # (cutlass/base_dsl/runtime/jit_arg_adapters.py), so the compiled function still
    # takes it at call time.
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

    # Invalid for wrap_cutedsl (it needs exactly one cuda.CUstream param), but
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

    # --- END TOY KERNELS ---

    # --- BEGIN CASE KERNELS ---
    # Kernels for specific cases of the tests below (not part of the toy block).

    # A runtime Float32 scalar whose default is a cutlass Numeric value: out = x + c.
    @cute.kernel
    def _add_c_k(gX: cute.Tensor, gOut: cute.Tensor, n: cutlass.Int32, c: cutlass.Float32):
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
        mX: cute.Tensor, mOut: cute.Tensor, n: cutlass.Int32, flag, stream: cuda.CUstream
    ):
        _select_k(mX, mOut, n, cutlass.Boolean(flag)).launch(
            grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    # --- END CASE KERNELS ---


DEVICE = "cuda"
# Error types acceptable for user errors (bad arguments to wrap_cutedsl / a wrapper).
USER_ERRORS = (TypeError, ValueError, RuntimeError)

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

    class Toy(torch.nn.Module):
        def forward(self, x, y):
            return toy(x, y)


def _rand(*shape, dtype=torch.float32, requires_grad=False):
    return torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=requires_grad)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _hop_nodes(gm, hop):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is hop]


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


def _normalize_indices(code):
    """Side-table indices depend on what ran before; blank them for expected code."""
    code = re.sub(r"\bkernel_idx = \d+", "kernel_idx = K", code)
    return re.sub(r"\bconstant_args_idx = \d+", "constant_args_idx = C", code)


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


def _pointer_alignment(t):
    ptr = t.data_ptr()
    return next(a for a in (16, 8, 4, 2, 1) if ptr % a == 0)


@functools.cache
def _compile_lock():
    from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

    return CUTEDSL_COMPILE_LOCK


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
    if mutates_args != "unknown":
        mutates_args = tuple(mutates_args)
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
    """Direct use of the HOPs and the side table (no wrap_cutedsl)."""

    def test_side_table_dedupes_constant_args(self):
        # Eager wrapper calls go through the HOP and this table is never reset, so
        # equal constants share one entry; equal values of other types do not.
        table = cutedsl_kernel_side_table
        idx = table.add_constant_args({"alpha": 2, "k": None})
        self.assertEqual(table.add_constant_args({"k": None, "alpha": 2}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": 2.0, "k": None}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": True, "k": None}), idx)
        # Lists by content; objects by identity, even when their reprs are equal.
        lst = table.add_constant_args({"tile": [128, 64]})
        self.assertEqual(table.add_constant_args({"tile": [128, 64]}), lst)

        class Cfg:
            def __init__(self, alpha):
                self.alpha = alpha

            def __repr__(self):
                return "Cfg()"

            __hash__ = None  # unhashable, like a mutable config object

        a = table.add_constant_args({"cfg": Cfg(1.0)})
        self.assertNotEqual(table.add_constant_args({"cfg": Cfg(2.0)}), a)

    def test_runtime_scalars_reach_the_dsl_as_python_values(self):
        # The DSL converts only exact bool/int/float: an IntEnum for an unannotated
        # param, and a cutlass Numeric default, are passed as plain Python values.
        import enum

        from torch._higher_order_ops.cutedsl_kernel_wrap import get_host_fn_info

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

        default = get_host_fn_info(add_c_kernel).defaults["c"]
        self.assertIs(type(default), float)
        self.assertEqual(default, 0.25)
        out = torch.empty_like(x)
        cutedsl_kernel_wrapper_mutation(
            kernel_idx=_add_kernel_to_table(add_c_kernel, ["mOut"]),
            constant_args_idx=cutedsl_kernel_side_table.add_constant_args({}),
            kwargs={"mX": x, "mOut": out, "n": 1000},
        )
        self.assertEqual(out, x + 0.25)

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
        import dataclasses

        r1 = CuteDSLKernel(add_kernel, ("mOut",))
        r2 = CuteDSLKernel(add_kernel, ("mOut",))
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
        # Constexpr and unannotated-param values travel in the side table; each
        # distinct constant is a distinct compiled kernel.
        x = _rand(1000)
        kidx = _add_kernel_to_table(mul_const_kernel, ["mOut"])
        for alpha in (3, 2.5):
            out = torch.zeros_like(x)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cutedsl_kernel_side_table.add_constant_args(
                    {"alpha": alpha}
                ),
                kwargs={"mX": x, "mOut": out, "n": 1000},
            )
            self.assertEqual(out, x * alpha)
        kidx = _add_kernel_to_table(add_scalar_kernel, ["mOut"])
        for k in (5, 7):
            out = torch.zeros_like(x)
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cutedsl_kernel_side_table.add_constant_args({"k": k}),
                kwargs={"mX": x, "mOut": out, "n": 1000},
            )
            self.assertEqual(out, x + k)

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

    @parametrize("api", ["python", "cpp"])
    def test_cutedsl_hop_functionalize_refuses_aliased_writes(self, api):
        # Each written argument is cloned and written back on its own, so the same
        # tensor passed to two written parameters would lose a write: functionalization
        # refuses it. Distinct views of one base are written back correctly.
        from torch._subclasses.functional_tensor import (
            FunctionalTensorMode,
            PythonFunctionalizeAPI,
        )

        kidx = _add_kernel_to_table(add_kernel, "unknown")
        cidx = cutedsl_kernel_side_table.add_constant_args({})

        def trace(f, *args):
            if api == "cpp":
                return make_fx(torch.func.functionalize(f))(*args)
            with FunctionalTensorMode():
                return make_fx(PythonFunctionalizeAPI().functionalize(f))(*args)

        def in_place(x, y):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=_add_kwargs(x, y, x)
            )
            return x

        def views(base, y):
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx,
                constant_args_idx=cidx,
                kwargs=_add_kwargs(base[0], y, base[1]),
            )
            return base

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(
            RuntimeError, "'mX' and 'mOut' are the same tensor"
        ):
            trace(in_place, x.clone(), y)
        base = torch.stack([x, torch.zeros_like(x)])
        gm = trace(views, base.clone(), y)
        self.assertEqual(gm(base.clone(), y), torch.stack([x, x + y]))

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

    def test_wrap_cutedsl_error_tensor_in_container(self):
        # Only top-level tensors become graph arguments.
        w = wrap_cutedsl(scale_by_first_kernel, mutates_args=["mOut"])
        x = _rand(1000)
        with self.assertRaisesRegex(RuntimeError, "inside a container"):
            w(x, torch.empty_like(x), 1000, (x,))

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
        self.assertEqual(type(w).__name__, "TraceableCuteDSLKernelWrapper")
        x, y = _rand(1000), _rand(1000)
        out = torch.zeros_like(x)
        self.assertIsNone(w(x, y, out, 1000))
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

    def test_wrap_cutedsl_error_stream_passed(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(USER_ERRORS, "stream"):
            add_w(x, y, out, 16, stream=_stream())
        with self.assertRaisesRegex(USER_ERRORS, "stream"):
            add_w(x, y, out, 16, _stream())

    def test_wrap_cutedsl_error_stream_params(self):
        # Exactly one cuda.CUstream parameter is required.
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        for fn in (no_stream_kernel, two_stream_kernel):
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(USER_ERRORS, "(?i)stream"):
                    wrap_cutedsl(fn, mutates_args=["mOut"])(x, y, out, 16)

    def test_wrap_cutedsl_error_bad_mutates_args(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        for bad, pattern in (
            (["not_a_param"], "not_a_param"),
            (["n"], "'n'"),  # a scalar parameter cannot be written
            (["stream"], "stream"),
            ("mOut", "mutates_args"),  # a bare string other than "unknown"
        ):
            with self.subTest(mutates_args=bad):
                with self.assertRaisesRegex(USER_ERRORS, pattern):
                    wrap_cutedsl(add_kernel, mutates_args=bad)(x, y, out, 16)

    def test_wrap_cutedsl_error_not_cute_jit(self):
        def plain(mX, mY, mOut, n, stream: cuda.CUstream):
            pass

        with self.assertRaisesRegex(USER_ERRORS, r"(?i)cute\.jit"):
            wrap_cutedsl(plain, mutates_args=["mOut"])
        # A @cute.kernel device function is not a host function either.
        with self.assertRaisesRegex(USER_ERRORS, r"(?i)cute\.jit"):
            wrap_cutedsl(_add_k, mutates_args=["gOut"])

    def test_wrap_cutedsl_error_unwrapped_host_fn(self):
        # inspect.unwrap(host_fn) is the user's undecorated function. Whether it
        # still carries _dsl_cls depends on whether it was compiled before (the DSL
        # deletes it then); either way it is not a @cute.jit function.
        _reference_launch(
            add_kernel, _rand(8), _rand(8), torch.zeros(8, device=DEVICE), 8
        )
        with self.assertRaisesRegex(USER_ERRORS, r"(?i)cute\.jit"):
            wrap_cutedsl(inspect.unwrap(add_kernel), mutates_args=["mOut"])

    def test_wrap_cutedsl_error_cute_tensor_arg(self):
        x, y = _rand(16), _rand(16)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(USER_ERRORS, "torch.Tensor"):
            add_w(from_dlpack(x, assumed_align=16), y, out, 16)

    def test_wrap_cutedsl_error_bad_call(self):
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
        # default applies when the kernel is launched (or generated for Inductor).
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
        from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

        held = []
        orig_compile = cute.compile

        def spy_compile(*args, **kwargs):
            held.append(CUTEDSL_COMPILE_LOCK.locked())
            return orig_compile(*args, **kwargs)

        x, y = _rand(1237), _rand(1237)  # a shape no other test uses: a cache miss
        out = torch.zeros_like(x)
        with mock.patch.object(cute, "compile", spy_compile):
            add_w(x, y, out, 1237)
        self.assertEqual(out, x + y)
        self.assertTrue(held, "the eager launch did not compile in-process")
        self.assertTrue(all(held), held)

    def test_wrap_cutedsl_export_not_serializable(self):
        import io

        class M(torch.nn.Module):
            def forward(self, x, y):
                out = torch.empty_like(x)
                add_w(x, y, out, x.numel())
                return out

        x, y = _rand(1000), _rand(1000)
        ep = torch.export.export(M(), (x, y), strict=False)
        # The node indexes a process-local side table: no serialized form.
        with self.assertRaisesRegex(Exception, "cutedsl user kernels"):
            torch.export.save(ep, io.BytesIO())


@requires_cutedsl
class CuteDSLDynamoTests(TestCase):
    """torch.compile front end: the wrapper (global or created inline) becomes one
    HOP node, and so does a raw @cute.jit call written as for eager (from_dlpack,
    cutlass.torch.current_stream()); recompiles follow the arg classes."""

    def test_wrap_of_a_wrapper_graph_breaks(self):
        # Eager refuses to wrap a wrapper (it is not a @cute.jit function), so Dynamo
        # must not silently replace its write set.
        def f(x, y):
            out = torch.empty_like(x)
            wrap_cutedsl(add_w, mutates_args=["mOut"])(x, y, out, x.numel())
            return out

        x, y = _rand(1000), _rand(1000)
        with self.assertRaises(RuntimeError):
            f(x, y)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(x, y)

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
        from torch._higher_order_ops.cutedsl_kernel_wrap import (
            TraceableCuteDSLKernelWrapper,
        )

        def f(x):
            return x + int(isinstance(add_w, TraceableCuteDSLKernelWrapper))

        x = _rand(4)
        self.assertEqual(torch.compile(f, backend="eager", fullgraph=True)(x), x + 1)

    def test_object_constants_fullgraph(self):
        # Constants that are not Python constants (a config object, a cutlass Numeric
        # value) are passed as the objects, guarded on their identity.
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
        # pointer's own, in the Python launcher); an option that changes the compiled
        # kernel graph-breaks.
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
        # A replacement stored under the DSL function's name (a functools.wraps
        # wrapper, a mock) is not taken for it, so Dynamo traces it like any callable
        # (its body runs) and a mock's (*args, **kwargs) signature cannot break capture.
        from torch._library.cutedsl import is_cutedsl_from_dlpack

        @functools.wraps(from_dlpack)
        def logged(*args, **kwargs):
            return from_dlpack(*args, **kwargs)

        self.assertTrue(is_cutedsl_from_dlpack(from_dlpack))
        self.assertFalse(is_cutedsl_from_dlpack(logged))

        def f(x, y):
            out = torch.zeros_like(x)
            add_kernel(
                cute.runtime.from_dlpack(x),
                cute.runtime.from_dlpack(y),
                cute.runtime.from_dlpack(out),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return out

        x, y = _rand(1000), _rand(1000)
        with mock.patch.object(cute.runtime, "from_dlpack", wraps=from_dlpack):
            self.assertEqual(torch.compile(f, backend="eager")(x, y), x + y)

    def test_raw_call_in_place(self):
        # x = x + y in place: the same tensor for mX and mOut. Every tensor of a raw
        # call counts as written, and two written arguments that are the same tensor
        # cannot be functionalized, so the compile is refused.
        def f(x, y):
            x = x.clone()
            add_kernel(
                from_dlpack(x),
                from_dlpack(y),
                from_dlpack(x),
                x.numel(),
                cutlass_torch.current_stream(),
            )
            return x

        x, y = _rand(1000), _rand(1000)
        self.assertEqual(f(x, y), x + y)
        with self.assertRaisesRegex(Exception, "'mX' and 'mOut' are the same tensor"):
            torch.compile(f, backend="aot_eager", fullgraph=True)(x, y)

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
        self.assertIn(
            "torch.ops.higher_order.cutedsl_kernel_wrapper_", ep.graph_module.code
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
        # serializing the compiled graph (aot_compile) fails loudly.
        def f(x, y):
            out = torch.empty_like(x)
            add_w(x, y, out, x.numel())
            return out * 2

        x, y = _rand(1000), _rand(1000)
        with self.assertRaisesRegex(Exception, "cannot be serialized"):
            torch.compile(f, fullgraph=True).aot_compile(((x, y), {}))

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
        # Launched on the stream Inductor fetched for the device (named
        # `raw_stream0` or `stream0` depending on the version), never a constant.
        self.assertRegex(launches[0][1], r"stream=(raw_)?stream\d+$")

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


instantiate_parametrized_tests(CuteDSLHOPTests)
instantiate_parametrized_tests(CuteDSLWrapTests)
instantiate_parametrized_tests(CuteDSLDynamoTests)
instantiate_parametrized_tests(CuteDSLInductorTests)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
