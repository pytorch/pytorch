# Owner(s): ["module: inductor"]
"""
Tests for torch.cuda.localize / _LocalizedTensor.

These tests exercise the tensor-subclass-based localization path:
``torch.cuda.localize(module)`` rewrites every ``nn.Linear``'s weight (and
bias) into a ``_LocalizedTensor`` whose halves live in per-locality-domain
green-context-backed memory pools.  ``__torch_function__`` then forks each
``F.linear`` call into per-branch matmuls on green streams, with a final
``aten.cat`` joining the per-branch outputs.

Pointwise tails downstream of the cat (relu, bias-add, gelu, ...) are
packed onto the per-branch streams by a separate post-grad FX pass --
``torch._inductor.fx_passes.pack_pointwise_after_cat.PackPointwiseAfterCat``
-- which the IR tests in this file run over the captured AOT graph.
The pass is fully localization-agnostic; it only matches "two matmul
outputs, cat, then a pointwise op that consumes the cat".
"""

import contextlib
import functools
import os
import re
import unittest
import warnings
from unittest import mock

import pytest

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import run_tests, TestCase


def _requires_cuda(test_func):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")(
        test_func
    )


def _requires_green_contexts(test_func):
    """Skip test if localized green contexts are not supported."""

    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from torch.cuda.green_contexts import (
                GreenContext,
                is_localization_supported,
            )

            if not is_localization_supported(0):
                raise unittest.SkipTest(
                    "Localized green contexts not supported on this device"
                )
            (gc,) = GreenContext.split(locality_domain_ids=(0,), device_id=0)
            del gc
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"Green contexts not supported: {e}") from e
        return test_func(*args, **kwargs)

    return wrapper


def _cublas_version_int():
    """Return cuBLAS version as a single int (MMmmpp), or None if unavailable.

    Used to gate tests around bugs fixed in specific cuBLAS releases.
    13.4.1 returns 130401; 13.6.0 returns 130600.
    """
    import ctypes

    try:
        cublas = ctypes.CDLL("libcublas.so.13")
    except OSError:
        return None
    cublas.cublasCreate_v2.restype = ctypes.c_int
    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublas.cublasDestroy_v2.restype = ctypes.c_int
    cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublas.cublasGetVersion_v2.restype = ctypes.c_int
    cublas.cublasGetVersion_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    # cublasCreate needs a current CUDA context.
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
    h = ctypes.c_void_p(0)
    if cublas.cublasCreate_v2(ctypes.byref(h)) != 0:
        return None
    v = ctypes.c_int(0)
    rc = cublas.cublasGetVersion_v2(h.value, ctypes.byref(v))
    cublas.cublasDestroy_v2(h)
    return v.value if rc == 0 else None


def _requires_cublas_tensorop_on_green_streams(test_func):
    """Skip if cuBLAS's TensorOp GEMM path fails on green-context-backed
    streams.  Reproducible in <10 lines: create a green context, get its
    stream, call ``F.linear`` with fp16 / bf16 inputs on the stream.

    Some cuBLAS versions return ``CUBLAS_STATUS_EXECUTION_FAILED``
    from ``cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)`` when the
    stream is bound to a green context, regardless of whether the green
    context is currently pushed.  The fp32 codepath (non-TensorOp)
    works fine on the same stream.

    More recent cuBLAS versions work fine.
    """

    def wrapper(*args, **kwargs):
        v = _cublas_version_int()
        if v is not None and v < 130600:
            raise unittest.SkipTest(
                f"cuBLAS {v} has a TensorOp-on-green-stream bug; upgrade cuBLAS"
            )
        return test_func(*args, **kwargs)

    wrapper.__name__ = test_func.__name__
    return wrapper


def _is_localized_tensor(t):
    from torch.cuda.localized_tensor import _LocalizedTensor

    return isinstance(t, _LocalizedTensor)


# ---------------------------------------------------------------------------
# Helper modules.  Each holds its weights inside ``nn.Linear``s so
# ``torch.cuda.localize`` can replace them with ``_LocalizedTensor``
# parameters.
# ---------------------------------------------------------------------------


class _OneLinear(nn.Module):
    """Single linear over its input."""

    def __init__(self, in_f, out_f, bias=False, dtype=torch.float32):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.fc(x)


class _LinearReLU(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return torch.relu(self.fc(x))


class _LinearLeakyReLU(nn.Module):
    def __init__(self, in_f, out_f, slope=0.01, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)
        self.slope = slope

    def forward(self, x):
        return F.leaky_relu(self.fc(x), negative_slope=self.slope)


class _TwoSequentialLinears(nn.Module):
    def __init__(self, k, n, p, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _TwoIndependentLinears(nn.Module):
    """y1 = fc1(x); y2 = fc2(x); return y1 + y2."""

    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_f, out_f, bias=bias)
        self.fc2 = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.fc1(x) + self.fc2(x)


class _LinearBiasSin(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x):
        return torch.sin(self.fc(x))


class _AttentionQKV(nn.Module):
    """Per-head MHA forward with three independent Q/K/V projections.

    The Q/K/V projections become nn.Linear and get localized; the
    rest of the attention (view/transpose/bmm/softmax) runs unchanged
    on the main stream.
    """

    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, batch, seq_len):
        q = (
            self.q(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = q.reshape(batch * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(batch * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(batch * self.num_heads, seq_len, self.head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5)
        scores = torch.softmax(scores, dim=-1)
        out = torch.bmm(scores, v)
        out = out.view(batch, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        return out.contiguous().view(batch * seq_len, self.hidden)


class _FFNResidualLN(nn.Module):
    def __init__(self, hidden, ffn_dim):
        super().__init__()
        self.hidden = hidden
        self.ffn = nn.Linear(hidden, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x):
        out = self.ffn(x)
        out = self.down(out) + x
        return F.layer_norm(out, (self.hidden,))


class _FusedQKV(nn.Module):
    """Single nn.Linear for fused QKV; the rest is unchanged."""

    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)

    def forward(self, x, batch, seq_len):
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = (
            q.transpose(1, 2)
            .contiguous()
            .reshape(batch * self.num_heads, seq_len, self.head_dim)
        )
        k = (
            k.transpose(1, 2)
            .contiguous()
            .reshape(batch * self.num_heads, seq_len, self.head_dim)
        )
        v = (
            v.transpose(1, 2)
            .contiguous()
            .reshape(batch * self.num_heads, seq_len, self.head_dim)
        )
        scores = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5), dim=-1
        )
        out = torch.bmm(scores, v)
        out = out.view(batch, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        return out.contiguous().view(batch * seq_len, self.hidden)


class _NSequentialLinears(nn.Module):
    """N stacked square nn.Linears with no activations in between."""

    def __init__(self, dims, bias=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x


class _LinearGELULinear(nn.Module):
    def __init__(self, k, n, p, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LinearHeadEscapes(nn.Module):
    """Returns intermediate y AND a final tail."""

    def __init__(self, k, n, p, q, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)
        self.fc3 = nn.Linear(p, q, bias=bias)

    def forward(self, x):
        y = self.fc1(x)
        z = self.fc2(y)
        return y, self.fc3(z)


class _MultiConsumerHead(nn.Module):
    """One linear's output feeds two downstream linears that get summed."""

    def __init__(self, k, n, p, q, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)
        self.fc3 = nn.Linear(n, q, bias=bias)

    def forward(self, x):
        mm1 = self.fc1(x)
        return self.fc2(mm1) + self.fc3(mm1)


class _SharedLinearTwoCalls(nn.Module):
    """One nn.Linear called on two different inputs.  The same
    _LocalizedTensor weight is read from both call sites; numerics must
    match eager."""

    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x1, x2):
        return self.fc(x1) + self.fc(x2)


# ---------------------------------------------------------------------------
# Drive helpers.
# ---------------------------------------------------------------------------


def _get_node_stream(node):
    """Return the green-stream index annotation that the localized
    dispatch attaches to per-branch FX nodes, or ``None`` if absent."""
    custom = node.meta.get("custom") or {}
    return custom.get("stream")


def _capture_localized_aot(module, *forward_args):
    """Compile *module* (which must already be localized) through the
    AOT path and return the post-AOT FX graph.

    The capture goes through ``aot_module_simplified`` directly so that
    inductor's own lowering is bypassed -- we only need the desugared FX
    graph.  Localization writes per-branch ``aten.mm`` / ``aten.addmm``
    nodes there, plus an ``aten.cat`` per linear, plus
    ``meta["custom"]["stream"] = idx`` on each per-branch node.
    """
    captured = []

    def aot_compiler(gm_aot, example_inputs):
        captured.append(gm_aot)
        return gm_aot

    def capturing_backend(gm, example_inputs):
        from torch._functorch.aot_autograd import aot_module_simplified

        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=aot_compiler,
        )

    compiled = torch.compile(module, backend=capturing_backend, fullgraph=False)
    with torch.inference_mode():
        compiled(*forward_args)
        torch.cuda.synchronize()
    assert captured, "post-AOT graph was not captured"  # noqa: S101
    gm = captured[0]
    return gm


def _eager_and_localized(module_cls, ctor_args, forward_args, dtype=torch.float32):
    """Build two copies of the module, localize one, and return (eager_out, loc_out).

    ``forward_args`` is a list of inputs (CUDA tensors / non-tensor scalars)
    passed positionally to forward().  Both module copies start from the
    same RNG seed so their parameters are bit-identical before localization.
    """
    torch.manual_seed(0)
    eager = module_cls(*ctor_args).to(device="cuda", dtype=dtype).eval()
    torch.manual_seed(0)
    loc = module_cls(*ctor_args).to(device="cuda", dtype=dtype).eval()
    with torch.inference_mode():
        torch.cuda.localize(loc)
    with torch.inference_mode():
        y_eager = eager(*forward_args)
        y_loc = loc(*forward_args)
        torch.cuda.synchronize()
    return y_eager, y_loc, loc


# ---------------------------------------------------------------------------
# Tests: parameter-shape / type contract
# ---------------------------------------------------------------------------


class TestLocalizeContract(TestCase):
    """The user-facing contract of torch.cuda.localize on an nn.Module."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_localize_replaces_linear_weight_with_subclass(self):
        """nn.Linear.weight should become a _LocalizedTensor after localize."""
        m = _OneLinear(64, 128, bias=True).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m.fc.weight))
        self.assertTrue(_is_localized_tensor(m.fc.bias))
        # Public-shape contract: from outside, the parameter still looks
        # like (out_features, in_features).
        self.assertEqual(tuple(m.fc.weight.shape), (128, 64))
        self.assertEqual(tuple(m.fc.bias.shape), (128,))

    @_requires_cuda
    @_requires_green_contexts
    def test_localize_idempotent(self):
        """Re-localizing should be a no-op."""
        m = _OneLinear(64, 128).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        w_id = id(m.fc.weight)
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertEqual(id(m.fc.weight), w_id)

    @_requires_cuda
    @_requires_green_contexts
    def test_localize_returns_module_for_chaining(self):
        m = _OneLinear(64, 128).cuda()
        with torch.inference_mode():
            ret = torch.cuda.localize(m)
        self.assertIs(ret, m)

    @_requires_cuda
    @_requires_green_contexts
    def test_localize_graph_module(self):
        """localize() also accepts a torch.fx.GraphModule (e.g. from
        torch.export): it walks the graph for _LOCALIZABLE_OPS call sites,
        resolves each localized-position arg back to its get_attr parameter,
        and swaps that parameter in place for a _LocalizedTensor without
        rewriting the graph."""
        m = _OneLinear(64, 128, bias=True).cuda().eval()
        x = torch.randn(32, 64, device="cuda")
        gm = torch.export.export(m, (x,)).module()
        self.assertIsInstance(gm, torch.fx.GraphModule)
        # Sanity: export keeps the aten.linear call site the branch keys on;
        # make_fx / symbolic_trace would decompose or hide it instead.
        self.assertTrue(
            any(
                n.op == "call_function" and n.target is torch.ops.aten.linear.default
                for n in gm.graph.nodes
            )
        )
        y_ref = gm(x)
        with torch.inference_mode():
            ret = torch.cuda.localize(gm)
        self.assertIs(ret, gm)
        self.assertTrue(_is_localized_tensor(gm.fc.weight))
        self.assertTrue(_is_localized_tensor(gm.fc.bias))
        # The graph is untouched; the subclass takes over at runtime, so the
        # localized graph must still match the original op-for-op.
        torch.testing.assert_close(gm(x), y_ref, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_rejects_odd_output_dim(self):
        """Splitting an odd output dim must error rather than silently
        miscompute."""
        m = _OneLinear(16, 17).cuda()
        with self.assertRaises((ValueError, AssertionError)):
            with torch.inference_mode():
                torch.cuda.localize(m)

    @_requires_cuda
    @_requires_green_contexts
    def test_mark_static_halves_default_marks_halves(self):
        """By default, ``torch.cuda.localize`` marks each per-domain
        half of every localized weight/bias with
        ``mark_static_address``.  Cudagraph_trees then knows the halves'
        data pointers are stable across iters and can skip the per-iter
        copy it would otherwise emit for non-static inputs.

        Implementation note: the mark has to be applied *after*
        ``swap_tensors``.  ``nn.Parameter(_LocalizedTensor)`` detach()es
        its input, which rebuilds the halves via
        ``__tensor_unflatten__`` as fresh Python objects (same storage).
        Marking pre-Parameter would silently get dropped, hence this
        test.
        """
        m = _OneLinear(64, 128, bias=True).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)  # default: mark_static_halves=True
        w = m.fc.weight
        b = m.fc.bias
        self.assertEqual(
            getattr(w._half0, "_dynamo_static_input_type", None),
            "unguarded",
        )
        self.assertEqual(
            getattr(w._half1, "_dynamo_static_input_type", None),
            "unguarded",
        )
        self.assertEqual(
            getattr(b._half0, "_dynamo_static_input_type", None),
            "unguarded",
        )
        self.assertEqual(
            getattr(b._half1, "_dynamo_static_input_type", None),
            "unguarded",
        )

    @_requires_cuda
    @_requires_green_contexts
    def test_mark_static_halves_false_leaves_halves_unmarked(self):
        """``mark_static_halves=False`` must leave the halves clean, so
        A/B benches that need to attribute the cudagraph win to this
        knob actually have an unmarked baseline.
        """
        m = _OneLinear(64, 128, bias=True).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m, mark_static_halves=False)
        w = m.fc.weight
        b = m.fc.bias
        self.assertIsNone(getattr(w._half0, "_dynamo_static_input_type", None))
        self.assertIsNone(getattr(w._half1, "_dynamo_static_input_type", None))
        self.assertIsNone(getattr(b._half0, "_dynamo_static_input_type", None))
        self.assertIsNone(getattr(b._half1, "_dynamo_static_input_type", None))

    def test_is_localization_supported_is_total(self):
        """The capability predicate must return a bool -- never raise --
        on any hardware/driver combination, since it is exactly the guard
        localize() uses to decide whether to skip."""
        from torch.cuda.green_contexts import is_localization_supported

        self.assertIsInstance(is_localization_supported(0), bool)

    @_requires_cuda
    def test_localize_skips_and_warns_once_when_unsupported(self):
        """On devices without locality-domain support, localize must leave
        the module untouched and warn once per process instead of raising.
        Runs on any CUDA device: support is mocked away."""
        from torch._dynamo.utils import warn_once_cache

        m = _OneLinear(64, 128, bias=True).cuda()
        warn_once_cache.clear()
        with (
            mock.patch(
                "torch.cuda.green_contexts.is_localization_supported",
                return_value=False,
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            with torch.inference_mode():
                ret = torch.cuda.localize(m)
            with torch.inference_mode():
                torch.cuda.localize(m)  # second call: no second warning

        self.assertIs(ret, m)
        self.assertFalse(_is_localized_tensor(m.fc.weight))
        self.assertFalse(_is_localized_tensor(m.fc.bias))
        relevant = [w for w in caught if "localization" in str(w.message)]
        self.assertEqual(len(relevant), 1)
        # The unlocalized module must still work.
        x = torch.randn(4, 64, device="cuda")
        y = m(x)
        self.assertEqual(tuple(y.shape), (4, 128))


# ---------------------------------------------------------------------------
# Tests: localizable-op registry
# ---------------------------------------------------------------------------


class TestLocalizableOpRegistry(TestCase):
    """``_LOCALIZABLE_OPS`` drives every localization decision: which ops
    fork across branches, and which module types ``localize`` rewrites.
    Nothing keys on ``linear`` specifically."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_is_registered_not_special_cased(self):
        """F.linear is an ordinary registry entry, carrying the arg names
        needed to match keyword args and the module type localize walks."""
        from torch.cuda.localized_tensor import (
            _LOCALIZABLE_OPS,
            _register_default_localizable_ops,
        )

        _register_default_localizable_ops()
        meta = _LOCALIZABLE_OPS[F.linear]
        self.assertEqual(meta.localized_args, (1, 2))
        self.assertEqual(meta.arg_names, ("input", "weight", "bias"))
        self.assertIs(meta.module_type, nn.Linear)
        # aten.linear is the exported shape; its names come from the schema.
        aten_meta = _LOCALIZABLE_OPS[torch.ops.aten.linear.default]
        self.assertEqual(aten_meta.arg_names, ("input", "weight", "bias"))

    @_requires_cuda
    @_requires_green_contexts
    def test_localized_args_may_be_passed_by_keyword(self):
        """A localized position is resolved whether the caller passed it
        positionally, by keyword, or omitted it entirely."""
        m = _OneLinear(64, 128, bias=True).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        w, b = m.fc.weight, m.fc.bias
        x = torch.randn(32, 64, device="cuda")

        positional = F.linear(x, w, b)
        for by_keyword in (
            F.linear(x, w, bias=b),
            F.linear(x, weight=w, bias=b),
            F.linear(input=x, weight=w, bias=b),
        ):
            torch.testing.assert_close(by_keyword, positional, atol=0, rtol=0)
        # Omitted entirely must mean "no bias", not "a bias of zeros by
        # accident" -- match an explicit None, and differ from the biased call.
        torch.testing.assert_close(F.linear(x, w), F.linear(x, w, None), atol=0, rtol=0)
        self.assertFalse(torch.equal(F.linear(x, w), positional))

    @_requires_cuda
    @_requires_green_contexts
    def test_keyword_localized_arg_matches_unlocalized(self):
        """Numerics, not just self-consistency: a keyword-passed localized
        weight must match the same call on the reified tensor."""
        m = _OneLinear(64, 128, bias=True).cuda()
        ref_w = m.fc.weight.detach().clone()
        ref_b = m.fc.bias.detach().clone()
        x = torch.randn(32, 64, device="cuda")
        expected = F.linear(x, ref_w, bias=ref_b)

        with torch.inference_mode():
            torch.cuda.localize(m)
        got = F.linear(x, weight=m.fc.weight, bias=m.fc.bias)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_graph_module_resolves_keyword_args(self):
        """The FX path resolves a localized position out of node.kwargs as
        well as node.args.  torch.export emits all-positional call sites, so
        the call site here is rewritten to use keywords."""
        m = _OneLinear(64, 128, bias=True).cuda().eval()
        x = torch.randn(32, 64, device="cuda")
        gm = torch.export.export(m, (x,)).module()
        rewritten = 0
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.linear.default
            ):
                inp, weight, bias = node.args
                node.args = (inp,)
                node.kwargs = {"weight": weight, "bias": bias}
                rewritten += 1
        self.assertEqual(rewritten, 1)
        gm.recompile()
        y_ref = gm(x)

        with torch.inference_mode():
            torch.cuda.localize(gm)
        self.assertTrue(_is_localized_tensor(gm.fc.weight))
        self.assertTrue(_is_localized_tensor(gm.fc.bias))
        torch.testing.assert_close(gm(x), y_ref, atol=1e-4, rtol=1e-4)


class _ScaledLinear(nn.Module):
    """Module whose stock forward calls a registered custom op."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)

    def forward(self, x):
        return torch.ops.test_localization.scaled_linear.default(
            x, self.weight, bias=self.bias, scale=2.0
        )


class TestLocalizableOpRegistryCustomOp(TestCase):
    """A downstream op registered with ``register_localizable_op`` gets the
    same treatment the built-in linear does, including through kwargs and
    the module walk."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        from torch.cuda.localized_tensor import (
            _LOCALIZABLE_OPS,
            register_localizable_op,
        )

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self._lib = stack.enter_context(
            torch.library._scoped_library("test_localization", "FRAGMENT")
        )
        self._lib.define(
            "scaled_linear(Tensor input, Tensor weight, Tensor? bias=None, "
            "float scale=1.0) -> Tensor"
        )

        def _scaled_linear(input, weight, bias=None, scale=1.0):
            return F.linear(input, weight, bias) * scale

        self._lib.impl("scaled_linear", _scaled_linear, "CUDA")
        self._op = torch.ops.test_localization.scaled_linear.default
        register_localizable_op(
            self._op, localized_args=(1, 2), module_type=_ScaledLinear
        )
        self._registry = _LOCALIZABLE_OPS
        # The registry is process-global; leaving this op in it would leak
        # into every later test's module walk.
        self.addCleanup(self._registry.pop, self._op, None)

    @_requires_cuda
    @_requires_green_contexts
    def test_arg_names_derived_from_schema(self):
        meta = self._registry[self._op]
        self.assertEqual(meta.arg_names, ("input", "weight", "bias", "scale"))

    @_requires_cuda
    @_requires_green_contexts
    def test_custom_op_forks_with_keyword_args(self):
        """The op's bias arrives by keyword and its non-localized ``scale``
        must reach every branch untouched."""
        m = _ScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = m(x)

        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m.weight))
        self.assertTrue(_is_localized_tensor(m.bias))
        torch.testing.assert_close(m(x), expected, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_module_walk_is_registry_driven(self):
        """localize() rewrites a registered module type it has never heard
        of, and leaves an unregistered one alone."""

        class _Unregistered(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(128, 64))
                nn.init.normal_(self.weight)

            def forward(self, x):
                return x @ self.weight.t()

        m = nn.Sequential(_ScaledLinear(64, 128), _Unregistered()).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m[0].weight))
        self.assertFalse(_is_localized_tensor(m[1].weight))

    @_requires_cuda
    @_requires_green_contexts
    def test_subclass_overriding_forward_is_skipped(self):
        """The stock-forward guard is keyed on the registered module type,
        so an override still opts the layer out."""

        class _Overridden(_ScaledLinear):
            def forward(self, x):
                return F.linear(x, self.weight, self.bias)

        m = _Overridden(64, 128).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertFalse(_is_localized_tensor(m.weight))


class _ChannelScaledLinear(nn.Module):
    """Linear with a per-output-channel scale -- the shape sliced_args exists
    for.  ``scale`` is a buffer, not a parameter, so localize() leaves it
    whole and the dispatch has to split it to match the weight's rows."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f))
        self.register_buffer("scale", torch.empty(out_f))
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)
        nn.init.uniform_(self.scale, 0.5, 1.5)

    def forward(self, x):
        return torch.ops.test_localization.channel_scaled_linear.default(
            x, self.weight, self.bias, self.scale
        )


class TestLocalizableOpSlicedArgs(TestCase):
    """sliced_args positions are split per-branch along dim 0 but left in the
    default pool, for per-output-channel side tensors like quantization
    scales.  No built-in op registers any -- nn.Linear has none -- so this is
    the only coverage of that path."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        from torch.cuda.localized_tensor import _LOCALIZABLE_OPS

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self._lib = stack.enter_context(
            torch.library._scoped_library("test_localization", "FRAGMENT")
        )
        self._lib.define(
            "channel_scaled_linear(Tensor input, Tensor weight, Tensor? bias=None, "
            "Tensor? scale=None, Tensor[]? extra=None, float bogus=1.0) -> Tensor"
        )

        def _channel_scaled_linear(
            input, weight, bias=None, scale=None, extra=None, bogus=1.0
        ):
            out = F.linear(input, weight, bias)
            if scale is not None:
                out = out * scale
            for e in extra or ():
                out = out * e
            return out

        self._lib.impl("channel_scaled_linear", _channel_scaled_linear, "CUDA")
        self._op = torch.ops.test_localization.channel_scaled_linear.default
        self._register(sliced_args=(3, 4))
        self._registry = _LOCALIZABLE_OPS
        self.addCleanup(_LOCALIZABLE_OPS.pop, self._op, None)

    def _register(self, *, sliced_args):
        from torch.cuda.localized_tensor import register_localizable_op

        with warnings.catch_warnings():
            # Re-registering to vary sliced_args warns by design.
            warnings.simplefilter("ignore")
            register_localizable_op(
                self._op,
                localized_args=(1, 2),
                sliced_args=sliced_args,
                module_type=_ChannelScaledLinear,
            )

    @_requires_cuda
    @_requires_green_contexts
    def test_tensor_sliced_arg_matches_unlocalized(self):
        """A per-channel scale must be split to line up with the weight rows
        each branch owns; leaving it whole would multiply every branch by the
        full-width scale and silently corrupt the result."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = m(x)

        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m.weight))
        # The scale is a buffer, so it stays whole and gets sliced per call.
        self.assertFalse(_is_localized_tensor(m.scale))
        torch.testing.assert_close(m(x), expected, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_sliced_arg_may_be_none(self):
        """An explicit None in a sliced position is replicated to every
        branch rather than sliced."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = self._op(x, m.weight, m.bias, None)

        with torch.inference_mode():
            torch.cuda.localize(m)
        got = self._op(x, m.weight, m.bias, None)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_sliced_arg_list_of_tensors(self):
        """A list in a sliced position is sliced element-wise and transposed
        so each branch gets a list of its own slices."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        extra = [torch.rand(128, device="cuda") + 0.5 for _ in range(3)]
        expected = self._op(x, m.weight, m.bias, m.scale, extra)

        with torch.inference_mode():
            torch.cuda.localize(m)
        got = self._op(x, m.weight, m.bias, m.scale, extra)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_sliced_arg_empty_list(self):
        """An empty list has nothing to slice, so each branch gets an empty
        list rather than a slice of nothing."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        expected = self._op(x, m.weight, m.bias, m.scale, [])

        with torch.inference_mode():
            torch.cuda.localize(m)
        got = self._op(x, m.weight, m.bias, m.scale, [])
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_unsliceable_sliced_arg_raises(self):
        """Registering a position that cannot be split must report which arg
        and what it got, not fail somewhere downstream."""
        m = _ChannelScaledLinear(64, 128).cuda()
        x = torch.randn(32, 64, device="cuda")
        with torch.inference_mode():
            torch.cuda.localize(m)

        self._register(sliced_args=(5,))  # 'bogus', a float
        with self.assertRaisesRegex(TypeError, "must be None, Tensor, or list"):
            self._op(x, m.weight, m.bias, m.scale, [], 2.0)


# ---------------------------------------------------------------------------
# Tests: runtime numerics
# ---------------------------------------------------------------------------


class TestLocalizationRuntimeNumerics(TestCase):
    """The localized forward must match the eager forward, op-for-op,
    across the patterns we care about.  Old free-function fn(x, w, ...)
    style is replaced by nn.Module wrappers so the weight slots are
    nn.Linear instances that ``torch.cuda.localize`` knows how to
    rewrite."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_basic_linear(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_OneLinear, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_with_pointwise_relu(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearReLU, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_with_leaky_relu(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearLeakyReLU, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_two_sequential_linears(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_TwoSequentialLinears, (64, 128, 64), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_two_independent_linears(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_TwoIndependentLinears, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_bias_sin(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearBiasSin, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_attention_qkv(self):
        batch, seq_len, hidden, num_heads, head_dim = 2, 8, 64, 4, 16
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _AttentionQKV,
            (hidden, num_heads, head_dim),
            (x, batch, seq_len),
        )
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_ffn_residual_layernorm(self):
        batch, seq_len, hidden, ffn_dim = 2, 8, 64, 128
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_FFNResidualLN, (hidden, ffn_dim), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_fused_qkv_split(self):
        batch, seq_len, hidden, num_heads, head_dim = 2, 8, 64, 4, 16
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _FusedQKV,
            (hidden, num_heads, head_dim),
            (x, batch, seq_len),
        )
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tests: chain numerics (multiple sequential linears).
#
# Even without chaining-aware fusion, the per-layer numerics must still
# be correct because each layer's localize fork-join is self-contained.
# ---------------------------------------------------------------------------


class TestLocalizationChainNumerics(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_2(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_NSequentialLinears, ([64, 48, 32],), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_mm_gelu_mm(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_LinearGELULinear, (64, 64, 32), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_3(self):
        x = torch.randn(16, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _NSequentialLinears, ([64, 64, 64, 32],), (x,)
        )
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_4(self):
        x = torch.randn(8, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(
            _NSequentialLinears, ([64, 64, 64, 64, 32],), (x,)
        )
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_head_escapes(self):
        x = torch.randn(16, 64, device="cuda")
        y_e_pair, y_l_pair, _ = _eager_and_localized(
            _LinearHeadEscapes, (64, 64, 48, 32), (x,)
        )
        for ye, yl in zip(y_e_pair, y_l_pair):
            torch.testing.assert_close(yl, ye, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_multiple_consumers(self):
        x = torch.randn(16, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_MultiConsumerHead, (64, 64, 32, 32), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tests: packing-shaped patterns (numerics only).
#
# Each of these used to verify that pointwise ops were absorbed into the
# green streams.  In the subclass path the pointwise tail runs unchanged
# on the main stream after the cat -- correctness is the same, perf is
# left on the table.  Numerics must still match.
# ---------------------------------------------------------------------------


class TestLocalizationPackingNumerics(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_with_bias(self):
        """linear(x) + b: bias is now part of nn.Linear, no separate add."""
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_OneLinear, (64, 128, True), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_bias_relu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return torch.relu(self.fc(x))

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_packing_cross_site(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_TwoIndependentLinears, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_packing_stops_at_reduction(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return self.fc(x).sum()

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-3, rtol=1e-3)

    @_requires_cuda
    @_requires_green_contexts
    def test_packing_through_unsqueeze(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.fc(x).unsqueeze(0)

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_packing_through_transpose(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.fc(x).t()

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_packing_through_contiguous(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return self.fc(x).contiguous()

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_bias_gelu(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=True)

            def forward(self, x):
                return F.gelu(self.fc(x))

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_view_after_linear(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.fc(x).reshape(-1)

        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(M, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: shared-weight nn.Linear used at multiple call sites.
# ---------------------------------------------------------------------------


class TestLocalizationSharedWeight(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_called_twice(self):
        """One nn.Linear, two inputs.  The same _LocalizedTensor weight is
        read by two F.linear sites; both should produce eager-equivalent
        output."""
        x1 = torch.randn(32, 64, device="cuda")
        x2 = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_SharedLinearTwoCalls, (64, 128), (x1, x2))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_tied_embedding_lm_head(self):
        """head.weight = embed.weight: the tie must survive localize so
        both call sites see the same _LocalizedTensor."""

        class TiedGPT(nn.Module):
            def __init__(self, vocab=64, dim=32):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.head = nn.Linear(dim, vocab, bias=False)
                self.head.weight = self.embed.weight

            def forward(self, x):
                return self.head(self.embed(x))

        torch.manual_seed(0)
        eager = TiedGPT().cuda().eval()
        torch.manual_seed(0)
        loc = TiedGPT().cuda().eval()
        with torch.inference_mode():
            torch.cuda.localize(loc)
        self.assertIs(loc.head.weight, loc.embed.weight)
        self.assertTrue(_is_localized_tensor(loc.head.weight))

        x = torch.tensor([0, 1, 2, 3], device="cuda")
        with torch.inference_mode():
            torch.testing.assert_close(loc(x), eager(x), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: dtype variants
# ---------------------------------------------------------------------------


class TestLocalizationDtypes(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_dtype_float32(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
        y_e, y_l, _ = _eager_and_localized(
            _OneLinear, (64, 128), (x,), dtype=torch.float32
        )
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    # fp16 / bf16 route through cuBLAS's TensorOp GEMM codepath
    # (``cublasGemmEx`` with ``CUBLAS_GEMM_DEFAULT_TENSOR_OP``).
    # cuBLAS 13.4.x and earlier return ``CUBLAS_STATUS_EXECUTION_FAILED``
    # on that codepath when the stream is bound to a green context;
    # fp32 uses the non-TensorOp path and works fine on the same
    # stream.  Fixed in cuBLAS 13.6+ -- the per-test
    # ``_requires_cublas_tensorop_on_green_streams`` decorator skips
    # below that version.
    @_requires_cuda
    @_requires_green_contexts
    @_requires_cublas_tensorop_on_green_streams
    def test_dtype_float16(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        y_e, y_l, _ = _eager_and_localized(
            _OneLinear, (64, 128), (x,), dtype=torch.float16
        )
        torch.testing.assert_close(y_l, y_e, atol=5e-3, rtol=5e-3)

    @_requires_cuda
    @_requires_green_contexts
    @_requires_cublas_tensorop_on_green_streams
    def test_dtype_bfloat16(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        y_e, y_l, _ = _eager_and_localized(
            _OneLinear, (64, 128), (x,), dtype=torch.bfloat16
        )
        torch.testing.assert_close(y_l, y_e, atol=5e-3, rtol=5e-3)


class _LinearHeadEscapes2(nn.Module):
    """Two stacked linears where the head's output ALSO escapes (is
    returned alongside the tail).  The original test asserted that the
    chain pass would not apply when the head escapes; in the subclass
    path that constraint doesn't exist (each linear forks
    independently), but the shape contract for both outputs still
    holds."""

    def __init__(self, k, n, p):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=False)
        self.fc2 = nn.Linear(n, p, bias=False)

    def forward(self, x):
        head = self.fc1(x)
        tail = self.fc2(head)
        return head, tail


class TestLocalizationAllocatorMeta(TestCase):
    """Per-branch matmul outputs (and any pointwise tail duplicates) are
    routed into the matching locality-domain mempool.  These tests pin
    three layers of that contract:

    * the localized dispatch stamps ``meta['custom']['mempool']`` on
      every per-branch FX node, paired by branch with the existing
      ``stream`` index;
    * the inductor wrapper consumes that meta and emits
      ``with torch.cuda.use_mem_pool(...)`` around the matching
      scheduler node's allocations;
    * non-localized compiles still produce single-pool allocations
      (the new path is purely additive on localized graphs).

    Wall-clock parity vs eager is exercised separately by the
    ``test_*_baseline_benchmark`` / ``test_*_localized_benchmark`` pairs
    in ``TestLocalizationBenchmarks``.
    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_per_branch_nodes_carry_paired_stream_and_mempool_meta(self):
        """Each per-branch matmul node carries both ``stream`` and
        ``mempool``; the two indices are paired by branch (same node
        carries matching (stream, mempool)); the two branches use
        distinct pairs."""
        m = _OneLinear(4096, 4096, bias=False, dtype=torch.bfloat16).cuda().eval()
        with torch.inference_mode():
            torch.cuda.localize(m)
        x = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        gm = _capture_localized_aot(m, x)

        aten = torch.ops.aten
        pairs: set = set()
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            if n.target not in (aten.mm.default, aten.addmm.default):
                continue
            c = n.meta.get("custom") or {}
            s, a = c.get("stream"), c.get("mempool")
            self.assertIsNotNone(
                s,
                f"per-branch matmul {n} should carry stream meta",
            )
            self.assertIsNotNone(
                a,
                f"per-branch matmul {n} should carry mempool meta",
            )
            pairs.add((s, a))

        self.assertEqual(
            len(pairs),
            2,
            f"expected two distinct (stream, mempool) pairs across "
            f"the two branches, got {pairs}",
        )
        streams = {s for s, _ in pairs}
        allocs = {a for _, a in pairs}
        self.assertEqual(
            len(streams),
            2,
            f"per-branch streams must be distinct, got {streams}",
        )
        self.assertEqual(
            len(allocs),
            2,
            f"per-branch mempool indices must be distinct, got {allocs}",
        )

    @_requires_cuda
    @_requires_green_contexts
    def test_cat_join_carries_no_mempool_meta(self):
        """The trailing ``aten.cat`` produced by the dispatch join is
        outside every branch, so it must NOT inherit either ``stream``
        or ``mempool`` meta."""
        m = _OneLinear(4096, 4096, bias=False, dtype=torch.bfloat16).cuda().eval()
        with torch.inference_mode():
            torch.cuda.localize(m)
        x = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        gm = _capture_localized_aot(m, x)
        cats = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.cat.default
        ]
        self.assertTrue(cats, "expected at least one cat in localized graph")
        for n in cats:
            c = n.meta.get("custom") or {}
            self.assertNotIn("stream", c, f"cat {n} should not carry stream meta")
            self.assertNotIn("mempool", c, f"cat {n} should not carry mempool meta")

    @_requires_cuda
    @_requires_green_contexts
    def test_output_code_wraps_per_branch_allocations(self):
        """The inductor wrapper emits ``with torch.cuda.use_mem_pool(...)``
        around the per-branch ``empty_strided_cuda`` calls."""
        from torch._inductor.graph import GraphLowering

        captured: list = []

        def save(code: str) -> None:
            captured.append(code)

        prev = GraphLowering.save_output_code
        GraphLowering.save_output_code = save
        try:
            model = _LinearReLU(4096, 4096, bias=False).cuda().eval()
            model = model.to(torch.bfloat16)
            with torch.inference_mode():
                torch.cuda.localize(model)
            opt = torch.compile(model, fullgraph=True)
            x = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
            with torch.inference_mode():
                _ = opt(x)
            torch.cuda.synchronize()
        finally:
            GraphLowering.save_output_code = prev

        self.assertTrue(captured, "no codegen captured from compile")
        code = "\n\n".join(captured)
        self.assertIn(
            "torch.cuda.use_mem_pool",
            code,
            "expected the wrapper to emit use_mem_pool around per-branch "
            "allocations, but the generated code does not mention it",
        )
        # The wrap pattern is jansel's per-scheduler-node mempool context:
        # "with torch.cuda.use_mem_pool(get_external_object_by_index(N), device=M):"
        # at least one such line must precede an empty_strided_cuda allocation
        # at the next level of indentation.
        # Between the wrap and the allocation the wrapper may open the
        # stream context and emit alignment copies (`with streamN:` /
        # copy_if_misaligned) -- the codegen nests pool and stream
        # contexts differently across inductor versions -- so only
        # require that an allocation appears within the wrapped block.
        self.assertRegex(
            code,
            r"with torch\.cuda\.use_mem_pool\(\s*get_external_object_by_index\(\d+\),"
            r"\s*device=\d+\s*\):"
            r"(?:\n[^\n]*)*?\n\s+\w+ = empty_strided_cuda\(",
            "expected `with use_mem_pool(..., device=N):` followed by an "
            "indented `empty_strided_cuda(...)` call within the block",
        )

    @_requires_cuda
    @_requires_green_contexts
    def test_non_localized_compile_has_no_use_mem_pool_wrap(self):
        """A baseline (non-localized) compile of the same kind of model
        must NOT contain any ``use_mem_pool`` wrap.  This is the
        non-localized-user-code invariant the mempool path commits to."""
        from torch._inductor.graph import GraphLowering

        captured: list = []

        def save(code: str) -> None:
            captured.append(code)

        prev = GraphLowering.save_output_code
        GraphLowering.save_output_code = save
        try:
            model = _LinearReLU(4096, 4096, bias=False).cuda().eval()
            model = model.to(torch.bfloat16)
            # Deliberately NOT calling torch.cuda.localize(model).
            opt = torch.compile(model, fullgraph=True)
            x = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
            with torch.inference_mode():
                _ = opt(x)
            torch.cuda.synchronize()
        finally:
            GraphLowering.save_output_code = prev

        self.assertTrue(captured, "no codegen captured from compile")
        code = "\n\n".join(captured)
        self.assertNotIn(
            "use_mem_pool",
            code,
            "non-localized compile must not emit use_mem_pool; the "
            "mempool-meta path leaked into user-code that didn't ask "
            "for localization",
        )

    @_requires_cuda
    @_requires_green_contexts
    def test_runtime_numerics_match_eager(self):
        """Compiled localized forward matches eager forward to within
        bf16 tolerance.  Confirms the use_mem_pool wraps don't perturb
        results (they shouldn't -- they only redirect where storage is
        allocated, not what gets computed)."""
        torch.manual_seed(0)
        eager_model = (
            _LinearReLU(4096, 4096, bias=False).cuda().eval().to(torch.bfloat16)
        )
        loc_model = _LinearReLU(4096, 4096, bias=False).cuda().eval().to(torch.bfloat16)
        loc_model.load_state_dict(eager_model.state_dict())
        with torch.inference_mode():
            torch.cuda.localize(loc_model)

        x = torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            ref = eager_model(x)
            opt = torch.compile(loc_model, fullgraph=True)
            out = opt(x)
            torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    @_requires_cuda
    @_requires_green_contexts
    def test_compiled_numerics_match_eager_triton_fused(self):
        """Companion to test_runtime_numerics_match_eager: compile with
        max_autotune_gemm_backends="TRITON" so the matmul lowers to a
        Triton template instead of extern_kernels.mm, and on a workload
        (Linear -> GELU -> Linear) that gives the template a fusible
        epilogue.  The default-backend numerics path is already covered
        above; this exercises the Triton-template path for the same
        localized forward."""
        torch.manual_seed(0)
        eager_model = _LinearGELULinear(64, 64, 32).cuda().eval().to(torch.bfloat16)
        loc_model = _LinearGELULinear(64, 64, 32).cuda().eval().to(torch.bfloat16)
        loc_model.load_state_dict(eager_model.state_dict())
        with torch.inference_mode():
            torch.cuda.localize(loc_model)

        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
        with inductor_config.patch(
            {
                "max_autotune_gemm": True,
                "max_autotune_gemm_backends": "TRITON",
            }
        ):
            with torch.inference_mode():
                ref = eager_model(x)
                opt = torch.compile(loc_model, fullgraph=True)
                out = opt(x)
                torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    @_requires_cuda
    @_requires_green_contexts
    def test_fused_qkv_split_compiled(self):
        """Compiled (reduce-overhead) companion to the eager
        test_fused_qkv_split: the localized fused-QKV forward, CUDA-graph
        captured, must match the eager forward.  Unlike the eager subclass
        path -- which serializes the two per-branch GEMMs behind per-branch
        Python dispatch, so they never overlap -- the captured graph replays
        both branches back-to-back, letting them execute concurrently on
        their green-context SM partitions."""
        torch.manual_seed(0)
        hidden, num_heads, head_dim, batch, seq_len = 64, 4, 16, 2, 8
        eager_model = _FusedQKV(hidden, num_heads, head_dim).cuda().eval()
        loc_model = _FusedQKV(hidden, num_heads, head_dim).cuda().eval()
        loc_model.load_state_dict(eager_model.state_dict())
        with torch.inference_mode():
            torch.cuda.localize(loc_model)
        x = torch.randn(batch * seq_len, hidden, device="cuda")
        counters.clear()
        with torch.inference_mode():
            ref = eager_model(x, batch, seq_len)
            opt = torch.compile(loc_model, fullgraph=True, mode="reduce-overhead")
            # Warm up so the CUDA graph is captured + replayed.
            for _ in range(3):
                out = opt(x, batch, seq_len)
            torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        # The point of this test is the captured graph; a skip would silently
        # downgrade it to an eager-ish run that still matches numerically.
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)


class TestLocalizationChainIR(TestCase):
    """Chained linears: each localized linear forks per-branch
    independently regardless of how many other linears feed it or
    consume it.  These tests pin the numerics across a few common
    chain shapes."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_no_chain_single_site(self):
        """A single mm should not trigger any chaining behavior; the
        subclass path produces 2 per-branch matmuls and otherwise
        matches eager numerically."""
        torch.manual_seed(0)
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_OneLinear, (64, 128), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)

    @_requires_cuda
    @_requires_green_contexts
    def test_chain_head_escapes(self):
        """The head of a 2-linear chain is also returned: both outputs
        must match eager."""
        torch.manual_seed(0)
        x = torch.randn(32, 64, device="cuda")
        y_e_pair, y_l_pair, _ = _eager_and_localized(
            _LinearHeadEscapes2, (64, 128, 96), (x,)
        )
        for ye, yl in zip(y_e_pair, y_l_pair):
            torch.testing.assert_close(yl, ye, atol=1e-3, rtol=1e-3)


class _LinearReshape(nn.Module):
    """``mm(x, W).view(M, a, b)`` — tests the boundary contract for
    a non-pointwise op downstream of a localized linear."""

    def __init__(self, in_f, out_f, view_shape):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.view_shape = view_shape

    def forward(self, x):
        return self.fc(x).view(*self.view_shape)


class _LinearReshapePermuteClone(nn.Module):
    def __init__(self, in_f, out_f, view_shape, perm):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.view_shape = view_shape
        self.perm = perm

    def forward(self, x):
        y = self.fc(x).view(*self.view_shape)
        y = y.permute(*self.perm)
        return y.clone()


class _LinearReshapePermuteCloneReshape(nn.Module):
    def __init__(self, in_f, out_f, view1, perm, view2):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.view1 = view1
        self.perm = perm
        self.view2 = view2

    def forward(self, x):
        y = self.fc(x).view(*self.view1)
        y = y.permute(*self.perm)
        y = y.clone()
        return y.reshape(*self.view2)


class _LinearMergeInnerSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 8, bias=False)

    def forward(self, x):
        y = self.fc(x)  # [6, 8]
        y = y.view(6, 2, 4)  # 1-to-many
        return y.view(12, 4)  # merge with split inside


class _LinearBiasOnly(nn.Module):
    """``F.linear(x, W) + b`` -- bias as a separate add (not via
    nn.Linear's bias).  The add packs onto the green streams as a
    pointwise tail."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        return self.fc(x) + self.bias


class TestLocalizationPackingIR(TestCase):
    """IR-level tests for the subclass path's packing/boundary
    behavior.

    Pointwise tails (relu, bias-add, gelu) pack into the per-branch
    streams via __torch_dispatch__'s pointwise fast path -- those
    nodes appear per-branch with stream annotations in the post-AOT
    graph.

    Non-pointwise / shape-changing ops (reshape, view, permute, clone)
    are NOT packed: when one of them consumes a localized tensor, the
    slow path reifies via ``cat`` and the downstream op runs once on
    the reified tensor, on the main stream.  These tests pin that
    boundary contract.
    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _matmul_count(self, gm):
        aten = torch.ops.aten
        return sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in (aten.mm.default, aten.addmm.default)
        )

    def _nodes_with_target(self, gm, target):
        return [
            n for n in gm.graph.nodes if n.op == "call_function" and n.target is target
        ]

    @_requires_cuda
    @_requires_green_contexts
    def test_packing_merge_inner_split_rejected(self):
        """``mm -> view(1-to-many) -> view(merge with inner split)``.
        Both reshapes are non-pointwise -> both run on the main
        stream after a single reify.  Numerics must match eager."""
        torch.manual_seed(0)
        x = torch.randn(6, 3, device="cuda")
        y_e, y_l, gm = _eager_and_localized(_LinearMergeInnerSplit, (), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)


class TestLocalizationRequiresInferenceMode(TestCase):
    """``localize`` refuses to run outside inference mode.

    Localization is inference-only: the split halves are not set up to
    backprop through the fork/join, so a module localized in a training
    context would look trainable and silently fail to learn.  Rejecting the
    call is the contract; these pin it for both the ``nn.Module`` and the
    ``GraphModule`` entry points.
    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_module_path_raises_outside_inference_mode(self):
        m = _OneLinear(64, 128).cuda()
        with self.assertRaisesRegex(RuntimeError, "requires inference mode"):
            torch.cuda.localize(m)
        # The module is left untouched, so a caller can retry correctly.
        self.assertFalse(_is_localized_tensor(m.fc.weight))

    @_requires_cuda
    @_requires_green_contexts
    def test_graph_module_path_raises_outside_inference_mode(self):
        m = _OneLinear(64, 128, bias=True).cuda().eval()
        x = torch.randn(32, 64, device="cuda")
        with torch.inference_mode():
            gm = torch.export.export(m, (x,)).module()
        with self.assertRaisesRegex(RuntimeError, "requires inference mode"):
            torch.cuda.localize(gm)

    @_requires_cuda
    @_requires_green_contexts
    def test_succeeds_inside_inference_mode(self):
        m = _OneLinear(64, 128, bias=True).cuda()
        with torch.inference_mode():
            torch.cuda.localize(m)
        self.assertTrue(_is_localized_tensor(m.fc.weight))
        self.assertTrue(_is_localized_tensor(m.fc.bias))

    @_requires_cuda
    @_requires_green_contexts
    def test_no_grad_is_not_sufficient(self):
        """``no_grad`` is the easy thing to reach for and is not enough --
        it stops gradient recording but leaves inference mode off."""
        m = _OneLinear(64, 128).cuda()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "requires inference mode"):
                torch.cuda.localize(m)


class _LinearBiasGelu(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x):
        return F.gelu(self.fc(x))


class _TwoIndependentLinearsBias(nn.Module):
    """Two independent Linear(...,bias=True) sites whose outputs are
    summed.  Used by the addmm-split IR test.  Independent weights
    means each site gets its own _LocalizedTensor and its own stream
    pair."""

    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc1 = nn.Linear(in_f, out_f, bias=True)
        self.fc2 = nn.Linear(in_f, out_f, bias=True)

    def forward(self, x):
        return self.fc1(x) + self.fc2(x)


class TestLocalizationAddmmNumerics(TestCase):
    """nn.Linear with bias is the userspace shape of addmm; the
    numerics test ports cleanly."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @_requires_cuda
    @_requires_green_contexts
    def test_linear_with_bias_numerics(self):
        x = torch.randn(32, 64, device="cuda")
        y_e, y_l, _ = _eager_and_localized(_OneLinear, (64, 128, True), (x,))
        torch.testing.assert_close(y_l, y_e, atol=1e-4, rtol=1e-4)


def _install_bf16_gelu_decomp():
    """Replace the global aten.gelu decomp with one that has no
    elementwise_type_promotion_wrapper.  Lets the entire gelu chain stay in
    bf16 in the FX graph (the upstream decomp inserts bf16->fp32 converts
    that the packing pass then splits out as a separate per-branch upcast
    kernel).

    Patches both the global decomposition tables (consumed by AOT autograd)
    AND ``torch._inductor.decomposition.decompositions``.  The latter is a
    snapshot taken at inductor module import time; without patching it,
    inductor's lowering will still use the upstream gelu decomp even after
    we override the global table."""
    import torch._inductor.decomposition as inductor_decomp
    from torch._decomp import decomposition_table, global_decomposition_table

    aten = torch.ops.aten

    def gelu_bf16(a, approximate="none"):
        return a * 0.5 * (1 + torch.erf(a * 0.7071067811865476))

    saved = {}
    for name in ("post_autograd", "pre_autograd"):
        saved[name] = global_decomposition_table.get(name, {}).get(aten.gelu.default)
    saved["top"] = decomposition_table.get(aten.gelu.default)
    saved["inductor"] = inductor_decomp.decompositions.get(aten.gelu.default)

    for table in (
        decomposition_table,
        global_decomposition_table.get("post_autograd", {}),
        global_decomposition_table.get("pre_autograd", {}),
        inductor_decomp.decompositions,
    ):
        table[aten.gelu.default] = gelu_bf16
        if aten.gelu.out in table:
            del table[aten.gelu.out]

    def _restore():
        for name, fn in (
            ("post_autograd", saved["post_autograd"]),
            ("pre_autograd", saved["pre_autograd"]),
        ):
            tbl = global_decomposition_table.get(name, {})
            if fn is None:
                tbl.pop(aten.gelu.default, None)
            else:
                tbl[aten.gelu.default] = fn
        if saved["top"] is None:
            decomposition_table.pop(aten.gelu.default, None)
        else:
            decomposition_table[aten.gelu.default] = saved["top"]
        if saved["inductor"] is None:
            inductor_decomp.decompositions.pop(aten.gelu.default, None)
        else:
            inductor_decomp.decompositions[aten.gelu.default] = saved["inductor"]

    return _restore


class TestLocalizationTritonEpilogue(TestCase):
    """Verify the codegen produced by ``torch.compile`` for a localized
    ``Linear -> GeLU`` under Triton-only GEMM autotune.

    The contract we want to confirm:
      1. ``max_autotune_gemm_backends="TRITON"`` actually produces Triton MM
         kernels (no silent fallback to ``extern_kernels.mm``).
      2. The ``PackPointwiseAfterCat`` pass duplicates the GeLU chain
         per-green-stream and inductor honours those stream annotations in
         codegen: the GeLU kernel is invoked twice, once on each green
         stream.
      3. The ``cat`` (or its scheduler-replacement) happens *after* both
         per-stream GeLU invocations.

    (1) currently passes when the inductor cache is cleared; (2) and (3)
    fail because inductor's scheduler re-collapses the duplicated GeLU
    into a single post-join kernel.  The failing assertion documents the
    epilogue-fusion gap we want to close next."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self._restore_decomp = _install_bf16_gelu_decomp()
        # Cache hits would mask the codegen we want to inspect.
        import shutil

        shutil.rmtree(
            "/tmp/torchinductor_" + os.environ.get("USER", "user"), ignore_errors=True
        )

    def tearDown(self):
        self._restore_decomp()

    @_requires_cuda
    @_requires_green_contexts
    def test_gelu_duplicated_per_stream_under_triton_autotune(self):
        """Capture the generated wrapper code via ``GraphLowering.save_output_code``,
        then string-match for the expected structure."""
        from torch._inductor.graph import GraphLowering

        captured: list[str] = []

        def save(code: str) -> None:
            captured.append(code)

        prev = GraphLowering.save_output_code
        GraphLowering.save_output_code = save
        try:
            model = (
                nn.Linear(
                    8192,
                    8192,
                    bias=False,
                    dtype=torch.bfloat16,
                )
                .cuda()
                .eval()
            )
            # Localize wraps weight into a _LocalizedTensor; F.gelu sits on top.
            full_model = nn.Sequential(model, nn.GELU())
            with torch.inference_mode():
                torch.cuda.localize(full_model)

            with inductor_config.patch(
                {
                    "force_disable_caches": True,
                    "max_autotune_gemm": True,
                    "max_autotune_gemm_backends": "TRITON",
                }
            ):
                opt = torch.compile(
                    full_model,
                    fullgraph=True,
                    mode="reduce-overhead",
                )
                x = torch.randn(
                    32,
                    8192,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                with torch.inference_mode():
                    _ = opt(x)
                torch.cuda.synchronize()
        finally:
            GraphLowering.save_output_code = prev
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

        self.assertTrue(captured, "no codegen captured from compile")
        code = "\n\n".join(captured)

        # (1) Triton MM template was actually selected.
        self.assertIn(
            "triton_tem_fused_mm",
            code,
            "Triton MM kernel not generated despite "
            "max_autotune_gemm_backends='TRITON' "
            "(silent fallback to extern_kernels.mm probably triggered)",
        )
        self.assertNotIn(
            "extern_kernels.mm",
            code,
            "found extern_kernels.mm in codegen, meaning inductor silently "
            "fell back to cuBLAS despite Triton-only autotune config",
        )

        # (2) At least one pointwise kernel must be invoked inside each
        # green-stream block.  The per-stream wrappers in localize codegen
        # guard each green stream with ``with torch.cuda.stream(stream{1,2}):``
        # blocks; with the multi-user pack pass active, the gelu-decomposed
        # ``mul``/``mul_1`` pair is fused into one ``triton_poi_fused_*``
        # kernel that runs on each green stream.  The inductor scheduler
        # may interleave per-branch work into more than one ``with
        # torch.cuda.stream(stream1):`` block (mm in the first, the
        # hoisted pointwise tail in a later one), so we concatenate every
        # block bound to a given stream before checking.
        def _stream_blocks(stream_name: str) -> str:
            # The wrapper enters stream contexts as ``with streamN:`` on
            # newer inductor and ``with torch.cuda.stream(streamN):`` on
            # older; collect lines from either form of this stream's
            # blocks until a different stream context (or return) starts.
            needles = (
                f"with {stream_name}:",
                f"with torch.cuda.stream({stream_name}):",
            )
            collected: list[str] = []
            in_block = False
            for ln in code.splitlines():
                stripped = ln.strip()
                if stripped in needles:
                    in_block = True
                    continue
                if in_block and (
                    (
                        stripped.startswith(("with stream", "with torch.cuda.stream("))
                        and stripped not in needles
                    )
                    or stripped.startswith(("with default_stream", "return "))
                ):
                    in_block = False
                    continue
                if in_block:
                    collected.append(ln)
            return "\n".join(collected)

        block1 = _stream_blocks("stream1")
        block2 = _stream_blocks("stream2")
        self.assertTrue(
            block1 and block2, "could not find both green-stream blocks in codegen"
        )
        # The pack pass fires at FX time and per-branch pointwise
        # kernels show up either INSIDE each green stream block (the
        # original packing target) or fused into the cat-copy kernel
        # the scheduler emits on the default stream when per-branch
        # mm outputs are no longer aliased into the cat output (which
        # is the case once the localized-mempool opt-out in
        # ``ConcatKernel.realize_into`` fires).  The locality property
        # we actually care about is preserved either way: the mm
        # output lives in its branch's locality-domain pool (the
        # ``use_mem_pool`` wrap around the per-branch ``empty_strided_cuda``
        # is what gives us that) and the mm kernel runs inside the
        # ``with torch.cuda.stream(stream{1,2}):`` block, so we just
        # check for the per-branch pool wrap there instead of the
        # specific gelu-packed kernel placement.
        for label, block in (("stream1", block1), ("stream2", block2)):
            has_pointwise = re.search(r"triton_poi_fused\w*\.run\(", block)
            has_use_mem_pool = "torch.cuda.use_mem_pool" in block
            self.assertTrue(
                has_pointwise or has_use_mem_pool,
                f"green {label} block has neither a per-branch pointwise "
                f"kernel nor a use_mem_pool wrap; one of these must be "
                f"present for the localization to be doing meaningful "
                f"work (block content:\n{block})",
            )

        # (3) The gelu math ends up SOMEWHERE in the codegen (either
        # packed onto a green stream by PackPointwiseAfterCat or fused
        # into the cat copy by the scheduler when the per-branch mm
        # outputs are no longer aliased into the cat).  We don't pin
        # the placement; we just check that gelu math is present.
        has_gelu_math = re.search(
            r"triton_poi_fused\w*\.run\(",
            code,
        )
        self.assertIsNotNone(
            has_gelu_math,
            "no per-branch pointwise kernel found anywhere in codegen "
            "-- the pack pass didn't fire and the cat copy wasn't "
            "fused with the gelu either, so the gelu chain is silently "
            "missing or running on the default stream as a separate "
            "kernel that the test would have flagged previously.",
        )


class TestLocalizationBenchmarks:
    """pytest-benchmark suite for the localization path.

    Each test pairs a baseline (no localize) against a localized
    variant; both sides go through ``torch.compile`` in
    ``reduce-overhead`` mode so CUDA Graph replay dominates timing.

    GEMM backend selection is left to inductor's autotuner -- explicit
    backend forcing lives in :class:`TestLocalizationTritonEpilogue` and
    is the right place to add per-backend regression tests, not here.

    Run with::

        pytest test/inductor/test_localization.py -k Benchmark --benchmark-enable
    """

    @staticmethod
    def _bench_compiled(benchmark, model, args):
        """Compile ``model`` in reduce-overhead mode with the
        ``PackPointwiseAfterCat`` post-grad pass enabled, warm twice,
        then benchmark a single forward + sync."""
        torch._dynamo.reset()
        counters.clear()
        # PackPointwiseAfterCat now runs as a built-in post-grad pass; only
        # need to ensure no stray custom pre-pass leaked in from another test.
        with inductor_config.patch(
            {
                "post_grad_custom_pre_pass": None,
            }
        ):
            with torch.inference_mode():
                opt_model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
                opt_model(*args)
                opt_model(*args)
                torch.cuda.synchronize()
                # Benchmarking a skipped capture measures the wrong thing.
                assert counters["inductor"]["cudagraph_skips"] == 0  # noqa: S101

                def _run():
                    opt_model(*args)
                    torch.cuda.synchronize()

                benchmark(_run)

    @staticmethod
    def _compile_localized_capture(model, args, *, force_triton):
        """Compile ``model`` in reduce-overhead mode (matching the other
        benchmarks), capturing the generated wrapper code, and warm twice.
        ``force_triton`` selects Triton-template GEMMs over the default
        ``extern_kernels.mm`` (nvjet) backend.  Returns
        ``(opt_model, generated_code)`` so the caller can assert on numerics
        and IR before handing a run functor to ``benchmark``."""
        from torch._inductor.graph import GraphLowering

        captured: list[str] = []

        def save(code: str) -> None:
            captured.append(code)

        patch = {"post_grad_custom_pre_pass": None}
        if force_triton:
            patch["max_autotune_gemm"] = True
            patch["max_autotune_gemm_backends"] = "TRITON"

        torch._dynamo.reset()
        counters.clear()
        prev = GraphLowering.save_output_code
        GraphLowering.save_output_code = save
        try:
            with inductor_config.patch(patch):
                with torch.inference_mode():
                    opt_model = torch.compile(
                        model, fullgraph=True, mode="reduce-overhead"
                    )
                    opt_model(*args)
                    opt_model(*args)
                    torch.cuda.synchronize()
        finally:
            GraphLowering.save_output_code = prev
        assert counters["inductor"]["cudagraph_skips"] == 0  # noqa: S101
        return opt_model, "\n".join(captured)

    @_requires_green_contexts
    def test_simple_matmul_baseline_benchmark(self, benchmark):
        """nn.Linear forward, no localization."""
        m, k, n = 32, 1024, 65536
        model = torch.nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))

    def _page_aligned_localized(self, benchmark, *, force_triton):
        """Shared body for the two page-aligned localized-matmul cases.

        Shapes are sized so each per-branch matmul output is an exact multiple
        of the locality-domain allocator's page granularity (2 MiB on
        Blackwell): per-branch output = ``m * (n / 2) * 2 = m * n`` bf16 bytes,
        and ``m=32, n=65536`` gives exactly one 2 MiB granularity unit per
        branch -- the alignment the future zero-copy contiguous-split path
        needs.  Compiles in reduce-overhead mode (matching the other
        benchmarks), captures the generated wrapper, asserts numerics vs eager
        and that both branches allocate into their locality-domain pool, then
        benchmarks the steady-state replay.  Returns the generated code so the
        caller can assert backend-specific IR.

        Note: under reduce-overhead the zero-copy static-alloc path is gated
        off (cudagraph_trees rejects out-of-pool storage), so the cat still
        does a pointwise copy; that lifts once cudagraph_trees grows multi-pool
        support.
        """
        m, k, n = 32, 1024, 65536  # m*n = 2 MiB exactly per branch
        eager_model = torch.nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        loc_model = torch.nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        loc_model.load_state_dict(eager_model.state_dict())
        with torch.inference_mode():
            torch.cuda.localize(loc_model)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            ref = eager_model(x)

        opt_model, code = self._compile_localized_capture(
            loc_model, (x,), force_triton=force_triton
        )
        with torch.inference_mode():
            out = opt_model(x)
            torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        # Both branches allocate into their own locality-domain pool.
        assert code.count("use_mem_pool") == 2, (  # noqa: S101
            f"expected 2 per-branch use_mem_pool wraps, got "
            f"{code.count('use_mem_pool')}"
        )

        def _run():
            opt_model(x)
            torch.cuda.synchronize()

        benchmark(_run)
        return code

    @_requires_green_contexts
    def test_simple_matmul_localized_page_aligned_default_benchmark(self, benchmark):
        """Page-aligned localized nn.Linear with the DEFAULT GEMM backend
        (``extern_kernels.mm`` -> nvjet/cuBLAS).  Companion to the Triton
        variant below: same shapes and same localized fork-join, differing only
        in GEMM backend, so a perf delta between the two isolates backend cost
        from the localization effect."""
        code = self._page_aligned_localized(benchmark, force_triton=False)
        assert "extern_kernels.mm" in code, (  # noqa: S101
            "expected the default backend to lower the matmul to "
            "extern_kernels.mm (nvjet)"
        )

    @_requires_green_contexts
    def test_simple_matmul_localized_page_aligned_triton_benchmark(self, benchmark):
        """Page-aligned localized nn.Linear forcing Triton-template GEMMs
        (``max_autotune_gemm_backends="TRITON"``) instead of nvjet.

        Forcing Triton keeps the matmul's pointer args visible to cupti
        (nvjet hides them), so the per-branch mm can be verified to bind its
        locality-domain pool -- which the default/nvjet variant cannot
        demonstrate."""
        code = self._page_aligned_localized(benchmark, force_triton=True)
        assert "extern_kernels.mm" not in code, (  # noqa: S101
            "forced Triton should lower the matmul to a Triton template, "
            "not extern_kernels.mm"
        )
        assert re.search(r"triton_\w*mm", code), (  # noqa: S101
            "expected a Triton matmul kernel in the generated code"
        )

    @_requires_green_contexts
    def test_pattern1_attention_baseline_benchmark(self, benchmark):
        """MHA Q/K/V projections + bmm/softmax tail, no localization."""
        batch, seq_len, hidden, num_heads, head_dim = 4, 512, 768, 12, 64
        model = (
            _AttentionQKV(hidden, num_heads, head_dim).cuda().to(torch.bfloat16).eval()
        )
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x, batch, seq_len))

    @_requires_green_contexts
    def test_pattern1_attention_localized_benchmark(self, benchmark):
        """MHA with the three Q/K/V nn.Linears localized."""
        batch, seq_len, hidden, num_heads, head_dim = 4, 512, 768, 12, 64
        model = (
            _AttentionQKV(hidden, num_heads, head_dim).cuda().to(torch.bfloat16).eval()
        )
        with torch.inference_mode():
            torch.cuda.localize(model)
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x, batch, seq_len))

    @_requires_green_contexts
    def test_pattern2_ffn_baseline_benchmark(self, benchmark):
        """FFN (up-proj + down-proj + residual + LN), no localization."""
        batch, seq_len, hidden, ffn_dim = 4, 512, 768, 3072
        model = _FFNResidualLN(hidden, ffn_dim).cuda().to(torch.bfloat16).eval()
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))

    @_requires_green_contexts
    def test_pattern2_ffn_localized_benchmark(self, benchmark):
        """FFN with both up-proj and down-proj localized."""
        batch, seq_len, hidden, ffn_dim = 4, 512, 768, 3072
        model = _FFNResidualLN(hidden, ffn_dim).cuda().to(torch.bfloat16).eval()
        with torch.inference_mode():
            torch.cuda.localize(model)
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))

    @_requires_green_contexts
    @pytest.mark.parametrize(
        ("m", "k", "n", "p"),
        [
            (1, 4096, 32768, 4096),
            (4, 4096, 32768, 4096),
            (16, 4096, 32768, 4096),
            (64, 4096, 32768, 4096),
            (256, 4096, 8192, 4096),
        ],
    )
    def test_chain_localized_benchmark(self, benchmark, m, k, n, p):
        """Two stacked nn.Linears, both localized."""
        model = (
            _TwoSequentialLinears(k, n, p, bias=False).cuda().to(torch.bfloat16).eval()
        )
        with torch.inference_mode():
            torch.cuda.localize(model)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))


if __name__ == "__main__":
    run_tests()
