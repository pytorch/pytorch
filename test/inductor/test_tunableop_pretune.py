# Owner(s): ["module: inductor"]

"""Unit tests for the TunableOp bucket ladder in tunableop.

The ladder decides which GEMMs get pre-tuned and at which sizes. Everything up to
the point of benchmarking is pure logic over tensor sizes, so it is tested here
without a GPU: `tunableop.submit` is patched to record the requests it is handed,
and the assertions are on those requests.

This is deliberately not an end-to-end test. It covers what the ladder decides,
not whether hipBLASLt accepts the reconstructed metas or whether the resulting
signatures match the ones runtime produces -- both need real hardware.
"""

import contextlib
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._inductor import config, ir, select_algorithm as sa, tunableop
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfWindows,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._sympy.value_ranges import ValueRanges


DEVICE = torch.device("cuda")
CPU_DEVICE = torch.device("cpu")
DTYPE = torch.float16


class _FakeSizevars:
    """The three sizevars entry points the ladder uses.

    `hint` is the traced value of the dynamic symbol; `declared` is the range
    ShapeEnv would report for it.
    """

    def __init__(self, sym, hint, declared):
        self._sym = sym
        self._hint = hint
        self.shape_env = mock.Mock()
        self.shape_env.bound_sympy = lambda expr: ValueRanges(*declared)
        self.var_to_hint_override: dict = {}

    # The real predicate over the fake's dict, not a stand-in for it.
    all_unbacked_explicitly_hinted = SizeVarAllocator.all_unbacked_explicitly_hinted

    def optimization_hint(self, expr):
        return int(sympy.sympify(expr).subs(self._sym, self._hint))

    def optimization_hints(self, exprs):
        return [self.optimization_hint(e) for e in exprs]


class _FakeGraph:
    def __init__(self, sizevars):
        self.sizevars = sizevars
        # Empty: no buffer is layout-constrained, so strides come from the node.
        self.buffer_layout_constraints = {}


def _buffer(name, sizes, strides):
    return ir.Buffer(name=name, layout=ir.FixedLayout(DEVICE, DTYPE, sizes, strides))


def _contiguous_strides(sizes):
    strides, acc = [], 1
    for s in reversed(sizes):
        strides.insert(0, acc)
        acc = acc * s if isinstance(s, int) else acc
    return strides


class _FakeWinner(sa.ExternKernelCaller):
    """Stands in for an ExternKernelCaller with the attributes the ladder reads.

    Subclasses rather than duck-types because the ladder gates on isinstance;
    the base __init__ is deliberately not called.
    """

    def __init__(
        self,
        name,
        input_nodes,
        out_sizes,
        choice_name=None,
        out_strides=None,
        kwargs=None,
    ):
        self.name = name
        # Production always has choice.name == name; pass choice_name to
        # drive a variant without building a real ExternKernelChoice.
        self.choice = SimpleNamespace(name=choice_name or name)
        # Set by the base __init__ in production; part of the memo key because
        # _scaled_mm's use_fast_accum reaches the TunableOp signature.
        self.kwargs = kwargs or {}
        self.input_nodes = input_nodes
        self.layout = ir.FixedLayout(
            DEVICE, DTYPE, out_sizes, out_strides or _contiguous_strides(out_sizes)
        )
        self.bmreq = mock.Mock()
        self.bmreq.input_tensor_meta = None
        self.bmreq.output_tensor_meta = None


@contextlib.contextmanager
def _ladder(sym, hint=512, declared=(2, 2**31 - 1), ladder=True):
    """Run the ladder with tuning faked on, collecting the requests it submits."""
    submitted = []

    def _record(job, timeout=None):
        submitted.append(job.bmreq)
        return mock.Mock(status="ok", time=1.0)

    graph = _FakeGraph(_FakeSizevars(sym, hint, declared))
    with contextlib.ExitStack() as stack:
        stack.enter_context(V.set_graph_handler(graph))
        stack.enter_context(
            mock.patch("torch._inductor.tunableop.is_tune_mode", return_value=True)
        )
        stack.enter_context(
            mock.patch("torch._inductor.tunableop.submit", side_effect=_record)
        )
        stack.enter_context(mock.patch("torch._inductor.tunableop.BenchmarkJob", _Job))
        # The winners carry a cuda layout so they reach the ladder, but
        # _cuda_tunableop_get_filename needs a CUDA build: patched for CPU shards.
        stack.enter_context(
            mock.patch(
                "torch._inductor.tunableop._resolve_and_pin_filename_locked",
                return_value="pretune.csv",
            )
        )
        stack.enter_context(config.patch({"cuda.autotune_tunableop_ladder": ladder}))
        yield submitted


class _Job:
    def __init__(self, bmreq, tuning=False, **kwargs):
        self.bmreq = bmreq


def _run(winner, **kwargs):
    """Drive the module-level ladder, returning the submitted bmreqs."""
    sym = kwargs.pop("sym")
    with _ladder(sym, **kwargs) as submitted:
        tunableop.maybe_range_pretune(winner)
    return submitted


@instantiate_parametrized_tests
class TestTunableOpPretune(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        # The memo is process-global: without this, tests sharing a traced
        # shape (most use _mm_winner at hint 512) pollute each other by order.
        tunableop._pretuned_gemms.cache_clear()

    def tearDown(self):
        tunableop._pretuned_gemms.cache_clear()
        super().tearDown()

    def _mm_winner(self, name, s):
        """A GEMM whose M is the dynamic dim.

        Operand lists match what each op is really lowered with: bias-first for
        addmm/baddbmm, and [mat_a, mat_b, scale_a, scale_b] for _scaled_mm --
        where scale_a carries the symbol too, so it goes through the meta
        rebuild and the density check like the matrices do.
        """
        m, k, n = s, 256, 512
        mat1 = _buffer("mat1", [m, k], [k, 1])
        mat2 = _buffer("mat2", [k, n], [n, 1])
        if name in ("addmm", "baddbmm"):
            bias = _buffer("bias", [m, n], [n, 1])
            return _FakeWinner(name, [bias, mat1, mat2], [m, n])
        if name.startswith("_scaled_mm"):
            scale_a = _buffer("scale_a", [m, 1], [1, 1])
            # Rowwise scale_b is w_scale.t(): [1, N] strided [1, 1].
            scale_b = _buffer("scale_b", [1, n], [1, 1])
            return _FakeWinner(name, [mat1, mat2, scale_a, scale_b], [m, n])
        return _FakeWinner(name, [mat1, mat2], [m, n])

    @parametrize("name", ("mm", "addmm", "bmm", "baddbmm", "_scaled_mm"))
    def test_supported_ops_produce_buckets(self, name):
        s = sympy.Symbol(f"s_{name}", positive=True, integer=True)
        submitted = _run(self._mm_winner(name, s), sym=s)
        self.assertTrue(submitted, f"{name} produced no buckets")
        if name == "_scaled_mm":
            metas = submitted[0].input_tensor_meta
            self.assertEqual(len(metas), 4, "scales must reach the worker")
            # scale_a carries the symbol, so it is rebuilt with the matrices
            self.assertEqual(metas[2].sizes[0], metas[0].sizes[0])

    def test_unsupported_op_is_skipped(self):
        s = sympy.Symbol("s_other", positive=True, integer=True)
        # Control first, so an empty result means the op gate, not the harness.
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s))
        self.assertEqual(_run(self._mm_winner("convolution", s), sym=s), [])

    @parametrize("name", ("addmm", "baddbmm"))
    def test_bias_first_ops_read_the_right_operands(self, name):
        """Bias-first ops take [bias, mat1, mat2]; reading nodes[0] as A would
        bucket the wrong dim. The bias carries the symbol too, so the tell is
        the submitted SHAPE, not whether anything was submitted."""
        s = sympy.Symbol(f"s_{name}", positive=True, integer=True)
        submitted = _run(self._mm_winner(name, s), sym=s)
        self.assertTrue(submitted)
        for bmreq in submitted:
            bias, mat1, mat2 = bmreq.input_tensor_meta
            self.assertEqual(mat2.sizes, [256, 512])
            self.assertEqual(mat1.sizes[1], 256)
            self.assertEqual(bias.sizes[0], mat1.sizes[0])

    def test_dynamic_reduction_dim_is_rejected(self):
        """A solution is only valid for the K it was tuned at."""
        s = sympy.Symbol("s_k", positive=True, integer=True)
        mat1 = _buffer("mat1", [128, s], [s, 1])
        mat2 = _buffer("mat2", [s, 512], [512, 1])
        winner = _FakeWinner("mm", [mat1, mat2], [128, 512])
        self.assertEqual(_run(winner, sym=s), [])

    def test_buckets_scale_only_the_dynamic_dim(self):
        s = sympy.Symbol("s_m", positive=True, integer=True)
        submitted = _run(self._mm_winner("mm", s), sym=s)
        seen = set()
        for bmreq in submitted:
            mat1, mat2 = bmreq.input_tensor_meta
            self.assertEqual(mat2.sizes, [256, 512], "static operand must not move")
            self.assertEqual(mat1.sizes[1], 256, "K must not move")
            seen.add(mat1.sizes[0])
        self.assertGreater(len(seen), 1, "ladder produced a single size")

    def test_expression_dim_only_takes_reachable_values(self):
        """For a dim of 200*s, powers of two are not values the dim can take."""
        s = sympy.Symbol("s_expr", positive=True, integer=True)
        m = 200 * s
        mat1 = _buffer("mat1", [m, 256], [256, 1])
        mat2 = _buffer("mat2", [256, 512], [512, 1])
        winner = _FakeWinner("mm", [mat1, mat2], [m, 512])
        submitted = _run(winner, sym=s)
        self.assertTrue(submitted)
        for bmreq in submitted:
            dim = bmreq.input_tensor_meta[0].sizes[0]
            self.assertEqual(dim % 200, 0, f"{dim} is not a multiple of 200")

    def test_stride_order_is_preserved(self):
        """A column-major operand must stay column-major: reordering strides turns
        a plain GEMM into a transposed one, which is a different solution."""
        s = sympy.Symbol("s_t", positive=True, integer=True)
        mat1 = _buffer("mat1", [s, 256], [256, 1])
        mat2 = _buffer("mat2", [256, 512], [1, 256])  # .t() view
        winner = _FakeWinner("mm", [mat1, mat2], [s, 512])
        submitted = _run(winner, sym=s)
        self.assertTrue(submitted)
        for bmreq in submitted:
            _, b = bmreq.input_tensor_meta
            self.assertLess(b.strides[0], b.strides[1], "mat2 lost its column order")

    def test_ladder_disabled_produces_nothing(self):
        s = sympy.Symbol("s_off", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        with _ladder(s, ladder=False) as submitted:
            tunableop.maybe_range_pretune(winner)
        self.assertEqual(submitted, [])

    def _bmm_winner(self, name, s):
        """A batched GEMM with real 3-D operands whose batch dim is dynamic."""
        b, m, k, n = s, 32, 256, 512
        mat1 = _buffer("mat1", [b, m, k], _contiguous_strides([b, m, k]))
        mat2 = _buffer("mat2", [b, k, n], _contiguous_strides([b, k, n]))
        if name == "baddbmm":
            bias = _buffer("bias", [b, m, n], _contiguous_strides([b, m, n]))
            return _FakeWinner(name, [bias, mat1, mat2], [b, m, n])
        return _FakeWinner(name, [mat1, mat2], [b, m, n])

    @parametrize("name", ("bmm", "baddbmm"))
    def test_batched_ops_reconstruct_batched_metadata(self, name):
        # Every operand keeps its batch axis, and they agree on it.
        s = sympy.Symbol(f"b_{name}", positive=True, integer=True)
        submitted = _run(self._bmm_winner(name, s), sym=s)
        self.assertTrue(submitted, f"{name} produced no buckets")
        for bmreq in submitted:
            metas = [*bmreq.input_tensor_meta, bmreq.output_tensor_meta]
            for meta in metas:
                self.assertEqual(len(meta.sizes), 3, "operand lost its batch axis")
            batches = {meta.sizes[0] for meta in metas}
            self.assertEqual(len(batches), 1, f"operands disagree on batch {batches}")

    def test_symbol_spanning_two_axes_is_rejected(self):
        # M = 1000*s and N = s share one symbol: capping N at 16384 would let
        # M reach 16,384,000. The single-axis control isolates the cause.
        s = sympy.Symbol("s_shared", positive=True, integer=True)
        k = 256
        shared = _FakeWinner(
            "mm",
            [_buffer("mat1", [1000 * s, k], [k, 1]), _buffer("mat2", [k, s], [s, 1])],
            [1000 * s, s],
        )
        # _contiguous_strides would yield [1, 1] here, masking the guard.
        shared.layout = ir.FixedLayout(DEVICE, DTYPE, [1000 * s, s], [s, 1])
        self.assertEqual(_run(shared, sym=s), [])
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s))

    def test_non_dense_operand_is_skipped(self):
        # [s, 256] strided [300, 1] is padded, and _meta_at rebuilds densely:
        # it would be tuned at [256, 1], an ld runtime never asks for.
        s = sympy.Symbol("s_pad", positive=True, integer=True)
        k, n = 256, 512
        padded = _FakeWinner(
            "mm",
            [_buffer("mat1", [s, k], [300, 1]), _buffer("mat2", [k, n], [n, 1])],
            [s, n],
        )
        self.assertEqual(_run(padded, sym=s), [])

    def test_expanded_bias_still_produces_buckets(self):
        # A broadcast bias arrives stride-0 and is exempt from the density
        # check; mat1/mat2/output are not.
        m, k, n = 32, 256, 512

        def _baddbmm(sym, mat1_strides):
            # bias is [N] broadcast to [B, M, N]: zero stride on both leading
            # dims, which is how mm_args hands it over.
            return _FakeWinner(
                "baddbmm",
                [
                    _buffer("bias", [sym, m, n], [0, 0, 1]),
                    _buffer("mat1", [sym, m, k], mat1_strides),
                    _buffer("mat2", [sym, k, n], _contiguous_strides([sym, k, n])),
                ],
                [sym, m, n],
            )

        s = sympy.Symbol("s_expbias", positive=True, integer=True)
        submitted = _run(_baddbmm(s, _contiguous_strides([s, m, k])), sym=s)
        self.assertTrue(submitted, "an expanded bias must not block the ladder")
        for bmreq in submitted:
            meta = bmreq.input_tensor_meta[0]
            self.assertEqual(meta.strides, _contiguous_strides(meta.sizes))
        # A non-dense mat1 is still rejected: that ld is in the signature.
        tunableop._pretuned_gemms.cache_clear()
        s2 = sympy.Symbol("s_expbias2", positive=True, integer=True)
        self.assertEqual(_run(_baddbmm(s2, [m * 300, 300, 1]), sym=s2), [])

    @parametrize("case", ("output_stride", "stride_order", "dynamic_axis"))
    def test_memo_distinguishes_layout(self, case):
        # Each variant has a different TunableOp signature from the plain mm it
        # follows, so it must ladder rather than hit that memo entry. Output
        # strides are explicit: _contiguous_strides cannot build symbolic N.
        s = sympy.Symbol(f"s_memo_{case}", positive=True, integer=True)
        mat1 = _buffer("mat1", [s, 256], [256, 1])
        variants = {
            # column-major output: a different ldc
            "output_stride": (
                [mat1, _buffer("mat2", [256, 512], [512, 1])],
                [s, 512],
                [1, s],
            ),
            # mat2 as a .t() view: NN vs NT
            "stride_order": (
                [mat1, _buffer("mat2", [256, 512], [1, 256])],
                [s, 512],
                [512, 1],
            ),
            # dynamic N instead of M: a different ladder
            "dynamic_axis": (
                [
                    _buffer("mat1", [512, 256], [256, 1]),
                    _buffer("mat2", [256, s], [s, 1]),
                ],
                [512, s],
                [s, 1],
            ),
        }
        inputs, out_sizes, out_strides = variants[case]
        second = _FakeWinner("mm", inputs, out_sizes, out_strides=out_strides)
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s))
        self.assertTrue(_run(second, sym=s), f"{case} must not share a memo key")

    def test_unbacked_symbol_is_skipped(self):
        # optimization_hint() invents a fallback rather than raising. Unbacked
        # ints are the "u" prefix (torch/utils/_sympy/symbol.py).
        u = sympy.Symbol("u0", positive=True, integer=True)
        self.assertEqual(_run(self._mm_winner("mm", u), sym=u), [])

    def test_unbacked_symbol_with_a_user_hint_is_tuned(self):
        # An explicit hint via var_to_hint_override is a real size, so the
        # ladder runs rather than being turned away.
        u = sympy.Symbol("u1", positive=True, integer=True)
        winner = self._mm_winner("mm", u)
        with _ladder(u) as submitted:
            V.graph.sizevars.var_to_hint_override = {u: 512}
            tunableop.maybe_range_pretune(winner)
        self.assertTrue(submitted, "an explicit hint must re-enable the ladder")

    def test_a_persistently_faulting_bucket_stops_being_retried(self):
        # A rung that keeps faulting is attempted a bounded number of times
        # across the process, not once per identical layer.
        s = sympy.Symbol("s_stuck", positive=True, integer=True)
        attempts = []

        def _always_timeout(job, timeout=None):
            attempts.append(job.bmreq)
            return mock.Mock(status="timeout", time=None)

        def _one_pass():
            with (
                _ladder(s),
                mock.patch(
                    "torch._inductor.tunableop.submit", side_effect=_always_timeout
                ),
            ):
                tunableop.maybe_range_pretune(self._mm_winner("mm", s))

        _one_pass()
        per_pass = len(attempts)
        self.assertTrue(per_pass, "the first pass must try every bucket")
        for _ in range(4):
            _one_pass()
        self.assertEqual(
            len(attempts),
            per_pass * tunableop._TUNABLEOP_MAX_BUCKET_ATTEMPTS,
            "a faulting bucket must stop after a bounded number of attempts",
        )

    def test_memo_does_not_depend_on_an_unpinned_filename(self):
        # get_filename() reads "" until pinned; keying on it directly would
        # scope the first GEMM apart from every later one. The real resolver
        # runs over a faked tunable, so the pin happens between the two calls.
        s = sympy.Symbol("s_pin", positive=True, integer=True)
        real = tunableop._resolve_and_pin_filename_locked
        pinned = [""]

        def _set_filename(name, insert_device_ordinal=False):
            pinned[0] = name

        with _ladder(s) as submitted:
            with (
                mock.patch.multiple(
                    "torch.cuda.tunable",
                    get_filename=lambda: pinned[0],
                    set_filename=_set_filename,
                ),
                mock.patch.object(
                    tunableop, "_resolve_and_pin_filename_locked", lambda _: real(0)
                ),
            ):
                tunableop.maybe_range_pretune(self._mm_winner("mm", s))
                first = len(submitted)
                self.assertTrue(pinned[0], "the resolver must pin the empty name")
                tunableop.maybe_range_pretune(self._mm_winner("mm", s))
        self.assertEqual(len(submitted), first, "pinning must not rekey")

    def test_memo_distinguishes_extern_kwargs(self):
        # use_fast_accum is in the signature but not the metas.
        s = sympy.Symbol("s_kwargs", positive=True, integer=True)
        m, k, n = s, 256, 512
        nodes = [_buffer("mat1", [m, k], [k, 1]), _buffer("mat2", [k, n], [n, 1])]
        slow = _FakeWinner(
            "_scaled_mm", nodes, [m, n], kwargs={"use_fast_accum": False}
        )
        fast = _FakeWinner("_scaled_mm", nodes, [m, n], kwargs={"use_fast_accum": True})
        with _ladder(s) as submitted:
            tunableop.maybe_range_pretune(slow)
            first = len(submitted)
            self.assertTrue(first)
            tunableop.maybe_range_pretune(fast)
            self.assertGreater(
                len(submitted), first, "use_fast_accum must not share a memo key"
            )

    @parametrize("choice_name", ("mm_dtype", "bmm_dtype", "_scaled_mm_v2"))
    def test_choice_variants_produce_buckets(self, choice_name):
        # Matching runs on winner.choice.name, covering the _dtype out-variants.
        s = sympy.Symbol(f"s_{choice_name}", positive=True, integer=True)
        submitted = _run(self._mm_winner(choice_name, s), sym=s)
        self.assertTrue(submitted, f"{choice_name} produced no buckets")

    @unittest.skipIf(torch.version.hip is None, "addmm_dtype is tunable on ROCm only")
    def test_addmm_dtype_produces_buckets(self):
        s = sympy.Symbol("s_addmm_dtype", positive=True, integer=True)
        submitted = _run(self._mm_winner("addmm_dtype", s), sym=s)
        self.assertTrue(submitted, "addmm_dtype produced no buckets")

    def test_traced_hint_is_always_tuned(self):
        # The traced shape is the size actually served: it must be a rung even
        # when it is not a power of two and neither endpoint.
        s = sympy.Symbol("s_hint", positive=True, integer=True)
        submitted = _run(self._mm_winner("mm", s), sym=s, hint=3000)
        dims = set()
        for bmreq in submitted:
            dims.add(bmreq.input_tensor_meta[0].sizes[0])
        self.assertIn(3000, dims, f"traced hint missing from {sorted(dims)}")

    def test_a_small_traced_hint_is_still_tuned(self):
        # A model served below _TUNABLEOP_MIN_DYN_DIM still needs that shape:
        # dropping it guarantees a miss on the only size runtime asks for.
        s = sympy.Symbol("s_small", positive=True, integer=True)
        submitted = _run(self._mm_winner("mm", s), sym=s, hint=4)
        dims = {b.input_tensor_meta[0].sizes[0] for b in submitted}
        self.assertIn(4, dims, f"the served dim was dropped: {sorted(dims)}")

    def test_traced_hint_first_then_largest_first(self):
        # The budget is shared and spent in order, so the size runtime is known
        # to ask for goes first; the rest descend (a big fault costs a restart).
        s = sympy.Symbol("s_order", positive=True, integer=True)
        submitted = _run(self._mm_winner("mm", s), sym=s, hint=512)
        dims = [bmreq.input_tensor_meta[0].sizes[0] for bmreq in submitted]
        self.assertEqual(dims[0], 512, f"traced hint must run first: {dims}")
        self.assertEqual(dims[1:], sorted(dims[1:], reverse=True))

    def test_rung_order_is_keyed_on_the_dim_not_the_symbol(self):
        # floor(s/2) is not injective: s=512 and the traced s=513 both give a
        # dim of 256, so dedup keeps 512 and no bucket equals the hint. The
        # served dim must still lead, or the budget drops it first.
        s = sympy.Symbol("s_noninj", positive=True, integer=True)
        m, k, n = sympy.floor(s / 2), 256, 512
        winner = _FakeWinner(
            "mm",
            [_buffer("mat1", [m, k], [k, 1]), _buffer("mat2", [k, n], [n, 1])],
            [m, n],
        )
        submitted = _run(winner, sym=s, hint=513, declared=(2, 1024))
        dims = [b.input_tensor_meta[0].sizes[0] for b in submitted]
        self.assertEqual(dims[0], 256, f"the served dim must run first: {dims}")

    def test_deadline_skips_remaining_buckets(self):
        s = sympy.Symbol("s_budget", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        timeouts: list = []

        def _fake_submit(job, timeout=None):
            timeouts.append(timeout)
            return mock.Mock(status="ok", time=1.0)

        def _fake_monotonic():
            # 0 until the first submission, then a sliver short of the
            # deadline: too little for a fair trial, so the loop must stop.
            budget = config.cuda.autotune_tunableop_timeout
            return 0.0 if not timeouts else budget - 1

        with _ladder(s):
            with (
                mock.patch(
                    "torch._inductor.tunableop.submit", side_effect=_fake_submit
                ),
                # The module, not time.monotonic, which is process-wide.
                mock.patch(
                    "torch._inductor.tunableop.time",
                    SimpleNamespace(monotonic=_fake_monotonic),
                ),
                # A full budget really spent: coverage was cut, so warn.
                self.assertLogs(tunableop.log, level="WARNING"),
            ):
                tunableop.maybe_range_pretune(winner)
        # One bucket ran; the rest had too little budget to be anything but
        # a timeout, so they were left rather than charged.
        self.assertEqual(len(timeouts), 1)
        self.assertGreaterEqual(timeouts[0], tunableop._TUNABLEOP_MIN_BUCKET_BUDGET)
        self.assertFalse(tunableop._pretuned_gemms.failed, "healthy rung charged")

    def test_a_short_timeout_still_tunes_the_traced_shape(self):
        # The per-bucket floor must not turn the ladder off wholesale when the
        # configured budget is smaller than it.
        s = sympy.Symbol("s_shortbudget", positive=True, integer=True)
        with config.patch({"cuda.autotune_tunableop_timeout": 30.0}):
            # Documented behaviour, once per GEMM: it must not warn.
            with self.assertNoLogs(tunableop.log, level="WARNING"):
                submitted = _run(self._mm_winner("mm", s), sym=s)
        dims = {b.input_tensor_meta[0].sizes[0] for b in submitted}
        self.assertEqual(dims, {512}, "only the traced shape fits a short budget")

    def test_a_short_budget_is_spent_on_the_served_rung_only(self):
        # Once the served rung is memoed it leaves the ladder, and its floor
        # exemption must not fall through to the largest rung behind it.
        s = sympy.Symbol("s_shortwiden", positive=True, integer=True)
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s, declared=(2, 4096)))
        with config.patch({"cuda.autotune_tunableop_timeout": 30.0}):
            wide = _run(self._mm_winner("mm", s), sym=s)
        self.assertEqual(wide, [], "no invented rung may claim the exemption")

    def test_non_ok_outcome_continues(self):
        s = sympy.Symbol("s_cont", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        statuses = ["timeout"]

        def _flaky_submit(job, timeout=None):
            submitted.append(job.bmreq)
            status = statuses.pop(0) if statuses else "ok"
            return mock.Mock(status=status, time=1.0)

        submitted = []
        with _ladder(s) as _:
            with mock.patch(
                "torch._inductor.tunableop.submit", side_effect=_flaky_submit
            ):
                tunableop.maybe_range_pretune(winner)
        # First bucket faulted, the rest still submitted.
        self.assertGreater(len(submitted), 1)

    def test_repeat_gemm_is_skipped(self):
        s = sympy.Symbol("s_memo", positive=True, integer=True)
        first = _run(self._mm_winner("mm", s), sym=s)
        self.assertTrue(first)
        second = _run(self._mm_winner("mm", s), sym=s)
        self.assertEqual(second, [])

    def test_flexible_operand_is_skipped(self):
        # Lowering-time hooks run before finalize(), so an operand that is
        # still flexible would be tuned at strides it is about to lose.
        s = sympy.Symbol("s_flex", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        winner.input_nodes[1].layout = ir.FlexibleLayout(DEVICE, DTYPE, [256, 512])
        self.assertEqual(_run(winner, sym=s), [])

    def test_cpu_winner_is_skipped(self):
        # The worker is a CUDA process: a CPU winner's bmreq would spend the
        # whole per-GEMM budget there and record nothing.
        s = sympy.Symbol("s_cpu", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        winner.layout = ir.FixedLayout(
            CPU_DEVICE, DTYPE, [s, 512], _contiguous_strides([s, 512])
        )
        self.assertEqual(_run(winner, sym=s), [])

    def test_a_failed_run_is_retried(self):
        # The memo key is recorded only once something was tuned, so a GEMM
        # whose buckets all faulted is retried rather than skipped for good.
        s = sympy.Symbol("s_retry", positive=True, integer=True)
        with _ladder(s) as submitted:
            with mock.patch(
                "torch._inductor.tunableop.submit",
                side_effect=lambda job, timeout=None: mock.Mock(
                    status="error", time=None
                ),
            ):
                tunableop.maybe_range_pretune(self._mm_winner("mm", s))
            self.assertEqual(submitted, [])
            tunableop.maybe_range_pretune(self._mm_winner("mm", s))
            self.assertTrue(submitted, "a failed GEMM must stay retryable")

    def test_a_narrower_range_reuses_tuned_buckets(self):
        # The narrow ladder's rungs are a subset of the wide one's, so every
        # one of them is already tuned.
        s = sympy.Symbol("s_range", positive=True, integer=True)
        wide = _run(self._mm_winner("mm", s), sym=s, declared=(2, 2**31 - 1))
        self.assertTrue(wide)
        narrow = _run(self._mm_winner("mm", s), sym=s, declared=(2, 4096))
        self.assertEqual(narrow, [], "already-tuned rungs must not re-run")

    def test_a_wider_range_tunes_only_the_new_buckets(self):
        s = sympy.Symbol("s_widen", positive=True, integer=True)
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s, declared=(2, 4096)))
        wide = _run(self._mm_winner("mm", s), sym=s, declared=(2, 2**31 - 1))
        dims = {b.input_tensor_meta[0].sizes[0] for b in wide}
        self.assertTrue(dims, "the rungs above the old ceiling are new")
        self.assertTrue(all(d > 4096 for d in dims), f"re-tuned old rungs: {dims}")

    def test_a_partial_run_retries_the_untuned_buckets(self):
        # The budget is shared, so a run can tune a few rungs and stop. Only
        # the rungs that returned ok are skipped next time.
        s = sympy.Symbol("s_partial", positive=True, integer=True)
        calls = []

        def _one_then_fail(job, timeout=None):
            calls.append(job.bmreq)
            return mock.Mock(status="ok" if len(calls) == 1 else "timeout", time=1.0)

        with _ladder(s):
            with mock.patch(
                "torch._inductor.tunableop.submit", side_effect=_one_then_fail
            ):
                tunableop.maybe_range_pretune(self._mm_winner("mm", s))
        tuned_dim = calls[0].input_tensor_meta[0].sizes[0]

        second = _run(self._mm_winner("mm", s), sym=s)
        self.assertTrue(second, "the untuned rungs must be retried")
        retried = {b.input_tensor_meta[0].sizes[0] for b in second}
        self.assertNotIn(tuned_dim, retried, "the tuned rung must be skipped")

    def test_bias_addmm_is_skipped(self):
        # bias_addmm rewrites its own bias before dispatching, so it is absent
        # from _TUNABLEOP_GEMM_CHOICES and never reaches the density guard.
        s = sympy.Symbol("s_biasaddmm", positive=True, integer=True)
        m, k, n = s, 256, 512
        winner = _FakeWinner(
            "bias_addmm",
            [
                _buffer("bias", [m, n], [n, 1]),
                _buffer("mat1", [m, k], [k, 1]),
                _buffer("mat2", [k, n], [n, 1]),
            ],
            [m, n],
            choice_name="bias_addmm",
        )
        self.assertEqual(_run(winner, sym=s), [])

    def test_oversize_bucket_is_dropped(self):
        # One batch of 4096x4096 fp16 plus operands is ~32.5 MiB, so the cap
        # admits s up to 252. Traced at 512: that rung is oversize but served,
        # so it is exempt, while the invented rungs above the cap are dropped.
        s = sympy.Symbol("s_oom", positive=True, integer=True)
        m, k, n = 32, 4096, 4096
        mat1 = _buffer("mat1", [s, m, k], _contiguous_strides([s, m, k]))
        mat2 = _buffer("mat2", [s, k, n], _contiguous_strides([s, k, n]))
        winner = _FakeWinner("bmm", [mat1, mat2], [s, m, n])
        submitted = _run(winner, sym=s, hint=512, declared=(2, 16384))
        batches = set()
        for bmreq in submitted:
            # Summed, as the cap is: the worker allocates them all at once.
            total = sum(
                math.prod(int(d) for d in meta.sizes) * meta.dtype.itemsize
                for meta in [*bmreq.input_tensor_meta, bmreq.output_tensor_meta]
            )
            batch = bmreq.input_tensor_meta[0].sizes[0]
            batches.add(batch)
            if batch != 512:
                self.assertLessEqual(total, tunableop._TUNABLEOP_MAX_LADDER_BYTES)
        self.assertIn(512, batches, "the served rung is exempt from the cap")
        self.assertNotIn(16384, batches, "an invented oversize rung must be absent")
        self.assertTrue(
            batches - {512}, "rungs under the cap must still submit alongside it"
        )

    @parametrize("path", ("single_choice", "deterministic"))
    def test_algorithm_selector_hook_fires_on_early_return(self, path):
        # Driven through __call__, not by calling the hook: both early returns
        # skip benchmarking, and a direct call passes with either line deleted.
        from torch._inductor import select_algorithm as _sa

        s = sympy.Symbol(f"s_sel_{path}", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        winner.output_node = mock.Mock(return_value=mock.Mock())
        choices = [winner]
        if path == "deterministic":
            choices.append(self._mm_winner("mm", s))
        with config.patch({"deterministic": path == "deterministic"}):
            with mock.patch("torch._inductor.tunableop.maybe_range_pretune") as pretune:
                _, chosen = _sa.AlgorithmSelectorCache()(
                    "mm", choices, winner.input_nodes, winner.layout
                )
                # Triton winners are turned away by the isinstance gate.
                _sa._maybe_pretune_tunableop(mock.Mock(spec=[]))
        self.assertIs(chosen, winner)
        pretune.assert_called_once_with(winner)

    @parametrize("scenario", ("clean", "conflict", "force_extern"))
    def test_finalize_hook_routes_extern_winners(self, scenario):
        # The finalizer ladders the extern winner except on the layout-conflict
        # fallback; force-extern is not that case, so the ladder must still run.
        import torch._inductor.ir as _ir
        from torch._inductor.scheduler import Scheduler, SchedulerNode

        conflict = scenario == "conflict"
        s = sympy.Symbol(f"s_fin_{scenario}", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        multi = mock.Mock(spec=_ir.MultiTemplateBuffer)
        multi.choices = [winner]
        multi.inputs = winner.input_nodes
        multi.get_min_choice.return_value = (winner, None)
        multi.choice_timings.return_value = {winner: 1.0}
        multi.origins = set()
        multi.layout = winner.layout
        multi.origin_node = None
        sched_node = mock.Mock(spec=SchedulerNode)
        sched_node.node = multi
        sched = mock.Mock(spec=Scheduler)
        sched.nodes = [sched_node]
        sched._has_layout_conflict_for_template.return_value = conflict

        # Spec'd so the isinstance checks pass and both branches run to the
        # end (_replace_node is a no-op on the mocked scheduler).
        out_buffer = mock.Mock(spec=_ir.OperationBuffer)
        out_storage = mock.Mock(spec=_ir.StorageBox, data=out_buffer)
        winner.output_node = mock.Mock(
            return_value=mock.Mock(spec=_ir.TensorBox, data=out_storage)
        )
        with config.patch(
            {
                "test_configs.force_extern_kernel_in_multi_template": scenario
                == "force_extern"
            }
        ):
            with mock.patch(
                "torch._inductor.select_algorithm._maybe_pretune_tunableop"
            ) as pretune:
                Scheduler.finalize_multi_template_buffers(sched)
        if conflict:
            pretune.assert_not_called()
        else:
            pretune.assert_called_once_with(winner)
        if scenario == "force_extern":
            # It freezes flexible input layouts; the flag must short out first.
            sched._has_layout_conflict_for_template.assert_not_called()
        winner.output_node.assert_called_once()


class TunableOpPretuneRealGpuTest(InductorTestCase):
    """The ladder against a real compile, worker and results file.

    The only test that sends a real ExternKernelGPUBenchmarkRequest through
    the pipe, so it is what proves the rebuilt metas describe shapes
    hipBLASLt accepts and that each reaches the file runtime reads.

    InductorTestCase enters fresh_cache(), isolating the cache dir and
    clearing _pretuned_gemms. The compile therefore always starts cold, so
    this does not cover a warm cache, where the FX graph is served from the
    cache and no autotuning -- hence no ladder -- runs at all.
    """

    hw_classification = HardwareClassification.CUDA

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_dynamic_mm_records_every_bucket(self):
        def fn(a, b):
            return a @ b

        tunable = torch.cuda.tunable
        # All process-global: leaving one behind makes a later test tune at
        # a single iteration. Cleanups run LIFO, so the worker goes first.
        self.addCleanup(tunable.set_filename, tunable.get_filename(), False)
        for getter, setter in (
            (tunable.is_enabled, tunable.enable),
            (tunable.get_max_tuning_duration, tunable.set_max_tuning_duration),
            (tunable.get_max_tuning_iterations, tunable.set_max_tuning_iterations),
            (tunable.get_rotating_buffer_size, tunable.set_rotating_buffer_size),
        ):
            self.addCleanup(setter, getter())
        self.addCleanup(tunableop.shutdown_worker)

        with tempfile.TemporaryDirectory(prefix="tunableop_pretune_gpu_") as d:
            results = os.path.join(d, "results.csv")
            tunable.enable(False)
            tunable.set_filename(results, False)
            # Cost is per candidate solution, so the caps keep this short.
            tunable.set_max_tuning_duration(1)
            tunable.set_max_tuning_iterations(1)
            tunable.set_rotating_buffer_size(0)
            with config.patch(
                {
                    "cuda.autotune_tunableop_mode": "tune",
                    "cuda.autotune_tunableop_ladder": True,
                }
            ):
                a = torch.randn(64, 64, device="cuda", dtype=DTYPE)
                b = torch.randn(64, 64, device="cuda", dtype=DTYPE)
                # One dynamic dim: the ladder needs exactly one free symbol.
                torch._dynamo.mark_dynamic(a, 0)
                torch.compile(fn)(a, b)
            with open(results) as f:
                content = f.read()

        # Rows are op,params,solution,time; params is ab_M_N_K_ld_... in the
        # BLAS frame, which transposes the operands, so compare the triple.
        shapes = set()
        for line in content.splitlines():
            parts = line.split(",")
            if len(parts) < 2 or parts[0] == "Validator":
                continue
            tokens = parts[1].split("_")
            if len(tokens) > 3:
                shapes.add(tuple(tokens[1:4]))

        # Fully determined: the symbol walks {2, hint 64, ceiling 16384} plus
        # the powers of two between, and the minimum drops 2 and 4.
        expected_dims = {1 << k for k in range(3, 15)}  # 8 .. 16384
        self.assertEqual(
            len(shapes),
            len(expected_dims),
            f"expected one row per ladder bucket, got {sorted(shapes)}",
        )
        # The varying position is the dynamic dim. Asserting the set catches
        # a bucket that ran but recorded the wrong size; a count would not.
        varying = {
            frozenset(int(t[i]) for t in shapes)
            for i in range(3)
            if len({t[i] for t in shapes}) > 1
        }
        self.assertEqual(
            varying,
            {frozenset(expected_dims)},
            f"recorded dims do not match the ladder: {sorted(shapes)}",
        )


if __name__ == "__main__":
    run_tests()
