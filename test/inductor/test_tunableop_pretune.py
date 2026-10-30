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
from unittest import mock

import sympy

import torch
from torch._inductor import config, ir, select_algorithm as sa, tunableop
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
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

    def __init__(self, name, input_nodes, out_sizes, choice_name=None):
        from types import SimpleNamespace

        self.name = name
        # The ladder matches on winner.choice.name (the ExternKernelChoice,
        # always set by the base __init__ in production). Pass choice_name to
        # exercise the _dtype / v2 / bias variants.
        self.choice = SimpleNamespace(name=choice_name or name)
        self.input_nodes = input_nodes
        self.layout = ir.FixedLayout(
            DEVICE, DTYPE, out_sizes, _contiguous_strides(out_sizes)
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
        stack.enter_context(config.patch({"cuda.aten_tunableop_ladder": ladder}))
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


class TestTunableOpPretune(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        # The memo is process-global: without this, tests sharing the same
        # traced shapes (most build _mm_winner("mm", s) at hint 512) pollute
        # each other depending on run order.
        tunableop._pretuned_gemms.cache_clear()

    def tearDown(self):
        tunableop._pretuned_gemms.cache_clear()
        super().tearDown()

    def _mm_winner(self, name, s):
        """A GEMM whose M is the dynamic dim. Bias-first ops get a third operand."""
        m, k, n = s, 256, 512
        mat1 = _buffer("mat1", [m, k], [k, 1])
        mat2 = _buffer("mat2", [k, n], [n, 1])
        if name in ("addmm", "baddbmm"):
            bias = _buffer("bias", [m, n], [n, 1])
            return _FakeWinner(name, [bias, mat1, mat2], [m, n])
        return _FakeWinner(name, [mat1, mat2], [m, n])

    def test_supported_ops_produce_buckets(self):
        for name in ("mm", "addmm", "bmm", "baddbmm", "_scaled_mm"):
            s = sympy.Symbol(f"s_{name}", positive=True, integer=True)
            submitted = _run(self._mm_winner(name, s), sym=s)
            self.assertTrue(submitted, f"{name} produced no buckets")

    def test_unsupported_op_is_skipped(self):
        s = sympy.Symbol("s_other", positive=True, integer=True)
        # Control first: an identical winner under a supported name DOES bucket,
        # so an empty result below means the op gate, not a broken harness.
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s))
        self.assertEqual(_run(self._mm_winner("convolution", s), sym=s), [])

    def test_bias_first_ops_read_the_right_operands(self):
        """addmm and baddbmm take [bias, mat1, mat2]; reading nodes[0] as A would
        make the bias the left matrix and bucket the wrong dim."""
        for name in ("addmm", "baddbmm"):
            s = sympy.Symbol(f"s_{name}", positive=True, integer=True)
            # Bias carries the dynamic dim too, so a wrong operand pick still
            # yields buckets -- the tell is the SHAPE that gets submitted.
            submitted = _run(self._mm_winner(name, s), sym=s)
            self.assertTrue(submitted)
            for bmreq in submitted:
                bias, mat1, mat2 = bmreq.input_tensor_meta
                # mat2 is static; mat1 and bias scale on the dynamic dim.
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

    def test_batched_ops_reconstruct_batched_metadata(self):
        # _mm_winner hands bmm 2-D operands, so a reconstruction that dropped or
        # mangled the batch axis would still pass the bucket-count assertions.
        for name in ("bmm", "baddbmm"):
            s = sympy.Symbol(f"b_{name}", positive=True, integer=True)
            submitted = _run(self._bmm_winner(name, s), sym=s)
            self.assertTrue(submitted, f"{name} produced no buckets")
            for bmreq in submitted:
                metas = [*bmreq.input_tensor_meta, bmreq.output_tensor_meta]
                for meta in metas:
                    self.assertEqual(
                        len(meta.sizes), 3, f"{name}: operand lost its batch axis"
                    )
                batches = {meta.sizes[0] for meta in metas}
                self.assertEqual(
                    len(batches), 1, f"{name}: operands disagree on batch {batches}"
                )

    def test_symbol_spanning_two_axes_is_rejected(self):
        # M = 1000*s and N = s share one symbol. The ceiling is derived from a
        # single expression, so capping N at 16384 would let M reach 16,384,000;
        # the buckets also differ in two dims, which nearest match will not
        # borrow across. The single-axis control below isolates the cause.
        s = sympy.Symbol("s_shared", positive=True, integer=True)
        k = 256
        shared = _FakeWinner(
            "mm",
            [_buffer("mat1", [1000 * s, k], [k, 1]), _buffer("mat2", [k, s], [s, 1])],
            [1000 * s, s],
        )
        # _contiguous_strides cannot build these (both dims are symbolic) and
        # would yield [1, 1], which the density guard rejects -- masking the
        # guard under test. Give it the real dense strides.
        shared.layout = ir.FixedLayout(DEVICE, DTYPE, [1000 * s, s], [s, 1])
        self.assertEqual(_run(shared, sym=s), [])
        self.assertTrue(_run(self._mm_winner("mm", s), sym=s))

    def test_non_dense_operand_is_skipped(self):
        # [s, 256] strided [300, 1] is padded. _meta_at rebuilds strides densely,
        # so it would be tuned as [256, 1] -- an ld runtime never asks for, and
        # one the nearest matcher will not borrow across either.
        s = sympy.Symbol("s_pad", positive=True, integer=True)
        k, n = 256, 512
        padded = _FakeWinner(
            "mm",
            [_buffer("mat1", [s, k], [300, 1]), _buffer("mat2", [k, n], [n, 1])],
            [s, n],
        )
        self.assertEqual(_run(padded, sym=s), [])

    def test_choice_variants_produce_buckets(self):
        # The old op-name allowlist missed the _dtype out-variants and
        # _scaled_mm_v2. Matching runs on winner.choice.name.
        from types import SimpleNamespace

        for choice_name in (
            "mm_dtype",
            "addmm_dtype",
            "bmm_dtype",
            "_scaled_mm_v2",
        ):
            # Clear the cross-GEMM memo: every variant below reuses the same
            # traced shapes by design (the point is the choice gate, not the
            # sizes), so without this only the first variant would submit.
            tunableop._pretuned_gemms.cache_clear()
            s = sympy.Symbol(f"s_{choice_name}", positive=True, integer=True)
            m, k, n = s, 256, 512
            if choice_name == "addmm_dtype":
                # Bias-first like addmm: [bias, mat1, mat2].
                winner = _FakeWinner(
                    choice_name,
                    [
                        _buffer("bias", [m, n], [n, 1]),
                        _buffer("mat1", [m, k], [k, 1]),
                        _buffer("mat2", [k, n], [n, 1]),
                    ],
                    [m, n],
                    choice_name=choice_name,
                )
            else:
                winner = self._mm_winner("mm", s)
                winner.choice = SimpleNamespace(name=choice_name)
            submitted = _run(winner, sym=s)
            self.assertTrue(submitted, f"{choice_name} produced no buckets")

    def test_traced_hint_is_always_tuned(self):
        # The traced shape is the size actually served: it must be a rung even
        # when it is not a power of two and neither endpoint.
        s = sympy.Symbol("s_hint", positive=True, integer=True)
        submitted = _run(self._mm_winner("mm", s), sym=s, hint=3000)
        dims = set()
        for bmreq in submitted:
            dims.add(bmreq.input_tensor_meta[0].sizes[0])
        self.assertIn(3000, dims, f"traced hint missing from {sorted(dims)}")

    def test_largest_first_order(self):
        s = sympy.Symbol("s_order", positive=True, integer=True)
        submitted = _run(self._mm_winner("mm", s), sym=s)
        dims = [bmreq.input_tensor_meta[0].sizes[0] for bmreq in submitted]
        self.assertEqual(dims, sorted(dims, reverse=True))

    def test_deadline_skips_remaining_buckets(self):
        # Patch time.monotonic so the first bucket sees a healthy budget and
        # every later check sees an expired one; assert the surviving submit
        # got the remaining budget as its timeout.
        import time as _time

        s = sympy.Symbol("s_budget", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        timeouts = []
        submit_calls = {"n": 0}

        def _fake_submit(job, timeout=None):
            timeouts.append(timeout)
            submit_calls["n"] += 1
            return mock.Mock(status="ok", time=1.0)

        real_monotonic = _time.monotonic
        t0 = real_monotonic()
        mono_calls = {"n": 0}

        def _fake_monotonic():
            # Call 1 computes the deadline; call 2 (first budget check) is
            # healthy; everything after is expired.
            mono_calls["n"] += 1
            if mono_calls["n"] <= 2:
                return t0
            return t0 + 10**6

        with _ladder(s):
            with (
                mock.patch(
                    "torch._inductor.tunableop.submit", side_effect=_fake_submit
                ),
                mock.patch(
                    "torch._inductor.tunableop.time.monotonic",
                    side_effect=_fake_monotonic,
                ),
            ):
                tunableop.maybe_range_pretune(winner)
        # Exactly one bucket submitted with a positive remaining budget; the
        # rest were skipped by the expired deadline.
        self.assertEqual(submit_calls["n"], 1)
        self.assertGreater(timeouts[0], 0)

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
        tunableop._pretuned_gemms.cache_clear()
        try:
            s = sympy.Symbol("s_memo", positive=True, integer=True)
            first = _run(self._mm_winner("mm", s), sym=s)
            self.assertTrue(first)
            second = _run(self._mm_winner("mm", s), sym=s)
            self.assertEqual(second, [])
        finally:
            tunableop._pretuned_gemms.cache_clear()

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
        # Recording the memo key before the submissions made one fault
        # permanent for the process; a GEMM that tuned nothing must retry.
        tunableop._pretuned_gemms.cache_clear()
        try:
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
        finally:
            tunableop._pretuned_gemms.cache_clear()

    def test_memo_distinguishes_declared_range(self):
        # A narrower declared range builds a different ladder for the same
        # operands, so the second compile must not be skipped.
        tunableop._pretuned_gemms.cache_clear()
        try:
            s = sympy.Symbol("s_range", positive=True, integer=True)
            first = _run(self._mm_winner("mm", s), sym=s, declared=(2, 2**31 - 1))
            self.assertTrue(first)
            second = _run(self._mm_winner("mm", s), sym=s, declared=(2, 4096))
            self.assertTrue(second, "a different declared range is a new ladder")
        finally:
            tunableop._pretuned_gemms.cache_clear()

    def test_memo_distinguishes_stride_order(self):
        # Same sizes, different stride order (plain vs transposed B): the
        # TunableOp signatures differ (NN vs NT), so the second GEMM must
        # still ladder. Guards the memo key carrying strides.
        tunableop._pretuned_gemms.cache_clear()
        try:
            s = sympy.Symbol("s_stride", positive=True, integer=True)
            first = _run(self._mm_winner("mm", s), sym=s)
            self.assertTrue(first)
            mat1 = _buffer("mat1", [s, 256], [256, 1])
            mat2 = _buffer("mat2", [256, 512], [1, 256])  # .t() view
            winner = _FakeWinner("mm", [mat1, mat2], [s, 512])
            second = _run(winner, sym=s)
            self.assertTrue(second)
        finally:
            tunableop._pretuned_gemms.cache_clear()

    def test_memo_distinguishes_dynamic_axis(self):
        # Same hinted sizes but a different dynamic axis (M vs N): the ladders
        # cover different dims, so the second GEMM must still run. Guards the
        # memo key carrying the symbolic expressions, not just hints.
        tunableop._pretuned_gemms.cache_clear()
        try:
            s = sympy.Symbol("s_axis", positive=True, integer=True)
            first = _run(self._mm_winner("mm", s), sym=s)
            self.assertTrue(first)
            # Same hint 512 on N instead of M.
            mat1 = _buffer("mat1", [512, 256], [256, 1])
            mat2 = _buffer("mat2", [256, s], [s, 1])
            winner = _FakeWinner("mm", [mat1, mat2], [512, s])
            # _contiguous_strides cannot build the N-dynamic output (it would
            # yield [1, 1]): give it the real dense strides like the
            # symbol-spanning test does.
            winner.layout = ir.FixedLayout(DEVICE, DTYPE, [512, s], [s, 1])
            second = _run(winner, sym=s)
            self.assertTrue(second)
        finally:
            tunableop._pretuned_gemms.cache_clear()

    def test_bias_addmm_is_skipped(self):
        # bias_addmm is only chosen for broadcast (stride-0) bias, which the
        # density guard rejects: rebuilding its metas densely would tune a
        # layout runtime never asks for. Excluded from _TUNABLEOP_GEMM_CHOICES.
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
        # Batch s declared at exactly 16384 of 4096x4096 fp16 is 512 GiB per
        # operand -- over the cap -- while smaller rungs must still submit.
        s = sympy.Symbol("s_oom", positive=True, integer=True)
        m, k, n = 32, 4096, 4096
        mat1 = _buffer("mat1", [s, m, k], _contiguous_strides([s, m, k]))
        mat2 = _buffer("mat2", [s, k, n], _contiguous_strides([s, k, n]))
        winner = _FakeWinner("bmm", [mat1, mat2], [s, m, n])
        submitted = _run(winner, sym=s, hint=16384, declared=(2, 16384))
        self.assertTrue(submitted, "smaller rungs must still submit")
        for bmreq in submitted:
            for meta in [*bmreq.input_tensor_meta, bmreq.output_tensor_meta]:
                numel = 1
                for d in meta.sizes:
                    numel *= int(d)
                self.assertLessEqual(
                    numel * meta.dtype.itemsize, tunableop._TUNABLEOP_MAX_LADDER_BYTES
                )
        # And the 16384 rung itself is absent.
        batches = set()
        for bmreq in submitted:
            batches.add(bmreq.input_tensor_meta[0].sizes[0])
        self.assertNotIn(16384, batches)

    def test_select_algorithm_hook_fires_on_extern(self):
        # The select_algorithm hook covers concrete extern winners; triton
        # winners are ignored by the isinstance gate.
        from torch._inductor import select_algorithm as _sa

        s = sympy.Symbol("s_hook", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        with mock.patch("torch._inductor.tunableop.maybe_range_pretune") as pretune:
            _sa._maybe_pretune_tunableop(winner)
            pretune.assert_called_once_with(winner)
            pretune.reset_mock()
            _sa._maybe_pretune_tunableop(mock.Mock(spec=[]))
            pretune.assert_not_called()

    def test_finalize_hook_routes_extern_winners(self):
        # finalize_multi_template_buffers sends the chosen extern winner through
        # the ladder. Spec'd mocks so the isinstance gates pass.
        import torch._inductor.ir as _ir
        from torch._inductor.scheduler import Scheduler, SchedulerNode

        s = sympy.Symbol("s_fin", positive=True, integer=True)
        winner = self._mm_winner("mm", s)
        multi = mock.Mock(spec=_ir.MultiTemplateBuffer)
        multi.choices = [winner]
        multi.get_min_choice.return_value = (winner, None)
        multi.origins = set()
        multi.layout = winner.layout
        multi.origin_node = None
        sched_node = mock.Mock(spec=SchedulerNode)
        sched_node.node = multi
        sched = mock.Mock(spec=Scheduler)
        sched.nodes = [sched_node]
        sched._has_layout_conflict_for_template.return_value = True

        # The finalizer calls output_node() after the ladder hook and then
        # replaces the node -- deep IR machinery this test does not cover.
        # Raise a sentinel there: reaching it proves the hook already fired.
        class _Stop(Exception):
            pass

        winner.output_node = mock.Mock(side_effect=_Stop)
        with mock.patch("torch._inductor.tunableop.maybe_range_pretune") as pretune:
            with self.assertRaises(_Stop):
                Scheduler.finalize_multi_template_buffers(sched)
            pretune.assert_called_once_with(winner)


if __name__ == "__main__":
    run_tests()
