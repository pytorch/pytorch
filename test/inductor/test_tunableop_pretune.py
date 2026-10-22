# Owner(s): ["module: inductor"]

"""Unit tests for the TunableOp bucket ladder in Scheduler.

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
from torch._inductor import config, ir
from torch._inductor import select_algorithm as sa
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)
from torch.utils._sympy.value_ranges import ValueRanges


DEVICE = torch.device("cpu")
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
    return ir.Buffer(
        name=name, layout=ir.FixedLayout(DEVICE, DTYPE, sizes, strides)
    )


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

    def __init__(self, name, input_nodes, out_sizes):
        self.name = name
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
        stack.enter_context(
            mock.patch("torch._inductor.tunableop.BenchmarkJob", _Job)
        )
        stack.enter_context(config.patch({"aten_tunableop_ladder": ladder}))
        yield submitted


class _Job:
    def __init__(self, bmreq, tuning=False, **kwargs):
        self.bmreq = bmreq


def _run(winner, **kwargs):
    """Drive the ladder; `self` is unused by the method so None suffices."""
    sym = kwargs.pop("sym")
    with _ladder(sym, **kwargs) as submitted:
        Scheduler._maybe_range_pretune_tunableop(None, winner)
    return submitted


class TestTunableOpPretune(TestCase):
    hw_classification = HardwareClassification.GENERIC

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
            Scheduler._maybe_range_pretune_tunableop(None, winner)
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


if __name__ == "__main__":
    run_tests()
