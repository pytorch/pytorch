# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import sympy

import torch
from torch._inductor.kernel.flex.interval_mask_packing import (
    select_packed_mask_intervals,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import MockGraphHandler


@instantiate_parametrized_tests
class TestIntervalMaskPacking(InductorTestCase):
    def _expected_packed_mask_interval(self, case_name, q_idx, kv_idx, lane):
        match case_name:
            case "causal":
                return q_idx >= kv_idx + lane
            case "strict_causal":
                return q_idx > kv_idx + lane
            case "sliding_window":
                return q_idx >= kv_idx + lane and q_idx - (kv_idx + lane) <= 256
            case "disjoint_or":
                return kv_idx + lane in (q_idx, q_idx + 2)
            case "block_equality":
                return q_idx // 16 == (kv_idx + lane) // 16
            case "row_bounded_causal":
                return q_idx >= kv_idx + lane and q_idx < 100
            case "not_causal":
                return not q_idx >= kv_idx + lane
            case "ne_diagonal":
                return kv_idx + lane != q_idx
            case "lane_uniform_eq":
                return q_idx == 16
            case _:
                raise AssertionError(case_name)

    def _eval_packed_mask_intervals(self, intervals, q_idx, kv_idx):
        def eval_index(expr):
            source = (
                expr.replace("cutlass.Int32", "")
                .replace("q_idx[0]", "q_idx")
                .replace("kv_idx[0]", "kv_idx")
            )
            return int(
                eval(
                    source,
                    {"__builtins__": {}},
                    {"max": max, "min": min, "q_idx": q_idx, "kv_idx": kv_idx},
                )
            )

        mask = 0
        for interval in intervals:
            lower = eval_index(interval.render_lower())
            upper = eval_index(interval.render_upper())
            for lane in range(32):
                if lower <= lane < upper:
                    mask |= 1 << lane
        return mask

    @parametrize(
        "case_name",
        [
            "causal",
            "strict_causal",
            "sliding_window",
            "disjoint_or",
            "block_equality",
            "row_bounded_causal",
            "not_causal",
            "ne_diagonal",
            "lane_uniform_eq",
            "batch_dependent",
        ],
    )
    def test_packed_mask_interval_selector(self, case_name):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        full = graph.call_function(
            torch.ops.aten.full.default,
            ([], True),
            {"dtype": torch.bool, "layout": torch.strided, "device": "cuda"},
        )
        causal = graph.call_function(torch.ops.aten.ge.Tensor, (q_idx, kv_idx))
        distance = graph.call_function(torch.ops.aten.sub.Tensor, (q_idx, kv_idx))
        match case_name:
            case "causal":
                output = causal
            case "strict_causal":
                output = graph.call_function(torch.ops.aten.gt.Tensor, (q_idx, kv_idx))
            case "sliding_window":
                in_window = graph.call_function(
                    torch.ops.aten.le.Scalar, (distance, 256)
                )
                output = graph.call_function(
                    torch.ops.aten.bitwise_and.Tensor,
                    (
                        graph.call_function(
                            torch.ops.aten.bitwise_and.Tensor, (full, in_window)
                        ),
                        causal,
                    ),
                )
            case "disjoint_or":
                q_plus_2 = graph.call_function(torch.ops.aten.add.Scalar, (q_idx, 2))
                eq_q = graph.call_function(torch.ops.aten.eq.Tensor, (kv_idx, q_idx))
                eq_q_plus_2 = graph.call_function(
                    torch.ops.aten.eq.Tensor, (kv_idx, q_plus_2)
                )
                output = graph.call_function(
                    torch.ops.aten.bitwise_or.Tensor, (eq_q, eq_q_plus_2)
                )
            case "block_equality":
                q_block = graph.call_function(
                    torch.ops.aten.div.Tensor_mode,
                    (q_idx, 16),
                    {"rounding_mode": "floor"},
                )
                kv_block = graph.call_function(
                    torch.ops.aten.div.Tensor_mode,
                    (kv_idx, 16),
                    {"rounding_mode": "floor"},
                )
                output = graph.call_function(
                    torch.ops.aten.eq.Tensor, (q_block, kv_block)
                )
            case "batch_dependent":
                output = graph.call_function(torch.ops.aten.ge.Tensor, (b, kv_idx))
            case "row_bounded_causal":
                row_bound = graph.call_function(torch.ops.aten.lt.Scalar, (q_idx, 100))
                output = graph.call_function(
                    torch.ops.aten.bitwise_and.Tensor, (causal, row_bound)
                )
            case "not_causal":
                output = graph.call_function(
                    torch.ops.aten.logical_not.default, (causal,)
                )
            case "ne_diagonal":
                output = graph.call_function(torch.ops.aten.ne.Tensor, (kv_idx, q_idx))
            case "lane_uniform_eq":
                output = graph.call_function(torch.ops.aten.eq.Scalar, (q_idx, 16))
            case _:
                raise AssertionError(case_name)
        graph.output(output)
        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))
        if case_name == "batch_dependent":
            self.assertIsNone(intervals)
            return
        self.assertIsNotNone(intervals)
        for q in (0, 15, 16, 31, 32, 127, 256):
            for kv in (0, 16, 31, 32, 64, 127, 255, 288):
                expected_mask = sum(
                    int(self._expected_packed_mask_interval(case_name, q, kv, lane))
                    << lane
                    for lane in range(32)
                )
                self.assertEqual(
                    self._eval_packed_mask_intervals(intervals, q, kv), expected_mask
                )

    def _set_node_dtype(self, node, dtype, shape=()):
        node.meta["tensor_meta"] = SimpleNamespace(dtype=dtype, shape=shape)

    def test_packed_mask_interval_selector_rejects_non_affine_floor_div_lane(self):
        graph = torch.fx.Graph()
        graph.placeholder("b")
        graph.placeholder("h")
        graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        lhs = graph.call_function(torch.ops.aten.sub.Scalar, (kv_idx, 17))
        shifted_kv = graph.call_function(torch.ops.aten.add.Scalar, (kv_idx, 74))
        rhs = graph.call_function(
            torch.ops.aten.div.Tensor_mode,
            (shifted_kv, 2),
            {"rounding_mode": "floor"},
        )
        graph.output(graph.call_function(torch.ops.aten.le.Tensor, (lhs, rhs)))

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_rejects_non_integral_aux_bound(self):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        thresholds = graph.placeholder("thresholds")
        start = graph.call_function(
            torch.ops.aten.index.Tensor, (thresholds, [b, q_idx])
        )
        self._set_node_dtype(start, torch.float32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_supports_int64_aux_index_load(self):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        doc_ids = graph.placeholder("doc_ids")
        offsets = graph.placeholder("offsets")
        self._set_node_dtype(doc_ids, torch.int64, (1, 128))
        self._set_node_dtype(offsets, torch.int32, (5,))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int64)
        start = graph.call_function(torch.ops.aten.index.Tensor, (offsets, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("aux_tensors[0][b_idx[0], q_idx[0]]", intervals[0].render_lower())
        self.assertIn("aux_tensors[1]", intervals[0].render_lower())
        self.assertIn(" if ", intervals[0].render_lower())
        self.assertIn("q_idx[0]", intervals[0].render_upper())
        self.assertIn("cutlass.Int32(1)", intervals[0].render_upper())

    def test_packed_mask_interval_selector_rejects_interval_explosion(self):
        graph = torch.fx.Graph()
        graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        terms = []
        for offset in range(9):
            shifted_q = graph.call_function(torch.ops.aten.add.Scalar, (q_idx, offset))
            terms.append(
                graph.call_function(torch.ops.aten.eq.Tensor, (kv_idx, shifted_q))
            )
        expr = terms[0]
        for term in terms[1:]:
            expr = graph.call_function(torch.ops.aten.bitwise_or.Tensor, (expr, term))
        graph.output(expr)

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_rejects_partial_aux_index_bound(self):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        table = graph.placeholder("table")
        doc_ids = graph.placeholder("doc_ids")
        self._set_node_dtype(table, torch.int32, (1, 5))
        self._set_node_dtype(doc_ids, torch.int32, (1, 128))
        row = graph.call_function(torch.ops.aten.index.Tensor, (table, [b]))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int32)
        start = graph.call_function(torch.ops.aten.index.Tensor, (row, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_mixes_aux_scalars_and_tensors(self):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        limit = graph.placeholder("limit")
        doc_ids = graph.placeholder("doc_ids")
        offsets = graph.placeholder("offsets")
        self._set_node_dtype(doc_ids, torch.int32, (1, 128))
        self._set_node_dtype(offsets, torch.int32, (5,))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int32)
        start = graph.call_function(torch.ops.aten.index.Tensor, (offsets, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        below_limit = graph.call_function(torch.ops.aten.lt.Tensor, (kv_idx, limit))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, below_limit)
            )
        )

        limit_symbol = sympy.Symbol("s_limit", integer=True, nonnegative=True)
        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(
                torch.fx.GraphModule({}, graph),
                other_buffers=[limit_symbol, object(), object()],
                aux_scalar_symbol_codes={limit_symbol: "aux_scalars[0]"},
            )

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("aux_scalars[0]", intervals[0].render_upper())
        self.assertIn("aux_tensors[0]", intervals[0].render_lower())
        self.assertIn("aux_tensors[1]", intervals[0].render_lower())

    def test_packed_mask_interval_selector_aux_loaded_lower_bound(self):
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        doc_ids = graph.placeholder("doc_ids")
        offsets = graph.placeholder("offsets")
        self._set_node_dtype(doc_ids, torch.int32, (1, 128))
        self._set_node_dtype(offsets, torch.int32, (5,))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int32)
        start = graph.call_function(torch.ops.aten.index.Tensor, (offsets, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("max(", intervals[0].render_lower())
        self.assertIn("aux_tensors[0]", intervals[0].render_lower())
        self.assertIn("aux_tensors[1]", intervals[0].render_lower())
        self.assertIn("min(", intervals[0].render_upper())
        self.assertIn("q_idx[0]", intervals[0].render_upper())
        self.assertIn("cutlass.Int32(1)", intervals[0].render_upper())

    def test_packed_mask_interval_selector_padding_mask(self):
        """Regression for attention-gym #208: BERT-style padding masks must pack.

        (q_idx < seq_lens[b]) & (kv_idx < seq_lens[b]) mixes a lane-affine kv bound
        with a lane-uniform q bound; the lane-uniform conjunct used to reject the
        whole mask into the per-lane fallback path.
        """
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        seq_lens = graph.placeholder("seq_lens")
        self._set_node_dtype(seq_lens, torch.int64, (8,))
        length = graph.call_function(torch.ops.aten.index.Tensor, (seq_lens, [b]))
        self._set_node_dtype(length, torch.int64)
        q_ok = graph.call_function(torch.ops.aten.lt.Tensor, (q_idx, length))
        kv_ok = graph.call_function(torch.ops.aten.lt.Tensor, (kv_idx, length))
        graph.output(
            graph.call_function(torch.ops.aten.bitwise_and.Tensor, (q_ok, kv_ok))
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("min(", intervals[0].render_upper())
        self.assertIn("aux_tensors[0]", intervals[0].render_upper())
        self.assertIn("cutlass.Int32(32)", intervals[0].render_upper())


if __name__ == "__main__":
    run_tests()
