# Owner(s): ["module: inductor"]

import torch
from torch._inductor import ir
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import JaggedRange
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def graph():
    return GraphLowering(torch.fx.symbolic_trace(lambda x: x))


def row_sum(inner_fn, j_range):
    return ir.Reduction(
        device=torch.device("cpu"),
        dtype=torch.float32,
        inner_fn=inner_fn,
        ranges=[4],
        reduction_ranges=[j_range],
        reduction_type="sum",
        src_dtype=torch.float32,
        reduction_hint=ir.ReductionHint.DEFAULT,
    )


class TestJaggedReductionIR(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_bound_dependencies(self):
        j_range = JaggedRange("j")

        def inner_fn(index, rindex):
            b = index[0]
            ops.register_jagged_range(ops.load("lengths", b), j_range)
            start = ops.indirect_indexing(ops.load("offsets", b), 6, wrap_neg=False)
            return ops.load("values", start + rindex[0])

        with V.set_graph_handler(graph()):
            reduction = row_sum(inner_fn, j_range)
            self.assertIn("ops.register_jagged_range(", str(reduction))
            tensor = ir.TensorBox.create(reduction)
            tensor.realize()
            buffer = tensor.data.data
            self.assertEqual(
                set(buffer.data.get_read_names()), {"values", "lengths", "offsets"}
            )
            self.assertEqual(
                {dep.name for dep in buffer.get_read_writes().reads},
                {"values", "lengths", "offsets"},
            )
            _, body, _ = buffer.get_default_sizes_body()
            self.assertTrue(body.has_op("register_jagged_range"))

    def test_rejects_pointwise_range(self):
        j_range = JaggedRange("j")

        def inner_fn(index):
            ops.register_jagged_range(ops.load("lengths", index[0]), j_range)
            return ops.load("values", index[0])

        with V.set_graph_handler(graph()):
            with self.assertRaisesRegex(AssertionError, "must be a reduction range"):
                ir.Pointwise.create(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[4, j_range],
                )

    @parametrize("op", ["minimum", "to_dtype", "indirect", "masked"])
    def test_rejects_bound_op(self, op):
        j_range = JaggedRange("j")

        def bound(b):
            length = ops.load("lengths", b)
            if op == "minimum":
                return ops.minimum(length, ops.constant(2, torch.int64))
            if op == "to_dtype":
                return ops.to_dtype(length, torch.int32)
            if op == "indirect":
                p = ops.indirect_indexing(length, 4, wrap_neg=False)
                return ops.load("lengths", p)
            return ops.masked(ops.load("mask", b), lambda: length, 0)

        def inner_fn(index, rindex):
            ops.register_jagged_range(bound(index[0]), j_range)
            return ops.load("values", index[0])

        targets = {
            "minimum": "minimum",
            "to_dtype": "to_dtype",
            "indirect": "set_indirect0",
            "masked": "masked_subblock1",
        }
        with V.set_graph_handler(graph()):
            buffer = ir.ComputedBuffer(
                name="out",
                layout=ir.FixedLayout(torch.device("cpu"), torch.float32, [4]),
                data=row_sum(inner_fn, j_range),
            )
            with self.assertRaisesRegex(AssertionError, f"bound uses {targets[op]};"):
                buffer.get_default_sizes_body()


instantiate_parametrized_tests(TestJaggedReductionIR)

if __name__ == "__main__":
    run_tests()
