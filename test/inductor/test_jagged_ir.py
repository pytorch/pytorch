# Owner(s): ["module: inductor"]

import torch
from torch._inductor import ir
from torch._inductor.dependencies import extract_read_writes
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
from torch.testing._internal.triton_utils import requires_cuda_and_triton


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


def trace(data):
    buffer = ir.ComputedBuffer(
        name="out",
        layout=ir.FixedLayout(torch.device("cpu"), torch.float32, data.get_size()),
        data=data,
    )
    return buffer.get_default_sizes_body()


@requires_cuda_and_triton
class TestJaggedReductionIR(TestCase):
    hw_classification = HardwareClassification.CUDA

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
            sizes, body, _ = buffer.get_default_sizes_body()
            self.assertTrue(body.has_op("register_jagged_range"))
            # The LoopBody fast path must not record the range as an index_expr.
            self.assertEqual(
                extract_read_writes(body, *sizes).index_exprs,
                buffer.get_read_writes().index_exprs,
            )

    @parametrize("size", ["j", "2*j"])
    def test_rejects_pointwise_range(self, size):
        j_range = JaggedRange("j")

        def inner_fn(index):
            return ops.load("values", index[0])

        with V.set_graph_handler(graph()):
            with self.assertRaisesRegex(AssertionError, "must be a reduction range"):
                ir.Pointwise.create(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[4, j_range if size == "j" else 2 * j_range],
                )

    @parametrize("case", ["pointwise", "dense", "missing", "twice", "masked"])
    def test_rejects_registration(self, case):
        j_range = JaggedRange("j")

        def register(b):
            ops.register_jagged_range(ops.load("lengths", b), j_range)

        def inner_fn(index, rindex=None):
            b = index[0]
            if case == "masked":

                def body():
                    register(b)
                    return ops.load("values", b)

                return ops.masked(ops.load("mask", b), body, 0.0)
            if case != "missing":
                register(b)
            if case == "twice":
                register(b)
            return ops.load("values", b)

        msg = (
            "inside subblocks"
            if case == "masked"
            else "must be called once for each JaggedRange reduction size"
        )
        with V.set_graph_handler(graph()):
            if case == "pointwise":
                data = ir.Pointwise(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[4],
                )
            else:
                data = row_sum(inner_fn, 8 if case == "dense" else j_range)
            with self.assertRaisesRegex(AssertionError, msg):
                trace(data)

    @parametrize(
        "op", ["minimum", "to_dtype", "masked", "indirect", "rindex", "jagged_range"]
    )
    def test_rejects_bound_op(self, op):
        j_range = JaggedRange("j")

        def bound(b, r):
            length = ops.load("lengths", b)
            if op == "minimum":
                return ops.minimum(length, ops.constant(2, torch.int64))
            if op == "to_dtype":
                return ops.to_dtype(length, torch.int32)
            if op == "masked":
                return ops.masked(
                    ops.load("mask", b), lambda: ops.load("lengths", b), 0
                )
            if op == "indirect":
                p = ops.indirect_indexing(length, 4, wrap_neg=False)
                return ops.load("lengths", p)
            if op == "rindex":
                return ops.load("lengths", r)
            return ops.index_expr(j_range, torch.int64)

        def inner_fn(index, rindex):
            ops.register_jagged_range(bound(index[0], rindex[0]), j_range)
            return ops.load("values", index[0])

        msgs = {
            "minimum": "bound uses minimum;",
            "to_dtype": "bound uses to_dtype;",
            "masked": "bound uses masked_subblock",
        }
        msg = msgs.get(op, "bound indexes with .*; it may not use")
        with V.set_graph_handler(graph()):
            with self.assertRaisesRegex(AssertionError, msg):
                trace(row_sum(inner_fn, j_range))

    def test_rejects_name_prefix(self):
        with self.assertRaisesRegex(AssertionError, "must start with 'j'"):
            JaggedRange("u0")


instantiate_parametrized_tests(TestJaggedReductionIR)

if __name__ == "__main__":
    run_tests()
