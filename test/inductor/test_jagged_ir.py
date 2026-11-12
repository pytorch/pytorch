# Owner(s): ["module: inductor"]

import torch
import torch.nn.functional as F
from torch._inductor import config, ir
from torch._inductor.codecache import PyCodeCache
from torch._inductor.debug import DebugContext
from torch._inductor.dependencies import extract_read_writes
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import JaggedRange, run_and_get_code
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


def compile_and_run(node, inputs):
    graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
    with (
        V.set_graph_handler(graph),
        DebugContext(),
        config.patch(benchmark_fusion=False),
    ):
        for name, tensor in inputs.items():
            layout = ir.FixedLayout(tensor.device, tensor.dtype, list(tensor.shape))
            buffer = ir.InputBuffer(name=name, layout=layout)
            graph.graph_inputs[name] = ir.TensorBox.create(buffer)
            graph.graph_inputs_original[name] = buffer
            graph.graph_input_names.append(name)
        result = ir.TensorBox.create(node)
        result.realize()
        graph.graph_outputs = [result]
        wrapper, _ = graph.codegen()
    module = PyCodeCache.load(wrapper.value)
    (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
    return actual, wrapper.value


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

    @parametrize("cooperative", [False, True])
    def test_triton_row_sum_bj_to_b(self, cooperative):
        # Row 2 is longer than any R0_BLOCK autotuning picks (at most 4096), so
        # its loop runs more than once; rows 1 and 4 are empty. Integer values
        # keep the sums exact. A forced cooperative reduction must still be
        # turned off.
        offsets = torch.tensor([0, 2, 2, 4099, 4100, 4100], device="cuda")
        values = torch.randint(-8, 8, (4100,), device="cuda").float()
        j_range = JaggedRange("j")

        def inner_fn(index, rindex):
            b = index[0]
            length = ops.load("offsets", b + 1) - ops.load("offsets", b)
            ops.register_jagged_range(length, j_range)
            row = ops.load("offsets", b) + ops.index_expr(rindex[0], torch.int64)
            return ops.load("values", ops.indirect_indexing(row, values.numel()))

        reduction = ir.Reduction(
            device=values.device,
            dtype=values.dtype,
            inner_fn=inner_fn,
            ranges=[5],
            reduction_ranges=[j_range],
            reduction_type="sum",
            src_dtype=values.dtype,
            reduction_hint=ir.ReductionHint.DEFAULT,
        )
        inputs = {"offsets": offsets, "values": values}
        with config.patch({"triton.force_cooperative_reductions": cooperative}):
            actual, _ = compile_and_run(reduction, inputs)
        self.assertEqual(actual, torch.segment_reduce(values, "sum", offsets=offsets))

    def test_triton_row_sum_bjd_to_bd(self):
        # Enough rows that a (y, x) tiling would otherwise be picked.
        lengths = torch.randint(0, 8, (256,), device="cuda")
        lengths[1], lengths[2] = 0, 4097
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
        values = torch.randint(-8, 8, (int(offsets[-1]), 3), device="cuda").float()
        j_range = JaggedRange("j")

        def inner_fn(index, rindex):
            b, d = index
            length = ops.load("offsets", b + 1) - ops.load("offsets", b)
            ops.register_jagged_range(length, j_range)
            row = ops.load("offsets", b) + ops.index_expr(rindex[0], torch.int64)
            address = row * 3 + ops.index_expr(d, torch.int64)
            address = ops.indirect_indexing(address, values.numel())
            return ops.load("values", address)

        reduction = ir.Reduction(
            device=values.device,
            dtype=values.dtype,
            inner_fn=inner_fn,
            ranges=[256, 3],
            reduction_ranges=[j_range],
            reduction_type="sum",
            src_dtype=values.dtype,
            reduction_hint=ir.ReductionHint.DEFAULT,
        )
        actual, code = compile_and_run(
            reduction, {"offsets": offsets, "values": values}
        )
        self.assertNotIn("ynumel", code)
        self.assertEqual(actual, torch.segment_reduce(values, "sum", offsets=offsets))

    @parametrize("reduce", ["sum", "mean", "max", "min", "prod", "amax", "amin"])
    @parametrize("shape", [(1030,), (1030, 3)])
    def test_segment_reduce_offsets(self, reduce, shape):
        offsets = torch.tensor([0, 2, 2, 1029, 1030], device="cuda")
        data = 1 + 0.001 * torch.randn(shape, device="cuda")

        def fn(data, offsets):
            return torch.segment_reduce(data, reduce, offsets=offsets)

        actual, codes = run_and_get_code(torch.compile(fn), data, offsets)
        self.assertIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(data, offsets))

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_segment_reduce_low_precision(self, dtype):
        offsets = torch.tensor([0, 2, 2, 1029, 1030], device="cuda")
        data = torch.rand(1030, 3, device="cuda", dtype=dtype) + 0.5

        def fn(data, offsets):
            return torch.segment_reduce(data, "sum", offsets=offsets)

        actual, codes = run_and_get_code(torch.compile(fn), data, offsets)
        self.assertIn("r0_bound", "\n".join(codes))
        # Eager accumulates in the data dtype; compare with an fp64 reference.
        expected = fn(data.double(), offsets).to(dtype)
        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)

    def test_segment_reduce_inner_axis(self):
        offsets = torch.tensor([[0, 3, 3, 7], [0, 0, 5, 7]], device="cuda").int()
        data = torch.randn(2, 7, 4, device="cuda")

        def fn(data, offsets):
            return torch.segment_reduce(data, "sum", offsets=offsets, axis=1)

        actual, codes = run_and_get_code(torch.compile(fn), data, offsets)
        self.assertIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(data, offsets))

    @parametrize("unsafe", [True, False])
    def test_segment_reduce_lengths(self, unsafe):
        lengths = torch.tensor([2, 0, 1027, 1], device="cuda")
        data = torch.randn(1030, 3, device="cuda")

        def fn(data, lengths):
            return torch.segment_reduce(data, "max", lengths=lengths, unsafe=unsafe)

        actual, codes = run_and_get_code(torch.compile(fn), data, lengths)
        code = "\n".join(codes)
        self.assertIn("r0_bound", code)
        # Without unsafe=True, eager's host checks become device asserts.
        self.assertEqual("lengths contains negative value" in code, not unsafe)
        self.assertEqual(actual, fn(data, lengths))

    @parametrize("segments", ["offsets", "lengths", "unsafe_lengths"])
    def test_segment_reduce_negative_axis(self, segments):
        offsets = torch.tensor([[0, 3, 3, 7], [0, 0, 5, 7]], device="cuda")
        data = torch.randn(2, 7, 4, device="cuda")
        if segments != "offsets":
            offsets = offsets.diff()

        def fn(data, offsets):
            if segments == "offsets":
                return torch.segment_reduce(data, "sum", offsets=offsets, axis=-2)
            return torch.segment_reduce(
                data,
                "sum",
                lengths=offsets,
                axis=-2,
                unsafe=segments == "unsafe_lengths",
            )

        actual, codes = run_and_get_code(torch.compile(fn), data, offsets)
        self.assertIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(data, offsets))

    @parametrize(
        "reduce,initial,lowered",
        [
            ("sum", 0, True),
            ("mean", 0, True),
            ("prod", 1, True),
            ("max", 0, True),
            ("max", float("-inf"), True),
            ("min", 0.5, True),
            ("sum", 1.0, False),
            ("prod", 2.0, False),
        ],
    )
    def test_segment_reduce_initial(self, reduce, initial, lowered):
        lengths = torch.tensor([2, 0, 1027, 1], device="cuda")
        data = torch.randn(1030, 3, device="cuda")
        if reduce == "prod":
            data = 1 + 0.001 * data

        def fn(data, lengths):
            return torch.segment_reduce(
                data, reduce, lengths=lengths, unsafe=True, initial=initial
            )

        actual, codes = run_and_get_code(torch.compile(fn), data, lengths)
        self.assertEqual("r0_bound" in "\n".join(codes), lowered)
        self.assertEqual(actual, fn(data, lengths))

    def test_segment_reduce_constant_offsets(self):
        data = torch.randn(1030, 3, device="cuda")

        def fn(data):
            offsets = F.pad(torch.arange(10, 1031, 10, device=data.device), (1, 0))
            return torch.segment_reduce(data, "sum", offsets=offsets)

        actual, codes = run_and_get_code(torch.compile(fn), data)
        self.assertIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(data))

    def test_segment_reduce_horizontal_fusion(self):
        offsets = torch.tensor([0, 2, 2, 1029, 1030], device="cuda")
        other = torch.tensor([0, 10, 20, 30, 1030], device="cuda")
        data = torch.randn(1030, 3, device="cuda")

        def same_offsets(data, offsets):
            return (
                torch.segment_reduce(data, "sum", offsets=offsets),
                torch.segment_reduce(data, "max", offsets=offsets),
            )

        def same_lengths(data, lengths):
            return (
                torch.segment_reduce(data, "sum", lengths=lengths, unsafe=True),
                torch.segment_reduce(data, "max", lengths=lengths, unsafe=True),
            )

        def different_offsets(data, offsets, other):
            return (
                torch.segment_reduce(data, "sum", offsets=offsets),
                torch.segment_reduce(data, "sum", offsets=other),
            )

        # One jagged reduction kernel per distinct offsets or lengths, plus
        # one scan for lengths.
        for fn, args, launches in [
            (same_offsets, (data, offsets), 1),
            (same_lengths, (data, offsets.diff()), 2),
            (different_offsets, (data, offsets, other), 2),
        ]:
            actual, codes = run_and_get_code(torch.compile(fn), *args)
            self.assertEqual("\n".join(codes).count(".run("), launches)
            self.assertEqual(actual, fn(*args))

    def test_segment_reduce_vertical_prologue(self):
        offsets = torch.tensor([0, 2, 2, 1029, 1030], device="cuda")
        data = torch.randn(1030, 3, device="cuda")

        def pointwise(data, offsets):
            return torch.segment_reduce((data * 2 + 1).relu(), "sum", offsets=offsets)

        def stored(data, offsets):
            y = (data * 2 + 1).relu()
            return y, torch.segment_reduce(y, "sum", offsets=offsets)

        # A pointwise prologue is inlined into the jagged loop. A prologue that
        # is stored runs as its own kernel.
        for fn, launches in [(pointwise, 1), (stored, 2)]:
            actual, codes = run_and_get_code(torch.compile(fn), data, offsets)
            self.assertEqual("\n".join(codes).count(".run("), launches)
            self.assertEqual(actual, fn(data, offsets), atol=1e-4, rtol=1e-4)

    def test_segment_reduce_vertical_epilogue(self):
        offsets = torch.tensor([0, 2, 2, 1029, 1030], device="cuda")
        data = torch.randn(1030, 3, device="cuda")
        bias = torch.randn(3, device="cuda")

        def pointwise(data, offsets, bias):
            out = torch.segment_reduce(data, "sum", offsets=offsets)
            return torch.relu(out * 2 + bias)

        def across_segments(data, offsets, bias):
            out = torch.segment_reduce(data, "sum", offsets=offsets)
            return out / out.sum(0) + bias

        # A pointwise epilogue runs after the jagged loop in the same kernel. An
        # epilogue that reduces across segments needs every segment first.
        for fn, launches in [(pointwise, 1), (across_segments, 2)]:
            actual, codes = run_and_get_code(torch.compile(fn), data, offsets, bias)
            self.assertEqual("\n".join(codes).count(".run("), launches)
            self.assertEqual(actual, fn(data, offsets, bias), atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestJaggedReductionIR)

if __name__ == "__main__":
    run_tests()
