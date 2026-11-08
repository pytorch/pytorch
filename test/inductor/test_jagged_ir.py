# Owner(s): ["module: inductor"]

import unittest

import torch
import torch.nn.functional as F
from torch._inductor import config, ir
from torch._inductor.codecache import PyCodeCache
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import JaggedRange, run_and_get_code
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


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
            reduction = ir.Reduction(
                device=torch.device("cpu"),
                dtype=torch.float32,
                inner_fn=inner_fn,
                ranges=[4],
                reduction_ranges=[j_range],
                reduction_type="sum",
                src_dtype=torch.float32,
                reduction_hint=ir.ReductionHint.DEFAULT,
            )
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
                data=ir.Reduction(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[4],
                    reduction_ranges=[j_range],
                    reduction_type="sum",
                    src_dtype=torch.float32,
                    reduction_hint=ir.ReductionHint.DEFAULT,
                ),
            )
            with self.assertRaisesRegex(AssertionError, f"bound uses {targets[op]};"):
                buffer.get_default_sizes_body()

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_triton_row_sum_bj_to_b(self):
        # Row 2 is longer than one reduction block; row 1 is empty.
        offsets = torch.tensor([0, 2, 2, 1029, 1030], device="cuda")
        values = torch.randn(1030, device="cuda")
        j_range = JaggedRange("j")

        def inner_fn(index, rindex):
            b = index[0]
            length = ops.load("offsets", b + 1) - ops.load("offsets", b)
            ops.register_jagged_range(length, j_range)
            start = ops.indirect_indexing(ops.load("offsets", b), values.numel())
            return ops.load("values", start + rindex[0])

        reduction = ir.Reduction(
            device=values.device,
            dtype=values.dtype,
            inner_fn=inner_fn,
            ranges=[4],
            reduction_ranges=[j_range],
            reduction_type="sum",
            src_dtype=values.dtype,
            reduction_hint=ir.ReductionHint.DEFAULT,
        )
        actual, _ = compile_and_run(reduction, {"offsets": offsets, "values": values})
        self.assertEqual(actual, torch.segment_reduce(values, "sum", offsets=offsets))

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_triton_row_sum_bjd_to_bd(self):
        # Enough rows that a (y, x) tiling would otherwise be picked.
        lengths = torch.randint(0, 8, (256,), device="cuda")
        lengths[1], lengths[2] = 0, 1027
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
        values = torch.randn(int(offsets[-1]), 3, device="cuda")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_segment_reduce_inner_axis(self):
        offsets = torch.tensor([[0, 3, 3, 7], [0, 0, 5, 7]], device="cuda").int()
        data = torch.randn(2, 7, 4, device="cuda")

        def fn(data, offsets):
            return torch.segment_reduce(data, "sum", offsets=offsets, axis=1)

        actual, codes = run_and_get_code(torch.compile(fn), data, offsets)
        self.assertIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(data, offsets))

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_segment_reduce_constant_offsets(self):
        data = torch.randn(1030, 3, device="cuda")

        def fn(data):
            offsets = F.pad(torch.arange(10, 1031, 10, device=data.device), (1, 0))
            return torch.segment_reduce(data, "sum", offsets=offsets)

        actual, codes = run_and_get_code(torch.compile(fn), data)
        self.assertIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(data))

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
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

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    @parametrize("mode", ["sum", "mean"])
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_embedding_bag(self, mode, dtype):
        # Bags of 2, 0, 1027 and 1 indices.
        offsets = torch.tensor([0, 2, 2, 1029], device="cuda")
        indices = torch.randint(0, 100, (1030,), device="cuda")
        weight = torch.randn(100, 16, device="cuda", dtype=dtype)

        def fn(weight, indices, offsets):
            return F.embedding_bag(indices, weight, offsets, mode=mode)

        actual, codes = run_and_get_code(torch.compile(fn), weight, indices, offsets)
        self.assertIn("r0_bound", "\n".join(codes))
        # The summation order differs from eager.
        self.assertEqual(actual, fn(weight, indices, offsets), atol=1e-2, rtol=1e-2)

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_embedding_bag_options(self):
        offsets = torch.tensor([0, 2, 2, 1029], device="cuda")
        indices = torch.randint(0, 5, (1030,), device="cuda")
        weight = torch.randn(5, 16, device="cuda")
        per_sample_weights = torch.randn(1030, device="cuda")

        def weighted(weight, indices, offsets):
            return F.embedding_bag(
                indices,
                weight,
                offsets,
                mode="sum",
                per_sample_weights=per_sample_weights,
            )

        def padded_mean(weight, indices, offsets):
            return F.embedding_bag(indices, weight, offsets, mode="mean", padding_idx=3)

        def last_offset(weight, indices, offsets):
            offsets = torch.cat([offsets, offsets.new_tensor([1030])])
            return F.embedding_bag(
                indices, weight, offsets, mode="mean", include_last_offset=True
            )

        for fn in (weighted, padded_mean, last_offset):
            torch._dynamo.reset()
            actual, codes = run_and_get_code(
                torch.compile(fn), weight, indices, offsets
            )
            self.assertIn("r0_bound", "\n".join(codes))
            expected = fn(weight, indices, offsets)
            self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    @parametrize("mode", ["sum", "mean"])
    def test_embedding_bag_backward(self, mode):
        # The backward stays a fallback that reads offset2bag and bag_size.
        offsets = torch.tensor([0, 2, 2, 1029], device="cuda")
        indices = torch.randint(0, 5, (1030,), device="cuda")

        def fn(weight, indices, offsets):
            out = F.embedding_bag(indices, weight, offsets, mode=mode, padding_idx=3)
            return (out**2).sum()

        eager = torch.randn(5, 16, device="cuda", requires_grad=True)
        compiled = eager.detach().clone().requires_grad_()
        fn(eager, indices, offsets).backward()
        torch.compile(fn)(compiled, indices, offsets).backward()
        self.assertEqual(compiled.grad, eager.grad, atol=1e-2, rtol=1e-2)

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_embedding_bag_max_falls_back(self):
        offsets = torch.tensor([0, 2, 2, 1029], device="cuda")
        indices = torch.randint(0, 100, (1030,), device="cuda")
        weight = torch.randn(100, 16, device="cuda")

        def fn(weight, indices, offsets):
            return F.embedding_bag(indices, weight, offsets, mode="max")

        actual, codes = run_and_get_code(torch.compile(fn), weight, indices, offsets)
        self.assertNotIn("r0_bound", "\n".join(codes))
        self.assertEqual(actual, fn(weight, indices, offsets))

    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_embedding_bag_fusion(self):
        offsets = torch.tensor([0, 2, 2, 1029], device="cuda")
        indices = torch.randint(0, 100, (1030,), device="cuda")
        weights = torch.randn(2, 100, 16, device="cuda")
        bias = torch.randn(16, device="cuda")

        def fn(weights, indices, offsets, bias):
            a = F.embedding_bag(indices, weights[0], offsets, mode="sum")
            b = F.embedding_bag(indices, weights[1], offsets, mode="mean")
            return torch.relu(a + b + bias)

        # One kernel appends len(indices) to offsets; one runs both bags and
        # the epilogue in a single loop.
        args = (weights, indices, offsets, bias)
        actual, codes = run_and_get_code(torch.compile(fn), *args)
        self.assertEqual("\n".join(codes).count(".run("), 2)
        self.assertEqual(actual, fn(*args), atol=1e-2, rtol=1e-2)


instantiate_parametrized_tests(TestJaggedReductionIR)

if __name__ == "__main__":
    run_tests()
