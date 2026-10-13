# Owner(s): ["module: inductor"]

from unittest import skipUnless

import torch
from torch._inductor import config, ir
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import FusedSchedulerNode
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase


@skipUnless(TEST_CUDA, "CUDA is required")
class TestDataDependentCodegen(TestCase):
    def test_pointwise_fusion_with_features(self):
        # for b in range(4):
        #     for j in range(offsets[b + 1] - offsets[b]):
        #         for d in range(2):
        #             for e in range(4):
        #                 p = offsets[b] + j
        #                 out[p, d, e] = (values[p, d, e] + bias[b, d, e]) * scale[d, e] + 1
        device = torch.device("cuda", 0)
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        j_range = ir.DataDependentRange("j")

        def output_indexer(index):
            b, j, d, e = index
            p = ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            return p * 8 + ops.index_expr(d * 4 + e, torch.int64)

        def add_body(index):
            b, _, d, e = index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
            address = ops.indirect_indexing(output_indexer(index), 48, wrap_neg=False)
            return ops.load("values", address) + ops.load("bias", b * 8 + d * 4 + e)

        with (
            V.set_graph_handler(graph),
            DebugContext(),
            config.patch(benchmark_fusion=False),
        ):
            for name, dtype, size in (
                ("values", torch.float32, [6, 2, 4]),
                ("bias", torch.float32, [4, 2, 4]),
                ("scale", torch.float32, [2, 4]),
                ("offsets", torch.int64, [5]),
            ):
                buffer = ir.InputBuffer(
                    name=name, layout=ir.FixedLayout(device, dtype, size)
                )
                graph.graph_inputs[name] = ir.TensorBox.create(buffer)
                graph.graph_inputs_original[name] = buffer
                graph.graph_input_names.append(name)

            added = ir.Pointwise.create(
                device=device,
                dtype=torch.float32,
                inner_fn=add_body,
                ranges=[4, j_range, 2, 4],
                flat_size=[6, 2, 4],
                iteration_numel=48,
                output_indexer=output_indexer,
            )
            added.realize()
            added.data.data.decide_layout()

            def scale_body(index):
                b, _, d, e = index
                ops.register_datadep_range(
                    ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
                )
                address = ops.indirect_indexing(
                    output_indexer(index), 48, wrap_neg=False
                )
                value = ops.load(added.get_name(), address)
                scale = ops.load("scale", d * 4 + e)
                return value * scale + ops.constant(1, torch.float32)

            result = ir.Pointwise.create(
                device=device,
                dtype=torch.float32,
                inner_fn=scale_body,
                ranges=[4, j_range, 2, 4],
                flat_size=[6, 2, 4],
                iteration_numel=48,
                output_indexer=output_indexer,
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]
            wrapper, _ = graph.codegen()

            self.assertEqual(len(graph.scheduler.nodes), 1)
            self.assertIsInstance(graph.scheduler.nodes[0], FusedSchedulerNode)
            self.assertIn("'grid_prefixes': ('x', 'z')", wrapper.value)
            self.assertIn("for yoffset in tl.range(", wrapper.value)
            self.assertNotIn("tl.load(out_ptr", wrapper.value)

    def test_bias_add_then_sum(self):
        # for b in range(4):
        #     out[b] = 0
        #     for j in range(offsets[b + 1] - offsets[b]):
        #         out[b] += values[offsets[b] + j] + bias[b]
        device = torch.device("cuda", 0)
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        j_range = ir.DataDependentRange("j")

        def output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        def add_body(index):
            b, _ = index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
            address = ops.indirect_indexing(output_indexer(index), 6, wrap_neg=False)
            return ops.load("values", address) + ops.load("bias", b)

        with (
            V.set_graph_handler(graph),
            DebugContext(),
            config.patch(benchmark_fusion=False),
        ):
            for name, dtype, size in (
                ("values", torch.float32, [6]),
                ("bias", torch.float32, [4]),
                ("offsets", torch.int64, [5]),
            ):
                buffer = ir.InputBuffer(
                    name=name, layout=ir.FixedLayout(device, dtype, size)
                )
                graph.graph_inputs[name] = ir.TensorBox.create(buffer)
                graph.graph_inputs_original[name] = buffer
                graph.graph_input_names.append(name)

            added = ir.Pointwise.create(
                device=device,
                dtype=torch.float32,
                inner_fn=add_body,
                ranges=[4, j_range],
                flat_size=[6],
                output_indexer=output_indexer,
            )
            added.realize()
            added.data.data.decide_layout()
            added_loader = added.make_loader()

            def sum_body(index, rindex):
                b = index[0]
                ops.register_datadep_range(
                    ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
                )
                address = ops.indirect_indexing(
                    output_indexer([b, rindex[0]]), 6, wrap_neg=False
                )
                return added_loader([address])

            result = ir.TensorBox.create(
                ir.Reduction(
                    device=device,
                    dtype=torch.float32,
                    inner_fn=sum_body,
                    ranges=[4],
                    reduction_ranges=[j_range],
                    iteration_numel=6,
                    reduction_type="sum",
                    src_dtype=torch.float32,
                    reduction_hint=ir.ReductionHint.DEFAULT,
                )
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]
            wrapper, _ = graph.codegen()

            self.assertEqual(len(graph.scheduler.nodes), 1)
            self.assertIsInstance(graph.scheduler.nodes[0], FusedSchedulerNode)
            self.assertIn("for r0_offset in tl.range(", wrapper.value)
            self.assertIn("tl.sum(", wrapper.value)
            self.assertNotIn("tl.load(out_ptr", wrapper.value)


if __name__ == "__main__":
    run_tests()
