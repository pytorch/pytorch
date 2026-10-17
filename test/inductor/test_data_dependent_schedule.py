# Owner(s): ["module: inductor"]

from unittest import skipUnless

import torch
from torch._inductor import config, ir
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import FusedSchedulerNode
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


@skipUnless(TEST_CUDA, "CUDA is required")
@instantiate_parametrized_tests
class TestDataDependentSchedule(TestCase):
    @parametrize("read_offsets", ["offsets", "other_offsets"])
    def test_bias_add_then_sum(self, read_offsets):
        device = torch.device("cuda", 0)
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        j_range = ir.DataDependentRange("j")

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        def add_body(index):
            b, _ = index
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            address = ops.indirect_indexing(
                datadep_output_indexer(index), 6, wrap_neg=False
            )
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
                ("other_offsets", torch.int64, [5]),
                ("lengths", torch.int64, [4]),
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
                datadep_output_flat_size=[6],
                datadep_output_indexer=datadep_output_indexer,
            )
            added.realize()
            added.data.data.decide_layout()
            added_loader = added.make_loader()

            def sum_body(index, rindex):
                b = index[0]
                ops.register_datadep_range(ops.load("lengths", b), j_range)
                address = ops.load(read_offsets, b) + ops.index_expr(
                    rindex[0], torch.int64
                )
                address = ops.indirect_indexing(address, 6, wrap_neg=False)
                return added_loader([address])

            result = ir.TensorBox.create(
                ir.Reduction(
                    device=device,
                    dtype=torch.float32,
                    inner_fn=sum_body,
                    ranges=[4],
                    reduction_ranges=[j_range],
                    datadep_output_flat_size=[4],
                    datadep_iteration_numel=6,
                    reduction_type="sum",
                    src_dtype=torch.float32,
                    reduction_hint=ir.ReductionHint.DEFAULT,
                )
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]

            # Use the scheduler entry from GraphLowering.codegen(), then stop.
            graph.init_wrapper_code()
            graph._update_scheduler()
            self.assertEqual(
                len(graph.scheduler.nodes), 1 if read_offsets == "offsets" else 2
            )
            if read_offsets == "offsets":
                self.assertIsInstance(graph.scheduler.nodes[0], FusedSchedulerNode)

    def test_two_sums_then_add(self):
        device = torch.device("cuda", 0)
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        j_range = ir.DataDependentRange("j")

        with (
            V.set_graph_handler(graph),
            DebugContext(),
            config.patch(benchmark_fusion=False),
        ):
            for name, dtype, size in (
                ("left", torch.float32, [6]),
                ("right", torch.float32, [6]),
                ("offsets", torch.int64, [5]),
                ("lengths", torch.int64, [4]),
            ):
                buffer = ir.InputBuffer(
                    name=name, layout=ir.FixedLayout(device, dtype, size)
                )
                graph.graph_inputs[name] = ir.TensorBox.create(buffer)
                graph.graph_inputs_original[name] = buffer
                graph.graph_input_names.append(name)

            loaders = []
            for name in ("left", "right"):

                def sum_body(index, rindex, name=name):
                    b = index[0]
                    ops.register_datadep_range(ops.load("lengths", b), j_range)
                    address = ops.load("offsets", b) + ops.index_expr(
                        rindex[0], torch.int64
                    )
                    address = ops.indirect_indexing(address, 6, wrap_neg=False)
                    return ops.load(name, address)

                reduced = ir.TensorBox.create(
                    ir.Reduction(
                        device=device,
                        dtype=torch.float32,
                        inner_fn=sum_body,
                        ranges=[4],
                        reduction_ranges=[j_range],
                        datadep_output_flat_size=[4],
                        datadep_iteration_numel=6,
                        reduction_type="sum",
                        src_dtype=torch.float32,
                        reduction_hint=ir.ReductionHint.DEFAULT,
                    )
                )
                reduced.realize()
                reduced.data.data.decide_layout()
                loaders.append(reduced.make_loader())

            def add_body(index):
                return loaders[0](index) + loaders[1](index)

            result = ir.Pointwise.create(
                device=device,
                dtype=torch.float32,
                inner_fn=add_body,
                ranges=[4],
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]

            graph.init_wrapper_code()
            graph._update_scheduler()
            self.assertEqual(len(graph.scheduler.nodes), 1)
            self.assertIsInstance(graph.scheduler.nodes[0], FusedSchedulerNode)


if __name__ == "__main__":
    run_tests()
