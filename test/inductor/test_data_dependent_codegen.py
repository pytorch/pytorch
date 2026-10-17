# Owner(s): ["module: inductor"]

from unittest import skipUnless

import torch
from torch._inductor import config, ir
from torch._inductor.codecache import PyCodeCache
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import FusedSchedulerNode
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


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

        def datadep_output_indexer(index):
            b, j, d, e = index
            p = ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            return p * 8 + ops.index_expr(d * 4 + e, torch.int64)

        def add_body(index):
            b, _, d, e = index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
            address = ops.indirect_indexing(
                datadep_output_indexer(index), 48, wrap_neg=False
            )
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
                datadep_output_flat_size=[6, 2, 4],
                datadep_iteration_numel=48,
                datadep_output_indexer=datadep_output_indexer,
            )
            added.realize()
            added.data.data.decide_layout()

            def scale_body(index):
                b, _, d, e = index
                ops.register_datadep_range(
                    ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
                )
                address = ops.indirect_indexing(
                    datadep_output_indexer(index), 48, wrap_neg=False
                )
                value = ops.load(added.get_name(), address)
                scale = ops.load("scale", d * 4 + e)
                return value * scale + ops.constant(1, torch.float32)

            result = ir.Pointwise.create(
                device=device,
                dtype=torch.float32,
                inner_fn=scale_body,
                ranges=[4, j_range, 2, 4],
                datadep_output_flat_size=[6, 2, 4],
                datadep_iteration_numel=48,
                datadep_output_indexer=datadep_output_indexer,
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]
            wrapper, _ = graph.codegen()

            self.assertEqual(len(graph.scheduler.nodes), 1)
            self.assertIsInstance(graph.scheduler.nodes[0], FusedSchedulerNode)
            self.assertIn("'grid_prefixes': ('y', 'z')", wrapper.value)
            self.assertIn("for xoffset in tl.range(", wrapper.value)
            self.assertNotIn("tl.load(out_ptr", wrapper.value)

    def test_bias_add_then_sum(self):
        # for b in range(4):
        #     out[b] = 0
        #     for j in range(offsets[b + 1] - offsets[b]):
        #         out[b] += values[offsets[b] + j] + bias[b]
        device = torch.device("cuda", 0)
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        j_range = ir.DataDependentRange("j")

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        def add_body(index):
            b, _ = index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
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
                ops.register_datadep_range(
                    ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
                )
                address = ops.indirect_indexing(
                    datadep_output_indexer([b, rindex[0]]), 6, wrap_neg=False
                )
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
            wrapper, _ = graph.codegen()

            self.assertEqual(len(graph.scheduler.nodes), 1)
            self.assertIsInstance(graph.scheduler.nodes[0], FusedSchedulerNode)
            self.assertIn("for r0_offset in tl.range(", wrapper.value)
            self.assertIn("tl.sum(", wrapper.value)
            self.assertNotIn("tl.load(out_ptr", wrapper.value)


class TestDataDependentNestedCodegen(TestCase):
    # The larger input spans multiple J and K blocks, with empty rows and depths.
    @parametrize(
        "row_counts,depth_counts",
        [
            ([2, 0, 1], [1, 0, 2]),
            ([35, 0, 3], [65] + [0] * 15 + [3] + [0] * 17 + [2, 1, 0, 66]),
        ],
    )
    def test_nested_csf_add_bias(self, device, row_counts, depth_counts):
        device = torch.device(device)
        row_lengths = torch.tensor(row_counts, device=device)
        depth_lengths = torch.tensor(depth_counts, device=device)
        offsets = torch.cat([row_lengths.new_zeros(1), row_lengths.cumsum(0)])
        depths = torch.cat([depth_lengths.new_zeros(1), depth_lengths.cumsum(0)])
        values = torch.arange(
            1, sum(depth_counts) + 1, dtype=torch.float32, device=device
        )
        bias = torch.tensor([10, 20, 30], dtype=values.dtype, device=device)
        j_range = ir.DataDependentRange("j")
        k_range = ir.DataDependentRange("k")

        def reference():
            result = torch.empty_like(values)
            for b in range(3):
                for j in range(int(row_lengths[b])):
                    parent = int(offsets[b]) + j
                    for k in range(int(depth_lengths[parent])):
                        address = int(depths[parent]) + k
                        result[address] = values[address] + bias[b]
            return result

        def datadep_output_indexer(index):
            b, j, k = index
            parent = ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            parent = ops.indirect_indexing(
                parent, depth_lengths.numel(), wrap_neg=False
            )
            return ops.load("depths", parent) + ops.index_expr(k, torch.int64)

        def inner_fn(index):
            b, j, k = index
            ops.register_datadep_range(ops.load("row_lengths", b), j_range)
            parent = ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            parent = ops.indirect_indexing(
                parent, depth_lengths.numel(), wrap_neg=False
            )
            ops.register_datadep_range(ops.load("depth_lengths", parent), k_range)
            address = ops.indirect_indexing(
                datadep_output_indexer(index), values.numel(), wrap_neg=False
            )
            return ops.load("values", address) + ops.load("bias", b)

        inputs = {
            "offsets": offsets,
            "depths": depths,
            "row_lengths": row_lengths,
            "depth_lengths": depth_lengths,
            "values": values,
            "bias": bias,
        }
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        with (
            V.set_graph_handler(graph),
            DebugContext(),
            config.patch(benchmark_fusion=False),
        ):
            for name, tensor in inputs.items():
                buffer = ir.InputBuffer(
                    name=name,
                    layout=ir.FixedLayout(device, tensor.dtype, list(tensor.shape)),
                )
                graph.graph_inputs[name] = ir.TensorBox.create(buffer)
                graph.graph_inputs_original[name] = buffer
                graph.graph_input_names.append(name)

            result = ir.Pointwise.create(
                device=device,
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[3, j_range, k_range],
                datadep_output_flat_size=[values.numel()],
                datadep_output_indexer=datadep_output_indexer,
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]
            wrapper, _ = graph.codegen()

        module = PyCodeCache.load(wrapper.value)
        (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
        torch.cuda.synchronize()
        self.assertEqual(actual, reference())


instantiate_device_type_tests(
    TestDataDependentNestedCodegen, globals(), only_for="cuda"
)


class TestDataDependentReductionCodegen(TestCase):
    @parametrize("width", [32, 8192])
    def test_sum_d(self, device, width):
        device = torch.device(device)
        offsets = torch.tensor([0, 513, 513, 516, 517], device=device)
        values = torch.arange(517 * width, dtype=torch.float32, device=device)
        values = values.reshape(517, width) % 7
        j_range = ir.DataDependentRange("j")

        def reference():
            off, x = offsets.cpu(), values.cpu()
            result = torch.empty(517)
            for b in range(4):
                for j in range(int(off[b + 1] - off[b])):
                    p = int(off[b]) + j
                    result[p] = x[p].sum()
            return result.to(device)

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        def inner_fn(index, reduction_index):
            b = index[0]
            (d,) = reduction_index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
            address = ops.indirect_indexing(
                datadep_output_indexer(index) * width + ops.index_expr(d, torch.int64),
                values.numel(),
                wrap_neg=False,
            )
            return ops.load("values", address)

        inputs = {"offsets": offsets, "values": values}
        graph = GraphLowering(torch.fx.symbolic_trace(lambda: 0))
        with (
            V.set_graph_handler(graph),
            DebugContext(),
            config.patch(benchmark_fusion=False),
        ):
            for name, tensor in inputs.items():
                buffer = ir.InputBuffer(
                    name=name,
                    layout=ir.FixedLayout(device, tensor.dtype, list(tensor.shape)),
                )
                graph.graph_inputs[name] = ir.TensorBox.create(buffer)
                graph.graph_inputs_original[name] = buffer
                graph.graph_input_names.append(name)

            result = ir.TensorBox.create(
                ir.Reduction(
                    device=device,
                    dtype=values.dtype,
                    inner_fn=inner_fn,
                    ranges=[4, j_range],
                    reduction_ranges=[width],
                    datadep_output_flat_size=[517],
                    datadep_iteration_numel=values.numel(),
                    datadep_output_indexer=datadep_output_indexer,
                    reduction_type="sum",
                    src_dtype=values.dtype,
                    reduction_hint=ir.ReductionHint.DEFAULT,
                )
            )
            result.realize()
            buffer = result.data.data
            buffer.decide_layout()
            self.assertEqual(buffer.get_size(), [517])
            self.assertEqual(buffer.get_size(), buffer.data.datadep_output_flat_size)
            graph.graph_outputs = [result]
            wrapper, _ = graph.codegen()

        if width == 32:
            self.assertIn("@triton_heuristics.persistent_reduction(", wrapper.value)
        else:
            self.assertIn("@triton_heuristics.reduction(", wrapper.value)
        module = PyCodeCache.load(wrapper.value)
        (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
        torch.cuda.synchronize()
        self.assertEqual(actual, reference())


instantiate_device_type_tests(
    TestDataDependentReductionCodegen, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
