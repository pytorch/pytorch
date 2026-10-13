# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config, ir
from torch._inductor.codecache import PyCodeCache
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestDataDependentAdvanced(TestCase):
    def test_outer_product(self, device):
        device = torch.device(device)
        left_offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        right_offsets = torch.tensor([0, 3, 5, 5, 7], device=device)
        left = torch.arange(1, 7, dtype=torch.float32, device=device)
        right = torch.arange(10, 17, dtype=torch.float32, device=device)
        counts = left_offsets.diff() * right_offsets.diff()
        output_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        packed_size = int(output_offsets[-1])
        i_range = ir.DataDependentRange("i")
        j_range = ir.DataDependentRange("j")

        def reference():
            output = left.new_empty(packed_size)
            for b in range(4):
                height = int(left_offsets[b + 1] - left_offsets[b])
                width = int(right_offsets[b + 1] - right_offsets[b])
                for i in range(height):
                    for j in range(width):
                        p = int(output_offsets[b]) + i * width + j
                        output[p] = (
                            left[int(left_offsets[b]) + i]
                            * right[int(right_offsets[b]) + j]
                        )
            return output

        def datadep_output_indexer(index):
            b, i, j = index
            width = ops.load("right_offsets", b + 1) - ops.load("right_offsets", b)
            return (
                ops.load("output_offsets", b)
                + ops.index_expr(i, torch.int64) * width
                + ops.index_expr(j, torch.int64)
            )

        def inner_fn(index):
            b, i, j = index
            ops.register_datadep_range(
                ops.load("left_offsets", b + 1) - ops.load("left_offsets", b),
                i_range,
            )
            ops.register_datadep_range(
                ops.load("right_offsets", b + 1) - ops.load("right_offsets", b),
                j_range,
            )
            left_address = ops.indirect_indexing(
                ops.load("left_offsets", b) + ops.index_expr(i, torch.int64),
                left.numel(),
                wrap_neg=False,
            )
            right_address = ops.indirect_indexing(
                ops.load("right_offsets", b) + ops.index_expr(j, torch.int64),
                right.numel(),
                wrap_neg=False,
            )
            return ops.load("left", left_address) * ops.load("right", right_address)

        inputs = {
            "left_offsets": left_offsets,
            "right_offsets": right_offsets,
            "output_offsets": output_offsets,
            "left": left,
            "right": right,
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
                dtype=left.dtype,
                inner_fn=inner_fn,
                ranges=[4, i_range, j_range],
                datadep_output_flat_size=[packed_size],
                datadep_iteration_numel=packed_size,
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

    def test_concat(self, device):
        device = torch.device(device)
        left_offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        right_offsets = torch.tensor([0, 3, 5, 5, 7], device=device)
        left = torch.arange(1, 19, dtype=torch.float32, device=device).reshape(6, 3)
        right = torch.arange(101, 122, dtype=torch.float32, device=device).reshape(7, 3)
        counts = left_offsets.diff() + right_offsets.diff()
        output_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        packed_size = int(output_offsets[-1])
        j_range = ir.DataDependentRange("j")

        def reference():
            output = left.new_empty(packed_size, 3)
            for b in range(4):
                left_length = int(left_offsets[b + 1] - left_offsets[b])
                right_length = int(right_offsets[b + 1] - right_offsets[b])
                for j in range(left_length + right_length):
                    for d in range(3):
                        p = int(output_offsets[b]) + j
                        if j < left_length:
                            output[p, d] = left[int(left_offsets[b]) + j, d]
                        else:
                            output[p, d] = right[
                                int(right_offsets[b]) + j - left_length, d
                            ]
            return output

        def datadep_output_indexer(index):
            b, j, d = index
            return (
                ops.load("output_offsets", b) + ops.index_expr(j, torch.int64)
            ) * 3 + ops.index_expr(d, torch.int64)

        def inner_fn(index):
            b, j, d = index
            left_length = ops.load("left_offsets", b + 1) - ops.load("left_offsets", b)
            right_length = ops.load("right_offsets", b + 1) - ops.load(
                "right_offsets", b
            )
            ops.register_datadep_range(left_length + right_length, j_range)
            from_left = ops.lt(ops.index_expr(j, torch.int64), left_length)

            def load_left():
                p = ops.load("left_offsets", b) + ops.index_expr(j, torch.int64)
                address = ops.indirect_indexing(
                    p * 3 + ops.index_expr(d, torch.int64), left.numel(), wrap_neg=False
                )
                return ops.load("left", address)

            def load_right():
                length = ops.load("left_offsets", b + 1) - ops.load("left_offsets", b)
                p = (
                    ops.load("right_offsets", b)
                    + ops.index_expr(j, torch.int64)
                    - length
                )
                address = ops.indirect_indexing(
                    p * 3 + ops.index_expr(d, torch.int64),
                    right.numel(),
                    wrap_neg=False,
                )
                return ops.load("right", address)

            # where evaluates both candidates, so each load needs its own mask.
            return ops.where(
                from_left,
                ops.masked(from_left, load_left, 0.0),
                ops.masked(ops.logical_not(from_left), load_right, 0.0),
            )

        inputs = {
            "left_offsets": left_offsets,
            "right_offsets": right_offsets,
            "output_offsets": output_offsets,
            "left": left,
            "right": right,
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
                dtype=left.dtype,
                inner_fn=inner_fn,
                ranges=[4, j_range, 3],
                datadep_output_flat_size=[packed_size, 3],
                datadep_iteration_numel=packed_size * 3,
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

    def test_split(self, device):
        device = torch.device(device)
        offsets = torch.tensor([0, 4, 4, 7, 12], device=device)
        starts = torch.tensor([1, 0, 2, 1], device=device)
        lengths = torch.tensor([2, 0, 1, 3], device=device)
        output_offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        values = torch.arange(1, 37, dtype=torch.float32, device=device).reshape(12, 3)
        packed_size = int(output_offsets[-1])
        j_range = ir.DataDependentRange("j")

        def reference():
            output = values.new_empty(packed_size, 3)
            for b in range(4):
                for j in range(int(lengths[b])):
                    for d in range(3):
                        source = int(offsets[b]) + int(starts[b]) + j
                        output[int(output_offsets[b]) + j, d] = values[source, d]
            return output

        def datadep_output_indexer(index):
            b, j, d = index
            return (
                ops.load("output_offsets", b) + ops.index_expr(j, torch.int64)
            ) * 3 + ops.index_expr(d, torch.int64)

        def inner_fn(index):
            b, j, d = index
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            p = (
                ops.load("offsets", b)
                + ops.load("starts", b)
                + ops.index_expr(j, torch.int64)
            )
            address = ops.indirect_indexing(
                p * 3 + ops.index_expr(d, torch.int64), values.numel(), wrap_neg=False
            )
            return ops.load("values", address)

        inputs = {
            "offsets": offsets,
            "starts": starts,
            "lengths": lengths,
            "output_offsets": output_offsets,
            "values": values,
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
                ranges=[4, j_range, 3],
                datadep_output_flat_size=[packed_size, 3],
                datadep_iteration_numel=packed_size * 3,
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

    @parametrize("d", [32, 8192])
    def test_split_then_sum_d(self, device, d):
        # Slice J independently for each batch, then reduce the dense D axis:
        #
        #     for b in range(B):
        #         for j in range(lengths[b]):
        #             source = offsets[b] + starts[b] + j
        #             output[output_offsets[b] + j] = values[source, :].sum()
        device = torch.device(device)
        offsets = torch.tensor([0, 4, 4, 7, 12], device=device)
        starts = torch.tensor([1, 0, 2, 1], device=device)
        lengths = torch.tensor([2, 0, 1, 3], device=device)
        output_offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        values = torch.arange(12 * d, dtype=torch.float32, device=device)
        values = values.reshape(12, d) % 7
        packed_size = int(output_offsets[-1])
        j_range = ir.DataDependentRange("j")

        def reference():
            off, start, length = offsets.cpu(), starts.cpu(), lengths.cpu()
            out_off, x = output_offsets.cpu(), values.cpu()
            output = torch.empty(packed_size)
            for b in range(4):
                for j in range(int(length[b])):
                    source = int(off[b]) + int(start[b]) + j
                    output[int(out_off[b]) + j] = x[source].sum()
            return output.to(device)

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("output_offsets", b) + ops.index_expr(j, torch.int64)

        def inner_fn(index, reduction_index):
            b, j = index
            (r,) = reduction_index
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            source = (
                ops.load("offsets", b)
                + ops.load("starts", b)
                + ops.index_expr(j, torch.int64)
            )
            address = ops.indirect_indexing(
                source * d + ops.index_expr(r, torch.int64),
                values.numel(),
                wrap_neg=False,
            )
            return ops.load("values", address)

        inputs = {
            "offsets": offsets,
            "starts": starts,
            "lengths": lengths,
            "output_offsets": output_offsets,
            "values": values,
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

            result = ir.TensorBox.create(
                ir.Reduction(
                    device=device,
                    dtype=values.dtype,
                    inner_fn=inner_fn,
                    ranges=[4, j_range],
                    reduction_ranges=[d],
                    datadep_output_flat_size=[packed_size],
                    datadep_iteration_numel=packed_size * d,
                    datadep_output_indexer=datadep_output_indexer,
                    reduction_type="sum",
                    src_dtype=values.dtype,
                    reduction_hint=ir.ReductionHint.DEFAULT,
                )
            )
            result.realize()
            result.data.data.decide_layout()
            graph.graph_outputs = [result]
            wrapper, _ = graph.codegen()
            self.assertEqual(len(graph.scheduler.nodes), 1)
            self.assertEqual(wrapper.value.count("async_compile.triton("), 1)

        if d == 32:
            self.assertIn("@triton_heuristics.persistent_reduction(", wrapper.value)
        else:
            self.assertIn("@triton_heuristics.reduction(", wrapper.value)
        module = PyCodeCache.load(wrapper.value)
        (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
        torch.cuda.synchronize()
        self.assertEqual(actual, reference())

    @parametrize("d", [32, 8192])
    def test_grouped_gemm_ir(self, device, d):
        # The reduction keeps logical (b, j, k) and stores into packed [N, K].
        device = torch.device(device)
        offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        left = torch.arange(6 * d, device=device).reshape(6, d)
        right = torch.arange(4 * d * 16, device=device).reshape(4, d, 16)
        left = (left % 7 - 3).to(torch.float16) / 8
        right = (right % 11 - 5).to(torch.float16) / 8
        j_range = ir.DataDependentRange("j")

        def reference():
            output = torch.empty(6, 16, dtype=torch.float32)
            off, a, w = offsets.cpu(), left.cpu().float(), right.cpu().float()
            for b in range(4):
                start, end = int(off[b]), int(off[b + 1])
                output[start:end] = (a[start:end, :, None] * w[b][None, :, :]).sum(1)
            return output.to(device)

        def datadep_output_indexer(index):
            b, j, k = index
            return (
                ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            ) * 16 + ops.index_expr(k, torch.int64)

        def inner_fn(index, reduction_index):
            b, j, k = index
            (r,) = reduction_index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
            p = ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            address = ops.indirect_indexing(
                p * d + ops.index_expr(r, torch.int64), left.numel(), wrap_neg=False
            )
            a = ops.load("left", address)
            w = ops.load("right", b * d * 16 + r * 16 + k)
            return ops.mul(a, w)

        inputs = {"offsets": offsets, "left": left, "right": right}
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
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[4, j_range, 16],
                    reduction_ranges=[d],
                    datadep_output_flat_size=[6, 16],
                    datadep_iteration_numel=6 * 16 * d,
                    datadep_output_indexer=datadep_output_indexer,
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
            self.assertEqual(wrapper.value.count("async_compile.triton("), 1)
            self.assertEqual(wrapper.value.count(".run("), 1)

        module = PyCodeCache.load(wrapper.value)
        (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
        torch.cuda.synchronize()
        self.assertEqual(actual, reference())

    @parametrize("lengths", [[2, 0, 3, 1], [513, 0, 3, 1]])
    def test_jagged_jagged_bmm(self, device, lengths):
        # A[b, J, D].T @ B[b, J, K] produces a dense [B, D, K] output.
        device = torch.device(device)
        batches, n, d, k = len(lengths), sum(lengths), 3, 5
        offsets = torch.tensor([0, *lengths], device=device).cumsum(0)
        left = torch.arange(n * d, device=device).reshape(n, d)
        right = torch.arange(n * k, device=device).reshape(n, k)
        left = (left % 7 - 3).to(torch.float32) / 8
        right = (right % 11 - 5).to(torch.float32) / 8
        j_range = ir.DataDependentRange("j")

        def reference():
            output = torch.empty(batches, d, k, dtype=torch.float32)
            off, a, w = offsets.cpu(), left.cpu(), right.cpu()
            for b in range(batches):
                start, end = int(off[b]), int(off[b + 1])
                output[b] = (a[start:end, :, None] * w[start:end, None, :]).sum(0)
            return output.to(device)

        def inner_fn(index, reduction_index):
            b, m, col = index
            (j,) = reduction_index
            ops.register_datadep_range(
                ops.load("offsets", b + 1) - ops.load("offsets", b), j_range
            )
            p = ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            left_address = ops.indirect_indexing(
                p * d + ops.index_expr(m, torch.int64), left.numel(), wrap_neg=False
            )
            right_address = ops.indirect_indexing(
                p * k + ops.index_expr(col, torch.int64), right.numel(), wrap_neg=False
            )
            return ops.mul(
                ops.load("left", left_address), ops.load("right", right_address)
            )

        inputs = {"offsets": offsets, "left": left, "right": right}
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
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[batches, d, k],
                    reduction_ranges=[j_range],
                    datadep_output_flat_size=[batches, d, k],
                    datadep_iteration_numel=n * d * k,
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
            self.assertEqual(wrapper.value.count("async_compile.triton("), 1)

        module = PyCodeCache.load(wrapper.value)
        (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
        torch.cuda.synchronize()
        self.assertEqual(actual, reference())

    @parametrize(
        "j_lengths,k_lengths,d",
        [
            ([2, 0, 3, 1], [3, 2, 0, 2], 3),
            ([65], [33], 129),
            ([513, 0, 3, 1], [129, 7, 0, 16], 17),
            ([2, 0, 3, 1], [3, 2, 0, 2], 8192),
        ],
    )
    def test_jagged_jagged_bmm_jagged_out(self, device, j_lengths, k_lengths, d):
        # A[b, J, D] @ B[b, K, D].T stores each J x K matrix in a packed 1D tensor.
        device = torch.device(device)
        batches = len(j_lengths)
        counts = [j * k for j, k in zip(j_lengths, k_lengths)]
        packed_size = sum(counts)
        left_offsets = torch.tensor([0, *j_lengths], device=device).cumsum(0)
        right_offsets = torch.tensor([0, *k_lengths], device=device).cumsum(0)
        output_offsets = torch.tensor([0, *counts], device=device).cumsum(0)
        left = torch.arange(sum(j_lengths) * d, device=device).reshape(-1, d)
        right = torch.arange(sum(k_lengths) * d, device=device).reshape(-1, d)
        left = (left % 7 - 3).to(torch.float32) / 8
        right = (right % 11 - 5).to(torch.float32) / 8
        j_range = ir.DataDependentRange("j")
        k_range = ir.DataDependentRange("k")

        def reference():
            output = torch.empty(packed_size, dtype=torch.float32)
            oa, ob, oc = left_offsets.cpu(), right_offsets.cpu(), output_offsets.cpu()
            a, w = left.cpu(), right.cpu()
            for b in range(batches):
                a0, a1 = int(oa[b]), int(oa[b + 1])
                b0, b1 = int(ob[b]), int(ob[b + 1])
                c0, c1 = int(oc[b]), int(oc[b + 1])
                output[c0:c1] = (a[a0:a1, None, :] * w[None, b0:b1, :]).sum(2).flatten()
            return output.to(device)

        def datadep_output_indexer(index):
            b, j, k = index
            width = ops.load("right_offsets", b + 1) - ops.load("right_offsets", b)
            return (
                ops.load("output_offsets", b)
                + ops.index_expr(j, torch.int64) * width
                + ops.index_expr(k, torch.int64)
            )

        def inner_fn(index, reduction_index):
            b, j, k = index
            (r,) = reduction_index
            ops.register_datadep_range(
                ops.load("left_offsets", b + 1) - ops.load("left_offsets", b), j_range
            )
            ops.register_datadep_range(
                ops.load("right_offsets", b + 1) - ops.load("right_offsets", b), k_range
            )
            p = ops.load("left_offsets", b) + ops.index_expr(j, torch.int64)
            q = ops.load("right_offsets", b) + ops.index_expr(k, torch.int64)
            left_address = ops.indirect_indexing(
                p * d + ops.index_expr(r, torch.int64), left.numel(), wrap_neg=False
            )
            right_address = ops.indirect_indexing(
                q * d + ops.index_expr(r, torch.int64), right.numel(), wrap_neg=False
            )
            return ops.mul(
                ops.load("left", left_address), ops.load("right", right_address)
            )

        inputs = {
            "left_offsets": left_offsets,
            "right_offsets": right_offsets,
            "output_offsets": output_offsets,
            "left": left,
            "right": right,
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

            result = ir.TensorBox.create(
                ir.Reduction(
                    device=device,
                    dtype=torch.float32,
                    inner_fn=inner_fn,
                    ranges=[batches, j_range, k_range],
                    reduction_ranges=[d],
                    datadep_output_flat_size=[packed_size],
                    datadep_iteration_numel=packed_size * d,
                    datadep_output_indexer=datadep_output_indexer,
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
            self.assertEqual(wrapper.value.count("async_compile.triton("), 1)
            self.assertEqual(
                graph.scheduler.nodes[0].get_tiling(batches * j_range * k_range, d),
                {"z": batches, "y": j_range, "x": k_range, "r0_": d},
            )

        module = PyCodeCache.load(wrapper.value)
        (actual,) = module.call([inputs[name] for name in graph.graph_input_names])
        torch.cuda.synchronize()
        self.assertEqual(actual, reference())


instantiate_device_type_tests(TestDataDependentAdvanced, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
