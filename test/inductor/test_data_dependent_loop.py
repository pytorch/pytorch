# Owner(s): ["module: inductor"]

"""Compare registered loop IR with ordinary Python loops.

The evaluator executes scalar IR operations; each test supplies the loops.
"""

from itertools import product
from math import prod

import sympy

import torch
from torch._inductor import dependencies, ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
from torch._inductor.virtualized import ops, V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class EagerOps:
    def __init__(self, buffers):
        self.buffers = buffers
        self.bounds = {}
        self.total = 0

    def load(self, name, index):
        return self.buffers[name][int(index)]

    def store(self, name, index, value, mode=None):
        self.buffers[name][int(index)] = value

    def constant(self, value, dtype):
        return value

    def index_expr(self, expr, dtype):
        return int(expr)

    def add(self, lhs, rhs):
        return lhs + rhs

    def mul(self, lhs, rhs):
        return lhs * rhs

    def indirect_indexing(self, value, size, check=True, wrap_neg=True):
        return sympy.Integer(int(value))

    def register_datadep_range(self, value, symbol):
        self.bounds[symbol] = int(value)
        return symbol

    def reduction(self, dtype, src_dtype, reduction_type, value):
        # Each body call contributes one element; the test resets total per row.
        combine = ir.get_reduction_combine_fn(reduction_type, src_dtype)
        self.total = combine(self.total, value).value
        return self.total

    def store_reduction(self, name, index, value):
        self.store(name, index, value)


class TestDataDependentIR(TestCase):
    def test_01_ops_address_capture(self, device):
        offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        lengths = offsets.diff()
        output = torch.full((6,), -1, dtype=torch.int64, device=device)
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.full_like(output, -1)
            for b in range(4):
                for j in range(int(lengths[b])):
                    result[int(offsets[b]) + j] = b + j
            return result

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        def inner_fn(index):
            b, j = index
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            return ops.index_expr(b + j, torch.int64)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            tensor = ir.Pointwise.create(
                device=torch.device(device),
                dtype=output.dtype,
                inner_fn=inner_fn,
                ranges=[4, j_range],
                datadep_output_flat_size=[6],
                datadep_output_indexer=datadep_output_indexer,
            )
            tensor.realize()
            buffer = tensor.data.data
            _, body, _ = buffer.get_default_sizes_body()
            evaluator = EagerOps(
                {"offsets": offsets, "lengths": lengths, buffer.name: output}
            )
            with V.set_ops_handler(evaluator):
                for b in range(4):
                    for j in range(int(lengths[b])):
                        body([b, j])
            self.assertEqual(output, reference())
            self.assertEqual(body.indirect_var_ranges[body.indirect_vars[-1]], 6)

    def test_02_bound_and_address_dependencies(self, device):
        j_range = ir.DataDependentRange("j")

        def inner_fn(index):
            ops.register_datadep_range(ops.load("lengths", index[0]), j_range)
            return ops.load("values", index[0])

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            tensor = ir.Pointwise.create(
                device=torch.device(device),
                dtype=torch.float32,
                inner_fn=inner_fn,
                ranges=[4, j_range],
                datadep_output_flat_size=[6],
                datadep_output_indexer=datadep_output_indexer,
            )
            tensor.realize()
            buffer = tensor.data.data
            rw = buffer.get_read_writes()
            self.assertEqual(
                set(buffer.data.get_read_names()), {"values", "lengths", "offsets"}
            )
            self.assertEqual(
                {dep.name for dep in rw.reads}, {"values", "lengths", "offsets"}
            )
            self.assertEqual({dep.name for dep in rw.writes}, {buffer.name})
            _, body, _ = buffer.get_default_sizes_body()
            self.assertTrue(body.has_op("register_datadep_range"))

    def test_03_copy_preserves_unit_dimension_and_bound(self, device):
        offsets = torch.tensor([0, 2, 2, 5], device=device)
        lengths = offsets.diff()
        output = torch.full((5,), -1, dtype=torch.int64, device=device)
        k_range = ir.DataDependentRange("k")
        sizes = [3, 1, k_range]

        def reference():
            result = torch.full_like(output, -1)
            for b in range(3):
                for k in range(int(lengths[b])):
                    result[int(offsets[b]) + k] = 10 * b + k
            return result

        def datadep_output_indexer(index):
            b, _, k = index
            return ops.load("offsets", b) + ops.index_expr(k, torch.int64)

        def inner_fn(index):
            b, unit, k = index
            ops.register_datadep_range(ops.load("lengths", b), k_range)
            return ops.index_expr(10 * b + unit + k, torch.int64)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            tensor = ir.Pointwise.create(
                device=torch.device(device),
                dtype=output.dtype,
                inner_fn=inner_fn,
                ranges=sizes,
                datadep_output_flat_size=[5],
                datadep_output_indexer=datadep_output_indexer,
            )
            tensor.realize()
            buffer = tensor.data.data
            _, body, _ = buffer.get_default_sizes_body()
            (renamed,), ranges = dependencies.index_vars_no_squeeze(sizes, prefix="r")
            copied = LoopBody(body, (renamed,), ranges, renamed, [])
            evaluator = EagerOps(
                {"offsets": offsets, "lengths": lengths, buffer.name: output}
            )
            with V.set_ops_handler(evaluator):
                for b in range(3):
                    for k in range(int(lengths[b])):
                        copied([b, 0, k])
                        self.assertEqual(evaluator.bounds[k_range], int(lengths[b]))
            self.assertEqual(len(copied.iter_vars), 3)
            self.assertEqual(output, reference())

    def test_04_realized_loader_uses_physical_coordinates(self, device):
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.empty(6, dtype=torch.int64, device=device)
            for b in range(3):
                for j in range(2):
                    result[2 * b + j] = b
            return result

        def inner_fn(index):
            ops.register_datadep_range(ops.index_expr(2, torch.int64), j_range)
            return ops.index_expr(index[0], torch.int64)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            tensor = ir.Pointwise.create(
                device=torch.device(device),
                dtype=torch.int64,
                inner_fn=inner_fn,
                ranges=[3, j_range],
                datadep_output_flat_size=[6],
                datadep_output_indexer=lambda index: 2 * index[0] + index[1],
            )
            tensor.realize()
            buffer = tensor.data.data
            buffer.freeze_layout()
            packed = reference()
            with V.set_ops_handler(EagerOps({buffer.name: packed})):
                logical = buffer.data.make_loader()([2, 1]).value
                physical = buffer.make_loader()([5]).value
            self.assertEqual(logical, int(packed[5]))
            self.assertEqual(physical, packed[5])

    def test_05_flat_layout_keeps_logical_loop_order(self, device):
        j_range = ir.DataDependentRange("j")
        ranges = [4, j_range]

        def inner_fn(index):
            ops.register_datadep_range(ops.load("lengths", index[0]), j_range)
            return ops.index_expr(index[0] + index[1], torch.int64)

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            tensor = ir.Pointwise.create(
                device=torch.device(device),
                dtype=torch.int64,
                inner_fn=inner_fn,
                ranges=ranges,
                datadep_output_flat_size=[6],
                datadep_output_indexer=datadep_output_indexer,
            )
            tensor.realize()
            buffer = tensor.data.data
            buffer.decide_layout()
            sizes, body = buffer.simplify_and_reorder()
            self.assertIsInstance(buffer.layout, ir.FixedLayout)
            self.assertEqual(buffer.get_size(), [6])
            self.assertEqual(buffer.get_stride(), [1])
            self.assertEqual(sizes, (ranges, []))
            self.assertIs(body.merge_loops(), body)

    def test_06_nonuniform_csr_add_bias(self, device):
        offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        lengths = offsets.diff()
        values = torch.arange(1, 7, dtype=torch.float32, device=device)
        bias = torch.tensor([10, 20, 30, 40], dtype=values.dtype, device=device)
        output = torch.zeros_like(values)
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.zeros_like(values)
            for b in range(4):
                for j in range(int(lengths[b])):
                    address = int(offsets[b]) + j
                    result[address] = values[address] + bias[b]
            return result

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("offsets", b) + ops.index_expr(j, torch.int64)

        def inner_fn(index):
            ops.register_datadep_range(ops.load("lengths", index[0]), j_range)
            address = ops.indirect_indexing(
                datadep_output_indexer(index), values.numel(), wrap_neg=False
            )
            return ops.load("values", address) + ops.load("bias", index[0])

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            lazy = ir.Pointwise.create(
                device=torch.device(device),
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[4, j_range],
                datadep_output_flat_size=[values.numel()],
                datadep_output_indexer=datadep_output_indexer,
            )
            lazy.realize()
            buffer = lazy.data.data
            _, body, _ = buffer.get_default_sizes_body()
            evaluator = EagerOps(
                {
                    "offsets": offsets,
                    "lengths": lengths,
                    "values": values,
                    "bias": bias,
                    buffer.name: output,
                }
            )
            with V.set_ops_handler(evaluator):
                for b in range(4):
                    for j in range(int(lengths[b])):
                        body([b, j])
                        self.assertEqual(evaluator.bounds[j_range], int(lengths[b]))
            self.assertEqual(output, reference())

    def test_07_csr_per_row_slice(self, device):
        source_offsets = torch.tensor([0, 3, 5, 9], device=device)
        destination_offsets = torch.tensor([0, 2, 2, 4], device=device)
        starts = torch.tensor([1, 0, 2], device=device)
        lengths = torch.tensor([2, 0, 2], device=device)
        values = torch.arange(1, 10, dtype=torch.float32, device=device)
        output = torch.zeros(4, dtype=values.dtype, device=device)
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.zeros_like(output)
            for b in range(3):
                for j in range(int(lengths[b])):
                    source = int(source_offsets[b]) + int(starts[b]) + j
                    destination = int(destination_offsets[b]) + j
                    result[destination] = values[source]
            return result

        def datadep_output_indexer(index):
            b, j = index
            return ops.load("destination_offsets", b) + ops.index_expr(j, torch.int64)

        def inner_fn(index):
            b, j = index
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            address = (
                ops.load("source_offsets", b)
                + ops.load("starts", b)
                + ops.index_expr(j, torch.int64)
            )
            address = ops.indirect_indexing(address, values.numel(), wrap_neg=False)
            return ops.load("values", address)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            lazy = ir.Pointwise.create(
                device=torch.device(device),
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[3, j_range],
                datadep_output_flat_size=[output.numel()],
                datadep_output_indexer=datadep_output_indexer,
            )
            lazy.realize()
            buffer = lazy.data.data
            _, body, _ = buffer.get_default_sizes_body()
            evaluator = EagerOps(
                {
                    "source_offsets": source_offsets,
                    "destination_offsets": destination_offsets,
                    "starts": starts,
                    "lengths": lengths,
                    "values": values,
                    buffer.name: output,
                }
            )
            with V.set_ops_handler(evaluator):
                for b in range(3):
                    for j in range(int(lengths[b])):
                        body([b, j])
                        self.assertEqual(evaluator.bounds[j_range], int(lengths[b]))
            self.assertEqual(output, reference())

    def test_08_nested_csf_add_bias(self, device):
        offsets = torch.tensor([0, 2, 2, 3], device=device)
        depths = torch.tensor([0, 1, 1, 3], device=device)
        row_lengths, depth_lengths = offsets.diff(), depths.diff()
        values = torch.arange(1, 4, dtype=torch.float32, device=device)
        bias = torch.tensor([10, 20, 30], dtype=values.dtype, device=device)
        output = torch.zeros_like(values)
        j_range, k_range = ir.DataDependentRange("j"), ir.DataDependentRange("k")

        def reference():
            result = torch.zeros_like(values)
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

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            lazy = ir.Pointwise.create(
                device=torch.device(device),
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[3, j_range, k_range],
                datadep_output_flat_size=[values.numel()],
                datadep_output_indexer=datadep_output_indexer,
            )
            lazy.realize()
            buffer = lazy.data.data
            _, body, _ = buffer.get_default_sizes_body()
            evaluator = EagerOps(
                {
                    "offsets": offsets,
                    "depths": depths,
                    "row_lengths": row_lengths,
                    "depth_lengths": depth_lengths,
                    "values": values,
                    "bias": bias,
                    buffer.name: output,
                }
            )
            with V.set_ops_handler(evaluator):
                for b in range(3):
                    for j in range(int(row_lengths[b])):
                        parent = int(offsets[b]) + j
                        for k in range(int(depth_lengths[parent])):
                            body([b, j, k])
                            self.assertEqual(
                                evaluator.bounds[j_range], int(row_lengths[b])
                            )
                            self.assertEqual(
                                evaluator.bounds[k_range], int(depth_lengths[parent])
                            )
            self.assertEqual(output, reference())

    @parametrize("feature_shape", [(2,), (2, 3), (2, 3, 2)])
    def test_packed_dense_features(self, device, feature_shape):
        offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        lengths = offsets.diff()
        values = torch.arange(
            6 * prod(feature_shape), dtype=torch.float32, device=device
        )
        values = values.reshape(6, *feature_shape)
        bias = torch.tensor([10, 20, 30, 40], device=device)
        output = torch.empty_like(values)
        strides = values.stride()
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.empty_like(values)
            for b in range(4):
                for j in range(int(lengths[b])):
                    packed = int(offsets[b]) + j
                    for feature in product(*(range(size) for size in feature_shape)):
                        result[(packed, *feature)] = (
                            values[(packed, *feature)] + bias[b]
                        )
            return result

        def datadep_output_indexer(index):
            b, j, *feature = index
            address = (
                ops.load("offsets", b) + ops.index_expr(j, torch.int64)
            ) * strides[0]
            for coordinate, stride in zip(feature, strides[1:]):
                address = address + ops.index_expr(coordinate * stride, torch.int64)
            return address

        def inner_fn(index):
            ops.register_datadep_range(ops.load("lengths", index[0]), j_range)
            address = ops.indirect_indexing(
                datadep_output_indexer(index), values.numel(), wrap_neg=False
            )
            return ops.load("values", address) + ops.load("bias", index[0])

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            tensor = ir.Pointwise.create(
                device=torch.device(device),
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[4, j_range, *feature_shape],
                datadep_output_flat_size=list(values.shape),
                datadep_iteration_numel=values.numel(),
                datadep_output_indexer=datadep_output_indexer,
            )
            tensor.realize()
            buffer = tensor.data.data
            buffer.decide_layout()
            sizes, body = buffer.simplify_and_reorder()
            self.assertEqual(sizes, ([4, prod(feature_shape), j_range], []))
            evaluator = EagerOps(
                {
                    "offsets": offsets,
                    "lengths": lengths,
                    "bias": bias,
                    "values": values.reshape(-1),
                    buffer.name: output.reshape(-1),
                }
            )
            with V.set_ops_handler(evaluator):
                for b in range(4):
                    for feature in range(prod(feature_shape)):
                        for j in range(int(lengths[b])):
                            body([b, feature, j])
                last = [5, *(size - 1 for size in feature_shape)]
                self.assertEqual(buffer.make_loader()(last).value, output[tuple(last)])
            self.assertEqual(output, reference())
            self.assertEqual(buffer.get_size(), list(values.shape))
            self.assertEqual(buffer.get_size(), buffer.data.datadep_output_flat_size)
            self.assertEqual(buffer.get_stride(), list(strides))
            self.assertEqual(
                body.indirect_var_ranges[body.indirect_vars[-1]], values.numel()
            )

    def test_row_sum_with_dense_features(self, device):
        offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        lengths = offsets.diff()
        values = torch.arange(36, dtype=torch.float32, device=device).reshape(6, 2, 3)
        output = torch.empty((4, 2, 3), device=device)
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.zeros_like(output)
            for b in range(4):
                for d in range(2):
                    for e in range(3):
                        for j in range(int(lengths[b])):
                            result[b, d, e] += values[int(offsets[b]) + j, d, e] + 1
            return result

        def inner_fn(index, rindex):
            b, d, e = index
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            address = (
                ops.load("offsets", b) + ops.index_expr(rindex[0], torch.int64)
            ) * 6
            address = address + ops.index_expr(d * 3 + e, torch.int64)
            address = ops.indirect_indexing(address, values.numel(), wrap_neg=False)
            return ops.load("values", address) + ops.constant(1, values.dtype)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            reduction = ir.Reduction(
                device=torch.device(device),
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[4, 2, 3],
                reduction_ranges=[j_range],
                datadep_output_flat_size=list(output.shape),
                reduction_type="sum",
                datadep_iteration_numel=values.numel(),
                src_dtype=values.dtype,
                reduction_hint=ir.ReductionHint.DEFAULT,
            )
            tensor = ir.TensorBox.create(reduction)
            tensor.realize()
            buffer = tensor.data.data
            buffer.decide_layout()
            self.assertEqual(buffer.get_size(), [4, 2, 3])
            self.assertEqual(buffer.get_size(), reduction.datadep_output_flat_size)
            _, body, _ = buffer.get_default_sizes_body()
            evaluator = EagerOps(
                {
                    "values": values.reshape(-1),
                    "offsets": offsets,
                    "lengths": lengths,
                    buffer.name: output.reshape(-1),
                }
            )
            with V.set_ops_handler(evaluator):
                for b in range(4):
                    for d in range(2):
                        for e in range(3):
                            evaluator.total = 0
                            for j in range(int(lengths[b])):
                                body([b, d, e], [j])
                            if int(lengths[b]) == 0:
                                evaluator.store_reduction(
                                    buffer.name, b * 6 + d * 3 + e, 0
                                )
            self.assertEqual(output, reference())

    def test_09_nonuniform_row_sum(self, device):
        values = torch.arange(1, 7, dtype=torch.float32, device=device)
        offsets = torch.tensor([0, 2, 2, 5, 6], device=device)
        lengths = offsets.diff()
        output = torch.full((4,), float("nan"), dtype=values.dtype, device=device)
        j_range = ir.DataDependentRange("j")

        def reference():
            result = torch.empty_like(output)
            for b in range(4):
                total = 0
                for j in range(int(offsets[b + 1] - offsets[b])):
                    total += values[int(offsets[b]) + j] + 1
                result[b] = total
            return result

        def inner_fn(index, rindex):
            b = index[0]
            ops.register_datadep_range(ops.load("lengths", b), j_range)
            start = ops.indirect_indexing(
                ops.load("offsets", b), values.numel(), wrap_neg=False
            )
            return ops.load("values", start + rindex[0]) + ops.constant(1, values.dtype)

        with V.set_graph_handler(GraphLowering(torch.fx.symbolic_trace(lambda x: x))):
            reduction = ir.Reduction(
                device=torch.device(device),
                dtype=values.dtype,
                inner_fn=inner_fn,
                ranges=[4],
                reduction_ranges=[j_range],
                datadep_output_flat_size=[4],
                datadep_iteration_numel=values.numel(),
                reduction_type="sum",
                src_dtype=values.dtype,
                reduction_hint=ir.ReductionHint.DEFAULT,
            )
            buffer = ir.ComputedBuffer(
                name="out",
                layout=ir.FixedLayout(torch.device(device), values.dtype, [4]),
                data=reduction,
            )
            _, body, _ = buffer.get_default_sizes_body()
            evaluator = EagerOps(
                {
                    "values": values,
                    "offsets": offsets,
                    "lengths": lengths,
                    buffer.name: output,
                }
            )
            with V.set_ops_handler(evaluator):
                for b in range(4):
                    evaluator.total = 0
                    length = int(lengths[b])
                    for j in range(length):
                        body([b], [j])
                        self.assertEqual(evaluator.bounds[j_range], length)
                    if length == 0:
                        # The parent loop stores the empty sum identity.
                        evaluator.store_reduction(buffer.name, b, evaluator.total)
            self.assertEqual(output, reference())


instantiate_device_type_tests(TestDataDependentIR, globals(), only_for="cpu")

if __name__ == "__main__":
    run_tests()
