# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR sequences translations."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder, GraphBuilder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops, core


aten = torch.ops.aten


def _atleast_sequence(self: Sequence[ir.Value] | ir.Value, ndim: int):
    if not isinstance(self, ir.Value):
        function = (core.aten_atleast_1d, core.aten_atleast_2d, core.aten_atleast_3d)[
            ndim - 1
        ]
        return [function(tensor) for tensor in self]

    builder = current_builder()
    tensor_type = self.type.elem_type
    tensor = ir.Value(name="atleast_tensor", type=tensor_type)
    body = ir.Graph([tensor], [], nodes=[], name="atleast_body")
    local = GraphBuilder(body, opset=builder.opset)
    rank = local.call("Size", local.call("Shape", tensor)[0])[0]

    def branch(name, shape=None, *, unsqueeze=False):
        graph = ir.Graph([], [], nodes=[], name=name)
        sub = GraphBuilder(graph, opset=builder.opset)
        if shape is None:
            result = sub.call("Identity", tensor)[0]
        else:
            result = sub.call("Unsqueeze" if unsqueeze else "Reshape", tensor, shape)[0]
        graph.outputs.append(result)
        for index, node in enumerate(graph):
            for output_index, output in enumerate(node.outputs):
                output.name = f"{name}_{index}_{output_index}"
        return graph

    unchanged = branch("atleast_unchanged")
    if ndim == 3:
        rank_two = local.call("Equal", rank, 2)[0]
        result = local.node(
            "If",
            [rank_two],
            attributes={
                "then_branch": branch("atleast_matrix", [-1], unsqueeze=True),
                "else_branch": unchanged,
            },
            output_types=[tensor_type],
        )[0]
        unchanged = ir.Graph([], [result], nodes=[], name="atleast_high_rank")
        # The second branch owns the nodes computing its captured rank test.
        for node in list(body):
            if node.op_type == "If":
                body.remove(node)
                unchanged.append(node)
    condition = local.call("LessOrEqual", rank, 0 if ndim == 1 else 1)[0]
    shape = ([1], [1, -1], [1, -1, 1])[ndim - 1]
    result = local.node(
        "If",
        [condition],
        attributes={
            "then_branch": branch("atleast_low_rank", shape),
            "else_branch": unchanged,
        },
        output_types=[tensor_type],
    )[0]
    body.outputs.append(result)
    for index, node in enumerate(body):
        for output_index, output in enumerate(node.outputs):
            output.name = f"atleast_body_{index}_{output_index}"
    return builder.node(
        "SequenceMap", [self], attributes={"body": body}, output_types=[self.type]
    )[0]


@onnx_impl(aten.atleast_1d.Sequence)
def aten_atleast_1d_sequence(self: Sequence[ir.Value]) -> ir.Value:
    """atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]"""
    return _atleast_sequence(self, 1)


@onnx_impl(aten.atleast_2d.Sequence)
def aten_atleast_2d_sequence(self: Sequence[ir.Value]) -> ir.Value:
    """atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]"""
    return _atleast_sequence(self, 2)


@onnx_impl(aten.atleast_3d.Sequence)
def aten_atleast_3d_sequence(self: Sequence[ir.Value]) -> ir.Value:
    """atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]"""
    return _atleast_sequence(self, 3)


# Do not register hstack - decomposed by PyTorch: https://github.com/pytorch/pytorch/blob/bedf96d7ffe74b34bcfe52c7ae1ae05f40d6c8ee/torch/_refs/__init__.py#L3918
def aten_hstack(tensors: Sequence[ir.Value]) -> ir.Value:
    """hstack(Tensor[] tensors) -> Tensor"""
    builder = current_builder()
    if not isinstance(tensors, ir.Value):
        if not tensors:
            raise ValueError(
                "hstack requires a nonempty tensor list; supply at least one tensor."
            )
        reshaped = _atleast_sequence(tensors, 1)
        return aten_cat(reshaped, 0 if len(tensors[0].shape) <= 1 else 1)
    tensors_atleast_2d = _atleast_sequence(tensors, 2)
    result = aten_cat(tensors_atleast_2d, 1)
    # hstack expects a non-empty sequence of tensors. So we don't need to check for length.
    first = aten_getitem(tensors, 0)
    rank_1d_or_less = builder.call(
        "Less", builder.call("Size", builder.call("Shape", first)[0])[0], 2
    )[0]
    then_branch = ir.Graph([], [], nodes=[], name="hstack_flatten")
    local = GraphBuilder(then_branch, opset=builder.opset)
    then_branch.outputs.append(local.call("Reshape", result, [-1])[0])
    else_branch = ir.Graph([], [], nodes=[], name="hstack_identity")
    local = GraphBuilder(else_branch, opset=builder.opset)
    else_branch.outputs.append(local.call("Identity", result)[0])
    for branch in (then_branch, else_branch):
        for index, node in enumerate(branch):
            for output_index, output in enumerate(node.outputs):
                output.name = f"{branch.name}_{index}_{output_index}"
    return builder.node(
        "If",
        [rank_1d_or_less],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[result.type],
    )[0]


# Do not register vstack - decomposed by PyTorch: https://github.com/pytorch/pytorch/blob/bedf96d7ffe74b34bcfe52c7ae1ae05f40d6c8ee/torch/_refs/__init__.py#L3918
def aten_vstack(tensors: Sequence[ir.Value]) -> ir.Value:
    """vstack(Tensor[] tensors) -> Tensor"""
    # The same logic as atleast_2d is shared explicitly in the native implementation.
    return aten_cat(_atleast_sequence(tensors, 2), 0)


@onnx_impl(
    (aten.cat.default, aten.concat.default, aten.concatenate.default), complex=True
)
def aten_cat_complex(tensors: Sequence[ir.Value], dim: int = 0) -> ir.Value:
    """cat(Tensor[] tensors, int dim=0) -> Tensor"""
    # Real representation unsqueezes the last dimension
    if dim < 0:
        dim = dim - 1
    return aten_cat(tensors, dim=dim)


@onnx_impl((aten.cat.default, aten.concat.default, aten.concatenate.default))
def aten_cat(tensors: Sequence[ir.Value], dim: int = 0) -> ir.Value:
    """cat(Tensor[] tensors, int dim=0) -> Tensor"""
    builder = current_builder()
    if isinstance(tensors, ir.Value):
        return builder.call(
            "ConcatFromSequence",
            tensors,
            axis=dim,
            _output_types=[tensors.type.elem_type],
        )[0]

    filtered_tensors = []
    for tensor in tensors:
        # Remove None tensors
        if tensor is None:
            continue
        # Remove empty tensors
        if tensor.shape is not None and tuple(tensor.shape) == (0,):
            continue
        filtered_tensors.append(tensor)
    if not filtered_tensors:
        empty = next((tensor for tensor in tensors if tensor is not None), None)
        if empty is None:
            raise ValueError(
                "aten::cat requires a nonempty list of tensors; supply at least one tensor."
            )
        return builder.call("Identity", empty)[0]
    if len(filtered_tensors) == 1:
        return builder.call("Identity", filtered_tensors[0])[0]
    return builder.call("Concat", *filtered_tensors, axis=dim)[0]


@onnx_impl(operator.getitem)
def aten_getitem(self: Sequence[ir.Value], i: ir.Value) -> ir.Value:
    builder = current_builder()
    if not isinstance(self, ir.Value):
        if not isinstance(i, int):
            self = builder.call("SequenceConstruct", *self)[0]
        else:
            return self[i]
    return builder.call("SequenceAt", self, i, _output_types=[self.type.elem_type])[0]


@onnx_impl(aten.repeat_interleave.self_int)
def aten_repeat_interleave_self_int(
    self: ir.Value,
    repeats: int,
    dim: int | None = None,
    output_size: int | None = None,
) -> ir.Value:
    """repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor

    The trick is to repeat in one direction orthogonal to reshape.

    .. code-block:: python

        x = torch.tensor([[0, 1, 2], [3, 4, 5]])
        x.repeat_interleave(2, dim=0)

    is equivalent to:

    .. code-block:: python

        x = torch.tensor([[0, 1, 2], [3, 4, 5]])
        x.repeat((1, 2)).reshape((-1, x.shape[1]))
    """
    builder = current_builder()
    if dim is None or (self.shape is not None and len(self.shape) == 0):
        self = builder.call("Reshape", self, [-1])[0]
        dim = 0
        self_rank = 1
    else:
        self_rank = len(self.shape)

    pos_dim = (dim + self_rank) % self_rank
    unsqueezed = builder.call("Unsqueeze", self, [pos_dim + 1])[0]
    if isinstance(repeats, int):
        tiles = [1] * (self_rank + 1)
        tiles[pos_dim + 1] = repeats
        tile_repeat = builder.call(
            "Constant", value=ir.tensor(tiles, dtype=ir.DataType.INT64)
        )[0]
    else:
        # repeats is a symbolic tensor
        tile_repeat = builder.call(
            "Concat",
            builder.call(
                "Constant",
                value=ir.tensor([1] * (pos_dim + 1), dtype=ir.DataType.INT64),
            )[0],
            builder.call(
                "Reshape",
                repeats,
                builder.call(
                    "Constant", value=ir.tensor([-1], dtype=ir.DataType.INT64)
                )[0],
            )[0],
            builder.call(
                "Constant",
                value=ir.tensor(
                    [1] * (self_rank - pos_dim - 1), dtype=ir.DataType.INT64
                ),
            )[0],
            axis=0,
        )[0]
    tiled = builder.call("Tile", unsqueezed, tile_repeat)[0]
    repeated_dim = builder.call(
        "Mul", builder.call("Shape", self, start=pos_dim, end=pos_dim + 1)[0], repeats
    )[0]
    final_shape = builder.call(
        "Concat",
        builder.call("Shape", self, start=0, end=pos_dim)[0],
        repeated_dim,
        builder.call("Shape", self, start=pos_dim + 1)[0],
        axis=0,
    )[0]
    return builder.call("Reshape", tiled, final_shape, allowzero=1)[0]


@onnx_impl(aten.repeat_interleave.Tensor)
def aten_repeat_interleave_Tensor(
    self: ir.Value,
    repeats: ir.Value | None = None,
    dim: int | None = None,
    output_size: int | None = None,
) -> ir.Value:
    """repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor

    When `repeats` is a tensor, each line is multiplied
    by a different number.
    There are multiple strategies. Here is one.

    .. code-block:: python

        import torch

        x = torch.tensor([[0, 1, 2], [3, 4, 5]])
        times = torch.tensor([2, 3], dtype=torch.int64)
        y = x.repeat_interleave(times, dim=0)
        print("repeat_interleave")
        print(y)

        ci = times.cumsum(dim=0)
        rows = torch.arange(ci[-1], dtype=torch.int64) < ci.reshape((-1, 1))
        srows = times.shape[0] - rows.to(torch.int64).sum(axis=0)
        indices = srows.reshape((-1,))
        print("decomposed")
        print(x[indices, :])
    """
    builder = current_builder()
    if repeats is None:
        repeats = self
        indices_only = True
    else:
        indices_only = False
    if not indices_only and (
        dim is None or (self.shape is not None and len(self.shape) == 0)
    ):
        # flatten
        self = builder.call("Reshape", self, [-1])[0]
        dim = 0
    repeats = builder.call("Cast", repeats, to=ir.DataType.INT64)[0]
    if not indices_only:
        # Expand a scalar or one-element vector to the selected axis length.
        rank = 1 if dim == 0 and self.shape is None else len(self.shape)
        axis = dim % rank
        repeats = builder.call(
            "Expand", repeats, builder.call("Shape", self, start=axis, end=axis + 1)[0]
        )[0]
    ci = builder.call("CumSum", repeats, 0)[0]
    total = builder.call("ReduceSum", repeats, keepdims=0)[0]
    trange = builder.call("Range", 0, total, 1)[0]
    rows = builder.call("Less", trange, builder.call("Unsqueeze", ci, [-1])[0])[0]
    srows = builder.call(
        "Sub",
        builder.call("Size", repeats)[0],
        builder.call(
            "ReduceSum",
            builder.call("Cast", rows, to=ir.DataType.INT64)[0],
            [0],
            keepdims=0,
        )[0],
    )[0]
    indices = builder.call("Reshape", srows, [-1])[0]
    if indices_only:
        return indices
    return builder.call("Gather", self, indices, axis=dim)[0]


@onnx_impl((aten.split.default, aten.split.Tensor))
def aten_split(self: ir.Value, split_size: ir.Value, dim: int = 0) -> ir.Value:
    """split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]"""
    builder = current_builder()

    return builder.call(
        "SplitToSequence",
        self,
        split_size,
        axis=dim,
        _output_types=[ir.SequenceType(self.type)],
    )[0]


@onnx_impl(aten.split_with_sizes.default)
def aten_split_with_sizes(
    self: ir.Value, split_sizes: ir.Value, dim: int = 0
) -> ir.Value:
    """split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]"""
    builder = current_builder()

    return builder.call(
        "SplitToSequence",
        self,
        common_ops.merge_dims(split_sizes),
        axis=dim,
        _output_types=[ir.SequenceType(self.type)],
    )[0]


@onnx_impl(aten.unsafe_split.Tensor)
def aten_unsafe_split(self: ir.Value, split_size: ir.Value, dim: int = 0) -> ir.Value:
    """unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]"""
    return aten_split(self, split_size, dim)


@onnx_impl(aten.chunk.default)
def aten_chunk(self: ir.Value, chunks: int, dim: int = 0) -> ir.Value:
    """chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]"""
    if chunks <= 0:
        raise ValueError(
            f"chunk requires a positive number of chunks, got {chunks}; pass chunks > 0."
        )
    builder = current_builder()
    # This will create a Sequence of tensors.
    # Get size of specified dim.
    dim_size = builder.call("Gather", builder.call("Shape", self)[0], dim, axis=0)[0]
    # Compute size/chunk to get the number of data in one chunk.
    num_per_chunk = builder.call(
        "Div", builder.call("Add", dim_size, chunks - 1)[0], chunks
    )[0]
    num_per_chunk = builder.call("Max", num_per_chunk, 1)[0]
    # Compute real chunk number, preserving all empty chunks when dim_size is zero.
    count = builder.call(
        "Div",
        builder.call("Add", dim_size, builder.call("Sub", num_per_chunk, 1)[0])[0],
        num_per_chunk,
    )[0]
    count = builder.call("Where", builder.call("Equal", dim_size, 0)[0], chunks, count)[
        0
    ]
    positions = builder.call("Range", 0, count, 1)[0]
    # Append the remainder to the [n, n, n, n, ..., r].
    sizes = builder.call(
        "Min",
        num_per_chunk,
        builder.call(
            "Max",
            builder.call(
                "Sub", dim_size, builder.call("Mul", positions, num_per_chunk)[0]
            )[0],
            0,
        )[0],
    )[0]
    return aten_split_with_sizes(self, sizes, dim)


@onnx_impl(aten.stack.default, complex=True)
def aten_stack_complex(tensors: Sequence[ir.Value], dim: int = 0) -> ir.Value:
    """stack(Tensor[] tensors, int dim=0) -> Tensor"""
    # Real representation unsqueezes the last dimension
    if dim < 0:
        dim = dim - 1
    return aten_stack(tensors, dim)


@onnx_impl(aten.stack.default)
def aten_stack(tensors: Sequence[ir.Value], dim: int = 0) -> ir.Value:
    """stack(Tensor[] tensors, int dim=0) -> Tensor"""
    builder = current_builder()
    if isinstance(tensors, Sequence):
        unsqueezed = [
            builder.call(
                "Unsqueeze", t, builder.constant([dim], dtype=ir.DataType.INT64)
            )[0]
            for t in tensors
        ]
        return builder.call("Concat", *unsqueezed, axis=dim)[0]
    return builder.call(
        "ConcatFromSequence",
        tensors,
        axis=dim,
        new_axis=1,
        _output_types=[tensors.type.elem_type],
    )[0]


@onnx_impl(aten.tile.default)
def aten_tile(self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """tile(Tensor self, int[] dims) -> Tensor"""
    builder = current_builder()

    self_rank = len(self.shape)
    dims_rank = len(dims)
    diff = self_rank - dims_rank

    if diff > 0:
        # dims is shorter than self.shape
        # pad dims with 1
        exapnd_ones = [1] * diff
        dims = [*exapnd_ones, *dims]

    elif diff < 0:
        # dims is longer than self.shape
        # pad self.shape with 1
        exapnd_ones = builder.constant([1] * (-diff), dtype=ir.DataType.INT64)
        self_shape = builder.call("Shape", self)[0]
        self_final_shape = builder.call("Concat", exapnd_ones, self_shape, axis=0)[0]
        self = builder.call("Reshape", self, self_final_shape, allowzero=True)[0]

    return builder.call("Tile", self, common_ops.merge_dims(dims))[0]


@onnx_impl(aten.unbind.int)
def aten_unbind(self: ir.Value, dim: int = 0) -> Sequence[ir.Value]:
    """unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]"""
    builder = current_builder()

    if isinstance(self.shape[dim], int):
        num_outputs = self.shape[dim]
        results = []
        for i in range(num_outputs):
            # Slice to get a single element at position i along dim
            sliced = builder.call(
                "Slice",
                self,
                builder.constant([i], dtype=ir.DataType.INT64),
                builder.constant([i + 1], dtype=ir.DataType.INT64),
                builder.constant([dim], dtype=ir.DataType.INT64),
            )[0]
            # Squeeze to remove the dimension of size 1
            squeezed = builder.call("Squeeze", sliced, [dim])[0]
            results.append(squeezed)
        return results

    return builder.call(
        "SplitToSequence",
        self,
        axis=dim,
        keepdims=False,
        _output_types=[ir.SequenceType(self.type)],
    )[0]
