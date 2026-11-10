# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR sequences translations."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import GraphBuilder, TranslationContext
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops, core


aten = torch.ops.aten


def _atleast_sequence(
    op: TranslationContext, self: Sequence[ir.Value] | ir.Value, ndim: int
):
    if not isinstance(self, ir.Value):
        function = (core.aten_atleast_1d, core.aten_atleast_2d, core.aten_atleast_3d)[
            ndim - 1
        ]
        return [function(op, tensor) for tensor in self]

    builder = op.builder
    tensor_type = self.type.elem_type
    tensor = ir.Value(name="atleast_tensor", type=tensor_type)
    body = ir.Graph([tensor], [], nodes=[], name="atleast_body")
    local = GraphBuilder(body, opset=builder.opset)
    local_op = TranslationContext(local)
    rank = local_op.Size(local_op.Shape(tensor))

    def branch(op: TranslationContext, name, shape=None, *, unsqueeze=False):
        graph = ir.Graph([], [], nodes=[], name=name)
        sub = GraphBuilder(graph, opset=builder.opset)
        sub_op = TranslationContext(sub)
        if shape is None:
            result = sub_op.Identity(tensor)
        else:
            result = (
                sub_op.Unsqueeze(tensor, shape)
                if unsqueeze
                else sub_op.Reshape(tensor, shape)
            )
        graph.outputs.append(result)
        for index, node in enumerate(graph):
            for output_index, output in enumerate(node.outputs):
                output.name = f"{name}_{index}_{output_index}"
        return graph

    unchanged = branch(op, "atleast_unchanged")
    if ndim == 3:
        rank_two = local_op.Equal(rank, 2)
        result = local.node(
            "If",
            [rank_two],
            attributes={
                "then_branch": branch(op, "atleast_matrix", [-1], unsqueeze=True),
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
    condition = local_op.LessOrEqual(rank, 0 if ndim == 1 else 1)
    shape = ([1], [1, -1], [1, -1, 1])[ndim - 1]
    result = local.node(
        "If",
        [condition],
        attributes={
            "then_branch": branch(op, "atleast_low_rank", shape),
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
def aten_atleast_1d_sequence(
    op: TranslationContext, self: Sequence[ir.Value]
) -> ir.Value:
    """atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]"""
    return _atleast_sequence(op, self, 1)


@onnx_impl(aten.atleast_2d.Sequence)
def aten_atleast_2d_sequence(
    op: TranslationContext, self: Sequence[ir.Value]
) -> ir.Value:
    """atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]"""
    return _atleast_sequence(op, self, 2)


@onnx_impl(aten.atleast_3d.Sequence)
def aten_atleast_3d_sequence(
    op: TranslationContext, self: Sequence[ir.Value]
) -> ir.Value:
    """atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]"""
    return _atleast_sequence(op, self, 3)


# Do not register hstack - decomposed by PyTorch: https://github.com/pytorch/pytorch/blob/bedf96d7ffe74b34bcfe52c7ae1ae05f40d6c8ee/torch/_refs/__init__.py#L3918
def aten_hstack(op: TranslationContext, tensors: Sequence[ir.Value]) -> ir.Value:
    """hstack(Tensor[] tensors) -> Tensor"""
    builder = op.builder
    if not isinstance(tensors, ir.Value):
        if not tensors:
            raise ValueError(
                "hstack requires a nonempty tensor list; supply at least one tensor."
            )
        reshaped = _atleast_sequence(op, tensors, 1)
        return aten_cat(op, reshaped, 0 if len(tensors[0].shape) <= 1 else 1)
    tensors_atleast_2d = _atleast_sequence(op, tensors, 2)
    result = aten_cat(op, tensors_atleast_2d, 1)
    # hstack expects a non-empty sequence of tensors. So we don't need to check for length.
    first = aten_getitem(op, tensors, 0)
    rank_1d_or_less = op.Less(op.Size(op.Shape(first)), 2)
    then_branch = ir.Graph([], [], nodes=[], name="hstack_flatten")
    local = GraphBuilder(then_branch, opset=builder.opset)
    local_op = TranslationContext(local)
    then_branch.outputs.append(local_op.Reshape(result, [-1]))
    else_branch = ir.Graph([], [], nodes=[], name="hstack_identity")
    local = GraphBuilder(else_branch, opset=builder.opset)
    local_op = TranslationContext(local)
    else_branch.outputs.append(local_op.Identity(result))
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
def aten_vstack(op: TranslationContext, tensors: Sequence[ir.Value]) -> ir.Value:
    """vstack(Tensor[] tensors) -> Tensor"""
    # The same logic as atleast_2d is shared explicitly in the native implementation.
    return aten_cat(op, _atleast_sequence(op, tensors, 2), 0)


@onnx_impl(
    (aten.cat.default, aten.concat.default, aten.concatenate.default), complex=True
)
def aten_cat_complex(
    op: TranslationContext, tensors: Sequence[ir.Value], dim: int = 0
) -> ir.Value:
    """cat(Tensor[] tensors, int dim=0) -> Tensor"""
    # Real representation unsqueezes the last dimension
    if dim < 0:
        dim = dim - 1
    return aten_cat(op, tensors, dim=dim)


@onnx_impl((aten.cat.default, aten.concat.default, aten.concatenate.default))
def aten_cat(
    op: TranslationContext, tensors: Sequence[ir.Value], dim: int = 0
) -> ir.Value:
    """cat(Tensor[] tensors, int dim=0) -> Tensor"""
    if isinstance(tensors, ir.Value):
        return op.ConcatFromSequence(
            tensors,
            axis=dim,
            output_types=[tensors.type.elem_type],
        )

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
        return op.Identity(empty)
    if len(filtered_tensors) == 1:
        return op.Identity(filtered_tensors[0])
    return op.Concat(*filtered_tensors, axis=dim)


@onnx_impl((operator.getitem, "aten::getitem"))
def aten_getitem(
    op: TranslationContext, self: Sequence[ir.Value], i: ir.Value
) -> ir.Value:
    if not isinstance(self, ir.Value):
        if not isinstance(i, int):
            self = op.SequenceConstruct(*self)
        else:
            return self[i]
    return op.SequenceAt(self, i, output_types=[self.type.elem_type])


@onnx_impl(aten.repeat_interleave.self_int)
def aten_repeat_interleave_self_int(
    op: TranslationContext,
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
    if dim is None or (self.shape is not None and len(self.shape) == 0):
        self = op.Reshape(self, [-1])
        dim = 0
        self_rank = 1
    else:
        self_rank = len(self.shape)

    pos_dim = (dim + self_rank) % self_rank
    unsqueezed = op.Unsqueeze(self, [pos_dim + 1])
    if isinstance(repeats, int):
        tiles = [1] * (self_rank + 1)
        tiles[pos_dim + 1] = repeats
        tile_repeat = op.Constant(value=ir.tensor(tiles, dtype=ir.DataType.INT64))
    else:
        # repeats is a symbolic tensor
        tile_repeat = op.Concat(
            op.Constant(
                value=ir.tensor([1] * (pos_dim + 1), dtype=ir.DataType.INT64),
            ),
            op.Reshape(
                repeats,
                op.Constant(value=ir.tensor([-1], dtype=ir.DataType.INT64)),
            ),
            op.Constant(
                value=ir.tensor(
                    [1] * (self_rank - pos_dim - 1), dtype=ir.DataType.INT64
                ),
            ),
            axis=0,
        )
    tiled = op.Tile(unsqueezed, tile_repeat)
    repeated_dim = op.Mul(op.Shape(self, start=pos_dim, end=pos_dim + 1), repeats)
    final_shape = op.Concat(
        op.Shape(self, start=0, end=pos_dim),
        repeated_dim,
        op.Shape(self, start=pos_dim + 1),
        axis=0,
    )
    return op.Reshape(tiled, final_shape, allowzero=1)


@onnx_impl(aten.repeat_interleave.Tensor)
def aten_repeat_interleave_Tensor(
    op: TranslationContext,
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
    if repeats is None:
        repeats = self
        indices_only = True
    else:
        indices_only = False
    if not indices_only and (
        dim is None or (self.shape is not None and len(self.shape) == 0)
    ):
        # flatten
        self = op.Reshape(self, [-1])
        dim = 0
    repeats = op.Cast(repeats, to=ir.DataType.INT64)
    if not indices_only:
        # Expand a scalar or one-element vector to the selected axis length.
        rank = 1 if dim == 0 and self.shape is None else len(self.shape)
        axis = dim % rank
        repeats = op.Expand(repeats, op.Shape(self, start=axis, end=axis + 1))
    ci = op.CumSum(repeats, 0)
    total = op.ReduceSum(repeats, keepdims=0)
    trange = op.Range(0, total, 1)
    rows = op.Less(trange, op.Unsqueeze(ci, [-1]))
    srows = op.Sub(
        op.Size(repeats),
        op.ReduceSum(
            op.Cast(rows, to=ir.DataType.INT64),
            [0],
            keepdims=0,
        ),
    )
    indices = op.Reshape(srows, [-1])
    if indices_only:
        return indices
    return op.Gather(self, indices, axis=dim)


@onnx_impl((aten.split.default, aten.split.Tensor))
def aten_split(
    op: TranslationContext, self: ir.Value, split_size: ir.Value, dim: int = 0
) -> ir.Value:
    """split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]"""

    return op.SplitToSequence(
        self,
        split_size,
        axis=dim,
        output_types=[ir.SequenceType(self.type)],
    )


@onnx_impl(aten.split_with_sizes.default)
def aten_split_with_sizes(
    op: TranslationContext, self: ir.Value, split_sizes: ir.Value, dim: int = 0
) -> ir.Value:
    """split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]"""

    return op.SplitToSequence(
        self,
        common_ops.merge_dims(op, split_sizes),
        axis=dim,
        output_types=[ir.SequenceType(self.type)],
    )


@onnx_impl(aten.unsafe_split.Tensor)
def aten_unsafe_split(
    op: TranslationContext, self: ir.Value, split_size: ir.Value, dim: int = 0
) -> ir.Value:
    """unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]"""
    return aten_split(op, self, split_size, dim)


@onnx_impl(aten.chunk.default)
def aten_chunk(
    op: TranslationContext, self: ir.Value, chunks: int, dim: int = 0
) -> ir.Value:
    """chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]"""
    if chunks <= 0:
        raise ValueError(
            f"chunk requires a positive number of chunks, got {chunks}; pass chunks > 0."
        )
    # This will create a Sequence of tensors.
    # Get size of specified dim.
    dim_size = op.Gather(op.Shape(self), dim, axis=0)
    # Compute size/chunk to get the number of data in one chunk.
    num_per_chunk = op.Div(op.Add(dim_size, chunks - 1), chunks)
    num_per_chunk = op.Max(num_per_chunk, 1)
    # Compute real chunk number, preserving all empty chunks when dim_size is zero.
    count = op.Div(
        op.Add(dim_size, op.Sub(num_per_chunk, 1)),
        num_per_chunk,
    )
    count = op.Where(op.Equal(dim_size, 0), chunks, count)
    positions = op.Range(0, count, 1)
    # Append the remainder to the [n, n, n, n, ..., r].
    sizes = op.Min(
        num_per_chunk,
        op.Max(
            op.Sub(dim_size, op.Mul(positions, num_per_chunk)),
            0,
        ),
    )
    return aten_split_with_sizes(op, self, sizes, dim)


@onnx_impl(aten.stack.default, complex=True)
def aten_stack_complex(
    op: TranslationContext, tensors: Sequence[ir.Value], dim: int = 0
) -> ir.Value:
    """stack(Tensor[] tensors, int dim=0) -> Tensor"""
    # Real representation unsqueezes the last dimension
    if dim < 0:
        dim = dim - 1
    return aten_stack(op, tensors, dim)


@onnx_impl(aten.stack.default)
def aten_stack(
    op: TranslationContext, tensors: Sequence[ir.Value], dim: int = 0
) -> ir.Value:
    """stack(Tensor[] tensors, int dim=0) -> Tensor"""
    builder = op.builder
    if isinstance(tensors, Sequence):
        unsqueezed = [
            op.Unsqueeze(t, builder.constant([dim], dtype=ir.DataType.INT64))
            for t in tensors
        ]
        return op.Concat(*unsqueezed, axis=dim)
    return op.ConcatFromSequence(
        tensors,
        axis=dim,
        new_axis=1,
        output_types=[tensors.type.elem_type],
    )


@onnx_impl(aten.tile.default)
def aten_tile(op: TranslationContext, self: ir.Value, dims: Sequence[int]) -> ir.Value:
    """tile(Tensor self, int[] dims) -> Tensor"""
    builder = op.builder

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
        self_shape = op.Shape(self)
        self_final_shape = op.Concat(exapnd_ones, self_shape, axis=0)
        self = op.Reshape(self, self_final_shape, allowzero=True)

    return op.Tile(self, common_ops.merge_dims(op, dims))


@onnx_impl(aten.unbind.int)
def aten_unbind(
    op: TranslationContext, self: ir.Value, dim: int = 0
) -> Sequence[ir.Value]:
    """unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]"""
    builder = op.builder

    if isinstance(self.shape[dim], int):
        num_outputs = self.shape[dim]
        results = []
        for i in range(num_outputs):
            # Slice to get a single element at position i along dim
            sliced = op.Slice(
                self,
                builder.constant([i], dtype=ir.DataType.INT64),
                builder.constant([i + 1], dtype=ir.DataType.INT64),
                builder.constant([dim], dtype=ir.DataType.INT64),
            )
            # Squeeze to remove the dimension of size 1
            squeezed = op.Squeeze(sliced, [dim])
            results.append(squeezed)
        return results

    return op.SplitToSequence(
        self,
        axis=dim,
        keepdims=False,
        output_types=[ir.SequenceType(self.type)],
    )
