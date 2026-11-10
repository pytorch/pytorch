# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR embedding translations."""

from __future__ import annotations

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import GraphBuilder, TranslationContext
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


aten = torch.ops.aten


@onnx_impl(aten.embedding.default)
def aten_embedding(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> ir.Value:
    # embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor

    return op.Gather(weight, indices)


@onnx_impl(aten.embedding_renorm.default)
def aten_embedding_renorm(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    max_norm: float,
    norm_type: float = 2.0,
) -> ir.Value:
    """embedding_renorm(Tensor weight, Tensor indices, float max_norm, float norm_type) -> Tensor"""
    builder = op.builder

    unique_indices, _, _, _ = op.Unique(indices, outputs=4)
    partial_weight = op.Gather(weight, unique_indices)
    # partial_weight_norm = sum(|w|^p)^(1/p)
    if norm_type == 1.0:
        # This is not necessary, but op.ReduceL1 is faster than function list in 'else'
        partial_weight_norm = op.ReduceL1(partial_weight, [1], keepdims=True)
    elif norm_type == 2.0:
        # This is not necessary, but op.ReduceL2 is faster than function list in 'else'
        partial_weight_norm = op.ReduceL2(partial_weight, [1], keepdims=True)
    else:
        # Abs -> Pow -> ReduceSum -> Pow -> Pow
        partial_weight_abs = op.Abs(partial_weight)
        partial_weight_pow = op.Pow(
            partial_weight_abs,
            builder.constant(norm_type, dtype=ir.DataType.FLOAT),
        )
        partial_weight_norm = op.ReduceSum(partial_weight_pow, [1], keepdims=True)
        pow_value = op.CastLike(1.0 / norm_type, weight)
        partial_weight_norm = op.Pow(partial_weight_norm, pow_value)

    max_norm = op.CastLike(builder.constant(max_norm, dtype=ir.DataType.FLOAT), weight)
    # This is to avoid weight is zero
    err = op.CastLike(builder.constant(1e-7, dtype=ir.DataType.FLOAT), weight)
    partial_weight_norm_ = op.Add(partial_weight_norm, err)
    scales = op.Div(max_norm, partial_weight_norm_)
    partial_weight_renorm = op.Mul(partial_weight, scales)
    # Set values to renormed values where weight_norm > max_norm, but keep the original values where weight_norm <= max_norm
    partial_weight_renorm = op.Where(
        op.Greater(partial_weight_norm, max_norm),
        partial_weight_renorm,
        partial_weight,
    )
    value = op.ScatterND(
        weight,
        op.Unsqueeze(unique_indices, [1]),
        partial_weight_renorm,
    )
    return value


@onnx_impl(aten.embedding_bag.default)
def aten_embedding_bag(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    offsets: ir.Value,
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,
    per_sample_weights: ir.Value | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    """embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)"""
    return aten_embedding_bag_padding_idx(
        op,
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


@onnx_impl(
    (
        aten.embedding_bag.padding_idx,
        aten._embedding_bag.default,
        aten._embedding_bag_forward_only.default,
    )
)
def aten_embedding_bag_padding_idx(
    op: TranslationContext,
    weight: ir.Value,
    indices: ir.Value,
    offsets: ir.Value,
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # [0,1,2] indicate ["sum", "mean", "max"]
    sparse: bool = False,
    per_sample_weights: ir.Value | None = None,
    include_last_offset: bool = False,
    padding_idx: int | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    """embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)

    We add default values for the attributes to accommodate _embedding_bag as well:
    _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1)
    """
    del scale_grad_by_freq, sparse
    builder = op.builder
    if mode not in (0, 1, 2):
        raise ValueError(
            f"embedding_bag mode must be 0 (sum), 1 (mean), or 2 (max), got {mode}."
        )
    if per_sample_weights is not None and mode != 0:
        raise ValueError("embedding_bag per_sample_weights requires mode=0 (sum).")
    # assert(rank(indices) in [1,2])
    # assert(rank(offsets) == 1)
    # assert(op.Size(per_sample_weights) == op.Size(indices))
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    # Assume indices is shape(5,2), indices_1d is shape(10,)
    indices_1d = op.Reshape(indices, neg_1)
    indices_size = op.Shape(indices_1d)
    if per_sample_weights is None:
        # Set per_sample_weights to 1.0, because cannot check 'None' in ONNX-Script
        # Size of persample_weights is the same as indices, and should be 1d tensor
        per_sample_weights = op.Expand(1, indices_size)
        # Dtype of per_sample_weights is the same as weight
        per_sample_weights = op.CastLike(per_sample_weights, weight)
    else:
        per_sample_weights = op.Reshape(per_sample_weights, neg_1)
    # Get weight out according to indices_1d,
    # Get weight out according to indices,
    # e.g. indices=[3,1,4,5,3] means get weight[[3,1,4,5,3]]
    new_weight = op.Gather(weight, indices_1d)
    # This happens after first step of Gather. Because Shape(indices)==Shape(per_sample_weights)
    new_weight = op.Mul(new_weight, op.Unsqueeze(per_sample_weights, [1]))
    weight_dim_1 = op.Shape(weight, start=1, end=2)
    normalized_indices = indices_1d
    # When padding_idx is None, use the standard embedding_bag implementation
    if padding_idx is not None:
        num_embeddings = op.Shape(
            weight, start=0, end=1
        )  # Get number of rows in weight
        num_embeddings_scalar = op.Squeeze(num_embeddings)
        if padding_idx < 0:
            padding_idx = op.Add(padding_idx, num_embeddings_scalar)
        normalized_indices = op.Where(
            op.Less(indices_1d, 0),
            op.Add(indices_1d, num_embeddings_scalar),
            indices_1d,
        )

    # Assume indices is shape(5,2) reshape to (10,), offsets=[0,2,3], include_last_offset = False
    # [0,2,3] -> [0:2], [2:3], [3:10]
    num_bag = op.Size(offsets)  # 3 bags, means 10 is the last index
    if include_last_offset:
        num_bag = op.Sub(num_bag, 1)  # 2 bags, means 3 is the last index
        if padding_idx is not None:
            # Change(by ScatterElement setting) the last element to 'end'
            # [0,2,3] -> [0,2,end]
            offsets = op.ScatterElements(offsets, [-1], indices_size)
    else:
        # Change [0,2,3] -> [0,2,3,end], means [0:2],[2:3],[3:end]
        offsets = op.Concat(offsets, indices_size, axis=0)  # Replace end with number
    # The element in sequence must be FLOAT32 dtype due to ORT bug
    new_weight = op.Cast(new_weight, to=ir.DataType.FLOAT)
    # FIXME: https://github.com/microsoft/onnxruntime/issues/16846
    float_sequence = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
    int_sequence = ir.SequenceType(ir.TensorType(ir.DataType.INT64))
    result = op.SequenceEmpty(dtype=ir.DataType.FLOAT)
    sizes = op.SequenceEmpty(dtype=ir.DataType.INT64)
    maxima = op.SequenceEmpty(dtype=ir.DataType.INT64)
    # Used for iterator
    i = ir.Value(
        name="bag_iteration", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
    )
    condition = ir.Value(
        name="bag_condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
    )
    rows = ir.Value(name="bag_rows", type=float_sequence)
    lengths = ir.Value(name="bag_lengths", type=int_sequence)
    max_rows = ir.Value(name="bag_maxima", type=int_sequence)
    body = ir.Graph(
        [i, condition, rows, lengths, max_rows], [], nodes=[], name="embedding_bag_body"
    )
    local = GraphBuilder(body, opset=builder.opset)
    local_op = TranslationContext(local)
    # Process each bag
    start_pos = local_op.Gather(offsets, i)
    end_pos = local_op.Gather(offsets, local_op.Add(i, 1))
    last_bag = local_op.Equal(i, local_op.Sub(num_bag, 1))
    range_end = end_pos
    if mode == 2 and padding_idx is None:
        # max
        # The last bag
        range_end = local_op.Where(last_bag, local_op.Size(indices_1d), end_pos)
    positions = local_op.Range(start_pos, range_end, 1)
    # empty tensor
    curr_indices = local_op.Gather(normalized_indices, positions)
    if padding_idx is not None:
        # Call the existing function for handling padding_idx
        # Something like the 'append' operation: keep only positions not equal to padding_idx.
        keep = local_op.Not(local_op.Equal(curr_indices, padding_idx))
        positions = local_op.Compress(positions, keep, axis=0)
        curr_indices = local_op.Compress(curr_indices, keep, axis=0)
    row_weight = local_op.Gather(new_weight, positions)
    count = local_op.Size(positions)
    empty = local_op.Equal(count, 0)
    if mode == 2 and padding_idx is None:
        empty = local_op.Equal(start_pos, end_pos)
    expand_shape = local_op.Concat([1], weight_dim_1, axis=0)
    then_branch = ir.Graph([], [], nodes=[], name="embedding_bag_empty")
    sub = GraphBuilder(then_branch, opset=builder.opset)
    sub_op = TranslationContext(sub)
    # Empty input get zero value output, not empty output
    # row_result should be 0, need to generate (1,N) shape tensor with 0 values
    zero_row = sub_op.Expand(sub.constant([0.0], dtype=ir.DataType.FLOAT), expand_shape)
    zero_max = sub_op.Expand(sub.constant([0], dtype=ir.DataType.INT64), expand_shape)
    common_ops.finish_subgraph(op, then_branch, [zero_row, zero_max])
    else_branch = ir.Graph([], [], nodes=[], name="embedding_bag_nonempty")
    sub = GraphBuilder(else_branch, opset=builder.opset)
    sub_op = TranslationContext(sub)
    if mode == 0:  # sum
        row_result = sub_op.ReduceSum(row_weight, [0])
    elif mode == 1:  # mean
        if padding_idx is None:
            # The last bag
            row_result = sub_op.ReduceSum(row_weight, [0])
            # When include_last_offset=False, offsets=[0,2,3] -> [0,2,3,10], denominator=10-3=7
            # When include_last_offset=True, offsets=[0,2,3], denominator=10-2=8
            denominator = sub_op.Sub(
                sub_op.Gather(sub_op.Shape(indices), 0),
                start_pos,
            )
            denominator = sub_op.Where(last_bag, denominator, count)
            denominator = sub_op.CastLike(sub_op.Max(denominator, 1), row_weight)
            row_result = sub_op.Div(row_result, denominator)
        else:
            row_result = sub_op.ReduceMean(row_weight, [0])
    else:  # mode == 2, max
        # max
        row_result = sub_op.ReduceMax(row_weight, [0])
    if mode == 2:
        argmax = sub_op.ArgMax(row_weight, axis=0, keepdims=0)
        max_indices = sub_op.Unsqueeze(sub_op.Gather(curr_indices, argmax), [0])
    else:
        max_indices = sub_op.Expand(
            sub.constant([0], dtype=ir.DataType.INT64), expand_shape
        )
    common_ops.finish_subgraph(op, else_branch, [row_result, max_indices])
    row_result, max_indices = local.node(
        "If",
        [empty],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[
            ir.TensorType(ir.DataType.FLOAT),
            ir.TensorType(ir.DataType.INT64),
        ],
    )
    # Something like the 'append' operation
    rows = local_op.SequenceInsert(rows, row_result)
    lengths = local_op.SequenceInsert(lengths, local_op.Reshape(count, [1]))
    max_rows = local_op.SequenceInsert(max_rows, max_indices)
    common_ops.finish_subgraph(op, body, [condition, rows, lengths, max_rows])
    result, sizes, maxima = builder.node(
        "Loop",
        [
            num_bag,
            builder.constant(True, dtype=ir.DataType.BOOL),
            result,
            sizes,
            maxima,
        ],
        attributes={"body": body},
        output_types=[float_sequence, int_sequence, int_sequence],
    )
    # Seed sequences after the loop so zero bags have a well-defined empty concatenation.
    empty_shape = op.Concat([0], weight_dim_1, axis=0)
    empty_row = op.Expand(builder.constant([0.0], dtype=ir.DataType.FLOAT), empty_shape)
    result = op.SequenceInsert(result, empty_row)
    result = op.ConcatFromSequence(
        result,
        axis=0,
        output_types=[ir.TensorType(ir.DataType.FLOAT)],
    )
    result = op.CastLike(result, weight)
    sizes = op.SequenceInsert(sizes, builder.constant([], dtype=ir.DataType.INT64))
    bag_size = op.ConcatFromSequence(
        sizes,
        axis=0,
        output_types=[ir.TensorType(ir.DataType.INT64)],
    )
    maxima = op.SequenceInsert(
        maxima,
        op.Expand(builder.constant([0], dtype=ir.DataType.INT64), empty_shape),
    )
    max_indices = op.ConcatFromSequence(
        maxima,
        axis=0,
        output_types=[ir.TensorType(ir.DataType.INT64)],
    )
    # Only compute the shape of other 3 outputs, we don't care the value
    # The native path also supplies the auxiliary values for consumers that use them.
    if mode == 0 and padding_idx is None:
        offset2bag = builder.constant([], dtype=ir.DataType.INT64)
    else:
        indices_length = op.Gather(op.Shape(indices), 0)
        positions = op.Range(0, indices_length, 1)
        starts = op.Slice(offsets, [0], op.Reshape(num_bag, [1]))
        membership = op.GreaterOrEqual(
            op.Unsqueeze(positions, [1]),
            op.Unsqueeze(starts, [0]),
        )
        offset2bag = op.Sub(
            op.ReduceSum(
                op.Cast(membership, to=ir.DataType.INT64),
                [1],
                keepdims=0,
            ),
            1,
        )
    if mode == 0:  # sum
        bag_size = op.Expand(
            builder.constant(0, dtype=ir.DataType.INT64),
            op.Shape(offsets),
        )
        if not include_last_offset:
            bag_size = op.Slice(bag_size, [0], [-1])
        max_indices = op.Identity(bag_size)
    elif mode == 1:  # mean
        max_indices = op.Identity(bag_size)
    # shape = (bag_size.dim[0], weight.dim[1])
    return result, offset2bag, bag_size, max_indices
