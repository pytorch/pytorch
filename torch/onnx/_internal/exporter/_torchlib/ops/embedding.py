# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR embedding translations."""

from __future__ import annotations

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder, GraphBuilder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


aten = torch.ops.aten


@onnx_impl(aten.embedding.default)
def aten_embedding(
    weight: ir.Value,
    indices: ir.Value,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> ir.Value:
    builder = current_builder()
    # embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor

    return builder.call("Gather", weight, indices)[0]


@onnx_impl(aten.embedding_renorm.default)
def aten_embedding_renorm(
    weight: ir.Value, indices: ir.Value, max_norm: float, norm_type: float = 2.0
) -> ir.Value:
    """embedding_renorm(Tensor weight, Tensor indices, float max_norm, float norm_type) -> Tensor"""
    builder = current_builder()

    unique_indices, _, _, _ = builder.call("Unique", indices, _num_outputs=4)
    partial_weight = builder.call("Gather", weight, unique_indices)[0]
    # partial_weight_norm = sum(|w|^p)^(1/p)
    if norm_type == 1.0:
        # This is not necessary, but op.ReduceL1 is faster than function list in 'else'
        partial_weight_norm = builder.call(
            "ReduceL1", partial_weight, [1], keepdims=True
        )[0]
    elif norm_type == 2.0:
        # This is not necessary, but op.ReduceL2 is faster than function list in 'else'
        partial_weight_norm = builder.call(
            "ReduceL2", partial_weight, [1], keepdims=True
        )[0]
    else:
        # Abs -> Pow -> ReduceSum -> Pow -> Pow
        partial_weight_abs = builder.call("Abs", partial_weight)[0]
        partial_weight_pow = builder.call(
            "Pow",
            partial_weight_abs,
            builder.constant(norm_type, dtype=ir.DataType.FLOAT),
        )[0]
        partial_weight_norm = builder.call(
            "ReduceSum", partial_weight_pow, [1], keepdims=True
        )[0]
        pow_value = builder.call("CastLike", 1.0 / norm_type, weight)[0]
        partial_weight_norm = builder.call("Pow", partial_weight_norm, pow_value)[0]

    max_norm = builder.call(
        "CastLike", builder.constant(max_norm, dtype=ir.DataType.FLOAT), weight
    )[0]
    # This is to avoid weight is zero
    err = builder.call(
        "CastLike", builder.constant(1e-7, dtype=ir.DataType.FLOAT), weight
    )[0]
    partial_weight_norm_ = builder.call("Add", partial_weight_norm, err)[0]
    scales = builder.call("Div", max_norm, partial_weight_norm_)[0]
    partial_weight_renorm = builder.call("Mul", partial_weight, scales)[0]
    # Set values to renormed values where weight_norm > max_norm, but keep the original values where weight_norm <= max_norm
    partial_weight_renorm = builder.call(
        "Where",
        builder.call("Greater", partial_weight_norm, max_norm)[0],
        partial_weight_renorm,
        partial_weight,
    )[0]
    value = builder.call(
        "ScatterND",
        weight,
        builder.call("Unsqueeze", unique_indices, [1])[0],
        partial_weight_renorm,
    )[0]
    return value


@onnx_impl(aten.embedding_bag.default)
def aten_embedding_bag(
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
    builder = current_builder()
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
    indices_1d = builder.call("Reshape", indices, neg_1)[0]
    indices_size = builder.call("Shape", indices_1d)[0]
    if per_sample_weights is None:
        # Set per_sample_weights to 1.0, because cannot check 'None' in ONNX-Script
        # Size of persample_weights is the same as indices, and should be 1d tensor
        per_sample_weights = builder.call("Expand", 1, indices_size)[0]
        # Dtype of per_sample_weights is the same as weight
        per_sample_weights = builder.call("CastLike", per_sample_weights, weight)[0]
    else:
        per_sample_weights = builder.call("Reshape", per_sample_weights, neg_1)[0]
    # Get weight out according to indices_1d,
    # Get weight out according to indices,
    # e.g. indices=[3,1,4,5,3] means get weight[[3,1,4,5,3]]
    new_weight = builder.call("Gather", weight, indices_1d)[0]
    # This happens after first step of Gather. Because Shape(indices)==Shape(per_sample_weights)
    new_weight = builder.call(
        "Mul", new_weight, builder.call("Unsqueeze", per_sample_weights, [1])[0]
    )[0]
    weight_dim_1 = builder.call("Shape", weight, start=1, end=2)[0]
    normalized_indices = indices_1d
    # When padding_idx is None, use the standard embedding_bag implementation
    if padding_idx is not None:
        num_embeddings = builder.call("Shape", weight, start=0, end=1)[
            0
        ]  # Get number of rows in weight
        num_embeddings_scalar = builder.call("Squeeze", num_embeddings)[0]
        if padding_idx < 0:
            padding_idx = builder.call("Add", padding_idx, num_embeddings_scalar)[0]
        normalized_indices = builder.call(
            "Where",
            builder.call("Less", indices_1d, 0)[0],
            builder.call("Add", indices_1d, num_embeddings_scalar)[0],
            indices_1d,
        )[0]

    # Assume indices is shape(5,2) reshape to (10,), offsets=[0,2,3], include_last_offset = False
    # [0,2,3] -> [0:2], [2:3], [3:10]
    num_bag = builder.call("Size", offsets)[0]  # 3 bags, means 10 is the last index
    if include_last_offset:
        num_bag = builder.call("Sub", num_bag, 1)[
            0
        ]  # 2 bags, means 3 is the last index
        if padding_idx is not None:
            # Change(by ScatterElement setting) the last element to 'end'
            # [0,2,3] -> [0,2,end]
            offsets = builder.call("ScatterElements", offsets, [-1], indices_size)[0]
    else:
        # Change [0,2,3] -> [0,2,3,end], means [0:2],[2:3],[3:end]
        offsets = builder.call("Concat", offsets, indices_size, axis=0)[
            0
        ]  # Replace end with number
    # The element in sequence must be FLOAT32 dtype due to ORT bug
    new_weight = builder.call("Cast", new_weight, to=ir.DataType.FLOAT)[0]
    # FIXME: https://github.com/microsoft/onnxruntime/issues/16846
    float_sequence = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
    int_sequence = ir.SequenceType(ir.TensorType(ir.DataType.INT64))
    result = builder.call("SequenceEmpty", dtype=ir.DataType.FLOAT)[0]
    sizes = builder.call("SequenceEmpty", dtype=ir.DataType.INT64)[0]
    maxima = builder.call("SequenceEmpty", dtype=ir.DataType.INT64)[0]
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
    # Process each bag
    start_pos = local.call("Gather", offsets, i)[0]
    end_pos = local.call("Gather", offsets, local.call("Add", i, 1)[0])[0]
    last_bag = local.call("Equal", i, local.call("Sub", num_bag, 1)[0])[0]
    range_end = end_pos
    if mode == 2 and padding_idx is None:
        # max
        # The last bag
        range_end = local.call(
            "Where", last_bag, local.call("Size", indices_1d)[0], end_pos
        )[0]
    positions = local.call("Range", start_pos, range_end, 1)[0]
    # empty tensor
    curr_indices = local.call("Gather", normalized_indices, positions)[0]
    if padding_idx is not None:
        # Call the existing function for handling padding_idx
        # Something like the 'append' operation: keep only positions not equal to padding_idx.
        keep = local.call("Not", local.call("Equal", curr_indices, padding_idx)[0])[0]
        positions = local.call("Compress", positions, keep, axis=0)[0]
        curr_indices = local.call("Compress", curr_indices, keep, axis=0)[0]
    row_weight = local.call("Gather", new_weight, positions)[0]
    count = local.call("Size", positions)[0]
    empty = local.call("Equal", count, 0)[0]
    if mode == 2 and padding_idx is None:
        empty = local.call("Equal", start_pos, end_pos)[0]
    expand_shape = local.call("Concat", [1], weight_dim_1, axis=0)[0]
    then_branch = ir.Graph([], [], nodes=[], name="embedding_bag_empty")
    sub = GraphBuilder(then_branch, opset=builder.opset)
    # Empty input get zero value output, not empty output
    # row_result should be 0, need to generate (1,N) shape tensor with 0 values
    zero_row = sub.call(
        "Expand", sub.constant([0.0], dtype=ir.DataType.FLOAT), expand_shape
    )[0]
    zero_max = sub.call(
        "Expand", sub.constant([0], dtype=ir.DataType.INT64), expand_shape
    )[0]
    common_ops.finish_subgraph(then_branch, [zero_row, zero_max])
    else_branch = ir.Graph([], [], nodes=[], name="embedding_bag_nonempty")
    sub = GraphBuilder(else_branch, opset=builder.opset)
    if mode == 0:  # sum
        row_result = sub.call("ReduceSum", row_weight, [0])[0]
    elif mode == 1:  # mean
        if padding_idx is None:
            # The last bag
            row_result = sub.call("ReduceSum", row_weight, [0])[0]
            # When include_last_offset=False, offsets=[0,2,3] -> [0,2,3,10], denominator=10-3=7
            # When include_last_offset=True, offsets=[0,2,3], denominator=10-2=8
            denominator = sub.call(
                "Sub",
                sub.call("Gather", sub.call("Shape", indices)[0], 0)[0],
                start_pos,
            )[0]
            denominator = sub.call("Where", last_bag, denominator, count)[0]
            denominator = sub.call(
                "CastLike", sub.call("Max", denominator, 1)[0], row_weight
            )[0]
            row_result = sub.call("Div", row_result, denominator)[0]
        else:
            row_result = sub.call("ReduceMean", row_weight, [0])[0]
    else:  # mode == 2, max
        # max
        row_result = sub.call("ReduceMax", row_weight, [0])[0]
    if mode == 2:
        argmax = sub.call("ArgMax", row_weight, axis=0, keepdims=0)[0]
        max_indices = sub.call(
            "Unsqueeze", sub.call("Gather", curr_indices, argmax)[0], [0]
        )[0]
    else:
        max_indices = sub.call(
            "Expand", sub.constant([0], dtype=ir.DataType.INT64), expand_shape
        )[0]
    common_ops.finish_subgraph(else_branch, [row_result, max_indices])
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
    rows = local.call("SequenceInsert", rows, row_result)[0]
    lengths = local.call(
        "SequenceInsert", lengths, local.call("Reshape", count, [1])[0]
    )[0]
    max_rows = local.call("SequenceInsert", max_rows, max_indices)[0]
    common_ops.finish_subgraph(body, [condition, rows, lengths, max_rows])
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
    empty_shape = builder.call("Concat", [0], weight_dim_1, axis=0)[0]
    empty_row = builder.call(
        "Expand", builder.constant([0.0], dtype=ir.DataType.FLOAT), empty_shape
    )[0]
    result = builder.call("SequenceInsert", result, empty_row)[0]
    result = builder.call(
        "ConcatFromSequence",
        result,
        axis=0,
        _output_types=[ir.TensorType(ir.DataType.FLOAT)],
    )[0]
    result = builder.call("CastLike", result, weight)[0]
    sizes = builder.call(
        "SequenceInsert", sizes, builder.constant([], dtype=ir.DataType.INT64)
    )[0]
    bag_size = builder.call(
        "ConcatFromSequence",
        sizes,
        axis=0,
        _output_types=[ir.TensorType(ir.DataType.INT64)],
    )[0]
    maxima = builder.call(
        "SequenceInsert",
        maxima,
        builder.call(
            "Expand", builder.constant([0], dtype=ir.DataType.INT64), empty_shape
        )[0],
    )[0]
    max_indices = builder.call(
        "ConcatFromSequence",
        maxima,
        axis=0,
        _output_types=[ir.TensorType(ir.DataType.INT64)],
    )[0]
    # Only compute the shape of other 3 outputs, we don't care the value
    # The native path also supplies the auxiliary values for consumers that use them.
    if mode == 0 and padding_idx is None:
        offset2bag = builder.constant([], dtype=ir.DataType.INT64)
    else:
        indices_length = builder.call("Gather", builder.call("Shape", indices)[0], 0)[0]
        positions = builder.call("Range", 0, indices_length, 1)[0]
        starts = builder.call(
            "Slice", offsets, [0], builder.call("Reshape", num_bag, [1])[0]
        )[0]
        membership = builder.call(
            "GreaterOrEqual",
            builder.call("Unsqueeze", positions, [1])[0],
            builder.call("Unsqueeze", starts, [0])[0],
        )[0]
        offset2bag = builder.call(
            "Sub",
            builder.call(
                "ReduceSum",
                builder.call("Cast", membership, to=ir.DataType.INT64)[0],
                [1],
                keepdims=0,
            )[0],
            1,
        )[0]
    if mode == 0:  # sum
        bag_size = builder.call(
            "Expand",
            builder.constant(0, dtype=ir.DataType.INT64),
            builder.call("Shape", offsets)[0],
        )[0]
        if not include_last_offset:
            bag_size = builder.call("Slice", bag_size, [0], [-1])[0]
        max_indices = builder.call("Identity", bag_size)[0]
    elif mode == 1:  # mean
        max_indices = builder.call("Identity", bag_size)[0]
    # shape = (bag_size.dim[0], weight.dim[1])
    return result, offset2bag, bag_size, max_indices
