# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common helpers for PyTorch-owned ONNX IR translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx_ir as ir


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    import onnx

    from torch.onnx._internal.exporter._ir_builder import TranslationContext


def require_dtype(op: TranslationContext, value: ir.Value | None) -> ir.DataType:
    if value is None:
        raise ValueError(
            "ONNX translation requires a tensor value, not None. "
            "Handle omitted optional inputs before requesting their dtype."
        )
    dtype = value.dtype
    if dtype is None:
        raise ValueError(
            f"ONNX translation requires a tensor dtype for value {value.name!r}. "
            "Set ir.Value.type before lowering this operator."
        )
    return dtype


def require_shape(op: TranslationContext, value: ir.Value | None) -> ir.Shape:
    if value is None:
        raise ValueError(
            "ONNX translation requires a tensor value, not None. "
            "Handle omitted optional inputs before requesting their shape."
        )
    shape = value.shape
    if shape is None:
        raise ValueError(
            f"ONNX translation requires a known rank for value {value.name!r}. "
            "Set ir.Value.shape; symbolic dimension sizes are supported."
        )
    return shape


def require_type(op: TranslationContext, value: ir.Value) -> ir.TypeProtocol:
    if value.type is None:
        raise ValueError(
            f"ONNX translation requires a type for value {value.name!r}. "
            "Set ir.Value.type before constructing typed outputs."
        )
    return value.type


def require_sequence_type(op: TranslationContext, value: ir.Value) -> ir.SequenceType:
    value_type = require_type(op, value)
    if not isinstance(value_type, ir.SequenceType):
        raise TypeError(
            f"ONNX translation requires a sequence type for value {value.name!r}, "
            f"got {value_type}. Supply a Tensor[] value with ir.SequenceType."
        )
    return value_type


def cast_to(op: TranslationContext, a: ir.Value, dtype: int | ir.DataType) -> ir.Value:
    """Cast input to dtype while handling complex types."""
    builder = op.builder
    dtype = ir.DataType(dtype)
    if dtype in (ir.DataType.COMPLEX128, ir.DataType.COMPLEX64):
        # Cast to the real representation of the complex type
        real_dtype = (
            ir.DataType.DOUBLE if dtype == ir.DataType.COMPLEX128 else ir.DataType.FLOAT
        )
        casted = op.Cast(a, to=real_dtype)
        # Create a complex number
        real_part = op.Unsqueeze(casted, [-1])
        zero = builder.constant(0.0, dtype=real_dtype)
        imag_part = op.Expand(zero, op.Shape(real_part))
        return op.Concat(real_part, imag_part, axis=-1)
    # Cast to real numbers
    return op.Cast(a, to=dtype)


def constant(
    op: TranslationContext,
    array: npt.ArrayLike | onnx.TensorProto | ir.DLPackCompatible | ir.ArrayCompatible,
    dtype: int | ir.DataType,
) -> ir.Value:
    """Utility for creating a constant tensor.

    Args:
        array: The array to convert to a constant tensor.
        dtype: The data type of the tensor.

    Returns:
        A constant node.
    """
    return op.builder.constant(array, dtype=ir.DataType(dtype))


def merge_dims(
    op: TranslationContext, dims: Sequence[int | ir.Value] | ir.Value
) -> ir.Value:
    """Concatenate dimensions into a single value."""
    builder = op.builder
    if isinstance(dims, ir.Value):
        return op.Cast(dims, to=ir.DataType.INT64)
    if not dims:
        return builder.constant([], dtype=ir.DataType.INT64)

    neg_one_1d = builder.constant([-1], dtype=ir.DataType.INT64)
    result_dims = [
        builder.constant([d], dtype=ir.DataType.INT64)
        if isinstance(d, int)
        else op.Reshape(d, neg_one_1d)
        for d in dims
    ]
    # Symbolic dimensions must already be INT64 so Concat can be used.
    for dim in result_dims:
        if dim.dtype != ir.DataType.INT64:
            raise TypeError(
                f"Expected an INT64 symbolic dimension, got {dim.dtype}. "
                "Cast dimensions to INT64 before merging them."
            )
    return op.Concat(*result_dims, axis=0)


def complex_scalar(
    op: TranslationContext,
    value: ir.Value | complex | float | int | bool,
    *,
    dtype: ir.DataType | None,
) -> ir.Value:
    """Construct a Python scalar's explicit real/imaginary representation."""
    if isinstance(value, ir.Value):
        return value
    if dtype is None or not dtype.is_floating_point():
        raise TypeError(
            f"A complex scalar needs a known floating-point component dtype, got {dtype}. "
            "Set the complex tensor's real component type before translation."
        )
    components = [value.real, value.imag] if isinstance(value, complex) else [value, 0]
    return op.builder.constant(components, dtype=dtype)


def finish_subgraph(
    op: TranslationContext, graph: ir.Graph, outputs: Sequence[ir.Value]
) -> ir.Graph:
    """Set explicit outputs and avoid shadowing captured ancestor value names."""
    graph.outputs.extend(outputs)
    for index, node in enumerate(graph):
        for output_index, output in enumerate(node.outputs):
            output.name = f"{graph.name}_{index}_{output_index}"
    return graph
