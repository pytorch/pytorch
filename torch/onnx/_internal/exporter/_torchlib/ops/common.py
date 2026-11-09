# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common helpers for PyTorch-owned ONNX IR translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    import onnx


def cast_to(a: ir.Value, dtype: int | ir.DataType) -> ir.Value:
    """Cast input to dtype while handling complex types."""
    builder = current_builder()
    dtype = ir.DataType(dtype)
    if dtype in (ir.DataType.COMPLEX128, ir.DataType.COMPLEX64):
        # Cast to the real representation of the complex type
        real_dtype = (
            ir.DataType.DOUBLE if dtype == ir.DataType.COMPLEX128 else ir.DataType.FLOAT
        )
        casted = builder.call("Cast", a, to=real_dtype)[0]
        # Create a complex number
        real_part = builder.call("Unsqueeze", casted, [-1])[0]
        zero = builder.constant(0.0, dtype=real_dtype)
        imag_part = builder.call("Expand", zero, builder.call("Shape", real_part)[0])[0]
        return builder.call("Concat", real_part, imag_part, axis=-1)[0]
    # Cast to real numbers
    return builder.call("Cast", a, to=dtype)[0]


def constant(
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
    return current_builder().constant(array, dtype=ir.DataType(dtype))


def merge_dims(dims: Sequence[int | ir.Value] | ir.Value) -> ir.Value:
    """Concatenate dimensions into a single value."""
    builder = current_builder()
    if isinstance(dims, ir.Value):
        return builder.call("Cast", dims, to=ir.DataType.INT64)[0]
    if not dims:
        return builder.constant([], dtype=ir.DataType.INT64)

    neg_one_1d = builder.constant([-1], dtype=ir.DataType.INT64)
    result_dims = [
        builder.constant([d], dtype=ir.DataType.INT64)
        if isinstance(d, int)
        else builder.call("Reshape", d, neg_one_1d)[0]
        for d in dims
    ]
    # Symbolic dimensions must already be INT64 so Concat can be used.
    for dim in result_dims:
        if dim.dtype != ir.DataType.INT64:
            raise TypeError(
                f"Expected an INT64 symbolic dimension, got {dim.dtype}. "
                "Cast dimensions to INT64 before merging them."
            )
    return builder.call("Concat", *result_dims, axis=0)[0]


def complex_scalar(
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
    return current_builder().constant(components, dtype=dtype)


def finish_subgraph(graph: ir.Graph, outputs: Sequence[ir.Value]) -> ir.Graph:
    """Set explicit outputs and avoid shadowing captured ancestor value names."""
    graph.outputs.extend(outputs)
    for index, node in enumerate(graph):
        for output_index, output in enumerate(node.outputs):
            output.name = f"{graph.name}_{index}_{output_index}"
    return graph
