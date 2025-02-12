"""Common operators shared in the torchlib library."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa
from __future__ import annotations

import numpy.typing as npt

import onnx
import onnxscript
import onnxscript.values
from onnxscript import INT64, ir, opset18 as op
from onnxscript.onnx_types import COMPLEX128, COMPLEX64, DOUBLE, FLOAT, TensorType

from torch.onnx._internal.exporter import _constants
from torch.onnx._internal.exporter._torchlib._tensor_typing import RealType, TTensor


COMPLEX64_TYPE = COMPLEX64.dtype
COMPLEX128_TYPE = COMPLEX128.dtype

torch_opset = onnxscript.values.Opset(domain=_constants.TORCHLIB_DOMAIN, version=1)


@onnxscript.script(torch_opset)
def Rank(input: TTensor) -> INT64:
    """Take the rank of the input tensor."""

    return op.Size(op.Shape(input))


def cast_to(a: RealType, dtype: int) -> RealType:
    """Cast input to dtype while handling complex types."""

    # Traced function because different if branches return different dtypes
    # which is not supported in an ONNX function
    if dtype == COMPLEX128_TYPE:
        # Cast to the real representation of the complex type
        casted = op.Cast(a, to=DOUBLE.dtype)
        # Create a complex number
        real_part = op.Unsqueeze(casted, axes=[-1])
        imag_part = op.Expand(op.Cast(0.0, to=DOUBLE.dtype), op.Shape(real_part))
        result = op.Concat(real_part, imag_part, axis=-1)
    elif dtype == COMPLEX64_TYPE:
        # Cast to the real representation of the complex type
        casted = op.Cast(a, to=FLOAT.dtype)
        # Create a complex number
        real_part = op.Unsqueeze(casted, axes=[-1])
        imag_part = op.Expand(0.0, op.Shape(real_part))
        result = op.Concat(real_part, imag_part, axis=-1)
    else:
        # Cast to real numbers
        result = op.Cast(a, to=dtype)

    return result


def constant(
    array: npt.ArrayLike | onnx.TensorProto | ir.DLPackCompatible | ir.ArrayCompatible,
    dtype: int | onnx.TensorProto.DataType | ir.DataType,
) -> TensorType:
    """Utility for creating a constant tensor.

    Args:
        array: The array to convert to a constant tensor.
        dtype: The data type of the tensor.

    Returns:
        A constant node.
    """
    return op.Constant(value=ir.tensor(array, dtype=ir.DataType(dtype)))
