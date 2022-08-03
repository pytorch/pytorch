"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

import enum
from typing import Dict, Optional, Union

from typing_extensions import Literal

import torch
from torch._C import _onnx as _C_onnx

ScalarName = Literal[
    "Byte",
    "Char",
    "Double",
    "Float",
    "Half",
    "Int",
    "Long",
    "Short",
    "Bool",
    "ComplexHalf",
    "ComplexFloat",
    "ComplexDouble",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "Undefined",
]

TorchName = Literal[
    "bool",
    "uint8_t",
    "int8_t",
    "double",
    "float",
    "half",
    "int",
    "int64_t",
    "int16_t",
    "complex32",
    "complex64",
    "complex128",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
]


class ScalarType(enum.IntEnum):
    """Scalar types defined in torch."""

    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    UINT8 = 0
    INT8 = enum.auto()  # 1
    INT16 = enum.auto()  # 2
    INT = enum.auto()  # 3
    INT64 = enum.auto()  # 4
    HALF = enum.auto()  # 5
    FLOAT = enum.auto()  # 6
    DOUBLE = enum.auto()  # 7
    COMPLEX32 = enum.auto()  # 8
    COMPLEX64 = enum.auto()  # 9
    COMPLEX128 = enum.auto()  # 10
    BOOL = enum.auto()  # 11
    QINT8 = enum.auto()  # 12
    QUINT8 = enum.auto()  # 13
    QINT32 = enum.auto()  # 14
    BFLOAT16 = enum.auto()  # 15
    UNDEFINED = enum.auto()  # 16

    @classmethod
    def from_name(cls, name: Union[ScalarName, TorchName, Optional[str]]) -> ScalarType:
        """Convert a JIT scalar type or torch type name to ScalarType.

        Args:
            name: JIT scalar type name (Byte) or torch type name (uint8_t).

        Returns:
            ScalarType.

        Raises:
            ValueError: if name is not a valid scalar type name or if it is None.
        """
        if name is None:
            raise ValueError("Scalar type name cannot be None")
        if valid_scalar_name(name):
            return _SCALAR_NAME_TO_TYPE[name]  # type: ignore[index]
        if valid_torch_name(name):
            return _TORCH_NAME_TO_SCALAR_TYPE[name]  # type: ignore[index]

        raise ValueError(f"Unknown torch or scalar type: '{name}'")

    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> ScalarType:
        """Convert a torch dtype to ScalarType."""
        if dtype not in _DTYPE_TO_SCALAR_TYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        return _DTYPE_TO_SCALAR_TYPE[dtype]

    def scalar_name(self) -> ScalarName:
        """Convert a ScalarType to a JIT scalar type name."""
        return _SCALAR_TYPE_TO_NAME[self]

    def torch_name(self) -> TorchName:
        """Convert a ScalarType to a torch type name."""
        return _SCALAR_TYPE_TO_TORCH_NAME[self]

    def dtype(self) -> torch.dtype:
        """Convert a ScalarType to a torch dtype."""
        return _SCALAR_TYPE_TO_DTYPE[self]

    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a ScalarType to an ONNX data type."""
        if self not in _SCALAR_TYPE_TO_ONNX:
            raise ValueError(f"Scalar type {self} cannot be converted to ONNX")
        return _SCALAR_TYPE_TO_ONNX[self]

    def onnx_compatible(self) -> bool:
        """Return whether this ScalarType is compatible with ONNX."""
        return (
            self in _SCALAR_TYPE_TO_ONNX
            and self != ScalarType.UNDEFINED
            and self != ScalarType.COMPLEX32
        )


def valid_scalar_name(scalar_name: Union[ScalarName, str]) -> bool:
    """Return whether the given scalar name is a valid JIT scalar type name."""
    return scalar_name in _SCALAR_NAME_TO_TYPE


def valid_torch_name(torch_name: Union[TorchName, str]) -> bool:
    """Return whether the given torch name is a valid torch type name."""
    return torch_name in _TORCH_NAME_TO_SCALAR_TYPE


# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
_SCALAR_TYPE_TO_NAME: Dict[ScalarType, ScalarName] = {
    ScalarType.BOOL: "Bool",
    ScalarType.UINT8: "Byte",
    ScalarType.INT8: "Char",
    ScalarType.INT16: "Short",
    ScalarType.INT: "Int",
    ScalarType.INT64: "Long",
    ScalarType.HALF: "Half",
    ScalarType.FLOAT: "Float",
    ScalarType.DOUBLE: "Double",
    ScalarType.COMPLEX32: "ComplexHalf",
    ScalarType.COMPLEX64: "ComplexFloat",
    ScalarType.COMPLEX128: "ComplexDouble",
    ScalarType.QINT8: "QInt8",
    ScalarType.QUINT8: "QUInt8",
    ScalarType.QINT32: "QInt32",
    ScalarType.BFLOAT16: "BFloat16",
    ScalarType.UNDEFINED: "Undefined",
}

_SCALAR_NAME_TO_TYPE: Dict[ScalarName, ScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_NAME.items()
}

_SCALAR_TYPE_TO_TORCH_NAME: Dict[ScalarType, TorchName] = {
    ScalarType.BOOL: "bool",
    ScalarType.UINT8: "uint8_t",
    ScalarType.INT8: "int8_t",
    ScalarType.INT16: "int16_t",
    ScalarType.INT: "int",
    ScalarType.INT64: "int64_t",
    ScalarType.HALF: "half",
    ScalarType.FLOAT: "float",
    ScalarType.DOUBLE: "double",
    ScalarType.COMPLEX32: "complex32",
    ScalarType.COMPLEX64: "complex64",
    ScalarType.COMPLEX128: "complex128",
    ScalarType.QINT8: "qint8",
    ScalarType.QUINT8: "quint8",
    ScalarType.QINT32: "qint32",
    ScalarType.BFLOAT16: "bfloat16",
}

_TORCH_NAME_TO_SCALAR_TYPE: Dict[TorchName, ScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_TORCH_NAME.items()
}

_SCALAR_TYPE_TO_ONNX = {
    ScalarType.BOOL: _C_onnx.TensorProtoDataType.BOOL,
    ScalarType.UINT8: _C_onnx.TensorProtoDataType.UINT8,
    ScalarType.INT8: _C_onnx.TensorProtoDataType.INT8,
    ScalarType.INT16: _C_onnx.TensorProtoDataType.INT16,
    ScalarType.INT: _C_onnx.TensorProtoDataType.INT32,
    ScalarType.INT64: _C_onnx.TensorProtoDataType.INT64,
    ScalarType.HALF: _C_onnx.TensorProtoDataType.FLOAT16,
    ScalarType.FLOAT: _C_onnx.TensorProtoDataType.FLOAT,
    ScalarType.DOUBLE: _C_onnx.TensorProtoDataType.DOUBLE,
    ScalarType.COMPLEX64: _C_onnx.TensorProtoDataType.COMPLEX64,
    ScalarType.COMPLEX128: _C_onnx.TensorProtoDataType.COMPLEX128,
    ScalarType.BFLOAT16: _C_onnx.TensorProtoDataType.BFLOAT16,
    ScalarType.UNDEFINED: _C_onnx.TensorProtoDataType.UNDEFINED,
    ScalarType.COMPLEX32: _C_onnx.TensorProtoDataType.UNDEFINED,
    ScalarType.QINT8: _C_onnx.TensorProtoDataType.INT8,
    ScalarType.QUINT8: _C_onnx.TensorProtoDataType.UINT8,
    ScalarType.QINT32: _C_onnx.TensorProtoDataType.INT32,
}

# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
_SCALAR_TYPE_TO_DTYPE = {
    ScalarType.BOOL: torch.bool,
    ScalarType.UINT8: torch.uint8,
    ScalarType.INT8: torch.int8,
    ScalarType.INT16: torch.short,
    ScalarType.INT: torch.int,
    ScalarType.INT64: torch.int64,
    ScalarType.HALF: torch.half,
    ScalarType.FLOAT: torch.float,
    ScalarType.DOUBLE: torch.double,
    ScalarType.COMPLEX32: torch.complex32,
    ScalarType.COMPLEX64: torch.complex64,
    ScalarType.COMPLEX128: torch.complex128,
    ScalarType.QINT8: torch.qint8,
    ScalarType.QUINT8: torch.quint8,
    ScalarType.QINT32: torch.qint32,
    ScalarType.BFLOAT16: torch.bfloat16,
}

_DTYPE_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_DTYPE.items()}
