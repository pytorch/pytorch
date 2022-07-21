"""Utilities for converting and operating on ONNX, JIT and torch types."""

from typing import Dict

import torch
import enum
from torch._C import _onnx as _C_onnx

from typing_extensions import Literal
from typing import Union


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

class ScalarType(enum.Enum):
    """A human-readable name for a key into scalar_type_to_pytorch_type."""

    UINT8 = enum.auto()
    INT8 = enum.auto()
    INT16 = enum.auto()
    INT = enum.auto()
    INT64 = enum.auto()
    HALF = enum.auto()
    FLOAT = enum.auto()
    DOUBLE = enum.auto()
    COMPLEX32 = enum.auto()
    COMPLEX64 = enum.auto()
    COMPLEX128 = enum.auto()
    BOOL = enum.auto()
    QINT8 = enum.auto()
    QUINT8 = enum.auto()
    QINT32 = enum.auto()
    BFLOAT16 = enum.auto()
    UNDEFINED = enum.auto()

    

    def to_scalar_name(self) -> ScalarName:
        return _SCALAR_TYPE_TO_NAME[self]

    def to_torch_name(self) -> TorchName:
        return _SCALAR_TYPE_TO_TORCH_NAME[self]

    def to_dtype(self) -> torch.dtype:
        return _SCALAR_TYPE_TO_DTYPE[self]

    def onnx_compatible(self) -> bool:
        return self in SCALAR_TYPE_TO_ONNX


# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
_SCALAR_TYPE_TO_NAME: Dict[ScalarType, ScalarName] = {
    ScalarType.BOOL : "Bool",
    ScalarType.UINT8: "Byte",
    ScalarType.INT8 : "Char",
    ScalarType.INT16 : "Short",
    ScalarType.INT : "Int",
    ScalarType.INT64 : "Int64",
    ScalarType.HALF : "Half",
    ScalarType.FLOAT : "Float",
    ScalarType.DOUBLE : "Double",
    ScalarType.COMPLEX32 : "ComplexHalf",
    ScalarType.COMPLEX64 : "ComplexFloat",
    ScalarType.COMPLEX128 : "ComplexDouble",
    ScalarType.QINT8 : "QInt8",
    ScalarType.QUINT8 : "QUInt8",
    ScalarType.QINT32 : "QInt32",
    ScalarType.BFLOAT16 : "BFloat16",
    ScalarType.UNDEFINED: "Undefined",
}

_SCALAR_NAME_TO_TYPE: Dict[ScalarName, ScalarType] = dict(map(reversed, _SCALAR_TYPE_TO_NAME.items()))

_SCALAR_TYPE_TO_TORCH_NAME: Dict[ScalarType, TorchName] = {
    ScalarType.BOOL : "bool",
    ScalarType.UINT8: "uint8_t",
    ScalarType.INT8 : "int8_t",
    ScalarType.INT16 : "int16_t",
    ScalarType.INT : "int",
    ScalarType.INT64 : "int64_t",
    ScalarType.HALF : "half",
    ScalarType.FLOAT : "float",
    ScalarType.DOUBLE : "double",
    ScalarType.COMPLEX32 : "complex32",
    ScalarType.COMPLEX64 : "complex64",
    ScalarType.COMPLEX128 : "complex128",
    ScalarType.QINT8 : "qint8",
    ScalarType.QUINT8 : "quint8",
    ScalarType.QINT32 : "qint32",
    ScalarType.BFLOAT16 : "bfloat16",
}

TORCH_NAME_TO_SCALAR_TYPE: Dict[TorchName, ScalarType] = dict(map(reversed, _SCALAR_TYPE_TO_TORCH_NAME.items()))


SCALAR_TYPE_TO_ONNX = {
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
    ScalarType.QINT8 : torch.qint8,
    ScalarType.QUINT8 : torch.quint8,
    ScalarType.QINT32 : torch.qint32,
    ScalarType.BFLOAT16: torch.bfloat16,
}
