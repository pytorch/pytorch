"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import numpy
import onnx

import torch
from torch._subclasses import fake_tensor

if TYPE_CHECKING:
    import onnx.defs.OpSchema.AttrType  # type: ignore[import]


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


def is_torch_complex_dtype(tensor_dtype: torch.dtype) -> bool:
    # NOTE: This is needed as TorchScriptTensor is nor supported by torch.is_complex()
    return tensor_dtype in _COMPLEX_TO_FLOAT


def from_complex_to_float(dtype: torch.dtype) -> torch.dtype:
    return _COMPLEX_TO_FLOAT[dtype]


def from_sym_value_to_torch_dtype(sym_value: SYM_VALUE_TYPE) -> torch.dtype:
    return _SYM_TYPE_TO_TORCH_DTYPE[type(sym_value)]


def is_optional_onnx_dtype_str(onnx_type_str: str) -> bool:
    return onnx_type_str in _OPTIONAL_ONNX_DTYPE_STR


def from_torch_dtype_to_onnx_dtype_str(dtype: Union[torch.dtype, type]) -> Set[str]:
    return _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[dtype]


def from_python_type_to_onnx_attribute_type(
    dtype: type, is_sequence: bool = False
) -> Optional[onnx.defs.OpSchema.AttrType]:
    import onnx.defs  # type: ignore[import]

    _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {
        float: onnx.defs.OpSchema.AttrType.FLOAT,
        int: onnx.defs.OpSchema.AttrType.INT,
        str: onnx.defs.OpSchema.AttrType.STRING,
        bool: onnx.defs.OpSchema.AttrType.INT,
    }

    _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {
        float: onnx.defs.OpSchema.AttrType.FLOATS,
        int: onnx.defs.OpSchema.AttrType.INTS,
        str: onnx.defs.OpSchema.AttrType.STRINGS,
        bool: onnx.defs.OpSchema.AttrType.INTS,
    }

    if is_sequence:
        return _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)
    return _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)


def from_python_type_to_onnx_tensor_element_type(type: type):
    """
    Converts a Python type to the corresponding ONNX tensor element type.
    For example, `from_python_type_to_onnx_tensor_element_type(float)` returns
    `onnx.TensorProto.FLOAT`.

    Args:
      type (type): The Python type to convert.

    Returns:
      int: The corresponding ONNX tensor element type.

    """
    _PYTHON_TYPE_TO_ONNX_TENSOR_ELEMENT_TYPE = {
        float: onnx.TensorProto.FLOAT,  # type: ignore[attr-defined]
        int: onnx.TensorProto.INT64,  # type: ignore[attr-defined]
        bool: onnx.TensorProto.BOOL,  # type: ignore[attr-defined]
    }
    return _PYTHON_TYPE_TO_ONNX_TENSOR_ELEMENT_TYPE.get(type)


def is_torch_symbolic_type(value: Any) -> bool:
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))


def from_torch_dtype_to_abbr(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return ""
    return _TORCH_DTYPE_TO_ABBREVIATION.get(dtype, "")


def from_scalar_type_to_torch_dtype(scalar_type: type) -> Optional[torch.dtype]:
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)


# NOTE: this is a mapping from torch dtype to a set of compatible onnx types
# It's used in dispatcher to find the best match overload for the input dtypes
_TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS: Dict[
    Union[torch.dtype, type], Set[str]
] = {
    torch.bfloat16: {"tensor(bfloat16)"},
    torch.bool: {"tensor(bool)"},
    torch.float64: {"tensor(double)"},
    torch.float32: {"tensor(float)"},
    torch.float16: {"tensor(float16)"},
    torch.float8_e4m3fn: {"tensor(float8_e4m3fn)"},
    torch.float8_e4m3fnuz: {"tensor(float8_e4m3fnuz)"},
    torch.float8_e5m2: {"tensor(float8_e5m2)"},
    torch.float8_e5m2fnuz: {"tensor(float8_e5m2fnuz)"},
    torch.int16: {"tensor(int16)"},
    torch.int32: {"tensor(int32)"},
    torch.int64: {"tensor(int64)"},
    torch.int8: {"tensor(int8)"},
    torch.uint8: {"tensor(uint8)"},
    str: {"tensor(string)"},
    int: {"tensor(int16)", "tensor(int32)", "tensor(int64)"},
    float: {"tensor(float16)", "tensor(float)", "tensor(double)"},
    bool: {"tensor(int32)", "tensor(int64)", "tensor(bool)"},
    complex: {"tensor(float)", "tensor(double)"},
    torch.complex32: {"tensor(float16)"},
    torch.complex64: {"tensor(float)"},
    torch.complex128: {"tensor(double)"},
}

_OPTIONAL_ONNX_DTYPE_STR: Set[str] = {
    f"optional({value})"
    for value_set in _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS.values()
    for value in value_set
}

_PYTHON_TYPE_TO_TORCH_DTYPE = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex64,
}

_COMPLEX_TO_FLOAT: Dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,  # NOTE: ORT doesn't support torch.float64
}

_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

_SCALAR_TYPE_TO_TORCH_DTYPE: Dict[Type, torch.dtype] = {
    **_PYTHON_TYPE_TO_TORCH_DTYPE,
    **_SYM_TYPE_TO_TORCH_DTYPE,
}

_TORCH_DTYPE_TO_ABBREVIATION = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.float8_e4m3fn: "e4m3fn",
    torch.float8_e4m3fnuz: "e4m3fnuz",
    torch.float8_e5m2: "f8e5m2",
    torch.float8_e5m2fnuz: "e5m2fnuz",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

_TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: numpy.float16,
    torch.float32: numpy.float32,
    torch.float64: numpy.float64,
    torch.uint8: numpy.uint8,
    torch.int8: numpy.int8,
    torch.int16: numpy.int16,
    torch.int32: numpy.int32,
    torch.int64: numpy.longlong,
    torch.bool: numpy.bool_,
}

_ONNX_TENSOR_ELEMENT_TYPE_TO_TORCH_DTYPE = {
    onnx.TensorProto.FLOAT: torch.float32,  # type: ignore[attr-defined]
    onnx.TensorProto.FLOAT16: torch.float16,  # type: ignore[attr-defined]
    onnx.TensorProto.FLOAT8E5M2: torch.float8_e5m2,  # type: ignore[attr-defined]
    onnx.TensorProto.FLOAT8E5M2FNUZ: torch.float8_e5m2fnuz,  # type: ignore[attr-defined]
    onnx.TensorProto.FLOAT8E4M3FN: torch.float8_e4m3fn,  # type: ignore[attr-defined]
    onnx.TensorProto.FLOAT8E4M3FNUZ: torch.float8_e4m3fnuz,  # type: ignore[attr-defined]
    onnx.TensorProto.DOUBLE: torch.float64,  # type: ignore[attr-defined]
    onnx.TensorProto.BOOL: torch.bool,  # type: ignore[attr-defined]
    onnx.TensorProto.UINT8: torch.uint8,  # type: ignore[attr-defined]
    onnx.TensorProto.INT8: torch.int8,  # type: ignore[attr-defined]
    onnx.TensorProto.INT16: torch.int16,  # type: ignore[attr-defined]
    onnx.TensorProto.INT32: torch.int32,  # type: ignore[attr-defined]
    onnx.TensorProto.INT64: torch.int64,  # type: ignore[attr-defined]
}

_TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE = {
    value: key for key, value in _ONNX_TENSOR_ELEMENT_TYPE_TO_TORCH_DTYPE.items()
}

SYM_VALUE_TYPE = Union[torch.SymInt, torch.SymFloat, torch.SymBool]
META_VALUE_TYPE = Union[fake_tensor.FakeTensor, SYM_VALUE_TYPE, int, float, bool]
# NOTE: Belows are from torch/fx/node.py
BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    torch._ops.OpOverload,
]
Argument = Optional[
    Union[
        Tuple[Any, ...],  # actually Argument, but mypy can't represent recursive types
        List[Any],  # actually Argument
        Dict[str, Any],  # actually Argument
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        range,
        "torch.fx.Node",
        BaseArgumentTypes,
    ]
]
