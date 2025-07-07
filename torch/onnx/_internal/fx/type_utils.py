# mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX, JIT and torch types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import Protocol, runtime_checkable

import onnx

import torch
from torch._subclasses import fake_tensor


if TYPE_CHECKING:
    import onnx.defs  # noqa: TCH004


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> torch.dtype | None: ...


def is_torch_complex_dtype(tensor_dtype: torch.dtype) -> bool:
    # NOTE: This is needed as TorchScriptTensor is nor supported by torch.is_complex()
    return tensor_dtype in _COMPLEX_TO_FLOAT


def from_complex_to_float(dtype: torch.dtype) -> torch.dtype:
    return _COMPLEX_TO_FLOAT[dtype]


def from_sym_value_to_torch_dtype(sym_value: SYM_VALUE_TYPE) -> torch.dtype:
    return _SYM_TYPE_TO_TORCH_DTYPE[type(sym_value)]


def is_optional_onnx_dtype_str(onnx_type_str: str) -> bool:
    return onnx_type_str in _OPTIONAL_ONNX_DTYPE_STR


def from_torch_dtype_to_onnx_dtype_str(dtype: torch.dtype | type) -> set[str]:
    return _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[dtype]


def from_python_type_to_onnx_attribute_type(
    dtype: type, is_sequence: bool = False
) -> onnx.defs.OpSchema.AttrType | None:
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


def is_torch_symbolic_type(value: Any) -> bool:
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))


def from_torch_dtype_to_abbr(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return ""
    return _TORCH_DTYPE_TO_ABBREVIATION.get(dtype, "")


def from_scalar_type_to_torch_dtype(scalar_type: type) -> torch.dtype | None:
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)


# NOTE: this is a mapping from torch dtype to a set of compatible onnx types
# It's used in dispatcher to find the best match overload for the input dtypes
_TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS: dict[torch.dtype | type, set[str]] = {
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

_OPTIONAL_ONNX_DTYPE_STR: set[str] = {
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

_COMPLEX_TO_FLOAT: dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,  # NOTE: ORT doesn't support torch.float64
}

_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

_SCALAR_TYPE_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
    **_PYTHON_TYPE_TO_TORCH_DTYPE,
    **_SYM_TYPE_TO_TORCH_DTYPE,  # type: ignore[dict-item]
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
    torch.SymInt,
    torch.SymFloat,
    torch.SymBool,
]
Argument = Optional[
    Union[
        tuple["Argument", ...],
        Sequence["Argument"],
        Mapping[str, "Argument"],
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        range,
        "torch.fx.Node",
        BaseArgumentTypes,
    ]
]
