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
    TYPE_CHECKING,
    Union,
)

import torch

if TYPE_CHECKING:
    import onnx.defs.OpSchema.AttrType  # type: ignore[import]


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


def is_torch_complex_dtype(tensor: TensorLike) -> bool:
    # NOTE: This is needed as TorchScriptTensor is nor supported by torch.is_complex()
    return tensor.dtype in _COMPLEX_TO_FLOAT


def from_complex_to_float(dtype: torch.dtype) -> torch.dtype:
    return _COMPLEX_TO_FLOAT[dtype]


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


def is_torch_symbolic_type(t: Any) -> bool:
    return isinstance(t, (torch.SymBool, torch.SymInt, torch.SymFloat))


def from_torch_dtype_to_abbr(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return ""
    return _TORCH_DTYPE_TO_ABBREVIATION.get(dtype, "")


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
    torch.int16: {"tensor(int16)"},
    torch.int32: {"tensor(int32)"},
    torch.int64: {"tensor(int64)"},
    torch.int8: {"tensor(int8)"},
    torch.uint8: {"tensor(uint8)"},
    str: {"tensor(string)"},
    int: {"tensor(int16)", "tensor(int32)", "tensor(int64)"},
    float: {"tensor(float16)", "tensor(float)", "tensor(double)"},
    bool: {"tensor(int32)", "tensor(int64)", "tensor(bool)"},
}

_COMPLEX_TO_FLOAT: Dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,  # NOTE: ORT doesn't support torch.float64
}

_TORCH_DTYPE_TO_ABBREVIATION = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
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
