import math
from enum import Enum

import torch


class ScalarDtype(Enum):
    bool = bool
    int = int
    float = float


SUPPORTED_TENSOR_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    # torch.float16,
    torch.float32,
    torch.float64,
    # torch.complex32,
    # torch.complex64,
    # torch.complex128,
    # torch.bfloat16,
]


class VariableType(Enum):
    Bool = bool
    Int = int
    Float = float
    String = str
    ScalarDtype = ScalarDtype
    TensorDtype = torch.dtype
    Tuple = tuple

    @staticmethod
    def contains(v):
        return v in [member.value for member in VariableType]


def check_vtype(vtype, v):
    if v is None:
        return
    if vtype in [bool, int, float]:
        if type(v) not in [bool, int, float]:
            raise TypeError("Variable type mismatch")
    else:
        if not isinstance(v, vtype):
            raise TypeError("Variable type mismatch")


def is_integer(v):
    if math.isnan(v) or not math.isfinite(v):
        return False
    return int(v) == v


def convert_to_vtype(vtype, v) -> None:
    if vtype == bool:
        return bool(v)
    if vtype == int:
        if not is_integer(v):
            return v
        return int(v)
    if vtype == float:
        return float(v)
    return v
