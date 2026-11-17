# Import dtype mappings lazily: this module imports cutlass.

import cutlass
from cutlass import Float32, Float64, Int32, Int64

import torch

from ..utils.scalar import f32
from .types import NumericType, Scalar


# torch dtype -> cute numeric type. Extend as new dtypes are supported.
torch2cute: dict[torch.dtype, NumericType] = {
    torch.float32: Float32,
    torch.float64: Float64,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.int32: Int32,
    torch.int64: Int64,
    torch.int8: cutlass.Int8,
    torch.int16: cutlass.Int16,
    torch.uint8: cutlass.Uint8,
    torch.uint16: cutlass.Uint16,
    torch.uint32: cutlass.Uint32,
    torch.uint64: cutlass.Uint64,
    torch.bool: cutlass.Boolean,
}

# Inverse mapping for allocating torch scratch from a trait's cute accumulator dtype.
cute2torch: dict[NumericType, torch.dtype] = {v: k for k, v in torch2cute.items()}


def scalar_arg(value: bool | int | float | complex, dtype: torch.dtype) -> Scalar:
    r"""Box runtime scalars with direct integer-to-FP32 rounding."""
    if isinstance(value, int) and dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        value = f32(value)
    return torch2cute[dtype](value)
