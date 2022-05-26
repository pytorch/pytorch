import torch
from torch._prims.utils import (
    TensorLikeType,
    get_higher_dtype,
    is_float_dtype,
    is_complex_dtype,
)

from typing import Optional


def check_fp_or_complex(X: TensorLikeType, fn_name: str, half: bool = False):
    dtype = X.dtype
    if not is_float_dtype(dtype) and not is_complex_dtype(dtype):
        raise RuntimeError(
            f"{fn_name}: Expected a floating point or complex tensor as input. Got {X.dtype}"
        )


def check_norm_dtype(dtype: Optional[torch.dtype], x_dtype: torch.dtype, fn_name: str):
    if dtype is not None:
        if not is_float_dtype(dtype) and not is_complex_dtype(dtype):
            raise RuntimeError(
                f"{fn_name}: dtype should be floating point or complex, but got {dtype}"
            )
        if is_complex_dtype(dtype) != is_complex_dtype(x_dtype):
            d = "complex" if is_complex_dtype(x_dtype) else "real"
            raise RuntimeError(
                f"{fn_name}: dtype should be {d} for {d} inputs, but got {dtype}"
            )
        if get_higher_dtype(dtype, x_dtype) != dtype:
            raise RuntimeError(
                f"{fn_name}: the dtype of the input ({x_dtype}) should be convertible "
                "without narrowing to the specified dtype ({dtype})"
            )
