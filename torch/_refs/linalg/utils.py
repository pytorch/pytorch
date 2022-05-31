import torch
from torch._prims.utils import (
    check,
    get_higher_dtype,
    is_half_dtype,
    is_float_dtype,
    is_complex_dtype,
)

from typing import Optional


def check_fp_or_complex(dtype: torch.dtype, fn_name: str, half: bool = False):
    check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    check(
        half or not is_half_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )


def check_norm_dtype(dtype: Optional[torch.dtype], x_dtype: torch.dtype, fn_name: str):
    if dtype is not None:
        check(
            is_float_dtype(dtype) or is_complex_dtype(dtype),
            lambda: f"{fn_name}: dtype should be floating point or complex. Got {dtype}",
        )
        check(
            is_complex_dtype(dtype) == is_complex_dtype(x_dtype),
            lambda: "{fn_name}: dtype should be {d} for {d} inputs. Got {dtype}".format(
                fn_name=fn_name,
                d="complex" if is_complex_dtype(x_dtype) else "real",
                dtype=dtype,
            ),
        )
        check(
            get_higher_dtype(dtype, x_dtype) == dtype,
            lambda: f"{fn_name}: the dtype of the input ({x_dtype}) should be convertible "
            "without narrowing to the specified dtype ({dtype})",
        )
