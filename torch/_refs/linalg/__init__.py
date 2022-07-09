import torch
from torch import Tensor

import torch._prims as prims
import torch._refs as refs
from torch._prims.wrappers import out_wrapper

from torch._prims.utils import (
    check,
    DimsType,
    TensorLikeType,
    NumberType,
    is_float_dtype,
    is_complex_dtype,
    get_computation_dtype,
    get_higher_dtype,
    reduction_dtypes,
    REDUCTION_OUTPUT_TYPE_KIND,
)
import torch._refs.linalg as linalg
import torch._refs.linalg.utils

from typing import Optional, List
from functools import partial

__all__ = [
    "vector_norm",
]


def check_norm_dtype(dtype: Optional[torch.dtype], x_dtype: torch.dtype, fn_name: str):
    """
    Checks related to the dtype kwarg in `linalg.*norm` functions
    """
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


@out_wrapper(exact_dtype=True)
def vector_norm(
    x: TensorLikeType,
    ord: float = 2.0,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    # Checks
    linalg.utils.check_fp_or_complex(x.dtype, "linalg.vector_norm")

    if isinstance(dim, int):
        dim = [dim]  # type: ignore[assignment]
    elif not isinstance(dim, List) and dim is not None:
        # refs.amin just accepts List rather than DimType (Tuple)
        dim = list(dim)  # type: ignore[assignment]

    if x.numel() == 0 and (ord < 0.0 or ord == float("inf")):
        check(
            dim is not None and len(dim) != 0,
            lambda: f"linalg.vector_norm cannot compute the {ord} norm on an empty tensor "
            "because the operation does not have an identity",
        )
        shape = x.shape
        assert dim is not None  # mypy does not seem to be able to see through check?
        for d in dim:
            check(
                shape[d] != 0,
                lambda: f"linalg.vector_norm cannot compute the {ord} norm on the "
                f"dimension {d} because this dimension is empty and the "
                "operation does not have an identity",
            )
    check_norm_dtype(dtype, x.dtype, "linalg.vector_norm")

    computation_dtype, result_dtype = reduction_dtypes(
        x, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT, dtype
    )

    to_result_dtype = partial(prims.convert_element_type, dtype=result_dtype)

    # Implementation
    if ord == 0.0:
        return refs.sum(refs.ne(x, 0.0), dim=dim, keepdim=keepdim, dtype=result_dtype)
    elif ord == float("inf"):
        return to_result_dtype(refs.amax(torch.abs(x), dim=dim, keepdim=keepdim))
    elif ord == float("-inf"):
        return to_result_dtype(refs.amin(torch.abs(x), dim=dim, keepdim=keepdim))
    else:
        # From here on the computation dtype is important as the reduction is non-trivial
        x = prims.convert_element_type(x, computation_dtype)
        reduce_sum = partial(refs.sum, dim=dim, keepdim=keepdim)

        # Avoid computing a sqrt in abs and then squaring (more stable)
        # This could potentially be done for complex dtypes as
        # x = torch.real(torch.conj(x) * x))
        # and it should be more stable, but it's not clear whether it'll be faster on, say
        # CPU (abs is 1 vectorised operation), so leaving it just for real dtypes for now
        if not (ord % 2.0 == 0.0 and is_float_dtype(x.dtype)):
            x = torch.abs(x)
        return to_result_dtype(torch.pow(reduce_sum(torch.pow(x, ord)), 1.0 / ord))
