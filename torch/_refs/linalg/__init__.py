from functools import partial

from typing import List, Optional, Tuple, Union

import torch

import torch._prims as prims

import torch._prims_common as utils
import torch._refs as refs
import torch._refs.linalg as linalg
from torch import Tensor
from torch._prims_common import (
    check_fp_or_complex,
    check_is_matrix,
    Dim,
    DimsType,
    NumberType,
    TensorLikeType,
)
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper

__all__ = ["diagonal", "matrix_norm", "norm", "svd", "svdvals", "vector_norm"]


def _check_norm_dtype(dtype: Optional[torch.dtype], x_dtype: torch.dtype, fn_name: str):
    """
    Checks related to the dtype kwarg in `linalg.*norm` functions
    """
    if dtype is not None:
        torch._check(
            utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
            lambda: f"{fn_name}: dtype should be floating point or complex. Got {dtype}",
        )
        torch._check(
            utils.is_complex_dtype(dtype) == utils.is_complex_dtype(x_dtype),
            lambda: "{fn_name}: dtype should be {d} for {d} inputs. Got {dtype}".format(
                fn_name=fn_name,
                d="complex" if utils.is_complex_dtype(x_dtype) else "real",
                dtype=dtype,
            ),
        )
        torch._check(
            utils.get_higher_dtype(dtype, x_dtype) == dtype,
            lambda: f"{fn_name}: the dtype of the input ({x_dtype}) should be convertible "
            "without narrowing to the specified dtype ({dtype})",
        )


# Utilities should come BEFORE this import
from torch._decomp import register_decomposition


def diagonal(
    input: TensorLikeType,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)


@register_decomposition(torch._ops.ops.aten.linalg_vector_norm)
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
    check_fp_or_complex(x.dtype, "linalg.vector_norm")

    if isinstance(dim, Dim):
        dim = [dim]  # type: ignore[assignment]

    if x.numel() == 0 and (ord < 0.0 or ord == float("inf")):
        torch._check(
            dim is not None and len(dim) != 0,
            lambda: f"linalg.vector_norm cannot compute the {ord} norm on an empty tensor "
            "because the operation does not have an identity",
        )
        shape = x.shape
        assert dim is not None  # mypy does not seem to be able to see through check?
        for d in dim:
            torch._check(
                shape[d] != 0,
                lambda: f"linalg.vector_norm cannot compute the {ord} norm on the "
                f"dimension {d} because this dimension is empty and the "
                "operation does not have an identity",
            )
    _check_norm_dtype(dtype, x.dtype, "linalg.vector_norm")

    computation_dtype, result_dtype = utils.reduction_dtypes(
        x, utils.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT, dtype
    )

    to_result_dtype = partial(_maybe_convert_to_dtype, dtype=result_dtype)

    # Implementation
    if ord == 0.0:
        return torch.sum(torch.ne(x, 0.0), dim=dim, keepdim=keepdim, dtype=result_dtype)
    elif ord == float("inf"):
        return to_result_dtype(torch.amax(torch.abs(x), dim=dim, keepdim=keepdim))  # type: ignore[return-value,arg-type]
    elif ord == float("-inf"):
        return to_result_dtype(torch.amin(torch.abs(x), dim=dim, keepdim=keepdim))  # type: ignore[return-value,arg-type]
    else:
        # From here on the computation dtype is important as the reduction is non-trivial
        x = _maybe_convert_to_dtype(x, computation_dtype)  # type: ignore[assignment]
        reduce_sum = partial(torch.sum, dim=dim, keepdim=keepdim)

        if not (ord % 2.0 == 0.0 and utils.is_float_dtype(x.dtype)):
            x = torch.abs(x)
        return to_result_dtype(torch.pow(reduce_sum(torch.pow(x, ord)), 1.0 / ord))  # type: ignore[return-value]


def _backshift_permutation(dim0, dim1, ndim):
    # Auxiliary function for matrix_norm
    # Computes the permutation that moves the two given dimensions to the back
    ret = [i for i in range(ndim) if i != dim0 and i != dim1]
    ret.extend((dim0, dim1))
    return ret


def _inverse_permutation(perm):
    # Given a permutation, returns its inverse. It's equivalent to argsort on an array
    return [i for i, j in sorted(enumerate(perm), key=lambda i_j: i_j[1])]


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def matrix_norm(
    A: TensorLikeType,
    ord: Union[float, str] = "fro",
    dim: DimsType = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # shape
    check_is_matrix(A, "linalg.matrix_norm")
    # dim
    dim = utils.canonicalize_dims(A.ndim, dim)
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    torch._check(
        len(dim) == 2, lambda: "linalg.matrix_norm: dim must be a 2-tuple. Got {dim}"
    )
    torch._check(
        dim[0] != dim[1],
        lambda: "linalg.matrix_norm: dims must be different. Got ({dim[0]}, {dim[1]})",
    )
    # dtype arg
    _check_norm_dtype(dtype, A.dtype, "linalg.matrix_norm")

    if isinstance(ord, str):
        # ord
        torch._check(
            ord in ("fro", "nuc"),
            lambda: "linalg.matrix_norm: Order {ord} not supported.",
        )
        # dtype
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != "nuc"
        )

        if ord == "fro":
            return vector_norm(A, 2, dim, keepdim, dtype=dtype)
        else:  # ord == "nuc"
            if dtype is not None:
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = torch.sum(svdvals(prims.transpose(A, perm)), -1, keepdim)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
    else:
        # ord
        abs_ord = abs(ord)
        torch._check(
            abs_ord in (2, 1, float("inf")),
            lambda: "linalg.matrix_norm: Order {ord} not supported.",
        )
        # dtype
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != 2
        )

        max_min = partial(torch.amax if ord > 0.0 else torch.amin, keepdim=keepdim)

        if abs_ord == 2.0:
            if dtype is not None:
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = max_min(svdvals(prims.transpose(A, perm)), dim=-1)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
        else:  # 1, -1, inf, -inf
            dim0, dim1 = dim
            if abs_ord == float("inf"):
                dim0, dim1 = dim1, dim0
            if not keepdim and (dim0 < dim1):
                dim1 -= 1
            return max_min(
                vector_norm(A, 1.0, dim=dim0, keepdim=keepdim, dtype=dtype), dim1
            )


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def norm(
    A: TensorLikeType,
    ord: Optional[Union[float, str]] = None,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    if dim is not None:
        if isinstance(dim, Dim):
            dim = (dim,)  # type: ignore[assignment]
        torch._check(
            len(dim) in (1, 2),
            lambda: "linalg.norm: If dim is specified, it must be of length 1 or 2. Got {dim}",
        )
    elif ord is not None:
        torch._check(
            A.ndim in (1, 2),
            lambda: "linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got {A.ndim}D",
        )

    if ord is not None and (
        (dim is not None and len(dim) == 2) or (dim is None and A.ndim == 2)
    ):
        if dim is None:
            dim = (0, 1)
        return matrix_norm(A, ord, dim, keepdim, dtype=dtype)
    else:
        if ord is None:
            ord = 2.0
        return vector_norm(A, ord, dim, keepdim, dtype=dtype)


# CompositeImplicitAutograd
@out_wrapper("U", "S", "Vh", exact_dtype=True)
def svd(A: TensorLikeType, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    return prims.svd(A, full_matrices=full_matrices)


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def svdvals(A: TensorLikeType) -> Tensor:
    return svd(A, full_matrices=False)[1]
