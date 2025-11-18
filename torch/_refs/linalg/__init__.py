# mypy: allow-untyped-defs
import math
from functools import partial
from typing import Optional, Union

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
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    IntLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    elementwise_type_promotion_wrapper,
    out_wrapper,
)


__all__ = [
    "diagonal",
    "matrix_norm",
    "norm",
    "svd",
    "svdvals",
    "vector_norm",
    "vecdot",
    "cross",
]


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
            f"without narrowing to the specified dtype ({dtype})",
        )


import operator

# Utilities should come BEFORE this import
from torch._decomp import register_decomposition
from torch._decomp.decompositions import pw_cast_for_opmath


@register_decomposition(torch._ops.ops.aten.linalg_cross)
@out_wrapper()
@pw_cast_for_opmath
def cross(a: Tensor, b: Tensor, dim: int = -1):
    torch._check(
        a.ndim == b.ndim,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    torch._check(
        a.size(dim) == 3 and b.size(dim) == 3,
        lambda: f"linalg.cross: inputs dim {dim} must have length 3, got {a.size(dim)} and {b.size(dim)}",
    )
    a, b = torch.broadcast_tensors(a, b)
    dim = utils.canonicalize_dim(a.ndim, dim)
    idx = torch.arange(3, device=a.device)
    return a.index_select(dim, (idx + 1) % 3) * b.index_select(
        dim, (idx + 2) % 3
    ) - a.index_select(dim, (idx + 2) % 3) * b.index_select(dim, (idx + 1) % 3)


def diagonal(
    input: TensorLikeType,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)


def _check_vector_norm_args(
    x: TensorLikeType, ord: Union[float, int] = 2, dim: Optional[DimsType] = None
):
    from torch.fx.experimental.symbolic_shapes import sym_or

    if not (ord < 0.0 or ord == float("inf")):
        return

    torch._check(
        sym_or(
            x.numel() != 0,
            not isinstance(dim, IntLike) and dim is not None and len(dim) != 0,
        ),
        lambda: f"linalg.vector_norm cannot compute the {ord} norm on an empty tensor "
        "because the operation does not have an identity",
    )

    shape = x.shape
    if dim is not None and not isinstance(dim, IntLike):
        for d in dim:
            torch._check(
                sym_or(x.numel() != 0, d < len(shape) and d >= 0 and shape[d] != 0),
                lambda: f"linalg.vector_norm cannot compute the {ord} norm on the "
                f"dimension {d} because this dimension is empty and the "
                "operation does not have an identity",
            )


@register_decomposition(torch._ops.ops.aten.linalg_vector_norm)
@out_wrapper(exact_dtype=True)
def vector_norm(
    x: TensorLikeType,
    ord: Union[float, int] = 2,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    check_fp_or_complex(x.dtype, "linalg.vector_norm")

    if isinstance(dim, Dim):
        dim = [dim]  # type: ignore[assignment]

    _check_vector_norm_args(x, ord, dim)

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

        is_ord_even = ord % 2 == 0 if isinstance(ord, IntLike) else ord % 2.0 == 0.0
        if dim == []:
            dim = None

        if (dim is None and x.numel() == 1) or (
            dim is not None
            and (x.ndim > 0 and all(guard_or_false(x.shape[d] == 1) for d in dim))
        ):
            if x.ndim > 64:
                raise RuntimeError(
                    f"Received a tensor with {x.ndim} dimensions, but only tensors with up to 64 dims are supported!"
                )
            x = torch.abs(x)
            if keepdim or x.ndim == 0:
                return to_result_dtype(x).contiguous()
            elif dim is None:
                return to_result_dtype(x).flatten()[0]
            else:
                new_shape = [s for d, s in enumerate(x.shape) if d not in dim]
                return to_result_dtype(x.view(new_shape)).contiguous()

        if not (is_ord_even and utils.is_float_dtype(x.dtype)):
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
    return [i for i, j in sorted(enumerate(perm), key=operator.itemgetter(1))]


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
        len(dim) == 2, lambda: f"linalg.matrix_norm: dim must be a 2-tuple. Got {dim}"
    )
    torch._check(
        # pyrefly: ignore [index-error]
        dim[0] != dim[1],
        # pyrefly: ignore [index-error]
        lambda: f"linalg.matrix_norm: dims must be different. Got ({dim[0]}, {dim[1]})",
    )
    # dtype arg
    _check_norm_dtype(dtype, A.dtype, "linalg.matrix_norm")

    if isinstance(ord, str):
        # ord
        torch._check(
            ord in ("fro", "nuc"),
            lambda: f"linalg.matrix_norm: Order {ord} not supported.",
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
            # pyrefly: ignore [index-error]
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
            lambda: f"linalg.matrix_norm: Order {ord} not supported.",
        )
        # dtype
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != 2
        )

        max_min = partial(torch.amax if ord > 0.0 else torch.amin, keepdim=keepdim)

        if abs_ord == 2.0:
            if dtype is not None:
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            # pyrefly: ignore [index-error]
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = max_min(svdvals(prims.transpose(A, perm)), dim=-1)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
        else:  # 1, -1, inf, -inf
            # pyrefly: ignore [bad-unpacking]
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
            lambda: f"linalg.norm: If dim is specified, it must be of length 1 or 2. Got {dim}",
        )
    elif ord is not None:
        torch._check(
            A.ndim in (1, 2),
            lambda: f"linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got {A.ndim}D",
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
        return vector_norm(A, ord, dim, keepdim, dtype=dtype)  # type: ignore[arg-type]


# CompositeImplicitAutograd
@out_wrapper("U", "S", "Vh", exact_dtype=True)
def svd(A: TensorLikeType, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    return prims.svd(A, full_matrices=full_matrices)


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
def svdvals(A: TensorLikeType) -> Tensor:
    return svd(A, full_matrices=False)[1]


# CompositeImplicitAutograd
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("x", "y"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def vecdot(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    check_fp_or_complex(x.dtype, "linalg.vecdot")
    return (x.conj() * y).sum(dim=dim)


def _pivots_to_permutation(pivots, shape, *, inverse=False):
    perm = torch.empty(shape, dtype=torch.int32, device=pivots.device)
    perm[..., :] = torch.arange(shape[-1], dtype=torch.int32, device=pivots.device)
    indices = range(shape[-1])
    if inverse:
        indices = reversed(indices)

    if len(shape) > 1:
        for i in indices:
            j_s = pivots[..., i]
            perm_i = perm[..., i].clone()
            j_idx = torch.meshgrid(
                *[torch.arange(s, device=perm.device) for s in j_s.shape], indexing="ij"
            ) + (j_s,)
            perm_j = perm[j_idx]
            perm.index_put_(j_idx, perm_i)
            perm[..., i].copy_(perm_j)

    else:
        for i in indices:
            j = pivots[i]
            perm_i = perm[i].clone()
            perm_j = perm[j].clone()
            perm[i].copy_(perm_j)
            perm[j].copy_(perm_i)

    return perm


def _apply_pivots(a, pivots, shape, *, inverse=False):
    perm = _pivots_to_permutation(pivots - 1, shape, inverse=inverse)

    if len(shape) == 1:
        return a[perm, :]
    else:
        idx = torch.meshgrid(
            *[torch.arange(s, device=a.device) for s in perm.shape], indexing="ij"
        )[:-1] + (perm, slice(None))
        return a[idx]


def linalg_lu_solve_out_mps(LU, pivots, B, *, left=True, adjoint=False, out):
    if out.numel() == 0:
        return

    if not left:
        adjoint = not adjoint
        B = B.mH

    if adjoint:
        lu_ = LU.mH
        x = torch.linalg.solve_triangular(lu_, B, left=True, upper=False)
        x = torch.linalg.solve_triangular(
            lu_, x, left=True, upper=True, unitriangular=True
        )
        x = _apply_pivots(x, pivots, LU.shape[:-1], inverse=True)
    else:
        x = _apply_pivots(B, pivots, LU.shape[:-1])
        x = torch.linalg.solve_triangular(
            LU, x, left=True, upper=False, unitriangular=True
        )
        x = torch.linalg.solve_triangular(LU, x, left=True, upper=True)

    if not left:
        x = x.mH

    out.copy_(x)


mps_lib = torch.library.Library("aten", "IMPL", "MPS")  # noqa: TOR901
mps_lib.impl("aten::linalg_lu_solve.out", linalg_lu_solve_out_mps)
