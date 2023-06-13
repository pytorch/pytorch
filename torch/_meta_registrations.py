import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch._prims_common as utils
from torch import Tensor
from torch._decomp import (
    _add_op_to_registry,
    _convert_out_params,
    global_decomposition_table,
    meta_table,
)
from torch._ops import OpOverload
from torch._prims import _elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
    check,
    corresponding_complex_dtype,
    corresponding_real_dtype,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    IntLike,
    make_contiguous_strides_for,
    TensorLike,
)

from torch._prims_common.wrappers import (
    _maybe_resize_out,
    _resize_output_check,
    _safe_copy_out,
    out_wrapper,
)
from torch._refs import _broadcast_shapes

from torch.utils._pytree import tree_map


aten = torch.ops.aten

_meta_lib_dont_use_me_use_register_meta = torch.library.Library("aten", "IMPL", "Meta")


def register_meta(op):
    def wrapper(fn):
        fn = _convert_out_params(fn)

        def register(op):
            _add_op_to_registry(meta_table, op, fn)

        tree_map(register, op)
        return fn

    return wrapper


def toRealValueType(dtype):
    from_complex = {
        torch.complex32: torch.half,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double,
    }
    return from_complex.get(dtype, dtype)


def check_inplace_broadcast(self_shape, *args_shape):
    broadcasted_shape = tuple(_broadcast_shapes(self_shape, *args_shape))
    check(
        broadcasted_shape == self_shape,
        lambda: f"output with shape {self_shape} doesn't match the broadcast shape {broadcasted_shape}",
    )


@register_meta([aten.take.default, aten.take.out])
@out_wrapper()
def meta_take(self, index):
    # Type and device checks
    check(
        index.dtype == torch.long,
        lambda: f"take(): Expected a long tensor for index, but got {index.dtype}",
    )
    # Index checks
    check(
        not (self.numel() == 0 and index.numel() != 0),
        lambda: "take(): tried to take from an empty tensor",
        IndexError,
    )
    return self.new_empty(index.shape)


@register_meta([aten.linalg_cross.default, aten.linalg_cross.out])
@out_wrapper()
def linalg_cross(self, other, *, dim=-1):
    x_d = self.ndim
    y_d = other.ndim
    check(
        x_d == y_d,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    check(
        self.size(dim) == 3 and other.size(dim) == 3,
        lambda: (
            f"linalg.cross: inputs dimension {dim} must have length 3. "
            f"Got {self.size(dim)} and {other.size(dim)}"
        ),
    )
    out_shape = _broadcast_shapes(self.shape, other.shape)
    return self.new_empty(out_shape)


@register_meta(aten.linalg_matrix_exp)
@out_wrapper()
def linalg_matrix_exp(self):
    squareCheckInputs(self, "linalg.matrix_exp")
    checkFloatingOrComplex(self, "matrix_exp")
    return torch.empty_like(self)


@register_meta(
    [aten.cummax.default, aten.cummax.out, aten.cummin.default, aten.cummin.out]
)
@out_wrapper("values", "indices")
def cummaxmin(self, dim):
    values = torch.empty(self.shape, device=self.device, dtype=self.dtype)
    indices = torch.empty(self.shape, device=self.device, dtype=torch.int64)
    if self.numel() != 0 and self.ndim != 0:
        # Checks that dim is within bounds
        maybe_wrap_dim(dim, self.ndim)
    return values, indices


@register_meta([aten.logcumsumexp.default, aten.logcumsumexp.out])
@out_wrapper()
def logcumsumexp(self, dim):
    # Checks that dim is within bounds
    maybe_wrap_dim(dim, self.ndim)
    return torch.empty_like(self).contiguous()


@register_meta([aten._fft_c2c.default, aten._fft_c2c.out])
@out_wrapper()
def meta_fft_c2c(self, dim, normalization, forward):
    assert self.dtype.is_complex
    return self.new_empty(self.size())


@register_meta([aten._fft_r2c.default, aten._fft_r2c.out])
@out_wrapper()
def meta_fft_r2c(self, dim, normalization, onesided):
    assert self.dtype.is_floating_point
    output_sizes = list(self.size())

    if onesided:
        last_dim = dim[-1]
        last_dim_halfsize = (output_sizes[last_dim] // 2) + 1
        output_sizes[last_dim] = last_dim_halfsize

    return self.new_empty(
        output_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
    )


@register_meta(aten.randperm.generator_out)
def meta_randperm(n, *, generator=None, out):
    assert out.ndim == 1 and out.size(0) == n
    return out


@register_meta(aten.randperm.default)
def meta_randperm_default(
    n, *, dtype=torch.long, layout=None, device=None, pin_memory=None
):
    return torch.empty(
        n, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.randint.default)
def meta_randint(
    high, size, *, dtype=torch.long, layout=None, device=None, pin_memory=None
):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.randint.low)
def meta_randint_low(
    low,
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.rand.default)
def meta_rand_default(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten._fft_c2r.default, aten._fft_c2r.out])
@out_wrapper()
def meta_fft_c2r(self, dim, normalization, lastdim):
    assert self.dtype.is_complex
    output_sizes = list(self.size())
    output_sizes[dim[-1]] = lastdim
    return self.new_empty(output_sizes, dtype=toRealValueType(self.dtype))


@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=False):
    return self


def inferUnsqueezeGeometry(tensor, dim):
    result_sizes = list(tensor.size())
    result_strides = list(tensor.stride())
    new_stride = 1 if dim >= tensor.dim() else result_sizes[dim] * result_strides[dim]
    result_sizes.insert(dim, 1)
    result_strides.insert(dim, new_stride)
    return result_sizes, result_strides


@register_meta(aten.unsqueeze_.default)
def meta_unsqueeze_(self, dim):
    dim = maybe_wrap_dim(dim, self.dim() + 1)
    g_sizes, g_strides = inferUnsqueezeGeometry(self, dim)
    self.as_strided_(g_sizes, g_strides)
    return self


# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@register_meta(aten.index_select.default)
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)


@register_meta(aten.index_select.out)
def meta_index_select_out(self, dim, index, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.index_select(self, dim, index))


@register_meta([aten.max.default, aten.max.unary_out])
@out_wrapper()
def meta_max(self):
    return self.new_empty(())


@register_meta(aten.max.dim)
def meta_max_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta([aten.min.default, aten.min.unary_out])
@out_wrapper()
def meta_min(self):
    return self.new_empty(())


@register_meta(aten.min.dim)
def meta_min_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.angle.default)
def meta_angle(self):
    if self.is_complex():
        result_dtype = corresponding_real_dtype(self.dtype)
    else:
        _, result_dtype = elementwise_dtypes(
            self,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        )
    return torch.empty_like(self, dtype=result_dtype)


@register_meta(aten.angle.out)
def meta_angle_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.angle(self))


@register_meta(aten._assert_async.default)
def assert_async(val):
    return


@register_meta(aten._assert_async.msg)
def assert_async_meta(val, assert_msg):
    return


# From aten/src/ATen/native/LinearAlgebraUtils.h
def squareCheckInputs(self: Tensor, f_name: str):
    assert (
        self.dim() >= 2
    ), f"{f_name}: The input tensor must have at least 2 dimensions."
    assert self.size(-1) == self.size(
        -2
    ), f"{f_name}: A must be batches of square matrices, but they are {self.size(-2)} by {self.size(-1)} matrices"


# Validates input shapes and devices
# for linear solve methods (solve, cholesky_solve, lu_solve, triangular_solve)
# From aten/src/ATen/native/LinearAlgebraUtils.h
def linearSolveCheckInputs(
    self: Tensor,
    A: Tensor,
    name: str,
):
    check(
        self.device == A.device,
        lambda: (
            f"Expected b and A to be on the same device, but found b on "
            f"{self.device} and A on {A.device} instead."
        ),
    )

    check(
        self.dtype == A.dtype,
        lambda: (
            f"Expected b and A to have the same dtype, but found b of type "
            f"{self.dtype} and A of type {A.dtype} instead."
        ),
    )

    check(
        A.size(-1) == A.size(-2),
        lambda: (
            f"A must be batches of square matrices, "
            f"but they are {A.size(-2)} by {A.size(-1)} matrices"
        ),
    )

    check(
        A.size(-1) == self.size(-2),
        lambda: (
            f"Incompatible matrix sizes for {name}: each A "
            f"matrix is {A.size(-1)} by {A.size(-1)}"
            f" but each b matrix is {self.size(-2)} by {self.size(-1)}"
        ),
    )


# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkFloatingOrComplex(
    t: Tensor, f_name: str, allow_low_precision_dtypes: bool = True
):
    dtype = t.dtype
    check(
        t.is_floating_point() or t.is_complex(),
        lambda: f"{f_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    if not allow_low_precision_dtypes:
        check(
            dtype in (torch.float, torch.double, torch.cfloat, torch.cdouble),
            lambda: f"{f_name}: Low precision dtypes not supported. Got {dtype}",
        )


# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkIsMatrix(A: Tensor, f_name: str, arg_name: str = "A"):
    check(
        A.dim() >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def checkInputsSolver(
    A: Tensor,
    B: Tensor,
    left: bool,
    f_name: str,
):
    squareCheckInputs(A, f_name)
    checkIsMatrix(B, f_name)
    check(
        A.size(-2) == B.size(-2) if left else A.size(-1) == B.size(-1),
        lambda: (
            f"{f_name}: Incompatible shapes of A and B for the equation "
            f"{'AX = B' if left else 'XA = B'}"
            f" ({A.size(-2)}x{A.size(-1)} and {B.size(-2)}x{B.size(-1)})"
        ),
    )


def checkSameDevice(
    fn_name: str, result: Tensor, input: Tensor, result_name: str = "result"
):
    check(
        result.device == input.device,
        lambda: (
            f"{fn_name}: Expected {result_name} and input tensors to be on the same device, but got "
            f"{result_name} on {result.device} and input on {input.device}"
        ),
    )


def checkUplo(UPLO: str):
    UPLO_uppercase = UPLO.upper()
    check(
        len(UPLO) == 1 and (UPLO_uppercase == "U" or UPLO_uppercase == "L"),
        lambda: f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}",
    )


@register_meta([aten._linalg_eigh.default, aten._linalg_eigh.eigenvalues])
@out_wrapper("eigenvalues", "eigenvectors")
def meta__linalg_eigh(
    A: Tensor,
    UPLO: str = "L",
    compute_v: bool = True,
):
    squareCheckInputs(A, "linalg.eigh")
    checkUplo(UPLO)

    shape = list(A.shape)
    if compute_v:
        vecs = A.new_empty(shape)
        vecs.as_strided_(shape, make_contiguous_strides_for(shape, row_major=False))
    else:
        vecs = A.new_empty([0])

    shape.pop()
    vals = A.new_empty(shape, dtype=toRealValueType(A.dtype))

    return vals, vecs


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
@register_meta(aten.linalg_cholesky_ex.default)
def linalg_cholesky_ex(A: Tensor, upper: bool = False, check_errors: bool = False):
    squareCheckInputs(A, "linalg.cholesky")
    checkFloatingOrComplex(A, "linalg.cholesky")

    A_shape = A.shape
    ndim = len(A_shape)

    # L
    L_strides = make_contiguous_strides_for(A_shape, False)
    L = A.new_empty(A_shape)
    L.as_strided_(A_shape, L_strides)

    # infos
    infos = A.new_empty(A_shape[0 : ndim - 2], dtype=torch.int32)
    return L, infos


@register_meta(
    [aten.linalg_householder_product.default, aten.linalg_householder_product.out]
)
@out_wrapper()
def linalg_householder_product(input: Tensor, tau: Tensor) -> Tensor:
    check(
        input.ndim >= 2,
        lambda: "torch.linalg.householder_product: input must have at least 2 dimensions.",
    )
    check(
        input.size(-2) >= input.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]",
    )
    check(
        input.size(-1) >= tau.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]",
    )

    check(
        input.ndim - tau.ndim == 1,
        lambda: (
            f"torch.linalg.householder_product: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )
    if input.ndim > 2:
        expected_batch_tau_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        check(
            actual_batch_tau_shape == expected_batch_tau_shape,
            lambda: (
                f"torch.linalg.householder_product: Expected batch dimensions of tau to be "
                f"equal to input.shape[:-2], but got {actual_batch_tau_shape}"
            ),
        )

    check(
        tau.dtype == input.dtype,
        lambda: (
            f"torch.linalg.householder_product: tau dtype {tau.dtype}"
            f" does not match input dtype {input.dtype}"
        ),
    )
    checkSameDevice("torch.linalg.householder_product", tau, input, "tau")

    return torch.empty_strided(
        size=input.shape,
        stride=make_contiguous_strides_for(input.shape, row_major=False),
        dtype=input.dtype,
        device=input.device,
    )


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
@register_meta(aten.linalg_inv_ex.default)
def linalg_inv_ex_meta(A: Tensor, check_errors: bool = False):
    squareCheckInputs(A, "linalg.inv_ex")
    checkFloatingOrComplex(A, "linalg.inv_ex", allow_low_precision_dtypes=False)

    L = A.new_empty(A.shape)
    L.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    infos = A.new_empty(A.shape[:-2], dtype=torch.int32)
    return L, infos


@register_meta([aten.linalg_ldl_factor_ex.default, aten.linalg_ldl_factor_ex.out])
@out_wrapper("LD", "pivots", "info")
def linalg_ldl_factor_ex_meta(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    squareCheckInputs(self, "torch.linalg.ldl_factor_ex")
    checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex")
    LD = torch.empty_strided(
        size=self.shape,
        stride=make_contiguous_strides_for(self.shape, row_major=False),
        dtype=self.dtype,
        device=self.device,
    )
    pivots = self.new_empty(self.shape[:-1], dtype=torch.int)
    info = self.new_empty(self.shape[:-2], dtype=torch.int)
    return LD, pivots, info


@register_meta([aten.linalg_ldl_solve.default, aten.linalg_ldl_solve.out])
@out_wrapper()
def linalg_ldl_solve_meta(
    LD: Tensor, pivots: Tensor, B: Tensor, *, hermitian: bool = False
) -> Tensor:
    squareCheckInputs(LD, "torch.linalg.ldl_solve")
    checkFloatingOrComplex(LD, "torch.linalg.ldl_solve")
    linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve")
    check(
        B.ndim >= 2,
        lambda: (
            f"torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, "
            f"but it has {B.ndim} dimensions instead"
        ),
    )
    expected_pivots_shape = LD.shape[:-1]
    check(
        expected_pivots_shape == pivots.shape,
        lambda: (
            f"torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )
    check(
        utils.is_integer_dtype(pivots.dtype),
        lambda: f"torch.linalg.ldl_solve: Expected pivots to be integers. Got {pivots.dtype}",
    )
    check(
        LD.dtype == B.dtype,
        lambda: f"torch.linalg.ldl_solve: LD dtype {LD.dtype} does not match b dtype {B.dtype}",
    )
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LD)
    return torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=False),
        dtype=B.dtype,
        device=B.device,
    )


@register_meta([aten.linalg_lu.default, aten.linalg_lu.out])
@out_wrapper("P", "L", "U")
def linalg_lu_meta(A: Tensor, *, pivot: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    check(
        A.ndim >= 2,
        lambda: f"linalg.lu: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)

    sizes[-1] = m
    if pivot:
        P = A.new_empty(sizes)
    else:
        P = A.new_empty([0])

    sizes[-1] = k
    L = A.new_empty(sizes)

    sizes[-2] = k
    sizes[-1] = n
    U = A.new_empty(sizes)
    return P, L, U


@register_meta([aten.linalg_lu_factor_ex.default, aten.linalg_lu_factor_ex.out])
@out_wrapper("LU", "pivots", "info")
def linalg_lu_factor_ex_meta(
    A: Tensor, *, pivot: bool = True, check_errors: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    check(
        A.ndim >= 2,
        lambda: f"torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]

    LU = torch.empty_strided(
        size=sizes,
        stride=make_contiguous_strides_for(sizes, row_major=False),
        dtype=A.dtype,
        device=A.device,
    )

    # Sets sizes to the size of pivots
    sizes.pop()
    sizes[-1] = min(m, n)
    pivots = A.new_empty(sizes, dtype=torch.int)

    # Sets sizes to the size of info
    sizes.pop()
    info = A.new_empty(sizes, dtype=torch.int)

    return LU, pivots, info


@register_meta([aten.linalg_lu_solve.default, aten.linalg_lu_solve.out])
@out_wrapper()
def linalg_lu_solve_meta(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> Tensor:
    # dtype
    checkFloatingOrComplex(LU, "torch.linalg.lu_solve")
    check(
        LU.dtype == B.dtype,
        lambda: (
            f"linalg.lu_solve: Expected LU and B to have the same dtype, "
            f"but found LU of type {LU.dtype} and B of type {B.dtype} instead"
        ),
    )
    check(
        pivots.dtype == torch.int,
        lambda: "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32",
    )

    # matrix shapes
    squareCheckInputs(LU, "torch.linalg.lu_solve")
    checkInputsSolver(LU, B, left, "linalg.lu_solve")
    check(
        LU.size(-1) == pivots.size(-1),
        lambda: "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix",
    )

    # batches
    check(
        LU.shape[:-1] == pivots.shape,
        lambda: (
            f"linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )

    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LU)

    result = torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=not left),
        dtype=B.dtype,
        device=B.device,
    )

    if result.numel() != 0 and not left:
        if result.is_complex():
            result = result.conj()

    return result


@register_meta(aten.lu_unpack)
@out_wrapper("P", "L", "U")
def lu_unpack_meta(
    LU: Tensor,
    pivots: Tensor,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    torch._check(
        LU.ndim >= 2,
        lambda: f"torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: {LU.shape} instead",
    )
    if unpack_pivots:
        torch._check(
            pivots.dtype == torch.int32,
            lambda: (
                "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
                "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor"
            ),
        )
    sizes = list(LU.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)
    sizes[-1] = m
    if unpack_pivots:
        P = LU.new_empty(sizes)
    else:
        P = LU.new_empty([0])
    if unpack_data:
        sizes[-1] = k
        L = LU.new_empty(sizes)
        sizes[-2] = k
        sizes[-1] = n
        U = LU.new_empty(sizes)
    else:
        L = LU.new_empty([0])
        U = LU.new_empty([0])
    return P, L, U


# parse the "mode" param in linalg_qr: return a tuple of bools (compute_q, reduced)
def _parse_qr_mode(mode: str) -> Tuple[bool, bool]:
    if mode == "reduced":
        compute_q = True
        reduced = True
    elif mode == "complete":
        compute_q = True
        reduced = False
    elif mode == "r":
        compute_q = False
        reduced = True  # this is actually irrelevant in this mode
    else:
        check(
            False,
            lambda: (
                f"qr received unrecognized mode '{mode}' "
                f"but expected one of 'reduced' (default), 'r', or 'complete'"
            ),
        )
    return compute_q, reduced


@register_meta([aten.linalg_qr.default, aten.linalg_qr.out])
@out_wrapper("Q", "R")
def linalg_qr_meta(
    A: Tensor,
    mode: str = "reduced",
) -> Tuple[Tensor, Tensor]:
    checkIsMatrix(A, "linalg.qr")
    checkFloatingOrComplex(A, "linalg.qr")

    compute_q, reduced_mode = _parse_qr_mode(mode)

    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)

    if compute_q:
        Q_shape = list(A.shape)
        Q_shape[-1] = k if reduced_mode else m
        Q = A.new_empty(Q_shape)
        Q.as_strided_(Q_shape, make_contiguous_strides_for(Q_shape, row_major=False))
    else:
        Q = A.new_empty([0])

    # For readability
    R_shape = list(A.shape)
    R_shape[-2] = k if reduced_mode or not compute_q else m
    R = A.new_empty(R_shape)
    R.as_strided_(R_shape, make_contiguous_strides_for(R_shape, row_major=False))
    return Q, R


@register_meta([aten._linalg_slogdet.default, aten._linalg_slogdet.sign])
@out_wrapper("sign", "logabsdet", "LU", "pivots")
def _linalg_slogdet(A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    squareCheckInputs(A, "linalg.slogdet")
    checkFloatingOrComplex(A, "linalg.slogdet", False)
    shape = A.shape
    sign = A.new_empty(shape[:-2])
    logabsdet = A.new_empty(shape[:-2], dtype=toRealValueType(A.dtype))
    LU = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    pivots = A.new_empty(shape[:-1], dtype=torch.int32)
    return sign, logabsdet, LU, pivots


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
# NOTE: matching defaults in aten/src/ATen/native/native_functions.yaml
@register_meta(aten._linalg_svd.default)
def _linalg_svd_meta(
    A: Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: str = None,
):
    checkIsMatrix(A, "linalg.svd")
    checkFloatingOrComplex(A, "linalg.svd")

    batch_dims = list(A.shape[:-2])
    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)

    if compute_uv:
        U_shape = batch_dims + [m, m if full_matrices else k]
        U = A.new_empty(U_shape)
        U.as_strided_(U_shape, make_contiguous_strides_for(U_shape, row_major=False))

        V_shape = batch_dims + [n if full_matrices else k, n]
        V = A.new_empty(V_shape)
        # NB: This checks for CUDA since there is no way to check for cuSolver.
        # Also, this might not work correctly on CPU when fake_device is not
        # available as device_hint just defaults to CUDA in that case. See
        # _linalg_svd meta in core.
        is_cuda = device_hint(A) == "cuda"
        V.as_strided_(V_shape, make_contiguous_strides_for(V_shape, row_major=is_cuda))
    else:
        # doesn't matter
        U = A.new_empty([0])
        V = A.new_empty([0])

    # S is always real, even when A is complex.
    S = A.new_empty(batch_dims + [k], dtype=toRealValueType(A.dtype))
    return U, S, V


def _linalg_broadcast_batch_dims(
    arg1: Tensor, arg2: Tensor
) -> Tuple[List[int], List[int]]:
    # broadcast the batch dimensions of arg1 and arg2.
    arg1_batch_sizes = arg1.shape[:-2]
    arg2_batch_sizes = arg2.shape[:-2]
    expand_batch_portion = _broadcast_shapes(arg1_batch_sizes, arg2_batch_sizes)

    arg1_expand_size = list(expand_batch_portion)
    arg1_expand_size += [arg1.size(-2), arg1.size(-1)]

    arg2_expand_size = list(expand_batch_portion)
    arg2_expand_size += [arg2.size(-2), arg2.size(-1)]
    return arg1_expand_size, arg2_expand_size


def _linalg_broadcast_batch_dims_name(
    arg1: Tensor, arg2: Tensor, name: Optional[str]
) -> Tuple[Tensor, Tensor]:
    # If there's no name we assume we don't want to check the errors
    if name:
        linearSolveCheckInputs(arg1, arg2, name)

    arg1_expand_size, arg2_expand_size = _linalg_broadcast_batch_dims(arg1, arg2)

    arg1_broadcasted = (
        arg1 if arg1_expand_size == arg1.shape else arg1.expand(arg1_expand_size)
    )
    arg2_broadcasted = (
        arg2 if arg2_expand_size == arg2.shape else arg2.expand(arg2_expand_size)
    )
    return arg1_broadcasted, arg2_broadcasted


def linalg_solve_is_vector_rhs(input: Tensor, other: Tensor) -> bool:
    expected_batched_rhs_shape = input.shape[:-1]
    vector_case = other.ndim == 1 or (
        input.ndim - 1 == other.ndim and other.shape == expected_batched_rhs_shape
    )
    return vector_case


@register_meta(aten._linalg_solve_ex)
def _linalg_solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    result: Optional[Tensor] = None,
    LU: Optional[Tensor] = None,
    pivots: Optional[Tensor] = None,
    info: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    checkFloatingOrComplex(A, "linalg.solve")
    torch._check(
        A.dtype == B.dtype,
        lambda: (
            f"linalg.solve: Expected A and B to have the same dtype, but found A of type "
            f"{A.dtype} and B of type {B.dtype} instead"
        ),
    )
    vector_case = linalg_solve_is_vector_rhs(A, B)
    B_ = B.unsqueeze(-1) if vector_case else B
    checkInputsSolver(A, B_, left, "linalg.solve")
    B_broad_shape, _ = _linalg_broadcast_batch_dims(B_, A)
    torch._check(
        left or not vector_case,
        lambda: (
            "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. "
            "In this case linalg.solve is equivalent to B / A.squeeze(-1)"
        ),
    )
    result_shape = B_broad_shape[:-1] if vector_case else B_broad_shape
    result_ = torch.empty_strided(
        size=result_shape,
        stride=make_contiguous_strides_for(result_shape, not left),
        dtype=B.dtype,
        device=B.device,
    )
    shape = A.shape
    ndim = A.ndim
    LU_ = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    pivots_ = A.new_empty(shape[:-1], dtype=torch.int32)
    info_ = A.new_empty(shape[:-2], dtype=torch.int32)
    out = (result, LU, pivots, info)
    res = (result_, LU_, pivots_, info_)
    if all(x is not None for x in out):
        for r, o in zip(res, out):
            # resize and copy operations are done in-place
            _maybe_resize_out(o, r.shape)  # type: ignore[arg-type]
            # strides are not copied in out_wrapper
            o.as_strided_(r.shape, r.stride())  # type: ignore[union-attr]
            _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=False)  # type: ignore[arg-type]
    return res


@register_meta([aten.linalg_solve_triangular.default, aten.linalg_solve_triangular.out])
def linalg_solve_triangular_meta(
    A: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    out: Tensor = None,
) -> Tensor:
    if out is None:
        out = A.new_empty([0])
    assert isinstance(out, TensorLike)
    checkInputsSolver(A, B, left, "linalg.solve_triangular")
    B_, A_ = _linalg_broadcast_batch_dims_name(B, A, None)
    avoid_copy_A = A_.transpose(-2, -1).is_contiguous() and A_.is_conj()
    if avoid_copy_A:
        out = _maybe_resize_out(out, B_.shape)
    else:
        # reimplementation of resize_output with result F-contig
        if _resize_output_check(out, B_.shape):
            out.resize_(B_.transpose(-2, -1).shape)
            out.transpose_(-2, -1)
    return out  # type: ignore[return-value]


# From aten/src/ATen/native/LinearAlgebra.cpp
@register_meta(aten._linalg_det.default)
def _linalg_det_meta(A):
    squareCheckInputs(A, "linalg.det")
    checkFloatingOrComplex(A, "linalg.det")

    det = A.new_empty(A.shape[:-2])

    LU = A.new_empty(A.shape)
    LU.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    pivots = A.new_empty(A.shape[:-1], dtype=torch.int32)
    return det, LU, pivots


# From aten/src/ATen/native/ReflectionPad.cpp
@register_meta(
    [
        aten.reflection_pad2d_backward.default,
        aten.replication_pad2d_backward.default,
    ]
)
def meta_pad2d_backward(grad_output, self, padding):
    dim_w = 2
    dim_h = 1
    dim_plane = 0
    nbatch = 1

    self_shape = self.shape
    if self.dim() == 4:
        nbatch = self_shape[0]
        dim_w += 1
        dim_h += 1
        dim_plane += 1

    pad_l = padding[0]
    pad_r = padding[1]
    pad_t = padding[2]
    pad_b = padding[3]

    nplane = self_shape[dim_plane]
    input_h = self_shape[dim_h]
    input_w = self_shape[dim_w]
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    check(
        output_w == grad_output.shape[dim_w],
        lambda: f"gradOutput width unexpected. Expected: {output_w}, Got: {grad_output.shape[dim_w]}",
    )
    check(
        output_h == grad_output.shape[dim_h],
        lambda: f"gradOutput height unexpected. Expected: {output_h}, Got: {grad_output.shape[dim_h]}",
    )
    return self.new_empty(self.shape)


@register_meta(aten.reflection_pad2d.default)
def meta_pad2d(self, padding):
    valid_dims = self.size(1) != 0 and self.size(2) != 0
    check(
        (self.ndim == 3 and valid_dims)
        or (self.ndim == 4 and valid_dims and self.size(3) != 0),
        lambda: f"3D or 4D (batch mode) tensor expected for input, but got: {self}",
    )
    if self.ndim == 4:
        nbatch, nplane, input_h, input_w = self.shape
    else:
        nbatch = 1
        nplane, input_h, input_w = self.shape

    pad_l, pad_r, pad_t, pad_b = padding

    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    if self.ndim == 3:
        return self.new_empty((nplane, output_h, output_w))
    else:
        return self.new_empty((nbatch, nplane, output_h, output_w))


@register_meta([aten.baddbmm.default, aten.baddbmm.out])
@out_wrapper()
def meta_baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(0)
    dim2 = batch1.size(1)
    dim3 = batch2.size(2)
    self = self.expand((dim1, dim2, dim3))
    check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    check(
        self.dtype == batch1.dtype == batch2.dtype,
        lambda: f"Input dtypes must be the same, got: input: {self.dtype}, batch1: {batch1.dtype}, batch2: {batch2.dtype}",
    )
    batch1_sizes = batch1.shape
    batch2_sizes = batch2.shape
    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: (
            f"Expected size for first two dimensions of batch2 tensor to be: "
            f"[{bs}, {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}]."
        ),
    )
    return self.new_empty(self.size())


@register_meta([aten.bernoulli.default, aten.bernoulli.out])
@out_wrapper()
def meta_bernoulli(self, *, generator=None):
    # https://github.com/pytorch/pytorch/issues/88612
    return torch.empty_like(self).contiguous()


@register_meta(aten.bernoulli_.float)
def meta_bernoulli_(self, p=0.5, generator=None):
    return self


@register_meta(aten.bernoulli.p)
def meta_bernoulli_p(self, p=0.5, generator=None):
    # https://github.com/pytorch/pytorch/issues/88612
    return torch.empty_like(self).contiguous()


@register_meta(aten._fused_moving_avg_obs_fq_helper.default)
def meta__fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    check(
        ch_axis < self.dim(),
        lambda: "Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()",
    )
    mask = torch.empty_like(self, dtype=torch.bool)
    return (torch.empty_like(self), mask)


def dot_check(self, other):
    check(
        self.dim() == 1 and other.dim() == 1,
        lambda: f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors",
    )


@register_meta(aten.dot.default)
def meta_dot(self, tensor):
    dot_check(self, tensor)
    return self.new_empty(())


@register_meta([aten.mm.default])
def meta_mm(a, b):
    check(a.dim() == 2, lambda: "a must be 2D")
    check(b.dim() == 2, lambda: "b must be 2D")
    N, M1 = a.shape
    M2, P = b.shape
    check(
        M1 == M2,
        lambda: f"a and b must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",
    )
    return a.new_empty(N, P)


def _compute_reduction_shape(self, dims, keepdim):
    if keepdim:
        return tuple(self.shape[i] if i not in dims else 1 for i in range(self.ndim))

    return utils.compute_reduction_output_shape(self.shape, dims)


# FakeTensors (meta tensors with a device) will report device as meta
# when running meta kernels. Here, access the "fake device" of FakeTensor if it
# exists so meta kernels which have diverge per device will be more
# accurate when run with FakeTensors
def device_hint(tensor) -> "str":
    if isinstance(tensor, torch._subclasses.FakeTensor):
        return tensor.fake_device.type
    else:
        return "cuda"  # default to cuda


def calc_conv_nd_return_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: Union[List[int], int],
    padding: Union[List[int], int],
    dilation: Union[List[int], int],
    is_transposed: bool,
    groups: int,
    output_padding: Optional[Union[List[int], int]] = None,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    kernel_size = weight.shape[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight.shape[1]
    else:
        out_channels = weight.shape[0]
        if weight.shape[1] * groups != input_tensor.shape[1]:
            raise RuntimeError("Invalid channel dimensions")

    ret_shape = [input_tensor.shape[0], out_channels]
    if isinstance(stride, IntLike):
        stride = [stride] * len(dims)
    elif len(stride) == 1:
        stride = [stride[0]] * len(dims)

    if isinstance(padding, IntLike):
        padding = [padding] * len(dims)
    elif len(padding) == 1:
        padding = [padding[0]] * len(dims)

    if isinstance(dilation, IntLike):
        dilation = [dilation] * len(dims)
    elif len(dilation) == 1:
        dilation = [dilation[0]] * len(dims)

    output_padding_list: Optional[List[int]] = None
    if output_padding:
        if isinstance(output_padding, IntLike):
            output_padding_list = [output_padding] * len(dims)
        elif len(output_padding) == 1:
            output_padding_list = [output_padding[0]] * len(dims)
        else:
            output_padding_list = output_padding

    for i in range(len(dims)):
        # If output_padding is present, we are dealing with a transposed convolution
        if output_padding_list:
            ret_shape.append(
                _formula_transposed(
                    dims[i],
                    padding[i],
                    dilation[i],
                    kernel_size[i],
                    stride[i],
                    output_padding_list[i],
                )
            )
        else:
            ret_shape.append(
                _formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i])
            )

    return ret_shape


def is_channels_last(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last


@register_meta(aten.convolution.default)
def meta_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    is_transposed: bool,
    output_padding: List[int],
    groups: int,
):
    def pick_memory_format():
        if device_hint(input_tensor) == "cuda":
            if is_channels_last(input_tensor) or is_channels_last(weight):
                return torch.channels_last
        else:
            if is_channels_last(input_tensor):
                return torch.channels_last
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif input_tensor.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    shape_out = calc_conv_nd_return_shape(
        input_tensor,
        weight,
        stride,
        padding,
        dilation,
        is_transposed,
        groups,
        output_padding if is_transposed else None,
    )

    out = input_tensor.new_empty(shape_out)
    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


if torch._C._has_mkldnn:
    _meta_lib_dont_use_me_use_register_meta_for_mkldnn = torch.library.Library(
        "mkldnn", "IMPL", "Meta"
    )

    @register_meta(torch.ops.mkldnn._convolution_pointwise.default)
    def meta_mkldnn_convolution_default(
        input_tensor,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
        attr,
        scalars,
        algorithm,
    ):
        shape_out = calc_conv_nd_return_shape(
            input_tensor, weight, stride, padding, dilation, False, groups, []
        )
        out = input_tensor.new_empty(shape_out)
        out_memory_format = torch.channels_last
        out = out.to(memory_format=out_memory_format)  # type: ignore[call-overload]
        return out

    @register_meta(torch.ops.mkldnn._linear_pointwise.default)
    def meta_linear_pointwise_default(
        input_tensor, weight, bias, attr, scalars, algorithm
    ):
        return input_tensor.new_empty((*input_tensor.shape[:-1], weight.shape[0]))

    if torch._C.has_mkl:
        _meta_lib_dont_use_me_use_register_meta_for_mkl = torch.library.Library(
            "mkl", "IMPL", "Meta"
        )

        @register_meta(torch.ops.mkl._mkl_linear)
        def meta_mkl_linear(
            input_tensor,
            packed_weight,
            orig_weight,
            bias,
            batch_size,
        ):
            return input_tensor.new_empty(
                (*input_tensor.shape[:-1], orig_weight.shape[0])
            )


# from check_dim_size() in aten/src/ATen/TensorUtils.cpp.
def check_dim_size(tensor, dim, dim_size, size):
    check(
        tensor.dim() == dim and tensor.shape[dim_size] == size,
        lambda: f"Expected a tensor of dimension {dim} and tensor.size[{dim_size}] == {size}, "
        + f"but got : dimension {tensor.dim()} and tensor.size[{dim_size}] = {tensor.shape[dim_size]}",
    )


@register_meta(aten.avg_pool2d.default)
def meta_avg_pool2d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    def unpack(name, val):
        check(
            len(val) in [1, 2],
            lambda: f"avg_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)
    check(
        len(stride) in [0, 1, 2],
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    elif len(stride) == 1:
        dH, dW = stride[0], stride[0]
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)

    check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    memory_format = utils.suggest_memory_format(input)
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        memory_format,
    )

    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return torch.empty(
        size,
        dtype=input.dtype,
        device=input.device,
        memory_format=memory_format,
    )


# from avg_pool2d_backward_shape_check() in aten/src/ATen/native/Pool.h.
def avg_pool2d_backward_shape_check(
    input,
    gradOutput,
    nbatch,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    mem_format,
):
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    ndim = input.dim()
    nOutputPlane = nInputPlane

    check_dim_size(gradOutput, ndim, ndim - 3, nOutputPlane)
    check_dim_size(gradOutput, ndim, ndim - 2, outputHeight)
    check_dim_size(gradOutput, ndim, ndim - 1, outputWidth)


# Don't override the C++ registration.
@register_meta(aten.avg_pool2d_backward.default)
def meta_avg_pool2d_backward(
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # From aten/src/ATen/native/AveragePool2d.cpp structured kernel meta func.
    check(
        len(kernel_size) == 1 or len(kernel_size) == 2,
        lambda: "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints",
    )
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]
    check(
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]
    check(
        len(padding) == 1 or len(padding) == 2,
        lambda: "avg_pool2d: padding must either be a single int, or a tuple of two ints",
    )
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    input_size = input.shape
    nbatch = input_size[-4] if input.dim() == 4 else 1
    nInputPlane = input_size[-3]
    inputHeight = input_size[-2]
    inputWidth = input_size[-1]

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    mem_format = utils.suggest_memory_format(input)

    avg_pool2d_backward_shape_check(
        input,
        gradOutput_,
        nbatch,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    return torch.empty(
        input_size,
        dtype=input.dtype,
        device=input.device,
        memory_format=mem_format,
    )


@register_meta(aten._adaptive_avg_pool2d.default)
def meta_adaptive_avg_pool2d(self, output_size):
    check(
        self.ndim == 3 or self.ndim == 4,
        lambda: f"Expected 3D or 4D tensor, but got {self.shape}",
    )
    output_shape = self.shape[:-2] + tuple(output_size)
    memory_format = utils.suggest_memory_format(self)
    # need to set memory_format to preserve the memory format of the input
    # channel last input should have channel last output
    return torch.empty(
        output_shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


@register_meta(aten._adaptive_avg_pool3d.default)
def meta_adaptive_avg_pool3d(self, output_size):
    check(
        self.ndim == 4 or self.ndim == 5,
        lambda: f"Expected 4D or 5D tensor, but got {self.shape}",
    )
    return self.new_empty(self.shape[:-3] + tuple(output_size))


@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta__adaptive_avg_pool2d_backward(grad_out, self):
    ndim = grad_out.ndim
    for i in range(1, ndim):
        check(
            grad_out.size(i) > 0,
            lambda: f"adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero \
                      size for non-batch dimensions, {grad_out.shape} with dimension {i} being empty",
        )
    check(
        ndim == 3 or ndim == 4,
        lambda: f"adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got {self.shape}",
    )
    check(
        self.dtype == grad_out.dtype,
        lambda: f"expected dtype {self.dtype} for `grad_output` but got dtype {grad_out.dtype}",
    )
    memory_format = torch.contiguous_format
    if is_channels_last(self):
        memory_format = torch.channels_last
    return self.new_empty(self.shape).to(memory_format=memory_format)


@register_meta(aten.repeat_interleave.Tensor)
def meta_repeat_interleave_Tensor(repeats, output_size=None):
    if output_size is None:
        raise RuntimeError("cannot repeat_interleave a meta tensor without output_size")
    return repeats.new_empty(output_size)


@register_meta([aten.complex.default, aten.complex.out])
@out_wrapper()
def meta_complex(real, imag):
    assert real.dtype.is_floating_point
    assert imag.dtype.is_floating_point
    out_shape = _broadcast_shapes(real.shape, imag.shape)
    return real.new_empty(out_shape, dtype=corresponding_complex_dtype(real.dtype))


@register_meta(aten.view.dtype)
def view_dtype(self, dtype):
    return utils.clone_preserve_strides(self).to(dtype)


@register_meta(aten.vdot.default)
def vdot(self, other):
    if not self.is_complex:
        return torch.dot(self, other)

    if self.is_conj():
        if other.is_conj():
            return torch.vdot(other.conj(), self.conj())
        else:
            return torch.dot(self.conj(), other)
    elif other.is_conj():
        return torch.dot(self, other.conj()).conj()

    dot_check(self, other)
    return self.new_empty(())


@register_meta([aten.nonzero_static.default, aten.nonzero_static.out])
def nonzero_static(self, *, size: int, fill_value: int = -1):
    return self.new_empty((size, self.dim()), dtype=torch.long)


@register_meta([aten.index.Tensor, aten._unsafe_index.Tensor])
def meta_index_Tensor(self, indices):
    check(indices, lambda: "at least one index must be provided")
    # aten::index is the internal advanced indexing implementation
    # checkIndexTensorTypes and expandTensors
    result: List[Optional[Tensor]] = []
    for i, index in enumerate(indices):
        if index is not None:
            check(
                index.dtype in [torch.long, torch.int, torch.int8, torch.bool],
                lambda: "tensors used as indices must be long, int, byte or bool tensors",
            )
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(result)
                check(
                    k + index.ndim <= self.ndim,
                    lambda: f"too many indices for tensor of dimension {self.ndim}",
                    IndexError,
                )
                for j in range(index.ndim):
                    check(
                        index.shape[j] == self.shape[k + j],
                        lambda: f"The shape of the mask {index.shape} at index {i} "
                        f"does not match the shape of the indexed tensor {self.shape} at index {k + j}",
                        IndexError,
                    )
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
    indices = result
    check(
        len(indices) <= self.ndim,
        lambda: f"too many indices for tensor of dimension {self.ndim} (got {len(indices)})",
    )
    # expand_outplace
    import torch._refs as refs  # avoid import cycle in mypy

    indices = list(refs._maybe_broadcast(*indices))
    # add missing null tensors
    while len(indices) < self.ndim:
        indices.append(None)

    # hasContiguousSubspace
    #   true if all non-null tensors are adjacent
    # See:
    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # https://stackoverflow.com/questions/53841497/why-does-numpy-mixed-basic-advanced-indexing-depend-on-slice-adjacency
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # transposeToFront
    # This is the logic that causes the newly inserted dimensions to show up
    # at the beginning of the tensor, if they're not contiguous
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        self = self.permute(dims)
        indices = transposed_indices

    # AdvancedIndex::AdvancedIndex
    # Now we can assume the indices have contiguous subspace
    # This is simplified from AdvancedIndex which goes to more effort
    # to put the input and indices in a form so that TensorIterator can
    # take them.  If we write a ref for this, probably that logic should
    # get implemented
    before_shape: List[int] = []
    after_shape: List[int] = []
    replacement_shape: List[int] = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            replacement_shape = list(index.shape)
    return self.new_empty(before_shape + replacement_shape + after_shape)


@register_meta([aten.convolution_backward.default])
def meta_convolution_backward(
    grad_output_,
    input_,
    weight_,
    bias_sizes_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    # High level logic taken from slow_conv3d_backward_cpu which should
    # be representative of all convolution_backward impls
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if output_mask[0]:
        backend_grad_input = grad_output_.new_empty(input_.size())
    if output_mask[1]:
        backend_grad_weight = grad_output_.new_empty(weight_.size())
    if output_mask[2]:
        backend_grad_bias = grad_output_.new_empty(bias_sizes_opt)

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta([aten.addbmm.default, aten.addbmm.out])
@out_wrapper()
def meta_addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(1)
    dim2 = batch2.size(2)
    self = self.expand((dim1, dim2))
    check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    check(
        batch1.size(0) == batch2.size(0),
        lambda: f"batch1 and batch2 must have same number of batches, got {batch1.size(0)} and {batch2.size(0)}",
    )
    check(
        batch1.size(2) == batch2.size(1),
        lambda: (
            f"Incompatible matrix sizes for bmm ({batch1.size(1)}x{batch1.size(2)} "
            f"and {batch2.size(1)}x{batch2.size(2)})"
        ),
    )
    check(
        self.size(0) == dim1 and self.size(1) == dim2,
        lambda: "self tensor does not match matmul output shape",
    )
    return self.new_empty(self.size())


@register_meta(
    [
        aten._foreach_neg_.default,
        aten._foreach_reciprocal_.default,
    ]
)
def meta__foreach_unaop_(self):
    check(
        isinstance(self, List),
        lambda: f"Expect List[Tensor] but got {type(self)}",
    )


@register_meta(
    [
        aten._foreach_neg.default,
        aten._foreach_reciprocal.default,
        aten._foreach_sqrt.default,
    ]
)
def meta__foreach_unaop(self):
    check(
        isinstance(self, List),
        lambda: f"Expect List[Tensor] but got {type(self)}",
    )
    return [torch.empty_like(s) for s in self]


def _check_foreach_binop_tensor_lists(self, other):
    check(
        isinstance(self, List) and isinstance(other, List),
        lambda: (
            "The first two arguments of must be List[Tensor], "
            f"but got {type(self)} and {type(other)}."
        ),
    )
    check(
        len(self) > 0 and len(self) == len(other),
        lambda: (
            "self and other must be non-empty and match in length, "
            f"but got {len(self)} and {len(other)}."
        ),
    )


@register_meta([aten._foreach_add.List])
def meta__foreach_add(self, other, alpha=1):
    _check_foreach_binop_tensor_lists(self, other)
    return [torch.empty_like(s) for s in self]


@register_meta([aten._foreach_add_.List])
def meta__foreach_add__list(self, other, alpha=1):
    _check_foreach_binop_tensor_lists(self, other)


@register_meta([aten._foreach_div_.List])
def meta__foreach_binop__list(self, other):
    _check_foreach_binop_tensor_lists(self, other)


@register_meta(
    [
        aten._foreach_div.List,
        aten._foreach_mul.List,
    ]
)
def meta__foreach_binop_list(self, other):
    _check_foreach_binop_tensor_lists(self, other)
    return [torch.empty_like(s) for s in self]


@register_meta(
    [
        aten._foreach_add_.Scalar,
        aten._foreach_mul_.Scalar,
        aten._foreach_sub_.Scalar,
    ]
)
def meta__foreach_binop__scalar(self, scalar=1):
    check(
        isinstance(self, List),
        lambda: f"The first argument of must be List[Tensor], but got {type(self)}.",
    )


@register_meta(
    [
        aten._foreach_add.Scalar,
        aten._foreach_div.Scalar,
        aten._foreach_mul.Scalar,
        aten._foreach_sub.Scalar,
    ]
)
def meta__foreach_binop_scalar(self, scalar=1):
    check(
        isinstance(self, List),
        lambda: f"The first argument of must be List[Tensor], but got {type(self)}.",
    )
    return [torch.empty_like(s) for s in self]


@register_meta(
    [
        aten._foreach_addcdiv_.Scalar,
        aten._foreach_addcmul_.Scalar,
    ]
)
def meta__foreach_addcop__scalar(self, tensor1, tensor2, scalar=1):
    check(
        all(isinstance(l, List) for l in [self, tensor1, tensor2]),
        lambda: (
            "All arguments of _foreach_addc*_ must be List[Tensor], "
            f"but got {type(self)}, {type(tensor1)}, and {type(tensor2)}"
        ),
    )
    check(len(self) > 0, lambda: "input tensor list must not be empty.")
    check(
        len(self) == len(tensor1) and len(self) == len(tensor2),
        lambda: "All input tensor lists must have the same length",
    )


@register_meta(
    [
        aten._foreach_addcdiv.Scalar,
        aten._foreach_addcmul.Scalar,
    ]
)
def meta__foreach_addcop_scalar(self, tensor1, tensor2, scalar=1):
    check(
        all(isinstance(l, List) for l in [self, tensor1, tensor2]),
        lambda: (
            "All arguments must be List[Tensor], "
            f"but got {type(self)}, {type(tensor1)}, and {type(tensor2)}"
        ),
    )
    check(len(self) > 0, lambda: "input tensor list must not be empty.")
    check(
        len(self) == len(tensor1) and len(self) == len(tensor2),
        lambda: "All input tensor lists must have the same length",
    )

    return [torch.empty_like(s) for s in self]


@register_meta([aten._foreach_pow.ScalarAndTensor])
def meta__foreach_pow_scalar_and_tensor(self, exponent):
    check(
        isinstance(exponent, List),
        lambda: f"exponent must be a tensor list but got {type(exponent)}",
    )
    return [torch.empty_like(e) for e in exponent]


@register_meta([aten._foreach_addcdiv_.Tensor, aten._foreach_addcmul_.Tensor])
def meta__foreach_addcop_tensor(self, tensor1, tensor2, scalars):
    check(
        all(isinstance(l, List) for l in [self, tensor1, tensor2])
        and isinstance(scalars, torch.Tensor),
        lambda: (
            "_foreach_addc*_ op expects arguments of type: List[Tensor], List[Tensor], List[Tensor], tensor, "
            f"but got: {type(self)}, {type(tensor1)}, {type(tensor2)}, and {type(scalars)}"
        ),
    )
    check(len(self) > 0, lambda: "input tensor list must not be empty.")
    check(
        len(self) == len(tensor1) and len(self) == len(tensor2),
        lambda: "All input tensor lists must have the same length",
    )


@register_meta([aten._fused_adam_.default])
def meta__fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    for l in [self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]:
        check(
            isinstance(l, List),
            lambda: f"exponent must be a tensor list but got {type(l)}",
        )


@register_meta([aten._fused_adam.default])
def meta__fused_adam(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    for l in [self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]:
        check(
            isinstance(l, List),
            lambda: f"exponent must be a tensor list but got {type(l)}",
        )

    def empty_like_list(tensor_list):
        return [torch.empty_like(t) for t in tensor_list]

    return (
        empty_like_list(self),
        empty_like_list(grads),
        empty_like_list(exp_avgs),
        empty_like_list(exp_avg_sqs),
        empty_like_list(max_exp_avg_sqs),
    )


@register_meta([aten._int_mm])
@out_wrapper()
def meta__int_mm(a, b):
    check(a.dim() == 2, lambda: "a must be a 2D tensor")
    check(b.dim() == 2, lambda: "b must be a 2D tensor")
    check(
        a.dtype is torch.int8,
        lambda: f"expected self to be int8, got {a.dtype}",
    )
    check(
        b.dtype is torch.int8,
        lambda: f"expected mat2 to be int8, got {b.dtype}",
    )
    check(
        a.size(1) == b.size(0),
        lambda: (
            f"Incompatible matrix sizes for _int_mm ({a.size(0)}x{a.size(1)} "
            f"and {b.size(0)}x{b.size(1)})"
        ),
    )
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.int32)


@register_meta(aten._cdist_forward.default)
def meta_cdist_forward(x1, x2, p, compute_mode):
    check(
        x1.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X1 got: {x1.dim()}D",
    )
    check(
        x2.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X2 got: {x2.dim()}D",
    )
    check(
        x1.size(-1) == x2.size(-1),
        lambda: f"X1 and X2 must have the same number of columns. X1: {x1.size(-1)} X2: {x2.size(-1)}",
    )
    check(
        utils.is_float_dtype(x1.dtype),
        lambda: "cdist only supports floating-point dtypes, X1 got: {x1.dtype}",
    )
    check(
        utils.is_float_dtype(x2.dtype),
        lambda: "cdist only supports floating-point dtypes, X2 got: {x2.dtype}",
    )
    check(p >= 0, lambda: "cdist only supports non-negative p values")
    check(
        compute_mode in (None, 1, 2),
        lambda: f"possible modes: None, 1, 2, but was: {compute_mode}",
    )
    r1 = x1.size(-2)
    r2 = x2.size(-2)
    batch_tensor1 = x1.shape[:-2]
    batch_tensor2 = x2.shape[:-2]
    output_shape = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    output_shape.extend([r1, r2])
    return x1.new_empty(output_shape)


@register_meta(aten._embedding_bag.default)
def meta_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    check(
        indices.dtype in (torch.long, torch.int),
        lambda: f"expected indices to be long or int, got {indices.dtype}",
    )
    check(
        offsets.dtype in (torch.long, torch.int),
        lambda: f"expected offsets to be long or int, got {offsets.dtype}",
    )
    check(
        utils.is_float_dtype(weight.dtype),
        lambda: f"expected weight to be floating point type, got {weight.dtype}",
    )

    num_bags = offsets.size(0)
    if include_last_offset:
        check(
            num_bags >= 1,
            lambda: "include_last_offset: numBags should be at least 1",
        )
        num_bags -= 1

    output = weight.new_empty(num_bags, weight.size(1))
    MODE_SUM, MODE_MEAN, MODE_MAX = range(3)

    if per_sample_weights is not None:
        check(
            mode == MODE_SUM,
            lambda: "embedding_bag: per_sample_weights only supported with mode='sum'",
        )
        check(
            per_sample_weights.dtype == weight.dtype,
            lambda: f"expected weight ({weight.dtype}) and per_sample_weights ({per_sample_weights.dtype}) to have same dtype",
        )
        check(
            per_sample_weights.ndim == 1,
            lambda: f"expected per_sample_weights to be 1D tensor, got {per_sample_weights.ndim}D",
        )
        check(
            per_sample_weights.numel() == indices.numel(),
            lambda: (
                f"expected per_sample_weights.numel() ({per_sample_weights.numel()} "
                f"to be the same as indices.numel() ({indices.numel()})"
            ),
        )

    def is_fast_path_index_select_scale(src, scale, output, padding_idx):
        return (
            is_fast_path_index_select(src, output, padding_idx) and scale.stride(0) == 1
        )

    def is_fast_path_index_select(src, output, padding_idx):
        return (
            (src.dtype == torch.float or src.dtype == torch.half)
            and src.stride(1) == 1
            and output.stride(1) == 1
            and padding_idx < 0
        )

    def is_fast_path(src, scale, output, padding_idx):
        if scale is not None:
            return is_fast_path_index_select_scale(src, scale, output, padding_idx)
        else:
            return is_fast_path_index_select(src, output, padding_idx)

    if device_hint(offsets) != "cpu":
        offset2bag = indices.new_empty(indices.size(0))
        bag_size = indices.new_empty(offsets.size())
        if mode == MODE_MAX:
            max_indices = indices.new_empty(num_bags, weight.size(1))
        else:
            max_indices = indices.new_empty(0)
    else:
        fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx)
        if mode in (MODE_MEAN, MODE_MAX) or not fast_path_sum:
            offset2bag = offsets.new_empty(indices.size(0))
        else:
            offset2bag = offsets.new_empty(0)
        bag_size = offsets.new_empty(num_bags)
        # This part of the logic comes from make_max_indices_out in EmbeddingBag.cpp
        numBags = offsets.shape[0]
        if mode == MODE_MAX:
            if include_last_offset:
                check(
                    numBags >= 1,
                    lambda: "include_last_offset: numBags should be at least 1",
                )
                numBags -= 1
            max_indices = offsets.new_empty(numBags, weight.shape[1])
        else:
            max_indices = offsets.new_empty(bag_size.size())
    return output, offset2bag, bag_size, max_indices


@register_meta(aten._embedding_bag_forward_only.default)
def meta_embedding_bag_forward_only(weight, indices, offsets, *args):
    output, offset2bag, bag_size, max_indices = meta_embedding_bag(
        weight, indices, offsets, *args
    )
    if device_hint(offsets) == "cpu":
        bag_size = offsets.new_empty(offsets.size())
    return output, offset2bag, bag_size, max_indices


def _get_reduction_dtype(input, dtype, promote_int_to_long=True):
    # if specified, dtype takes precedence
    if dtype:
        return dtype

    if input.dtype.is_floating_point or input.dtype.is_complex:
        return input.dtype
    elif promote_int_to_long:
        return torch.long

    return input.dtype


@register_meta([aten.nansum.default, aten.nansum.out])
@out_wrapper()
def meta_nansum(input, dims=None, keepdim=False, *, dtype=None):
    output_dtype = _get_reduction_dtype(input, dtype, promote_int_to_long=True)
    dims = utils.reduction_dims(input.shape, dims)
    output_shape = _compute_reduction_shape(input, dims, keepdim)
    return input.new_empty(output_shape, dtype=output_dtype)


@register_meta(aten.nanmedian.default)
def meta_nanmedian(input):
    output_shape = utils.compute_reduction_output_shape(
        input.shape, tuple(range(input.dim()))
    )
    return input.new_empty(output_shape)


@register_meta([aten.nanmedian.dim, aten.nanmedian.dim_values])
@out_wrapper("values", "indices")
def meta_nanmedian_dim(input, dim=-1, keepdim=False):
    dim = utils.reduction_dims(input.shape, (dim,))
    output_shape = _compute_reduction_shape(input, dim, keepdim)
    return (
        input.new_empty(output_shape),
        input.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.logical_not_.default)
def meta_logical_not_(self):
    return self


@register_meta(aten.repeat.default)
def meta_repeat(self, repeats):
    check(
        len(repeats) >= self.dim(),
        lambda: "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )
    # Add new leading dimensions to the tensor if the
    # number of target dimensions is larger than the
    # number of source dimensions.
    num_new_dimensions = len(repeats) - self.dim()
    padded_size = (1,) * num_new_dimensions + tuple(self.shape)
    target_size = [padded_size[i] * repeats[i] for i in range(len(repeats))]
    return self.new_empty(target_size)


@register_meta(aten.zero_.default)
def meta_zero_(self):
    return self


@register_meta(
    [
        aten.mul_.Scalar,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.logical_and_.default,
        aten.logical_or_.default,
        aten.logical_xor_.default,
    ],
)
def meta_binop_inplace(self, other):
    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    return self


@register_meta(
    [
        aten.add_.Scalar,
        aten.sub_.Scalar,
        aten.add_.Tensor,
        aten.sub_.Tensor,
    ],
)
def meta_binop_inplace_alpha(self, other, alpha=1):
    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    return self


@register_meta([aten.round.default, aten.round.decimals])
def meta_round(self, **kwargs):
    return _elementwise_meta(
        self, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


@register_meta(aten.zero.default)
def meta_zero(self):
    return self.new_empty(self.shape)


@register_meta([aten.fill_.Tensor, aten.fill_.Scalar])
def meta_fill_(self, val):
    return self


@register_meta([aten.fill.Tensor, aten.fill.Scalar])
def meta_fill(self, val):
    return torch.empty_like(self)


@register_meta(aten.relu_.default)
def meta_relu_(self):
    return self


@register_meta([aten.index_put.default, aten._unsafe_index_put.default])
def meta_index_put(self, indices, values, accumulate=False):
    return torch.empty_like(self)


@register_meta(aten.masked_fill_.Scalar)
def meta_masked_fill_(self, mask, value):
    check_inplace_broadcast(self.shape, mask.shape)
    return self


@register_meta(aten.index_put_.default)
def meta_index_put_(self, indices, values, accumulate=False):
    return self


@register_meta(aten.alias.default)
def meta_alias(self):
    return self.view(self.shape)


def common_meta_baddbmm_bmm(batch1, batch2, is_bmm, self_baddbmm=None):
    check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")

    batch1_sizes = batch1.size()
    batch2_sizes = batch2.size()

    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    res_rows = batch1_sizes[1]
    res_cols = batch2_sizes[2]
    output_size = (bs, res_rows, res_cols)

    check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: f"Expected size for first two dimensions of batch2 tensor to be: [{bs}"
        f", {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].",
    )

    # TODO: handle out

    output = batch2.new_empty(output_size)

    if not is_bmm and self_baddbmm is not None:
        check(self_baddbmm.dim() == 3, lambda: "self must be a 3D tensor")
        check(
            self_baddbmm.size() == output_size,
            lambda: f"Expected an input tensor shape with shape {output_size} but got shape: {self_baddbmm.size()}",
        )

    return output


@register_meta(aten.bmm.default)
def meta_bmm(self, mat2):
    return common_meta_baddbmm_bmm(self, mat2, True)


def div_rtn(x, y):
    q = x // y
    r = x % y
    # WARNING: explicit bool conversion here is necessary;
    # would be fixed by SymBool
    if r != 0 and (bool(r < 0) != bool(y < 0)):
        q -= 1
    return q


def pooling_output_shape_pad_lr(
    inputSize, kernelSize, pad_l, pad_r, stride, dilation, ceil_mode
):
    outputSize = (
        div_rtn(
            inputSize
            + pad_l
            + pad_r
            - dilation * (kernelSize - 1)
            - 1
            + (stride - 1 if ceil_mode else 0),
            stride,
        )
        + 1
    )
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize -= 1
    return outputSize


def pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, ceil_mode):
    check(stride != 0, lambda: "stride should not be zero")
    check(pad >= 0, lambda: f"pad must be non-negative, but got pad: {pad}")
    check(
        pad <= kernelSize // 2,
        lambda: f"pad should be at most half of kernel size, but got pad={pad} and kernel_size={kernelSize}",
    )
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode
    )


def pool2d_shape_check(
    input,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    dilationH,
    dilationW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    memory_format,
):
    ndim = input.dim()
    nOutputPlane = nInputPlane

    check(
        kW > 0 and kH > 0,
        lambda: "kernel size should be greater than zero, but got kH: {kH}, kW: {kW}",
    )
    check(
        dW > 0 and dH > 0,
        lambda: "stride should be greater than zero, but got dH: {dH}, dW: {dW}",
    )
    check(
        dilationH > 0 and dilationW > 0,
        lambda: "dilation should be greater than zero, but got dilationH: {dilationH}, dilationW: {dilationW}",
    )

    valid_dims = input.size(1) != 0 and input.size(2) != 0

    if memory_format == torch.channels_last:
        check(
            ndim == 4 and valid_dims and input.size(3) != 0,
            lambda: "Expected 4D (batch mode) tensor expected for input with channels_last layout"
            " with optional 0 dim batch size for input, but got: {input.size()}",
        )
    else:
        check(
            (ndim == 3 and input.size(0) != 0 and valid_dims)
            or (ndim == 4 and valid_dims and input.size(3) != 0),
            lambda: f"Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got: {input.size()}",
        )

    check(
        kW // 2 >= padW and kH // 2 >= padH,
        lambda: "pad should be smaller than or equal to half of kernel size, but got "
        f"padW = {padW}, padH = {padH}, kW = {kW}, kH = {kH}",
    )

    check(
        outputWidth >= 1 and outputHeight >= 1,
        lambda: f"Given input size: ({nInputPlane}x{inputHeight}x{inputWidth}). "
        f"Calculated output size: ({nOutputPlane}x{outputHeight}x{outputWidth}). "
        "Output size is too small",
    )


def max_pool2d_checks_and_compute_shape(
    input, kernel_size, stride, padding, dilation, ceil_mode
):
    # Reference: aten/src/ATen/native/DilatedMaxPool2d.cpp
    def unpack(name, val):
        check(
            len(val) in [1, 2],
            lambda: f"max_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)

    check(
        len(stride) in [0, 1, 2],
        lambda: "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)
    dilationH, dilationW = unpack("dilation", dilation)
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    memory_format = utils.suggest_memory_format(input)
    if memory_format == torch.channels_last:
        check(
            input.dim() == 4,
            lambda: "non-empty 4D (batch mode) tensor expected for input with channels_last layout",
        )
    elif memory_format == torch.contiguous_format:
        check(
            input.dim() in [3, 4],
            lambda: "non-empty 3D or 4D (batch mode) tensor expected for input",
        )
    else:
        check(
            False,
            lambda: "Unsupport memory format. Supports only ChannelsLast, Contiguous",
        )

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, dilationW, ceil_mode)

    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        dilationH,
        dilationW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        memory_format,
    )

    return nInputPlane, outputHeight, outputWidth


@register_meta(aten.max_pool2d_with_indices_backward.default)
def meta_max_pool2d_with_indices_backward(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    (
        nInputPlane,
        outputHeight,
        outputWidth,
    ) = max_pool2d_checks_and_compute_shape(
        self, kernel_size, stride, padding, dilation, ceil_mode
    )

    check(
        self.dtype == grad_output.dtype,
        lambda: f"Expected dtype {self.dtype} for `gradOutput` but got dtype {grad_output.dtype}",
    )

    nOutputPlane = nInputPlane
    ndim = self.ndim

    def _check_dim_size(t):
        check_dim_size(t, ndim, ndim - 3, nOutputPlane)
        check_dim_size(t, ndim, ndim - 2, outputHeight)
        check_dim_size(t, ndim, ndim - 1, outputWidth)

    _check_dim_size(grad_output)
    _check_dim_size(indices)

    memory_format = utils.suggest_memory_format(self)
    return torch.empty(
        self.shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


@register_meta(aten.max_pool2d_with_indices.default)
def meta_max_pool2d_with_indices(
    input, kernel_size, stride=(), padding=(0,), dilation=(1,), ceil_mode=False
):
    (
        nInputPlane,
        outputHeight,
        outputWidth,
    ) = max_pool2d_checks_and_compute_shape(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    memory_format = utils.suggest_memory_format(input)
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return (
        torch.empty(
            size,
            dtype=input.dtype,
            device=input.device,
            memory_format=memory_format,
        ),
        torch.empty(
            size,
            dtype=torch.int64,
            device=input.device,
            memory_format=memory_format,
        ),
    )


@register_meta(aten.grid_sampler_2d_backward.default)
def grid_sampler_2d_backward_meta(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format)
    else:
        grad_input = None
    grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format)
    return (grad_input, grad_grid)


@register_meta([aten.full.default])
def full(size, fill_value, *args, **kwargs):
    return torch.empty(size, *args, **kwargs)


# zeros_like is special cased to work for sparse
@register_meta(aten.zeros_like.default)
def zeros_like(
    self,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    if layout == torch.sparse_coo:
        check(
            memory_format is None,
            lambda: "memory format option is only supported by strided tensors",
        )

        res = torch.empty(
            0,
            dtype=self.dtype if dtype is None else dtype,
            layout=layout,
            device=self.device if device is None else device,
            pin_memory=pin_memory,
        )

        if self.is_sparse:
            res.sparse_resize_and_clear_(
                self.size(), self.sparse_dim(), self.dense_dim()
            )
        else:
            res.sparse_resize_and_clear_(self.size(), self.dim(), 0)

        res._coalesced_(True)
        return res
    res = aten.empty_like.default(
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
    # device can be not "meta"
    res.fill_(0)
    return res


@register_meta(aten.select.int)
def meta_select(self, dim, index):
    ndim = self.dim()
    check(
        ndim != 0,
        lambda: "select() cannot be applied to a 0-dim tensor.",
        IndexError,
    )

    dim = dim if dim >= 0 else dim + ndim
    size = self.size(dim)

    check(
        not (-index > size or index >= size),
        lambda: f"select(): index {index} out of range for tensor of size "
        f"{self.size()} at dimension {dim}",
        IndexError,
    )

    index = index if index >= 0 else index + size

    new_size = list(self.size())
    new_stride = list(self.stride())

    new_storage_offset = self.storage_offset() + index * new_stride[dim]
    del new_size[dim]
    del new_stride[dim]

    return self.as_strided(new_size, new_stride, new_storage_offset)


@register_meta(aten.select_scatter.default)
def meta_select_scatter(self, src, dim, index):
    return utils.clone_preserve_strides(self)


@register_meta(aten.slice_scatter.default)
def meta_slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    return utils.clone_preserve_strides(self)


# TODO: Deduplicate this with canonicalize_dim
def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max), f"dim {dim} out of bounds ({min}, {max})"
    if dim < 0:
        dim += dim_post_expr
    return dim


def ensure_nonempty_size(t, dim):
    return 1 if t.dim() == 0 else t.shape[dim]


# From aten/src/ATen/native/ScatterGatherChecks.h
def gather_shape_check(self, dim, index):
    self_dims = max(self.dim(), 1)
    index_dims = max(index.dim(), 1)
    check(
        self_dims == index_dims,
        lambda: "Index tensor must have the same number of dimensions as input tensor",
    )
    for i in range(self_dims):
        if i != dim:
            check(
                ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
                lambda: f"Size does not match at dimension {i} expected index {index.shape}"
                + f" to be smaller than self {self.shape} apart from dimension {dim}",
            )


@register_meta(aten.gather.default)
def meta_gather(self, dim, index, sparse_grad=False):
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    is_index_empty = index.numel() == 0
    if not is_index_empty:
        check(
            index.dtype == torch.long,
            lambda: f"gather(): Expected dtype int64 for index, but got {index.dtype}",
        )
        gather_shape_check(self, wrapped_dim, index)
    return self.new_empty(index.shape)


# From aten/src/ATen/native/TensorAdvancedIndexing.cpp
def get_operator_enum(reduce_, use_new_options=False):
    if use_new_options:
        if reduce_ == "sum":
            return "REDUCE_ADD"
        elif reduce_ == "prod":
            return "REDUCE_MULTIPLY"
        elif reduce_ == "mean":
            return "REDUCE_MEAN"
        elif reduce_ == "amax":
            return "REDUCE_MAXIMUM"
        elif reduce_ == "amin":
            return "REDUCE_MINIMUM"
        check(
            False,
            lambda: "reduce argument must be either sum, prod, mean, amax or amin.",
        )
        return
    else:
        if reduce_ == "add":
            return "REDUCE_ADD"
        elif reduce_ == "multiply":
            return "REDUCE_MULTIPLY"
        check(False, lambda: "reduce argument must be either add or multiply.")
        return


# From aten/src/ATen/native/ScatterGatherChecks.h
def scatter_gather_dtype_check(method_name, self, index, src_opt=None):
    if index.numel() != 0:
        check(
            index.dtype == torch.long,
            lambda: f"{method_name}(): Expected dtype int64 for index",
        )

    if src_opt is not None:
        check(
            self.dtype == src_opt.dtype,
            lambda: f"{method_name}(): Expected self.dtype to be equal to src.dtype",
        )


def ensure_nonempty_dim(dim):
    return max(dim, 1)


# From aten/src/ATen/native/ScatterGatherChecks.h
def scatter_shape_check(self, dim, index, src_opt=None):
    if index.numel() == 0:
        return
    check(
        ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
        lambda: "Index tensor must have the same number of dimensions as self tensor",
    )

    is_wrong_shape = False
    self_dims = ensure_nonempty_dim(self.dim())

    # Check: index.size(d) <= self.size(d) for all d != dim
    for d in range(self_dims):
        index_d_size = ensure_nonempty_size(index, d)
        if d == dim:
            continue
        if index_d_size > ensure_nonempty_size(self, d):
            is_wrong_shape = True
            break

    # Check: index.size(d) <= src.size(d) for all d if src is Tensor
    if not is_wrong_shape and src_opt is not None:
        for d in range(self_dims):
            index_d_size = ensure_nonempty_size(index, d)
            if index_d_size > ensure_nonempty_size(src_opt, d):
                is_wrong_shape = True
                break

    if src_opt is not None:
        check(
            ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
            lambda: "Index tensor must have the same number of dimensions as self tensor",
        )
        check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be smaller than self {self.shape}"
            + f" apart from dimension {dim} and to be smaller than src {src_opt.shape}",
        )
    else:
        check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be smaller than self {self.shape}"
            + f" apart from dimension {dim}",
        )


# From aten/src/ATen/native/TensorAdvancedIndexing.cpp
def scatter_meta_impl(self, dim, index, src=None, reduce_=None, use_new_options=False):
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    scatter_gather_dtype_check("scatter", self, index, src)
    scatter_shape_check(self, wrapped_dim, index, src)
    if reduce_ is not None:
        # Check if we have a valid reduce operator.
        get_operator_enum(reduce_, use_new_options)


@register_meta(aten.scatter_add.default)
def meta_scatter_add(self, dim, index, src):
    scatter_meta_impl(self, dim, index, src, "add")
    return self.new_empty(self.shape)


@register_meta(aten.scatter_add_)
def meta_scatter_add_(self, dim, index, src):
    scatter_meta_impl(self, dim, index, src, "add")
    return self


@register_meta(
    [
        aten.scatter.src,
        aten.scatter.value,
        aten.scatter.reduce,
        aten.scatter.value_reduce,
    ]
)
@out_wrapper()
def meta_scatter(self, dim, index, src_or_value, reduce=None):
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    scatter_meta_impl(self, dim, index, src, reduce)
    return self.new_empty(self.shape)


@register_meta(
    [
        aten.scatter_.src,
        aten.scatter_.value,
        aten.scatter_.reduce,
        aten.scatter_.value_reduce,
    ]
)
def meta_scatter_(self, dim, index, src_or_value, reduce=None):
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    scatter_meta_impl(self, dim, index, src, reduce)
    return self


@register_meta(
    [
        aten._scaled_dot_product_flash_attention,
    ]
)
def meta__scaled_dot_product_flash(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
):
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_seqlen_batch_q = query.size(2)
    head_dim = query.size(3)

    max_seqlen_batch_k = key.size(2)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    Nnz_q = batch_size * max_seqlen_batch_q

    output = torch.empty(
        (Nnz_q, num_heads, head_dim), dtype=query.dtype, device=query.device
    )
    output = output.view(batch_size, max_seqlen_batch_q, num_heads, head_dim).transpose(
        1, 2
    )
    max_seqlen_q = math.ceil(max_seqlen_batch_q / 16) * 16
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q),
        dtype=torch.float,
        device=query.device,
    )
    cumulative_sequence_length_q = torch.empty(
        batch_size + 1, dtype=torch.int32, device="meta"
    )
    cumulative_sequence_length_k = torch.empty(
        batch_size + 1, dtype=torch.int32, device="meta"
    )

    if return_debug_mask:
        blocksize_c = 128 if head_dim > 64 else 256
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        debug_mask = torch.empty(
            (batch_size, num_heads, max_seqlen_q, max_seqlen_k),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        debug_mask = torch.empty(0, dtype=query.dtype, device=query.device)

    # note: device for seed and offset below depends on whether we are
    # capturing or not, but at the time of tracing we don't know if we
    # are going to use cudagraphs or not, so we return cpu tensors here
    # it's possible we'll need to have some special handling in inductor for sdpa

    return (
        output,
        logsumexp,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        torch.empty((), dtype=torch.long, device="meta"),
        torch.empty((), dtype=torch.long, device="meta"),
        debug_mask,
    )


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_backward,
    ]
)
def meta__scaled_dot_product_flash_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: int,
    philox_offset: int,
    scale: Optional[float] = None,
):
    batch_size = query.size(0)
    num_heads = query.size(1)
    head_dim = query.size(3)

    grad_q = torch.empty_permuted(
        (batch_size, num_heads, max_q, head_dim),
        (0, 2, 1, 3),
        dtype=query.dtype,
        device=query.device,
    )
    grad_k = torch.empty_permuted(
        (batch_size, num_heads, max_k, head_dim),
        (0, 2, 1, 3),
        dtype=key.dtype,
        device=key.device,
    )
    grad_v = torch.empty_permuted(
        (batch_size, num_heads, max_k, head_dim),
        (0, 2, 1, 3),
        dtype=value.dtype,
        device=value.device,
    )

    return grad_q, grad_k, grad_v


@register_meta(
    [
        aten._scaled_dot_product_efficient_attention,
    ]
)
def meta__scaled_dot_product_efficient(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    compute_log_sumexp: bool,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    B = query.size(0)
    M = query.size(1)
    N = key.size(1)
    num_heads = query.size(-2)
    K = query.size(-1)
    Kv = value.size(-1)

    res = torch.empty(B, M, num_heads, Kv, dtype=query.dtype, device=query.device)

    logsumexp_dim = math.ceil(M / 32) * 32 if compute_log_sumexp else 0
    logsum_exp = torch.empty(
        (B, num_heads, logsumexp_dim),
        dtype=torch.float,
        device=query.device,
    )

    res = res.transpose(1, 2)

    return res, logsum_exp


@register_meta(
    [
        aten._scaled_dot_product_efficient_attention_backward,
    ]
)
def meta__scaled_dot_product_efficient_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    is_causal: bool = False,
    chunk_grad_outputs=False,
    scale: Optional[float] = None,
):
    grad_out = grad_out.transpose(1, 2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    B = query.size(0)
    M = query.size(1)
    N = key.size(1)
    nH = query.size(2)
    K = query.size(3)

    grad_kv_needs_init = is_causal and N > M

    grad_q = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    grad_k = (
        torch.zeros(key.shape, dtype=key.dtype, device=key.device)
        if grad_kv_needs_init
        else torch.empty(key.shape, dtype=key.dtype, device=key.device)
    )
    grad_v = (
        torch.zeros(value.shape, dtype=value.dtype, device=value.device)
        if grad_kv_needs_init
        else torch.empty(value.shape, dtype=value.dtype, device=value.device)
    )
    return (
        grad_q.transpose(1, 2),
        grad_k.transpose(1, 2),
        grad_v.transpose(1, 2),
    )


@register_meta([aten.scatter_reduce.two, aten.scatter_reduce.two_out])
@out_wrapper()
def meta_scatter_reduce_two(self, dim, index, src, reduce, include_self=True):
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    return self.new_empty(self.shape)


@register_meta(aten.scatter_reduce_.two)
def meta_scatter_reduce__two(self, dim, index, src, reduce, include_self=True):
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    return self


@register_meta([aten.multinomial.default, aten.multinomial.out])
@out_wrapper()
def meta_multinomial(input, num_samples, replacement=False, *, generator=None):
    check(
        0 < input.dim() <= 2,
        lambda: f"The probabilty distributions dimensions must be 1 or 2, but got {input.dim()}",
    )
    if input.dim() == 1:
        return torch.empty(num_samples, dtype=torch.long, device=input.device)
    return torch.empty(
        input.size(0), num_samples, dtype=torch.long, device=input.device
    )


def multiply_integers(vs):
    r = 1
    for v in vs:
        r *= v
    return r


def upsample_common_check(input_size, output_size, num_spatial_dims):
    check(
        len(output_size) == num_spatial_dims,
        lambda: f"It is expected output_size equals to {num_spatial_dims}, but got size {len(output_size)}",
    )
    expected_input_dims = num_spatial_dims + 2  # N, C, ...
    check(
        len(input_size) == expected_input_dims,
        lambda: f"It is expected input_size equals to {expected_input_dims}, but got size {len(input_size)}",
    )

    check(
        all(s > 0 for s in input_size[2:]) and all(s > 0 for s in output_size),
        lambda: f"Input and output sizes should be greater than 0, but got "
        f"input size {input_size} and output size {output_size}",
    )

    nbatch, channels = input_size[:2]
    return (nbatch, channels, *output_size)


@register_meta(aten.upsample_nearest1d.default)
def upsample_nearest1d(input, output_size, scales=None):
    check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 3D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=1
    )
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


@register_meta(aten.upsample_nearest2d.default)
def upsample_nearest2d(input, output_size, scales_h=None, scales_w=None):
    check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    output = input.new_empty(full_output_size)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    _, n_channels, _, _ = input.shape
    if input.device.type == "cuda" and n_channels < 4:
        memory_format = torch.contiguous_format

    output = output.contiguous(memory_format=memory_format)

    return output


@register_meta(aten.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(
    grad_output: Tensor,
    output_size: Sequence[Union[int, torch.types.SymInt]],
    input_size: Sequence[Union[int, torch.types.SymInt]],
    scales_h: float = None,
    scales_w: float = None,
):
    full_output_size = upsample_common_check(
        input_size, output_size, num_spatial_dims=2
    )
    check(
        grad_output.ndim == 4,
        lambda: f"Expected grad_output to be a tensor of dimension 4 but got: dimension {grad_output.ndim}",
    )
    for i in range(4):
        check(
            grad_output.size(i) == full_output_size[i],
            lambda: (
                f"Expected grad_output to have the same shape as output;"
                f" output.size({i}) = {full_output_size[i]}"
                f" but got grad_output.size({i}) = {grad_output.size(i)}"
            ),
        )

    return grad_output.new_empty(input_size).to(
        memory_format=utils.suggest_memory_format(grad_output)
    )  # type: ignore[call-overload]


@register_meta(aten.upsample_nearest3d.default)
def upsample_nearest3d(input, output_size, scales_d=None, scales_h=None, scales_w=None):
    check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 5D data tensor expected but got a tensor with sizes {input.size()}",
    )
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=3
    )
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


@register_meta(
    [
        aten.sort.default,
        aten.sort.stable,
        aten.sort.values,
        aten.sort.values_stable,
    ]
)
def meta_sort(self, stable=None, dim=-1, descending=False, values=None, indices=None):
    v, i = torch.empty_like(self), torch.empty_like(self, dtype=torch.int64)
    if values is not None and indices is not None:
        assert isinstance(values, TensorLike)
        assert isinstance(indices, TensorLike)
        # Makes sure values and indices have the same strides. For cases where
        # these have different shapes, like (5, 10, 5) and (0) in msort.
        out_shape = v.shape
        out_stride = v.stride()
        values = _maybe_resize_out(values, out_shape)
        indices = _maybe_resize_out(indices, out_shape)
        values.as_strided_(out_shape, out_stride)
        indices.as_strided_(out_shape, out_stride)
        _safe_copy_out(copy_from=v, copy_to=values)  # type: ignore[arg-type]
        _safe_copy_out(copy_from=i, copy_to=indices)  # type: ignore[arg-type]
        return values, indices
    return v, i


def rnn_cell_checkSizes(
    input_gates, hidden_gates, input_bias, hidden_bias, factor, prev_hidden
):
    check(input_gates.ndim == 2, lambda: f"{input_gates.ndim} != 2")
    check(
        input_gates.shape == hidden_gates.shape,
        lambda: f"{input_gates.shape} != {hidden_gates.shape}",
    )
    gates_size = input_gates.size(1)
    if input_bias is not None:
        check(input_bias.ndim == 1, lambda: f"{input_bias.ndim} != 1")
        check(
            input_bias.numel() == gates_size,
            lambda: f"{input_bias.numel()} != {gates_size}",
        )
        check(
            input_bias.shape == hidden_bias.shape,
            lambda: f"{input_bias.shape} != {hidden_bias.shape}",
        )
    check(prev_hidden.ndim == 2, lambda: f"{prev_hidden.ndim} != 2")
    expected_prev_hidden_numel = input_gates.size(0) * gates_size // factor
    check(
        prev_hidden.numel() == expected_prev_hidden_numel,
        lambda: f"{prev_hidden.numel()} != {input_gates.size(0)} * {gates_size} // {factor} (aka {expected_prev_hidden_numel})",
    )
    check(
        all(
            x.device == input_gates.device
            for x in [hidden_gates, input_bias, hidden_bias, prev_hidden]
        ),
        lambda: "expected all inputs to be same device",
    )


@register_meta(aten._thnn_fused_lstm_cell.default)
def _thnn_fused_lstm_cell_meta(
    input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None
):
    rnn_cell_checkSizes(input_gates, hidden_gates, input_bias, hidden_bias, 4, cx)
    workspace = torch.empty_like(input_gates, memory_format=torch.contiguous_format)
    hy = torch.empty_like(cx, memory_format=torch.contiguous_format)
    cy = torch.empty_like(cx, memory_format=torch.contiguous_format)
    return (hy, cy, workspace)


@register_meta(aten._cudnn_rnn.default)
def _cudnn_rnn(
    input,
    weight,
    weight_stride0,
    weight_buf,
    hx,
    cx,
    mode,
    hidden_size,
    proj_size,
    num_layers,
    batch_first,
    dropout,
    train,
    bidirectional,
    batch_sizes,
    dropout_state,
):
    is_input_packed = len(batch_sizes) != 0
    if is_input_packed:
        seq_length = len(batch_sizes)
        mini_batch = batch_sizes[0]
        batch_sizes_sum = input.shape[0]
    else:
        seq_length = input.shape[1] if batch_first else input.shape[0]
        mini_batch = input.shape[0] if batch_first else input.shape[1]
        batch_sizes_sum = -1

    num_directions = 2 if bidirectional else 1
    out_size = proj_size if proj_size != 0 else hidden_size
    if is_input_packed:
        out_shape = [batch_sizes_sum, out_size * num_directions]
    else:
        out_shape = (
            [mini_batch, seq_length, out_size * num_directions]
            if batch_first
            else [seq_length, mini_batch, out_size * num_directions]
        )
    output = input.new_empty(out_shape)

    cell_shape = [num_layers * num_directions, mini_batch, hidden_size]
    if cx is None:
        cy = torch.empty(0, device=input.device)
    else:
        cy = cx.new_empty(cell_shape)

    hy = hx.new_empty([num_layers * num_directions, mini_batch, out_size])

    # TODO: Query cudnnGetRNNTrainingReserveSize (expose to python)
    reserve_shape = 0 if train else 0
    reserve = input.new_empty(reserve_shape, dtype=torch.uint8)

    return output, hy, cy, reserve, weight_buf


@register_meta(aten.mkldnn_rnn_layer.default)
def mkldnn_rnn_layer(
    input,
    w0,
    w1,
    w2,
    w3,
    hx_,
    cx_,
    reverse,
    batch_sizes,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    train,
):
    seq_length = input.shape[1] if batch_first else input.shape[0]
    mini_batch = input.shape[0] if batch_first else input.shape[1]
    output_chanels = hidden_size
    out_shape = (
        [mini_batch, seq_length, output_chanels]
        if batch_first
        else [seq_length, mini_batch, output_chanels]
    )
    output = input.new_empty(out_shape)
    if hx_ is None:
        hy = torch.empty(0, device=input.device)
    else:
        hy = hx_.new_empty(hx_.shape)
    if cx_ is None:
        cy = torch.empty(0, device=input.device)
    else:
        cy = cx_.new_empty(cx_.shape)
    workspace = torch.empty(0, device=input.device, dtype=torch.uint8)
    return output, hy, cy, workspace


def zero_numel_check_dims(self, dim, fn_name):
    if self.ndim == 0:
        check(
            dim == 0 or dim == -1,
            lambda: f"{fn_name}: Expected reduction dim -1 or 0 for scalar but got {dim}",
            IndexError,
        )
    else:
        check(
            self.size(dim) != 0,
            lambda: f"{fn_name}: Expected reduction dim {dim} to have non-zero size.",
            IndexError,
        )


# From aten/src/ATen/native/ReduceOps.cpp
def check_argmax_argmin(name, self, dim):
    if dim is not None:
        dim = maybe_wrap_dim(dim, self.dim())
        zero_numel_check_dims(self, dim, name)
    else:
        check(
            self.numel() != 0,
            lambda: f"{name}: Expected reduction dim to be specified for input.numel() == 0.",
        )


@register_meta([aten.argmax.default, aten.argmin.default])
def argmax_argmin_meta(self, dim=None, keepdim=False):
    check_argmax_argmin("argmax", self, dim)
    dims = utils.reduction_dims(self.shape, (dim,) if dim is not None else None)
    shape = _compute_reduction_shape(self, dims, keepdim)
    return self.new_empty(shape, dtype=torch.int64)


@register_meta(aten.scalar_tensor.default)
def scalar_tensor(s, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.empty(
        (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.topk.default)
def topk_meta(self, k, dim=-1, largest=True, sorted=True):
    # From aten/src/ATen/native/Sorting.cpp
    dim = maybe_wrap_dim(dim, self.dim(), wrap_scalar=True)
    check(
        k >= 0 and k <= (self.size(dim) if self.dim() > 0 else 1),
        lambda: "selected index k out of range",
    )
    sliceSize = 1 if self.dim() == 0 else self.size(dim)
    check(k >= 0 and k <= sliceSize, lambda: "k not in range for dimension")

    topKSize = list(self.shape)
    if len(topKSize) > 0:
        topKSize[dim] = k
    return self.new_empty(topKSize), self.new_empty(topKSize, dtype=torch.int64)


legacy_contiguous_memory_format = torch.contiguous_format


# From aten/src/ATen/native/cuda/RNN.cu
def checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace):
    defined_grad = grad_hy if grad_hy is not None else grad_cy
    check(defined_grad.dim() == 2, lambda: "")
    exp_size = defined_grad.size()
    if grad_hy is not None:
        check(grad_hy.size() == exp_size, lambda: "")
    if grad_cy is not None:
        check(grad_cy.size() == exp_size, lambda: "")
    check(cx.size() == exp_size, lambda: "")
    check(cy.size() == exp_size, lambda: "")
    check(workspace.dim() == 2, lambda: "")
    check(workspace.numel() == exp_size[0] * exp_size[1] * 4, lambda: "")


# From aten/src/ATen/native/cuda/RNN.cu
@register_meta(aten._thnn_fused_lstm_cell_backward_impl.default)
def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    if grad_hy is None and grad_cy is None:
        return None, None, None
    checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace)
    grad_gates = torch.empty_like(
        workspace, memory_format=legacy_contiguous_memory_format
    )
    grad_cx = torch.empty_like(cx, memory_format=legacy_contiguous_memory_format)
    grad_bias = grad_gates.sum(0, keepdim=False) if has_bias else None
    return grad_gates, grad_cx, grad_bias


@register_meta(aten.pixel_shuffle.default)
def meta_pixel_shuffle(self, upscale_factor):
    assert (
        len(self.shape) > 2 and self.shape[-3] % (upscale_factor * upscale_factor) == 0
    ), f"Invalid input shape for pixel_shuffle: {self.shape} with upscale_factor = {upscale_factor}"

    def is_channels_last(ten):
        return torch._prims_common.suggest_memory_format(ten) == torch.channels_last

    def pick_memory_format():
        if is_channels_last(self):
            if device_hint(self) == "cuda":
                return torch.contiguous_format
            else:
                return torch.channels_last
        elif self.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif self.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    C = self.shape[-3] // (upscale_factor * upscale_factor)
    Hr = self.shape[-2] * upscale_factor
    Wr = self.shape[-1] * upscale_factor
    out_shape = (*self.shape[:-3], C, Hr, Wr)

    out = self.new_empty(out_shape)
    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


@register_meta(aten.mkldnn_rnn_layer_backward.default)
def mkldnn_rnn_layer_backward(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_tmp,
    output,
    hy_,
    cy_,
    grad_output_r_opt,
    grad_hy_r_opt,
    grad_cy_r_opt,
    reverse,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    train,
    bidirectional,
    batch_sizes,
    batch_first,
    workspace,
):
    diff_x = input.new_empty(input.shape)
    diff_hx = hx_.new_empty(hx_.shape)
    diff_cx = cx_tmp.new_empty(cx_tmp.shape)
    diff_w1 = weight0.new_empty(weight0.shape)
    diff_w2 = weight1.new_empty(weight1.shape)
    diff_b = weight2.new_empty(weight2.shape)
    return diff_x, diff_w1, diff_w2, diff_b, diff_b, diff_hx, diff_cx


@register_meta([aten.bucketize.Tensor, aten.bucketize.Tensor_out])
@out_wrapper()
def meta_bucketize(self, boundaries, *, out_int32=False, right=False):
    return torch.empty_like(
        self, dtype=torch.int32 if out_int32 else torch.int64
    ).contiguous()


@register_meta(aten._upsample_bilinear2d_aa.default)
def meta_upsample_bilinear2d_aa(
    input, output_size, align_corners, scales_h=None, scales_w=None
):
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    check(
        input.numel() != 0 or all(size > 0 for size in input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


# From aten/src/ATen/native/cuda/AmpKernels.cu
@register_meta(aten._amp_foreach_non_finite_check_and_unscale_.default)
def _amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale):
    check(found_inf.numel() == 1, lambda: "found_inf must be a 1-element tensor.")
    check(inv_scale.numel() == 1, lambda: "inv_scale must be a 1-element tensor.")
    check(
        found_inf.dtype.is_floating_point,
        lambda: "found_inf must be a float tensor.",
    )
    check(
        inv_scale.dtype.is_floating_point,
        lambda: "inv_scale must be a float tensor.",
    )


# From aten/src/ATen/native/UnaryOps.cpp
@register_meta([aten.nan_to_num.default, aten.nan_to_num.out])
@out_wrapper()
def nan_to_num(self, nan=None, posinf=None, neginf=None):
    result_size = list(self.size())
    return self.new_empty(result_size)


@register_meta(torch.ops.aten.transpose_)
def transpose_(self, dim0, dim1):
    assert self.layout not in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }, f"torch.transpose_: in-place transposition is not supported for {self.layout} layout"

    ndims = self.ndim

    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)

    if dim0 == dim1:
        return self

    size = list(self.size())
    stride = list(self.stride())

    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    size[dim0], size[dim1] = size[dim1], size[dim0]

    self.as_strided_(size, stride)
    return self


@register_meta(torch.ops.aten.t_)
def t_(self):
    ndims = self.ndim

    if self.is_sparse:
        sparse_dim = self.sparse_dim()
        dense_dim = self.dense_dim()
        assert (
            sparse_dim <= 2 and dense_dim == 0
        ), f"t_ expects a tensor with <= 2 sparse and 0 dense dimensions, but got {sparse_dim} sparse and {dense_dim} dense dimensions"  # noqa: B950
    else:
        assert (
            self.dim() <= 2
        ), f"t_ expects a tensor with <= 2 dimensions, but self is {ndims}D"

    return transpose_(self, 0, 0 if ndims < 2 else 1)


@register_meta([aten.searchsorted.Tensor, aten.searchsorted.Tensor_out])
@out_wrapper()
def meta_searchsorted(
    sorted_sequence, self, *, out_int32=False, right=False, side=None, sorter=None
):
    dtype = torch.int32 if out_int32 else torch.int64
    return torch.empty_like(self, dtype=dtype).contiguous()


# We must also trigger meta registrations from PrimTorch ref
# decompositions
import torch._refs
import torch._refs.nn.functional
import torch._refs.special


def activate_meta():
    activate_meta_table = {}

    # For a given op, we pick the most specific decomp function from
    # global_decomp_table in the precedence order of meta > post_autograd > pre_autograd
    for type in ["meta", "post_autograd", "pre_autograd"]:
        registry = global_decomposition_table[type]

        for opo in registry:
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]

    for op_overload, fn in activate_meta_table.items():
        assert isinstance(op_overload, OpOverload)

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        if torch._C._dispatch_has_kernel_for_dispatch_key(
            op_overload.name(), "CompositeImplicitAutograd"
        ):
            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            if op_overload in global_decomposition_table["meta"]:
                raise RuntimeError(
                    f"{op_overload} is a CompositeImplicitAutograd op, we shouldn't "
                    "register meta function for it. Instead, we should let the decomposition run and write "
                    "meta kernels for the base operators."
                )
            pass
        elif op_overload.is_view:
            # Attempting to register a python meta kernel for a view operator.
            # We shouldn't do this, because the output will report as not having aliased storages.
            # All view ops have meta kernels in C++ today, so we should use those instead.
            pass
        elif op_overload.name() in {
            "aten::empty_strided",  # causing infinite recursion, test_meta.py
            "aten::clone",  # causing infinite recursion
            "aten::_to_copy",  # causing infinite recursion, test_serialization.py -k test_tensor_subclass_getstate_overwrite  # noqa: B950
            "aten::copy_",  # Exception not raised, test_torch.py -k test_storage_meta_errors_cpu_int64  # noqa: B950
            "aten::constant_pad_nd",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_amp_istft_cuda_float32  # noqa: B950
            "aten::rot90",  # requires_grad mismatch! test_ops.py -k test_fake_crossref_backward_amp_rot90_cuda_float32  # noqa: B950
            "aten::as_strided_scatter",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_no_amp_as_strided_scatter_cuda_float32  # noqa: B950
        }:
            pass
        else:
            if "mkldnn::" in op_overload.name():
                _meta_lib_dont_use_me_use_register_meta_for_mkldnn.impl(op_overload, fn)
            elif "mkl::" in op_overload.name():
                _meta_lib_dont_use_me_use_register_meta_for_mkl.impl(op_overload, fn)
            else:
                _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)


activate_meta()
