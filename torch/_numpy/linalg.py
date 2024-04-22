# mypy: ignore-errors

from __future__ import annotations

import functools
import math
from typing import Sequence

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, KeepDims, normalizer


class LinAlgError(Exception):
    pass


def _atleast_float_1(a):
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        a = a.to(_dtypes_impl.default_dtypes().float_dtype)
    return a


def _atleast_float_2(a, b):
    dtyp = _dtypes_impl.result_type_impl(a, b)
    if not (dtyp.is_floating_point or dtyp.is_complex):
        dtyp = _dtypes_impl.default_dtypes().float_dtype

    a = _util.cast_if_needed(a, dtyp)
    b = _util.cast_if_needed(b, dtyp)
    return a, b


def linalg_errors(func):
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        try:
            return func(*args, **kwds)
        except torch._C._LinAlgError as e:
            raise LinAlgError(*e.args)  # noqa: TRY200

    return wrapped


# ### Matrix and vector products ###


@normalizer
@linalg_errors
def matrix_power(a: ArrayLike, n):
    a = _atleast_float_1(a)
    return torch.linalg.matrix_power(a, n)


@normalizer
@linalg_errors
def multi_dot(inputs: Sequence[ArrayLike], *, out=None):
    return torch.linalg.multi_dot(inputs)


# ### Solving equations and inverting matrices ###


@normalizer
@linalg_errors
def solve(a: ArrayLike, b: ArrayLike):
    a, b = _atleast_float_2(a, b)
    return torch.linalg.solve(a, b)


@normalizer
@linalg_errors
def lstsq(a: ArrayLike, b: ArrayLike, rcond=None):
    a, b = _atleast_float_2(a, b)
    # NumPy is using gelsd: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/umath_linalg.cpp#L3991
    # on CUDA, only `gels` is available though, so use it instead
    driver = "gels" if a.is_cuda or b.is_cuda else "gelsd"
    return torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)


@normalizer
@linalg_errors
def inv(a: ArrayLike):
    a = _atleast_float_1(a)
    result = torch.linalg.inv(a)
    return result


@normalizer
@linalg_errors
def pinv(a: ArrayLike, rcond=1e-15, hermitian=False):
    a = _atleast_float_1(a)
    return torch.linalg.pinv(a, rtol=rcond, hermitian=hermitian)


@normalizer
@linalg_errors
def tensorsolve(a: ArrayLike, b: ArrayLike, axes=None):
    a, b = _atleast_float_2(a, b)
    return torch.linalg.tensorsolve(a, b, dims=axes)


@normalizer
@linalg_errors
def tensorinv(a: ArrayLike, ind=2):
    a = _atleast_float_1(a)
    return torch.linalg.tensorinv(a, ind=ind)


# ### Norms and other numbers ###


@normalizer
@linalg_errors
def det(a: ArrayLike):
    a = _atleast_float_1(a)
    return torch.linalg.det(a)


@normalizer
@linalg_errors
def slogdet(a: ArrayLike):
    a = _atleast_float_1(a)
    return torch.linalg.slogdet(a)


@normalizer
@linalg_errors
def cond(x: ArrayLike, p=None):
    x = _atleast_float_1(x)

    # check if empty
    # cf: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    if x.numel() == 0 and math.prod(x.shape[-2:]) == 0:
        raise LinAlgError("cond is not defined on empty arrays")

    result = torch.linalg.cond(x, p=p)

    # Convert nans to infs (numpy does it in a data-dependent way, depending on
    # whether the input array has nans or not)
    # XXX: NumPy does this: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    return torch.where(torch.isnan(result), float("inf"), result)


@normalizer
@linalg_errors
def matrix_rank(a: ArrayLike, tol=None, hermitian=False):
    a = _atleast_float_1(a)

    if a.ndim < 2:
        return int((a != 0).any())

    if tol is None:
        # follow https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1885
        atol = 0
        rtol = max(a.shape[-2:]) * torch.finfo(a.dtype).eps
    else:
        atol, rtol = tol, 0
    return torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


@normalizer
@linalg_errors
def norm(x: ArrayLike, ord=None, axis=None, keepdims: KeepDims = False):
    x = _atleast_float_1(x)
    return torch.linalg.norm(x, ord=ord, dim=axis)


# ### Decompositions ###


@normalizer
@linalg_errors
def cholesky(a: ArrayLike):
    a = _atleast_float_1(a)
    return torch.linalg.cholesky(a)


@normalizer
@linalg_errors
def qr(a: ArrayLike, mode="reduced"):
    a = _atleast_float_1(a)
    result = torch.linalg.qr(a, mode=mode)
    if mode == "r":
        # match NumPy
        result = result.R
    return result


@normalizer
@linalg_errors
def svd(a: ArrayLike, full_matrices=True, compute_uv=True, hermitian=False):
    a = _atleast_float_1(a)
    if not compute_uv:
        return torch.linalg.svdvals(a)

    # NB: ignore the hermitian= argument (no pytorch equivalent)
    result = torch.linalg.svd(a, full_matrices=full_matrices)
    return result


# ### Eigenvalues and eigenvectors ###


@normalizer
@linalg_errors
def eig(a: ArrayLike):
    a = _atleast_float_1(a)
    w, vt = torch.linalg.eig(a)

    if not a.is_complex() and w.is_complex() and (w.imag == 0).all():
        w = w.real
        vt = vt.real
    return w, vt


@normalizer
@linalg_errors
def eigh(a: ArrayLike, UPLO="L"):
    a = _atleast_float_1(a)
    return torch.linalg.eigh(a, UPLO=UPLO)


@normalizer
@linalg_errors
def eigvals(a: ArrayLike):
    a = _atleast_float_1(a)
    result = torch.linalg.eigvals(a)
    if not a.is_complex() and result.is_complex() and (result.imag == 0).all():
        result = result.real
    return result


@normalizer
@linalg_errors
def eigvalsh(a: ArrayLike, UPLO="L"):
    a = _atleast_float_1(a)
    return torch.linalg.eigvalsh(a, UPLO=UPLO)
