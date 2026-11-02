"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""

# Contents of this module ends up in the main namespace via _funcs.py
# where type annotations are used in conjunction with the @normalizer decorator.
from __future__ import annotations

import builtins
import itertools
import operator
from typing import TYPE_CHECKING

import torch

from . import _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._dtypes import DType
    from ._ndarray import ndarray
    from ._normalizations import (
        ArrayLike,
        ArrayLikeOrScalar,
        CastingModes,
        DTypeLike,
        NDArray,
        NotImplementedType,
        OutArray,
    )


def copy(
    a: ArrayLike, order: NotImplementedType = "K", subok: NotImplementedType = False
) -> torch.Tensor:
    return a.clone()


def copyto(
    dst: NDArray,
    src: ArrayLike,
    casting: CastingModes = "same_kind",
    where: NotImplementedType = None,
) -> None:
    (src,) = _util.typecast_tensors((src,), dst.dtype, casting=casting)
    dst.copy_(src)


def atleast_1d(*arys: ArrayLike) -> torch.Tensor | list[torch.Tensor]:
    res = torch.atleast_1d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_2d(*arys: ArrayLike) -> torch.Tensor | list[torch.Tensor]:
    res = torch.atleast_2d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_3d(*arys: ArrayLike) -> torch.Tensor | list[torch.Tensor]:
    res = torch.atleast_3d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def _concat_check(
    tup: Sequence[torch.Tensor], dtype: torch.dtype | None, out: object
) -> None:
    if tup == ():
        raise ValueError("need at least one array to concatenate")

    """Check inputs in concatenate et al."""
    if out is not None and dtype is not None:
        # mimic numpy
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an "
            "argument, but both were provided."
        )


def _concat_cast_helper(
    tensors: Sequence[torch.Tensor],
    out: ndarray | None = None,
    dtype: torch.dtype | None = None,
    casting: str = "same_kind",
) -> tuple[torch.Tensor, ...]:
    """Figure out dtypes, cast if necessary."""

    if out is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    elif dtype is not None:
        out_dtype = dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl(*tensors)

    # cast input arrays if necessary; do not broadcast them against `out`
    return _util.typecast_tensors(tensors, out_dtype, casting)


def _concatenate(
    tensors: Sequence[torch.Tensor],
    axis: int | None = 0,
    out: ndarray | None = None,
    dtype: torch.dtype | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    # pure torch implementation, used below and in cov/corrcoef below
    tensors, axis = _util.axis_none_flatten(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)
    return torch.cat(tensors, axis)


def concatenate(
    ar_tuple: Sequence[ArrayLike],
    axis: int | None = 0,
    out: OutArray | None = None,
    dtype: DTypeLike | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    _concat_check(ar_tuple, dtype, out=out)
    result = _concatenate(ar_tuple, axis=axis, out=out, dtype=dtype, casting=casting)
    return result


def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.vstack(tensors)


row_stack = vstack


def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.hstack(tensors)


def dstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.dstack(tensors)


def column_stack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.column_stack(tensors)


def stack(
    arrays: Sequence[ArrayLike],
    axis: int = 0,
    out: OutArray | None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes = "same_kind",
) -> torch.Tensor:
    _concat_check(arrays, dtype, out=out)

    tensors = _concat_cast_helper(arrays, dtype=dtype, casting=casting)
    result_ndim = tensors[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)
    # torch.stack accepts `axis` as an alias for `dim` at runtime.
    return torch.stack(tensors, axis=axis)  # pyrefly: ignore[unexpected-keyword]


def append(arr: ArrayLike, values: ArrayLike, axis: int | None = None) -> torch.Tensor:
    if axis is None:
        if arr.ndim != 1:
            arr = arr.flatten()
        values = values.flatten()
        axis = arr.ndim - 1
    return _concatenate((arr, values), axis=axis)


# ### split ###


def _split_helper(
    tensor: torch.Tensor,
    indices_or_sections: int | Sequence[int],
    axis: int,
    strict: bool = False,
) -> tuple[torch.Tensor, ...]:
    if isinstance(indices_or_sections, int):
        return _split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        # NB: drop split=..., it only applies to split_helper_int
        return _split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        raise TypeError(f"split_helper: {type(indices_or_sections)}")


def _split_helper_int(
    tensor: torch.Tensor, indices_or_sections: int, axis: int, strict: bool = False
) -> tuple[torch.Tensor, ...]:
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("split: indices_or_sections")

    axis = _util.normalize_axis_index(axis, tensor.ndim)

    # numpy: l%n chunks of size (l//n + 1), the rest are sized l//n
    l, n = tensor.shape[axis], indices_or_sections

    if n <= 0:
        raise ValueError

    if l % n == 0:
        num, sz = n, l // n
        lst = [sz] * num
    else:
        if strict:
            raise ValueError("array split does not result in an equal division")

        num, sz = l % n, l // n + 1
        lst = [sz] * num

    lst += [sz - 1] * (n - num)

    return torch.split(tensor, lst, axis)


def _split_helper_list(
    tensor: torch.Tensor, indices_or_sections: list[int], axis: int
) -> tuple[torch.Tensor, ...]:
    if not isinstance(indices_or_sections, list):
        raise NotImplementedError("split: indices_or_sections: list")
    # numpy expects indices, while torch expects lengths of sections
    # also, numpy appends zero-size arrays for indices above the shape[axis]
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)

    lst.append(tensor.shape[axis])
    lst = [
        lst[0],
    ] + [a - b for a, b in zip(lst[1:], lst[:-1])]
    lst += [0] * num_extra

    return torch.split(tensor, lst, axis)


def array_split(
    ary: ArrayLike, indices_or_sections: int | Sequence[int], axis: int = 0
) -> tuple[torch.Tensor, ...]:
    return _split_helper(ary, indices_or_sections, axis)


def split(
    ary: ArrayLike, indices_or_sections: int | Sequence[int], axis: int = 0
) -> tuple[torch.Tensor, ...]:
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def hsplit(
    ary: ArrayLike, indices_or_sections: int | Sequence[int]
) -> tuple[torch.Tensor, ...]:
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    axis = 1 if ary.ndim > 1 else 0
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def vsplit(
    ary: ArrayLike, indices_or_sections: int | Sequence[int]
) -> tuple[torch.Tensor, ...]:
    if ary.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return _split_helper(ary, indices_or_sections, 0, strict=True)


def dsplit(
    ary: ArrayLike, indices_or_sections: int | Sequence[int]
) -> tuple[torch.Tensor, ...]:
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return _split_helper(ary, indices_or_sections, 2, strict=True)


def kron(a: ArrayLike, b: ArrayLike) -> torch.Tensor:
    return torch.kron(a, b)


def vander(
    x: ArrayLike, N: int | None = None, increasing: bool = False
) -> torch.Tensor:
    return torch.vander(x, N, increasing)


# ### linspace, geomspace, logspace and arange ###


def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> torch.Tensor:
    if axis != 0 or retstep or not endpoint:
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    # XXX: raises TypeError if start or stop are not scalars
    return torch.linspace(start, stop, num, dtype=dtype)


def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> torch.Tensor:
    if axis != 0 or not endpoint:
        raise NotImplementedError
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)
    # torch.logspace accepts tensor start/end/base at runtime.
    return torch.logspace(  # pyrefly: ignore[no-matching-overload]
        torch.log(start) / logbase,
        torch.log(stop) / logbase,
        num,
        base=base,
    )


def logspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> torch.Tensor:
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return torch.logspace(start, stop, num, base=base, dtype=dtype)


def arange(
    start: ArrayLikeOrScalar | None = None,
    stop: ArrayLikeOrScalar | None = None,
    step: ArrayLikeOrScalar | None = 1,
    dtype: DTypeLike | None = None,
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if step == 0:
        raise ZeroDivisionError
    if stop is None:
        if start is None:
            raise TypeError
        # One positional value is stop: arange(4) == arange(0, 4).
        # arange(start=4) is rejected in _funcs.py before this remapping.
        start, stop = 0, start
    if start is None:
        start = 0

    # the dtype of the result
    # NB: a None step is not handled here (numpy would treat it as 1); passing
    # step=None reaches these helpers and raises, matching prior behavior.
    if dtype is None:
        dtype = (
            _dtypes_impl.default_dtypes().float_dtype
            # pyrefly: ignore[bad-argument-type]
            if any(_dtypes_impl.is_float_or_fp_tensor(x) for x in (start, stop, step))
            else _dtypes_impl.default_dtypes().int_dtype
        )
    work_dtype = torch.float64 if dtype.is_complex else dtype

    # RuntimeError: "lt_cpu" not implemented for 'ComplexFloat'. Fall back to eager.
    # pyrefly: ignore[bad-argument-type]
    if any(_dtypes_impl.is_complex_or_complex_tensor(x) for x in (start, stop, step)):
        raise NotImplementedError

    # The complex check above guarantees start/stop/step are real and orderable.
    # pyrefly: ignore[unsupported-operation]
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        # empty range
        return torch.empty(0, dtype=dtype)

    # torch.arange accepts scalar tensors as bounds at runtime.
    # pyrefly: ignore[no-matching-overload]
    result = torch.arange(start, stop, step, dtype=work_dtype)
    result = _util.cast_if_needed(result, dtype)
    return result


# ### zeros/ones/empty/full ###


def empty(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.empty(shape, dtype=dtype)


# NB: *_like functions deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.


def empty_like(
    prototype: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape: int | Sequence[int] | None = None,
) -> torch.Tensor:
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(
    shape: int | Sequence[int],
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = fill_value.dtype
    if not isinstance(shape, (tuple, list)):
        # defensive wrap for non-int scalar shapes (e.g. numpy scalars) that
        # fall outside the annotated int | Sequence[int] domain
        shape = (shape,)  # pyrefly: ignore[bad-assignment]
    # torch.full accepts a tensor fill_value at runtime.
    # pyrefly: ignore[no-matching-overload]
    return torch.full(shape, fill_value, dtype=dtype)


def full_like(
    a: ArrayLike,
    fill_value: int | float | complex | bool,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape: int | Sequence[int] | None = None,
) -> torch.Tensor:
    # XXX: fill_value broadcasts
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.ones(shape, dtype=dtype)


def ones_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape: int | Sequence[int] | None = None,
) -> torch.Tensor:
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def zeros(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.zeros(shape, dtype=dtype)


def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape: int | Sequence[int] | None = None,
) -> torch.Tensor:
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# ### cov & corrcoef ###


def _xy_helper_corrcoef(
    x_tensor: torch.Tensor, y_tensor: torch.Tensor | None = None, rowvar: bool = True
) -> torch.Tensor:
    """Prepare inputs for cov and corrcoef."""

    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y_tensor is not None:
        # make sure x and y are at least 2D
        ndim_extra = 2 - x_tensor.ndim
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT
        x_tensor = x_tensor.clone()

        ndim_extra = 2 - y_tensor.ndim
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT
        y_tensor = y_tensor.clone()

        x_tensor = _concatenate((x_tensor, y_tensor), axis=0)

    return x_tensor


def corrcoef(
    x: ArrayLike,
    y: ArrayLike | None = None,
    rowvar: bool = True,
    bias: int | None = None,
    ddof: int | None = None,
    *,
    dtype: DTypeLike | None = None,
) -> torch.Tensor:
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    xy_tensor = _xy_helper_corrcoef(x, y, rowvar)

    is_half = (xy_tensor.dtype == torch.float16) and xy_tensor.is_cpu
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    result = torch.corrcoef(xy_tensor)

    if is_half:
        result = result.to(torch.float16)

    return result


def cov(
    m: ArrayLike,
    y: ArrayLike | None = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: ArrayLike | None = None,
    aweights: ArrayLike | None = None,
    *,
    dtype: DTypeLike | None = None,
) -> torch.Tensor:
    m = _xy_helper_corrcoef(m, y, rowvar)

    if ddof is None:
        ddof = 1 if bias == 0 else 0

    is_half = (m.dtype == torch.float16) and m.is_cpu
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    m = _util.cast_if_needed(m, dtype)
    result = torch.cov(m, correction=ddof, aweights=aweights, fweights=fweights)

    if is_half:
        result = result.to(torch.float16)

    return result


def _conv_corr_impl(a: torch.Tensor, v: torch.Tensor, mode: str) -> torch.Tensor:
    dt = _dtypes_impl.result_type_impl(a, v)
    a = _util.cast_if_needed(a, dt)
    v = _util.cast_if_needed(v, dt)

    padding = v.shape[0] - 1 if mode == "full" else mode

    if padding == "same" and v.shape[0] % 2 == 0:
        # UserWarning: Using padding='same' with even kernel lengths and odd
        # dilation may require a zero-padded copy of the input be created
        # (Triggered internally at pytorch/aten/src/ATen/native/Convolution.cpp:1010.)
        raise NotImplementedError("mode='same' and even-length weights")

    # NumPy only accepts 1D arrays; PyTorch requires 2D inputs and 3D weights
    aa = a[None, :]
    vv = v[None, None, :]

    result = torch.nn.functional.conv1d(aa, vv, padding=padding)

    # torch returns a 2D result, numpy returns a 1D array
    return result[0, :]


def convolve(a: ArrayLike, v: ArrayLike, mode: str = "full") -> torch.Tensor:
    # NumPy: if v is longer than a, the arrays are swapped before computation
    if a.shape[0] < v.shape[0]:
        a, v = v, a

    # flip the weights since numpy does and torch does not
    v = torch.flip(v, (0,))

    return _conv_corr_impl(a, v, mode)


def correlate(a: ArrayLike, v: ArrayLike, mode: str = "valid") -> torch.Tensor:
    v = torch.conj_physical(v)
    return _conv_corr_impl(a, v, mode)


# ### logic & element selection ###


def bincount(
    x: ArrayLike, /, weights: ArrayLike | None = None, minlength: int = 0
) -> torch.Tensor:
    if x.numel() == 0:
        # edge case allowed by numpy; torch accepts the python `int` type as dtype
        x = x.new_empty(0, dtype=int)  # pyrefly: ignore[no-matching-overload]

    int_dtype = _dtypes_impl.default_dtypes().int_dtype
    (x,) = _util.typecast_tensors((x,), int_dtype, casting="safe")

    return torch.bincount(x, weights, minlength)


def where(
    condition: ArrayLike,
    x: ArrayLikeOrScalar | None = None,
    y: ArrayLikeOrScalar | None = None,
    /,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if (x is None) != (y is None):
        raise ValueError("either both or neither of x and y should be given")

    if condition.dtype != torch.bool:
        condition = condition.to(torch.bool)

    # the guard above guarantees x and y are either both None or both non-None
    if x is None or y is None:
        result = torch.where(condition)
    else:
        result = torch.where(condition, x, y)
    return result


# ###### module-level queries of object properties


def ndim(a: ArrayLike) -> int:
    return a.ndim


def shape(a: ArrayLike) -> tuple[int, ...]:
    return tuple(a.shape)


def size(a: ArrayLike, axis: int | None = None) -> int:
    if axis is None:
        return a.numel()
    else:
        return a.shape[axis]


# ###### shape manipulations and indexing


def expand_dims(a: ArrayLike, axis: int | tuple[int, ...]) -> torch.Tensor:
    shape = _util.expand_shape(a.shape, axis)
    return a.view(shape)  # never copies


def flip(m: ArrayLike, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    # XXX: semantic difference: np.flip returns a view, torch.flip copies
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)


def flipud(m: ArrayLike) -> torch.Tensor:
    return torch.flipud(m)


def fliplr(m: ArrayLike) -> torch.Tensor:
    return torch.fliplr(m)


def rot90(m: ArrayLike, k: int = 1, axes: tuple[int, ...] = (0, 1)) -> torch.Tensor:
    axes = _util.normalize_axis_tuple(axes, m.ndim)
    return torch.rot90(m, k, axes)


# ### broadcasting and indices ###


def broadcast_to(
    array: ArrayLike, shape: int | Sequence[int], subok: NotImplementedType = False
) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return torch.broadcast_to(array, size=shape)


# This is a function from tuples to tuples, so we just reuse it.  However,
# dynamo expects its __module__ to be torch._numpy
def broadcast_shapes(*args: int | Sequence[int]) -> torch.Size:
    return torch.broadcast_shapes(*args)


def broadcast_arrays(
    *args: ArrayLike, subok: NotImplementedType = False
) -> list[torch.Tensor]:
    return torch.broadcast_tensors(*args)


def meshgrid(
    *xi: ArrayLike,
    copy: bool = True,
    sparse: bool = False,
    indexing: str = "xy",
) -> list[torch.Tensor]:
    ndim = len(xi)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = torch.broadcast_tensors(*output)

    if copy:
        output = [x.clone() for x in output]

    return list(output)  # match numpy, return a list


def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = torch.int64,
    sparse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1691-L1791
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    idxs = [
        torch.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1 :])
        for i, dim in enumerate(dimensions)
    ]
    if sparse:
        return tuple(idxs)
    res = torch.empty((N,) + dimensions, dtype=dtype)
    for i, idx in enumerate(idxs):
        res[i] = idx
    return res


# ### tri*-something ###


def tril(m: ArrayLike, k: int = 0) -> torch.Tensor:
    return torch.tril(m, k)


def triu(m: ArrayLike, k: int = 0) -> torch.Tensor:
    return torch.triu(m, k)


def tril_indices(n: int, k: int = 0, m: int | None = None) -> torch.Tensor:
    if m is None:
        m = n
    return torch.tril_indices(n, m, offset=k)


def triu_indices(n: int, k: int = 0, m: int | None = None) -> torch.Tensor:
    if m is None:
        m = n
    return torch.triu_indices(n, m, offset=k)


def tril_indices_from(arr: ArrayLike, k: int = 0) -> torch.Tensor:
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # Return a tensor rather than a tuple to avoid a graphbreak
    return torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)


def triu_indices_from(arr: ArrayLike, k: int = 0) -> torch.Tensor:
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # Return a tensor rather than a tuple to avoid a graphbreak
    return torch.triu_indices(arr.shape[0], arr.shape[1], offset=k)


def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = None,
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    return torch.tril(tensor, diagonal=k)


# ### equality, equivalence, allclose ###


def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = False,
) -> torch.Tensor:
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _tensor_equal(a1: torch.Tensor, a2: torch.Tensor, equal_nan: bool = False) -> bool:
    # Implementation of array_equal/array_equiv.
    if a1.shape != a2.shape:
        return False
    cond = a1 == a2
    if equal_nan:
        cond = cond | (torch.isnan(a1) & torch.isnan(a2))
    return bool(cond.all().item())


def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = False) -> bool:
    return _tensor_equal(a1, a2, equal_nan=equal_nan)


def array_equiv(a1: ArrayLike, a2: ArrayLike) -> bool:
    # *almost* the same as array_equal: _equiv tries to broadcast, _equal does not
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1, a2)
    except RuntimeError:
        # failed to broadcast => not equivalent
        return False
    return _tensor_equal(a1_t, a2_t)


def nan_to_num(
    x: ArrayLike,
    copy: NotImplementedType = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> torch.Tensor:
    # work around RuntimeError: "nan_to_num" not implemented for 'ComplexDouble'
    if x.is_complex():
        re = torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf)
        im = torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf)
        return re + 1j * im
    else:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


# ### put/take_along_axis ###


def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out: OutArray | None = None,
    mode: NotImplementedType = "raise",
) -> torch.Tensor:
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    idx = (slice(None),) * axis + (indices, ...)
    result = a[idx]
    return result


def take_along_axis(
    arr: ArrayLike, indices: ArrayLike, axis: int | None
) -> torch.Tensor:
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    return torch.take_along_dim(arr, indices, axis)


def put(
    a: NDArray,
    indices: ArrayLike,
    values: ArrayLike,
    mode: NotImplementedType = "raise",
) -> None:
    v = values.type(a.dtype)
    # If indices is larger than v, expand v to at least the size of indices. Any
    # unnecessary trailing elements are then trimmed.
    if indices.numel() > v.numel():
        ratio = (indices.numel() + v.numel() - 1) // v.numel()
        v = v.unsqueeze(0).expand((ratio,) + v.shape)
    # Trim unnecessary elements, regardless if v was expanded or not. Note
    # np.put() trims v to match indices by default too.
    if indices.numel() < v.numel():
        v = v.flatten()
        v = v[: indices.numel()]
    a.put_(indices, v)
    return None


def put_along_axis(
    arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis: int | None
) -> None:
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)

    indices, values = torch.broadcast_tensors(indices, values)
    values = _util.cast_if_needed(values, arr.dtype)
    result = torch.scatter(arr, axis, indices, values)
    arr.copy_(result.reshape(arr.shape))
    return None


def choose(
    a: ArrayLike,
    choices: Sequence[ArrayLike],
    out: OutArray | None = None,
    mode: NotImplementedType = "raise",
) -> torch.Tensor:
    # First, broadcast elements of `choices`
    choices_t = torch.stack(torch.broadcast_tensors(*choices))

    # Use an analog of `gather(choices, 0, a)` which broadcasts `choices` vs `a`:
    # (taken from https://github.com/pytorch/pytorch/issues/9407#issuecomment-1427907939)
    idx_list: list[torch.Tensor] = [
        torch.arange(dim).view((1,) * i + (dim,) + (1,) * (choices_t.ndim - i - 1))
        for i, dim in enumerate(choices_t.shape)
    ]

    idx_list[0] = a
    return choices_t[tuple(idx_list)].squeeze(0)


# ### unique et al. ###


def unique(
    ar: ArrayLike,
    return_index: NotImplementedType = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    *,
    equal_nan: NotImplementedType = True,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    (ar,), axis = _util.axis_none_flatten(ar, axis=axis)
    axis = _util.normalize_axis_index(axis, ar.ndim)

    result = torch.unique(
        ar, return_inverse=return_inverse, return_counts=return_counts, dim=axis
    )

    return result


def nonzero(a: ArrayLike) -> tuple[torch.Tensor, ...]:
    return torch.nonzero(a, as_tuple=True)


def argwhere(a: ArrayLike) -> torch.Tensor:
    return torch.argwhere(a)


def flatnonzero(a: ArrayLike) -> torch.Tensor:
    return torch.flatten(a).nonzero(as_tuple=True)[0]


def clip(
    a: ArrayLike,
    min: ArrayLike | None = None,
    max: ArrayLike | None = None,
    out: OutArray | None = None,
) -> torch.Tensor:
    return torch.clamp(a, min, max)


def repeat(
    a: ArrayLike, repeats: ArrayLikeOrScalar, axis: int | None = None
) -> torch.Tensor:
    # torch.repeat_interleave accepts a tensor/int repeats and optional axis.
    # pyrefly: ignore[no-matching-overload]
    return torch.repeat_interleave(a, repeats, axis)


def tile(A: ArrayLike, reps: int | Sequence[int]) -> torch.Tensor:
    if isinstance(reps, int):
        reps = (reps,)
    return torch.tile(A, reps)


def resize(a: ArrayLike, new_shape: int | Sequence[int] | None = None) -> torch.Tensor:
    # implementation vendored from
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/fromnumeric.py#L1420-L1497
    if new_shape is None:
        return a

    if isinstance(new_shape, int):
        new_shape = (new_shape,)

    a = a.flatten()

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")

    if a.numel() == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return torch.zeros(new_shape, dtype=a.dtype)

    repeats = -(-new_size // a.numel())  # ceil division
    a = concatenate((a,) * repeats)[:new_size]

    return reshape(a, new_shape)


# ### diag et al. ###


def diagonal(
    a: ArrayLike, offset: int = 0, axis1: int = 0, axis2: int = 1
) -> torch.Tensor:
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.diagonal(a, offset, axis1, axis2)


def trace(
    a: ArrayLike,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
) -> torch.Tensor:
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


def identity(
    n: int, dtype: DTypeLike | None = None, *, like: NotImplementedType = None
) -> torch.Tensor:
    return torch.eye(n, dtype=dtype)


def diag(v: ArrayLike, k: int = 0) -> torch.Tensor:
    return torch.diag(v, k)


def diagflat(v: ArrayLike, k: int = 0) -> torch.Tensor:
    return torch.diagflat(v, k)


def diag_indices(n: int, ndim: int = 2) -> tuple[torch.Tensor, ...]:
    idx = torch.arange(n)
    return (idx,) * ndim


def diag_indices_from(arr: ArrayLike) -> tuple[torch.Tensor, ...]:
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(s[0], arr.ndim)


def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap: bool = False) -> torch.Tensor:
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if val.numel() == 0 and not wrap:
        # NB: latent bug -- Tensor.fill_diagonal_ requires a scalar, so this
        # path raises at runtime for a tensor `val` (numpy treats it as a no-op).
        a.fill_diagonal_(val)  # pyrefly: ignore[bad-argument-type]
        return a

    if val.ndim == 0:
        val = val.unsqueeze(0)

    # torch.Tensor.fill_diagonal_ only accepts scalars
    # If the size of val is too large, then val is trimmed
    if a.ndim == 2:
        tall = a.shape[0] > a.shape[1]
        # wrap does nothing for wide matrices...
        if not wrap or not tall:
            # Never wraps
            diag = a.diagonal()
            diag.copy_(val[: diag.numel()])
        else:
            # wraps and tall... leaving one empty line between diagonals?!
            max_, min_ = a.shape
            idx = torch.arange(max_ - max_ // (min_ + 1))
            mod = idx % min_
            div = idx // min_
            a[(div * (min_ + 1) + mod, mod)] = val[: idx.numel()]
    else:
        idx = diag_indices_from(a)
        # a.shape = (n, n, ..., n)
        a[idx] = val[: a.shape[0]]

    return a


def vdot(a: ArrayLike, b: ArrayLike, /) -> torch.Tensor:
    # 1. torch only accepts 1D arrays, numpy flattens
    # 2. torch requires matching dtype, while numpy casts (?)
    t_a, t_b = torch.atleast_1d(a, b)
    if t_a.ndim > 1:
        t_a = t_a.flatten()
    if t_b.ndim > 1:
        t_b = t_b.flatten()

    dtype = _dtypes_impl.result_type_impl(t_a, t_b)
    is_half = dtype == torch.float16 and (t_a.is_cpu or t_b.is_cpu)
    is_bool = dtype == torch.bool

    # work around torch's "dot" not implemented for 'Half', 'Bool'
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    t_a = _util.cast_if_needed(t_a, dtype)
    t_b = _util.cast_if_needed(t_b, dtype)

    result = torch.vdot(t_a, t_b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)

    return result


def tensordot(
    a: ArrayLike,
    b: ArrayLike,
    axes: int | Sequence[int] | Sequence[Sequence[int]] = 2,
) -> torch.Tensor:
    if isinstance(axes, (list, tuple)):
        axes = [[ax] if isinstance(ax, int) else ax for ax in axes]

    target_dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, target_dtype)
    b = _util.cast_if_needed(b, target_dtype)

    return torch.tensordot(a, b, dims=axes)


def dot(a: ArrayLike, b: ArrayLike, out: OutArray | None = None) -> torch.Tensor:
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_bool = dtype == torch.bool
    if is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    if a.ndim == 0 or b.ndim == 0:
        result = a * b
    else:
        result = torch.matmul(a, b)

    if is_bool:
        result = result.to(torch.bool)

    return result


def inner(a: ArrayLike, b: ArrayLike, /) -> torch.Tensor:
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_half = dtype == torch.float16 and (a.is_cpu or b.is_cpu)
    is_bool = dtype == torch.bool

    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    result = torch.inner(a, b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result


def outer(a: ArrayLike, b: ArrayLike, out: OutArray | None = None) -> torch.Tensor:
    return torch.outer(a, b)


def cross(
    a: ArrayLike,
    b: ArrayLike,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> torch.Tensor:
    # implementation vendored from
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1486-L1685
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    # Check axisa and axisb are within bounds
    axisa = _util.normalize_axis_index(axisa, a.ndim)
    axisb = _util.normalize_axis_index(axisb, b.ndim)

    # Move working axis to the end of the shape
    a = torch.moveaxis(a, axisa, -1)
    b = torch.moveaxis(b, axisb, -1)
    msg = "incompatible dimensions for cross product\n(dimension must be 2 or 3)"
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)

    # Create the output array
    shape = broadcast_shapes(a[..., 0].shape, b[..., 0].shape)
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # Check axisc is within bounds
        axisc = _util.normalize_axis_index(axisc, len(shape))
    dtype = _dtypes_impl.result_type_impl(a, b)
    cp = torch.empty(shape, dtype=dtype)

    # recast arrays as dtype
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    b0 = b[..., 0]
    b1 = b[..., 1]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cp[...] = a0 * b1 - a1 * b0
            return cp
        # from here the output is 3D, so cp.shape[-1] == 3
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]
        if b.shape[-1] != 3:
            raise AssertionError(f"b.shape[-1] must be 3, got {b.shape[-1]}")
        b2 = b[..., 2]
        # cp0 = a1 * b2 - 0  (a2 = 0)
        # cp1 = 0 - a0 * b2  (a2 = 0)
        # cp2 = a0 * b1 - a1 * b0
        cp0[...] = a1 * b2
        cp1[...] = -a0 * b2
        cp2[...] = a0 * b1 - a1 * b0
    else:
        if a.shape[-1] != 3:
            raise AssertionError(f"a.shape[-1] must be 3, got {a.shape[-1]}")
        # from here the output is 3D, so cp.shape[-1] == 3
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]
        a2 = a[..., 2]
        if b.shape[-1] == 3:
            b2 = b[..., 2]
            cp0[...] = a1 * b2 - a2 * b1
            cp1[...] = a2 * b0 - a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
        else:
            if b.shape[-1] != 2:
                raise AssertionError(f"b.shape[-1] must be 2, got {b.shape[-1]}")
            cp0[...] = -a2 * b1
            cp1[...] = a2 * b0
            cp2[...] = a0 * b1 - a1 * b0

    return torch.moveaxis(cp, -1, axisc)


def einsum(
    *operands: object,
    out: object = None,
    dtype: object = None,
    order: str = "K",
    casting: str = "safe",
    optimize: bool | str = False,
) -> object:
    # Have to manually normalize *operands and **kwargs, following the NumPy signature
    # We have a local import to avoid polluting the global space, as it will be then
    # exported in funcs.py
    from ._ndarray import ndarray
    from ._normalizations import (
        maybe_copy_to,
        normalize_array_like,
        normalize_casting,
        normalize_dtype,
        wrap_tensors,
    )

    dtype = normalize_dtype(dtype)
    casting = normalize_casting(casting)
    if out is not None and not isinstance(out, ndarray):
        raise TypeError("'out' must be an array")
    if order != "K":
        raise NotImplementedError("'order' parameter is not supported.")

    # parse arrays and normalize them
    subscripts: object = None
    sublist_format = not isinstance(operands[0], str)
    if sublist_format:
        # op, str, op, str ... [sublistout] format: normalize every other argument

        # - if sublistout is not given, the length of operands is even, and we pick
        #   odd-numbered elements, which are arrays.
        # - if sublistout is given, the length of operands is odd, we peel off
        #   the last one, and pick odd-numbered elements, which are arrays.
        #   Without [:-1], we would have picked sublistout, too.
        array_operands = operands[:-1][::2]
    else:
        # ("ij->", arrays) format
        subscripts, array_operands = operands[0], operands[1:]

    tensors = [normalize_array_like(op) for op in array_operands]
    target_dtype = _dtypes_impl.result_type_impl(*tensors) if dtype is None else dtype

    # work around 'bmm' not implemented for 'Half' etc
    is_half = target_dtype == torch.float16 and all(t.is_cpu for t in tensors)
    if is_half:
        target_dtype = torch.float32

    is_short_int = target_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32]
    if is_short_int:
        target_dtype = torch.int64

    tensors = _util.typecast_tensors(tensors, target_dtype, casting)

    from torch.backends import opt_einsum

    # Read current state unconditionally so it is always bound for the finally
    # block below (only actually restored when opt_einsum is available).
    old_strategy = torch.backends.opt_einsum.strategy
    old_enabled = torch.backends.opt_einsum.enabled

    try:
        # set the global state to handle the optimize=... argument, restore on exit
        if opt_einsum.is_available():
            # torch.einsum calls opt_einsum.contract_path, which runs into
            # https://github.com/dgasmith/opt_einsum/issues/219
            # for strategy={True, False}
            if optimize is True:
                optimize = "auto"
            elif optimize is False:
                torch.backends.opt_einsum.enabled = False

            # `optimize` may be False here, which the setter ignores while
            # opt_einsum is disabled.
            torch.backends.opt_einsum.strategy = (
                optimize  # pyrefly: ignore[bad-assignment]
            )

        if sublist_format:
            # recombine operands
            sublists = operands[1::2]
            has_sublistout = len(operands) % 2 == 1
            sublistout = operands[-1] if has_sublistout else None
            einsum_operands = list(
                itertools.chain.from_iterable(zip(tensors, sublists))
            )
            if has_sublistout:
                einsum_operands.append(sublistout)

            result = torch.einsum(*einsum_operands)
        else:
            result = torch.einsum(subscripts, *tensors)

    finally:
        if opt_einsum.is_available():
            torch.backends.opt_einsum.strategy = old_strategy
            torch.backends.opt_einsum.enabled = old_enabled

    result = maybe_copy_to(out, result)
    return wrap_tensors(result)


# ### sort and partition ###


def _sort_helper(
    tensor: torch.Tensor, axis: int | None, kind: str | None, order: object
) -> tuple[torch.Tensor, int, bool]:
    if tensor.dtype.is_complex:
        raise NotImplementedError(f"sorting {tensor.dtype} is not supported")
    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    stable = kind == "stable"

    return tensor, axis, stable


def sort(
    a: ArrayLike,
    axis: int = -1,
    kind: str | None = None,
    order: NotImplementedType = None,
) -> torch.Tensor:
    # `order` keyword arg is only relevant for structured dtypes; so not supported here.
    a, axis, stable = _sort_helper(a, axis, kind, order)
    result = torch.sort(a, dim=axis, stable=stable)
    return result.values


def argsort(
    a: ArrayLike,
    axis: int = -1,
    kind: str | None = None,
    order: NotImplementedType = None,
) -> torch.Tensor:
    a, axis, stable = _sort_helper(a, axis, kind, order)
    return torch.argsort(a, dim=axis, stable=stable)


def searchsorted(
    a: ArrayLike, v: ArrayLike, side: str = "left", sorter: ArrayLike | None = None
) -> torch.Tensor:
    if a.dtype.is_complex:
        raise NotImplementedError(f"searchsorted with dtype={a.dtype}")

    return torch.searchsorted(a, v, side=side, sorter=sorter)


# ### swap/move/roll axis ###


def moveaxis(
    a: ArrayLike, source: int | Sequence[int], destination: int | Sequence[int]
) -> torch.Tensor:
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    return torch.moveaxis(a, source, destination)


def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> torch.Tensor:
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.swapaxes(a, axis1, axis2)


def rollaxis(a: ArrayLike, axis: int, start: int = 0) -> torch.Tensor:
    # Straight vendor from:
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1259
    #
    # Also note this function in NumPy is mostly retained for backwards compat
    # (https://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing)
    # so let's not touch it unless hard pressed.
    n = a.ndim
    axis = _util.normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise _util.AxisError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        # numpy returns a view, here we try returning the tensor itself
        # return tensor[...]
        return a
    axes = list(range(n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)


def roll(
    a: ArrayLike,
    shift: int | tuple[int, ...],
    axis: int | Sequence[int] | None = None,
) -> torch.Tensor:
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    # torch.roll accepts axis=None (rolls the flattened tensor) at runtime.
    return torch.roll(a, shift, axis)  # pyrefly: ignore[bad-argument-type]


# ### shape manipulations ###


def squeeze(a: ArrayLike, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    if axis == ():
        result = a
    elif axis is None:
        result = a.squeeze()
    else:
        if isinstance(axis, tuple):
            result = a
            for ax in axis:
                result = a.squeeze(ax)
        else:
            result = a.squeeze(axis)
    return result


def reshape(
    a: ArrayLike, newshape: Sequence[int], order: NotImplementedType = "C"
) -> torch.Tensor:
    # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
    shape = newshape[0] if len(newshape) == 1 else newshape
    return a.reshape(shape)


# NB: cannot use torch.reshape(a, newshape) above, because of
# (Pdb) torch.reshape(torch.as_tensor([1]), 1)
# *** TypeError: reshape(): argument 'shape' (position 2) must be tuple of SymInts, not int


def transpose(a: ArrayLike, axes: int | Sequence[int] | None = None) -> torch.Tensor:
    # numpy allows both .transpose(sh) and .transpose(*sh)
    # also older code uses axes being a list
    if axes in [(), None, (None,)]:
        axes = tuple(reversed(range(a.ndim)))
    elif isinstance(axes, (tuple, list)) and len(axes) == 1:
        axes = axes[0]
    # axes is never None here (the membership check above replaces it).
    return a.permute(axes)  # pyrefly: ignore[no-matching-overload]


def ravel(a: ArrayLike, order: NotImplementedType = "C") -> torch.Tensor:
    return torch.flatten(a)


def diff(
    a: ArrayLike,
    n: int = 1,
    axis: int = -1,
    prepend: ArrayLike | None = None,
    append: ArrayLike | None = None,
) -> torch.Tensor:
    axis = _util.normalize_axis_index(axis, a.ndim)

    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    if n == 0:
        # match numpy and return the input immediately
        return a

    if prepend is not None:
        shape = list(a.shape)
        shape[axis] = prepend.shape[axis] if prepend.ndim > 0 else 1
        prepend = torch.broadcast_to(prepend, shape)

    if append is not None:
        shape = list(a.shape)
        shape[axis] = append.shape[axis] if append.ndim > 0 else 1
        append = torch.broadcast_to(append, shape)

    # torch.diff accepts `axis` as an alias for `dim` at runtime.
    # pyrefly: ignore[unexpected-keyword]
    return torch.diff(a, n, axis=axis, prepend=prepend, append=append)


# ### math functions ###


def angle(z: ArrayLike, deg: bool = False) -> torch.Tensor:
    result = torch.angle(z)
    if deg:
        result = result * (180 / torch.pi)
    return result


def sinc(x: ArrayLike) -> torch.Tensor:
    return torch.sinc(x)


# NB: have to normalize *varargs manually
def gradient(
    f: ArrayLike,
    *varargs: object,
    axis: int | tuple[int, ...] | None = None,
    edge_order: int = 1,
) -> torch.Tensor | list[torch.Tensor]:
    N = f.ndim  # number of dimensions

    # ndarrays_to_tensors preserves the tuple structure, converting any ndarray
    # spacing arguments to tensors and leaving scalars intact.
    spacing = _util.ndarrays_to_tensors(varargs)
    if not isinstance(spacing, tuple):
        raise AssertionError

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _util.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(spacing)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and (
        _dtypes_impl.is_scalar(spacing[0])
        or (isinstance(spacing[0], torch.Tensor) and spacing[0].ndim == 0)
    ):
        # single scalar or 0D tensor for all axes (np.ndim(spacing[0]) == 0)
        dx = spacing * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(spacing)
        for i, distances in enumerate(dx):
            distances = torch.as_tensor(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )
            if not (distances.dtype.is_floating_point or distances.dtype.is_complex):
                distances = distances.double()

            diffx = torch.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1: list[slice | int] = [slice(None)] * N
    slice2: list[slice | int] = [slice(None)] * N
    slice3: list[slice | int] = [slice(None)] * N
    slice4: list[slice | int] = [slice(None)] * N

    otype = f.dtype
    if _dtypes_impl.python_type_for_torch(otype) in (int, bool):
        # Convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        f = f.double()
        otype = torch.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )
        # result allocation
        out = torch.empty_like(f, dtype=otype)

        # spacing for the current axis (NB: np.ndim(ax_dx) == 0)
        uniform_spacing = _dtypes_impl.is_scalar(ax_dx) or (
            isinstance(ax_dx, torch.Tensor) and ax_dx.ndim == 0
        )

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            # non-uniform spacing is always a 1d tensor (as_tensor is a no-op)
            ax_dx = torch.as_tensor(ax_dx)
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = [1] * N
            shape[axis] = -1
            a = a.reshape(shape)
            b = b.reshape(shape)
            c = c.reshape(shape)
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            if uniform_spacing:
                dx_0 = ax_dx
            else:
                ax_dx = torch.as_tensor(ax_dx)
                dx_0 = ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            if uniform_spacing:
                dx_n = ax_dx
            else:
                ax_dx = torch.as_tensor(ax_dx)
                dx_n = ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                ax_dx = torch.as_tensor(ax_dx)
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                ax_dx = torch.as_tensor(ax_dx)
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


# ### Type/shape etc queries ###


def round(a: ArrayLike, decimals: int = 0, out: OutArray | None = None) -> torch.Tensor:
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = torch.complex(
            torch.round(a.real, decimals=decimals),
            torch.round(a.imag, decimals=decimals),
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = a
    return result


around = round
round_ = round


def real_if_close(a: ArrayLike, tol: float = 100) -> torch.Tensor:
    if not torch.is_complex(a):
        return a
    if tol > 1:
        # Undocumented in numpy: if tol < 1, it's an absolute tolerance!
        # Otherwise, tol > 1 is relative tolerance, in units of the dtype epsilon
        # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L577
        tol = tol * torch.finfo(a.dtype).eps

    mask = torch.abs(a.imag) < tol
    return a.real if mask.all() else a


def real(a: ArrayLike) -> torch.Tensor:
    return torch.real(a)


def imag(a: ArrayLike) -> torch.Tensor:
    if a.is_complex():
        return a.imag
    return torch.zeros_like(a)


def iscomplex(x: ArrayLike) -> torch.Tensor:
    if torch.is_complex(x):
        return x.imag != 0
    return torch.zeros_like(x, dtype=torch.bool)


def isreal(x: ArrayLike) -> torch.Tensor:
    if torch.is_complex(x):
        return x.imag == 0
    return torch.ones_like(x, dtype=torch.bool)


def iscomplexobj(x: ArrayLike) -> bool:
    return torch.is_complex(x)


def isrealobj(x: ArrayLike) -> bool:
    return not torch.is_complex(x)


def isneginf(x: ArrayLike, out: OutArray | None = None) -> torch.Tensor:
    return torch.isneginf(x)


def isposinf(x: ArrayLike, out: OutArray | None = None) -> torch.Tensor:
    return torch.isposinf(x)


def i0(x: ArrayLike) -> torch.Tensor:
    return torch.special.i0(x)


def isscalar(a: object) -> bool:
    # We need to use normalize_array_like, but we don't want to export it in funcs.py
    from ._normalizations import normalize_array_like

    try:
        t = normalize_array_like(a)
        return t.numel() == 1
    except Exception:
        return False


# ### Filter windows ###


def hamming(M: int) -> torch.Tensor:
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hamming_window(M, periodic=False, dtype=dtype)


def hanning(M: int) -> torch.Tensor:
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hann_window(M, periodic=False, dtype=dtype)


def kaiser(M: int, beta: float) -> torch.Tensor:
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.kaiser_window(M, beta=beta, periodic=False, dtype=dtype)


def blackman(M: int) -> torch.Tensor:
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.blackman_window(M, periodic=False, dtype=dtype)


def bartlett(M: int) -> torch.Tensor:
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.bartlett_window(M, periodic=False, dtype=dtype)


# ### Dtype routines ###

# vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L666


array_type = [
    [torch.float16, torch.float32, torch.float64],
    [None, torch.complex64, torch.complex128],
]
array_precision = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.complex64: 1,
    torch.complex128: 2,
}


def common_type(*tensors: ArrayLike) -> torch.dtype | None:
    is_complex = False
    precision = 0
    for a in tensors:
        t = a.dtype
        if iscomplexobj(a):
            is_complex = True
        if not (t.is_floating_point or t.is_complex):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = builtins.max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]


# ### histograms ###


def histogram(
    a: ArrayLike,
    # the `10` default is normalized to a tensor only when passed explicitly
    bins: ArrayLike = 10,  # pyrefly: ignore[bad-function-definition]
    range: tuple[float, float] | None = None,
    normed: bool | None = None,
    weights: ArrayLike | None = None,
    density: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")

    if weights is not None and weights.dtype.is_complex:
        raise NotImplementedError("complex weights histogram.")

    is_a_int = not (a.dtype.is_floating_point or a.dtype.is_complex)
    is_w_int = weights is None or not weights.dtype.is_floating_point
    if is_a_int:
        a = a.double()

    if weights is not None:
        weights = _util.cast_if_needed(weights, a.dtype)

    n_bins: torch.Tensor | int = bins
    if isinstance(n_bins, torch.Tensor):
        if n_bins.ndim == 0:
            # bins was a single int
            n_bins = operator.index(n_bins)
        else:
            n_bins = _util.cast_if_needed(n_bins, a.dtype)

    # torch.histogram accepts either a tensor of edges or an int count of bins.
    if range is None:
        h, b = torch.histogram(  # pyrefly: ignore[no-matching-overload]
            a, n_bins, weight=weights, density=bool(density)
        )
    else:
        h, b = torch.histogram(  # pyrefly: ignore[no-matching-overload]
            a, n_bins, range=range, weight=weights, density=bool(density)
        )

    if not density and is_w_int:
        h = h.long()
    if is_a_int:
        b = b.long()

    return h, b


def histogram2d(
    x: torch.Tensor | Sequence[float],
    y: torch.Tensor | Sequence[float],
    bins: int | Sequence[int] = 10,
    range: ArrayLike | None = None,
    normed: bool | None = None,
    weights: ArrayLike | None = None,
    density: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/twodim_base.py#L655-L821
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    try:
        # a scalar `bins` raises TypeError from len(), handled below
        N = len(bins)  # pyrefly: ignore[bad-argument-type]
    except TypeError:
        N = 1

    hist_bins: int | Sequence[object] = bins
    if N != 1 and N != 2:
        hist_bins = [bins, bins]

    h, e = histogramdd((x, y), hist_bins, range, normed, weights, density)

    return h, e[0], e[1]


def histogramdd(
    sample: object,
    bins: int | Sequence[object] = 10,
    range: ArrayLike | None = None,
    normed: bool | None = None,
    weights: ArrayLike | None = None,
    density: bool | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # have to normalize manually because `sample` interpretation differs
    # for a list of lists and a 2D array
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")

    from ._normalizations import normalize_array_like, normalize_seq_array_like

    if isinstance(sample, (list, tuple)):
        sample = normalize_array_like(sample).T
    else:
        sample = normalize_array_like(sample)

    sample = torch.atleast_2d(sample)

    if not (sample.dtype.is_floating_point or sample.dtype.is_complex):
        sample = sample.double()

    # bins is either an int, or a sequence of ints or a sequence of arrays
    bins_dtypes: list[torch.dtype] = []
    bins_seq: int | Sequence[object]
    if isinstance(bins, int) or builtins.all(isinstance(b, int) for b in bins):
        bins_is_array = False
        bins_seq = bins
    else:
        bins_is_array = True
        bins_tensors = normalize_seq_array_like(bins)
        bins_dtypes = [b.dtype for b in bins_tensors]
        bins_seq = [_util.cast_if_needed(b, sample.dtype) for b in bins_tensors]

    dd_range: list[float] | tuple[float, ...] | None
    w_kwd: dict[str, torch.Tensor]
    if weights is not None:
        # range=... is required : interleave min and max values per dimension
        mm = sample.aminmax(dim=0)
        edges = torch.cat(mm).reshape(2, -1).T.flatten()
        dd_range = tuple(edges.tolist())
        weights = _util.cast_if_needed(weights, sample.dtype)
        w_kwd = {"weight": weights}
    else:
        dd_range = range.flatten().tolist() if range is not None else None
        w_kwd = {}

    # torch.histogramdd accepts an int/sequence of bins and an optional range.
    h, b = torch.histogramdd(  # pyrefly: ignore[no-matching-overload]
        sample, bins_seq, dd_range, density=bool(density), **w_kwd
    )

    if bins_is_array:
        b = [_util.cast_if_needed(bb, dtyp) for bb, dtyp in zip(b, bins_dtypes)]

    return h, b


# ### odds and ends


def min_scalar_type(a: ArrayLike, /) -> DType:
    # https://github.com/numpy/numpy/blob/maintenance/1.24.x/numpy/core/src/multiarray/convert_datatype.c#L1288

    from ._dtypes import DType

    if a.numel() > 1:
        # numpy docs: "For non-scalar array a, returns the vector's dtype unmodified."
        return DType(a.dtype)

    # fallback dtype; the branches below always find a fit for a 1-element array
    dtype: torch.dtype = a.dtype
    if a.dtype == torch.bool:
        dtype = torch.bool

    elif a.dtype.is_complex:
        fi = torch.finfo(torch.float32)
        fits_in_single = a.dtype == torch.complex64 or (
            fi.min <= a.real <= fi.max and fi.min <= a.imag <= fi.max
        )
        dtype = torch.complex64 if fits_in_single else torch.complex128

    elif a.dtype.is_floating_point:
        for dt in [torch.float16, torch.float32, torch.float64]:
            fi = torch.finfo(dt)
            if fi.min <= a <= fi.max:
                dtype = dt
                break
    else:
        # must be integer
        for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            # Prefer unsigned int where possible, as numpy does.
            ii = torch.iinfo(dt)
            if ii.min <= a <= ii.max:
                dtype = dt
                break

    return DType(dtype)


def pad(
    array: ArrayLike, pad_width: ArrayLike, mode: str = "constant", **kwargs: object
) -> torch.Tensor:
    if mode != "constant":
        raise NotImplementedError
    value = kwargs.get("constant_values", 0)
    # `value` must be a python scalar for torch.nn.functional.pad; `typ` is the
    # array's python scalar type, chosen dynamically so the constructor call is
    # opaque to the type checker.
    typ = _dtypes_impl.python_type_for_torch(array.dtype)
    value = typ(value)  # pyrefly: ignore[no-matching-overload, bad-argument-type]

    pad_width = torch.broadcast_to(pad_width, (array.ndim, 2))
    pad_width = torch.flip(pad_width, (0,)).flatten()

    # `value` is a python scalar accepted by pad even when not a plain float.
    return torch.nn.functional.pad(
        array,
        tuple(pad_width),
        value=value,  # pyrefly: ignore[bad-argument-type]
    )
