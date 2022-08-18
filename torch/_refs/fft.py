import math

from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

from typing_extensions import Literal

import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import check, DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import out_wrapper

__all__ = [
    # Transforms
    "fft",
    "fft2",
    "fftn",
    "hfft",
    "hfft2",
    "hfftn",
    "rfft",
    "rfft2",
    "rfftn",
    "ifft",
    "ifft2",
    "ifftn",
    "ihfft",
    "ihfft2",
    "ihfftn",
    "irfft",
    "irfft2",
    "irfftn",
    # Helpers
    "fftshift",
    "ifftshift",
]

NormType = Union[None, Literal["forward"], Literal["backward"], Literal["ortho"]]
_NORM_VALUES = {None, "forward", "backward", "ortho"}


def _apply_norm(
    x: TensorLikeType, norm: NormType, signal_numel: int, forward: bool
) -> TensorLikeType:
    """Apply normalization to the un-normalized FFT result"""
    check(norm in _NORM_VALUES, lambda: f"Invalid normalization mode: {norm}")

    if norm == "ortho":
        return x * (1 / math.sqrt(signal_numel))

    normalize = (not forward and (norm is None or norm == "backward")) or (
        forward and norm == "forward"
    )
    return x * (1 / signal_numel) if normalize else x


def _promote_type_fft(dtype: torch.dtype, require_complex: bool) -> torch.dtype:
    """Helper to promote a dtype to one supported by the FFT primitives"""
    if dtype.is_complex:
        return dtype

    # Promote integral to default float type
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)

    return dtype


def _maybe_promote_tensor_fft(
    t: TensorLikeType, require_complex: bool = False
) -> TensorLikeType:
    """Helper to promote a tensor to a dtype supported by the FFT primitives"""
    cur_type = t.dtype
    new_type = _promote_type_fft(cur_type, require_complex)
    if cur_type == new_type:
        return t
    return prims.convert_element_type(t, new_type)


def _resize_fft_input(
    x: TensorLikeType, dims: Tuple[int, ...], sizes: Tuple[int, ...]
) -> TensorLikeType:
    """
    Fixes the shape of x such that x.size(dims[i]) == sizes[i],
    either by zero-padding, or by slicing x starting from 0.
    """
    assert len(dims) == len(sizes)
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    for i in range(len(dims)):
        if sizes[i] == -1:
            continue

        if x_sizes[dims[i]] < sizes[i]:
            must_copy = True
            pad_idx = len(pad_amount) - 2 * dims[i] - 1
            pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]]

        if x_sizes[dims[i]] > sizes[i]:
            x = x.narrow(dims[i], 0, sizes[i])

    return torch.constant_pad_nd(x, pad_amount) if must_copy else x


def _fft_c2r(
    func_name: str,
    input: TensorLikeType,
    n: Optional[int],
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    """Common code for performing any complex to real FFT (irfft or hfft)"""
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    dims = (utils.canonicalize_dim(input.ndim, dim),)
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)
    check(last_dim_size >= 1, lambda: f"Invalid number of data points ({n}) specified")

    if n is not None:
        input = _resize_fft_input(input, dims=dims, sizes=(last_dim_size // 2 + 1,))

    if forward:
        input = torch.conj(input)

    output = prims.fft_c2r(input, dim=dims, last_dim_size=last_dim_size)
    return _apply_norm(output, norm=norm, signal_numel=last_dim_size, forward=forward)


def _fft_r2c(
    func_name: str,
    input: TensorLikeType,
    n: Optional[int],
    dim: int,
    norm: NormType,
    forward: bool,
    onesided: bool,
) -> TensorLikeType:
    """Common code for performing any real to complex FFT (rfft or ihfft)"""
    check(
        not input.dtype.is_complex,
        lambda: f"{func_name} expects a floating point input tensor, but got {input.dtype}",
    )
    input = _maybe_promote_tensor_fft(input)
    dims = (utils.canonicalize_dim(input.ndim, dim),)

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)
    ret = _apply_norm(ret, norm, input.shape[dim], forward)
    return ret if forward else torch.conj(ret)


def _fft_c2c(
    func_name: str,
    input: TensorLikeType,
    n: Optional[int],
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    """Common code for performing any complex to complex FFT (fft or ifft)"""
    check(
        input.dtype.is_complex,
        lambda: f"{func_name} expects a complex input tensor, but got {input.dtype}",
    )
    dims = (utils.canonicalize_dim(input.ndim, dim),)

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_c2c(input, dim=dims, forward=forward)
    return _apply_norm(ret, norm, input.shape[dim], forward)


@register_decomposition(torch.ops.aten.fft_fft)
@out_wrapper()
def fft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    if input.dtype.is_complex:
        return _fft_c2c("fft", input, n, dim, norm, forward=True)
    else:
        return _fft_r2c("fft", input, n, dim, norm, forward=True, onesided=False)


@register_decomposition(torch.ops.aten.fft_ifft)
@out_wrapper()
def ifft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    if input.dtype.is_complex:
        return _fft_c2c("ifft", input, n, dim, norm, forward=False)
    else:
        return _fft_r2c("ifft", input, n, dim, norm, forward=False, onesided=False)


@register_decomposition(torch.ops.aten.fft_rfft)
@out_wrapper()
def rfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_r2c("rfft", input, n, dim, norm, forward=True, onesided=True)


@register_decomposition(torch.ops.aten.fft_irfft)
@out_wrapper()
def irfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_c2r("irfft", input, n, dim, norm, forward=False)


@register_decomposition(torch.ops.aten.fft_hfft)
@out_wrapper()
def hfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_c2r("hfft", input, n, dim, norm, forward=True)


@register_decomposition(torch.ops.aten.fft_ihfft)
@out_wrapper()
def ihfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_r2c("ihfft", input, n, dim, norm, forward=False, onesided=True)


class _ShapeAndDims(NamedTuple):
    shape: Tuple[int, ...]
    dims: Tuple[int, ...]


def _canonicalize_fft_shape_and_dim_args(
    input: TensorLikeType, shape: Optional[ShapeType], dim: Optional[DimsType]
) -> _ShapeAndDims:
    """Convert the shape and dim arguments into a canonical form where neither are optional"""
    input_dim = input.ndim
    input_sizes = input.shape

    if dim is not None:
        if not isinstance(dim, Sequence):
            dim = (dim,)
        ret_dims = utils.canonicalize_dims(input_dim, dim)

        # Check dims are unique
        check(len(set(dim)) == len(dim), lambda: "FFT dims must be unique")

    if shape is not None:
        if not isinstance(shape, Sequence):
            shape = (shape,)

        # Has shape, might have dim
        check(
            dim is None or len(dim) == len(shape),
            lambda: "When given, dim and shape arguments must have the same length",
        )
        transform_ndim = len(shape)

        check(
            transform_ndim <= input_dim,
            lambda: f"Got shape with {transform_ndim} values but input tensor "
            f"only has {input_dim} dimensions.",
        )

        # If shape is given, dims defaults to the last len(shape) dimensions
        if dim is None:
            ret_dims = tuple(range(input_dim - transform_ndim, input_dim))

        # Translate any -1 values in shape to the default length
        ret_shape = tuple(
            s if s != -1 else input_sizes[d] for (s, d) in zip(shape, ret_dims)
        )
    elif dim is None:
        # No shape, no dim
        ret_dims = tuple(range(input_dim))
        ret_shape = tuple(input_sizes)
    else:
        # No shape, has dim
        ret_shape = tuple(input_sizes[d] for d in ret_dims)

    for n in ret_shape:
        check(n > 0, lambda: f"Invalid number of data points ({n}) specified")

    return _ShapeAndDims(shape=ret_shape, dims=ret_dims)


def _prod(xs: Iterable[int]) -> int:
    """Compute product of a list"""
    prod = 1
    for x in xs:
        prod *= x
    return prod


def _fftn_c2c(
    function_name: str,
    input: TensorLikeType,
    shape: Tuple[int, ...],
    dim: Tuple[int, ...],
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    """Common code for n-dimensional complex to complex FFTs (fftn or ifftn)"""
    check(
        input.dtype.is_complex,
        lambda: f"{function_name} expects a complex input tensor, "
        f"but got {input.dtype}",
    )
    x = _resize_fft_input(input, dim, shape)
    output = prims.fft_c2c(x, dim=dim, forward=forward)
    return _apply_norm(output, norm=norm, signal_numel=_prod(shape), forward=forward)


@register_decomposition(torch.ops.aten.fft_fftn)
@out_wrapper()
def fftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _maybe_promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("fftn", x, shape, dim, norm, forward=True)


@register_decomposition(torch.ops.aten.fft_ifftn)
@out_wrapper()
def ifftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _maybe_promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("ifftn", x, shape, dim, norm, forward=False)


@register_decomposition(torch.ops.aten.fft_rfftn)
@out_wrapper()
def rfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    check(
        not input.dtype.is_complex,
        lambda: f"rfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    input = _maybe_promote_tensor_fft(input, require_complex=False)
    input = _resize_fft_input(input, dim, shape)
    out = prims.fft_r2c(input, dim=dim, onesided=True)
    return _apply_norm(out, norm=norm, signal_numel=_prod(shape), forward=True)


@register_decomposition(torch.ops.aten.fft_ihfftn)
@out_wrapper()
def ihfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    check(
        not input.dtype.is_complex,
        lambda: f"ihfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    check(len(shape) > 0, lambda: "ihfftn must transform at least one axis")
    input = _maybe_promote_tensor_fft(input, require_complex=False)
    input = _resize_fft_input(input, dim, shape)

    tmp = prims.fft_r2c(input, dim=dim[-1:], onesided=True)

    if len(dim) == 1:
        tmp = _apply_norm(tmp, norm=norm, signal_numel=shape[0], forward=False)
        return prims.conj(tmp)

    tmp = prims.conj_physical(tmp)
    tmp = prims.fft_c2c(tmp, dim=dim[:-1], forward=False)
    return _apply_norm(tmp, norm=norm, signal_numel=_prod(shape), forward=False)


class _CanonicalizeC2rReturn(NamedTuple):
    shape: Tuple[int, ...]
    dim: Tuple[int, ...]
    last_dim_size: int


def _canonicalize_fft_c2r_shape_and_dim_args(
    fname: str,
    input: TensorLikeType,
    s: Optional[ShapeType],
    dim: Optional[DimsType],
) -> _CanonicalizeC2rReturn:
    """Canonicalize shape and dim arguments for n-dimensional c2r transforms,
    as well as calculating the last_dim_size which is shape[dim[-1]] for the output"""
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    check(len(shape) > 0, lambda: f"{fname} must transform at least one axis")

    if s is None or s[-1] == -1:
        last_dim_size = 2 * (input.shape[dim[-1]] - 1)
    else:
        last_dim_size = shape[-1]

    check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    shape_list = list(shape)
    shape_list[-1] = last_dim_size // 2 + 1
    return _CanonicalizeC2rReturn(
        shape=tuple(shape_list), dim=dim, last_dim_size=last_dim_size
    )


@register_decomposition(torch.ops.aten.fft_irfftn)
@out_wrapper()
def irfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "irfftn", input, s, dim
    )
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    input = _resize_fft_input(input, dim, shape)
    out = prims.fft_c2r(input, dim=dim, last_dim_size=last_dim_size)
    return _apply_norm(out, norm, _prod(out.shape[d] for d in dim), forward=False)


@register_decomposition(torch.ops.aten.fft_hfftn)
@out_wrapper()
def hfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "hfftn", input, s, dim
    )
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    input = _resize_fft_input(input, dim, shape)

    tmp = prims.fft_c2c(input, dim=dim[:-1], forward=True) if len(dim) > 1 else input
    tmp = _apply_norm(tmp, norm, _prod(shape[:-1]), forward=True)
    tmp = prims.conj_physical(tmp)
    out = prims.fft_c2r(tmp, dim=dim[-1:], last_dim_size=last_dim_size)
    return _apply_norm(out, norm, last_dim_size, forward=True)


@register_decomposition(torch.ops.aten.fft_fft2)
@out_wrapper()
def fft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(torch.ops.aten.fft_ifft2)
@out_wrapper()
def ifft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.ifftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(torch.ops.aten.fft_rfft2)
@out_wrapper()
def rfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.rfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(torch.ops.aten.fft_irfft2)
@out_wrapper()
def irfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.irfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(torch.ops.aten.fft_hfft2)
@out_wrapper()
def hfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.hfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(torch.ops.aten.fft_ihfft2)
@out_wrapper()
def ihfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    return torch.fft.ihfftn(input, s=s, dim=dim, norm=norm)


def _default_alldims(dim: Optional[DimsType], x: TensorLikeType) -> List[int]:
    """Convert Optional[DimsType] to a simple list, defaulting to all dimensions"""
    if dim is None:
        return list(range(x.ndim))
    elif not isinstance(dim, Sequence):
        return [dim]
    else:
        return list(dim)


@register_decomposition(torch.ops.aten.fft_fftshift)
def fftshift(input: TensorLikeType, dim: Optional[DimsType] = None) -> TensorLikeType:
    dims = _default_alldims(dim, input)
    shift = [input.shape[d] // 2 for d in dims]
    return torch.roll(input, shift, dims)


@register_decomposition(torch.ops.aten.fft_ifftshift)
def ifftshift(input: TensorLikeType, dim: Optional[DimsType] = None) -> TensorLikeType:
    dims = _default_alldims(dim, input)
    shift = [(input.shape[d] + 1) // 2 for d in dims]
    return torch.roll(input, shift, dims)
