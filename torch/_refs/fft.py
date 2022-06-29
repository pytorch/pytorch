import torch
import torch._prims as prims
import torch._refs as refs
import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLike,
    TensorLikeType,
    ShapeType,
    DimsType,
    DimsSequenceType,
)
from torch._prims.wrappers import (
    out_wrapper,
)

from typing import Union, Tuple, Iterable, Sequence, Optional, NamedTuple
from typing_extensions import Literal
import math

__all__ = ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fftn", "ifftn"]

NormType = Union[None, Literal["forward"], Literal["backward"], Literal["ortho"]]


def _apply_norm(x: TensorLike, norm: NormType, signal_numel: int, forward: bool):
    if norm == "ortho":
        return prims.mul(x, 1 / math.sqrt(signal_numel))

    normalize = (not forward and (norm is None or norm == "backward")) or (
        forward and norm == "forward"
    )
    return prims.mul(x, 1 / signal_numel) if normalize else x


def _promote_type_fft(
    dtype: torch.dtype, require_complex: bool, device: torch.device
) -> torch.dtype:
    if dtype.is_complex:
        return dtype

    # Promote integral to default float type
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    is_rocm = False  # TODO: How to discern rocm from CUDA?
    if dtype == torch.half and (is_rocm or device.type != "cuda"):
        raise RuntimeError("Unsupported dtype Half")

    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)

    return dtype


def _promote_tensor_fft(t: TensorLikeType, require_complex: bool = False):
    cur_type = t.dtype
    new_type = _promote_type_fft(cur_type, require_complex, t.device)
    if cur_type == new_type:
        return t
    return prims.convert_element_type(t, new_type)


# Fixes the shape of x such that x.size(dims[i]) == sizes[i],
# either by zero-padding, or by slicing x starting from 0.
def _resize_fft_input(
    x: TensorLikeType, dims: Tuple[int, ...], sizes: Tuple[int, ...]
) -> TensorLikeType:
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
            x = refs.narrow(x, dims[i], 0, sizes[i])

    return refs.constant_pad_nd(x, pad_amount) if must_copy else x


def _fft_c2r(
    func_name: str,
    input: TensorLikeType,
    n: Optional[int],
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    input = _promote_tensor_fft(input, require_complex=True)
    dims = (utils.canonicalize_dim(input.ndim, dim),)
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)
    if last_dim_size < 1:
        raise RuntimeError(f"Invalid number of data points ({n}) specified")

    if n is not None:
        input = _resize_fft_input(input, dims=dims, sizes=(last_dim_size // 2 + 1,))

    if forward:
        input = prims.conj(input)

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
    if not input.is_floating_point:
        raise RuntimeError(
            f"{func_name} expects a floating point input tensor, but got {input.dtype}"
        )
    input = _promote_tensor_fft(input)
    dims = (utils.canonicalize_dim(input.ndim, dim),)

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)
    ret = _apply_norm(ret, norm, input.shape[dim], forward)
    return ret if forward else refs.conj(ret)


def _fft_c2c(
    func_name: str,
    input: TensorLikeType,
    n: Optional[int],
    dim: int,
    norm: NormType,
    forward: bool,
):
    if not input.dtype.is_complex:
        raise RuntimeError(
            f"{func_name} expects a complex input tensor, but got {input.dtype}"
        )
    dims = (utils.canonicalize_dim(input.ndim, dim),)

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_c2c(input, dim=dims, forward=forward)
    return _apply_norm(ret, norm, input.shape[dim], forward)


@out_wrapper
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


@out_wrapper
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


@out_wrapper
def rfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_r2c("rfft", input, n, dim, norm, forward=True, onesided=True)


@out_wrapper
def irfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_c2r("irfft", input, n, dim, norm, forward=False)


@out_wrapper
def hfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_c2r("hfft", input, n, dim, norm, forward=True)


@out_wrapper
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
    input_dim = input.ndim
    input_sizes = input.shape

    if dim is not None:
        if not isinstance(dim, Sequence):
            dim = (dim,)
        ret_dims = utils.canonicalize_dims(input_dim, dim)

        # Check dims are unique
        if len(set(dim)) != len(dim):
            raise RuntimeError("FFT dims must be unique")

    if shape is not None:
        if not isinstance(shape, Sequence):
            shape = (shape,)

        # Has shape, might have dim
        if dim is not None and len(dim) != len(shape):
            raise RuntimeError(
                "When given, dim and shape arguments must have the same length"
            )
        transform_ndim = len(shape)

        if transform_ndim > input_dim:
            raise RuntimeError(
                f"Got shape with {len(shape)} values but input tensor "
                f"only has {input_dim} dimensions."
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
        if n <= 0:
            raise RuntimeError(f"Invalid number of data points ({n}) specified")

    return _ShapeAndDims(shape=ret_shape, dims=ret_dims)


def _prod(xs: Iterable[int]):
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
    if not input.dtype.is_complex:
        raise RuntimeError(
            f"{function_name} expects a complex input tensor, " f"but got {input.dtype}"
        )
    x = _resize_fft_input(input, dim, shape)
    output = prims.fft_c2c(x, dim=dim, forward=forward)
    return _apply_norm(output, norm=norm, signal_numel=_prod(shape), forward=forward)


@out_wrapper
def fftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsSequenceType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("fftn", x, shape, dim, norm, forward=True)


@out_wrapper
def ifftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsSequenceType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("ifftn", x, shape, dim, norm, forward=False)
