import torch
import torch._prims as prims
import torch._prims.utils as utils
from torch._prims.utils import (
    check,
    TensorLikeType,
)
from torch._prims.wrappers import (
    out_wrapper,
)
from torch._decomp import register_decomposition

from typing import Union, Tuple, Optional
from typing_extensions import Literal
import math

__all__ = ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]

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
