# mypy: ignore-errors

from __future__ import annotations

import functools

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


def upcast(func):
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""

    @functools.wraps(func)
    def wrapped(tensor, *args, **kwds):
        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        tensor = _util.cast_if_needed(tensor, target_dtype)
        return func(tensor, *args, **kwds)

    return wrapped


@normalizer
@upcast
def fft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.fft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ifft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.ifft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def rfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.rfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def irfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.irfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def fftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.fftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.ifftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.rfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.irfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def fft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.fft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.ifft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.rfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.irfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def hfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.hfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ihfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.ihfft(a, n, dim=axis, norm=norm)


@normalizer
def fftfreq(n, d=1.0):
    return torch.fft.fftfreq(n, d)


@normalizer
def rfftfreq(n, d=1.0):
    return torch.fft.rfftfreq(n, d)


@normalizer
def fftshift(x: ArrayLike, axes=None):
    return torch.fft.fftshift(x, axes)


@normalizer
def ifftshift(x: ArrayLike, axes=None):
    return torch.fft.ifftshift(x, axes)
