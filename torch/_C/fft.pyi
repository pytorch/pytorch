# Defined in aten/src/ATen/native/SpectralOps.cpp
from torch import Tensor
from torch.types import _int

def fft(
    input: Tensor, s: _int | None = ..., dim: _int = ..., norm: str | None = ...
) -> Tensor: ...
def ifft(
    input: Tensor, s: _int | None = ..., dim: _int = ..., norm: str | None = ...
) -> Tensor: ...
def fft2(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def ifft2(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def fftn(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def ifftn(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def rfft(
    input: Tensor, s: _int | None = ..., dim: _int = ..., norm: str | None = ...
) -> Tensor: ...
def irfft(
    input: Tensor, s: _int | None = ..., dim: _int = ..., norm: str | None = ...
) -> Tensor: ...
def rfft2(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def irfft2(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def rfftn(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def irfftn(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def hfft(
    input: Tensor, s: _int | None = ..., dim: _int = ..., norm: str | None = ...
) -> Tensor: ...
def ihfft(
    input: Tensor, s: _int | None = ..., dim: _int = ..., norm: str | None = ...
) -> Tensor: ...
def hfft2(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def ihfft2(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def hfftn(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
def ihfftn(
    input: Tensor,
    s: tuple[_int] | None = ...,
    dim: tuple[_int] = ...,
    norm: str | None = ...,
) -> Tensor: ...
