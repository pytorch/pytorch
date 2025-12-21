# Stub file for torch.fft
from collections.abc import Sequence

import torch
from torch import Tensor

# FFT operations
def fft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def ifft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def fft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def ifft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def fftn(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def ifftn(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...

# Real FFT
def rfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def irfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def rfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def irfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def rfftn(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def irfftn(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...

# Hermitian FFT
def hfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def ihfft(
    input: Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def hfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def ihfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def hfftn(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def ihfftn(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: int | list[int] | tuple[int, ...] | None = None,
    norm: str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...

# Helper functions
def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> Tensor: ...
def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> Tensor: ...
def fftshift(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
) -> Tensor: ...
def ifftshift(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
) -> Tensor: ...
