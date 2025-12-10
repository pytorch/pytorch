from collections.abc import Sequence
from typing import Literal as L
from typing import TypeAlias

from numpy import complex128, float64
from numpy._typing import ArrayLike, NDArray, _ArrayLikeNumber_co

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "rfftn",
    "irfftn",
    "rfft2",
    "irfft2",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
]

_NormKind: TypeAlias = L["backward", "ortho", "forward"] | None

def fft(
    a: ArrayLike,
    n: int | None = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def ifft(
    a: ArrayLike,
    n: int | None = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def rfft(
    a: ArrayLike,
    n: int | None = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def irfft(
    a: ArrayLike,
    n: int | None = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: NDArray[float64] | None = ...,
) -> NDArray[float64]: ...

# Input array must be compatible with `np.conjugate`
def hfft(
    a: _ArrayLikeNumber_co,
    n: int | None = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: NDArray[float64] | None = ...,
) -> NDArray[float64]: ...

def ihfft(
    a: ArrayLike,
    n: int | None = ...,
    axis: int = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def rfftn(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def irfftn(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[float64] | None = ...,
) -> NDArray[float64]: ...

def fft2(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def ifft2(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def rfft2(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[complex128] | None = ...,
) -> NDArray[complex128]: ...

def irfft2(
    a: ArrayLike,
    s: Sequence[int] | None = ...,
    axes: Sequence[int] | None = ...,
    norm: _NormKind = ...,
    out: NDArray[float64] | None = ...,
) -> NDArray[float64]: ...
