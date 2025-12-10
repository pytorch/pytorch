from collections.abc import Container, Iterable
from typing import Any, Protocol, TypeAlias, overload, type_check_only
from typing import Literal as L

from _typeshed import Incomplete
from typing_extensions import TypeVar

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _16Bit,
    _32Bit,
    _64Bit,
    _ArrayLike,
    _NestedSequence,
    _ScalarLike_co,
    _SupportsArray,
)

__all__ = [
    "common_type",
    "imag",
    "iscomplex",
    "iscomplexobj",
    "isreal",
    "isrealobj",
    "mintypecode",
    "nan_to_num",
    "real",
    "real_if_close",
    "typename",
]

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, covariant=True)
_RealT = TypeVar("_RealT", bound=np.floating | np.integer | np.bool)

_FloatMax32: TypeAlias = np.float32 | np.float16
_ComplexMax128: TypeAlias = np.complex128 | np.complex64
_RealMax64: TypeAlias = np.float64 | np.float32 | np.float16 | np.integer
_Real: TypeAlias = np.floating | np.integer
_InexactMax32: TypeAlias = np.inexact[_32Bit] | np.float16
_NumberMax64: TypeAlias = np.number[_64Bit] | np.number[_32Bit] | np.number[_16Bit] | np.integer

@type_check_only
class _HasReal(Protocol[_T_co]):
    @property
    def real(self, /) -> _T_co: ...

@type_check_only
class _HasImag(Protocol[_T_co]):
    @property
    def imag(self, /) -> _T_co: ...

@type_check_only
class _HasDType(Protocol[_ScalarT_co]):
    @property
    def dtype(self, /) -> np.dtype[_ScalarT_co]: ...

###

def mintypecode(typechars: Iterable[str | ArrayLike], typeset: str | Container[str] = "GDFgdf", default: str = "d") -> str: ...

#
@overload
def real(val: _HasReal[_T]) -> _T: ...  # type: ignore[overload-overlap]
@overload
def real(val: _ArrayLike[_RealT]) -> NDArray[_RealT]: ...
@overload
def real(val: ArrayLike) -> NDArray[Any]: ...

#
@overload
def imag(val: _HasImag[_T]) -> _T: ...  # type: ignore[overload-overlap]
@overload
def imag(val: _ArrayLike[_RealT]) -> NDArray[_RealT]: ...
@overload
def imag(val: ArrayLike) -> NDArray[Any]: ...

#
@overload
def iscomplex(x: _ScalarLike_co) -> np.bool: ...
@overload
def iscomplex(x: NDArray[Any] | _NestedSequence[ArrayLike]) -> NDArray[np.bool]: ...
@overload
def iscomplex(x: ArrayLike) -> np.bool | NDArray[np.bool]: ...

#
@overload
def isreal(x: _ScalarLike_co) -> np.bool: ...
@overload
def isreal(x: NDArray[Any] | _NestedSequence[ArrayLike]) -> NDArray[np.bool]: ...
@overload
def isreal(x: ArrayLike) -> np.bool | NDArray[np.bool]: ...

#
def iscomplexobj(x: _HasDType[Any] | ArrayLike) -> bool: ...
def isrealobj(x: _HasDType[Any] | ArrayLike) -> bool: ...

#
@overload
def nan_to_num(
    x: _ScalarT,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> _ScalarT: ...
@overload
def nan_to_num(
    x: NDArray[_ScalarT] | _NestedSequence[_ArrayLike[_ScalarT]],
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def nan_to_num(
    x: _SupportsArray[np.dtype[_ScalarT]],
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> _ScalarT | NDArray[_ScalarT]: ...
@overload
def nan_to_num(
    x: _NestedSequence[ArrayLike],
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> NDArray[Incomplete]: ...
@overload
def nan_to_num(
    x: ArrayLike,
    copy: bool = True,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> Incomplete: ...

# NOTE: The [overload-overlap] mypy error is a false positive
@overload
def real_if_close(a: _ArrayLike[np.complex64], tol: float = 100) -> NDArray[np.float32 | np.complex64]: ...  # type: ignore[overload-overlap]
@overload
def real_if_close(a: _ArrayLike[np.complex128], tol: float = 100) -> NDArray[np.float64 | np.complex128]: ...
@overload
def real_if_close(a: _ArrayLike[np.clongdouble], tol: float = 100) -> NDArray[np.longdouble | np.clongdouble]: ...
@overload
def real_if_close(a: _ArrayLike[_RealT], tol: float = 100) -> NDArray[_RealT]: ...
@overload
def real_if_close(a: ArrayLike, tol: float = 100) -> NDArray[Any]: ...

#
@overload
def typename(char: L['S1']) -> L['character']: ...
@overload
def typename(char: L['?']) -> L['bool']: ...
@overload
def typename(char: L['b']) -> L['signed char']: ...
@overload
def typename(char: L['B']) -> L['unsigned char']: ...
@overload
def typename(char: L['h']) -> L['short']: ...
@overload
def typename(char: L['H']) -> L['unsigned short']: ...
@overload
def typename(char: L['i']) -> L['integer']: ...
@overload
def typename(char: L['I']) -> L['unsigned integer']: ...
@overload
def typename(char: L['l']) -> L['long integer']: ...
@overload
def typename(char: L['L']) -> L['unsigned long integer']: ...
@overload
def typename(char: L['q']) -> L['long long integer']: ...
@overload
def typename(char: L['Q']) -> L['unsigned long long integer']: ...
@overload
def typename(char: L['f']) -> L['single precision']: ...
@overload
def typename(char: L['d']) -> L['double precision']: ...
@overload
def typename(char: L['g']) -> L['long precision']: ...
@overload
def typename(char: L['F']) -> L['complex single precision']: ...
@overload
def typename(char: L['D']) -> L['complex double precision']: ...
@overload
def typename(char: L['G']) -> L['complex long double precision']: ...
@overload
def typename(char: L['S']) -> L['string']: ...
@overload
def typename(char: L['U']) -> L['unicode']: ...
@overload
def typename(char: L['V']) -> L['void']: ...
@overload
def typename(char: L['O']) -> L['object']: ...

# NOTE: The [overload-overlap] mypy errors are false positives
@overload
def common_type() -> type[np.float16]: ...
@overload
def common_type(a0: _HasDType[np.float16], /, *ai: _HasDType[np.float16]) -> type[np.float16]: ...  # type: ignore[overload-overlap]
@overload
def common_type(a0: _HasDType[np.float32], /, *ai: _HasDType[_FloatMax32]) -> type[np.float32]: ...  # type: ignore[overload-overlap]
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[np.float64 | np.integer],
    /,
    *ai: _HasDType[_RealMax64],
) -> type[np.float64]: ...
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[np.longdouble],
    /,
    *ai: _HasDType[_Real],
) -> type[np.longdouble]: ...
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[np.complex64],
    /,
    *ai: _HasDType[_InexactMax32],
) -> type[np.complex64]: ...
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[np.complex128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[np.clongdouble],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[_FloatMax32],
    array1: _HasDType[np.float32],
    /,
    *ai: _HasDType[_FloatMax32],
) -> type[np.float32]: ...
@overload
def common_type(
    a0: _HasDType[_RealMax64],
    array1: _HasDType[np.float64 | np.integer],
    /,
    *ai: _HasDType[_RealMax64],
) -> type[np.float64]: ...
@overload
def common_type(
    a0: _HasDType[_Real],
    array1: _HasDType[np.longdouble],
    /,
    *ai: _HasDType[_Real],
) -> type[np.longdouble]: ...
@overload
def common_type(  # type: ignore[overload-overlap]
    a0: _HasDType[_InexactMax32],
    array1: _HasDType[np.complex64],
    /,
    *ai: _HasDType[_InexactMax32],
) -> type[np.complex64]: ...
@overload
def common_type(
    a0: _HasDType[np.float64],
    array1: _HasDType[_ComplexMax128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_ComplexMax128],
    array1: _HasDType[np.float64],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_NumberMax64],
    array1: _HasDType[np.complex128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_ComplexMax128],
    array1: _HasDType[np.complex128 | np.integer],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[np.complex128 | np.integer],
    array1: _HasDType[_ComplexMax128],
    /,
    *ai: _HasDType[_NumberMax64],
) -> type[np.complex128]: ...
@overload
def common_type(
    a0: _HasDType[_Real],
    /,
    *ai: _HasDType[_Real],
) -> type[np.floating]: ...
@overload
def common_type(
    a0: _HasDType[np.number],
    array1: _HasDType[np.clongdouble],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.longdouble],
    array1: _HasDType[np.complexfloating],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.complexfloating],
    array1: _HasDType[np.longdouble],
    /,
    *ai: _HasDType[np.number],
) -> type[np.clongdouble]: ...
@overload
def common_type(
    a0: _HasDType[np.complexfloating],
    array1: _HasDType[np.number],
    /,
    *ai: _HasDType[np.number],
) -> type[np.complexfloating]: ...
@overload
def common_type(
    a0: _HasDType[np.number],
    array1: _HasDType[np.complexfloating],
    /,
    *ai: _HasDType[np.number],
) -> type[np.complexfloating]: ...
@overload
def common_type(
    a0: _HasDType[np.number],
    array1: _HasDType[np.number],
    /,
    *ai: _HasDType[np.number],
) -> type[Any]: ...
