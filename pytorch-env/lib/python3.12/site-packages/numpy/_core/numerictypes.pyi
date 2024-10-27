from typing import (
    Literal as L,
    Any,
    TypeVar,
    TypedDict,
)

import numpy as np
from numpy import (
    dtype,
    generic,
    ubyte,
    ushort,
    uintc,
    ulong,
    ulonglong,
    byte,
    short,
    intc,
    long,
    longlong,
    half,
    single,
    double,
    longdouble,
    csingle,
    cdouble,
    clongdouble,
    datetime64,
    timedelta64,
    object_,
    str_,
    bytes_,
    void,
)

from numpy._core._type_aliases import (
    sctypeDict as sctypeDict,
)

from numpy._typing import DTypeLike

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=generic)

class _TypeCodes(TypedDict):
    Character: L['c']
    Integer: L['bhilqnp']
    UnsignedInteger: L['BHILQNP']
    Float: L['efdg']
    Complex: L['FDG']
    AllInteger: L['bBhHiIlLqQnNpP']
    AllFloat: L['efdgFDG']
    Datetime: L['Mm']
    All: L['?bhilqnpBHILQNPefdgFDGSUVOMm']

__all__: list[str]

def isdtype(
    dtype: dtype[Any] | type[Any],
    kind: DTypeLike | tuple[DTypeLike, ...]
) -> bool: ...

def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool: ...

typecodes: _TypeCodes
ScalarType: tuple[
    type[int],
    type[float],
    type[complex],
    type[bool],
    type[bytes],
    type[str],
    type[memoryview],
    type[np.bool],
    type[csingle],
    type[cdouble],
    type[clongdouble],
    type[half],
    type[single],
    type[double],
    type[longdouble],
    type[byte],
    type[short],
    type[intc],
    type[long],
    type[longlong],
    type[timedelta64],
    type[datetime64],
    type[object_],
    type[bytes_],
    type[str_],
    type[ubyte],
    type[ushort],
    type[uintc],
    type[ulong],
    type[ulonglong],
    type[void],
]
