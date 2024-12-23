from collections.abc import Callable
from typing import Any, Literal, TypedDict, SupportsIndex

# Using a private class is by no means ideal, but it is simply a consequence
# of a `contextlib.context` returning an instance of aforementioned class
from contextlib import _GeneratorContextManager

import numpy as np
from numpy import (
    integer,
    timedelta64,
    datetime64,
    floating,
    complexfloating,
    void,
    longdouble,
    clongdouble,
)
from numpy._typing import NDArray, _CharLike_co, _FloatLike_co

_FloatMode = Literal["fixed", "unique", "maxprec", "maxprec_equal"]

class _FormatDict(TypedDict, total=False):
    bool: Callable[[np.bool], str]
    int: Callable[[integer[Any]], str]
    timedelta: Callable[[timedelta64], str]
    datetime: Callable[[datetime64], str]
    float: Callable[[floating[Any]], str]
    longfloat: Callable[[longdouble], str]
    complexfloat: Callable[[complexfloating[Any, Any]], str]
    longcomplexfloat: Callable[[clongdouble], str]
    void: Callable[[void], str]
    numpystr: Callable[[_CharLike_co], str]
    object: Callable[[object], str]
    all: Callable[[object], str]
    int_kind: Callable[[integer[Any]], str]
    float_kind: Callable[[floating[Any]], str]
    complex_kind: Callable[[complexfloating[Any, Any]], str]
    str_kind: Callable[[_CharLike_co], str]

class _FormatOptions(TypedDict):
    precision: int
    threshold: int
    edgeitems: int
    linewidth: int
    suppress: bool
    nanstr: str
    infstr: str
    formatter: None | _FormatDict
    sign: Literal["-", "+", " "]
    floatmode: _FloatMode
    legacy: Literal[False, "1.13", "1.21"]

def set_printoptions(
    precision: None | SupportsIndex = ...,
    threshold: None | int = ...,
    edgeitems: None | int = ...,
    linewidth: None | int = ...,
    suppress: None | bool = ...,
    nanstr: None | str = ...,
    infstr: None | str = ...,
    formatter: None | _FormatDict = ...,
    sign: Literal[None, "-", "+", " "] = ...,
    floatmode: None | _FloatMode = ...,
    *,
    legacy: Literal[None, False, "1.13", "1.21"] = ...,
    override_repr: None | Callable[[NDArray[Any]], str] = ...,
) -> None: ...
def get_printoptions() -> _FormatOptions: ...
def array2string(
    a: NDArray[Any],
    max_line_width: None | int = ...,
    precision: None | SupportsIndex = ...,
    suppress_small: None | bool = ...,
    separator: str = ...,
    prefix: str = ...,
    # NOTE: With the `style` argument being deprecated,
    # all arguments between `formatter` and `suffix` are de facto
    # keyworld-only arguments
    *,
    formatter: None | _FormatDict = ...,
    threshold: None | int = ...,
    edgeitems: None | int = ...,
    sign: Literal[None, "-", "+", " "] = ...,
    floatmode: None | _FloatMode = ...,
    suffix: str = ...,
    legacy: Literal[None, False, "1.13", "1.21"] = ...,
) -> str: ...
def format_float_scientific(
    x: _FloatLike_co,
    precision: None | int = ...,
    unique: bool = ...,
    trim: Literal["k", ".", "0", "-"] = ...,
    sign: bool = ...,
    pad_left: None | int = ...,
    exp_digits: None | int = ...,
    min_digits: None | int = ...,
) -> str: ...
def format_float_positional(
    x: _FloatLike_co,
    precision: None | int = ...,
    unique: bool = ...,
    fractional: bool = ...,
    trim: Literal["k", ".", "0", "-"] = ...,
    sign: bool = ...,
    pad_left: None | int = ...,
    pad_right: None | int = ...,
    min_digits: None | int = ...,
) -> str: ...
def array_repr(
    arr: NDArray[Any],
    max_line_width: None | int = ...,
    precision: None | SupportsIndex = ...,
    suppress_small: None | bool = ...,
) -> str: ...
def array_str(
    a: NDArray[Any],
    max_line_width: None | int = ...,
    precision: None | SupportsIndex = ...,
    suppress_small: None | bool = ...,
) -> str: ...
def printoptions(
    precision: None | SupportsIndex = ...,
    threshold: None | int = ...,
    edgeitems: None | int = ...,
    linewidth: None | int = ...,
    suppress: None | bool = ...,
    nanstr: None | str = ...,
    infstr: None | str = ...,
    formatter: None | _FormatDict = ...,
    sign: Literal[None, "-", "+", " "] = ...,
    floatmode: None | _FloatMode = ...,
    *,
    legacy: Literal[None, False, "1.13", "1.21"] = ...
) -> _GeneratorContextManager[_FormatOptions]: ...
