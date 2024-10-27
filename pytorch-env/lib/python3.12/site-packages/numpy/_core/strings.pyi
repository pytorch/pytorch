from typing import Any, overload

import numpy as np
from numpy._typing import (
    NDArray,
    _ArrayLikeStr_co as U_co,
    _ArrayLikeBytes_co as S_co,
    _ArrayLikeInt_co as i_co,
    _ArrayLikeBool_co as b_co,
)

@overload
def equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def not_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def not_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def greater_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def less_equal(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less_equal(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def greater(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def greater(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def less(x1: U_co, x2: U_co) -> NDArray[np.bool]: ...
@overload
def less(x1: S_co, x2: S_co) -> NDArray[np.bool]: ...

@overload
def add(x1: U_co, x2: U_co) -> NDArray[np.str_]: ...
@overload
def add(x1: S_co, x2: S_co) -> NDArray[np.bytes_]: ...

@overload
def multiply(a: U_co, i: i_co) -> NDArray[np.str_]: ...
@overload
def multiply(a: S_co, i: i_co) -> NDArray[np.bytes_]: ...

@overload
def mod(a: U_co, value: Any) -> NDArray[np.str_]: ...
@overload
def mod(a: S_co, value: Any) -> NDArray[np.bytes_]: ...

def isalpha(x: U_co | S_co) -> NDArray[np.bool]: ...
def isalnum(a: U_co | S_co) -> NDArray[np.bool]: ...
def isdigit(x: U_co | S_co) -> NDArray[np.bool]: ...
def isspace(x: U_co | S_co) -> NDArray[np.bool]: ...
def isdecimal(x: U_co) -> NDArray[np.bool]: ...
def isnumeric(x: U_co) -> NDArray[np.bool]: ...
def islower(a: U_co | S_co) -> NDArray[np.bool]: ...
def istitle(a: U_co | S_co) -> NDArray[np.bool]: ...
def isupper(a: U_co | S_co) -> NDArray[np.bool]: ...

def str_len(x: U_co | S_co) -> NDArray[np.int_]: ...

@overload
def find(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def find(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def rfind(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def rfind(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def index(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...
@overload
def index(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...

@overload
def rindex(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...
@overload
def rindex(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: None | i_co = ...,
) -> NDArray[np.int_]: ...

@overload
def count(
    a: U_co,
    sub: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...
@overload
def count(
    a: S_co,
    sub: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.int_]: ...

@overload
def startswith(
    a: U_co,
    prefix: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...
@overload
def startswith(
    a: S_co,
    prefix: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...

@overload
def endswith(
    a: U_co,
    suffix: U_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...
@overload
def endswith(
    a: S_co,
    suffix: S_co,
    start: i_co = ...,
    end: i_co | None = ...,
) -> NDArray[np.bool]: ...

def decode(
    a: S_co,
    encoding: None | str = ...,
    errors: None | str = ...,
) -> NDArray[np.str_]: ...

def encode(
    a: U_co,
    encoding: None | str = ...,
    errors: None | str = ...,
) -> NDArray[np.bytes_]: ...

@overload
def expandtabs(a: U_co, tabsize: i_co = ...) -> NDArray[np.str_]: ...
@overload
def expandtabs(a: S_co, tabsize: i_co = ...) -> NDArray[np.bytes_]: ...

@overload
def center(a: U_co, width: i_co, fillchar: U_co = ...) -> NDArray[np.str_]: ...
@overload
def center(a: S_co, width: i_co, fillchar: S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def ljust(a: U_co, width: i_co, fillchar: U_co = ...) -> NDArray[np.str_]: ...
@overload
def ljust(a: S_co, width: i_co, fillchar: S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def rjust(
    a: U_co,
    width: i_co,
    fillchar: U_co = ...,
) -> NDArray[np.str_]: ...
@overload
def rjust(
    a: S_co,
    width: i_co,
    fillchar: S_co = ...,
) -> NDArray[np.bytes_]: ...

@overload
def lstrip(a: U_co, chars: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def lstrip(a: S_co, chars: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def rstrip(a: U_co, char: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def rstrip(a: S_co, char: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def strip(a: U_co, chars: None | U_co = ...) -> NDArray[np.str_]: ...
@overload
def strip(a: S_co, chars: None | S_co = ...) -> NDArray[np.bytes_]: ...

@overload
def zfill(a: U_co, width: i_co) -> NDArray[np.str_]: ...
@overload
def zfill(a: S_co, width: i_co) -> NDArray[np.bytes_]: ...

@overload
def upper(a: U_co) -> NDArray[np.str_]: ...
@overload
def upper(a: S_co) -> NDArray[np.bytes_]: ...

@overload
def lower(a: U_co) -> NDArray[np.str_]: ...
@overload
def lower(a: S_co) -> NDArray[np.bytes_]: ...

@overload
def swapcase(a: U_co) -> NDArray[np.str_]: ...
@overload
def swapcase(a: S_co) -> NDArray[np.bytes_]: ...

@overload
def capitalize(a: U_co) -> NDArray[np.str_]: ...
@overload
def capitalize(a: S_co) -> NDArray[np.bytes_]: ...

@overload
def title(a: U_co) -> NDArray[np.str_]: ...
@overload
def title(a: S_co) -> NDArray[np.bytes_]: ...

@overload
def replace(
    a: U_co,
    old: U_co,
    new: U_co,
    count: i_co = ...,
) -> NDArray[np.str_]: ...
@overload
def replace(
    a: S_co,
    old: S_co,
    new: S_co,
    count: i_co = ...,
) -> NDArray[np.bytes_]: ...

@overload
def join(sep: U_co, seq: U_co) -> NDArray[np.str_]: ...
@overload
def join(sep: S_co, seq: S_co) -> NDArray[np.bytes_]: ...

@overload
def split(
    a: U_co,
    sep: None | U_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...
@overload
def split(
    a: S_co,
    sep: None | S_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...

@overload
def rsplit(
    a: U_co,
    sep: None | U_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...
@overload
def rsplit(
    a: S_co,
    sep: None | S_co = ...,
    maxsplit: None | i_co = ...,
) -> NDArray[np.object_]: ...

@overload
def splitlines(a: U_co, keepends: None | b_co = ...) -> NDArray[np.object_]: ...
@overload
def splitlines(a: S_co, keepends: None | b_co = ...) -> NDArray[np.object_]: ...

@overload
def partition(a: U_co, sep: U_co) -> NDArray[np.str_]: ...
@overload
def partition(a: S_co, sep: S_co) -> NDArray[np.bytes_]: ...

@overload
def rpartition(a: U_co, sep: U_co) -> NDArray[np.str_]: ...
@overload
def rpartition(a: S_co, sep: S_co) -> NDArray[np.bytes_]: ...

@overload
def translate(
    a: U_co,
    table: U_co,
    deletechars: None | U_co = ...,
) -> NDArray[np.str_]: ...
@overload
def translate(
    a: S_co,
    table: S_co,
    deletechars: None | S_co = ...,
) -> NDArray[np.bytes_]: ...
