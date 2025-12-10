"""Typing tests for `_core._ufunc_config`."""

from collections.abc import Callable
from typing import Any, assert_type

from _typeshed import SupportsWrite

import numpy as np

def func(a: str, b: int) -> None: ...

class Write:
    def write(self, value: str) -> None: ...

assert_type(np.seterr(all=None), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(divide="ignore"), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(over="warn"), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(under="call"), np._core._ufunc_config._ErrDict)
assert_type(np.seterr(invalid="raise"), np._core._ufunc_config._ErrDict)
assert_type(np.geterr(), np._core._ufunc_config._ErrDict)

assert_type(np.setbufsize(4096), int)
assert_type(np.getbufsize(), int)

assert_type(np.seterrcall(func), Callable[[str, int], Any] | SupportsWrite[str] | None)
assert_type(np.seterrcall(Write()), Callable[[str, int], Any] | SupportsWrite[str] | None)
assert_type(np.geterrcall(), Callable[[str, int], Any] | SupportsWrite[str] | None)

assert_type(np.errstate(call=func, all="call"), np.errstate)
assert_type(np.errstate(call=Write(), divide="log", over="log"), np.errstate)
