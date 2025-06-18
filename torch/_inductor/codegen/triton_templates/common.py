import inspect
from typing import Any, Callable

import sympy

from ....utils._sympy.functions import CeilDiv
from ...utils import ceildiv


class SymbolicGridFn:
    """
    Wrapper around a grid function that allows either int or sympy inputs.

        @SymbolicGridFn
        def grid(x, meta, *, cdiv):
            return cdiv(x, meta["BLOCK_X"])
    """

    def __init__(self, fn: Callable[..., tuple[Any, Any, Any]]):
        self.fn = fn
        self.kwargs_int = {}
        self.kwargs_sym = {}
        params = inspect.signature(fn).parameters
        for name, fn_sym, fn_int in [
            ("cdiv", CeilDiv, ceildiv),
            ("min", sympy.Min, min),
            ("max", sympy.Max, max),
        ]:
            if name in params:
                self.kwargs_int[name] = fn_int
                self.kwargs_sym[name] = fn_sym

    def __call__(self, *args, **kwargs) -> tuple[int, int, int]:
        return self.fn(*args, **kwargs, **self.kwargs_int)

    def sympy_call(self, *args, **kwargs):
        return self.fn(*args, **kwargs, **self.kwargs_sym)
