import sympy
import sys
from typing import Optional

import torch
from torch import SymInt, SymFloat, SymBool
from torch._dynamo import allow_in_graph
from torch._guards import TracingContext, detect_fake_mode
from torch.fx.experimental.symbolic_shapes import _constrain_symbol_range


def _wrap_concert_int_with_symbolic(symbol, min: Optional[int], max: int):
    assert isinstance(symbol, int)

    # Make sure whatever we constraining is valid in the specified range
    # min == None means user didn't specify any constraint on the minimum
    # range for this symbol.
    if min is None:
        if (symbol > max):
            raise ValueError(f"Invalid value {symbol} for range [{0}:{max}]")
    else:
        if not (min <= symbol <= max):
            raise ValueError(f"Invalid value {symbol} for range [{min}:{max}]")

    if (
        (fake_mode := detect_fake_mode()) is not None and
        getattr(fake_mode, "shape_env", None) is not None
    ):
        # If we are tracing with a fake mode then add this integer to the
        # shape_env's var_to_range
        sym_integer = sympy.Integer(symbol)
        shape_env = fake_mode.shape_env
        if min is None:
            _constrain_symbol_range_as_size(shape_env, sym_integer, max)
        else:
            _constrain_symbol_range(shape_env, sym_integer, min, max)

        shape_env.var_to_stack[sym_integer] = TracingContext(fake_mode).extract_stack()


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
# NOTE: We put allow_in_graph here so that Dynamo doesn't need to trace through here
@allow_in_graph
def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time. If called in eager mode,
    it will still check if the input value is within the specified range.
    """

    if min is None:
        min = 0
    if max is None:
        max = sys.maxsize

    if isinstance(symbol, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")

    if isinstance(symbol, int):
        # While we are tracing, if the symbol is resolved to static value
        # we still register it in the shape_env and later error out.
        _wrap_concert_int_with_symbolic(symbol, min, max)
        return

    assert isinstance(symbol, SymInt)
    torch.sym_constrain_range(symbol, min, max)


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
# NOTE: We put allow_in_graph here so that Dynamo doesn't need to trace through here
@allow_in_graph
def constrain_as_size(symbol, max: Optional[int] = None):
    """
    Add max constraint on the intermediate symbol which will be used as a size
    in another tensor. If called in eager mode, it will still check if the input
    value is within the specified range.
    """
    # NOTE: If min value is not passed, we will assume it means there is no
    # user specified min value provided. If this happens, we will assume min value to
    # be 2 for compiler only! Runtime will assume min value will be 0.

    if max is None:
        max = sys.maxsize

    if isinstance(symbol, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")

    if isinstance(symbol, int):
        # While we are tracing, if the symbol is resolved to static value
        # we still register it in the shape_env and later error out.
        _wrap_concert_int_with_symbolic(symbol, None, max)
        return

    assert isinstance(symbol, SymInt)
    return torch.sym_constrain_range(symbol, min=None, max=max)
