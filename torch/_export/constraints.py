from typing import Optional

import torch


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time. If called in eager mode,
    it will still check if the input value is within the specified range.
    """
    torch.sym_constrain_range(symbol, min=min, max=max)


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
def constrain_as_size(symbol):
    """
    Add compiler hints to the intermediate symbol which will be used as a size
    in another tensor.
    """
    # NOTE: If min, max value are not passed, we will assume it means this is only used for compiler hint.
    # Runtime will assume min value will be 0 and max value will be INT_MAX
    torch.sym_constrain_for_size(symbol)
