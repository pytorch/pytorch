"""
:mod:`torch.optim._fused` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""
from functools import partialmethod
from torch import optim

def partialclass(cls, *args, **kwargs):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


Adam = partialclass(optim.Adam, fused=True)
