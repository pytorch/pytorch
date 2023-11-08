"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.

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


Adam = partialclass(optim.Adam, foreach=True)
AdamW = partialclass(optim.AdamW, foreach=True)
NAdam = partialclass(optim.NAdam, foreach=True)
SGD = partialclass(optim.SGD, foreach=True)
RAdam = partialclass(optim.RAdam, foreach=True)
RMSprop = partialclass(optim.RMSprop, foreach=True)
Rprop = partialclass(optim.Rprop, foreach=True)
ASGD = partialclass(optim.ASGD, foreach=True)
Adamax = partialclass(optim.Adamax, foreach=True)
Adadelta = partialclass(optim.Adadelta, foreach=True)
Adagrad = partialclass(optim.Adagrad, foreach=True)
