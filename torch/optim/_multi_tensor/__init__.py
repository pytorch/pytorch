"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""
from functools import partialmethod

from .. import (
    Adadelta as _Adadelta,
    Adagrad as _Adagrad,
    Adam as _Adam,
    Adamax as _Adamax,
    AdamW as _AdamW,
    ASGD as _ASGD,
    NAdam as _NAdam,
    RAdam as _RAdam,
    RMSprop as _RMSprop,
    Rprop as _Rprop,
    SGD as _SGD,
)


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


Adam = partialclass(_Adam, foreach=True)
AdamW = partialclass(_AdamW, foreach=True)
NAdam = partialclass(_NAdam, foreach=True)
SGD = partialclass(_SGD, foreach=True)
RAdam = partialclass(_RAdam, foreach=True)
RMSprop = partialclass(_RMSprop, foreach=True)
Rprop = partialclass(_Rprop, foreach=True)
ASGD = partialclass(_ASGD, foreach=True)
Adamax = partialclass(_Adamax, foreach=True)
Adadelta = partialclass(_Adadelta, foreach=True)
Adagrad = partialclass(_Adagrad, foreach=True)


del _Adam
del _AdamW
del _NAdam
del _SGD
del _RAdam
del _RMSprop
del _Rprop
del _ASGD
del _Adamax
del _Adadelta
del _Adagrad
