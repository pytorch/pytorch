r"""Functional optimizer APIs."""

from collections.abc import Callable as _Callable
from functools import wraps as _wraps
from typing import ParamSpec as _ParamSpec, TypeVar as _TypeVar

from torch.optim._adafactor import adafactor as _adafactor
from torch.optim._functional import sparse_adam as _sparse_adam
from torch.optim._muon import muon as _muon
from torch.optim.adadelta import adadelta as _adadelta
from torch.optim.adagrad import adagrad as _adagrad
from torch.optim.adam import adam as _adam
from torch.optim.adamax import adamax as _adamax
from torch.optim.adamw import adamw as _adamw
from torch.optim.asgd import asgd as _asgd
from torch.optim.nadam import nadam as _nadam
from torch.optim.radam import radam as _radam
from torch.optim.rmsprop import rmsprop as _rmsprop
from torch.optim.rprop import rprop as _rprop
from torch.optim.sgd import sgd as _sgd


__all__ = [
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamw",
    "asgd",
    "muon",
    "nadam",
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
    "sparse_adam",
]

_P = _ParamSpec("_P")
_R = _TypeVar("_R")


# All this does is have a function, say, `adadelta`, redefined in this
# module as its own API while calling the exact same implementation in
# `torch.optim.adadelta`. This is done so that the functional API can
# be documented in this module without messing with the metadata of the
# original function `torch.optim.adadelta.adadelta`.
def _wrap_functional(function: _Callable[_P, _R]) -> _Callable[_P, _R]:
    @_wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return function(*args, **kwargs)

    wrapper.__module__ = __name__
    return wrapper


adadelta = _wrap_functional(_adadelta)
adafactor = _wrap_functional(_adafactor)
adagrad = _wrap_functional(_adagrad)
adam = _wrap_functional(_adam)
adamax = _wrap_functional(_adamax)
adamw = _wrap_functional(_adamw)
asgd = _wrap_functional(_asgd)
muon = _wrap_functional(_muon)
nadam = _wrap_functional(_nadam)
radam = _wrap_functional(_radam)
rmsprop = _wrap_functional(_rmsprop)
rprop = _wrap_functional(_rprop)
sgd = _wrap_functional(_sgd)
sparse_adam = _wrap_functional(_sparse_adam)

del _wrap_functional
