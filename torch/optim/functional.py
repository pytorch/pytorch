r"""Functional optimizer APIs."""

from torch.optim._adafactor import adafactor
from torch.optim._muon import muon
from torch.optim.adadelta import adadelta
from torch.optim.adagrad import adagrad
from torch.optim.adam import adam
from torch.optim.adamax import adamax
from torch.optim.adamw import adamw
from torch.optim.asgd import asgd
from torch.optim.nadam import nadam
from torch.optim.radam import radam
from torch.optim.rmsprop import rmsprop
from torch.optim.rprop import rprop
from torch.optim.sgd import sgd
from torch.optim.sparse_adam import sparse_adam


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

for _functional_api in (
    adadelta,
    adafactor,
    adagrad,
    adam,
    adamax,
    adamw,
    asgd,
    muon,
    nadam,
    radam,
    rmsprop,
    rprop,
    sgd,
    sparse_adam,
):
    _functional_api.__module__ = __name__

del _functional_api
