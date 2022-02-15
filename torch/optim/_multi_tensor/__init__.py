"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""
from functools import partial
from torch import optim

Adam = partial(optim.Adam, foreach=True)
from .adamw import AdamW
NAdam = partial(optim.NAdam, foreach=True)
SGD = partial(optim.SGD, foreach=True)
RAdam = partial(optim.RAdam, foreach=True)
RMSprop = partial(optim.RMSprop, foreach=True)
Rprop = partial(optim.Rprop, foreach=True)
ASGD = partial(optim.ASGD, foreach=True)
Adamax = partial(optim.Adamax, foreach=True)
Adadelta = partial(optim.Adadelta, foreach=True)
Adagrad = partial(optim.Adagrad, foreach=True)

del adamw
