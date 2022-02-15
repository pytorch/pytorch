from functools import partial
from torch import optim

Adam = partial(optim.Adam, foreach=True)
from .adamw import AdamW as AdamW
NAdam = partial(optim.NAdam, foreach=True)
SGD = partial(optim.SGD, foreach=True)
RAdam = partial(optim.RAdam, foreach=True)
from .rmsprop import RMSprop as RMSprop
from .rprop import Rprop as Rprop
ASGD = partial(optim.ASGD, foreach=True)
Adamax = partial(optim.Adamax, foreach=True)
Adadelta = partial(optim.Adadelta, foreach=True)
Adagrad = partial(optim.Adagrad, foreach=True)
