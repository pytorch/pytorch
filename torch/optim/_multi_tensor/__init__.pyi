from functools import partial
from torch import optim

from .adam import Adam as Adam
from .adamw import AdamW as AdamW
from .nadam import NAdam as NAdam
from .sgd import SGD as SGD
from .radam import RAdam as RAdam
from .rmsprop import RMSprop as RMSprop
from .rprop import Rprop as Rprop
from .asgd import ASGD as ASGD
Adamax = partial(optim.Adamax, foreach=True)
Adadelta = partial(optim.Adadelta, foreach=True)
Adagrad = partial(optim.Adagrad, foreach=True)
