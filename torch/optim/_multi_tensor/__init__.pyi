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
from .adamax import Adamax as Adamax
Adadelta = partial(optim.Adadelta, foreach=True)
from .adagrad import Adagrad as Adagrad
