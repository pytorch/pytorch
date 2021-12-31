from functools import partial
from torch import optim

from .adam import Adam as Adam
AdamW = partial(optim.AdamW, foreach=True)
from .nadam import NAdam as NAdam
from .sgd import SGD as SGD
from .radam import RAdam as RAdam
from .rmsprop import RMSprop as RMSprop
from .rprop import Rprop as Rprop
from .asgd import ASGD as ASGD
from .adamax import Adamax as Adamax
from .adadelta import Adadelta as Adadelta
from .adagrad import Adagrad as Adagrad
