"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .lazy_adam import LazyAdam
from .optimizerw import AdamW
from .adamax import Adamax
from .asgd import ASGD
from .sgd import SGD
from .rprop import Rprop
from .rmsprop import RMSprop
from .optimizer import Optimizer
from .lbfgs import LBFGS
from . import lr_scheduler

# backward compatibility
SparseAdam = LazyAdam

del adadelta
del adagrad
del adam
del optimizerw
del adamax
del asgd
del sgd
del rprop
del rmsprop
del optimizer
del lbfgs
