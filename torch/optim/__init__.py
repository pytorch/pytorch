"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from . import lr_scheduler, swa_utils
from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .lbfgs import LBFGS
from .optimizer import Optimizer
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD
from .sparse_adam import SparseAdam

del adadelta
del adagrad
del adam
del adamw
del sparse_adam
del adamax
del asgd
del sgd
del rprop
del rmsprop
del optimizer
del lbfgs
