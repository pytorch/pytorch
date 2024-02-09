"""
:mod:`torch.optim` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can also be easily integrated in the
future.
"""

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamw import AdamW
from .sparse_adam import SparseAdam
from .adamax import Adamax
from .asgd import ASGD
from .sgd import SGD
from .radam import RAdam
from .rprop import Rprop
from .rmsprop import RMSprop
from .optimizer import Optimizer
from .nadam import NAdam
from .lbfgs import LBFGS
from . import lr_scheduler
from . import swa_utils

del adadelta  # noqa: F821
del adagrad  # noqa: F821
del adam  # noqa: F821
del adamw  # noqa: F821
del sparse_adam  # noqa: F821
del adamax  # noqa: F821
del asgd  # noqa: F821
del sgd  # noqa: F821
del radam  # noqa: F821
del rprop  # noqa: F821
del rmsprop  # noqa: F821
del optimizer  # noqa: F821
del nadam  # noqa: F821
del lbfgs  # noqa: F821
