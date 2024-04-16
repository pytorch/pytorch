"""
:mod:`torch.optim` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can also be easily integrated in the
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
from .nadam import NAdam
from .optimizer import Optimizer
from .radam import RAdam
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD
from .sparse_adam import SparseAdam

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
