"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .adadelta import Adadelta
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD

del adam
del adamw
del sgd
del rmsprop
del rprop
del asgd
del adamax
del adadelta
