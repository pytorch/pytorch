"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .adam import Adam
from .adamw import AdamW
from .sgd import SGD
from .rmsprop import RMSprop

del adam
del adamw
del sgd
del rmsprop
