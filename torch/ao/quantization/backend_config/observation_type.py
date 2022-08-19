from enum import Enum

__all__ = ['ObservationType']

class ObservationType(Enum):
    # this means input and output are observed with different observers, based
    # on qconfig.activation
    # example: conv, linear, softmax
    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    # this means the output will use the same observer instance as input, based
    # on qconfig.activation
    # example: torch.cat, maxpool
    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    # this means the output is never observed
    # example: x.shape, x.size
    OUTPUT_NOT_OBSERVED = 2
