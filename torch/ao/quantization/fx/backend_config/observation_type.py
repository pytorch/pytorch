from enum import Enum

class ObservationType(Enum):
    # this means input and output are observed with different observers, based
    # on qconfig.activation
    # example: conv, linear, softmax
    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    # this means the output will use the same observer instance as input, based
    # on qconfig.activation
    # example: torch.cat, maxpool
    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
