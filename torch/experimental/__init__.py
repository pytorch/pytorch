import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

# Write:
#
#   torch.experimental.deterministic = True
#
# to globally enforce deterministic algorithms

def set_flags(_deterministic, _deterministic_error_level):
    global deterministic, deterministic_error_level
    orig_flags = (torch._C._get_deterministic(),
                  torch._C._get_deterministic_error_level())
    torch._C._set_deterministic(_deterministic)
    torch._C._set_deterministic_error_level(_deterministic_error_level)
    return orig_flags


@contextmanager
def flags(deterministic=False, deterministic_error_level=2):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(deterministic, deterministic_error_level)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0], orig_flags[1])


class ExperimentalModule(PropModule):
    def __init__(self, m, name):
        super(ExperimentalModule, self).__init__(m, name)

    deterministic = ContextProp(torch._C._get_deterministic, torch._C._set_deterministic)
    deterministic_error_level = ContextProp(torch._C._get_deterministic_error_level, torch._C._set_deterministic_error_level)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = ExperimentalModule(sys.modules[__name__], __name__)
