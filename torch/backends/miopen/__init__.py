# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def set_flags(
    _immediate=None,
):
    orig_flags = (torch._C._get_miopen_immediate(),)
    if _immediate is not None:
        torch._C._set_miopen_immediate(_immediate)
    return orig_flags


@contextmanager
def flags(
    immediate=False,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            immediate,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<miopen|mkldnn>.immediate = True


class MiopenModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    immediate = ContextProp(
        torch._C._get_miopen_immediate, torch._C._set_miopen_immediate
    )


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MiopenModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
immediate: bool
