import os
import sys
import warnings
from contextlib import contextmanager
from typing import Optional

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule



def set_flags(
    _deterministic=None,
):
    orig_flags = (torch._C._get_onednn_deterministic())
    if _deterministic is not None:
        torch._C._set_onednn_deterministic(_deterministic)
    return orig_flags


@contextmanager
def flags(
    deterministic=False,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(deterministic)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class OnednnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    deterministic = ContextProp(
        torch._C._get_onednn_deterministic, torch._C._set_onednn_deterministic
    )


sys.modules[__name__] = OnednnModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
deterministic: bool
