# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def is_available():
    return torch._C._has_hipdnn


def set_flags(
    _enabled=None,
):
    orig_flags = (torch._C._get_hipdnn_enabled(),)
    if _enabled is not None:
        torch._C._set_hipdnn_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(
    enabled=None,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            enabled,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class HipdnnModule(PropModule):
    enabled = ContextProp(torch._C._get_hipdnn_enabled, torch._C._set_hipdnn_enabled)


sys.modules[__name__] = HipdnnModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
enabled: bool
