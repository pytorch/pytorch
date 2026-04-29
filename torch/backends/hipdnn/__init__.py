# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def is_available():
    return torch._C._has_hipdnn


@contextmanager
def flags(enabled=None):
    orig_enabled = torch._C._get_hipdnn_enabled()
    with __allow_nonbracketed_mutation():
        if enabled is not None:
            torch._C._set_hipdnn_enabled(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            torch._C._set_hipdnn_enabled(orig_enabled)


class HipdnnModule(PropModule):
    enabled = ContextProp(torch._C._get_hipdnn_enabled, torch._C._set_hipdnn_enabled)


sys.modules[__name__] = HipdnnModule(sys.modules[__name__], __name__)

# Type annotation for the replaced module.
enabled: bool
