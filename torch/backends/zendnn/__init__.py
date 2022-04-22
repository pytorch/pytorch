import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def is_available():
    r"""Returns whether PyTorch is built with ZENDNN support."""
    return torch._C.has_zendnn

def set_flags(_enabled):
    orig_flags = (torch._C._get_zendnn_enabled(),)
    torch._C._set_zendnn_enabled(_enabled)
    return orig_flags

@contextmanager
def flags(enabled=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])

class ZendnnModule(PropModule):
    def __init__(self, m, name):
        super(ZendnnModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_zendnn_enabled, torch._C._set_zendnn_enabled)

sys.modules[__name__] = ZendnnModule(sys.modules[__name__], __name__)
