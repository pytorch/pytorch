import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def is_available():
    r"""Returns whether PyTorch is built with MLC support."""
    return torch._C.has_mlc

def set_flags(_enabled):
    orig_flags = (torch._C._get_mlc_enabled(),)
    torch._C._set_mlc_enabled(_enabled)
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

class MLCModule(PropModule):
    def __init__(self, m, name):
        super(MLCModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_mlc_enabled, torch._C._set_mlc_enabled)

# Cool stuff from torch/backends/cudnn/__init__.py and
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MLCModule(sys.modules[__name__], __name__)
import torch.mlc.optim
