import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def set_flags(_cuda_prefer_cusolver=None):
    orig_flags = (torch._C._get_linalg_cuda_prefer_cusolver(),)
    if _cuda_prefer_cusolver is not None:
        torch._C._set_linalg_cuda_prefer_cusolver(_cuda_prefer_cusolver)
    return orig_flags


@contextmanager
def flags(cuda_prefer_cusolver=True):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(cuda_prefer_cusolver)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<cudnn|mkldnn>.enabled = True

class LinalgModule(PropModule):
    def __init__(self, m, name):
        super(LinalgModule, self).__init__(m, name)

    cuda_prefer_cusolver = ContextProp(torch._C._get_linalg_cuda_prefer_cusolver, torch._C._set_linalg_cuda_prefer_cusolver)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = LinalgModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
cuda_prefer_cusolver: bool
