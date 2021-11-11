import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def set_flags(_preferred=None):
    orig_flags = (torch._C._get_linalg_preferred_backend(),)
    if _preferred is not None:
        if not isinstance(_preferred, torch.linalg_backend):
            raise RuntimeError("must choose a linalg backend from: "
                               "torch.linalg_default, torch.linalg_cusolver, torch.linalg_magma.")
        torch._C._set_linalg_preferred_backend(_preferred)
    return orig_flags


@contextmanager
def flags(preferred=torch.linalg_default):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(preferred)
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

    preferred = ContextProp(torch._C._get_linalg_preferred_backend, torch._C._set_linalg_preferred_backend)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = LinalgModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
preferred: torch.linalg_backend
