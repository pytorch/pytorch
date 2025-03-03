import sys

import torch.utils._cxx_pytree


sys.modules[__name__] = torch.utils._cxx_pytree

del sys, torch
