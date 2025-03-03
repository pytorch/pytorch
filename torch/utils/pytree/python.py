import sys

import torch.utils._pytree


sys.modules[__name__] = torch.utils._pytree

del sys, torch
