# Owner(s): ["module: nvfuser"]

import torch
from torch.testing._internal.common_utils import set_default_dtype

try:
    from _nvfuser.test_torchscript import *  # noqa: F403,F401
except ImportError:
    def run_tests():
        return
    pass

if __name__ == '__main__':
    # TODO: Update nvfuser to work with float default dtype
    with set_default_dtype(torch.double):
        run_tests()
