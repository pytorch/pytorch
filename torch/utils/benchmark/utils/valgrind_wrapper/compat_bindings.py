"""Allow Timer.collect_callgrind to be used on earlier versions of PyTorch

FIXME: Remove this module once we no longer need to back test.
"""
import os
import textwrap
from typing import List

from torch.utils.cpp_extension import load_inline

raise NotImplementedError(
    "Compat bindings should not be needed for a stable release. If you "
    "did not monkey patch benchmark utils into an earlier version of "
    "PyTorch, then this is a bug and should be reported to the PyTorch team."
)
