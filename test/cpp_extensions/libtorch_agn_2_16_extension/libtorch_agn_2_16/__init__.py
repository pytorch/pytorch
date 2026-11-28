import sys
from pathlib import Path

import torch


so_files = list(
    Path(__file__).parent.glob("_C*" + (".pyd" if sys.platform == "win32" else ".so"))
)
if len(so_files) != 1:
    raise AssertionError(f"Expected one _C*.{{so,pyd}} file, found {len(so_files)}")

torch.ops.load_library(str(so_files[0]))

# _interop is a separate importable module exposing the Python interop helpers
# used by the interop tests.
from . import _interop, ops


__all__ = [
    "_interop",
    "ops",
]
