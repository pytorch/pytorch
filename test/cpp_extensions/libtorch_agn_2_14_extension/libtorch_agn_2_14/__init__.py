import sys
from pathlib import Path

import torch


so_files = list(
    Path(__file__).parent.glob("_C*" + (".pyd" if sys.platform == "win32" else ".so"))
)
if len(so_files) != 1:
    raise AssertionError(f"Expected one _C*.{{so,pyd}} file, found {len(so_files)}")

# load_library registers the STABLE_TORCH_LIBRARY ops onto torch.ops (as in the
# other libtorch_agn extensions); importing _C additionally exposes its
# PyMethodDef helpers, used by the PyObject interop tests.
torch.ops.load_library(str(so_files[0]))
from . import _C  # noqa: F401
from . import ops


__all__ = [
    "_C",
    "ops",
]
