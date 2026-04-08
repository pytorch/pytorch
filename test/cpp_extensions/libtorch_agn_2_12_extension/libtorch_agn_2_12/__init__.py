import ctypes
import sys
from pathlib import Path

import torch


so_files = list(
    Path(__file__).parent.glob("_C*" + (".pyd" if sys.platform == "win32" else ".so"))
)
if len(so_files) != 1:
    raise AssertionError(f"Expected one _C*.{{so,pyd}} file, found {len(so_files)}")

# use ctypes.CDLL instead of load_library to be able to test the unload logic
# below code is reduced from the load_library code
with torch._ops.dl_open_guard():
    loaded_lib = ctypes.CDLL(str(so_files[0]))

from . import ops


__all__ = [
    "loaded_lib",
    "ops",
]
