import ctypes
import sys
from pathlib import Path

import torch


if sys.platform == "win32":
    lib_files = list(Path(__file__).parent.glob("_C*.dll"))
    lib_extension = "dll"
else:
    lib_files = list(Path(__file__).parent.glob("_C*.so"))
    lib_extension = "so"

assert len(lib_files) == 1, (
    f"Expected one _C*.{lib_extension} file, found {len(lib_files)}"
)

# use ctypes.CDLL instead of load_library to be able to test the unload logic
# below code is reduced from the load_library code
with torch._ops.dl_open_guard():
    loaded_lib = ctypes.CDLL(lib_files[0])

from . import ops


__all__ = [
    "loaded_lib",
    "ops",
]
