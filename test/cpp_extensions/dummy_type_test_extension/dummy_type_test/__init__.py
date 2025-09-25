import ctypes
from pathlib import Path

import torch


so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"

# use ctypes.CDLL instead of load_library to be able to test the unload logic
# below code is reduced from the load_library code
with torch._ops.dl_open_guard():
    loaded_lib = ctypes.CDLL(so_files[0])

# Import the C++ extension module for direct pybind access
import importlib.util


spec = importlib.util.spec_from_file_location("_C", so_files[0])
_C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_C)

from . import ops


__all__ = [
    "loaded_lib",
    "_C",
    "ops",
]
