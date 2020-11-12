"""JIT C++ strings into executables."""
import os
import threading
from typing import List, Optional

import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
from torch.utils import cpp_extension


LOCK = threading.Lock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]

if hasattr(torch.__config__, "_cxx_flags"):
    CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
    if "-g" not in CXX_FLAGS:
        CXX_FLAGS.append("-g")
else:
    # FIXME: Remove when back testing is no longer required.
    CXX_FLAGS = ["-O2", "-fPIC", "-g"]

EXTRA_INCLUDE_PATHS: List[str] = [os.path.join(SOURCE_ROOT, "valgrind_wrapper")]
CONDA_PREFIX = os.getenv("CONDA_PREFIX")
if CONDA_PREFIX is not None:
    # load will automatically search /usr/include, but not conda include.
    EXTRA_INCLUDE_PATHS.append(os.path.join(CONDA_PREFIX, "include"))


COMPAT_CALLGRIND_BINDINGS: Optional[CallgrindModuleType] = None
def get_compat_bindings() -> CallgrindModuleType:
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(
                name="callgrind_bindings",
                sources=[os.path.join(
                    SOURCE_ROOT,
                    "valgrind_wrapper",
                    "compat_bindings.cpp"
                )],
                extra_cflags=CXX_FLAGS,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
            )
    return COMPAT_CALLGRIND_BINDINGS
