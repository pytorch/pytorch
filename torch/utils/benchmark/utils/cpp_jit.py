"""JIT C++ strings into executables."""
import os
import threading
from typing import List, Optional

import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
from torch.utils import cpp_extension


LOCK = threading.Lock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]

# BACK_TESTING_NOTE:
#   There are two workflows where this code could be used. One is the obvious
#   case where someone simply builds or installs PyTorch and uses Timer.
#   The other is that the entire `torch/utils/benchmark` folder from a CURRENT
#   PyTorch checkout is copy-pasted into a much OLDER version of the PyTorch
#   source code. This is what we refer to here as "back testing". The rationale
#   is that we might want to use current tooling to study some aspect of an
#   earlier version of PyTorch. (e.g. a regression.)
#
#   The problem is that Timer relies on several aspects of core PyTorch, namely
#   some binding functions for Valgrind symbols in `torch._C` and the
#   `torch.__config__._cxx_flags()` method. If we were to naively copy code
#   around this wouldn't work as the symbols of interest aren't present in
#   earlier versions of PyTorch. In order to work around this, we must add back
#   testing shims. These shims will never activate during normal use, but will
#   allow Timer to function outside of the "correct" version of PyTorch by
#   emulating functionality that was added later.
#
#   These shims are temporary, and as Timer becomes more integrated with
#   PyTorch the cost and complexity of such shims will increase. Once back
#   testing is no longer required (which is to say we have done enough historic
#   analysis and the shims no longer justify their maintenance and code
#   complexity costs) back testing paths will be removed.

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
    # Load will automatically search /usr/include, but not conda include.
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
