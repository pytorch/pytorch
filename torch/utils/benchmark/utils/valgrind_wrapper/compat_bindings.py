"""Allow Timer.collect_callgrind to be used on earlier versions of PyTorch

FIXME: Remove this module once we no longer need to back test.
"""
import os
import textwrap
from typing import List

from torch.utils.cpp_extension import load_inline


# load_inline will automatically search /usr/include, but not conda include.
extra_include_paths: List[str] = []
conda_prefix = os.getenv("CONDA_PREFIX")
if conda_prefix is not None:
    extra_include_paths = [os.path.join(conda_prefix, "include")]

bindings = load_inline(
    name="callgrind_bindings",
    cpp_sources=textwrap.dedent("""
    #include <valgrind/callgrind.h>

    bool _valgrind_supported_platform() {
        #if defined(NVALGRIND)
        return false;
        #else
        return true;
        #endif
    }

    void _valgrind_toggle() {
        #if defined(NVALGRIND)
        TORCH_CHECK(false, "Valgrind is not supported.");
        #else
        CALLGRIND_TOGGLE_COLLECT;
        #endif
    }
    """),
    extra_include_paths=extra_include_paths,
    functions=["_valgrind_supported_platform", "_valgrind_toggle"],
)
