"""JIT C++ strings into executables.

1) Enable timeit and Callgrind collection for C++ snippets.
2) Allow Timer.collect_callgrind to be used on earlier versions of PyTorch
"""
import atexit
import os
import re
import shutil
import textwrap
import threading
from types import ModuleType
from typing import cast, List, Optional, Union
import uuid

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import torch
from torch.utils import cpp_extension


SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]
with open(os.path.join(SOURCE_ROOT, "timer_timeit_template.cpp"), "rt") as f:
    CXX_TIMEIT_TEMPLATE = f.read()


with open(os.path.join(SOURCE_ROOT, "timer_callgrind_template.cpp"), "rt") as f:
    CXX_CALLGRIND_TEMPLATE = f.read()


BUILD_ROOT = os.path.join(
    torch._appdirs.user_cache_dir(appname="benchmark_utils_jit"),
    f"build_{uuid.uuid4()}".replace("-", "")
)
os.makedirs(BUILD_ROOT, exist_ok=True)
WRITE_SOURCE_LOCK = threading.Lock()
atexit.register(shutil.rmtree, BUILD_ROOT)


CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
if "-g" not in CXX_FLAGS:
    CXX_FLAGS.append("-g")


# load will automatically search /usr/include, but not conda include.
extra_include_paths: List[str] = []
conda_prefix = os.getenv("CONDA_PREFIX")
if conda_prefix is not None:
    extra_include_paths = [os.path.join(conda_prefix, "include")]
extra_include_paths.append(SOURCE_ROOT)


# MyPy has no way of knowing the methods on a C module, so we must
# define API classes which mirror the PyBind11 methods.
class TimeitModuleType(Protocol):
    def timeit(self, number: int) -> float: ...


class CallgrindModuleType(Protocol):
    __file__: str
    __name__: str
    def _valgrind_supported_platform(self) -> bool: ...


_COMPAT_CALLGRIND_BINDINGS: Optional[CallgrindModuleType] = None
def get_compat_bindings() -> CallgrindModuleType:
    global _COMPAT_CALLGRIND_BINDINGS
    if _COMPAT_CALLGRIND_BINDINGS is None:
        _COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(
            name="callgrind_bindings",
            sources=os.path.join(SOURCE_ROOT, "compat_bindings.cpp"),
            extra_cflags=CXX_FLAGS,
            extra_include_paths=extra_include_paths,
        )
    return _COMPAT_CALLGRIND_BINDINGS


def compile_template(
    stmt: str,
    setup: str,
    template: str,
    is_standalone: bool
) -> Union[ModuleType, str]:
    src = template
    for before, after, indentation in (
        ("// SETUP_TEMPLATE_LOCATION", setup, 4),
        ("// STMT_TEMPLATE_LOCATION", stmt, 8)
    ):
        # C++ doesn't care about indentation so this code isn't load
        # bearing the way it is with Python, but this makes the source
        # look nicer if a human has to look at it.
        src = re.sub(
            before,
            textwrap.indent(after, " " * indentation)[indentation:],
            src
        )

    # We want to isolate different Timers. However `cpp_extension` will
    # cache builds which will significantly reduce the cost of repeated
    # invocations.
    with WRITE_SOURCE_LOCK:
        name = f"timer_cpp_{abs(hash(src))}"
        build_dir = os.path.join(BUILD_ROOT, name)
        os.makedirs(build_dir, exist_ok=True)

        src_path = os.path.join(build_dir, "timer_src.cpp")
        with open(src_path, "wt") as f:
            f.write(src)

    output = cpp_extension.load(
        name=name,
        sources=src_path,
        build_directory=build_dir,
        extra_cflags=CXX_FLAGS,
        extra_include_paths=extra_include_paths,
        is_python_module=not is_standalone,
        is_standalone=is_standalone,
    )

    return (
        # This is purely for MyPy and human comprehension.
        cast(str, output) if is_standalone else
        cast(ModuleType, output)
    )
