"""JIT C++ strings into executables."""
import atexit
import os
import re
import shutil
import tempfile
import textwrap
import threading
import uuid
from typing import Any, List, Optional

import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.historic.back_testing import IS_BACK_TESTING, CXX_FLAGS
from torch.utils.benchmark.utils.historic.cpp_jit import compat_jit

if not IS_BACK_TESTING:
    from torch.utils import cpp_extension


LOCK = threading.RLock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]

# We calculate uuid once at import time so that separate processes will have
# separate build roots, but threads will share the same build root.
# `cpp_extension` uses build root as part of the cache key, so per-invocation
# uuid's (e.g. different build root per _compile_template call) would lead to
# a 0% cache hit rate and spurious recompilation. Consider the following:
#   ```
#   setup = "auto x = torch::ones({1024, 1024});"
#   stmt = "torch::mm(x, x);"
#   for num_threads in [1, 2, 4, 8]:
#     print(Timer(stmt, setup, num_threads=num_threads, language="c++").blocked_autorange())
#   ````
# `setup` and `stmt` do not change, so we can reuse the executable from the
# first pass through the loop.
BUILD_ROOT = os.path.join(
    tempfile.gettempdir(),
    f"benchmark_utils_jit_build_{uuid.uuid4()}".replace("-", "")
)

def _get_build_root() -> str:
    with LOCK:
        if not os.path.exists(BUILD_ROOT):
            os.makedirs(BUILD_ROOT)
            atexit.register(shutil.rmtree, BUILD_ROOT)
        return BUILD_ROOT


EXTRA_INCLUDE_PATHS: List[str] = [
    os.path.join(SOURCE_ROOT, "valgrind_wrapper"),
]
CONDA_PREFIX = os.getenv("CONDA_PREFIX")
if CONDA_PREFIX is not None:
    # Load will automatically search /usr/include, but not conda include.
    EXTRA_INCLUDE_PATHS.append(os.path.join(CONDA_PREFIX, "include"))


COMPAT_CALLGRIND_BINDINGS: Optional[CallgrindModuleType] = None
def get_compat_bindings() -> CallgrindModuleType:
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            build_dir = os.path.join(_get_build_root(), "compat_bindings")
            os.makedirs(build_dir)
            src_file = os.path.join(SOURCE_ROOT, "historic", "compat_bindings.cpp")
            dest_file = os.path.join(build_dir, "callgrind_bindings.cpp")
            shutil.copyfile(src_file, dest_file)
            COMPAT_CALLGRIND_BINDINGS = compat_jit(
                fpath=dest_file, cxx_flags=CXX_FLAGS, is_standalone=False)
    return COMPAT_CALLGRIND_BINDINGS


def _compile_template(
    template_name: str,
    template_path: str,
    stmt: str,
    setup: str,
    global_setup: str,
    is_standalone: bool
) -> Any:
    with open(template_path, "rt") as f:
        src = f.read()

    for before, after, indentation in (
        ("// GLOBAL_SETUP_TEMPLATE_LOCATION", global_setup, 0),
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
    # invocations. For compat bindings, we don't care and just use a unique
    # ID to guarantee Safety.
    with LOCK:
        build_root = _get_build_root()
        if IS_BACK_TESTING:
            build_dir = os.path.join(build_root, f"timer_cpp_{uuid.uuid4()}")
        else:
            name = f"timer_cpp_{abs(hash(src))}"
            build_dir = os.path.join(BUILD_ROOT, name)

        os.makedirs(build_dir, exist_ok=True)

        src_path = os.path.join(build_dir, f"{template_name}.cpp")
        with open(src_path, "wt") as f:
            f.write(src)

    if IS_BACK_TESTING:
        return compat_jit(
            fpath=src_path,
            cxx_flags=CXX_FLAGS,
            is_standalone=is_standalone,
        )

    # `cpp_extension` has its own locking scheme, so we don't need our lock.
    return cpp_extension.load(
        name=name,
        sources=[src_path],
        build_directory=build_dir,
        extra_cflags=CXX_FLAGS,
        extra_include_paths=EXTRA_INCLUDE_PATHS,
        is_python_module=not is_standalone,
        is_standalone=is_standalone,
    )


def compile_timeit_template(stmt: str, setup: str, global_setup: str) -> TimeitModuleType:
    template_path: str = os.path.join(SOURCE_ROOT, "timeit_template.cpp")
    module = _compile_template("timer_timeit", template_path, stmt, setup, global_setup, is_standalone=False)
    assert isinstance(module, TimeitModuleType)
    return module


def compile_callgrind_template(stmt: str, setup: str, global_setup: str) -> str:
    template_path: str = os.path.join(SOURCE_ROOT, "valgrind_wrapper", "timer_callgrind_template.cpp")
    target = _compile_template("timer_callgrind", template_path, stmt, setup, global_setup, is_standalone=True)
    assert isinstance(target, str)
    return target
