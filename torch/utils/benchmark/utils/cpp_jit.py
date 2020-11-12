"""JIT C++ strings into executables."""
import atexit
import os
import re
import shutil
import sys
import textwrap
import threading
import uuid
from typing import Any, List, Optional

import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils import cpp_extension


LOCK = threading.Lock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]
BUILD_ROOT = os.path.join(
    torch._appdirs.user_cache_dir(appname="benchmark_utils_jit"),
    f"build_{uuid.uuid4()}".replace("-", "")
)

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


def _compile_template(stmt: str, setup: str, src: str, is_standalone: bool) -> Any:
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
    with LOCK:
        if not os.path.exists(BUILD_ROOT):
            os.makedirs(BUILD_ROOT)
            atexit.register(shutil.rmtree, BUILD_ROOT)

        name = f"timer_cpp_{abs(hash(src))}"
        build_dir = os.path.join(BUILD_ROOT, name)
        os.makedirs(build_dir, exist_ok=True)

        src_path = os.path.join(build_dir, "timer_src.cpp")
        with open(src_path, "wt") as f:
            f.write(src)

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


def compile_timeit_template(stmt: str, setup: str) -> TimeitModuleType:
    template_path: str = os.path.join(SOURCE_ROOT, "timeit_template.cpp")
    with open(template_path, "rt") as f:
        src: str = f.read()

    module = _compile_template(stmt, setup, src, is_standalone=False)
    assert isinstance(module, TimeitModuleType)
    return module


def compile_callgrind_template(stmt: str, setup: str) -> str:
    template_path: str = os.path.join(SOURCE_ROOT, "valgrind_wrapper", "timer_callgrind_template.cpp")
    with open(template_path, "rt") as f:
        src: str = f.read()

    target = _compile_template(stmt, setup, src, is_standalone=True)
    assert isinstance(target, str)
    return target
