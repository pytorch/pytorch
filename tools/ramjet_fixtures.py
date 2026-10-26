"""Build native fixtures for Python tests without rebuilding PyTorch."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path


FIXTURE_SOURCES = {
    "torchbind_test": ("test/cpp/jit/test_custom_class_registrations.cpp",),
    "jitbackend_test": ("test/cpp/jit/test_backend_lib.cpp",),
    "backend_with_compiler": (
        "test/cpp/jit/test_backend_compiler_lib.cpp",
        "test/cpp/jit/test_backend_compiler_preprocess.cpp",
    ),
    "aoti_custom_ops": ("test/inductor/custom_ops.cpp",),
}
EDGE_PROFILER = "torch/csrc/jit/mobile/profiler_edge.cpp"
FIXTURE_FILES = (
    *sorted(path for sources in FIXTURE_SOURCES.values() for path in sources),
    "test/cpp/jit/test_custom_class_registrations.h",
    EDGE_PROFILER,
)


def build_fixtures(source_root: str = "/opt/ramjet-fixtures") -> None:
    import torch
    from torch.utils.cpp_extension import load

    root = Path(source_root)
    for relative in FIXTURE_FILES:
        if not root.joinpath(relative).is_file():
            raise FileNotFoundError(f"Missing native test fixture source: {relative}")
    installed = Path(torch.__file__).resolve().parent / "lib"
    cache = Path(tempfile.gettempdir()) / "ramjet-fixture-ccache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(CCACHE_DIR=str(cache), CCACHE_TEMPDIR=str(cache), MAX_JOBS="32")
    kineto = torch.profiler.kineto_available()
    for name, relative_sources in FIXTURE_SOURCES.items():
        sources = [str(root / path) for path in relative_sources]
        flags = []
        if name == "backend_with_compiler":
            if kineto:
                sources.append(str(root / EDGE_PROFILER))
                flags.append("-DUSE_KINETO")
            else:
                flags.append("-DNO_PROFILING")
        directory = root / "build" / name
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        library = load(
            name=name,
            sources=sources,
            extra_include_paths=[str(root)],
            extra_cflags=flags,
            extra_ldflags=[f"-Wl,-soname,lib{name}.so"],
            build_directory=str(directory),
            is_python_module=False,
            with_cuda=False,
            verbose=True,
        )
        shutil.copy2(library, installed / f"lib{name}.so")
        print(
            f"Ramjet built {name} in {time.monotonic() - started:.2f} seconds",
            flush=True,
        )
