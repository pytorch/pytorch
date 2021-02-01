"""Compile specific Timer related C++ files.

This file provides a limited subset of the functionality of
`torch.utils.cpp_extension`, however it does so in a much simpler way which
tends to be more robust for the purpose of back testing.
"""
import atexit
import importlib.abc
import importlib.util
import os
import shutil
import subprocess
import sysconfig
import tempfile
from typing import cast

try:
    import pybind11
    PYBIND11_INSTALLED = True
except ImportError:
    PYBIND11_INSTALLED = False

import torch


_TARGETS = {
    # name:  link_libtorch,
    "timer_timeit": True,
    "timer_callgrind": True,
    "callgrind_bindings": False,
}


def compat_jit(*, fpath: str, cxx_flags: str, is_standalone: bool):
    build_dir, fname = os.path.split(fpath)
    name, ext = os.path.splitext(fname)

    assert name in _TARGETS, f"Valid targets: {', '.join(_TARGETS.keys())}"
    assert ext == ".cpp"
    assert PYBIND11_INSTALLED, "PyBind11 must be installed for compat JIT to function."

    link_libtorch = _TARGETS[name]
    valgrind_include = os.path.abspath(os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        "..", "valgrind_wrapper"))

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix is not None:
        lib_path = os.path.join(conda_prefix, "lib")
    else:
        lib_path = os.path.split(sysconfig.get_paths()["stdlib"])[0]

    cwd = os.path.split(os.path.abspath(__file__))[0]
    shutil.copyfile(
        os.path.join(cwd, "CMakeLists.txt"),
        os.path.join(build_dir, "CMakeLists.txt")
    )

    cmake_vars = {
        "NAME": name,
        "TORCH_ROOT": os.path.split(torch.__file__)[0],
        "VALGRIND_INCLUDE": valgrind_include,
        "LIB_PATH": lib_path,
        "CMAKE_CXX_FLAGS": " ".join(cxx_flags),
        "IS_STANDALONE": str(int(is_standalone)),
        "LINK_LIBTORCH": str(int(link_libtorch)),
    }
    def_args = ' '.join([f'-D{k}="{v}"' for k, v in cmake_vars.items()])

    build_result = subprocess.run(
        f"cmake {def_args} . && "
        "cmake --build . --config Release",
        shell=True,
        cwd=build_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )

    if build_result.returncode:
        raise ValueError(f"Failed to build: {name}\n{build_result.stdout}")

    def is_exec(i):
        full_path = os.path.join(build_dir, i)
        return os.access(full_path, os.X_OK) and os.path.isfile(full_path)

    # For Python modules, the final filename will depend on the platform.
    # e.g. "collect_wall_time.cpython-38-x86_64-linux-gnu.so"
    result_candidates = [i for i in os.listdir(build_dir) if is_exec(i)]
    assert len(result_candidates) == 1, ", ".join(result_candidates)
    target_path = os.path.join(build_dir, result_candidates[0])

    if is_standalone:
        return target_path

    module_spec = importlib.util.spec_from_file_location(name, target_path)
    assert module_spec, f"Failed to load module: {target_path}"
    module = importlib.util.module_from_spec(module_spec)
    loader = module_spec.loader
    assert loader is not None
    cast(importlib.abc.Loader, loader).exec_module(module)
    return module
