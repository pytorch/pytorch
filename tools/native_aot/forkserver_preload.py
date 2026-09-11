"""Preload the installed torch package in native-AOT's forkserver."""

import importlib
import os
import sys


_REPO = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_TORCH_PARENT_ENV = "TORCH_NATIVE_AOT_FORKSERVER_TORCH_PARENT"
_TORCH_PARENT = os.environ.pop(_TORCH_PARENT_ENV, None)
if not _TORCH_PARENT:
    raise RuntimeError(f"{_TORCH_PARENT_ENV} was not set by the native-AOT exporter")

_TORCH_PARENT_REAL = os.path.realpath(_TORCH_PARENT)
_ORIGINAL_PATH = sys.path[:]
try:
    # A forkserver starts through ``python -c``, which puts its cwd first and ignores
    # the parent path passed by multiprocessing. Import from the location export.py
    # selected, whether that is a wheel or an editable install.
    sys.path[:] = [_TORCH_PARENT] + [
        path
        for path in _ORIGINAL_PATH
        if os.path.realpath(path or os.getcwd()) not in {_REPO, _TORCH_PARENT_REAL}
    ]
    try:
        importlib.import_module("torch")
    except ImportError as e:
        # multiprocessing silently ignores ImportError from a preload module.
        raise RuntimeError(
            f"native-AOT forkserver could not import torch from {_TORCH_PARENT}"
        ) from e
finally:
    sys.path[:] = _ORIGINAL_PATH
