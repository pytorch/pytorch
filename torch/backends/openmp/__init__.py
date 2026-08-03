# mypy: allow-untyped-defs
import torch


def is_available():
    r"""Return whether PyTorch is built with OpenMP support."""
    return torch._C.has_openmp


def find_openmp_lib() -> str | None:
    r"""Return the loaded OpenMP runtime path, or ``None`` if none is loaded.

    Raises:
        RuntimeError: If multiple OpenMP runtimes are loaded in the process.
    """
    return torch._find_openmp_lib()
