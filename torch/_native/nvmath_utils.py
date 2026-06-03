"""Shared helpers for native ops backed by nvmath-python / cuSOLVER.

Centralizes nvmath availability checks, cuSOLVER version gating, and cuSOLVER
handle management so multiple native ops (e.g. linalg.polar, and upcoming
matmul-family ops) don't each reimplement them.

All functions here are import-light at module load (nvmath is only imported
inside the function bodies, lazily, on first use) so importing this module does
not pull in the nvmath/cuda runtime or initialize CUDA.
"""

import torch

from .common_utils import _unavailable_reason


# (package_name, importable_module) pairs probed without importing.
_NVMATH_DEPS = [
    ("nvmath-python", "nvmath.bindings"),
]


def nvmath_unavailable_reason() -> str | None:
    """Return a human-readable reason nvmath is unavailable, or None if present.

    Does not import nvmath (uses importlib.util.find_spec under the hood).
    """
    return _unavailable_reason(_NVMATH_DEPS)


def cusolver_version() -> int | None:
    """cuSOLVER runtime version as major*1000 + minor*100 + patch, or None.

    Queries cusolverGetVersion through nvmath, against the exact cuSOLVER
    library nvmath dlopens (which is whatever the loader resolves -- the system
    CUDA install or the nvidia-cu* wheels a torch build depends on). The encoded
    value does not include the build number, so it can gate out runtimes that
    lack a symbol entirely but cannot distinguish patch builds.
    """
    try:
        from nvmath.bindings import cusolver  # pyrefly: ignore[missing-import]

        return cusolver.get_version()
    except Exception:
        return None


# One cuSOLVER handle per device, reused for the process lifetime. Handles are
# bound to the CUDA device that is current when cusolverDnCreate runs, so the
# handle must be created -- and later used -- with its device active.
#
# NOTE: this creates a dedicated handle rather than reusing aten's cuSOLVER
# handle. A future improvement is to reuse the handle PyTorch already maintains
# (see at::cuda::getCurrentCUDASolverDnHandle) to avoid the extra context.
_handles: dict[torch.device, int] = {}


def get_cusolver_handle(device: torch.device) -> int:
    """Return a cached cuSOLVER Dn handle bound to ``device``."""
    from nvmath.bindings import cusolverDn as cs  # pyrefly: ignore[missing-import]

    handle = _handles.get(device)
    if handle is None:
        # Create under the target device so the handle is bound to it.
        with torch.cuda.device(device):
            handle = cs.create()
        _handles[device] = handle
    return handle
