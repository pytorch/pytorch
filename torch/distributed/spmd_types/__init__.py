"""
Re-export shim for the ``spmd_types`` package.

Use ``import torch.distributed.spmd_types as spmd`` and guard attribute
access with ``spmd.is_available()``.  When the package is installed,
all public names from ``spmd_types`` are available on this module.
"""

import importlib as _importlib


_HAS_SPMD_TYPES = _importlib.util.find_spec("spmd_types") is not None


def is_available() -> bool:
    """Return True if the real spmd_types package is installed."""
    return _HAS_SPMD_TYPES


if _HAS_SPMD_TYPES:
    from spmd_types import *  # noqa: F403  # pyrefly: ignore[missing-import]
else:
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def no_typecheck():
        """No-op stub when spmd_types is not installed."""
        yield
