"""
Re-export shim for the ``spmd_types`` package.

Use ``import torch.distributed.spmd_types as spmd`` and guard attribute
access with ``spmd.is_available()``.  When the package is installed,
all names from ``spmd_types`` are available on this module.
"""

import importlib as _importlib


_HAS_SPMD_TYPES = _importlib.util.find_spec("spmd_types") is not None


def is_available() -> bool:
    """Return True if the real spmd_types package is installed."""
    return _HAS_SPMD_TYPES


if _HAS_SPMD_TYPES:
    import spmd_types as _spmd_types  # pyrefly: ignore[missing-import]

    def __getattr__(name):
        attr = getattr(_spmd_types, name)
        globals()[name] = attr
        return attr

else:
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def no_typecheck():
        """No-op stub when spmd_types is not installed."""
        yield
