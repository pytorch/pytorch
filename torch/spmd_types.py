import importlib as _importlib


_mod = _importlib.import_module("spmd_types")

_public = [name for name in dir(_mod) if not name.startswith("_")]
globals().update({name: getattr(_mod, name) for name in _public})
__all__ = _public  # noqa: PLE0605

# Re-export submodule APIs needed by tests and internal users.
from spmd_types._checker import get_partition_spec  # noqa: F401  # pyrefly: ignore
from spmd_types._mesh_axis import _reset  # noqa: F401  # pyrefly: ignore
from spmd_types._type_attr import set_local_type  # noqa: F401  # pyrefly: ignore
from spmd_types.runtime import (  # noqa: F401  # pyrefly: ignore
    _set_partition_spec,
    has_local_type,
)
from spmd_types.types import normalize_axis  # noqa: F401  # pyrefly: ignore
