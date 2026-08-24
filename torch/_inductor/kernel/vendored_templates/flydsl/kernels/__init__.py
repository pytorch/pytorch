"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


_EXPORT_MODULES = {
    "build_flex_attn_bwd_module": "flex_attn_bwd_gfx950",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    submodule = _EXPORT_MODULES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value
    return value
