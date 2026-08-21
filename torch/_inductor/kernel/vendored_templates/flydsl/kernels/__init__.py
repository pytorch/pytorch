"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


_EXPORT_MODULES = {
    "build_flex_attn_fwd_module": "flex_attn_fwd_gfx950",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _EXPORT_MODULES[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
