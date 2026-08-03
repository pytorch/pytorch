"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


__all__ = ["make_mxfp8_scaled_mm_gfx950"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.gemm_mxfp8_gfx950")
    value = getattr(module, name)
    globals()[name] = value
    return value
