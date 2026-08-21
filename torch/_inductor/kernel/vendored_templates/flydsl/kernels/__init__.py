"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


_MODULE_BY_NAME = {
    "MXFP4GemmParams": "gemm_mxfp4_gfx950",
    "make_mxfp4_param_and_validate": "gemm_mxfp4_gfx950",
    "make_mxfp4_scaled_mm_gfx950": "gemm_mxfp4_gfx950",
    "mxfp4_gemm_derived": "gemm_mxfp4_gfx950",
    "MXFP8GemmParams": "gemm_mxfp8_gfx950",
    "make_mxfp8_param_and_validate": "gemm_mxfp8_gfx950",
    "make_mxfp8_scaled_mm_gfx950": "gemm_mxfp8_gfx950",
    "mxfp8_gemm_derived": "gemm_mxfp8_gfx950",
}

__all__ = list(_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.{_MODULE_BY_NAME[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
