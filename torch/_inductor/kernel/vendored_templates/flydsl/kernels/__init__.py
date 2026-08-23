"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


_MODULE_BY_NAME = {
    "MXFPFormat": "gemm_mxfp_gfx950",
    "MXFPGemmDerived": "gemm_mxfp_gfx950",
    "MXFPGemmParams": "gemm_mxfp_gfx950",
    "make_mxfp_param_and_validate": "gemm_mxfp_gfx950",
    "make_mxfp_scaled_mm_gfx950": "gemm_mxfp_gfx950",
    "mxfp_gemm_derived": "gemm_mxfp_gfx950",
    "mxfp_pipeline_schedule": "gemm_mxfp_gfx950",
}

__all__ = list(_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.{_MODULE_BY_NAME[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value
