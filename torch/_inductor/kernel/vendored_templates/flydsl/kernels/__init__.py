"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


# Maps each exported symbol to the submodule that defines it. Upstream's version
# hardcoded `gemm_gfx950` because that was the only vendored kernel; the MXFP8
# grouped GEMM adds a second, so the lookup is a map rather than a fixed import.
_EXPORT_MODULES = {
    "GEMM_DTYPE_BF16": "gemm_gfx950",
    "GEMM_DTYPE_FP16": "gemm_gfx950",
    "infer_has_k_tail": "gemm_gfx950",
    "make_gemm_gfx950_param": "gemm_gfx950",
    "make_gemm_param_and_validate": "gemm_gfx950",
    "launch_mxfp8_grouped_gemm_gfx950": "mxfp8_grouped_gemm_gfx950",
    "make_mxfp8_grouped_gemm_param": "mxfp8_grouped_gemm_gfx950",
    "make_mxfp8_grouped_gemm_param_and_validate": "mxfp8_grouped_gemm_gfx950",
    "pick_mxfp8_grouped_gemm_tile": "mxfp8_grouped_gemm_gfx950",
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
