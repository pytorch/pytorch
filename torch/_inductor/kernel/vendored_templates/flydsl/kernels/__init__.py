"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


_EXPORT_MODULES = {
    "GEMM_DTYPE_BF16": "gemm_gfx950",
    "GEMM_DTYPE_FP16": "gemm_gfx950",
    "infer_has_k_tail": "gemm_gfx950",
    "make_gemm_gfx950_param": "gemm_gfx950",
    "make_gemm_param_and_validate": "gemm_gfx950",
    "get_grouped_gemm_persistent_grid_size": "grouped_gemm_gfx950",
    "launch_gemm_gfx950_grouped": "grouped_gemm_gfx950",
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
