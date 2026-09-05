"""Lazy exports for optional FlyDSL vendored kernels."""

import importlib
from typing import Any


__all__ = [
    "GEMM_DTYPE_BF16",
    "GEMM_DTYPE_FP16",
    "infer_has_k_tail",
    "make_gemm_gfx950_param",
    "make_gemm_param_and_validate",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f"{__name__}.gemm_gfx950")
    value = getattr(module, name)
    globals()[name] = value
    return value
