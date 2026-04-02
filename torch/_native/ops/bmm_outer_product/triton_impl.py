import functools
import os

import torch

from ... import triton_utils as tu


_BACKEND_ENV = "TORCH_BMM_OUTER_PRODUCT_BACKEND"
_DEFAULT_BACKEND = "triton"
_VALID_BACKENDS = frozenset({"triton", "cutedsl"})


def _is_outer_product(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[2] == 1
        and b.shape[1] == 1
        and a.numel() > 0
        and b.numel() > 0
        and not a.is_complex()
    )


def _get_bmm_outer_product_backend() -> str:
    backend = os.getenv(_BACKEND_ENV, _DEFAULT_BACKEND).lower()
    if backend in _VALID_BACKENDS:
        return backend
    return _DEFAULT_BACKEND


def _run_bmm_outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    match _get_bmm_outer_product_backend():
        case "cutedsl":
            from .cutedsl_kernels import bmm_outer_product
        case _:
            from .triton_kernels import bmm_outer_product

    return bmm_outer_product(a, b)


def _bmm_outer_product_impl(
    dispatch_keys: torch.DispatchKeySet,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    fallback_kernel,
) -> torch.Tensor:
    if _is_outer_product(a, b):
        return _run_bmm_outer_product(a, b)
    return fallback_kernel.call_boxed(dispatch_keys, a, b)


def register_to_dispatch() -> None:
    fallback_kernel = torch.library.get_kernel("aten::bmm", "CUDA")
    tu.register_op_override(
        "aten",
        "bmm",
        "CUDA",
        functools.partial(_bmm_outer_product_impl, fallback_kernel=fallback_kernel),
        allow_multiple_override=True,
    )
