import functools
import importlib.util

import torch

from ... import triton_utils as tu


@functools.cache
def _has_triton() -> bool:
    try:
        return importlib.util.find_spec("triton") is not None
    except ModuleNotFoundError:
        return False


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


def _bmm_outer_product_impl(
    dispatch_keys: torch.DispatchKeySet,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    fallback_kernel,
) -> torch.Tensor:
    a_is_cow = torch._C._is_cow_tensor(a)  # pyrefly: ignore[missing-attribute]
    b_is_cow = torch._C._is_cow_tensor(b)  # pyrefly: ignore[missing-attribute]
    if _has_triton() and _is_outer_product(a, b) and not (a_is_cow or b_is_cow):
        from .triton_kernels import bmm_outer_product

        return bmm_outer_product(a, b)
    return fallback_kernel.call_boxed(dispatch_keys, a, b)


def _register_for_dispatch_key(dispatch_key: str) -> None:
    fallback_kernel = torch.library.get_kernel("aten::bmm", dispatch_key)
    tu.register_op_override(
        "aten",
        "bmm",
        dispatch_key,
        functools.partial(_bmm_outer_product_impl, fallback_kernel=fallback_kernel),
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    if not _has_triton():
        return

    _register_for_dispatch_key("CUDA")
    if torch.xpu._is_compiled():
        _register_for_dispatch_key("XPU")
