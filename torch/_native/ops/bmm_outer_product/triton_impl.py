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
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    from .triton_kernels import bmm_outer_product

    return bmm_outer_product(a, b)


def _bmm_outer_product_cond(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> bool:
    a_is_cow = torch._C._is_cow_tensor(a)  # pyrefly: ignore[missing-attribute]
    b_is_cow = torch._C._is_cow_tensor(b)  # pyrefly: ignore[missing-attribute]
    if (
        _has_triton()
        and a.is_cuda
        and b.is_cuda
        and _is_outer_product(a, b)
        and not (a_is_cow or b_is_cow)
    ):
        return True
    return False


def _register_for_dispatch_key(dispatch_key: str) -> None:
    tu.register_op_override(
        "aten",
        "bmm",
        dispatch_key,
        cond=_bmm_outer_product_cond,
        impl=_bmm_outer_product_impl,
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    if not _has_triton():
        return

    _register_for_dispatch_key("CUDA")
    if torch.xpu._is_compiled():
        _register_for_dispatch_key("XPU")
