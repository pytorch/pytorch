import torch

from ... import triton_utils as tu


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

    device = a.get_device()
    if device == torch.cuda.current_device():
        return bmm_outer_product(a, b)

    with torch.cuda.device(device):
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
        a.is_cuda
        and b.is_cuda
        and a.device == b.device
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
    _register_for_dispatch_key("CUDA")
    if torch.xpu._is_compiled():
        _register_for_dispatch_key("XPU")
