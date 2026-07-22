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

    with torch.accelerator.device_index(a.get_device()):
        return bmm_outer_product(a, b)


def _is_acc_tensor(t: torch.Tensor) -> bool:
    acc = torch.accelerator.current_accelerator()
    return acc is not None and acc.type == t.device.type


def _bmm_outer_product_cond(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> bool:
    # a and b are read-only here: the kernel wraps them in ConstTensorWrapper and
    # reads through const_data_ptr(), so copy-on-write inputs are not
    # materialized and need not be excluded.
    if _is_acc_tensor(a) and a.device == b.device and _is_outer_product(a, b):
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
