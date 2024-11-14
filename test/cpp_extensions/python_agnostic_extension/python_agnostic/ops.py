import torch
from torch import Tensor

lib = torch.library.Library("torchao", "FRAGMENT")
lib.define("unpack_tensor_core_tiled_layout(Tensor packed_w, int inner_k_tiles) -> Tensor")
lib.define("dequantize_tensor_core_tiled_layout(Tensor packed_w, Tensor scales_and_zeros, int group_size, int inner_k_tiles) -> Tensor")


def register_custom_op(name):
    def decorator(func):
        return torch.library.register_fake(f"{name}")(func)
    return decorator


def unpack_tensor_core_tiled_layout(packed_w: Tensor, inner_k_tiles: int) -> Tensor:
    """
    Unpacks weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`.

    Assumes that the packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `inner_k_tiles = 2 | 4 | 8`"

    Args:
        packed_w: torch.tensor: 4D tensor with shape (N / 8) x (K / (inner_k_tiles * 16)) x 32 x inner_k_tiles, dtype is torch.int32
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.int32

    """
    return torch.ops.torchao.unpack_tensor_core_tiled_layout.default(
        packed_w=packed_w, inner_k_tiles=inner_k_tiles
    )


@register_custom_op("torchao::unpack_tensor_core_tiled_layout")
def _(packed_w: Tensor, inner_k_tiles: int) -> Tensor:
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 42d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        lambda: "inner_k_tiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == inner_k_tiles / 2,
        lambda: "packed weight must have inner_k_tiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * inner_k_tiles * 16

    return torch.empty((N, K), dtype=torch.int32, device=packed_w.device)

def dequantize_tensor_core_tiled_layout(packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int) -> Tensor:
    """
    Dequantizes by:
    - Unpacking weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`
    - Upcasting to bfloat16
    - Dequantizing with the scales_and_zeros that were packed with `torchao.quantization.utils.pack_tinygemm_scales_and_zeros`

    Assumes:
    - packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `inner_k_tiles = 2 | 4 | 8`"
    - packed scales_and_zeros were generated with `torchao.quantization.utils.pack_tinygemm_scales_and_zeros`
    - qGroupSize is 32 | 64 | 128 | 256

    Args:
        packed_w: torch.tensor: 4D tensor with shape `(N / 8) x (K / (inner_k_tiles * 16)) x 32 x inner_k_tiles / 2`, dtype is torch.int32
        scales_and_zeros: torch.tensor: 3D tensor with shape `numQGroups x N x 2`, dtype is torch.bfloat16 where numQGroups is K / qGroupSize
        group_size: int
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.bfloat16

    """
    return torch.ops.torchao.dequantize_tensor_core_tiled_layout.default(
        packed_w, scales_and_zeros, group_size, inner_k_tiles
    )


@register_custom_op("torchao::dequantize_tensor_core_tiled_layout")
def _(packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int) -> Tensor:
    # packed_w preconditions
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 4d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        lambda: "inner_k_tiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == inner_k_tiles / 2,
        lambda: "packed weight must have inner_k_tiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * inner_k_tiles * 16

    # scales_and_zeros preconditions
    torch._check(scales_and_zeros.dtype is torch.bfloat16, lambda: "scales_and_zeros must be bfloat16")
    torch._check(scales_and_zeros.dim() == 3, lambda: "scales_and_zeros must be 3D, got {scales_and_zeros.dim()}")
    torch._check(group_size == 32 or group_size == 64 or group_size == 128 or group_size == 256, lambda: "qGroupSize must be 32, 64, 128, or 256")
    torch._check(scales_and_zeros.size(0) == K // group_size, lambda: "scales_and_zeros must have K // qGroupSize at dim 0")
    torch._check(scales_and_zeros.size(1) == N, lambda: "scales_and_zeros must have N at dim 1")
    torch._check(scales_and_zeros.size(2) == 2, lambda: "scales_and_zeros must have 2 at dim 2")

    return torch.empty((N, K), dtype=torch.bfloat16, device=packed_w.device)

