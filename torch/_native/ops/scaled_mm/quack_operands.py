"""Views from PyTorch's block-scaled GEMM operands to QuACK's operand ABI.

Shared by the eager ``_scaled_mm_v2`` override and FlexGEMM's Inductor runtime.
"""

import functools

import torch


def quack_packed_operand(operand: torch.Tensor, k_dim: int) -> torch.Tensor:
    """View a K-major float6_e2m3fn_x4 GEMM operand as QuACK's packed uint8 bit stream.

    Each 3-byte element becomes 3 bytes along K (A: [M, K/4] with k_dim=-1 ->
    [M, 3K/4]; B: column-major [K/4, N] with k_dim=-2 -> [3K/4, N]).
    """
    if operand.dtype is not torch.float6_e2m3fn_x4:
        return operand
    if operand.stride(k_dim) != 1:
        raise NotImplementedError(
            "MXFP6 GEMM operands must be contiguous along K, got strides "
            f"{operand.stride()} for shape {tuple(operand.shape)}"
        )
    if k_dim == -1:
        return operand.view(torch.uint8)
    return operand.mT.view(torch.uint8).mT


def quack_blockscaled_scale_view(
    scale: torch.Tensor, mn: int, storage_k: int, format_name: str
) -> torch.Tensor:
    """View a public flat SWIZZLE_32_4_4 scale as QuACK's (rm, rk, 32, 4, 4) blocked tensor."""
    sf_k = -(-_logical_k(format_name, storage_k) // _sf_vec_size(format_name))
    return scale.view(-(-mn // 128), -(-sf_k // 4), 32, 4, 4)


@functools.cache
def _format(format_name: str):
    from torch._vendor.quack.blockscaled import operand as blockscaled

    return blockscaled.BlockScaledFormat.from_name(format_name)


@functools.cache
def _logical_k(format_name: str, storage_k: int) -> int:
    return _format(format_name).logical_k(storage_k)


def _sf_vec_size(format_name: str) -> int:
    return _format(format_name).sf_vec_size
