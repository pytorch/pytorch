# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# PyTorch-local blockscaled layout helpers shared by DTensor and quack wrappers.
# The packing formulas mirror the blocked MX/NVFP layouts used by quack and
# cuBLAS block-scaled GEMM paths.

import torch

MX_BLOCK_SIZE = 32
NVFP4_BLOCK_SIZE = 16


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def pack_scale_2d_to_blocked_contig(scale_2d: torch.Tensor) -> torch.Tensor:
    """Rearrange `(l, mn, sf_k)` or `(mn, sf_k)` scales into blocked layout."""
    if scale_2d.dim() == 2:
        scale_2d = scale_2d.unsqueeze(0)
    assert scale_2d.dim() == 3, f"expected (l, mn, sf_k), got shape {tuple(scale_2d.shape)}"
    orig_dtype = scale_2d.dtype
    l, mn, sf_k = scale_2d.shape
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    mn_pad = rm * 128
    sf_k_pad = rk * 4
    u8 = scale_2d.contiguous().view(torch.uint8)
    if mn_pad != mn or sf_k_pad != sf_k:
        padded = torch.zeros(l, mn_pad, sf_k_pad, device=scale_2d.device, dtype=torch.uint8)
        padded[:, :mn, :sf_k] = u8
    else:
        padded = u8
    blocks = padded.view(l, rm, 128, rk, 4).permute(0, 1, 3, 2, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, 4).transpose(3, 4).contiguous()
    return blocks.view(l, rm, rk, 512).view(orig_dtype)


def unpack_scale_blocked_contig(scale_contig: torch.Tensor, mn: int, sf_k: int) -> torch.Tensor:
    """Inverse of pack_scale_2d_to_blocked_contig."""
    squeeze_l = False
    if scale_contig.dim() == 3:
        scale_contig = scale_contig.unsqueeze(0)
        squeeze_l = True
    assert scale_contig.dim() == 4, (
        f"expected (l, rm, rk, 512) or (rm, rk, 512), got shape {tuple(scale_contig.shape)}"
    )
    orig_dtype = scale_contig.dtype
    l, rm, rk, inner = scale_contig.shape
    exp_rm = ceil_div(mn, 128)
    exp_rk = ceil_div(sf_k, 4)
    assert inner == 512, f"expected innermost tile size 512, got {inner}"
    assert rm == exp_rm and rk == exp_rk, (
        f"expected (rm, rk)=({exp_rm}, {exp_rk}), got ({rm}, {rk})"
    )
    u8 = scale_contig.contiguous().view(torch.uint8)
    blocks = u8.view(l, rm, rk, 32, 4, 4).transpose(3, 4).reshape(l, rm, rk, 128, 4)
    padded = blocks.permute(0, 1, 3, 2, 4).reshape(l, rm * 128, rk * 4)
    scale_2d = padded[:, :mn, :sf_k].contiguous().view(orig_dtype)
    return scale_2d.squeeze(0) if squeeze_l else scale_2d


def scale_view_for_kernel(scale_contig: torch.Tensor, mn: int, sf_k: int, l: int) -> torch.Tensor:
    """Validate a `(l, rm, rk, 512)` scale tensor and return it unchanged.

    Only the innermost 512-B tile must be contiguous (stride 1, size 512);
    outer `(l, rm, rk)` strides are free.
    """
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    assert scale_contig.shape == (l, rm, rk, 512), (
        f"expected (l, rm, rk, 512) = ({l}, {rm}, {rk}, 512), got {tuple(scale_contig.shape)}"
    )
    assert scale_contig.stride(-1) == 1, (
        f"innermost 512-B dim must be unit-stride, got stride {scale_contig.stride(-1)}"
    )
    return scale_contig


def scale_blocked_for_cublas(
    scale_contig: torch.Tensor, mn: int, sf_k: int, l_idx: int = 0
) -> torch.Tensor:
    """Flatten a blocked scale tensor to the 1D swizzled cuBLAS layout."""
    assert scale_contig.is_contiguous() and scale_contig.dim() == 4
    return scale_contig[l_idx].reshape(-1)


def scale_2d_from_cublas(scale_flat: torch.Tensor, mn: int, sf_k: int) -> torch.Tensor:
    """Unflatten a 1D cuBLAS block-scaled payload to logical `(mn, sf_k)`."""
    assert scale_flat.dim() == 1, f"expected 1D flat scale, got shape {tuple(scale_flat.shape)}"
    rm = ceil_div(mn, 128)
    rk = ceil_div(sf_k, 4)
    expected_size = rm * rk * 512
    assert scale_flat.numel() == expected_size, (
        f"expected {expected_size} elements for (mn={mn}, sf_k={sf_k}), got {scale_flat.numel()}"
    )
    return unpack_scale_blocked_contig(scale_flat.contiguous().view(rm, rk, 512), mn, sf_k)
