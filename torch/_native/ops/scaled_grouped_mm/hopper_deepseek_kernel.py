"""Launch point for SM90 DeepSeek FP8 grouped mm."""

import functools

import torch

from ._common import BLOCKWISE_1X128
from .group_meta import build_group_metadata
from .hopper_config import select_kernel_config


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(
        torch.device("cuda", device_index)
    ).multi_processor_count


def run_deepseek_grouped_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    recipe_a: int,
    recipe_b: int,
    offs: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    device_index = mat_a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    b_is_2d = mat_b.dim() == 2
    a_is_3d = mat_a.dim() == 3
    batched = a_is_3d and not b_is_2d
    jagged_n = a_is_3d and b_is_2d
    total_m = mat_a.size(-2)
    n = mat_b.size(-1)
    group_count = mat_a.size(0) if batched else offs.numel()
    config = select_kernel_config(
        total_m=total_m,
        n=n,
        k=mat_a.size(-1),
        group_count=group_count,
        num_sms=num_sms,
        groups_split_k=b_is_2d and not jagged_n,
        batched=batched,
        jagged_n=jagged_n,
    )
    # A group_start // 128 truncation needs 128-aligned boundaries: on the
    # A-scale row when offs splits M, on the B-scale column when it splits N.
    # A ragged N additionally starts B's and C's per-group TMA descriptors at
    # the group's column, and a bf16 C needs 8 elements for 16-byte alignment.
    # The metadata kernel asserts this on device; reading offs back here would
    # sync, which cannot be captured into a CUDA graph.
    if b_is_2d and not jagged_n:
        offs_align = 128
    elif jagged_n:
        offs_align = 128 if recipe_b != BLOCKWISE_1X128 else 8
    else:
        offs_align = 128 if recipe_a != BLOCKWISE_1X128 else 1
    problem_sizes, ptrs_abc, _, tile_offsets, total_tiles = build_group_metadata(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        recipe_a,
        offs,
        out,
        config=config,
        offs_align=offs_align,
    )
    # Batched needs no group metadata; offs is unread, so reuse a tensor of the
    # right dtype/rank rather than allocating one.
    offs_arg = tile_offsets if batched else offs
    # Local import: keeps cutlass off the `import torch` path.
    from .wgmma_kernel import launch_deepseek_grouped_wgmma

    launch_deepseek_grouped_wgmma(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        recipe_a,
        recipe_b,
        offs_arg,
        problem_sizes,
        tile_offsets,
        total_tiles,
        ptrs_abc,
        out,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        num_sms,
        config.a_scale_wide,
        config.b_scale_wide,
    )
    return out
