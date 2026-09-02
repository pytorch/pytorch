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
    batched = mat_a.dim() == 3
    total_m = mat_a.size(-2)
    n = mat_b.size(-1)
    group_count = mat_a.size(0) if batched else offs.numel()
    config = select_kernel_config(
        total_m=total_m,
        n=n,
        k=mat_a.size(-1),
        group_count=group_count,
        num_sms=num_sms,
        groups_split_k=mat_b.dim() == 2,
        batched=batched,
    )
    if recipe_a != BLOCKWISE_1X128 and not batched and offs.numel() > 1:
        # group_start // 128 truncation (accumulate_scaled) needs 128-aligned
        # boundaries. Checked here, not in _cond, because it needs a sync.
        torch._check(
            bool((offs[:-1] % 128 == 0).all()),
            lambda: "DeepSeek grouped mm with a BlockWise128x128 A-scale "
            "recipe requires all group boundaries in `offs` (except the "
            "last) to be a multiple of 128",
        )
    problem_sizes, ptrs_abc, _, tile_offsets, total_tiles = build_group_metadata(
        mat_a, mat_b, scale_a, scale_b, recipe_a, offs, out, config=config
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
