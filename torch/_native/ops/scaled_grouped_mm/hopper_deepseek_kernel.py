"""Launch point for SM90 DeepSeek FP8 grouped mm."""

import functools

import torch

from ._common import BLOCKWISE_1X128, fp32_scale_stage_size
from .group_meta import build_group_metadata
from .hopper_config import select_kernel_config


_launch_deepseek_grouped_wgmma = None


def _get_launch_deepseek_grouped_wgmma():
    global _launch_deepseek_grouped_wgmma
    if _launch_deepseek_grouped_wgmma is None:
        from .wgmma_kernel import launch_deepseek_grouped_wgmma

        _launch_deepseek_grouped_wgmma = launch_deepseek_grouped_wgmma
    return _launch_deepseek_grouped_wgmma


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
    total_m = mat_a.size(0)
    n = mat_b.size(-1)
    group_count = mat_b.size(0)
    config = select_kernel_config(
        total_m=total_m,
        n=n,
        k=mat_a.size(1),
        group_count=group_count,
        num_sms=num_sms,
    )
    torch._check(
        n >= config.tile_n,
        lambda: f"DeepSeek grouped mm requires n ({n}) >= tile_n ({config.tile_n})",
    )
    if recipe_b == BLOCKWISE_1X128 and config.b_scale_wide:
        scale_b_cols = fp32_scale_stage_size(config.tile_n)
        torch._check(
            n >= scale_b_cols,
            lambda: f"DeepSeek grouped mm requires n ({n}) >= scale B cols ({scale_b_cols})",
        )
    if recipe_a != BLOCKWISE_1X128 and offs.numel() > 1:
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
    _get_launch_deepseek_grouped_wgmma()(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        recipe_a,
        recipe_b,
        offs,
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
