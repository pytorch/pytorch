# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import operator

import torch
from torch.utils._ordered_set import OrderedSet

from ..hints import HeuristicType, TileHint, TRITON_MAX_BLOCK
from .common import (
    _handle_combo_kernel_per_subkernel_blocks,
    _maybe_filter_configs_for_tma_restrictions,
    autotune_hints_to_configs,
    cached_autotune,
    triton_config,
)
from .registry import register_triton_heuristic


# Register the pointwise heuristic for both "cuda" and None device types
# to ensure it is available for CUDA and as a fallback for other devices.
@register_triton_heuristic("pointwise", None)
@register_triton_heuristic("pointwise", "cuda")
def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
    return_configs=False,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta

    configs = _handle_combo_kernel_per_subkernel_blocks(
        size_hints,
        inductor_meta,
        triton_meta,
        filename=filename,
        tile_hint=tile_hint,
        min_elem_per_thread=min_elem_per_thread,
    )
    if configs is not None:
        return cached_autotune(
            None,
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.POINTWISE,
            filename=filename,
        )

    assert not inductor_meta.get("no_x_dim")

    numel = functools.reduce(operator.mul, size_hints.values())
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", OrderedSet()),
        size_hints,
        bs,
        triton_meta["device"],
    )

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )

    configs = None
    if len(size_hints) == 1:
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, bs)]
        else:
            configs = [
                triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
                triton_config_with_settings(
                    size_hints, bs // 2, num_elements_per_warp=64
                ),
                *hinted_configs,
            ]
            # Additional configs appended for ROCm builds
            if torch.version.hip:
                if inductor_meta.get("max_autotune_pointwise"):
                    configs.extend(
                        [
                            triton_config_with_settings(
                                size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2
                            ),
                            triton_config_with_settings(
                                size_hints,
                                4096,  # wrt: better than the max_block for some kernel
                            ),
                            triton_config_with_settings(
                                size_hints,
                                2048,
                                num_warps=8,
                                num_stages=2,
                                waves_per_eu=1,  # 20% improvement
                            ),
                        ]
                    )
                if inductor_meta.get("atomic_add_found"):
                    configs.extend(
                        [
                            triton_config_with_settings(
                                size_hints,
                                64,
                                num_warps=1,
                                num_stages=1,  # 250% improvement
                            )
                        ]
                    )
            if torch.xpu.is_available():
                configs.extend(
                    [  # intel-xpu-backend-for-triton #5133
                        triton_config_with_settings(size_hints, 32),
                    ]
                )
    if len(size_hints) == 2:
        # Only avoiding tuning on TileHint.SQUARE if not on ROCm builds
        # ROCm has observed improvement by diverging here
        if (
            not inductor_meta.get("autotune_pointwise", True)
            or (
                torch.version.hip is None
                and tile_hint == TileHint.SQUARE
                and torch.version.xpu is None
            )
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, 32, 32)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ]
            # Additional configs appended for ROCm builds
            if torch.version.hip:
                configs.extend(
                    [
                        triton_config_with_settings(
                            size_hints, 64, 32
                        ),  # better for some kernels
                        triton_config_with_settings(
                            size_hints, 128, 16
                        ),  # +10% for some kernels
                        triton_config_with_settings(
                            size_hints, 128, 32
                        ),  # additional 10% more
                        triton_config_with_settings(
                            size_hints, 32, 512
                        ),  # +30% for some kernels
                    ]
                )
            if torch.xpu.is_available():
                configs.extend(
                    [
                        # intel-xpu-backend-for-triton #5198
                        triton_config_with_settings(size_hints, 32, 32, num_warps=8),
                        # intel-xpu-backend-for-triton #5199
                        triton_config_with_settings(size_hints, 4, 256),
                    ]
                )
    if len(size_hints) == 3:
        if not (
            inductor_meta.get("max_autotune_pointwise") or torch.xpu.is_available()
        ):
            configs = [triton_config_with_settings(size_hints, 16, 16, 16)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ]

    if not configs:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _maybe_filter_configs_for_tma_restrictions(inductor_meta, configs)
    if return_configs:
        return configs

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )
