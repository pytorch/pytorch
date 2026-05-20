"""Codegen heuristics for Triton pointwise kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the dimension methods that
differ from the base.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import TileHint, TRITON_MAX_BLOCK


if TYPE_CHECKING:
    from collections.abc import Callable


# ----------------------------------------------------------------------
# Default (CUDA) pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise")
class PointwiseHeuristic(CodegenConfigHeuristics):
    """Default pointwise autotuning configs (CUDA non-ROCm)."""

    def get_configs(
        self,
        size_hints: dict[str, int],
        bs: int,
        triton_config_fn: Callable[..., Any],
        hinted_configs: list[Any],
        tile_hint: TileHint | None = None,
        inductor_meta: dict[str, Any] | None = None,
    ) -> list[Any]:
        inductor_meta = inductor_meta or {}
        n = len(size_hints)
        if n == 1:
            return self._configs_1d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        if n == 2:
            return self._configs_2d(
                size_hints,
                bs,
                hinted_configs,
                triton_config_fn,
                tile_hint,
                inductor_meta,
            )
        if n == 3:
            return self._configs_3d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        raise NotImplementedError(f"size_hints: {size_hints}")

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, bs)]
        return [
            triton_config_fn(size_hints, bs, num_elements_per_warp=256),
            triton_config_fn(size_hints, bs // 2, num_elements_per_warp=64),
            *hinted_configs,
        ]

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        if (
            not inductor_meta.get("autotune_pointwise", True)
            or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        if not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 16, 16, 16)]
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]


# ----------------------------------------------------------------------
# ROCm/HIP pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
class ROCmPointwiseHeuristic(PointwiseHeuristic):
    """Pointwise configs for ROCm/HIP devices.

    Inherits _configs_3d from PointwiseHeuristic (same configs).
    """

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.extend(
                [
                    triton_config_fn(size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2),
                    triton_config_fn(
                        size_hints,
                        4096,  # wrt: better than the max_block for some kernel
                    ),
                    triton_config_fn(
                        size_hints,
                        2048,
                        num_warps=8,
                        num_stages=2,
                        waves_per_eu=1,  # 20% improvement
                    ),
                ]
            )
            if inductor_meta.get("atomic_add_found"):
                configs.append(
                    triton_config_fn(
                        size_hints,
                        64,
                        num_warps=1,
                        num_stages=1,  # 250% improvement
                    )
                )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # ROCm doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            triton_config_fn(size_hints, 64, 32),  # better for some kernels
            triton_config_fn(size_hints, 128, 16),  # +10% for some kernels
            triton_config_fn(size_hints, 128, 32),  # additional 10% more
            triton_config_fn(size_hints, 32, 512),  # +30% for some kernels
        ]


# ----------------------------------------------------------------------
# XPU pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "xpu", register=torch.xpu._is_compiled())
class XPUPointwiseHeuristic(PointwiseHeuristic):
    """Pointwise configs for XPU devices."""

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.append(
                # intel-xpu-backend-for-triton #5133
                triton_config_fn(size_hints, 32)
            )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # XPU doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            # intel-xpu-backend-for-triton #5198
            triton_config_fn(size_hints, 32, 32, num_warps=8),
            # intel-xpu-backend-for-triton #5199
            triton_config_fn(size_hints, 4, 256),
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # XPU always uses full set of configs (no autotune gate)
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]
