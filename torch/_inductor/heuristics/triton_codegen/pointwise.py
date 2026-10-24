"""Codegen heuristics for Triton pointwise kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the dimension methods that
differ from the base.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import (
    CAPPED_GRID_DEFAULT_MAX_XBLOCK,
    capped_grid_max_block,
    TileHint,
    TRITON_DEFAULT_BLOCK_SIZES,
    TRITON_MAX_BLOCK,
)
from torch._inductor.runtime.runtime_utils import last_power_of_2, next_power_of_2
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _pointer_element_bytes(arg_type: object) -> int:
    """Bytes per element of a pointer or TMA descriptor in a Triton signature (2 for
    "*bf16"), 4 if it names no width (as "nvTmaDesc"), and 0 for other arguments."""
    if not isinstance(arg_type, str):
        return 0
    if arg_type == "nvTmaDesc":
        return 4
    if not arg_type.startswith(("*", "tensordesc<")):
        return 0
    bits = re.search(r"\d+", arg_type)
    return max(1, int(bits.group()) // 8) if bits else 4


# ----------------------------------------------------------------------
# Default (CUDA) pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise")
class PointwiseHeuristic(CodegenConfigHeuristics):
    """Default pointwise autotuning configs (CUDA non-ROCm)."""

    # (XBLOCK, num_warps) for a capped grid, from one H100 sweep of a bf16-to-fp16
    # cast at 44 and 66 programs; 65,536-element blocks on 8 warps spilled registers
    # on an A100. Entries above grid_max_xblock are skipped.
    capped_grid_candidates: tuple[tuple[int, int], ...] = (
        (16384, 16),
        (16384, 32),
        (32768, 8),
        (32768, 16),
    )
    # Bytes of the kernel's tensors per block: half of an SM's 65,536 32-bit
    # registers, which the largest candidate fills at the cast's 4 bytes per element
    capped_grid_bytes_per_block: int = 131072
    # The device types the candidates and byte budget are for: NVIDIA GPUs (ROCm
    # reports "hip"). This class is also the fallback for device types without their
    # own heuristic, and a "cuda" registration would shadow heuristics registered for
    # every device type. A subclass for another device names its type here.
    capped_grid_device_types: tuple[str, ...] = ("cuda",)

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

    def get_capped_grid_configs(
        self,
        *,
        size_hints: Mapping[str, int],
        inductor_meta: Mapping[str, Any],
        triton_meta: Mapping[str, Any],
        min_elem_per_thread: int = 0,
    ) -> list[Any]:
        """Configs for a capped pointwise kernel, replacing get_configs' (the only ones
        autotune hints add to); none outside capped_grid_device_types.

        It keeps the candidates whose XBLOCK is within grid_max_xblock and whose block
        holds at most capped_grid_bytes_per_block bytes of the kernel's tensors. For a
        static xnumel, XBLOCK also leaves each program a block for the fewest elements
        the size hint stands for, or is at most 128 elements or the rounded hint, if
        smaller. With none kept, it gives the largest such block with warps for 4
        elements per thread and, when autotuning, the table's smaller warp counts.

        Warps stay within the device's thread limit, and blocks of 128 elements or more
        get at least 4, as in triton_config. The raise for min_elem_per_thread also
        follows it, within the max block and the byte budget; where XBLOCK stops short,
        the warps drop, not below that floor. A candidate still short (fp8 conversions
        need the elements) is dropped unless a static xnumel's rounded hint is below the
        block it needs; with none left, pointwise() uses get_configs' configs.
        """
        # Type checkers read Triton's Config; triton_compat's stands in without Triton
        if TYPE_CHECKING:
            from triton import Config
        else:
            from torch._inductor.runtime.triton_compat import Config

        device = triton_meta["device"]
        if device.type not in self.capped_grid_device_types:
            return []
        warp_size = device.warp_size_or_default
        threads = device.max_threads_per_block or 1024
        max_warps = last_power_of_2(max(1, threads // warp_size))

        def warps(xblock: int, num_warps: int) -> int:
            # At least 4 warps from 128 elements outside ROCm, as in triton_config
            if xblock >= 128 and not torch.version.hip:
                num_warps = max(num_warps, 4)
            return max(1, min(num_warps, max_warps))

        grid_cap = inductor_meta["grid_cap"]
        # Every tensor argument counts in full, also one read once per program, such
        # as a 0-d tensor or an RNG seed
        signature = triton_meta.get("signature", {})
        element_bytes = sum(map(_pointer_element_bytes, signature.values()))
        limit = inductor_meta.get("grid_max_xblock")
        if limit is None:
            limit = CAPPED_GRID_DEFAULT_MAX_XBLOCK
        # The largest XBLOCK, also for a raise: the max block, within the byte budget
        largest = capped_grid_max_block(limit)
        if element_bytes:
            budget = self.capped_grid_bytes_per_block // element_bytes
            largest = min(largest, last_power_of_2(max(1, budget)))
        dynamic = inductor_meta.get("grid_dynamic_xnumel", False)
        if dynamic:
            # A dynamic xnumel's hint is one example of its sizes: it bounds nothing
            size_bound = share = largest
        else:
            size_bound = next_power_of_2(size_hints["x"])
            # The fewest elements a hint rounded up to a power of 2 stands for
            fewest = size_hints["x"] // 2 + 1
            share = last_power_of_2(max(1, fewest // grid_cap))
        floor = min(TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"], size_bound)
        max_xblock = min(limit, max(share, floor), largest)
        candidates = [
            (xblock, warps(xblock, num_warps))
            for xblock, num_warps in self.capped_grid_candidates
            if xblock <= max_xblock
        ]
        if not candidates:
            # No entry fits: the largest block that does, at 4 elements per thread and
            # the table's smaller warp counts, for the autotuner to measure
            most = warps(max_xblock, max_xblock // (4 * warp_size))
            counts = OrderedSet([most]) | OrderedSet(
                warps(max_xblock, w) for _, w in self.capped_grid_candidates if w < most
            )
            candidates = [(max_xblock, w) for w in sorted(counts, reverse=True)]
        # As in _configs_1d: one config without pointwise autotuning or max-autotune
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            candidates = candidates[:1]

        def needed(num_warps: int) -> int:
            # A power of 2, so that a raised block divides the max block
            return next_power_of_2(min_elem_per_thread * warp_size * num_warps)

        def raised(xblock: int, num_warps: int) -> int:
            return min(max(xblock, needed(num_warps)), size_bound, largest)

        def fitted(xblock: int, num_warps: int) -> tuple[int, int] | None:
            # XBLOCK grows as in triton_config; where it stops short, this hook
            # lowers the warps, and a floor that adds warps grows XBLOCK again
            if not min_elem_per_thread:
                return xblock, num_warps
            xblock = raised(xblock, num_warps)
            fitting_warps = xblock // (min_elem_per_thread * warp_size)
            num_warps = warps(xblock, last_power_of_2(min(num_warps, fitting_warps)))
            xblock = raised(xblock, num_warps)
            # Dropped if the max block or the byte budget left it short
            if xblock < needed(num_warps) and (
                dynamic or size_bound >= needed(num_warps)
            ):
                return None
            return xblock, num_warps

        pairs: list[tuple[int, int]] = []
        for xblock, num_warps in candidates:
            pair = fitted(xblock, num_warps)
            # Capped warps and raised blocks can make two candidates equal
            if pair is not None and pair not in pairs:
                pairs.append(pair)
        return [
            Config({"XBLOCK": xblock}, num_warps=num_warps, num_stages=1)
            for xblock, num_warps in pairs
        ]

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
                # intel-xpu-backend-for-triton #5124
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
