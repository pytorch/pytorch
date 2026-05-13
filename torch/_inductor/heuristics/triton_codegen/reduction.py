"""Codegen heuristics for Triton reduction kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the methods that differ
from the base.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import ReductionHint, TRITON_MAX_BLOCK
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import prefix_is_reduction


if TYPE_CHECKING:
    from torch._inductor.runtime.triton_compat import Config


# ------------------------------------------------------------------
# Helpers (moved from triton_heuristics.py, only used by reduction)
# ------------------------------------------------------------------


def _get_tiling_scores(
    inductor_meta: dict[str, Any],
    size_hints: dict[str, int],
) -> dict[str, float]:
    """Retrieve tiling scores, providing suitable defaults if missing."""
    return inductor_meta.get("tiling_scores") or dict.fromkeys(size_hints, 1)


def _match_target_block_product(
    size_hints,
    tiling_scores,
    target_block_product,
    min_block_size=1,
    min_red_block: int | None = 4,
):
    """
    Distribute block sizes across dimensions according to tiling scores,
    aiming to match a target product of block sizes.
    """
    min_red_block = (
        min_block_size if min_red_block is None else max(min_red_block, min_block_size)
    )
    total_score = sum(tiling_scores.values())
    if total_score == 0:
        min_block_size = 1
        tiling_scores = dict.fromkeys(tiling_scores.keys(), target_block_product)
        total_score = target_block_product * len(tiling_scores)

    block_sizes: dict[str, int] = {}
    relative_scores: dict[str, float] = {}
    curr_block_product = 1

    for dim, score in tiling_scores.items():
        if score == 0 and "r" not in dim:
            block_sizes[dim] = 1
            relative_scores[dim] = 0
            continue

        size = min_block_size if "r" not in dim else min_red_block
        block_sizes[dim] = size
        curr_block_product *= size
        relative_scores[dim] = score / total_score

    while curr_block_product < target_block_product and relative_scores:
        dim, score = max(relative_scores.items(), key=lambda item: item[1])

        if (
            block_sizes[dim] >= TRITON_MAX_BLOCK[dim.capitalize()]
            or block_sizes[dim] >= size_hints[dim]
        ):
            del relative_scores[dim]
            continue

        block_sizes[dim] *= 2
        relative_scores[dim] /= 2
        curr_block_product *= 2

    return block_sizes


def _adapt_config_for_tiling(
    size_hints,
    tiling_scores,
    original_x,
    original_r,
    num_warps=None,
    num_stages=1,
    register_intensive=False,
    persistent_reduction=False,
    waves_per_eu=None,
) -> Config:
    """
    Create an adapted configuration based on tiling scores,
    redistributing the same total block size (x * r) according to tiling scores.
    """
    from torch._inductor.runtime.triton_heuristics import triton_config_tiled_reduction

    assert all(s in tiling_scores for s in size_hints)
    target_block_product = original_x * original_r
    block_sizes = _match_target_block_product(
        size_hints, tiling_scores, target_block_product
    )

    return triton_config_tiled_reduction(
        size_hints,
        block_sizes["x"],
        block_sizes["y"],
        block_sizes["r0_"],
        num_stages=num_stages,
        register_intensive=register_intensive,
        waves_per_eu=waves_per_eu,
    )


def _outer_config_opt(
    make_config, size_hints, rnumel, inductor_meta, num_dynamic, register_intensive
):
    """Optimized outer config for CUDA (non-HIP)."""
    max_x_block, x_block = 256, 64
    load_factor = inductor_meta.get("num_load", 0)
    x = size_hints["x"]
    num_warps = None

    if x <= 1024:
        x_block = max(min(x // 128, 8), 2)
        outer_r_block = min(rnumel, 64)
    elif x // 4096 <= 8:
        x_block = 16
        outer_r_block = 512 // x_block
    elif num_dynamic > 1:
        outer_r_block = max(min((rnumel // 64), 64), 8)
    elif num_dynamic == 1:
        outer_r_block = (
            1 if load_factor >= 3 else min(next_power_of_2(max(rnumel, 128) // 128), 8)
        )
    else:
        x_block = max(min(max_x_block, next_power_of_2(x // 4096)), x_block)
        if load_factor < 4 or rnumel <= 128:
            outer_r_block = 512 // x_block
        else:
            if rnumel >= 2048:
                outer_r_block = 64
            else:
                outer_r_block = 32
            x_block = min(x_block, 32)
            num_warps = 4

    return make_config(
        x_block,
        outer_r_block,
        num_warps=num_warps,
        register_intensive=register_intensive,
    )


# ------------------------------------------------------------------
# Default (CUDA) reduction heuristic
# ------------------------------------------------------------------


@register_codegen_heuristic("reduction")
class ReductionHeuristic(CodegenConfigHeuristics):
    """Default reduction autotuning configs (CUDA non-ROCm)."""

    def get_configs(
        self,
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
        num_dynamic: int = 0,
    ) -> list[Config]:
        """Generate non-persistent reduction autotuning configs."""
        from torch._inductor.runtime.triton_heuristics import (
            get_total_reduction_numel,
            make_matmul_triton_config,
            triton_config_reduction,
            triton_native_bmm_configs,
            triton_native_mm_configs,
        )

        reduction_hint = inductor_meta.get("reduction_hint")
        rnumel = get_total_reduction_numel(size_hints)

        max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
            "max_autotune_pointwise"
        )

        register_intensive = False
        loads_and_red = inductor_meta.get("num_load", 0) + inductor_meta.get(
            "num_reduction", 0
        )

        device_major = triton_meta["device"].major
        MAX_R0_BLOCK = 1024 if device_major is not None and device_major >= 10 else 2048
        if size_hints["x"] >= 1024 and loads_and_red >= 10:
            MAX_R0_BLOCK = 1024
            register_intensive = True

        if triton_meta.get("native_matmul"):
            if len(size_hints) == 3:
                return [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_mm_configs
                ]
            elif len(size_hints) == 4:
                return [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_bmm_configs
                ]
            else:
                raise NotImplementedError("native matmul only supports mm/bmm pattern")

        def make_config(
            x,
            r,
            num_warps=None,
            num_stages=1,
            register_intensive=False,
            dynamic_scale_rblock=True,
            waves_per_eu=None,
        ):
            if "y" in size_hints:
                tiling_scores = _get_tiling_scores(inductor_meta, size_hints)
                return _adapt_config_for_tiling(
                    size_hints,
                    tiling_scores,
                    x,
                    r,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    register_intensive=register_intensive,
                    waves_per_eu=waves_per_eu,
                )
            else:
                return triton_config_reduction(
                    size_hints,
                    x,
                    r,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    register_intensive=register_intensive,
                    waves_per_eu=waves_per_eu,
                    dynamic_scale_rblock=dynamic_scale_rblock,
                    reduction_hint=reduction_hint,
                )

        contiguous_config = make_config(
            2 if rnumel <= 2048 else 1,
            min(rnumel, MAX_R0_BLOCK),
            register_intensive=register_intensive,
        )
        tiny_config = make_config(
            2 * (256 // rnumel) if rnumel <= 256 else 1,
            min(rnumel, MAX_R0_BLOCK),
            register_intensive=register_intensive,
        )

        outer_config = self._get_outer_config(
            make_config,
            size_hints,
            rnumel,
            inductor_meta,
            num_dynamic,
            register_intensive,
        )

        configs: list[Config] = []

        if inductor_meta.get("add_persistent_rblock") and loads_and_red <= 8:
            xnumel = max(4096 // rnumel, 1)
            c = make_config(
                xnumel,
                min(rnumel, 32768),
                register_intensive=register_intensive,
                dynamic_scale_rblock=False,
            )
            configs.append(c)

        # For 3d tiling, default to more autotuning initially
        if "y" in size_hints:
            pass
        elif max_autotune_enabled:
            pass
        elif reduction_hint == ReductionHint.INNER:
            return configs + [contiguous_config]
        elif reduction_hint == ReductionHint.OUTER:
            return configs + [outer_config]
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return configs + [tiny_config]

        result_configs = configs + [
            contiguous_config,
            outer_config,
            tiny_config,
            make_config(64, 64),
            make_config(8, 512),
            make_config(64, 4, num_warps=8),
        ]

        return self._finalize_configs(
            result_configs, make_config, size_hints, inductor_meta
        )

    def get_persistent_configs(
        self,
        *,
        size_hints: dict[str, int],
        reduction_hint: Any = False,
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
    ) -> list[Config]:
        """Generate persistent reduction autotuning configs."""
        from torch._inductor.runtime.triton_heuristics import (
            get_total_reduction_numel,
            make_matmul_triton_config,
            triton_config_reduction,
            triton_config_tiled_reduction,
            triton_native_persistent_bmm_configs,
            triton_native_persistent_mm_configs,
        )

        # Deterministic batch invariance: canonicalize the batch-dim hint
        if inductor_meta and inductor_meta.get("batch_invariant"):
            size_hints = dict(size_hints)
            if "x" in size_hints:
                size_hints["x"] = max(size_hints["x"], 4096)

        xnumel = size_hints["x"]
        rnumel = get_total_reduction_numel(size_hints)

        MAX_PERSISTENT_BLOCK_NUMEL = 4096

        if triton_meta.get("native_matmul"):
            if len(size_hints) == 3:
                return [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_persistent_mm_configs
                ]
            elif len(size_hints) == 4:
                return [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_persistent_bmm_configs
                ]
            else:
                raise NotImplementedError("native matmul only supports mm/bmm pattern")

        max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
            "max_autotune_pointwise"
        )

        xblock_vals = self._persistent_xblock_vals()

        if "y" not in size_hints:
            configs = [
                triton_config_reduction(
                    size_hints,
                    xblock,
                    rnumel,
                    register_intensive=True,
                    reduction_hint=reduction_hint,
                )
                for xblock in xblock_vals
                if xblock == 1
                or (rnumel * xblock <= MAX_PERSISTENT_BLOCK_NUMEL and xblock <= xnumel)
            ]
        else:
            configs = []
            tiling_scores = _get_tiling_scores(inductor_meta, size_hints)
            x_y_scores = {dim: tiling_scores[dim] for dim in ("x", "y")}
            for target_block_size in xblock_vals:
                if target_block_size * rnumel > MAX_PERSISTENT_BLOCK_NUMEL:
                    continue

                block_sizes = _match_target_block_product(
                    size_hints, x_y_scores, target_block_size
                )
                configs.append(
                    triton_config_tiled_reduction(
                        size_hints, block_sizes["x"], block_sizes["y"], rnumel
                    )
                )

        tiny_configs = [
            triton_config_reduction(
                size_hints,
                2 * (256 // rnumel) if rnumel <= 256 else 1,
                rnumel,
            )
        ]

        # Defer to more autotuning for 3d tiling
        if "y" in size_hints:
            pass
        elif not max_autotune_enabled:
            if reduction_hint == ReductionHint.INNER and rnumel >= 256:
                if (
                    rnumel > 1024
                    or xnumel // 8 < 128
                    or inductor_meta.get("RSPLIT_SIZE")
                ):
                    configs = configs[:1]
                else:
                    configs = self._persistent_inner_config(
                        size_hints,
                        rnumel,
                        xnumel,
                        inductor_meta,
                        reduction_hint,
                    )
            elif reduction_hint == ReductionHint.OUTER:
                configs = configs[-1:]
            elif reduction_hint == ReductionHint.OUTER_TINY:
                configs = tiny_configs
        else:
            configs = self._persistent_max_autotune_extras(configs, tiny_configs)

        for c in configs:
            for p in size_hints:
                if prefix_is_reduction(p):
                    c.kwargs.pop(f"{p.upper()}BLOCK")  # type: ignore[union-attr]

        return configs

    # ------------------------------------------------------------------
    # Hook methods for device-specific overrides
    # ------------------------------------------------------------------

    def _get_outer_config(
        self,
        make_config,
        size_hints,
        rnumel,
        inductor_meta,
        num_dynamic,
        register_intensive,
    ):
        """Outer config. CUDA uses optimized heuristic."""
        return _outer_config_opt(
            make_config,
            size_hints,
            rnumel,
            inductor_meta,
            num_dynamic,
            register_intensive,
        )

    def _finalize_configs(self, configs, make_config, size_hints, inductor_meta):
        """Post-process non-persistent configs."""
        return configs

    def _persistent_xblock_vals(self) -> list[int]:
        """XBLOCK values for persistent reduction."""
        return [1, 8, 32, 128]

    def _persistent_inner_config(
        self,
        size_hints,
        rnumel,
        xnumel,
        inductor_meta,
        reduction_hint,
    ) -> list[Config]:
        """Config for INNER hint in persistent reduction."""
        from torch._inductor.runtime.triton_heuristics import triton_config_reduction

        x_block = min(1024 // rnumel, 8)
        return [
            triton_config_reduction(
                size_hints,
                x_block,
                rnumel,
                register_intensive=True,
                num_warps=1,
                min_num_warps=1,
                reduction_hint=reduction_hint,
            )
        ]

    def _persistent_max_autotune_extras(self, configs, tiny_configs):
        """Extra processing in max_autotune mode for persistent reduction."""
        return configs

    def get_cooperative_configs(
        self,
        *,
        size_hints: dict[str, int],
        reduction_hint: Any,
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
    ) -> list[Config]:
        """Generate configs for cooperative reduction (RSPLIT)."""
        from torch._inductor.runtime.hints import TRITON_MAX_RSPLIT
        from torch._inductor.runtime.runtime_utils import last_power_of_2

        assert len(size_hints) == 2, (
            "Cooperative reductions don't support tiling reduction dims"
        )
        xnumel, rnumel = size_hints["x"], size_hints["r0_"]

        target = last_power_of_2(triton_meta["device"].multi_processor_count)
        split = max(1, min((rnumel, target // xnumel, TRITON_MAX_RSPLIT)))
        if inductor_meta.get("persistent_reduction", False):
            configs = self.get_persistent_configs(
                size_hints={"x": xnumel, "r0_": rnumel // split},
                reduction_hint=reduction_hint,
                inductor_meta=inductor_meta,
                triton_meta=triton_meta,
            )
        else:
            configs = self.get_configs(
                size_hints={"x": xnumel, "r0_": rnumel // split},
                inductor_meta=inductor_meta,
                triton_meta=triton_meta,
            )
        for config in configs:
            config.kwargs["RSPLIT"] = split  # type: ignore[union-attr]
        return configs

    def apply_rsplit_size(
        self,
        configs: list[Config],
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
    ) -> list[Config]:
        """Apply RSPLIT_SIZE / mix-order post-processing to persistent configs."""
        import copy

        from torch._inductor.runtime.triton_heuristics import unique_configs

        max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
            "max_autotune_pointwise"
        )

        rsplit_size = inductor_meta.get("RSPLIT_SIZE")
        if not rsplit_size:
            return configs

        rnumel_hint = size_hints["r0_"]
        min_x_block = 1
        if rnumel_hint <= 512:
            min_x_block = 4
        required_x_block = 1
        if (
            tma_min_block_sizes := inductor_meta.get("tma_min_block_sizes")
        ) is not None:
            required_x_block = max(
                required_x_block, tma_min_block_sizes.get("XBLOCK", 1)
            )
        x_block = min(max(rsplit_size // 32, min_x_block, required_x_block), 16)

        new_configs: list[Config] = []
        for c in configs:
            c.kwargs["RSPLIT_SIZE"] = rsplit_size  # type: ignore[union-attr]
            c.kwargs["XBLOCK"] = x_block  # type: ignore[union-attr]

            num_iters = rsplit_size // x_block

            if inductor_meta.get("mix_order_reduction_allow_multi_stages", True):
                MAX_NUM_STAGES = 2 if rnumel_hint > 8192 else 3
            else:
                MAX_NUM_STAGES = 1
            c.kwargs["NUM_STAGES"] = min(  # type: ignore[union-attr]
                max(num_iters // 4, 1), MAX_NUM_STAGES
            )

            if rnumel_hint <= 1024:
                c.num_warps //= 2  # type: ignore[union-attr]
                c.num_warps = max(c.num_warps, 1)  # type: ignore[union-attr]
                new_configs.append(c)

                if max_autotune_enabled:
                    newc = copy.deepcopy(c)
                    newc.num_warps = 2  # type: ignore[union-attr]
                    new_configs.append(newc)
            else:
                new_configs.append(c)

                max_warps_limit = self._max_warps_limit()
                if max_autotune_enabled and c.num_warps < max_warps_limit:  # type: ignore[union-attr]
                    newc = copy.deepcopy(c)
                    newc.num_warps *= 2  # type: ignore[union-attr]
                    new_configs.append(newc)
        return unique_configs(new_configs)

    def _max_warps_limit(self) -> int:
        """Max warps limit for RSPLIT_SIZE processing."""
        return 32

    def get_split_scan_configs(
        self,
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
    ) -> list[Config]:
        """Generate configs for split scan kernels."""
        configs = self.get_configs(
            size_hints=size_hints,
            inductor_meta=inductor_meta,
            triton_meta=triton_meta,
        )
        min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
        for cfg in configs:
            for var in list(cfg.kwargs.keys()):  # type: ignore[union-attr]
                if var.startswith("R") and cfg.kwargs[var] < min_rblock:  # type: ignore[union-attr]
                    cfg.kwargs[var] = min_rblock  # type: ignore[union-attr]
        return configs


# ------------------------------------------------------------------
# ROCm/HIP reduction heuristic
# ------------------------------------------------------------------


@register_codegen_heuristic("reduction", "hip", register=torch.version.hip is not None)
class ROCmReductionHeuristic(ReductionHeuristic):
    """Reduction configs for ROCm/HIP devices."""

    def _get_outer_config(
        self,
        make_config,
        size_hints,
        rnumel,
        inductor_meta,
        num_dynamic,
        register_intensive,
    ):
        # HIP uses simple outer config (no outer_config_opt)
        return make_config(64, 8, register_intensive=register_intensive)

    def _finalize_configs(self, configs, make_config, size_hints, inductor_meta):
        hip_configs = [
            make_config(1024, 8, num_warps=4, num_stages=1, waves_per_eu=2),
            make_config(512, 8, num_warps=4, num_stages=1, waves_per_eu=1),
        ]
        configs.extend(hip_configs)

        max_persistent_rblock = inductor_meta.get("max_persistent_rblock", 0)
        if max_persistent_rblock > 0:
            configs = [
                c
                for c in configs
                if c.kwargs.get("XBLOCK", 0) * max_persistent_rblock <= 4096
            ]
        return configs

    def _persistent_xblock_vals(self) -> list[int]:
        return [1, 4, 8, 16, 32, 64, 128, 256]

    def _persistent_max_autotune_extras(self, configs, tiny_configs):
        for conf in tiny_configs:
            if conf not in configs:
                configs.append(conf)
        return configs

    def _max_warps_limit(self) -> int:
        return 16


# ------------------------------------------------------------------
# XPU reduction heuristic
# ------------------------------------------------------------------


@register_codegen_heuristic("reduction", "xpu", register=torch.xpu._is_compiled())
class XPUReductionHeuristic(ReductionHeuristic):
    """Reduction configs for XPU devices."""

    def _persistent_inner_config(
        self,
        size_hints,
        rnumel,
        xnumel,
        inductor_meta,
        reduction_hint,
    ) -> list[Config]:
        from torch._inductor.runtime.triton_heuristics import triton_config_reduction

        # TODO(Intel): CUDA uses num_warps = 1 to disable shared memory.
        # We apply different configurations from #168335.
        # We currently let cost model in Triton to decide whether to use
        # shared memory.
        loads_and_stores = inductor_meta.get("num_load", 0) + inductor_meta.get(
            "num_store", 0
        )
        x_block = 8
        if xnumel // x_block < 128 or loads_and_stores >= 5:
            x_block = 1
        return [
            triton_config_reduction(
                size_hints,
                x_block,
                rnumel,
                register_intensive=True,
                num_warps=None,
                min_num_warps=None,
                reduction_hint=None,
            )
        ]
