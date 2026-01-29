from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_utils import (
    get_nvmatmul_gpu_enum,
)
from torch._inductor.utils import ensure_nvmatmul_heuristics_available
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet

from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from ..kernel_inputs import KernelInputs, MMKernelInputs


log = logging.getLogger(__name__)
# Use autotuning artifact logger for detailed nvMatmulHeuristics logging
# Enable with TORCH_LOGS="+autotuning"
autotuning_log = getArtifactLogger(__name__, "autotuning")


@dataclass
class HeuristicConfig:
    """Configuration recommended by nvMatmulHeuristics."""

    tile_m: int
    tile_n: int
    tile_k: int
    cluster_m: int
    cluster_n: int
    stages: int
    split_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    instr_tile_m: int
    instr_tile_n: int
    instr_tile_k: int
    swizzle_factor: int
    cta_order: int  # 0 = raster along M, 1 = raster along N
    estimated_runtime: float


def _kernel_matches_heuristic(kernel, cfg: HeuristicConfig) -> tuple[bool, int]:
    """
    Check if kernel matches heuristic config on all fields it supports.

    Returns (matches, missing_count) where:
    - matches: True if kernel matches on all fields it supports
    - missing_count: Number of heuristic fields the kernel doesn't have
    """
    design = kernel.metadata.design
    missing = 0

    # tile_shape -> (tile_m, tile_n, tile_k)
    if hasattr(design, "tile_shape") and len(design.tile_shape) >= 3:
        if design.tile_shape[0] != cfg.tile_m:
            return (False, 0)
        if design.tile_shape[1] != cfg.tile_n:
            return (False, 0)
        if design.tile_shape[2] != cfg.tile_k:
            return (False, 0)
    else:
        missing += 3

    # cluster_shape -> (cluster_m, cluster_n)
    if hasattr(design, "cluster_shape") and len(design.cluster_shape) >= 2:
        if design.cluster_shape[0] != cfg.cluster_m:
            return (False, 0)
        if design.cluster_shape[1] != cfg.cluster_n:
            return (False, 0)
    else:
        missing += 2

    # All other fields: match by same name
    for field in dataclasses.fields(HeuristicConfig):
        name = field.name
        # Skip fields already handled or not config values
        if name in (
            "tile_m",
            "tile_n",
            "tile_k",
            "cluster_m",
            "cluster_n",
            "estimated_runtime",
        ):
            continue
        if hasattr(design, name):
            if getattr(design, name) != getattr(cfg, name):
                return (False, 0)
        else:
            missing += 1

    return (True, missing)


class NVUniversalGemmHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """
    Heuristics for NVGEMM kernel selection using nvMatmulHeuristics.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        """Check if heuristics should be used.

        Args:
            inputs: KernelInputs
        """
        return super().should_run(inputs) and ensure_nvmatmul_heuristics_available()

    def filter_kernels(
        self,
        kernels: list,
        inputs: MMKernelInputs,
        count: int,
        accumulator_type: torch.dtype = torch.float32,
    ) -> list:
        """
        Filter and rank kernels using nvMatmulHeuristics.

        Matches on (tile_m, tile_n, tile_k, cluster_m, cluster_n).
        Returns kernels sorted by estimated runtime.

        If nvMatmulHeuristics is not installed or max_autotune is disabled,
        returns the first `count` kernels without heuristic ranking.

        Args:
            kernels: List of cutlass_api.Kernel objects
            inputs: MMKernelInputs with matrix shapes, dtypes, and strides
            count: Maximum number of kernels to return
            accumulator_type: Accumulator dtype

        Returns:
            Filtered list of kernels, sorted by estimated performance
        """
        if not self.should_run(inputs):
            return kernels[:count]

        m, n, k = inputs.mnk_hinted()
        batch_size = inputs.batch_hinted()
        dtype_a = inputs.dtype(inputs._mat1_idx)
        strides = inputs.strides_hinted()
        layout_a = "row" if strides[inputs._mat1_idx][-1] == 1 else "col"
        layout_b = "row" if strides[inputs._mat2_idx][-1] == 1 else "col"

        heuristic_configs = self._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a,
            layout_a,
            layout_b,
            count,
            kernels,
            accumulator_type,
            batch_size,
        )

        if not heuristic_configs:
            log.debug("No heuristic configs found, using first %d kernels", count)
            return kernels[:count]

        # Match kernels to heuristic configs
        # Priority: kernels matching more heuristic fields rank higher
        matched: list[tuple[object, float, int]] = []  # (kernel, runtime, missing)
        for cfg in heuristic_configs:
            for kernel in kernels:
                matches, missing_count = _kernel_matches_heuristic(kernel, cfg)
                if matches:
                    matched.append((kernel, cfg.estimated_runtime, missing_count))

        if not matched:
            log.debug(
                "No kernels matched heuristic configs, using first %d kernels", count
            )
            return kernels[:count]

        # Sort by (missing_count, runtime) - prefer kernels matching more fields
        matched.sort(key=lambda x: (x[2], x[1]))
        selected = matched[:count]
        result = [k for k, _, _ in selected]

        log.debug(
            "Heuristic filtered to %d kernels from %d total", len(result), len(kernels)
        )

        self._log_selected_kernels(heuristic_configs, matched, kernels, selected)

        return result

    def _log_selected_kernels(
        self,
        heuristic_configs: list[HeuristicConfig],
        matched: list[tuple[object, float, int]],
        kernels: list,
        selected: list[tuple[object, float, int]],
    ) -> None:
        """Log details about selected kernels."""
        autotuning_log.info(
            "nvMatmulHeuristics kernel filtering: %d heuristic configs matched %d "
            "of %d available kernels, returning top %d",
            len(heuristic_configs),
            len(matched),
            len(kernels),
            len(selected),
        )
        for i, (kernel, runtime, missing) in enumerate(selected):
            design = kernel.metadata.design  # pyrefly: ignore[missing-attribute]
            # Log fields the kernel supports
            field_strs: list[str] = []
            for field in dataclasses.fields(HeuristicConfig):
                name = field.name
                if name == "estimated_runtime":
                    continue
                if hasattr(design, name):
                    field_strs.append(f"{name}={getattr(design, name)}")
                elif name.startswith("tile_") and hasattr(design, "tile_shape"):
                    idx = {"tile_m": 0, "tile_n": 1, "tile_k": 2}.get(name)
                    if idx is not None:
                        field_strs.append(f"{name}={design.tile_shape[idx]}")
                elif name.startswith("cluster_") and hasattr(design, "cluster_shape"):
                    idx = {"cluster_m": 0, "cluster_n": 1}.get(name)
                    if idx is not None:
                        field_strs.append(f"{name}={design.cluster_shape[idx]}")
            autotuning_log.info(
                "  Selected kernel %d: [%s], missing=%d, runtime=%.2f us",
                i,
                ", ".join(field_strs),
                missing,
                runtime * 1e6,
            )

    def _get_layout_enum(self, layout_a: str, layout_b: str):
        """Map layout strings to NvMatmulHeuristicsMatmulLayout enum."""
        import nvMatmulHeuristics

        trans_a = "T" if layout_a == "row" else "N"
        trans_b = "T" if layout_b == "row" else "N"
        layout_str = f"{trans_a}{trans_b}_ROW_MAJOR"
        return nvMatmulHeuristics.NvMatmulHeuristicsMatmulLayout[layout_str]

    def _make_validity_callback(self, kernels: list):
        """
        Create callback for nvMatmulHeuristics that only accepts configurations
        matching the available kernel tile/cluster shapes.
        """
        # Build set of (tile_m, tile_n, tile_k, cluster_m, cluster_n) tuples
        valid_shapes: OrderedSet[tuple[int, int, int, int, int]] = OrderedSet()
        for kernel in kernels:
            design = kernel.metadata.design
            if hasattr(design, "tile_shape") and hasattr(design, "cluster_shape"):
                valid_shapes.add(
                    (
                        design.tile_shape[0],
                        design.tile_shape[1],
                        design.tile_shape[2],
                        design.cluster_shape[0],
                        design.cluster_shape[1],
                    )
                )

        def validity_check(kernel_config_ptr, problem_ptr):
            k = kernel_config_ptr.contents
            key = (k.cta[0], k.cta[1], k.cta[2], k.cluster[0], k.cluster[1])
            return 1 if key in valid_shapes else 0

        return validity_check

    def _get_heuristic_configs(
        self,
        m: int,
        n: int,
        k: int,
        dtype_a: torch.dtype,
        layout_a: str,
        layout_b: str,
        count: int,
        kernels: list,
        accumulator_type: torch.dtype = torch.float32,
        batch_size: int = 1,
    ) -> list[HeuristicConfig]:
        """
        Get kernel configurations recommended by nvMatmulHeuristics.

        Uses validity callback to filter to cutlass_api-compatible configs.
        """
        import nvMatmulHeuristics

        dtype_to_cublas = {
            torch.float64: "D",
            torch.float32: "S",
            torch.float16: "H",
            torch.bfloat16: "T",
        }
        a_char = dtype_to_cublas.get(dtype_a, "H")
        acc_char = dtype_to_cublas.get(accumulator_type, "S")
        precision = f"{a_char}{acc_char}{a_char}"

        # NvMatmulHeuristicsInterfaceEx configuration:
        # - backend=CUTLASS3: Use CUTLASS 3.x kernel database for Hopper+ GPUs
        #   TODO(nikhilap): Update when nvMatmulHeuristics supports CUTLASS 4
        # - flags=PERF_MODEL_BASED_AUTO_TUNING: Rank kernels using analytical
        #   performance model (faster than empirical profiling)
        # - load_discovery_implicitly=True: Auto-load kernel discovery sets on demand
        # - gpu: Set to detected GPU for accurate performance predictions
        gpu_enum = get_nvmatmul_gpu_enum()
        lh = nvMatmulHeuristics.NvMatmulHeuristicsInterfaceEx(
            backend=nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3,
            flags=nvMatmulHeuristics.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
            load_discovery_implicitly=True,
            gpu=gpu_enum,
        )

        backend = lh.createBackend(nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3)

        validity_callback = self._make_validity_callback(kernels)
        lh.setBackendCallbackProperty(
            backend,
            nvMatmulHeuristics.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK,
            validity_callback,
        )

        layout = self._get_layout_enum(layout_a, layout_b)

        lh.loadInternalDiscoverySet(layout, precision=precision)

        problem = lh.makeNvMatmulHeuristicsProblem(
            m, n, k, layout, batch_size=batch_size
        )
        raw_configs = lh.getEx(problem, count, backend, precision=precision)
        lh.destroyBackend(backend)

        if not raw_configs:
            autotuning_log.debug(
                "nvMatmulHeuristics returned 0 configs for M=%d, N=%d, K=%d, "
                "dtype=%s, layout=(%s, %s), precision=%s",
                m,
                n,
                k,
                dtype_a,
                layout_a,
                layout_b,
                precision,
            )
            return []

        configs = []
        for cfg in raw_configs:
            kernel = cfg["kernel"]
            configs.append(
                HeuristicConfig(
                    tile_m=kernel.cta_tile_m,
                    tile_n=kernel.cta_tile_n,
                    tile_k=kernel.cta_tile_k,
                    cluster_m=kernel.cluster_m,
                    cluster_n=kernel.cluster_n,
                    stages=kernel.stages,
                    split_k=kernel.split_k,
                    warp_tile_m=kernel.warp_tile_m,
                    warp_tile_n=kernel.warp_tile_n,
                    warp_tile_k=kernel.warp_tile_k,
                    instr_tile_m=kernel.instr_tile_m,
                    instr_tile_n=kernel.instr_tile_n,
                    instr_tile_k=kernel.instr_tile_k,
                    swizzle_factor=kernel.swizzle_factor,
                    cta_order=kernel.cta_order,
                    estimated_runtime=cfg["runtime"],
                )
            )

        autotuning_log.info(
            "nvMatmulHeuristics for M=%d, N=%d, K=%d, dtype=%s, layout=(%s, %s), "
            "precision=%s: %d configs returned",
            m,
            n,
            k,
            dtype_a,
            layout_a,
            layout_b,
            precision,
            len(configs),
        )
        for i, cfg in enumerate(configs):
            runtime_us = cfg.estimated_runtime * 1e6
            raster_dir = "M" if cfg.cta_order == 0 else "N"
            autotuning_log.info(
                "  Config %d: tile=(%d, %d, %d), cluster=(%d, %d), "
                "swizzle=%d, raster=%s, stages=%d, split_k=%d, "
                "warp_tile=(%d, %d, %d), instr_tile=(%d, %d, %d), "
                "estimated_runtime=%.2f us",
                i,
                cfg.tile_m,
                cfg.tile_n,
                cfg.tile_k,
                cfg.cluster_m,
                cfg.cluster_n,
                cfg.swizzle_factor,
                raster_dir,
                cfg.stages,
                cfg.split_k,
                cfg.warp_tile_m,
                cfg.warp_tile_n,
                cfg.warp_tile_k,
                cfg.instr_tile_m,
                cfg.instr_tile_n,
                cfg.instr_tile_k,
                runtime_us,
            )

        return configs


# Singleton instance for use in add_nv_universal_gemm_choices
_nvgemm_heuristics: NVUniversalGemmHeuristics | None = None


def get_nvgemm_heuristics() -> NVUniversalGemmHeuristics:
    """Get the singleton NVUniversalGemmHeuristics instance."""
    global _nvgemm_heuristics
    if _nvgemm_heuristics is None:
        _nvgemm_heuristics = NVUniversalGemmHeuristics()
    return _nvgemm_heuristics
