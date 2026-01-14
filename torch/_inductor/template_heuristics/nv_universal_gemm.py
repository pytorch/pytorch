from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
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

# Type alias for kernel config key tuple.
# Currently matches on (tile_m, tile_n, tile_k, cluster_m, cluster_n).
# #TODO(nikhilap) When cutlass_api adds support for stages/split_k, extend this tuple and
# update the _make_config_key_* helper functions below.
ConfigKey = tuple[int, int, int, int, int]


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
    estimated_runtime: float


def _make_config_key_from_heuristic(cfg: HeuristicConfig) -> ConfigKey:
    """Build config key from HeuristicConfig returned by nvMatmulHeuristics."""
    return (cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.cluster_m, cfg.cluster_n)


def _make_config_key_from_kernel_design(design) -> ConfigKey | None:
    """Build config key from cutlass_api kernel metadata.design."""
    if (
        hasattr(design, "tile_shape")
        and len(design.tile_shape) >= 3
        and hasattr(design, "cluster_shape")
        and len(design.cluster_shape) >= 2
    ):
        return (
            design.tile_shape[0],
            design.tile_shape[1],
            design.tile_shape[2],
            design.cluster_shape[0],
            design.cluster_shape[1],
        )
    return None


def _make_config_key_from_heuristics_kernel(kernel) -> ConfigKey:
    """Build config key from nvMatmulHeuristics kernel config struct."""
    return (
        kernel.cta[0],
        kernel.cta[1],
        kernel.cta[2],
        kernel.cluster[0],
        kernel.cluster[1],
    )


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
        dtype_a = inputs.dtype(inputs._mat1_idx)
        strides = inputs.strides_hinted()
        layout_a = "row" if strides[inputs._mat1_idx][-1] == 1 else "col"
        layout_b = "row" if strides[inputs._mat2_idx][-1] == 1 else "col"

        config_to_kernels = self._extract_config_to_kernels(kernels)

        if not config_to_kernels:
            log.debug("Could not extract kernel configs, using first %d kernels", count)
            return kernels[:count]

        heuristic_configs = self._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a,
            layout_a,
            layout_b,
            count,
            OrderedSet(config_to_kernels.keys()),
            accumulator_type,
        )

        if not heuristic_configs:
            log.debug("No heuristic configs found, using first %d kernels", count)
            return kernels[:count]

        # Match kernels to heuristic configs
        matched: list[tuple] = []
        for cfg in heuristic_configs:
            key = _make_config_key_from_heuristic(cfg)
            kernels_for_key = config_to_kernels.get(key)
            if not kernels_for_key:
                continue
            for kernel in kernels_for_key:
                matched.append((kernel, cfg.estimated_runtime))

        if not matched:
            log.debug(
                "No kernels matched heuristic configs, using first %d kernels", count
            )
            return kernels[:count]

        matched.sort(key=lambda x: x[1])
        selected = matched[:count]
        result = [k for k, _ in selected]

        log.debug(
            "Heuristic filtered to %d kernels from %d total", len(result), len(kernels)
        )

        autotuning_log.info(
            "nvMatmulHeuristics kernel filtering: %d heuristic configs matched %d "
            "of %d available kernels, returning top %d",
            len(heuristic_configs),
            len(matched),
            len(kernels),
            len(result),
        )
        for i, (kernel, runtime) in enumerate(selected):
            design = kernel.metadata.design
            autotuning_log.info(
                "  Selected kernel %d: tile=(%d, %d, %d), cluster=(%d, %d), "
                "estimated_runtime=%.2f us",
                i,
                design.tile_shape[0],
                design.tile_shape[1],
                design.tile_shape[2],
                design.cluster_shape[0],
                design.cluster_shape[1],
                runtime * 1e6,
            )

        return result

    def _extract_config_to_kernels(self, kernels: list) -> dict[ConfigKey, list]:
        """Build a map from config key to kernels."""
        config_to_kernels: dict[ConfigKey, list] = defaultdict(list)

        for kernel in kernels:
            key = _make_config_key_from_kernel_design(kernel.metadata.design)
            if key is not None:
                config_to_kernels[key].append(kernel)

        return config_to_kernels

    def _get_layout_enum(self, layout_a: str, layout_b: str):
        """Map layout strings to NvMatmulHeuristicsMatmulLayout enum."""
        import nvMatmulHeuristics

        trans_a = "T" if layout_a == "row" else "N"
        trans_b = "T" if layout_b == "row" else "N"
        layout_str = f"{trans_a}{trans_b}_ROW_MAJOR"
        return nvMatmulHeuristics.NvMatmulHeuristicsMatmulLayout[layout_str]

    def _make_validity_callback(
        self,
        valid_configs: OrderedSet[ConfigKey],
    ):
        """
        Create callback for nvMatmulHeuristics that only accepts configurations
        matching the available cutlass_api kernel tile/cluster shapes.
        """

        def validity_check(kernel_config_ptr, problem_ptr):
            kernel = kernel_config_ptr.contents
            key = _make_config_key_from_heuristics_kernel(kernel)
            return 1 if key in valid_configs else 0

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
        valid_configs: OrderedSet[ConfigKey],
        accumulator_type: torch.dtype = torch.float32,
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
        lh = nvMatmulHeuristics.NvMatmulHeuristicsInterfaceEx(
            backend=nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3,
            flags=nvMatmulHeuristics.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
            load_discovery_implicitly=True,
        )

        backend = lh.createBackend(nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3)

        validity_callback = self._make_validity_callback(valid_configs)
        lh.setBackendCallbackProperty(
            backend,
            nvMatmulHeuristics.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK,
            validity_callback,
        )

        layout = self._get_layout_enum(layout_a, layout_b)

        lh.loadInternalDiscoverySet(layout, precision=precision)

        # TODO(nikhilap) support different batch sizes
        problem = lh.makeNvMatmulHeuristicsProblem(m, n, k, layout, batch_size=1)
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
            autotuning_log.info(
                "  Config %d: tile=(%d, %d, %d), cluster=(%d, %d), "
                "stages=%d, split_k=%d, warp_tile=(%d, %d, %d), "
                "estimated_runtime=%.2f us",
                i,
                cfg.tile_m,
                cfg.tile_n,
                cfg.tile_k,
                cfg.cluster_m,
                cfg.cluster_n,
                cfg.stages,
                cfg.split_k,
                cfg.warp_tile_m,
                cfg.warp_tile_n,
                cfg.warp_tile_k,
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
