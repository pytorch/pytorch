from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from torch._inductor.utils import ensure_nvmatmul_heuristics_available
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet

from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from ..kernel_inputs import KernelInputs


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
    estimated_runtime: float


class NVUniversalGemmHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """
    Heuristics for NVGEMM kernel selection using nvMatmulHeuristics.

    Inherits from GemmMaxAutotuneTemplateConfigHeuristics for the should_run()
    implementation that checks max_autotune or max_autotune_gemm config.

    Unlike TemplateConfigHeuristics which generates config dicts for Triton templates,
    this class filters/ranks pre-built kernel objects from cutlass_api.
    """

    def should_run(self, inputs: Optional[KernelInputs] = None) -> bool:
        """Check if heuristics should be used.

        Only requires nvMatmulHeuristics to be available. The max_autotune check
        is already done before we reach add_nv_universal_gemm_choices().

        Args:
            inputs: KernelInputs (optional, kept for API consistency with parent class)
        """
        return ensure_nvmatmul_heuristics_available()

    def filter_kernels(
        self,
        kernels: list,
        m: int,
        n: int,
        k: int,
        dtype_a: torch.dtype,
        layout_a: str,
        layout_b: str,
        count: int,
    ) -> list:
        """
        Filter and rank kernels using nvMatmulHeuristics.

        Matches on (tile_m, tile_n, cluster_m, cluster_n).
        Returns kernels sorted by estimated runtime.

        Note: Caller should check should_run() before calling this method.

        Args:
            kernels: List of cutlass_api.Kernel objects
            m, n, k: GEMM problem dimensions
            dtype_a: Input dtype (fp16 or bf16)
            layout_a, layout_b: "row" or "col" for each input
            count: Maximum number of kernels to return

        Returns:
            Filtered list of kernels, sorted by estimated performance
        """
        valid_tile_m, valid_tile_n, valid_cluster_m, valid_cluster_n = (
            self._extract_valid_configs_from_kernels(kernels)
        )

        if not valid_tile_m:
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
            valid_tile_m,
            valid_tile_n,
            valid_cluster_m,
            valid_cluster_n,
        )

        if not heuristic_configs:
            log.debug("No heuristic configs found, using first %d kernels", count)
            return kernels[:count]

        # Build map from config key to estimated runtime
        config_key_to_runtime: dict[tuple, float] = {}
        for cfg in heuristic_configs:
            key = (cfg.tile_m, cfg.tile_n, cfg.cluster_m, cfg.cluster_n)
            if key not in config_key_to_runtime:
                config_key_to_runtime[key] = cfg.estimated_runtime

        # Match kernels to heuristic configs
        matched: list[tuple] = []
        for kernel in kernels:
            meta = kernel.metadata
            if hasattr(meta, "design"):
                design = meta.design
                tile_shape = design.tile_shape
                cluster_shape = design.cluster_shape
                key = (
                    tile_shape[0],
                    tile_shape[1],
                    cluster_shape[0],
                    cluster_shape[1],
                )
                if key in config_key_to_runtime:
                    matched.append((kernel, config_key_to_runtime[key]))

        if not matched:
            log.debug(
                "No kernels matched heuristic configs, using first %d kernels", count
            )
            return kernels[:count]

        matched.sort(key=lambda x: x[1])
        result = [k for k, _ in matched[:count]]

        log.debug(
            "Heuristic filtered to %d kernels from %d total", len(result), len(kernels)
        )

        # Log kernel matching results
        autotuning_log.info(
            "nvMatmulHeuristics kernel filtering: %d heuristic configs matched %d "
            "of %d available kernels, returning top %d",
            len(config_key_to_runtime),
            len(matched),
            len(kernels),
            len(result),
        )
        for i, (kernel, runtime) in enumerate(matched[:count]):
            meta = kernel.metadata
            if hasattr(meta, "design"):
                design = meta.design
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

    def _extract_valid_configs_from_kernels(
        self, kernels: list
    ) -> tuple[OrderedSet, OrderedSet, OrderedSet, OrderedSet]:
        """Extract valid tile/cluster configurations from available kernels."""
        tile_m_set: OrderedSet[int] = OrderedSet()
        tile_n_set: OrderedSet[int] = OrderedSet()
        cluster_m_set: OrderedSet[int] = OrderedSet()
        cluster_n_set: OrderedSet[int] = OrderedSet()

        for kernel in kernels:
            meta = kernel.metadata
            if hasattr(meta, "design"):
                design = meta.design
                if hasattr(design, "tile_shape") and len(design.tile_shape) >= 2:
                    tile_m_set.add(design.tile_shape[0])
                    tile_n_set.add(design.tile_shape[1])
                if hasattr(design, "cluster_shape") and len(design.cluster_shape) >= 2:
                    cluster_m_set.add(design.cluster_shape[0])
                    cluster_n_set.add(design.cluster_shape[1])

        return tile_m_set, tile_n_set, cluster_m_set, cluster_n_set

    def _get_layout_enum(self, layout_a: str, layout_b: str):
        """Map layout strings to NvMatmulHeuristicsMatmulLayout enum."""
        import nvMatmulHeuristics

        trans_a = "T" if layout_a == "row" else "N"
        trans_b = "T" if layout_b == "row" else "N"
        layout_str = f"{trans_a}{trans_b}_ROW_MAJOR"
        return nvMatmulHeuristics.NvMatmulHeuristicsMatmulLayout[layout_str]

    def _make_validity_callback(
        self,
        valid_tile_m: OrderedSet,
        valid_tile_n: OrderedSet,
        valid_cluster_m: OrderedSet,
        valid_cluster_n: OrderedSet,
    ):
        """
        Create callback for nvMatmulHeuristics that only accepts configurations
        matching the available cutlass_api kernel tile/cluster shapes.
        """

        def validity_check(kernel_config_ptr, problem_ptr):
            kernel = kernel_config_ptr.contents
            tile_m = kernel.cta[0]
            tile_n = kernel.cta[1]
            cluster_m = kernel.cluster[0]
            cluster_n = kernel.cluster[1]

            if tile_m not in valid_tile_m:
                return 0
            if tile_n not in valid_tile_n:
                return 0
            if cluster_m not in valid_cluster_m:
                return 0
            if cluster_n not in valid_cluster_n:
                return 0
            if cluster_m * cluster_n > 16:
                return 0
            return 1

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
        valid_tile_m: OrderedSet,
        valid_tile_n: OrderedSet,
        valid_cluster_m: OrderedSet,
        valid_cluster_n: OrderedSet,
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
        precision = f"{a_char}S{a_char}"  # e.g., "HSH" for fp16

        lh = nvMatmulHeuristics.NvMatmulHeuristicsInterfaceEx(
            backend=nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3,
            flags=nvMatmulHeuristics.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
            load_discovery_implicitly=True,
        )

        backend = lh.createBackend(nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3)

        # Set validity callback to filter to available kernel configs
        validity_callback = self._make_validity_callback(
            valid_tile_m, valid_tile_n, valid_cluster_m, valid_cluster_n
        )
        try:
            lh.setBackendCallbackProperty(
                backend,
                nvMatmulHeuristics.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK,
                validity_callback,
            )
        except Exception:
            log.debug("Could not set validity callback", exc_info=True)

        # Set CTA_TILE_N_DIV_REQUIREMENT
        cta_n_div = ctypes.c_int(32)
        lh.setBackendValueProperty(
            backend,
            nvMatmulHeuristics.NvMatmulHeuristicsBackendProperty.CTA_TILE_N_DIV_REQUIREMENT,
            ctypes.byref(cta_n_div),  # pyre-ignore[6]: ctypes typing
            ctypes.sizeof(cta_n_div),  # pyre-ignore[6]: ctypes typing
        )

        layout = self._get_layout_enum(layout_a, layout_b)

        try:
            lh.loadInternalDiscoverySet(layout, precision=precision)
        except Exception:
            pass

        problem = lh.makeNvMatmulHeuristicsProblem(m, n, k, layout, batch_size=1)
        raw_configs = lh.getEx(problem, count, backend, precision=precision)
        lh.destroyBackend(backend)

        if not raw_configs:
            autotuning_log.info(
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
                    estimated_runtime=cfg["runtime"],
                )
            )

        configs.sort(key=lambda c: c.estimated_runtime)

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
                "estimated_runtime=%.2f us",
                i,
                cfg.tile_m,
                cfg.tile_n,
                cfg.tile_k,
                cfg.cluster_m,
                cfg.cluster_n,
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
