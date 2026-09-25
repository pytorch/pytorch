from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.utils import ensure_nvmatmul_heuristics_available, has_free_symbols
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet

from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs, MMKernelInputs


log = logging.getLogger(__name__)
# Use autotuning artifact logger for detailed nvMatmulHeuristics logging
# Enable with TORCH_LOGS="+autotuning"
autotuning_log = getArtifactLogger(__name__, "autotuning")

# Type alias for kernel config key tuple.
# Currently matches on (tile_m, tile_n, cluster_m, cluster_n).
# tile_k excluded because nvMatmulHeuristics and cutlass.operators use it to mean different things.
# TODO(nikhilap): Extend config key for stages/split_k https://github.com/pytorch/pytorch/issues/177578
ConfigKey = tuple[int, int, int, int]

_NVFP4_MAX_PROFILING_CONFIGS = 3
_NVFP4_AUTOTUNE_CUDAGRAPH_UNROLL = 16
_NVFP4_PDL_MAX_M = 256
_NVFP4_PDL_MIN_WIDTH = 1024
_NVFP4_PDL_MAX_WIDTH = 65536

# Hand-picked configs in the space nvMatmulHeuristics does not currently
# explore. The final entries are NVFP4 oracle-best configs from an 83-shape
# LLM sweep; together, the supplemental pool reaches the oracle-best config on
# 73 of those shapes.
_SUPPLEMENT_CONFIGS: tuple[ConfigKey, ...] = (
    (64, 128, 1, 1),
    (64, 128, 1, 2),
    (64, 128, 1, 4),
    (64, 128, 1, 8),
    (64, 128, 1, 16),
    (64, 256, 1, 8),
    (64, 256, 1, 16),
    (64, 32, 1, 2),
    (64, 32, 1, 4),
    (128, 64, 1, 1),
    (128, 64, 1, 4),
    (128, 64, 2, 1),
    (128, 128, 1, 8),
    (128, 128, 1, 16),
    (128, 128, 2, 2),
    (128, 128, 2, 4),
    (128, 128, 2, 8),
    (128, 192, 1, 1),
    (128, 192, 1, 2),
    (128, 192, 1, 4),
    (128, 192, 2, 1),
    (128, 192, 2, 2),
    (128, 256, 1, 4),
    (128, 256, 1, 8),
    (128, 256, 1, 16),
    (128, 256, 2, 1),
    (128, 256, 2, 8),
    (256, 192, 1, 1),
    (256, 192, 1, 2),
    (256, 192, 1, 4),
    (256, 192, 2, 1),
    (256, 256, 2, 1),
    (256, 256, 2, 4),
    (256, 256, 4, 2),
    (256, 256, 4, 4),
    (256, 256, 8, 1),
    (256, 256, 8, 2),
    (256, 128, 2, 1),
    (256, 128, 2, 2),
    (128, 64, 1, 2),
    (128, 128, 1, 1),
    (128, 128, 1, 4),
    (128, 256, 1, 1),
    (256, 64, 2, 1),
    (256, 128, 4, 1),
    (256, 192, 2, 2),
    (256, 192, 4, 1),
    (256, 192, 4, 2),
    (256, 256, 4, 1),
)


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
    return (cfg.tile_m, cfg.tile_n, cfg.cluster_m, cfg.cluster_n)


def _make_config_key_from_kernel_design(design) -> ConfigKey | None:
    """Build config key from cutlass.operators kernel metadata.design."""
    if (
        hasattr(design, "tile_shape")
        and len(design.tile_shape) >= 2
        and hasattr(design, "cluster_shape")
        and len(design.cluster_shape) >= 2
    ):
        return (
            design.tile_shape[0],
            design.tile_shape[1],
            design.cluster_shape[0],
            design.cluster_shape[1],
        )
    return None


def _make_config_key_from_heuristics_kernel(kernel) -> ConfigKey:
    """Build config key from nvMatmulHeuristics kernel config struct."""
    return (
        kernel.cta[0],
        kernel.cta[1],
        kernel.cluster[0],
        kernel.cluster[1],
    )


def is_nvfp4_problem(
    is_scaled_gemm: bool,
    input_dtype_a: torch.dtype,
    input_dtype_b: torch.dtype | None,
    scale_type_a: Any | None,
    scale_type_b: Any | None,
) -> bool:
    """Whether this is the block-scaled NVFP4 recipe tuned below."""
    from torch.nn.functional import ScalingType  # type: ignore[attr-defined]

    return (
        is_scaled_gemm
        and input_dtype_a == torch.float4_e2m1fn_x2
        and input_dtype_b == torch.float4_e2m1fn_x2
        and scale_type_a == ScalingType.BlockWise1x16
        and scale_type_b == ScalingType.BlockWise1x16
    )


def use_swap_ab_for_scaled_gemm(
    input_dtype_a: torch.dtype,
    input_dtype_b: torch.dtype,
    scale_type_a: Any,
    scale_type_b: Any,
    logical_m: int,
    logical_n: int,
) -> bool:
    """Expose the transposed candidate family for decode-shaped NVFP4."""
    return config.nvgemm_swap_ab or (
        logical_m <= 256
        and logical_n >= 1024
        and is_nvfp4_problem(
            True,
            input_dtype_a,
            input_dtype_b,
            scale_type_a,
            scale_type_b,
        )
    )


def nvgemm_max_configs(is_nvfp4: bool, available: int) -> int:
    """Bound the measured candidate count while retaining the global cap."""
    general_limit = config.nvgemm_max_profiling_configs
    if not general_limit:
        return available
    if is_nvfp4:
        return min(general_limit, _NVFP4_MAX_PROFILING_CONFIGS)
    return general_limit


def nvgemm_cudagraph_unroll(
    is_scaled_gemm: bool,
    input_dtype_a: torch.dtype,
    input_dtype_b: torch.dtype | None,
    output_shape: torch.Size | list[int] | tuple[int, ...],
    scale_type_a: Any | None,
    scale_type_b: Any | None,
    allow_mixed_backend_unroll: bool = False,
) -> int:
    """Return one common CUDA-graph unroll for an NVGEMM problem.

    Candidate-dependent unrolls would make the fixed replay overhead differ
    across choices, so this policy depends only on the GEMM problem. The
    scoped value is limited to the decode-range NVFP4 cases measured in LLM
    inference. Other problems retain the global replay count so candidates in
    the same autotune decision stay comparable.
    """
    from torch._inductor.utils import _is_only_autotune_backend

    if (
        is_nvfp4_problem(
            is_scaled_gemm,
            input_dtype_a,
            input_dtype_b,
            scale_type_a,
            scale_type_b,
        )
        and len(output_shape) == 2
        and output_shape[0] <= 256
        and (_is_only_autotune_backend("NVGEMM") or allow_mixed_backend_unroll)
    ):
        return _NVFP4_AUTOTUNE_CUDAGRAPH_UNROLL
    return max(1, config.autotune_cudagraph_benchmarking_iters)


def nvgemm_cold_cache_shape(
    output_shape: torch.Size | list[int] | tuple[int, ...],
) -> bool:
    """Whether to rotate weights for the small-M inference regime."""
    return len(output_shape) == 2 and 0 < output_shape[0] <= 256


def _logical_nvfp4_k(packed_k: int) -> int:
    return 2 * packed_k


def use_nvfp4_pdl(logical_m: int, logical_n: int, packed_k: int) -> bool:
    """Whether an NVFP4 shape should use programmatic dependent launch."""
    if config.nvgemm_pdl == "1":
        return True
    if config.nvgemm_pdl != "auto":
        return False
    logical_k = _logical_nvfp4_k(packed_k)
    width_min = min(logical_n, logical_k)
    width_max = max(logical_n, logical_k)
    return (
        0 < logical_m <= _NVFP4_PDL_MAX_M
        and _NVFP4_PDL_MIN_WIDTH <= width_min
        and width_max <= _NVFP4_PDL_MAX_WIDTH
        # Hybrid models regress in this intermediate-width batch 33-64 band.
        and not (32 < logical_m <= 64 and 2048 < width_min < 4096)
    )


def use_nvfp4_late_pdl_wait(
    *, use_pdl: bool, sf_vec_size: int, logical_m: int, logical_k: int
) -> bool:
    """Whether descriptor setup should overlap the NVFP4 PDL wait."""
    return (
        use_pdl
        and sf_vec_size == 16
        and (
            (logical_m <= 32 and logical_k <= 4096)
            or (32 < logical_m <= 64 and logical_k <= 5120)
        )
    )


def kernel_uses_pdl(kernel) -> bool:
    return bool(
        getattr(
            getattr(kernel, "impl", None),
            "use_pdl",
            getattr(kernel.metadata.design, "use_pdl", False),
        )
    )


def prefer_pdl_kernels(
    non_efc_kernels: list[Any],
    efc_kernels: list[Any],
    use_pdl: bool,
) -> tuple[list[Any], list[Any]]:
    """Prefer generated PDL variants while retaining a safe base fallback."""
    if not use_pdl:
        return non_efc_kernels, efc_kernels
    pdl_non_efc_kernels = [
        kernel for kernel in non_efc_kernels if kernel_uses_pdl(kernel)
    ]
    pdl_efc_kernels = [kernel for kernel in efc_kernels if kernel_uses_pdl(kernel)]
    if not pdl_non_efc_kernels and not pdl_efc_kernels:
        return non_efc_kernels, efc_kernels
    return pdl_non_efc_kernels, pdl_efc_kernels


def _nvfp4_candidate_configs(
    *,
    logical_m: int,
    logical_n: int,
    packed_k: int,
    n_is_static: bool,
    swap_ab: bool,
) -> tuple[OrderedSet[ConfigKey], OrderedSet[ConfigKey], OrderedSet[ConfigKey]]:
    """Return primary, all-variant, and prefetch configs for one NVFP4 shape."""
    targeted: OrderedSet[ConfigKey] = OrderedSet()
    all_variants: OrderedSet[ConfigKey] = OrderedSet()
    prefetch: OrderedSet[ConfigKey] = OrderedSet()

    # nvMatmulHeuristics currently ranks only 128-wide MMA tiles for these
    # large-M projections. The 256x192 c2x2 family is consistently faster.
    if not swap_ab and logical_m >= 1024 and config.nvgemm_supplement_configs:
        config_key = (256, 192, 2, 2)
        targeted.add(config_key)
        all_variants.add(config_key)

    # Medium-M native-orientation QKV and projection winners.
    if not swap_ab and 64 < logical_m <= 256:
        targeted.update(
            (
                (128, 64, 1, 1),
                (128, 64, 1, 2),
                (128, 128, 1, 1),
                (128, 128, 1, 2),
            )
        )
        if logical_n <= 16384:
            prefetch.update(
                (
                    (128, 64, 1, 4),
                    (128, 128, 1, 2),
                    (128, 128, 1, 4),
                )
            )

    # Medium-M transposed projection winners from a complete candidate sweep.
    if swap_ab and n_is_static and 64 < logical_m <= 256 and logical_n >= 1024:
        targeted.update(((256, 128, 4, 1), (256, 64, 2, 2), (128, 64, 2, 2)))
        if logical_n <= 16384:
            prefetch.update(((256, 64, 2, 2), (256, 64, 4, 2), (128, 64, 2, 2)))

    # Batch 33-64 transposed projection winners.
    if swap_ab and n_is_static and 32 < logical_m <= 64 and logical_n >= 1024:
        targeted.add((128, 64, 1, 1))
        prefetch.update(((256, 64, 2, 2), (256, 64, 4, 2)))

    # Native batch 33-64 down-projection winner.
    if not swap_ab and 32 < logical_m <= 64 and logical_n >= 1024 and packed_k >= 3000:
        targeted.add((256, 64, 2, 2))

    # Decode-shaped transposed winners absent from the CUTLASS3 discovery set.
    if swap_ab and n_is_static and logical_m <= 32 and logical_n >= 1024:
        targeted.update(((128, 32, 1, 1), (128, 64, 1, 1), (128, 32, 4, 1)))
        logical_k = _logical_nvfp4_k(packed_k)
        if min(logical_n, logical_k) <= 4608 or max(logical_n, logical_k) <= 8192:
            targeted.add((128, 32, 2, 1))
            prefetch.add((128, 32, 4, 1))
        if logical_m <= 8:
            targeted.update(((128, 8, 1, 1), (128, 8, 4, 1), (128, 16, 1, 1)))

    if config.nvgemm_supplement_configs:
        targeted.update(_SUPPLEMENT_CONFIGS)

    return targeted, all_variants, prefetch


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
        *,
        is_nvfp4: bool = False,
        swap_ab: bool = False,
        fallback_count: int | None = None,
    ) -> list:
        """
        Filter and rank kernels using nvMatmulHeuristics.

        Matches on (tile_m, tile_n, tile_k, cluster_m, cluster_n).
        Returns kernels sorted by estimated runtime.

        If nvMatmulHeuristics is not installed or max_autotune is disabled,
        returns the first `count` kernels without heuristic ranking.

        Args:
            kernels: List of cutlass.operators.Operator objects
            inputs: MMKernelInputs with matrix shapes, dtypes, and strides
            count: Maximum number of kernels to return
            accumulator_type: Accumulator dtype

        Returns:
            Filtered list of kernels, sorted by estimated performance
        """
        _, symbolic_n, _ = inputs.mnk_symbolic()
        n_is_static = not has_free_symbols((symbolic_n,))
        if is_nvfp4 and not n_is_static:
            kernels = [
                kernel
                for kernel in kernels
                if getattr(kernel.metadata.design, "tile_shape", (0, 64))[1] >= 64
            ]

        def is_primary_variant(kernel) -> bool:
            return not getattr(kernel.metadata.design, "use_prefetch", False)

        primary_kernels = [kernel for kernel in kernels if is_primary_variant(kernel)]
        fallback_kernels = primary_kernels or kernels
        fallback_count = count if fallback_count is None else fallback_count
        if not self.should_run(inputs):
            return fallback_kernels[:fallback_count]

        m, n, k = inputs.mnk_hinted()
        logical_m, logical_n = (n, m) if swap_ab else (m, n)
        batch_size = inputs.batch_hinted()
        dtype_a = inputs.dtype(inputs._mat1_idx)
        dtype_b = inputs.dtype(inputs._mat2_idx)
        out_dtype = inputs.out_dtype()
        strides = inputs.strides_hinted()
        layout_a = "row" if strides[inputs._mat1_idx][-1] == 1 else "col"
        layout_b = "row" if strides[inputs._mat2_idx][-1] == 1 else "col"

        config_to_kernels = self._extract_config_to_kernels(kernels)

        if not config_to_kernels:
            log.debug(
                "Could not extract kernel configs, using first %d kernels",
                fallback_count,
            )
            return fallback_kernels[:fallback_count]

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
            batch_size,
            dtype_b=dtype_b,
            out_dtype=out_dtype,
        )

        if not heuristic_configs:
            log.debug(
                "No heuristic configs found, using first %d kernels", fallback_count
            )
            return fallback_kernels[:fallback_count]

        # Match kernels to each distinct heuristic config at its best estimate.
        config_runtimes: dict[ConfigKey, float] = {}
        for cfg in heuristic_configs:
            key = _make_config_key_from_heuristic(cfg)
            config_runtimes[key] = min(
                cfg.estimated_runtime, config_runtimes.get(key, float("inf"))
            )

        matched: list[tuple] = []

        for key, runtime in config_runtimes.items():
            kernels_for_key = config_to_kernels.get(key)
            if not kernels_for_key:
                continue
            for kernel in kernels_for_key:
                if is_primary_variant(kernel):
                    matched.append((kernel, runtime))

        if not matched:
            log.debug(
                "No kernels matched heuristic configs, using first %d kernels",
                fallback_count,
            )
            return fallback_kernels[:fallback_count]

        matched.sort(key=lambda x: x[1])
        selected = matched[:count]
        result = [k for k, _ in selected]

        targeted_configs: OrderedSet[ConfigKey] = OrderedSet()
        all_kernel_variants_configs: OrderedSet[ConfigKey] = OrderedSet()
        prefetch_configs: OrderedSet[ConfigKey] = OrderedSet()
        if is_nvfp4:
            (
                targeted_configs,
                all_kernel_variants_configs,
                prefetch_configs,
            ) = _nvfp4_candidate_configs(
                logical_m=logical_m,
                logical_n=logical_n,
                packed_k=k,
                n_is_static=n_is_static,
                swap_ab=swap_ab,
            )
        elif config.nvgemm_supplement_configs:
            targeted_configs.update(_SUPPLEMENT_CONFIGS)

        selected_keys = OrderedSet(
            [_make_config_key_from_kernel_design(k.metadata.design) for k in result]
        )
        for key, key_kernels in config_to_kernels.items():
            if key in all_kernel_variants_configs:
                for kernel in key_kernels:
                    if is_primary_variant(kernel) and kernel not in result:
                        result.append(kernel)
            elif key not in selected_keys and key in targeted_configs:
                primary = next(filter(is_primary_variant, key_kernels), None)
                if primary is not None:
                    result.append(primary)
            if key in prefetch_configs:
                prefetch = next(
                    (
                        kernel
                        for kernel in key_kernels
                        if getattr(kernel.metadata.design, "use_prefetch", False)
                    ),
                    None,
                )
                if prefetch is not None and prefetch not in result:
                    result.append(prefetch)

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
        matching the available cutlass.operators kernel tile/cluster shapes.
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
        batch_size: int = 1,
        dtype_b: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> list[HeuristicConfig]:
        """
        Get kernel configurations recommended by nvMatmulHeuristics.

        Uses validity callback to filter to cutlass.operators-compatible configs.
        """
        import nvMatmulHeuristics

        dtype_to_cublas = {
            torch.float64: "D",
            torch.float32: "S",
            torch.float16: "H",
            torch.bfloat16: "T",
            torch.float8_e4m3fn: "Q",
            torch.float8_e5m2: "R",
            torch.float4_e2m1fn_x2: "F4",
        }
        a_char = dtype_to_cublas.get(dtype_a, "H")
        b_char = dtype_to_cublas.get(dtype_b or dtype_a, a_char)
        out_char = dtype_to_cublas.get(out_dtype or dtype_a, a_char)
        acc_char = dtype_to_cublas.get(accumulator_type, "S")

        # nvMatmulHeuristics precision string formats:
        # - 3-letter {A}{B}{out}: used for standard GEMM and multi-char tokens (F4, BF)
        # - 5-letter {A}{B}{C}{compute}{D}: used for single-char FP8 types (Q, R, O)
        has_multichar = any(len(c) > 1 for c in (a_char, b_char, out_char))
        if has_multichar:
            precision = f"{a_char}{b_char}{out_char}"
        elif a_char != b_char or a_char in ("Q", "R", "O"):
            precision = f"{a_char}{b_char}{out_char}{acc_char}{out_char}"
        else:
            precision = f"{a_char}{acc_char}{out_char}"

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
