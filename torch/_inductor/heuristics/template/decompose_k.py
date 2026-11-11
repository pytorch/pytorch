from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor import config
from torch._inductor.heuristics.registry import register_template_heuristic
from torch.utils._ordered_set import OrderedSet

from ...ir import get_free_symbols
from ...kernel.decompose_k import (
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    decompose_k_subgraph_template,
    effective_two_ctas,
)
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...runtime.hints import DeviceProperties
from ...utils import (
    _rocm_decompose_k_auto,
    ceildiv,
    get_k_splits,
    use_triton_blackwell_tma_template,
)
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .triton import BaseConfig, CUDAConfigHeuristic


if TYPE_CHECKING:
    from collections.abc import Generator

    import torch


def decompose_k_split_bounds(device: torch.device) -> tuple[int, int]:
    """Return (min_output_tiles, max_workspace_bytes) for decompose-K splits.

    0 means unbounded. Tuned by timing every legal split against ATen on B200,
    H100 and MI350X and checking that no split that beat ATen gets filtered out.
    """
    device_properties = DeviceProperties.create(device)
    if device_properties.type == "hip" and _rocm_decompose_k_auto():
        # MI350X default (-1 splits): at least 32 output tiles and at most 64 MiB of
        # FP32 workspace. An explicit split count is not pruned, as before.
        return 32, 64 * 1024 * 1024
    if device_properties.type != "cuda":
        return 0, 0
    if (device_properties.major or 0) >= 10:
        # SM100+: ensure splits fill at least one wave, assuming two CTAs per output
        # tile (decompose_k_min_output_tile_size).
        return (device_properties.multi_processor_count + 1) // 2, 0
    # SM90 and earlier: prune splits that use more than 8 MiB of FP32 workspace (no
    # split that beat ATen used more than ~5 MiB) or launch fewer than 8 output tiles.
    return 8, 8 * 1024 * 1024


def filter_decompose_k_splits(
    k_splits: list[int],
    m: int,
    n: int,
    tile_size: int,
    min_output_tiles: int,
    max_workspace_bytes: int,
) -> list[int]:
    """Drop split choices that rarely win: too few output tiles or a large workspace.

    Only removes candidates, and never all of them.
    """
    fits = [
        split
        for split in k_splits
        if max_workspace_bytes <= 0 or split * m * n * 4 <= max_workspace_bytes
    ]
    output_tiles = ceildiv(m, tile_size) * ceildiv(n, tile_size)
    kept = [split for split in fits if split * output_tiles >= min_output_tiles]
    if kept:
        return kept
    # Rather than dropping decompose-K, keep the split closest to the tile bound that
    # fits the workspace, else the one with the least workspace.
    if fits:
        return [max(fits)]
    return [min(k_splits)] if k_splits else []


@register_template_heuristic(decompose_k_subgraph_template.uid, None, op_name="mm")
class EmptyDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip decompose k on anything not cuda"""


@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "xpu",
    op_name="mm",
)
# Register on CUDA (both NVIDIA and ROCm/HIP)
# Runtime enablement is controlled by config.triton.num_decompose_k_splits (0 disables,
# -1 resolves per device in utils.decompose_k_num_splits)
@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)
# by either adding specific register_template_heuristic tags, or setting the
# device to None (enabled on all devices)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """Generate backend-specific decompose-K partial-BMM configurations."""

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")

        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return

        m, n, k = kernel_inputs.mnk_symbolic()
        bmm_backends = OrderedSet(
            backend.strip().upper()
            for backend in config.triton.decompose_k_bmm_backends.split(",")
        )
        k_splits = get_k_splits(m, n, k)
        tile_size = config.triton.decompose_k_min_output_tile_size
        if (
            tile_size > 0
            and config.max_autotune_gemm_search_space != "EXHAUSTIVE"
            and all(not isinstance(x, sympy.Expr) or x.is_number for x in (m, n, k))
        ):
            min_output_tiles, max_workspace_bytes = decompose_k_split_bounds(
                kernel_inputs.device()
            )
            k_splits = filter_decompose_k_splits(
                k_splits,
                int(m),
                int(n),
                tile_size,
                min_output_tiles,
                max_workspace_bytes,
            )
        exact_k_splits = [
            k_split
            for k_split in k_splits
            if V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            )
        ]

        if "ATEN" in bmm_backends:
            for k_split in exact_k_splits:
                yield {"k_split": k_split, "bmm_backend": "aten"}

        # Return before the TMA checks below, which install guards.
        if "TRITON" not in bmm_backends or not exact_k_splits:
            return

        mat1, mat2 = kernel_inputs.mat1mat2()
        layout = kernel_inputs.output_layout()
        if not use_triton_blackwell_tma_template(
            mat1,
            mat2,
            output_layout=layout,
            add_guards=True,
        ):
            return

        # The partial template uses compile-time descriptor geometry, so backed
        # dynamic dimensions must be specialized explicitly. Unbacked symbols
        # were rejected above.
        m_hint, n_hint, k_hint = V.graph.sizevars.guard_int_seq((m, n, k))
        config_indices = [0, 3]
        if m_hint > 128:
            config_indices.extend((1, 4) if n_hint <= 128 else (2, 5))
        # Same opt-in shared-memory pruning as the mm heuristics.
        exceeds_smem = None
        if config.max_autotune_prune_choices_based_on_shared_mem:
            exceeds_smem = CUDAConfigHeuristic()._get_exceeding_shared_memory_checker(
                has_sm_layout_conversion=False, layout_conversion_byte_size=0
            )
        dtype_size = mat1.get_dtype().itemsize

        for k_split in exact_k_splits:
            for config_index in config_indices:
                partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
                if exceeds_smem is not None and exceeds_smem(
                    BaseConfig(
                        partial_config.block_m,
                        partial_config.block_n,
                        partial_config.block_k,
                        partial_config.num_stages,
                        partial_config.num_warps,
                    ),
                    dtype_size,
                ):
                    continue
                m_tiles = math.ceil(m_hint / partial_config.block_m)
                if effective_two_ctas(partial_config):
                    m_tiles = math.ceil(m_tiles / 2) * 2
                k_part = (
                    math.ceil(math.ceil(k_hint / k_split) / partial_config.block_k)
                    * partial_config.block_k
                )
                workspace_bytes = (
                    k_split * m_tiles * partial_config.block_m * n_hint * 4
                )
                if (k_split - 1) * k_part < k_hint and workspace_bytes <= 128 * 1024**2:
                    yield {
                        "k_split": k_split,
                        "bmm_backend": "triton",
                        "bmm_config_index": config_index,
                    }
