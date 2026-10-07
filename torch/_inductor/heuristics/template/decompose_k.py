from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor import config
from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.mm import decompose_k_subgraph_template
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...runtime.hints import DeviceProperties
from ...utils import ceildiv, get_k_splits
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator

    import torch


def decompose_k_split_bounds(device: torch.device) -> tuple[int, int]:
    """Return (min_output_tiles, max_workspace_bytes) for decompose-K splits.

    0 means unbounded. Tuned by timing every legal split against ATen on B200 and
    H100 and checking that no split that beat ATen gets filtered out.
    """
    device_properties = DeviceProperties.create(device)
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
# Runtime enablement is controlled by config.triton.num_decompose_k_splits (0 disables)
@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)
# by either adding specific register_template_heuristic tags, or setting the
# device to None (enabled on all devices)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
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

        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}
