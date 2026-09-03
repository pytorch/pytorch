from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor import config
from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.decompose_k import (
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    blackwell_decompose_k_subgraph_template,
)
from ...kernel.mm import decompose_k_subgraph_template
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...runtime.hints import DeviceProperties
from ...utils import get_k_splits
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


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
        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}


@register_template_heuristic(
    blackwell_decompose_k_subgraph_template.uid,
    None,
    op_name="mm",
)
class EmptyBlackwellDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """Skip the experimental partial template outside NVIDIA CUDA."""


@register_template_heuristic(
    blackwell_decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
class BlackwellDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """Generate one bounded 1CTA and one bounded 2CTA complete-plan seed."""

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        if not config.triton.enable_blackwell_decompose_k_partial:
            return
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        if any(
            len(get_free_symbols(value, unbacked_only=True)) > 0
            for value in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        ):
            return

        device = DeviceProperties.create(kernel_inputs.device())
        if device.type != "cuda" or device.major != 10:
            return
        m_sym, n_sym, k_sym = kernel_inputs.mnk_symbolic()
        m, n, k = int(m_sym), int(n_sym), int(k_sym)

        config_indices = [0, 3]
        if m > 128:
            config_indices.extend((1, 4) if n <= 128 else (2, 5))

        for config_index in config_indices:
            partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
            m_tiles = math.ceil(m / partial_config.block_m)
            if partial_config.two_ctas:
                m_tiles = math.ceil(m_tiles / 2) * 2
            output_tiles = m_tiles * math.ceil(n / partial_config.block_n)
            waves = 1 if partial_config.two_ctas else 2
            k_split = max(
                2,
                round(waves * device.multi_processor_count / output_tiles),
            )
            k_split = min(k_split, k // partial_config.block_k)
            while k_split >= 2:
                k_part = (
                    math.ceil(math.ceil(k / k_split) / partial_config.block_k)
                    * partial_config.block_k
                )
                workspace_bytes = k_split * m_tiles * partial_config.block_m * n * 4
                if (k_split - 1) * k_part < k and workspace_bytes <= 128 * 1024**2:
                    break
                k_split -= 1
            if k_split >= 2:
                yield {"k_split": k_split, "config_index": config_index}
