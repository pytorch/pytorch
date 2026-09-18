from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor import config
from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
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
        output_tile_size = config.triton.decompose_k_min_output_tile_size
        m_is_static = not isinstance(m, sympy.Expr) or bool(m.is_number)
        n_is_static = not isinstance(n, sympy.Expr) or bool(n.is_number)
        if output_tile_size > 0 and m_is_static and n_is_static:
            device_properties = DeviceProperties.create(kernel_inputs.device())
            m_hint = int(m)
            n_hint = int(n)
            output_ctas = (
                2
                * ((m_hint + output_tile_size - 1) // output_tile_size)
                * ((n_hint + output_tile_size - 1) // output_tile_size)
            )
            min_k_split = (
                device_properties.multi_processor_count + output_ctas - 1
            ) // output_ctas
            k_splits = get_k_splits(m, n, k, min_k_split=min_k_split)
        else:
            k_splits = get_k_splits(m, n, k)

        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}
