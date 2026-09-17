from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

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
        device_properties = DeviceProperties.create(kernel_inputs.device())
        m_hint = V.graph.sizevars.guard_int(m)
        n_hint = V.graph.sizevars.guard_int(n)
        # Conservatively estimate two CTAs per 64x64 output tile.
        output_ctas = 2 * ((m_hint + 63) // 64) * ((n_hint + 63) // 64)
        min_k_split = (
            device_properties.multi_processor_count + output_ctas - 1
        ) // output_ctas
        k_splits = get_k_splits(
            m,
            n,
            k,
            min_k_split=min_k_split,
        )

        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}
