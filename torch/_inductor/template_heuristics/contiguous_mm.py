from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ..ir import get_free_symbols
from ..kernel_inputs import KernelInputs, MMKernelInputs
from .base import TemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..ir import Layout


@register_template_heuristic("contiguous_mm", None, op_name="mm")
@register_template_heuristic("contiguous_addmm", None, op_name="addmm")
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip contiguous mm on not hip"""


@register_template_heuristic(
    "contiguous_mm", "cuda", register=torch.version.hip is not None, op_name="mm"
)
@register_template_heuristic(
    "contiguous_addmm", "cuda", register=torch.version.hip is not None, op_name="addmm"
)
class ContiguousMMHeuristics(TemplateConfigHeuristics):
    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        assert isinstance(kernel_inputs, MMKernelInputs), (
            f"{self.__class__.__name__} requires MMKernelInputs"
        )

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

        yield {}
