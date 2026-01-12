from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from ..ir import get_free_symbols
from ..kernel.mm import (
    addmm_contiguous_subgraph_template,
    mm_contiguous_subgraph_template,
)
from ..kernel_inputs import KernelInputs, MMKernelInputs
from ..utils import use_contiguous
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, None, op_name="addmm"
)
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip contiguous mm on not cuda"""


@register_template_heuristic(
    mm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="mm",
)
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="addmm",
)
class ContiguousMMHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
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
        mat2 = kernel_inputs.mat1mat2()[1]
        if mat2.get_layout().is_contiguous():
            # no need for contiguous decomposition
            return
        m, n, k = kernel_inputs.mnk_symbolic()
        if not use_contiguous(m, n, k):
            return
        yield {}
