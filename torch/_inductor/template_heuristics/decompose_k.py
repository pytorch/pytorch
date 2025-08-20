from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

import torch

from ..ir import Layout
from ..kernel_inputs import MMKernelInputs

from ..utils import get_k_splits
from ..virtualized import V
from .registry import register_template_heuristic

if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(
    "decompose_k", "cuda", register=torch.version.hip is None, op_name="mm"
)
class DecomposeKConfigHeuristics:
    def get_template_configs(
        self,
        kernel_inputs: MMKernelInputs,
        layout: Layout,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        m, n, k = kernel_inputs.mnk_symbolic()
        k_splits = get_k_splits(m, n, k)
        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}
