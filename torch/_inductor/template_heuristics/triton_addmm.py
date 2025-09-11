from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..kernel.mm_common import addmm_epilogue
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ..ir import Layout
    from ..kernel_inputs import KernelInputs


class AddMMConfigMixin(TemplateConfigHeuristics):
    """
    Simple mixin to handle scalars for addmm like operators (addmm, baddbmm)
    """

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, layout, op_name)
        assert op_name in [
            "addmm",
            "baddbmm",
        ], f"op_name={op_name} invalid for AddMMConfigMixin"
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "epilogue_fn": addmm_epilogue(layout.dtype, alpha, beta),
            "epilogue_fn_hash": str(["addmm_epilogue", layout.dtype, alpha, beta]),
            "prefix_args": 1,
        }
