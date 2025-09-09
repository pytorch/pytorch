from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor.kernel_inputs import MMKernelInputs

from ..ir import TensorBox
from ..kernel.mm_common import addmm_epilogue
from ..lowering import expand
from ..select_algorithm import realize_inputs
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ..kernel_inputs import KernelInputs


class AddMMBiasExpansionConfigMixin(TemplateConfigHeuristics):
    """
    handle bias input expansion
    """

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        assert isinstance(kernel_inputs, MMKernelInputs), (
            "MMKernelInputs expected for AddMMBiasExpansionConfigMixin"
        )
        nodes = kernel_inputs.nodes()
        bias = nodes[0]
        if not isinstance(bias, TensorBox):
            # bias has already been expanded
            return kernel_inputs
        layout = kernel_inputs.output_layout()
        bias = realize_inputs(expand(bias, layout.size))
        return MMKernelInputs(
            [bias, *nodes[1:]],
            scalars=kernel_inputs.scalars(),
            mat1_idx=kernel_inputs._mat1_idx,
            mat2_idx=kernel_inputs._mat2_idx,
        )


class AddMMConfigMixin(AddMMBiasExpansionConfigMixin):
    """
    Simple mixin to handle scalars for addmm like operators (addmm, baddbmm)
    """

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        assert op_name in [
            "addmm",
            "baddbmm",
        ], f"op_name={op_name} invalid for AddMMConfigMixin"
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "epilogue_fn": addmm_epilogue(kernel_inputs.out_dtype(), alpha, beta),
            "epilogue_fn_hash": str(
                ["addmm_epilogue", kernel_inputs.out_dtype(), alpha, beta]
            ),
            "prefix_args": 1,
        }
