from __future__ import annotations

from typing import TYPE_CHECKING

from ..kernel_inputs import MMKernelInputs
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ..kernel_inputs import KernelInputs


class AddMMBiasExpansionConfigMixin(TemplateConfigHeuristics):
    """
    expand the bias to match the output size
    """

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        from ..ir import TensorBox
        from ..lowering import expand

        assert isinstance(kernel_inputs, MMKernelInputs)
        output_size = kernel_inputs.output_layout(flexible=False).size
        nodes = kernel_inputs.nodes()
        bias = nodes[0]
        bias = expand(TensorBox(bias), output_size)
        return MMKernelInputs(
            input_nodes=[bias, *nodes[1:]],
            scalars=kernel_inputs.scalars(),
            out_dtype=kernel_inputs.out_dtype(),
            mat1_idx=kernel_inputs._mat1_idx,
            mat2_idx=kernel_inputs._mat2_idx,
        )
