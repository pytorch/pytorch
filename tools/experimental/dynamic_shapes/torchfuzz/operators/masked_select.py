"""Masked select operator implementation."""

from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class MaskedSelectOperator(Operator):
    """Operator for selecting elements from a tensor based on a mask."""

    def __init__(self):
        super().__init__("masked_select")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.masked_select"

    def can_produce(self, output_spec: Spec) -> bool:
        """Masked select produces a 1D tensor with data-dependent size."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Output is always 1D with data-dependent size
        # Be very restrictive to avoid shape mismatches
        return (
            len(output_spec.size) == 1
            and output_spec.size[0] <= 10  # Reasonable size
            and output_spec.dtype not in [torch.bool]
        )  # Avoid bool outputs

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Generate input specs for masked_select operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("MaskedSelectOperator can only produce TensorSpec outputs")

        # Input tensor - can be any shape and type
        input_tensor_spec = TensorSpec(
            size=(2, 3),  # Fixed size for consistency
            stride=(3, 1),  # Contiguous
            dtype=output_spec.dtype,  # Match output dtype
        )

        # Mask tensor - must be boolean and broadcastable to input
        mask_spec = TensorSpec(
            size=(2, 3),  # Same size as input for simplicity
            stride=(3, 1),  # Contiguous
            dtype=torch.bool,
        )

        return [input_tensor_spec, mask_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for masked_select operation."""
        if len(input_names) != 2:
            raise ValueError("MaskedSelectOperator requires exactly two inputs")

        return (
            f"{output_name} = torch.masked_select({input_names[0]}, {input_names[1]})"
        )
