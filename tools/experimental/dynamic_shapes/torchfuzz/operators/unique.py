"""Unique operator implementation."""

from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class UniqueOperator(Operator):
    """Operator for finding unique elements in a tensor."""

    def __init__(self):
        super().__init__("unique")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.unique"

    def can_produce(self, output_spec: Spec) -> bool:
        """Unique produces a 1D tensor with data-dependent size."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Output is always 1D with data-dependent size
        # Be very restrictive to avoid shape mismatches
        return (
            len(output_spec.size) == 1
            and output_spec.size[0] <= 10  # Reasonable size
            and output_spec.dtype not in [torch.bool]
        )  # Avoid bool outputs

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Generate input spec for unique operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("UniqueOperator can only produce TensorSpec outputs")

        # Input can be any tensor - unique will flatten and find unique values
        input_spec = TensorSpec(
            size=(2, 3),  # Fixed size for consistency
            stride=(3, 1),  # Contiguous
            dtype=output_spec.dtype,  # Match output dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for unique operation."""
        if len(input_names) != 1:
            raise ValueError("UniqueOperator requires exactly one input")

        return f"{output_name} = torch.unique({input_names[0]})"
