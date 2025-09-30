"""Nonzero operator implementation."""

from typing import Optional

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class NonzeroOperator(Operator):
    """Operator for finding nonzero elements in a tensor."""

    def __init__(self):
        super().__init__("nonzero")

    @property
    def torch_op_name(self) -> Optional[str]:
        """Return the torch operation name."""
        return "torch.nonzero"

    def can_produce(self, output_spec: Spec) -> bool:
        """Nonzero produces a tensor with shape (n_nonzero, n_dims)."""
        if not isinstance(output_spec, TensorSpec):
            return False

        # Output shape is (n_nonzero, n_dims) where both are data-dependent
        # We can only produce integer tensors (indices) and only 2D tensors
        # Restrict to very specific shapes to avoid shape mismatches
        return (
            output_spec.dtype in [torch.int64, torch.long]
            and len(output_spec.size) == 2
            and output_spec.size[1] <= 4
        )  # Reasonable input dimensionality

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Generate input spec for nonzero operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("NonzeroOperator can only produce TensorSpec outputs")

        # Input can be any tensor type that supports comparison with zero
        # Use boolean tensors for simplicity to ensure some nonzero elements
        input_spec = TensorSpec(
            size=(3, 4),  # Fixed size that will have some nonzero elements
            stride=(4, 1),  # Contiguous
            dtype=torch.bool,  # Boolean tensors are good for nonzero testing
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for nonzero operation."""
        if len(input_names) != 1:
            raise ValueError("NonzeroOperator requires exactly one input")

        return f"{output_name} = torch.nonzero({input_names[0]})"
