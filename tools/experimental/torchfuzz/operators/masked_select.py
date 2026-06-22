"""Masked select operator implementation."""

import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class MaskedSelectOperator(Operator):
    """Operator for selecting elements from a tensor based on a mask."""

    def __init__(self):
        super().__init__("masked_select")

    @property
    def torch_op_name(self) -> str | None:
        """Return the torch operation name."""
        return "torch.masked_select"

    def can_produce(self, output_spec: Spec) -> bool:
        """Masked select produces a 1D tensor; we'll synthesize inputs to match size."""
        return isinstance(output_spec, TensorSpec) and len(output_spec.size) == 1

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
        """Generate code for masked_select with synthesized inputs to match size.

        Constructs an input tensor and mask so that exactly k elements are selected,
        where k = output_spec.size[0]. No data-dependent guards.
        """
        if len(input_names) != 2:
            raise ValueError("MaskedSelectOperator requires exactly two inputs")
        if not isinstance(output_spec, TensorSpec) or len(output_spec.size) != 1:
            raise ValueError("MaskedSelectOperator requires 1D TensorSpec output")
        k = output_spec.size[0]
        # Build a 1D input of length >= k and a mask with first k positions True
        # Use input's device and output dtype to avoid mismatches
        return (
            f"_x_ms = torch.arange(max({k}, 1), device={input_names[0]}.device).to({input_names[0]}.dtype)\n"
            f"_mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)\n"
            f"_mask_ms[:{k}] = True\n"
            f"{output_name} = torch.masked_select(_x_ms, _mask_ms)"
        )
