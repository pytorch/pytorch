"""Unique operator implementation."""

from typing import Optional

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
        """Unique can produce 1D tensor outputs of arbitrary length without guards.

        We will synthesize an input with exactly the desired number of unique
        elements so that torch.unique returns the target size deterministically.
        """
        return isinstance(output_spec, TensorSpec) and len(output_spec.size) == 1

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
        """Generate code for unique with deterministic target size input (no guards)."""
        if len(input_names) != 1:
            raise ValueError("UniqueOperator requires exactly one input")
        # Desired output length
        desired_len = output_spec.size[0] if isinstance(output_spec, TensorSpec) else 0
        # Synthesize an input tensor with exactly desired_len distinct elements
        # Use the input's device to avoid device mismatches
        return (
            f"_inp_unique = torch.arange({desired_len}, device={input_names[0]}.device).to({input_names[0]}.dtype)\n"
            f"{output_name} = torch.unique(_inp_unique)"
        )
