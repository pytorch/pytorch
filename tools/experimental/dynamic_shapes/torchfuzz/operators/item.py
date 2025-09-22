"""Item operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


class ItemOperator(Operator):
    """Operator for extracting a scalar from a tensor."""

    def __init__(self):
        super().__init__("torch.ops.aten.item")

    def can_produce(self, output_spec: Spec) -> bool:
        """Item can only produce scalars."""
        return isinstance(output_spec, ScalarSpec)

    def supports_variable_inputs(self) -> bool:
        """Item operator does not support variable number of inputs."""
        return False

    def decompose(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Decompose scalar into a single-element tensor for item operation."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError("ItemOperator can only produce ScalarSpec outputs")

        # Create a tensor spec that can produce a scalar via .item()
        # Use a 1-D tensor with 1 element
        tensor_spec = TensorSpec(size=(1,), stride=(1,), dtype=output_spec.dtype)

        return [tensor_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for item operation."""
        if len(input_names) != 1:
            raise ValueError("ItemOperator requires exactly one input")

        return f"{output_name} = {input_names[0]}.item()"
