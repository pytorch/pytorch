"""Item operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


class ItemOperator(Operator):
    """Operator for converting 0-d tensor to scalar."""

    def __init__(self):
        super().__init__("item")

    @property
    def torch_op_name(self) -> str | None:
        """Item is a tensor method, not a direct torch operation."""
        return None

    def can_produce(self, output_spec: Spec) -> bool:
        """Item produces scalars from 0-d tensors."""
        return isinstance(output_spec, ScalarSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        """Decompose scalar into a single-element tensor for item operation."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError("ItemOperator can only produce ScalarSpec outputs")

        # Create a tensor spec that can produce a scalar via .item()
        # Use a 0-D tensor (scalar tensor) to ensure .item() works reliably
        tensor_spec = TensorSpec(size=(), stride=(), dtype=output_spec.dtype)

        return [tensor_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for item operation."""
        if len(input_names) != 1:
            raise ValueError("ItemOperator requires exactly one input")

        return f"{output_name} = {input_names[0]}.item()"
