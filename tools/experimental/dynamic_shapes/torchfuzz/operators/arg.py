"""Arg operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec


class ArgOperator(Operator):
    """Operator for function arguments/parameters."""

    def __init__(self):
        super().__init__("arg")

    def can_produce(self, output_spec: Spec) -> bool:
        """Arg can produce any type of output."""
        return True

    def decompose(self, output_spec: Spec, num_inputs: int = 0) -> list[Spec]:
        """Arg requires no inputs."""
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for arg operation."""
        # The actual argument name assignment will be handled separately
        # in the codegen.py when processing arg operations
        return f"# {output_name} will be assigned an argument value"
