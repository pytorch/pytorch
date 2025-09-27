"""Base operator implementation."""

from abc import ABC, abstractmethod
from typing import Optional

from torchfuzz.tensor_fuzzer import Spec


class Operator(ABC):
    """Base class for all operators in torchfuzz."""

    def __init__(self, name: str):
        """Initialize operator with name."""
        self.name = name

    @property
    @abstractmethod
    def torch_op_name(self) -> Optional[str]:
        """
        Return the torch operation name this operator represents.

        Returns:
            Optional[str]: The torch operation name (e.g., "torch.ops.aten.add", "torch.nonzero").
                          Returns None for non-torch operations like "arg" and "constant".
        """

    @abstractmethod
    def can_produce(self, output_spec: Spec) -> bool:
        """Check if this operator can produce the given output spec."""

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """
        Get input specifications for fuzzing. By default, delegates to decompose.
        Leaf operators should override this to return an empty list.
        """
        return self.decompose(output_spec)

    @abstractmethod
    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for this operation."""

    def __str__(self) -> str:
        """String representation of the operator."""
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        """Repr representation of the operator."""
        return self.__str__()
