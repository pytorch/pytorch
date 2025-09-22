"""Base operator implementation."""

from abc import ABC, abstractmethod

from torchfuzz.tensor_fuzzer import Spec


class Operator(ABC):
    """Base class for all operators in torchfuzz."""

    def __init__(self, name: str):
        """Initialize operator with name."""
        self.name = name

    @abstractmethod
    def can_produce(self, output_spec: Spec) -> bool:
        """Check if this operator can produce the given output spec."""

    @abstractmethod
    def supports_variable_inputs(self) -> bool:
        """Check if this operator supports variable number of inputs."""

    @abstractmethod
    def decompose(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose output spec into input specs."""

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
