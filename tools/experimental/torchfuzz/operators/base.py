"""Base operator implementation."""

from abc import ABC, abstractmethod

from torchfuzz.tensor_fuzzer import Spec


class Operator(ABC):
    """Base class for all operators in torchfuzz."""

    def __init__(self, name: str, weight: float = 1.0):
        """Initialize operator with name and optional selection weight.

        Args:
            name: Unique operator name used in the registry
            weight: Relative selection weight when sampling among compatible operators
                    (default 1.0). Higher values increase selection likelihood.
        """
        self.name = name
        self.weight: float = float(weight)

    @property
    @abstractmethod
    def torch_op_name(self) -> str | None:
        """
        Return the torch operation name this operator represents.

        Returns:
            Optional[str]: The torch operation name (e.g., "torch.ops.aten.add", "torch.nonzero").
                          Returns None for non-torch operations like "arg" and "constant".
        """
        raise NotImplementedError("Subclasses must implement torch_op_name")

    @abstractmethod
    def can_produce(self, output_spec: Spec) -> bool:
        """Check if this operator can produce the given output spec."""
        raise NotImplementedError("Subclasses must implement can_produce()")

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """
        Get input specifications for fuzzing.

        Subclasses must implement this to return a list of input Specs that,
        when used with this operator, can produce the given output_spec. Leaf
        operators should return an empty list.
        """
        raise NotImplementedError("Subclasses must implement fuzz_inputs_specs()")

    @abstractmethod
    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for this operation."""
        raise NotImplementedError("Subclasses must implement codegen()")

    def get_weight(
        self,
        *,
        target_spec: Spec | None = None,
        depth: int | None = None,
        stack_size: int | None = None,
        template: str | None = None,
    ) -> float:
        """
        Return the selection weight for this operator.

        Subclasses may override to implement context-sensitive weighting.
        The default implementation returns the static attribute `self.weight`.
        """
        return self.weight

    def __str__(self) -> str:
        """String representation of the operator."""
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        """Repr representation of the operator."""
        return self.__str__()
