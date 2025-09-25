"""Operator registry for mapping operation names to operator instances."""

from typing import Optional

from torchfuzz.operators.arg import ArgOperator
from torchfuzz.operators.base import Operator
from torchfuzz.operators.constant import ConstantOperator
from torchfuzz.operators.item import ItemOperator
from torchfuzz.operators.scalar_pointwise import ScalarPointwiseOperator
from torchfuzz.operators.tensor_pointwise import TensorPointwiseOperator


class OperatorRegistry:
    """Registry for managing operator instances."""

    def __init__(self):
        """Initialize the registry with default operators."""
        self._operators: dict[str, Operator] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """Register the default set of operators."""
        self.register(TensorPointwiseOperator())
        self.register(ScalarPointwiseOperator())
        self.register(ItemOperator())
        self.register(ConstantOperator())
        self.register(ArgOperator())

    def register(self, operator: Operator):
        """Register an operator in the registry."""
        self._operators[operator.name] = operator

    def get(self, op_name: str) -> Optional[Operator]:
        """Get an operator by name."""
        # Handle special arg_ operations by mapping them to the ArgOperator
        if op_name.startswith("arg_"):
            return self._operators.get("arg")
        return self._operators.get(op_name)

    def list_operators(self) -> dict[str, Operator]:
        """List all registered operators."""
        return self._operators.copy()


# Global registry instance
_global_registry = OperatorRegistry()


def get_operator(op_name: str) -> Optional[Operator]:
    """Get an operator from the global registry."""
    return _global_registry.get(op_name)


def register_operator(operator: Operator):
    """Register an operator in the global registry."""
    _global_registry.register(operator)


def list_operators() -> dict[str, Operator]:
    """List all operators in the global registry."""
    return _global_registry.list_operators()
