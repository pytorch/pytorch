"""Operator registry for mapping operation names to operator instances."""

from typing import Dict, Optional
from torchfuzz.operators.base import Operator
from torchfuzz.operators.add import AddOperator
from torchfuzz.operators.mul import MulOperator
from torchfuzz.operators.item import ItemOperator
from torchfuzz.operators.scalar_add import ScalarAddOperator
from torchfuzz.operators.scalar_multiply import ScalarMultiplyOperator
from torchfuzz.operators.constant import ConstantOperator
from torchfuzz.operators.arg import ArgOperator


class OperatorRegistry:
    """Registry for managing operator instances."""

    def __init__(self):
        """Initialize the registry with default operators."""
        self._operators: Dict[str, Operator] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """Register the default set of operators."""
        self.register(AddOperator())
        self.register(MulOperator())
        self.register(ItemOperator())
        self.register(ScalarAddOperator())
        self.register(ScalarMultiplyOperator())
        self.register(ConstantOperator())
        self.register(ArgOperator())

    def register(self, operator: Operator):
        """Register an operator in the registry."""
        self._operators[operator.name] = operator

    def get(self, op_name: str) -> Optional[Operator]:
        """Get an operator by name."""
        return self._operators.get(op_name)

    def list_operators(self) -> Dict[str, Operator]:
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


def list_operators() -> Dict[str, Operator]:
    """List all operators in the global registry."""
    return _global_registry.list_operators()
