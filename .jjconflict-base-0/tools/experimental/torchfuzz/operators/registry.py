"""Operator registry for mapping operation names to operator instances."""

import importlib
import inspect
from types import ModuleType

from torchfuzz.operators.base import Operator


_OPERATOR_MODULES: list[str] = [
    "activations",
    "arg",
    "argsort",
    "bitwise",
    "comparison",
    "constant",
    "elementwise_math",
    "gather",
    "index_select",
    "item",
    "layout",
    "logical",
    "loss_functions",
    "manipulation_indexing",
    "masked_select",
    "matrix_multiply",
    "nn_functional",
    "nonzero",
    "reduction",
    "scalar_pointwise",
    "special_functions",
    "tensor_creation",
    "tensor_pointwise",
    "unique",
]


def _concrete_operator_classes(module: ModuleType) -> list[type[Operator]]:
    """Discover all concrete Operator subclasses defined in a module.

    Filters out:
    - Non-Operator classes
    - The Operator ABC itself
    - Re-exported classes (whose __module__ doesn't match)
    - Abstract classes
    - Classes with names ending in "Base" (reserved for abstract bases)
    """
    out: list[type[Operator]] = []
    for _name, cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(cls, Operator):
            continue
        if cls is Operator:
            continue
        if cls.__module__ != module.__name__:
            continue
        if inspect.isabstract(cls):
            continue
        if cls.__name__.endswith("Base"):
            continue
        out.append(cls)
    return out


class OperatorRegistry:
    """Registry for managing operator instances."""

    def __init__(self):
        """Initialize the registry with default operators."""
        self._operators: dict[str, Operator] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """Register the default set of operators via introspection."""
        for module_name in _OPERATOR_MODULES:
            module = importlib.import_module(
                f".{module_name}", package="torchfuzz.operators"
            )
            for cls in _concrete_operator_classes(module):
                self.register(cls())  # pyrefly: ignore[missing-argument]

    def register(self, operator: Operator):
        """Register an operator in the registry."""
        self._operators[operator.name] = operator

    def get(self, op_name: str) -> Operator | None:
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


def get_operator(op_name: str) -> Operator | None:
    """Get an operator from the global registry."""
    return _global_registry.get(op_name)


def register_operator(operator: Operator):
    """Register an operator in the global registry."""
    _global_registry.register(operator)


def list_operators() -> dict[str, Operator]:
    """List all operators in the global registry."""
    return _global_registry.list_operators()


def set_operator_weight(op_name: str, weight: float) -> None:
    """Set the selection weight for a specific operator.

    Args:
        op_name: The registered operator name (e.g., "add", "arg") OR fully-qualified torch op
                 (e.g., "torch.nn.functional.relu", "torch.matmul")
        weight: New relative selection weight (must be > 0)
    """
    if weight <= 0:
        raise ValueError("Operator weight must be > 0")

    # Try by registry key
    op = _global_registry.get(op_name)
    if op is not None:
        op.weight = float(weight)
        return

    # Fallback: try to locate by fully-qualified torch op name
    for candidate in _global_registry.list_operators().values():
        if getattr(candidate, "torch_op_name", None) == op_name:
            candidate.weight = float(weight)
            return

    raise KeyError(f"Operator '{op_name}' not found by registry name or torch op name")


def set_operator_weights(weights: dict[str, float]) -> None:
    """Bulk-update operator weights from a mapping of name -> weight."""
    for name, w in weights.items():
        set_operator_weight(name, w)


def set_operator_weight_by_torch_op(torch_op_name: str, weight: float) -> None:
    """Set operator weight by fully-qualified torch op name."""
    if weight <= 0:
        raise ValueError("Operator weight must be > 0")
    for candidate in _global_registry.list_operators().values():
        if getattr(candidate, "torch_op_name", None) == torch_op_name:
            candidate.weight = float(weight)
            return
    raise KeyError(f"Torch op '{torch_op_name}' not found in registry")


def set_operator_weights_by_torch_op(weights: dict[str, float]) -> None:
    """Bulk-update weights by fully-qualified torch op names."""
    for name, w in weights.items():
        set_operator_weight_by_torch_op(name, w)
