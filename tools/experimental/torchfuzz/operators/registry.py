"""Operator registry for mapping operation names to operator instances."""

from torchfuzz.operators.arg import ArgOperator
from torchfuzz.operators.argsort import ArgsortOperator
from torchfuzz.operators.base import Operator
from torchfuzz.operators.constant import ConstantOperator
from torchfuzz.operators.gather import GatherOperator
from torchfuzz.operators.index_select import IndexSelectOperator
from torchfuzz.operators.item import ItemOperator
from torchfuzz.operators.layout import (
    CatOperator,
    ChunkOperator,
    FlattenOperator,
    ReshapeOperator,
    SqueezeOperator,
    StackOperator,
    UnsqueezeOperator,
    ViewOperator,
)
from torchfuzz.operators.masked_select import MaskedSelectOperator
from torchfuzz.operators.matrix_multiply import (
    AddmmOperator,
    BmmOperator,
    MatmulOperator,
    MMOperator,
)
from torchfuzz.operators.nn_functional import (
    BatchNormOperator,
    DropoutOperator,
    ELUOperator,
    EmbeddingOperator,
    GELUOperator,
    GroupNormOperator,
    LayerNormOperator,
    LeakyReLUOperator,
    LinearOperator,
    MultiHeadAttentionForwardOperator,
    ReLUOperator,
    RMSNormOperator,
    ScaledDotProductAttentionOperator,
    SigmoidOperator,
    SiLUOperator,
    SoftmaxOperator,
    TanhOperator,
)
from torchfuzz.operators.nonzero import NonzeroOperator
from torchfuzz.operators.scalar_pointwise import (
    ScalarAddOperator,
    ScalarDivOperator,
    ScalarMulOperator,
    ScalarSubOperator,
)
from torchfuzz.operators.tensor_pointwise import (
    AddOperator,
    ClampOperator,
    CumsumOperator,
    DivOperator,
    MulOperator,
    SubOperator,
)
from torchfuzz.operators.unique import UniqueOperator


class OperatorRegistry:
    """Registry for managing operator instances."""

    def __init__(self):
        """Initialize the registry with default operators."""
        self._operators: dict[str, Operator] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """Register the default set of operators."""
        # Individual tensor pointwise operators (preferred)
        self.register(AddOperator())
        self.register(MulOperator())
        self.register(SubOperator())
        self.register(DivOperator())
        self.register(ClampOperator())
        self.register(CumsumOperator())

        # Individual scalar pointwise operators (preferred)
        self.register(ScalarAddOperator())
        self.register(ScalarMulOperator())
        self.register(ScalarSubOperator())
        self.register(ScalarDivOperator())

        # Leaf Input operators
        self.register(ConstantOperator())
        self.register(ArgOperator())

        # # Data-dependent operators
        self.register(NonzeroOperator())
        self.register(MaskedSelectOperator())
        self.register(GatherOperator())
        self.register(IndexSelectOperator())
        self.register(ArgsortOperator())
        self.register(ItemOperator())
        self.register(UniqueOperator())

        # Tensor layout operators
        self.register(ViewOperator())
        self.register(ReshapeOperator())
        self.register(FlattenOperator())
        self.register(SqueezeOperator())
        self.register(UnsqueezeOperator())
        self.register(CatOperator())
        self.register(StackOperator())
        self.register(ChunkOperator())

        # Matrix multiplication operators
        self.register(MMOperator())
        self.register(AddmmOperator())
        self.register(BmmOperator())
        self.register(MatmulOperator())

        # Neural network functional operators
        self.register(EmbeddingOperator())
        self.register(LinearOperator())
        self.register(ScaledDotProductAttentionOperator())
        self.register(MultiHeadAttentionForwardOperator())

        # Activation functions
        self.register(ReLUOperator())
        self.register(LeakyReLUOperator())
        self.register(ELUOperator())
        self.register(GELUOperator())
        self.register(SiLUOperator())
        self.register(SigmoidOperator())
        self.register(TanhOperator())
        self.register(SoftmaxOperator())

        # Normalization layers
        self.register(LayerNormOperator())
        self.register(RMSNormOperator())
        self.register(BatchNormOperator())
        self.register(GroupNormOperator())

        # Regularization
        self.register(DropoutOperator())

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
