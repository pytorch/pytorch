"""Torchfuzz operators module."""

from torchfuzz.operators.arg import ArgOperator
from torchfuzz.operators.base import Operator
from torchfuzz.operators.constant import ConstantOperator
from torchfuzz.operators.index_select import IndexSelectOperator
from torchfuzz.operators.item import ItemOperator
from torchfuzz.operators.layout import (
    FlattenOperator,
    ReshapeOperator,
    SplitOperator,
    SqueezeOperator,
    UnsqueezeOperator,
    ViewOperator,
)
from torchfuzz.operators.matrix_multiply import (
    AddmmOperator,
    BmmOperator,
    MatmulOperator,
    MMOperator,
)
from torchfuzz.operators.nn_functional import (
    DropoutOperator,
    EmbeddingOperator,
    LayerNormOperator,
    LinearOperator,
    MultiHeadAttentionForwardOperator,
    ReLUOperator,
    ScaledDotProductAttentionOperator,
    SoftmaxOperator,
)
from torchfuzz.operators.registry import (
    get_operator,
    list_operators,
    register_operator,
    set_operator_weight,
    set_operator_weight_by_torch_op,
    set_operator_weights,
    set_operator_weights_by_torch_op,
)
from torchfuzz.operators.scalar_pointwise import (
    ScalarAddOperator,
    ScalarDivOperator,
    ScalarMulOperator,
    ScalarPointwiseOperator,
    ScalarSubOperator,
)
from torchfuzz.operators.tensor_pointwise import (
    AddOperator,
    DivOperator,
    MulOperator,
    PointwiseOperator,
    SubOperator,
)


__all__ = [
    "Operator",
    "PointwiseOperator",
    "AddOperator",
    "MulOperator",
    "SubOperator",
    "DivOperator",
    "ScalarPointwiseOperator",
    "ScalarAddOperator",
    "ScalarMulOperator",
    "ScalarSubOperator",
    "ScalarDivOperator",
    "ItemOperator",
    "ConstantOperator",
    "ArgOperator",
    "IndexSelectOperator",
    "ViewOperator",
    "ReshapeOperator",
    "FlattenOperator",
    "SqueezeOperator",
    "UnsqueezeOperator",
    "SplitOperator",
    "MMOperator",
    "AddmmOperator",
    "BmmOperator",
    "MatmulOperator",
    "EmbeddingOperator",
    "LinearOperator",
    "MultiHeadAttentionForwardOperator",
    "ReLUOperator",
    "ScaledDotProductAttentionOperator",
    "SoftmaxOperator",
    "DropoutOperator",
    "LayerNormOperator",
    "get_operator",
    "register_operator",
    "list_operators",
    "set_operator_weight",
    "set_operator_weights",
    "set_operator_weight_by_torch_op",
    "set_operator_weights_by_torch_op",
]
