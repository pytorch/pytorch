from enum import Enum
from typing import Optional, Union

import torch


class _EffectType(Enum):
    ORDERED = "Ordered"


_op_identifier = Union[
    str,
    "torch._ops.OpOverload",
    "torch._library.custom_ops.CustomOpDef",
    "torch._ops.HigherOrderOperator",
]


def register_effectful_op(
    op: _op_identifier,
    effect: Optional[_EffectType] = _EffectType.ORDERED,
):
    r"""
    Registers an effect for this operator.

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        effect: Effect type to register. By default it will register as being
        ORDERED, meaning function calls to this operator should not be reordered.
    """
    from torch._higher_order_ops.effects import _register_effectful_op

    _register_effectful_op(op, effect)
