# flake8: noqa
from typing import Any
from typing_extensions import assert_type

from torch import randn, Tensor


# See ../pass/arithmetic_ops.py for more information

TENSOR, INT, FLOAT = randn(3), 2, 1.5

assert_type(
    INT & TENSOR,  # E: Unsupported operand types for & ("int" and "Tensor")  [operator]
    Any,
)
assert_type(
    INT | TENSOR,  # E: Unsupported operand types for | ("int" and "Tensor")  [operator]
    Any,
)
assert_type(
    INT ^ TENSOR,  # E: Unsupported operand types for ^ ("int" and "Tensor")  [operator]
    Any,
)

assert_type(
    FLOAT  # E: Unsupported operand types for & ("float" and "Tensor")  [operator]
    & TENSOR,
    Tensor,
)
assert_type(
    FLOAT  # E: Unsupported operand types for | ("float" and "Tensor")  [operator]
    | TENSOR,
    Tensor,
)
assert_type(
    FLOAT  # E: Unsupported operand types for ^ ("float" and "Tensor")  [operator]
    ^ TENSOR,
    Tensor,
)
