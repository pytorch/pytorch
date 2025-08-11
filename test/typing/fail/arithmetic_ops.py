# flake8: noqa
from typing import Any
from typing_extensions import assert_type

from torch import randn, Tensor


# See ../pass/arithmetic_ops.py for more information

TENSOR, FLOAT = randn(3), 1.5

FLOAT & TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
FLOAT | TENSOR  # E: Unsupported operand types for | ("float" and "Tensor")
FLOAT ^ TENSOR  # E: Unsupported operand types for ^ ("float" and "Tensor")
# FIXME: false negatives (https://github.com/pytorch/pytorch/issues/155701)
#
# FLOAT << TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
# FLOAT >> TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
#
# TENSOR & FLOAT  # E: Unsupported operand types for & ("Tensor" and "float" )
# TENSOR | FLOAT  # E: Unsupported operand types for | ("Tensor" and "float" )
# TENSOR ^ FLOAT  # E: Unsupported operand types for ^ ("Tensor" and "float" )
# TENSOR << FLOAT  # E: Unsupported operand types for & ("Tensor" and "float")
# TENSOR >> FLOAT  # E: Unsupported operand types for & ("Tensor" and "float")
