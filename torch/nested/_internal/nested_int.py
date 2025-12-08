from typing import *  # noqa: F403

import torch
from torch.fx.experimental._constant_symnode import ConstantIntNode


__all__ = ["NestedIntNode"]


# Python version of aten/src/ATen/core/NestedIntSymNodeImpl.cpp
def _eq(lhs: Any, rhs: Any) -> bool:
    return (
        isinstance(lhs, NestedIntNode)
        and isinstance(rhs, NestedIntNode)
        and lhs.t_id == rhs.t_id
        and lhs.coeff == rhs.coeff
    )


def _ge(lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
        if lhs.t_id == rhs.t_id:
            return lhs.coeff >= rhs.coeff
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(lhs, NestedIntNode):
        if rhs.is_constant() and rhs.constant_int() <= 2:
            return True
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(rhs, NestedIntNode):
        if lhs.is_constant() and lhs.constant_int() < 2:
            return False
        raise ValueError("ge: relation is indeterminate")
    else:
        raise ValueError("inputs unsupported")


class NestedIntNode:
    def __init__(self, t_id: int, coeff: int) -> None:
        self.t_id = t_id
        self.coeff = coeff

    def nested_int_coeff(self) -> int:
        return self.coeff

    def maybe_as_int(self) -> Optional[int]:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return True

    def clone(self) -> "NestedIntNode":
        return self

    def _str(self) -> Any:
        if self.coeff == 1:
            return f"j{self.t_id}"
        return f"{self.coeff}*j{self.t_id}"

    def str(self) -> Any:
        return self._str()

    def __str__(self) -> Any:
        return self._str()

    def __repr__(self) -> Any:
        return self._str()

    def _graph_repr(self) -> Any:
        return self._str()

    def mul(self, other: Any) -> "NestedIntNode":
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.t_id, self.coeff * other)

    def eq(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_eq(self, other))

    def ne(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _eq(self, other))

    def gt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(other, self))

    def lt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(self, other))

    def le(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(other, self))

    def ge(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(self, other))

    def is_symbolic(self) -> bool:
        return False

    def nested_int(self) -> int:
        return self.t_id

    def is_constant(self) -> bool:
        return False

    def wrap_int(self, num: int) -> ConstantIntNode:
        assert type(num) is int
        return ConstantIntNode(num)
