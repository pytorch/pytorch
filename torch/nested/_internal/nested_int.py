from typing import *  # noqa: F403

import torch
from torch.fx.experimental.constant_symnode import ConstantIntNode
from torch.nested._internal.tensor_registry import register_tensor, try_get_int
from torch.nested._internal.utils import apply_func


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


def _get_tensor_id(t) -> int:
    ret = None

    def func(t):
        nonlocal ret

        if try_get_int(t) is None:
            ret = register_tensor(t)
        else:
            ret = try_get_int(t)

    apply_func(t, func, only_source_fields=True)
    assert ret is not None
    return ret


class NestedIntNode:
    def __init__(self, cache: torch.Tensor, coeff: int):
        self.cache = cache
        # Wait cache CAN mutate in eager! but t_id should change anyway.
        self.t_id = _get_tensor_id(cache)
        self.coeff = coeff

    def nested_int_coeff(self) -> int:
        return self.coeff

    def nested_int_cache(self) -> Any:
        return self.cache

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

    def _str(self) -> str:
        if self.coeff == 1:
            return f"j{self.t_id}"
        return f"{self.coeff}*j{self.t_id}"

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def mul(self, other: Any) -> "NestedIntNode":
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.cache, self.coeff * other)

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

def get_metadata(x: torch.SymInt):
    if isinstance(x.node, NestedIntNode):
        return x.node.nested_int_cache()
    else:
        return x.node.hint.node.nested_int_cache()
