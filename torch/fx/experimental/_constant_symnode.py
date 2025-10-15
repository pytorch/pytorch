from typing import *  # noqa: F403


# Python version of c10/core/ConstantSymNodeImpl.cpp
# This needs to exist because the Python version of nested int is not compatible
# with the C++ version of constant symnode.
class ConstantIntNode:
    def __init__(self, val: int):
        self.val = val

    def is_constant(self) -> bool:
        return True

    def maybe_as_int(self) -> int:
        return self.val

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def clone(self) -> "ConstantIntNode":
        return self

    def _str(self) -> str:
        return str(self.val)

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def mul(self, other: Any) -> Any:
        return other.mul(self)

    def eq(self, other: Any) -> Any:
        return other.eq(self)

    def ne(self, other: Any) -> Any:
        return other.ne(self)

    def gt(self, other: Any) -> Any:
        return other.lt(self)

    def lt(self, other: Any) -> Any:
        return other.gt(self)

    def le(self, other: Any) -> Any:
        return other.ge(self)

    def ge(self, other: Any) -> Any:
        return other.le(self)

    def is_symbolic(self) -> bool:
        return False

    def constant_int(self) -> int:
        return self.val
