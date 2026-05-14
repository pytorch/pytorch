from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable
    from typing import Literal

from .dispatch import dispatch
from .utils import hashable


_global_logic_variables: set[Hashable] = set()
_glv = _global_logic_variables


class Var:
    """Logic Variable"""

    _id = 1

    def __new__(cls, *token: Hashable) -> Var:  # noqa: PYI034
        if len(token) == 0:
            token = f"_{Var._id}"  # type: ignore[assignment]
            Var._id += 1
        elif len(token) == 1:
            token = token[0]  # pyrefly: ignore[bad-assignment]

        obj = object.__new__(cls)
        obj.token = token  # type: ignore[attr-defined]
        return obj

    def __str__(self) -> str:
        return "~" + str(self.token)  # type: ignore[attr-defined]

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.token == other.token  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self), self.token))  # type: ignore[attr-defined]


def var() -> Callable[..., Var]:
    return lambda *args: Var(*args)


def vars() -> Callable[[int], list[Callable[..., Var]]]:
    return lambda n: [var() for i in range(n)]


@dispatch(Var)
def isvar(v: Var) -> Literal[True]:
    return True


isvar


@dispatch(object)  # type: ignore[no-redef]
def isvar(o: object) -> bool:
    return bool(_glv and hashable(o) and o in _glv)


@contextmanager
def variables(*variables: Hashable) -> Generator[None, None, None]:
    """
    Context manager for logic variables

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> from __future__ import with_statement
        >>> with variables(1):
        ...     print(isvar(1))
        True
        >>> print(isvar(1))
        False
        >>> # Normal approach
        >>> from unification import unify
        >>> x = var("x")
        >>> unify(x, 1)
        {~x: 1}
        >>> # Context Manager approach
        >>> with variables("x"):
        ...     print(unify("x", 1))
        {'x': 1}
    """
    old_global_logic_variables = _global_logic_variables.copy()
    _global_logic_variables.update(set(variables))
    try:
        yield
    finally:
        _global_logic_variables.clear()
        _global_logic_variables.update(old_global_logic_variables)
