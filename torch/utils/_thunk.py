from collections.abc import Callable
from typing import Generic, Optional, TypeVar


R = TypeVar("R")


class Thunk(Generic[R]):
    """
    A simple lazy evaluation implementation that lets you delay
    execution of a function.  It properly handles releasing the
    function once it is forced.
    """

    f: Optional[Callable[[], R]]
    r: Optional[R]

    __slots__ = ["f", "r"]

    def __init__(self, f: Callable[[], R]):
        self.f = f
        self.r = None

    def force(self) -> R:
        if self.f is None:
            return self.r  # type: ignore[return-value]
        self.r = self.f()
        self.f = None
        return self.r
