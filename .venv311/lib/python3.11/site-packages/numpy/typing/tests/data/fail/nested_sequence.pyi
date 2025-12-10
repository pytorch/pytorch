from collections.abc import Sequence
from numpy._typing import _NestedSequence

a: Sequence[float]
b: list[complex]
c: tuple[str, ...]
d: int
e: str

def func(a: _NestedSequence[int]) -> None: ...

reveal_type(func(a))  # type: ignore[arg-type, misc]
reveal_type(func(b))  # type: ignore[arg-type, misc]
reveal_type(func(c))  # type: ignore[arg-type, misc]
reveal_type(func(d))  # type: ignore[arg-type, misc]
reveal_type(func(e))  # type: ignore[arg-type, misc]
