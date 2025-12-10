from collections.abc import Sequence
from typing import Any, assert_type

from numpy._typing import _NestedSequence

a: Sequence[int]
b: Sequence[Sequence[int]]
c: Sequence[Sequence[Sequence[int]]]
d: Sequence[Sequence[Sequence[Sequence[int]]]]
e: Sequence[bool]
f: tuple[int, ...]
g: list[int]
h: Sequence[Any]

def func(a: _NestedSequence[int]) -> None: ...

assert_type(func(a), None)
assert_type(func(b), None)
assert_type(func(c), None)
assert_type(func(d), None)
assert_type(func(e), None)
assert_type(func(f), None)
assert_type(func(g), None)
assert_type(func(h), None)
assert_type(func(range(15)), None)
