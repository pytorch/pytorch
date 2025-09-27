# Test ``inspect.formatannotation``
# https://github.com/python/cpython/issues/96073

from typing import Union

ann = Union[list[str], int]

# mock typing._type_repr behaviour
class A: ...

A.__module__ = 'testModule.typing'
A.__qualname__ = 'A'

ann1 = Union[list[A], int]
