"""
Python polyfills for builtins
"""

from ..decorators import substitute_in_graph
from typing import MutableMapping, Iterable, Any


__all__ = []

try:
    import _collections

    @substitute_in_graph(_collections._count_elements)
    def _count_elements(
        mapping: MutableMapping[Any, int],
        iterable: Iterable[Any],
    ) -> None:
        'Tally elements from the iterable.'
        mapping_get = mapping.get
        for elem in iterable:
            mapping[elem] = mapping_get(elem, 0) + 1

    __all__.append("_count_elements")

except ImportError:
    pass