"""
Python polyfills for builtins
"""

from collections.abc import Iterable, MutableMapping
from typing import Any

from ..decorators import substitute_in_graph


__all__ = []

try:
    import _collections  # type: ignore[import-not-found]

    @substitute_in_graph(_collections._count_elements)
    def _count_elements(
        mapping: MutableMapping[Any, int],
        iterable: Iterable[Any],
    ) -> None:
        "Tally elements from the iterable."
        mapping_get = mapping.get
        for elem in iterable:
            mapping[elem] = mapping_get(elem, 0) + 1

    __all__.append("_count_elements")

except ImportError:
    pass
