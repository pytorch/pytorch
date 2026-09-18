"""Python polyfills for collections.abc."""

from __future__ import annotations

import sys
from collections.abc import Set as AbstractSet
from typing import Any

from ..decorators import substitute_in_graph


__all__ = ["set_hash"]


@substitute_in_graph(AbstractSet._hash)
def set_hash(self: AbstractSet[Any]) -> int:
    if type(self) is frozenset:
        # Use Dynamo's FrozensetVariable.hash_impl directly.  In particular,
        # this keeps identity-derived element hashes out of Python control flow.
        return hash(self)

    # Mirror collections.abc.Set._hash exactly for arbitrary implementations:
    # their len() and iteration may have observable behavior.
    max_value = sys.maxsize
    mask = 2 * max_value + 1
    n = len(self)
    h = 1927868237 * (n + 1)
    h &= mask
    for x in self:
        hx = hash(x)
        h ^= (hx ^ (hx << 16) ^ 89869747) * 3644798167
        h &= mask
    h ^= (h >> 11) ^ (h >> 25)
    h = h * 69069 + 907133923
    h &= mask
    if h > max_value:
        h -= mask + 1
    if h == -1:
        h = 590923713
    return h
