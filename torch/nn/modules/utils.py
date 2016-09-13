import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

