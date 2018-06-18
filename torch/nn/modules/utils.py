import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def set_output_size(input_size, output_size):
    dim = len(input_size)
    if isinstance(output_size, int):
        return _ntuple(dim)(output_size)
    if len(output_size) == 1:
        return _ntuple(dim)(output_size[0])
    assert(len(output_size) == dim)

    shape = ()
    for i,s in enumerate(output_size):
        shape += ((s or input_size[i]),)

    return shape
