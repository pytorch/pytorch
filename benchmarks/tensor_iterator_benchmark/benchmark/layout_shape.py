from typing import NamedTuple, Tuple
from collections import defaultdict
from operator import mul
from functools import reduce
import torch


selected_layouts = [
    ("contiguous",),
    ("non_contiguous", "non_contiguous", "non_contiguous"),
    ("contiguous", "non_contiguous", "non_contiguous", "non_contiguous"),
]

more_layouts = [
    ("contiguous",),
    ("contiguous", "contiguous", "contiguous"),
    ("non_contiguous",),
    ("non_contiguous", "non_contiguous", "non_contiguous"),
    ("contiguous", "non_contiguous"),
    ("contiguous", "non_contiguous", "non_contiguous", "non_contiguous"),
]

padding = 1

def trivial_1d(shape, dtype=None, device=None):
    return torch.empty(shape, dtype=dtype, device=device)

def contiguous_last_dim(shape, dtype=None, device=None):
    underlying_shape = tuple(s + padding for s in shape)
    t = trivial_1d(underlying_shape, dtype, device)
    for i, s in enumerate(shape):
        t = t.narrow(i, 0, s)
    return t

def non_contiguous(shape, dtype=None, device=None):
    return contiguous_last_dim((*shape, 1), dtype, device).squeeze(-1)

class LayoutShape(NamedTuple):
    problem_size: int
    factory: callable
    shape: Tuple

    def new(self, dtype=None, device=None):
        return self.factory(self.shape, dtype, device)


def split_size(size, dims):
    s1 = size // dims
    s2 = size - s1 * (dims - 1)
    return (s1,) * (dims - 1) + (s2,)


def make_layout_shape(layout, contiguous_size=0, non_contiguous_size=0):
    countiguous_dims = 0
    non_contiguous_dims = 0
    for i in layout:
        if i == 'contiguous':
            countiguous_dims += 1
        else:
            assert i == 'non_contiguous'
            non_contiguous_dims += 1
    if non_contiguous_dims == 0:
        assert countiguous_dims > 0
        name = f"all contiguous {countiguous_dims}d"
        problem_size = contiguous_size
        factory = trivial_1d
        shape = split_size(contiguous_size, countiguous_dims)
    elif countiguous_dims == 0:
        assert non_contiguous_dims > 0
        name = f"all non-contiguous {non_contiguous_dims}d"
        problem_size = non_contiguous_size
        factory = non_contiguous
        shape = split_size(non_contiguous_size, non_contiguous_dims)
    else:
        assert countiguous_dims == 1
        assert non_contiguous_dims > 0
        assert layout[0] == 'contiguous'
        name = f"contiguous 1d and non-contiguous {non_contiguous_dims}d"
        problem_size = contiguous_size + non_contiguous_size
        factory = contiguous_last_dim
        shape = split_size(non_contiguous_size, non_contiguous_dims) + (contiguous_size,)
    shape = tuple(2 ** x for x in shape)
    return name, LayoutShape(problem_size, factory, shape)


def numel_after_pad(layout, shape):
    ret = 1
    for x in shape:
        ret *= x + padding
    if set(layout) == {'contiguous', 'non_contiguous'}:
        return ret
    elif set(layout) == {'non_contiguous'}:
        return (1 + padding) * ret
    assert set(layout) == {'contiguous'}
    return reduce(mul, shape, 1)


def sizeof(dtype):
    return torch.empty((), dtype=dtype).element_size()


def combine_layouts_and_shapes(layouts, dtype):
    max_size = 30  # number of bytes in power of 2
    step = 1
    ret1d = defaultdict(list)
    ret2d = defaultdict(lambda: defaultdict(list))
    for layout in layouts:
        if set(layout) == {'contiguous'}:
            for size in range(0, max_size + 1, step):
                name, result = make_layout_shape(layout, contiguous_size=size)
                if numel_after_pad(layout, result.shape) * sizeof(dtype) > 2 ** max_size:
                    break
                ret1d[name].append(result)
        elif set(layout) == {'non_contiguous'}:
            for size in range(0, max_size + 1, step):
                name, result = make_layout_shape(layout, non_contiguous_size=size)
                if numel_after_pad(layout, result.shape) * sizeof(dtype) > 2 ** max_size:
                    break
                ret1d[name].append(result)
        else:
            assert set(layout) == {'contiguous', 'non_contiguous'}
            for size1 in range(0, max_size + 1, step):
                for size2 in range(0, max_size + 1, step):
                    name, result = make_layout_shape(layout, size1, size2)
                    if numel_after_pad(layout, result.shape) * sizeof(dtype) <= 2 ** max_size:
                        ret2d[name][size2].append(result)
    return ret1d, ret2d


def get(dtype, more):
    layout = more_layouts if more else selected_layouts
    return combine_layouts_and_shapes(layout, dtype)
