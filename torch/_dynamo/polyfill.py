# mypy: ignore-errors

"""
Python polyfills for common builtins.
"""
import math
from typing import Any, Callable, Sequence

import torch


def all(iterator):
    for elem in iterator:
        if not elem:
            return False
    return True


def any(iterator):
    for elem in iterator:
        if elem:
            return True
    return False


def index(iterator, item, start=0, end=None):
    for i, elem in enumerate(list(iterator)[start:end], start):
        if item == elem:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item, count):
    for i in range(count):
        yield item


def radians(x):
    return math.pi / 180.0 * x


def accumulate_grad(x, new_grad):
    new_grad = torch.clone(new_grad)
    if x.grad is None:
        x.grad = new_grad
    else:
        x.grad.add_(new_grad)


def list_cmp(op: Callable[[Any, Any], bool], left: Sequence[Any], right: Sequence[Any]):
    """emulate `(1,2,3) > (1,2)` etc"""
    for a, b in zip(left, right):
        if a != b:
            return op(a, b)
    return op(len(left), len(right))


def set_isdisjoint(set1, set2):
    for x in set1:
        if x in set2:
            return False
    return True


def set_intersection(set1, set2):
    intersection_set = set()
    for x in set1:
        if x in set2:
            intersection_set.add(x)
    return intersection_set


def set_union(set1, set2):
    union_set = set1.copy()
    for x in set2:
        if x not in union_set:
            union_set.add(x)
    return union_set


def set_difference(set1, set2):
    difference_set = set()
    for x in set1:
        if x not in set2:
            difference_set.add(x)
    return difference_set


def dropwhile(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) -> 6 4 1
    iterable = iter(iterable)
    for x in iterable:
        if not predicate(x):
            yield x
            break
    yield from iterable


def zip_longest(*iterables, fillvalue=None):
    # Create a list of iterators from the input iterables
    iterators = [iter(it) for it in iterables]
    result = []
    while True:
        row = []
        active = False
        for it in iterators:
            try:
                # Try to get the next item from the iterator
                value = next(it)
                row.append(value)
                active = True
            except StopIteration:
                # If the iterator is exhausted, use the fillvalue
                row.append(fillvalue)
        if not active:
            break
        result.append(tuple(row))
    return result


def getattr_and_trace(*args, **kwargs):
    wrapper_obj = args[0]
    attr_name = args[1]
    fn = getattr(wrapper_obj, attr_name)
    return fn(*args[2:], **kwargs)


def enumerate(iterable, start=0):
    n = start
    for elem in iterable:
        yield n, elem
        n += 1
