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
    for i, elem in enumerate(list(iterator))[start:end]:
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


def dropwhile(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) → 6 4 1
    iterable = iter(iterable)
    for x in iterable:
        if not predicate(x):
            yield x
            break
    yield from iterable
