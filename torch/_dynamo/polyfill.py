# mypy: ignore-errors

"""
Python polyfills for common builtins.
"""
import math

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
