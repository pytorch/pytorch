"""
Python polyfills for common builtins.
"""
import math


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


def index(iterator, item, start=0, end=-1):
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
