"""
Python polyfills for common builtins.
"""

# NOTE: 1. Please do not import any submodule in the directory here to avoid circular imports.
#       2. While adding a new polyfill module, also add it to POLYFILLED_MODULE_NAMES in loader.py.

# mypy: allow-untyped-defs

import math
from typing import Any, Callable, Sequence

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._device import DeviceContext


def index(iterator, item, start=0, end=None):
    for i, elem in islice(enumerate(iterator), start, end):
        if item == elem:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def islice(iterator, start=0, end=None, step=1):
    if start < 0 or (end is not None and end < 0) or step < 0:
        raise ValueError("Indices must be non-negative")
    if step == 0:
        raise ValueError("Step cannot be 0")

    it = iter(iterator)

    for _ in range(start):
        next(it)

    if end is None:
        for i, element in enumerate(it):
            if i % step == 0:
                yield element
    else:
        for i, element in enumerate(it):
            if i % step == 0 and i + start < end - start:
                yield element
            elif i + start >= end - start:
                break


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


# These classes handle support for TorchFunctionModes across
# graph breaks
# Today the TorchFunctionMode enter (for the classes we support)
# simply pushes the mode onto the stack. Since after this occurs
# the stack is mutated, and we replay these mutations, we don't need
# any cleanup logic to be run once the graph break occurs, we simply replay
# these mutations to ensure at the graph break the torch function mode stack is correct
#  and reconstruct the torch function mode stack normally
# when we compile the resume function on the other side of the break.
# However, to ensure we exit properly
# in the resume function, we need to re-enter the contexts as we do other contexts.
# These contexts do nothing on enter, but provide the correct exit logic to ensure
# the stack state is correct.
class NoEnterTorchFunctionMode(TorchFunctionMode):
    def __enter__(self):
        pass


class NoEnterDeviceTorchFunctionMode(DeviceContext):
    def __enter__(self):
        pass


def getattr_and_trace(*args, **kwargs):
    wrapper_obj = args[0]
    attr_name = args[1]
    fn = getattr(wrapper_obj, attr_name)
    return fn(*args[2:], **kwargs)


def mapping_get(obj, key, value=None):
    try:
        return obj.__getitem__(key)
    except KeyError:
        return value


def instantiate_user_defined_class_object(cls, /, *args, **kwargs):
    obj = cls.__new__(cls, *args, **kwargs)

    # Only call __init__ if the object is an instance of the class
    # Reference: https://github.com/python/cpython/blob/3.12/Objects/typeobject.c#L1670-L1673
    if isinstance(obj, cls):
        obj.__init__(*args, **kwargs)
    return obj
