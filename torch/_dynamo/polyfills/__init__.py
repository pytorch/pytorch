"""
Python polyfills for common builtins.
"""

# NOTE: 1. Please do not import any submodule in the directory here to avoid circular imports.
#       2. While adding a new polyfill module, also add it to POLYFILLED_MODULE_NAMES in loader.py.
#          Add it in the TYPE_CHECKING block below as well.

# mypy: allow-untyped-defs

import types
from collections import OrderedDict
from collections.abc import Hashable, Iterable, MutableMapping, Sequence
from itertools import repeat as _repeat
from typing import Any, Callable, TYPE_CHECKING

import torch

from ..utils import dict_keys


if TYPE_CHECKING:
    # Load by torch._dynamo.polyfills.loader
    # See also the POLYFILLED_MODULE_NAMES in torch/_dynamo/polyfills/loader.py
    # Put the submodules here to avoid circular imports
    from . import (
        builtins as builtins,
        functools as functools,
        itertools as itertools,
        operator as operator,
        os as os,
        pytree as pytree,
        struct as struct,
        sys as sys,
    )

from torch.overrides import BaseTorchFunctionMode


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
class NoEnterTorchFunctionMode(BaseTorchFunctionMode):
    def __enter__(self):
        pass


def index(iterator, item, start=0, end=None):
    from itertools import islice

    for i, elem in islice(enumerate(iterator), start, end):
        if item == elem:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item, count):
    for _ in range(count):
        yield item


def radians(x):
    import math

    return math.pi / 180.0 * x


def impl_CONTAINS_OP_fallback(a, b):
    # performs fallback "a in b"
    if hasattr(b, "__iter__"):
        # use __iter__ if __contains__ is not available
        for x in b:
            if x == a:
                return True
        return False
    raise TypeError(f"argument of type {type(b)} is not iterable")


def accumulate_grad(x, new_grad):
    # polyfills according to the Gradient Layout Contract
    if new_grad is None:
        return
    new_grad_strided = torch.empty_like(x)
    new_grad_strided.copy_(new_grad)
    if x.grad is None:
        x.grad = new_grad_strided
    elif torch.is_grad_enabled():
        x.grad = x.grad + new_grad_strided
    else:
        x.grad.add_(new_grad_strided)


# This mirrors
# https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/listobject.c#L3352-L3413
def list_cmp(op: Callable[[Any, Any], bool], left: Sequence[Any], right: Sequence[Any]):
    """emulate `(1,2,3) > (1,2)` etc"""
    # Apply `op` to the first pair that differ
    for a, b in zip(left, right):
        if a != b:
            return op(a, b)

    # No more pairs to compare, so compare sizes.
    return op(len(left), len(right))


def dict___eq__(d, other):
    if (len(d) != len(other)) or (d.keys() != other.keys()):
        return False

    if all(isinstance(a, OrderedDict) for a in (d, other)):
        return list(d.items()) == list(other.items())

    for k, v in d.items():
        if v != other[k]:
            return False

    return True


def set_symmetric_difference(set1, set2):
    symmetric_difference_set = set()
    for x in set1:
        if x not in set2:
            symmetric_difference_set.add(x)
    for x in set2:
        if x not in set1:
            symmetric_difference_set.add(x)
    return symmetric_difference_set


def set_symmetric_difference_update(set1, set2):
    result = set1.symmetric_difference(set2)
    set1.clear()
    set1.update(result)


def set_isdisjoint(set1, set2):
    if not isinstance(set2, Iterable):
        raise TypeError(f"'{type(set2)}' object is not iterable")

    for x in set1:
        for y in set2:
            if not isinstance(y, Hashable):
                raise TypeError(f"unhashable type: '{type(y)}'")
            if x == y:
                return False
    return True


def set_intersection(set1, *others):
    if len(others) == 0:
        return set1.copy()

    if not all(isinstance(s, Iterable) for s in others):
        raise TypeError(f"set.difference expected an iterable, got {type(others)}")

    for s in others:
        if any(not isinstance(x, Hashable) for x in s):
            raise TypeError("unhashable type")

    # return a new set with elements common in all sets
    intersection_set = set()
    for x in set1:
        for set2 in others:
            if not any(x == y for y in set2):
                break
        else:
            intersection_set.add(x)
    return intersection_set


def set_intersection_update(set1, *others):
    result = set1.intersection(*others)
    set1.clear()
    set1.update(result)


def set_union(set1, *others):
    # frozenset also uses this function
    if len(others) == 0:
        return set1.copy()

    if not all(isinstance(s, Iterable) for s in others):
        raise TypeError(f"set.union expected an iterable, got {type(others)}")

    for s in others:
        if any(not isinstance(x, Hashable) for x in s):
            raise TypeError("unhashable type")

    union_set = set(set1.copy())
    for set2 in others:
        set_update(union_set, set2)

    # frozenset also uses this function
    return type(set1)(union_set)


def set_update(set1, *others):
    if len(others) == 0:
        return set1

    for set2 in others:
        for x in set2:
            if x not in set1:
                set1.add(x)


def set_difference(set1, *others):
    if len(others) == 0:
        return set1.copy()

    if not all(isinstance(s, Iterable) for s in others):
        raise TypeError(f"set.difference expected an iterable, got {type(others)}")

    for s in others:
        if any(not isinstance(x, Hashable) for x in s):
            raise TypeError("unhashable type")

    difference_set = set()
    for x in set1:
        for set2 in others:
            if x in set2:
                break
        else:
            difference_set.add(x)
    return difference_set


def set_difference_update(set1, *others):
    result = set1.difference(*others)
    set1.clear()
    set1.update(result)


def assert_dict_equal(self_, d1, d2, msg=None):
    self_.assertTrue(d1 == d2, msg)


def assert_multi_line_equal(self_, first, second, msg=None):
    return self_.assertTrue(first == second, msg)


# The original impl. uses difflib
def assert_sequence_equal(self_, seq1, seq2, msg=None, seq_type=None):
    return self_.assertTrue(seq1 == seq2, msg)


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


# Used with something like dict(obj)
def construct_dict(cls, /, *args, **kwargs):
    dst = cls.__new__(cls)

    if args:
        src = args[0]

        if not isinstance(src, Iterable):
            raise TypeError(f"{type(src)} object is not iterable")

        # Ensure that the overridden __iter__ method is invoked
        if isinstance(src, (dict, MutableMapping, types.MappingProxyType)):
            for key in src:
                # This will inline the __getitem__ of the src object
                dst[key] = src[key]
        else:
            # likely a sequence like tuple of pairs
            for key, value in src:
                dst[key] = value

    if kwargs:
        for key in kwargs:
            dst[key] = kwargs[key]

    return dst


def foreach_map_fn(*args):
    op = args[0]
    new_args: list[Any] = []
    at_least_one_list = False
    for arg in args[1:]:
        if not isinstance(arg, (list, tuple)):
            new_args.append(_repeat(arg))
        else:
            at_least_one_list = True
            new_args.append(arg)

    # Just apply op once to args if there are no lists
    if not at_least_one_list:
        return op(*args[1:])

    out = []
    for unpacked in zip(*new_args):
        out.append(op(*unpacked))

    return out


def foreach_lerp_inplace(self, end, weight):
    # decompose foreach lerp into constituent ops, prevents a graph break due to
    # converting a value to a scalar when arg[2] is a single tensor
    result = torch._foreach_sub(end, self)
    result = torch._foreach_mul(result, weight)
    return torch._foreach_add_(self, result)


def foreach_pow_scalar(scalar, exps):
    return torch._foreach_pow([scalar for _ in exps], exps)


def addcmul_inplace(self, tensor1, tensor2, value):
    return self.add_(tensor1 * tensor2 * value)


def predicate(obj: Any) -> bool:
    # This will cause the rest of dynamo to handle the if statement correctly, so we don't have to rewrite it here.
    # We can't just use bool() here since we can't trace into that in general.
    if obj:
        return True
    return False


def cmp_eq(a, b):
    # Note that the commented `is` check should ideally be removed. This is a
    # CPython optimization that skips the __eq__ checks it the obj id's are
    # same. But, these lines adds many `is` nodes in the Fx graph for
    # SymNodeVariable. For now, we can just skip this check. This is STILL
    # correct because one of the __eq__ checks will pass later, just could be
    # slow in some corner cases.
    # if a is b:
    #     return True
    result = a.__eq__(b)
    if result is NotImplemented:
        result = b.__eq__(a)
    return result is not NotImplemented and result


def cmp_ne(a, b):
    # Check if __ne__ is overridden
    if isinstance(type(a).__ne__, types.FunctionType):
        return a.__ne__(b)
    return not cmp_eq(a, b)


def cmp_lt(a, b):
    result = a.__lt__(b)
    if result is NotImplemented:
        raise TypeError(f"{type(a)} does not support the < operator")
    return result


def cmp_le(a, b):
    # Check if __le__ is overridden
    if isinstance(type(a).__le__, types.FunctionType):
        return a.__le__(b)
    return cmp_eq(a, b) or cmp_lt(a, b)


def cmp_gt(a, b):
    # Check if __gt__ is overridden
    if isinstance(type(a).__gt__, types.FunctionType):
        return a.__gt__(b)
    # a > b is equivalent to b < a
    return cmp_lt(b, a)


def cmp_ge(a, b):
    # Check if __ge__ is overridden
    if isinstance(type(a).__ge__, types.FunctionType):
        return a.__ge__(b)
    return cmp_eq(a, b) or cmp_gt(a, b)
