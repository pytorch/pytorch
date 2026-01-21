"""
Python polyfills for common builtins.
"""

# NOTE: 1. Please do not import any submodule in the directory here to avoid circular imports.
#       2. While adding a new polyfill module, also add it to POLYFILLED_MODULE_NAMES in loader.py.
#          Add it in the TYPE_CHECKING block below as well.

import types
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from itertools import repeat as _repeat
from operator import eq, ne
from typing import Any, TYPE_CHECKING, TypeVar

import torch


T = TypeVar("T")
U = TypeVar("U")
C = TypeVar("C")


if TYPE_CHECKING:
    from ..utils import dict_keys

    # Load by torch._dynamo.polyfills.loader
    # See also the POLYFILLED_MODULE_NAMES in torch/_dynamo/polyfills/loader.py
    # Put the submodules here to avoid circular imports
    from . import (
        _collections as _collections,
        builtins as builtins,
        functools as functools,
        itertools as itertools,
        operator as operator,
        os as os,
        pytree as pytree,
        struct as struct,
        sys as sys,
        torch_c_nn as torch_c_nn,
        traceback as traceback,
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
    def __enter__(self) -> None:
        pass


def index(
    iterator: Iterator[T], item: T, start: int = 0, end: int | None = None
) -> int:
    from itertools import islice

    for i, elem in islice(enumerate(iterator), start, end):
        if item == elem:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item: T, count: int) -> Iterator[T]:
    for _ in range(count):
        yield item


def radians(x: float) -> float:
    import math

    return math.pi / 180.0 * x


def impl_CONTAINS_OP_fallback(a: T, b: Iterable[T]) -> bool:
    # performs fallback "a in b"
    if hasattr(b, "__iter__"):
        # use __iter__ if __contains__ is not available
        for x in b:
            if x == a:
                return True
        return False
    raise TypeError(f"argument of type {type(b)} is not iterable")


def accumulate_grad(x: torch.Tensor, new_grad: torch.Tensor | None) -> None:
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
def list_cmp(
    op: Callable[[Any, Any], bool], left: Sequence[T], right: Sequence[T]
) -> bool:
    """emulate `(1,2,3) > (1,2)` etc"""

    # Optimization: For equality, short-circuit if lengths differ
    # This avoids iterating through elements and triggering guards on SymInts
    left_len = len(left)
    right_len = len(right)

    if op is eq and left_len != right_len:
        return False
    if op is ne and left_len != right_len:
        return True

    # Apply `op` to the first pair that differ
    for a, b in zip(left, right):
        if a != b:
            return op(a, b)

    # No more pairs to compare, so compare sizes.
    return op(left_len, right_len)


def dict___eq__(d: dict[T, U], other: dict[T, U]) -> bool:
    if (len(d) != len(other)) or (d.keys() != other.keys()):
        return False

    if all(isinstance(a, OrderedDict) for a in (d, other)):
        return list(d.items()) == list(other.items())

    for k, v in d.items():
        if v != other[k]:
            return False

    return True


def set_symmetric_difference(
    set1: Iterable[T],
    set2: Iterable[T],
    cls: type[Any] = set,
) -> Any:
    symmetric_difference_set: set[T] = set()
    for x in set1:
        if x not in set2:
            symmetric_difference_set.add(x)
    for x in set2:
        if x not in set1:
            symmetric_difference_set.add(x)
    return cls(symmetric_difference_set)


def set_symmetric_difference_update(set1: set[T], set2: set[T]) -> None:
    result = set1.symmetric_difference(set2)
    set1.clear()
    set1.update(result)


def set_isdisjoint(set1: set[T], set2: set[T]) -> bool:
    if not isinstance(set2, Iterable):
        raise TypeError(f"'{type(set2)}' object is not iterable")

    for x in set1:
        for y in set2:
            if not isinstance(y, Hashable):
                raise TypeError(f"unhashable type: '{type(y)}'")
            if x == y:
                return False
    return True


def set_intersection(
    set1: set[T],
    *others: Iterable[T],
    # See facebook/pyrefly#1496 - leave generic
    cls: type[Any] = set,
) -> Any:
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
    return cls(intersection_set)


def set_intersection_update(set1: set[T], *others: Iterable[T]) -> None:
    result = set1.intersection(*others)
    set1.clear()
    set1.update(result)


def set_union(
    set1: set[T], *others: Iterable[T], cls: type[C] | None = None
) -> C | set[T]:
    # frozenset also uses this function
    if cls is None:
        # pyrefly: ignore[bad-assignment]
        cls = type(set1)

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
    # pyrefly: ignore[not-callable]
    return cls(union_set)


def set_update(set1: set[T], *others: Iterable[T]) -> set[T]:
    if len(others) == 0:
        return set1

    for set2 in others:
        for x in set2:
            if x not in set1:
                set1.add(x)


def set_difference(
    set1: set[T],
    *others: Iterable[T],
    cls: type[Any] = set,
) -> Any:
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
    return cls(difference_set)


def set_difference_update(set1: set[T], *others: Iterable[T]) -> None:
    result = set1.difference(*others)
    set1.clear()
    set1.update(result)


def assert_dict_equal(
    self_: Any, d1: dict[T, U], d2: dict[T, U], msg: str | None = None
) -> None:
    self_.assertTrue(d1 == d2, msg)


def assert_multi_line_equal(
    self_: Any, first: T, second: T, msg: str | None = None
) -> None:
    return self_.assertTrue(first == second, msg)


# The original impl. uses difflib
def assert_sequence_equal(
    self_: Any,
    seq1: Sequence[T],
    seq2: Sequence[T],
    msg: str | None = None,
    seq_type: type[Any] | None = None,
) -> None:
    return self_.assertTrue(seq1 == seq2, msg)


def getattr_and_trace(*args: Any, **kwargs: Any) -> Any:
    wrapper_obj = args[0]
    attr_name = args[1]
    fn = getattr(wrapper_obj, attr_name)
    return fn(*args[2:], **kwargs)


def mapping_get(obj: Mapping[T, U], key: T, value: U | None = None, /) -> U | None:
    try:
        return obj.__getitem__(key)
    except KeyError:
        return value


def instantiate_user_defined_class_object(
    cls: type[T], /, *args: Any, **kwargs: Any
) -> T:
    obj = cls.__new__(cls, *args, **kwargs)

    # Only call __init__ if the object is an instance of the class
    # Reference: https://github.com/python/cpython/blob/3.12/Objects/typeobject.c#L1670-L1673
    if isinstance(obj, cls):
        obj.__init__(*args, **kwargs)
    return obj


def mutable_mapping_update(
    self,
    data: Mapping[T, U] | Iterable[tuple[T, U]] = (),
    /,
    **kwargs: Any,
) -> None:
    if isinstance(data, Mapping):
        # Merge standard mapping with PyMapping_Items
        for key, value in data.items():
            self[key] = value
    # FIXME: Enabling the `elif`-branch below needs too many `VariableClass.call_obj_hasattr` changes.
    #   >>> class Foo:
    #   ...     def __init__(self):
    #   ...         self.keys = lambda: ['a', 'b', 'c']  # not required to be a method
    #   ...
    #   ...     def __getitem__(self, key):
    #   ...         return 0
    #   ...
    #   >>> dict(Foo())
    #   {'a': 0, 'b': 0, 'c': 0}
    #
    # > This is a rare case, so we comment it out for now.
    #
    # elif hasattr(data, "keys"):
    #     # Merge mapping-like object with PyMapping_Keys + PyObject_GetItem
    #     for key in data.keys():
    #         self[key] = data[key]
    else:
        if not isinstance(data, Iterable):
            raise TypeError(f"{type(data).__name__!r} object is not iterable")
        # Likely a sequence of pairs
        for key, value in data:
            self[key] = value

    if kwargs:
        for key, value in kwargs.items():
            self[key] = value


# Used with something like dict(obj)
def construct_dict(
    cls: type[T],
    data: Mapping[object, object] | Iterable[tuple[object, object]] = (),
    /,
    **kwargs: Any,
) -> T:
    self = cls.__new__(cls)
    mutable_mapping_update(self, data, **kwargs)
    return self


def foreach_map_fn(*args: Any) -> Any:
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


def foreach_lerp_inplace(
    self,
    end: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    weight: Sequence[bool | complex | float | int],
) -> None:
    # decompose foreach lerp into constituent ops, prevents a graph break due to
    # converting a value to a scalar when arg[2] is a single tensor
    result = torch._foreach_sub(end, self)
    result = torch._foreach_mul(result, weight)
    return torch._foreach_add_(self, result)


def foreach_pow_scalar(
    scalar: Any, exps: Sequence[bool | complex | float | int]
) -> tuple[torch.Tensor, ...]:
    return torch._foreach_pow([scalar for _ in exps], exps)


def addcmul_inplace(
    self, tensor1: torch.Tensor, tensor2: torch.Tensor, value: Any
) -> None:
    return self.add_(tensor1 * tensor2 * value)


def predicate(obj: object) -> bool:
    # This will cause the rest of dynamo to handle the if statement correctly, so we don't have to rewrite it here.
    # We can't just use bool() here since we can't trace into that in general.
    if obj:
        return True
    return False


def cmp_eq(a: object, b: object) -> bool:
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


def cmp_ne(a: object, b: object) -> bool:
    # Check if __ne__ is overridden
    if isinstance(type(a).__ne__, types.FunctionType):
        result = a.__ne__(b)
        if result is not NotImplemented:
            return result
        # Fall through to try b.__ne__(a) or cmp_eq
    if isinstance(type(b).__ne__, types.FunctionType):
        result = b.__ne__(a)
        if result is not NotImplemented:
            return result
    return not cmp_eq(a, b)


def cmp_lt(a: Any, b: Any) -> bool:
    result = a.__lt__(b)
    if result is NotImplemented:
        raise TypeError(f"{type(a)} does not support the < operator")
    return result


def cmp_le(a: Any, b: Any) -> bool:
    # Check if __le__ is overridden
    if isinstance(type(a).__le__, types.FunctionType):
        return a.__le__(b)
    return cmp_eq(a, b) or cmp_lt(a, b)


def cmp_gt(a: Any, b: Any) -> bool:
    # Check if __gt__ is overridden
    if isinstance(type(a).__gt__, types.FunctionType):
        return a.__gt__(b)
    # a > b is equivalent to b < a
    return cmp_lt(b, a)


def cmp_ge(a: Any, b: Any) -> bool:
    # Check if __ge__ is overridden
    if isinstance(type(a).__ge__, types.FunctionType):
        return a.__ge__(b)
    return cmp_eq(a, b) or cmp_gt(a, b)


def group_tensors_by_device_and_dtype(
    tensorlistlist: list[list[torch.Tensor | None]], with_indices: bool = False
) -> dict[tuple[torch.device, torch.dtype], tuple[list[list[Any]], list[int]]]:
    """Pure Python implementation of torch._C._group_tensors_by_device_and_dtype.

    Groups tensors by their device and dtype. This is useful before sending
    tensors off to a foreach implementation, which requires tensors to be on
    one device and dtype.

    Args:
        tensorlistlist: A list of lists of tensors (tensors can be None).
        with_indices: If True, track original indices in the output.

    Returns:
        A dict mapping (device, dtype) tuples to (grouped_tensorlistlist, indices).
    """
    # Result dict: (device, dtype) -> (list of lists, indices)
    result: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Any]], list[int]]
    ] = {}

    if not tensorlistlist or not tensorlistlist[0]:
        return result

    num_lists = len(tensorlistlist)
    num_tensors = len(tensorlistlist[0])

    for idx in range(num_tensors):
        # Find the first non-None tensor at this index to get device and dtype
        first_tensor = None
        for tlist in tensorlistlist:
            if tlist is not None and idx < len(tlist) and tlist[idx] is not None:
                first_tensor = tlist[idx]
                break

        if first_tensor is None:
            # All tensors at this index are None, skip
            continue

        key = (first_tensor.device, first_tensor.dtype)

        if key not in result:
            # Initialize empty lists for each tensorlist
            result[key] = ([[] for _ in range(num_lists)], [])

        grouped_lists, indices = result[key]

        # Add tensors from each list at this index
        for list_idx, tlist in enumerate(tensorlistlist):
            if tlist is not None and idx < len(tlist):
                grouped_lists[list_idx].append(tlist[idx])
            else:
                grouped_lists[list_idx].append(None)

        if with_indices:
            indices.append(idx)

    return result
