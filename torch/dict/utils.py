# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import inspect

import math
import time
from collections import defaultdict, OrderedDict
from collections.abc import KeysView
from copy import copy
from functools import wraps
from importlib import import_module
from numbers import Number
from textwrap import indent
from typing import Any, Callable, Iterator, List, Sequence, Tuple, TypeVar, Union

import numpy as np
from packaging.version import parse

import torch
from torch import Tensor
from torch._C import _disabled_torch_function_impl
from torch._C._functorch import get_unwrapped, is_batchedtensor
from torch.nn.parameter import _ParameterMeta

T = TypeVar("T", bound="TensorDictBase")

_STRDTYPE2DTYPE = {
    str(dtype): dtype
    for dtype in (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.quint4x2,
    )
}

IndexType = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]
DeviceType = Union[torch.device, str, int]
NestedKey = Union[str, Tuple[str, ...]]

_KEY_ERROR = 'key "{}" not found in {} with ' "keys {}"
_LOCK_ERROR = (
    "Cannot modify locked TensorDict. For in-place modification, consider "
    "using the `set_()` method and make sure the key is present."
)


def _sub_index(tensor: Tensor, idx: IndexType) -> Tensor:
    """Allows indexing of tensors with nested tuples.

     >>> sub_tensor1 = tensor[tuple1][tuple2]
     >>> sub_tensor2 = _sub_index(tensor, (tuple1, tuple2))
     >>> assert torch.allclose(sub_tensor1, sub_tensor2)

    Args:
        tensor (Tensor): tensor to be indexed.
        idx (tuple of indices): indices sequence to be used.

    """
    if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
        idx0 = idx[0]
        idx1 = idx[1:]
        return _sub_index(_sub_index(tensor, idx0), idx1)
    return tensor[idx]


def convert_ellipsis_to_idx(
    idx: tuple[int | Ellipsis] | Ellipsis, batch_size: list[int]
) -> tuple[int, ...]:
    """Given an index containing an ellipsis or just an ellipsis, converts any ellipsis to slice(None).

    Example:
        >>> idx = (..., 0)
        >>> batch_size = [1,2,3]
        >>> new_index = convert_ellipsis_to_idx(idx, batch_size)
        >>> print(new_index)
        (slice(None, None, None), slice(None, None, None), 0)

    Args:
        idx (tuple, Ellipsis): Input index
        batch_size (list): Shape of tensor to be indexed

    Returns:
        new_index (tuple): Output index
    """
    istuple = isinstance(idx, tuple)
    if (not istuple and idx is not Ellipsis) or (
        istuple and all(_idx is not Ellipsis for _idx in idx)
    ):
        return idx
    new_index = ()
    num_dims = len(batch_size)

    if idx is Ellipsis:
        idx = (...,)

    num_ellipsis = sum(_idx is Ellipsis for _idx in idx)
    if num_dims < (len(idx) - num_ellipsis - sum(item is None for item in idx)):
        raise RuntimeError("Not enough dimensions in TensorDict for index provided.")

    start_pos, after_ellipsis_length = None, 0
    for i, item in enumerate(idx):
        if item is Ellipsis:
            if start_pos is not None:
                raise RuntimeError("An index can only have one ellipsis at most.")
            else:
                start_pos = i
        if item is not Ellipsis and start_pos is not None:
            after_ellipsis_length += 1
        if item is None:
            # unsqueeze
            num_dims += 1

    before_ellipsis_length = start_pos
    if start_pos is None:
        return idx
    else:
        ellipsis_length = num_dims - after_ellipsis_length - before_ellipsis_length

    new_index += idx[:start_pos]

    ellipsis_start = start_pos
    ellipsis_end = start_pos + ellipsis_length
    new_index += (slice(None),) * (ellipsis_end - ellipsis_start)

    new_index += idx[start_pos + 1 : start_pos + 1 + after_ellipsis_length]

    if len(new_index) != num_dims:
        raise RuntimeError(
            f"The new index {new_index} is incompatible with the dimensions of the batch size {num_dims}."
        )

    return new_index


def _copy(self: list[int]) -> list[int]:
    return list(self)


def infer_size_impl(shape: list[int], numel: int) -> list[int]:
    """Infers the shape of an expanded tensor whose number of elements is indicated by :obj:`numel`.

    Copied from pytorch for compatibility issues (See #386).
    See https://github.com/pytorch/pytorch/blob/35d4fa444b67cbcbe34a862782ddf2d92f5b1ce7/torch/jit/_shape_functions.py
    for the original copy.

    """
    newsize = 1
    infer_dim: int | None = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise AssertionError("invalid shape dimensions")
    if not (
        numel == newsize
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise AssertionError("invalid shape")
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out


def _unwrap_value(value: Tensor) -> Tensor:
    # batch_dims = value.ndimension()
    if not isinstance(value, Tensor):
        out = value
    elif is_batchedtensor(value):
        out = get_unwrapped(value)
    else:
        out = value
    return out
    # batch_dims = out.ndimension() - batch_dims
    # batch_size = out.shape[:batch_dims]
    # return out, batch_size


if hasattr(math, "prod"):  # Python 3.8+

    def prod(sequence):
        """General prod function, that generalised usage across math and np.

        Created for multiple python versions compatibility.

        """
        return math.prod(sequence)

else:

    def prod(sequence):
        """General prod function, that generalised usage across math and np.

        Created for multiple python versions compatibility.

        """
        return int(np.prod(sequence))


def expand_as_right(
    tensor: Tensor | TensorDictBase,
    dest: Tensor | TensorDictBase,
) -> Tensor | TensorDictBase:
    """Expand a tensor on the right to match another tensor shape.

    Args:
        tensor: tensor to be expanded
        dest: tensor providing the target shape

    Returns:
         a tensor with shape matching the dest input tensor shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> dest = torch.zeros(3,4,5)
        >>> print(expand_as_right(tensor, dest).shape)
        torch.Size([3,4,5])

    """
    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have less "
            f"dimensions than the input tensor, got"
            f" tensor.ndimension()={tensor.ndimension()} and "
            f"dest.ndimension()={dest.ndimension()}"
        )
    if not (tensor.shape == dest.shape[: tensor.ndimension()]):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, "
            f"got: tensor.shape={tensor.shape}, dest={dest.shape}"
        )
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand(dest.shape)


def expand_right(tensor: Tensor, shape: Sequence[int]) -> Tensor:
    """Expand a tensor on the right to match a desired shape.

    Args:
        tensor: tensor to be expanded
        shape: target shape

    Returns:
         a tensor with shape matching the target shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> shape = (3,4,5)
        >>> print(expand_right(tensor, shape).shape)
        torch.Size([3,4,5])

    """
    tensor_expand = tensor
    while tensor_expand.ndimension() < len(shape):
        tensor_expand = tensor_expand.unsqueeze(-1)
    tensor_expand = tensor_expand.expand(shape)
    return tensor_expand


NUMPY_TO_TORCH_DTYPE_DICT = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
TORCH_TO_NUMPY_DTYPE_DICT = {
    value: key for key, value in NUMPY_TO_TORCH_DTYPE_DICT.items()
}


def is_nested_key(key: NestedKey) -> bool:
    """Returns True if key is a NestedKey."""
    if isinstance(key, str):
        return True
    if key and isinstance(key, (list, tuple)):
        return all(isinstance(subkey, str) for subkey in key)
    return False


def is_seq_of_nested_key(seq: Sequence[NestedKey]) -> bool:
    """Returns True if seq is a Sequence[NestedKey]."""
    if seq and isinstance(seq, Sequence):
        return all(is_nested_key(k) for k in seq)
    elif isinstance(seq, Sequence):
        # we allow empty inputs
        return True
    return False


def _ndimension(tensor: Tensor) -> int:
    if isinstance(tensor, Tensor):
        return tensor.ndimension()
    else:
        return tensor.ndimension()


def _shape(tensor: Tensor) -> torch.Size:
    if not isinstance(tensor, Tensor):
        return tensor.shape
    if tensor.is_nested:
        shape = []
        for i in range(tensor.ndim):
            try:
                shape.append(tensor.size(i))
            except RuntimeError:
                shape.append(-1)
        return torch.Size(shape)
    return tensor.shape


def _device(tensor: Tensor) -> torch.device:
    if isinstance(tensor, Tensor):
        return tensor.device
    else:
        return tensor.device


def _is_shared(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
        if torch._C._functorch.is_batchedtensor(tensor):
            return None
        return tensor.is_shared()
    from functorch import dim as ftdim

    if isinstance(tensor, ftdim.Tensor):
        return None
    else:
        return tensor.is_shared()


def _is_meta(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
        return tensor.is_meta
    else:
        return tensor.is_meta


def _dtype(tensor: Tensor) -> torch.dtype:
    if isinstance(tensor, Tensor):
        return tensor.dtype
    else:
        return tensor.dtype


def _get_item(tensor: Tensor, index: IndexType) -> Tensor:
    if isinstance(tensor, Tensor):
        return tensor[index]
    else:
        return tensor[index]


def _set_item(tensor: Tensor, index: IndexType, value: Tensor, *, validated) -> Tensor:
    # the tensor must be validated
    if not validated:
        raise RuntimeError
    if isinstance(tensor, Tensor):
        tensor[index] = value
        return tensor
        return tensor
    else:
        tensor[index] = value
        return tensor


def _requires_grad(tensor: Tensor) -> bool:
    if isinstance(tensor, Tensor):
        return tensor.requires_grad
    else:
        return tensor.requires_grad


class timeit:
    """A dirty but easy to use decorator for profiling code."""

    _REG = {}

    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, fn):
        @wraps(fn)
        def decorated_fn(*args, **kwargs):
            with self:
                out = fn(*args, **kwargs)
                return out

        return decorated_fn

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t0
        val = self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = val[2]
        N = count + 1
        val[0] = val[0] * (count / N) + t / N
        val[1] += t
        val[2] = N

    @staticmethod
    def print(prefix=None):
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)"
            )
            print(" -- ".join(strings))

    @staticmethod
    def erase():
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]


def int_generator(seed):
    """A pseudo-random chaing generator.

    To be used to produce deterministic integer sequences

    Examples:
        >>> for _ in range(2):
        ...     init_int = 10
        ...     for _ in range(10):
        ...        init_int = int_generator(init_int)
        ...        print(init_int, end=", ")
        ...     print("")
        6756, 1717, 4410, 9740, 9611, 9716, 5397, 7745, 4521, 7523,
        6756, 1717, 4410, 9740, 9611, 9716, 5397, 7745, 4521, 7523,
    """
    max_seed_val = 10_000
    rng = np.random.default_rng(seed)
    seed = int.from_bytes(rng.bytes(8), "big")
    return seed % max_seed_val


def _is_lis_of_list_of_bools(index, first_level=True):
    # determines if an index is a list of list of bools.
    # this is aimed at catching a deprecation feature where list of list
    # of bools are valid indices
    if first_level:
        if not isinstance(index, list):
            return False
        if not len(index):
            return False
        if isinstance(index[0], list):
            return _is_lis_of_list_of_bools(index[0], False)
        return False
    # then we know it is a list of lists
    if isinstance(index[0], bool):
        return True
    if isinstance(index[0], list):
        return _is_lis_of_list_of_bools(index[0], False)
    return False


def is_tensorclass(obj: type | Any) -> bool:
    """Returns True if obj is either a tensorclass or an instance of a tensorclass."""
    cls = obj if isinstance(obj, type) else type(obj)
    return _is_tensorclass(cls)


def _is_tensorclass(cls) -> bool:
    return (
        dataclasses.is_dataclass(cls)
        and "to_tensordict" in cls.__dict__
        and "_from_tensordict" in cls.__dict__
    )


class implement_for:
    """A version decorator that checks the version in the environment and implements a function with the fitting one.

    If specified module is missing or there is no fitting implementation, call of the decorated function
    will lead to the explicit error.
    In case of intersected ranges, last fitting implementation is used.

    Args:
        module_name (str or callable): version is checked for the module with this
            name (e.g. "gym"). If a callable is provided, it should return the
            module.
        from_version: version from which implementation is compatible. Can be open (None).
        to_version: version from which implementation is no longer compatible. Can be open (None).

    Examples:
        >>> @implement_for("torch", None, "1.13")
        >>> def fun(self, x):
        ...     # Older torch versions will return x + 1
        ...     return x + 1
        ...
        >>> @implement_for("torch", "0.13", "2.0")
        >>> def fun(self, x):
        ...     # More recent torch versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for(lambda: import_module("torch"), "0.", None)
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for("gymnasium", "0.27", None)
        >>> def fun(self, x):
        ...     # If gymnasium is to be used instead of gym, x+3 will be returned
        ...     return x + 3
        ...

        This indicates that the function is compatible with gym 0.13+, but doesn't with gym 0.14+.
    """

    # Stores pointers to fitting implementations: dict[func_name] = func_pointer
    _implementations = {}
    _setters = []

    def __init__(
        self,
        module_name: Union[str, Callable],
        from_version: str = None,
        to_version: str = None,
    ):
        self.module_name = module_name
        self.from_version = from_version
        self.to_version = to_version
        implement_for._setters.append(self)

    @staticmethod
    def check_version(version, from_version, to_version):
        return (from_version is None or parse(version) >= parse(from_version)) and (
            to_version is None or parse(version) < parse(to_version)
        )

    @staticmethod
    def get_class_that_defined_method(f):
        """Returns the class of a method, if it is defined, and None otherwise."""
        return f.__globals__.get(f.__qualname__.split(".")[0], None)

    @property
    def func_name(self):
        return self.fn.__name__

    def module_set(self):
        """Sets the function in its module, if it exists already."""
        cls = self.get_class_that_defined_method(self.fn)
        if cls is None:
            # class not yet defined
            return
        if cls.__class__.__name__ == "function":
            cls = inspect.getmodule(self.fn)
        setattr(cls, self.fn.__name__, self.fn)

    @staticmethod
    def import_module(module_name: Union[Callable, str]) -> str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = import_module(module_name)
        else:
            module = module_name()
        return module.__version__

    def __call__(self, fn):
        self.fn = fn

        # If the module is missing replace the function with the mock.
        func_name = self.func_name
        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args, **kwargs):
            raise ModuleNotFoundError(
                f"Supported version of '{func_name}' has not been found."
            )

        do_set = False
        # Return fitting implementation if it was encountered before.
        if func_name in implementations:
            try:
                # check that backends don't conflict
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    do_set = True
                if not do_set:
                    return implementations[func_name]
            except ModuleNotFoundError:
                # then it's ok, there is no conflict
                return implementations[func_name]
        else:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    do_set = True
            except ModuleNotFoundError:
                return unsupported
        if do_set:
            implementations[func_name] = fn
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters=None):
        if setters is None:
            setters = copy(cls._setters)
        cls._setters = []
        cls._implementations = {}
        for setter in setters:
            setter(setter.fn)
            cls._setters.append(setter)


def _unfold_sequence(seq):
    for item in seq:
        if isinstance(item, (list, tuple)):
            yield tuple(_unfold_sequence(item))
        else:
            if isinstance(item, (str, int, slice)) or item is Ellipsis:
                yield item
            else:
                yield id(item)


def _make_cache_key(args, kwargs):
    """Creats a key for the cache such that memory footprint is minimized."""
    return (
        tuple(_unfold_sequence(args)),
        tuple(_unfold_sequence(sorted(kwargs.items()))),
    )


def cache(fun):
    """A cache for TensorDictBase subclasses.

    This decorator will cache the values returned by a method as long as the
    input arguments match.
    Leaves (tensors and such) are not cached.
    The cache is stored within the tensordict such that it can be erased at any
    point in time.

    Examples:
        >>> import timeit
        >>> from tensordict import TensorDict
        >>> class SomeOtherTd(TensorDict):
        ...     @cache
        ...     def all_keys(self):
        ...         return set(self.keys(include_nested=True))
        >>> td = SomeOtherTd({("a", "b", "c", "d", "e", "f", "g"): 1.0}, [])
        >>> td.lock_()
        >>> print(timeit.timeit("set(td.keys(True))", globals={'td': td}))
        11.057
        >>> print(timeit.timeit("set(td.all_keys())", globals={'td': td}))
        0.88
    """

    @wraps(fun)
    def newfun(_self: TensorDictBase, *args, **kwargs):
        if not _self.is_locked:
            return fun(_self, *args, **kwargs)
        cache = _self._cache
        if cache is None:
            cache = _self._cache = defaultdict(dict)
        cache = cache[fun.__name__]
        key = _make_cache_key(args, kwargs)
        if key not in cache:
            out = fun(_self, *args, **kwargs)
            if not isinstance(out, Tensor):
                # we don't cache tensors to avoid filling the mem and / or
                # stacking them from their origin
                cache[key] = out
        else:
            out = cache[key]
        return out

    return newfun


def erase_cache(fun):
    """A decorator to erase the cache at each call."""

    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        self._erase_cache()
        return fun(self, *args, **kwargs)

    return new_fun


_NON_STR_KEY_TUPLE_ERR = "Nested membership checks with tuples of strings is only supported when setting `include_nested=True`."
_NON_STR_KEY_ERR = "TensorDict keys are always strings. Membership checks are only supported for strings or non-empty tuples of strings (for nested TensorDicts)"
_GENERIC_NESTED_ERR = "Only NestedKeys are supported. Got key {}."


class _StringKeys(KeysView):
    """A key view where contains is restricted to strings."""

    def __contains__(self, item):
        if not isinstance(item, str):
            try:
                unravel_item = _unravel_key_to_tuple(item)
                if not unravel_item:  # catch errors during unravel
                    raise TypeError
            except Exception:
                raise TypeError(_NON_STR_KEY_ERR)
            if len(unravel_item) > 1:
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)
            else:
                item = unravel_item[0]
        return super().__contains__(item)


class _StringOnlyDict(dict):
    """A dict class where contains is restricted to strings."""

    # kept here for debugging
    # def __setitem__(self, key, value):
    #     if not isinstance(key, str):
    #         raise RuntimeError
    #     return super().__setitem__(key, value)

    def __contains__(self, item):
        if not isinstance(item, str):
            try:
                unravel_item = _unravel_key_to_tuple(item)
                if not unravel_item:  # catch errors during unravel
                    raise TypeError
            except Exception:
                raise TypeError(_NON_STR_KEY_ERR)
            if len(unravel_item) > 1:
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)
            else:
                item = unravel_item[0]
        return super().__contains__(item)

    def keys(self):
        return _StringKeys(self)


def lock_blocked(func):
    """Checks that the tensordict is unlocked before executing a function."""

    @wraps(func)
    def new_func(self, *args, **kwargs):
        if self.is_locked:
            raise RuntimeError(_LOCK_ERROR)
        return func(self, *args, **kwargs)

    return new_func


class as_decorator:
    """Converts a method to a decorator.

    Examples:
        >>> from tensordict import TensorDict
        >>> data = TensorDict({}, [])
        >>> with data.lock_(): # lock_ is decorated
        ...     assert data.is_locked
        >>> assert not data.is_locked
    """

    def __init__(self, attr=None):
        self.attr = attr

    def __call__(self, func):
        @wraps(func)
        def new_func(_self, *args, **kwargs):
            if self.attr is not None:
                _attr_pre = getattr(_self, self.attr)
            out = func(_self, *args, **kwargs)
            if self.attr is not None:
                _attr_post = getattr(_self, self.attr)
            if self.attr is None or (_attr_post is not _attr_pre):
                out._last_op = (new_func.__name__, (args, kwargs, _self))
            else:
                out._last_op = None
            return out

        return new_func


def _split_tensordict(td, chunksize, num_chunks, num_workers, dim):
    if chunksize is None and num_chunks is None:
        num_chunks = num_workers
    if chunksize is not None and num_chunks is not None:
        raise ValueError(
            "Either chunksize or num_chunks must be provided, but not both."
        )
    if num_chunks is not None:
        num_chunks = min(td.shape[dim], num_chunks)
        return td.chunk(num_chunks, dim=dim)
    else:
        chunksize = min(td.shape[dim], chunksize)
        return td.split(chunksize, dim=dim)


def _parse_to(*args, **kwargs):
    batch_size = kwargs.pop("batch_size", None)
    other = kwargs.pop("other", None)
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
        *args, **kwargs
    )
    if other is not None:
        if device is not None and device != other.device:
            raise ValueError("other and device cannot be both passed")
        device = other.device
        dtypes = {val.dtype for val in other.values(True, True)}
        if len(dtypes) > 1 or len(dtypes) == 0:
            dtype = None
        elif len(dtypes) == 1:
            dtype = list(dtypes)[0]
    return device, dtype, non_blocking, convert_to_format, batch_size


class _ErrorInteceptor:
    """Context manager for catching errors and modifying message.

    Intended for use with stacking / concatenation operations applied to TensorDicts.

    """

    DEFAULT_EXC_MSG = "Expected all tensors to be on the same device"

    def __init__(
        self,
        key: NestedKey,
        prefix: str,
        exc_msg: str | None = None,
        exc_type: type[Exception] | None = None,
    ) -> None:
        self.exc_type = exc_type if exc_type is not None else RuntimeError
        self.exc_msg = exc_msg if exc_msg is not None else self.DEFAULT_EXC_MSG
        self.prefix = prefix
        self.key = key

    def _add_key_to_error_msg(self, msg: str) -> str:
        if msg.startswith(self.prefix):
            return f'{self.prefix} "{self.key}" /{msg[len(self.prefix):]}'
        return f'{self.prefix} "{self.key}". {msg}'

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, _):
        if exc_type is self.exc_type and (
            self.exc_msg is None or self.exc_msg in str(exc_value)
        ):
            exc_value.args = (self._add_key_to_error_msg(str(exc_value)),)


def _nested_keys_to_dict(keys: Iterator[NestedKey]) -> dict[str, Any]:
    nested_keys = {}
    for key in keys:
        if isinstance(key, str):
            nested_keys.setdefault(key, {})
        else:
            d = nested_keys
            for subkey in key:
                d = d.setdefault(subkey, {})
    return nested_keys


def _dict_to_nested_keys(
    nested_keys: dict[NestedKey, NestedKey], prefix: tuple[str, ...] = ()
) -> tuple[str, ...]:
    for key, subkeys in nested_keys.items():
        if subkeys:
            yield from _dict_to_nested_keys(subkeys, prefix=(*prefix, key))
        elif prefix:
            yield (*prefix, key)
        else:
            yield key


def _default_hook(td: T, key: tuple[str, ...]) -> None:
    """Used to populate a tensordict.

    For example, ``td.set(("a", "b"))`` may require to create ``"a"``.

    """
    out = td.get(key[0], None)
    if out is None:
        td._create_nested_str(key[0])
        out = td._get_str(key[0], None)
    return out


def _get_leaf_tensordict(
    tensordict: T, key: tuple[str, ...], hook: Callable = None
) -> tuple[TensorDictBase, str]:
    # utility function for traversing nested tensordicts
    # hook should return the default value for tensordit.get(key)
    while len(key) > 1:
        if hook is not None:
            tensordict = hook(tensordict, key)
        else:
            tensordict = tensordict.get(key[0])
        key = key[1:]
    return tensordict, key[0]


def assert_allclose_td(
    actual: T,
    expected: T,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = True,
    msg: str = "",
) -> bool:
    """Compares two tensordicts and raise an exception if their content does not match exactly."""
    from torch.dict.base import _is_tensor_collection

    if not _is_tensor_collection(actual.__class__) or not _is_tensor_collection(
        expected.__class__
    ):
        raise TypeError("assert_allclose inputs must be of TensorDict type")

    set1 = set(actual.keys())
    set2 = set(expected.keys())
    if not (len(set1.difference(set2)) == 0 and len(set2) == len(set1)):
        raise KeyError(
            "actual and expected tensordict keys mismatch, "
            f"keys {(set1 - set2).union(set2 - set1)} appear in one but not "
            f"the other."
        )
    keys = sorted(actual.keys(), key=str)
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        if _is_tensor_collection(input1.__class__):
            assert_allclose_td(input1, input2, rtol=rtol, atol=atol)
            continue

        mse = (input1.to(torch.float) - input2.to(torch.float)).pow(2).sum()
        mse = mse.div(input1.numel()).sqrt().item()

        default_msg = f"key {key} does not match, got mse = {mse:4.4f}"
        msg = "\t".join([default_msg, msg]) if len(msg) else default_msg
        torch.testing.assert_close(
            input1, input2, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
    return True


def _get_repr(tensor: Tensor) -> str:
    s = ", ".join(
        [
            f"shape={_shape(tensor)}",
            f"device={_device(tensor)}",
            f"dtype={_dtype(tensor)}",
            f"is_shared={_is_shared(tensor)}",
        ]
    )
    return f"{tensor.__class__.__name__}({s})"


def _get_repr_custom(cls, shape, device, dtype, is_shared) -> str:
    s = ", ".join(
        [
            f"shape={shape}",
            f"device={device}",
            f"dtype={dtype}",
            f"is_shared={is_shared}",
        ]
    )
    return f"{cls.__name__}({s})"


def _make_repr(key: str, item: CompatibleType, tensordict: T) -> str:
    from torch.dict.base import _is_tensor_collection

    if _is_tensor_collection(type(item)):
        return f"{key}: {repr(tensordict.get(key))}"
    return f"{key}: {_get_repr(item)}"


def _td_fields(td: T, keys=None) -> str:
    strs = []
    if keys is None:
        keys = td.keys()
    for key in keys:
        shape = td.get_item_shape(key)
        if -1 not in shape:
            item = td.get(key)
            strs.append(_make_repr(key, item, td))
        else:
            # we know td is lazy stacked and the key is a leaf
            # so we can get the shape and escape the error
            temp_td = td
            tensor = temp_td.get(key)
            from torch.dict import TensorDictBase

            if isinstance(tensor, TensorDictBase):
                substr = _td_fields(tensor)
            else:
                substr = _get_repr_custom(
                    tensor.__class__,
                    shape=shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    is_shared=tensor.is_shared(),
                )
            strs.append(f"{key}: {substr}")

    return indent(
        "\n" + ",\n".join(sorted(strs)),
        4 * " ",
    )


def _check_keys(
    list_of_tensordicts: Sequence[TensorDictBase],
    strict: bool = False,
    include_nested: bool = False,
    leaves_only: bool = False,
) -> set[str]:
    if not len(list_of_tensordicts):
        return set()
    keys: set[str] = set(
        list_of_tensordicts[0].keys(
            include_nested=include_nested, leaves_only=leaves_only
        )
    )
    for td in list_of_tensordicts[1:]:
        k = td.keys(include_nested=include_nested, leaves_only=leaves_only)
        if not strict:
            keys = keys.intersection(k)
        else:
            if set(k) != keys:
                raise KeyError(
                    f"got keys {keys} and {set(td.keys())} which are incompatible"
                )
    return keys


def _expand_to_match_shape(
    parent_batch_size: torch.Size,
    tensor: Tensor,
    self_batch_dims: int,
    self_device: DeviceType,
) -> Tensor | TensorDictBase:
    if hasattr(tensor, "dtype"):
        return torch.zeros(
            (
                *parent_batch_size,
                *_shape(tensor)[self_batch_dims:],
            ),
            dtype=tensor.dtype,
            device=self_device,
        )
    else:
        # tensordict
        from torch.dict import TensorDict

        out = TensorDict(
            {},
            [*parent_batch_size, *_shape(tensor)[self_batch_dims:]],
            device=self_device,
        )
        return out


def _set_max_batch_size(source: T, batch_dims=None):
    """Updates a tensordict with its maximium batch size."""
    tensor_data = list(source.values())

    for val in tensor_data:
        from torch.dict.base import _is_tensor_collection

        if _is_tensor_collection(val.__class__):
            _set_max_batch_size(val, batch_dims=batch_dims)
    batch_size = []
    if not tensor_data:  # when source is empty
        source.batch_size = batch_size
        return
    curr_dim = 0
    while True:
        if tensor_data[0].dim() > curr_dim:
            curr_dim_size = tensor_data[0].size(curr_dim)
        else:
            source.batch_size = batch_size
            return
        for tensor in tensor_data[1:]:
            if tensor.dim() <= curr_dim or tensor.size(curr_dim) != curr_dim_size:
                source.batch_size = batch_size
                return
        if batch_dims is None or len(batch_size) < batch_dims:
            batch_size.append(curr_dim_size)
        curr_dim += 1


def _clone_value(value: CompatibleType, recurse: bool) -> CompatibleType:
    from torch.dict.base import _is_tensor_collection

    if recurse:
        return value.clone()
    elif _is_tensor_collection(value.__class__):
        return value.clone(recurse=False)
    else:
        return value


def _is_number(item):
    if isinstance(item, Tensor) and item.ndim == 0:
        return True
    if isinstance(item, np.ndarray) and item.ndim == 0:
        return True
    from functorch import dim as ftdim

    if isinstance(item, (Number, ftdim.Dim)):
        return True
    return False


def _expand_index(index, batch_size):
    len_index = sum(True for idx in index if idx is not None)
    if len_index > len(batch_size):
        raise ValueError
    if len_index < len(batch_size):
        index = index + (slice(None),) * (len(batch_size) - len_index)
    return index


def _broadcast_tensors(index):
    # tensors and range need to be broadcast
    tensors = {
        i: tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        for i, tensor in enumerate(index)
        if isinstance(tensor, (range, list, np.ndarray, Tensor))
    }
    if tensors:
        shape = torch.broadcast_shapes(*[tensor.shape for tensor in tensors.values()])
        tensors = {i: tensor.expand(shape) for i, tensor in tensors.items()}
        index = tuple(
            idx if i not in tensors else tensors[i] for i, idx in enumerate(index)
        )
    return index


def _reduce_index(index):
    if all(
        idx is Ellipsis or (isinstance(idx, slice) and idx == slice(None))
        for idx in index
    ):
        index = ()
    return index


def _get_shape_from_args(*args, kwarg_name="size", **kwargs):
    if not args and not kwargs:
        return ()
    if args:
        if len(args) > 1 or isinstance(args[0], Number):
            size = args
        else:
            size = args[0]
        if len(kwargs):
            raise TypeError(
                f"Either the kwarg `{kwarg_name}`, a single shape argument or a sequence of integers can be passed. Got args={args} and kwargs={kwargs}."
            )
    else:
        size = kwargs.pop(kwarg_name, None)
        if size is None:
            raise TypeError(
                f"Either the kwarg `{kwarg_name}`, a single shape argument or a sequence of integers can be passed. Got args={args} and kwargs={kwargs}."
            )
    return size


def _unravel_key_to_tuple(key):
    if isinstance(key, str):
        return (key,)
    if not isinstance(key, tuple):
        return None
    result = ()
    for elt in key:
        elt = _unravel_key_to_tuple(elt)
        if elt is None:
            return None
        result = result + elt
    return result


def unravel_key_list(keys):
    raise NotImplementedError


def unravel_key(key):
    if isinstance(key, str):
        return key
    result = ()
    for elt in key:
        result = result + _unravel_key_to_tuple(elt)
    if len(result) == 1:
        return result[0]
    return result


class Buffer(Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module buffer.

    Args:
        data (Tensor): buffer tensor.
        requires_grad (bool, optional): if the buffer requires gradient. See
            :ref:`locally-disable-grad-doc` for more details. Default: `False`
    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.empty(0)
        if type(data) is Tensor or type(data) is Buffer:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        t._is_buffer = True
        return t

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Buffer containing:\n" + super(Buffer, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict()),
        )

    __torch_function__ = _disabled_torch_function_impl
