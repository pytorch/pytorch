r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.

`default_collate` and `default_convert` are exposed to users via 'dataloader.py'.
"""

import collections
import contextlib
import re
import torch

from typing import Callable, Dict, Optional, Tuple, Type, Union
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""
        Function that converts each NumPy array element into a :class:`torch.Tensor`. If the input is a `Sequence`,
        `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
        If the input is not an NumPy array, it is left unchanged.
        This is used as the default function for collation when both `batch_sampler` and
        `batch_size` are NOT defined in :class:`~torch.utils.data.DataLoader`.

        The general input type to output type mapping is similar to that
        of :func:`~torch.utils.data.default_collate`. See the description there for more details.

        Args:
            data: a single data point to be converted

        Examples:
            >>> # Example with `int`
            >>> default_convert(0)
            0
            >>> # Example with NumPy array
            >>> # xdoctest: +SKIP
            >>> default_convert(np.array([0, 1]))
            tensor([0, 1])
            >>> # Example with NamedTuple
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_convert(Point(0, 0))
            Point(x=0, y=0)
            >>> default_convert(Point(np.array(0), np.array(0)))
            Point(x=tensor(0), y=tensor(0))
            >>> # Example with List
            >>> default_convert([np.array([0, 1]), np.array([2, 3])])
            [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    r"""
        General collate function that handles collection type of element within each batch
        and opens function registry to deal with specific element types. `default_collate_fn_map`
        provides default collate functions for tensors, numpy arrays, numbers and strings.

        Args:
            batch: a single batch to be collated
            collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
              If the element type isn't present in this dictionary,
              this function will go through each key of the dictionary in the insertion order to
              invoke the corresponding collate function if the element type is a subclass of the key.

        Examples:
            >>> # Extend this function to handle batch of tensors
            >>> def collate_tensor_fn(batch, *, collate_fn_map):
            ...     return torch.stack(batch, 0)
            >>> def custom_collate(batch):
            ...     collate_map = {torch.Tensor: collate_tensor_fn}
            ...     return collate(batch, collate_fn_map=collate_map)
            >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
            >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

        Note:
            Each collate function requires a positional argument for batch and a keyword argument
            for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.as_tensor(batch)


def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch)


def collate_str_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np
    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[string_classes] = collate_str_fn


def default_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:

            * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
            * NumPy Arrays -> :class:`torch.Tensor`
            * `float` -> :class:`torch.Tensor`
            * `int` -> :class:`torch.Tensor`
            * `str` -> `str` (unchanged)
            * `bytes` -> `bytes` (unchanged)
            * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
            * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
              default_collate([V2_1, V2_2, ...]), ...]`
            * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
              default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> # xdoctest: +SKIP
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Two options to extend `default_collate` to handle specific type
            >>> # Option 1: Write custom collate function and invoke `default_collate`
            >>> def custom_collate(batch):
            ...     elem = batch[0]
            ...     if isinstance(elem, CustomType):  # Some custom condition
            ...         return ...
            ...     else:  # Fall back to `default_collate`
            ...         return default_collate(batch)
            >>> # Option 2: In-place modify `default_collate_fn_map`
            >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
            ...     return ...
            >>> default_collate_fn_map.update(CustoType, collate_customtype_fn)
            >>> default_collate(batch)  # Handle `CustomType` automatically
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)
