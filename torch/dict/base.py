from __future__ import annotations

import abc
import collections
import numbers
from collections import defaultdict
from collections.abc import MutableMapping
from copy import copy
from numbers import Number
from textwrap import indent
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    Optional,
    OrderedDict,
    overload,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

import torch
from torch import distributed as dist, Tensor
from .utils import (
    _GENERIC_NESTED_ERR,
    _is_tensorclass,
    _KEY_ERROR,
    _shape,
    _split_tensordict,
    _td_fields,
    _unravel_key_to_tuple,
    as_decorator,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    erase_cache,
    IndexType,
    int_generator,
    lock_blocked,
    NestedKey,
    prod,
)

# NO_DEFAULT is used as a placeholder whenever the default is not provided.
# Using None is not an option since `td.get(key, default=None)` is a valid usage.
NO_DEFAULT = "_no_default_"

T = TypeVar("T", bound="TensorDictBase")


class _BEST_ATTEMPT_INPLACE:
    def __bool__(self):
        raise NotImplementedError


BEST_ATTEMPT_INPLACE = _BEST_ATTEMPT_INPLACE()

# some complex string used as separator to concatenate and split keys in
# distributed frameworks
CompatibleType = Union[Tensor,]

_STR_MIXED_INDEX_ERROR = "Received a mixed string-non string index. Only string-only or string-free indices are supported."

_HEURISTIC_EXCLUDED = (Tensor, tuple, list, set, dict, np.ndarray)

_TENSOR_COLLECTION_MEMO = {}


class TensorDictBase(MutableMapping):
    """TensorDictBase is an abstract parent class for TensorDicts, a torch.Tensor data container."""

    _safe = False
    _lazy = False
    _inplace_set = False
    is_meta = False
    _is_locked = False
    _cache = None

    def __bool__(self) -> bool:
        raise RuntimeError("Converting a tensordict to boolean value is not permitted")

    @abc.abstractmethod
    def __ne__(self, other: object) -> T:
        """NOT operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __xor__(self, other):
        """XOR operation over two tensordicts, for evey key.

        The two tensordicts must have the same key set.

        Args:
            other (TensorDictBase, dict, or float): the value to compare against.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> T:
        """Compares two tensordicts against each other, for every key. The two tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        ...

    def __del__(self):
        for td in getattr(self, "_locked_tensordicts", ()):
            td._remove_lock(id(self))

    def __repr__(self) -> str:
        fields = _td_fields(self)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
        device_str = indent(f"device={self.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        return f"{type(self).__name__}(\n{string})"

    def __iter__(self) -> Generator:
        """Iterates over the first shape-dimension of the tensordict."""
        if not self.batch_dims:
            raise StopIteration
        yield from self.unbind(0)

    def __len__(self) -> int:
        """Returns the length of first dimension, if there is, otherwise 0."""
        return self.shape[0] if self.batch_dims else 0

    def __contains__(self, key: NestedKey) -> bool:
        # by default a Mapping will implement __contains__ by calling __getitem__ and
        # returning False if a KeyError is raised, True otherwise. TensorDict has a
        # complex __getitem__ method since we support more than just retrieval of values
        # by key, and so this can be quite inefficient, particularly if values are
        # evaluated lazily on access. Hence, we don't support use of __contains__ and
        # direct the user to use TensorDict.keys() instead
        raise NotImplementedError(
            "TensorDict does not support membership checks with the `in` keyword. If "
            "you want to check if a particular key is in your TensorDict, please use "
            "`key in tensordict.keys()` instead."
        )

    def __getitem__(self, index: IndexType) -> T:
        """Indexes all tensors according to the provided index.

        The index can be a (nested) key or any valid shape index given the
        tensordict batch size.

        Examples:
            >>> td = TensorDict({"root": torch.arange(2), ("nested", "entry"): torch.arange(2)}, [2])
            >>> td["root"]
            torch.tensor([0, 1])
            >>> td["nested", "entry"]
            torch.tensor([0, 1])
            >>> td[:1]
            TensorDict(
                fields={
                    nested: TensorDict(
                        fields={
                            entry: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([1]),
                        device=None,
                        is_shared=False),
                    root: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([1]),
                device=None,
                is_shared=False)
        """
        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
            # _unravel_key_to_tuple will return an empty tuple if the index isn't a NestedKey
            idx_unravel = _unravel_key_to_tuple(index)
            if idx_unravel:
                return self._get_tuple(idx_unravel, NO_DEFAULT)
        if (istuple and not index) or (not istuple and index is Ellipsis):
            # empty tuple returns self
            return self
        if not istuple:
            if isinstance(index, int):
                return self._index_tensordict(index)
            # we only want tuple indices
            index = (index,)
        # # convert range/np.ndarray to tensor: this is not cheap
        # index = tuple(
        #     torch.tensor(idx) if isinstance(idx, (np.ndarray, range)) else idx
        #     for idx in index
        # )
        if istuple and any(idx is Ellipsis for idx in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        if all(isinstance(idx, slice) and idx == slice(None) for idx in index):
            return self

        return self._index_tensordict(index)

    # this is necessary for data collectors for instance, otherwise indexing
    # will always be achieved one element at a time.
    __getitems__ = __getitem__

    def _get_sub_tensordict(self, idx: IndexType) -> T:
        """Returns a _SubTensorDict with the desired index."""
        from torch.dict.tensordict import _SubTensorDict

        return _SubTensorDict(source=self, idx=idx)

    @abc.abstractmethod
    def __setitem__(
        self,
        index: IndexType,
        value: T | dict | numbers.Number | CompatibleType,
    ) -> None:
        ...

    def __delitem__(self, key: NestedKey) -> T:
        return self.del_(key)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        from ._torch_func import TD_HANDLED_FUNCTIONS

        if kwargs is None:
            kwargs = {}
        if func not in TD_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, TensorDictBase)) for t in types
        ):
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @abc.abstractmethod
    def all(self, dim: int = None) -> bool | TensorDictBase:
        """Checks if all values are True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating
                whether all tensors return `tensor.all() == True`
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.

        """
        ...

    @abc.abstractmethod
    def any(self, dim: int = None) -> bool | TensorDictBase:
        """Checks if any value is True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating
                whether all tensors return `tensor.any() == True`.
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with
                the tensordict shape.

        """
        ...

    # Module interaction
    @staticmethod
    def from_module(module, as_module: bool = False, lock: bool = True):
        """Copies the params and buffers of a module in a tensordict.

        Args:
            as_module (bool, optional): if ``True``, a :class:`~tensordict.nn.TensorDictParams`
                instance will be returned which can be used to store parameters
                within a :class:`torch.nn.Module`. Defaults to ``False``.
            lock (bool, optional): if ``True``, the resulting tensordict will be locked.
                Defaults to ``True``.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1)
            >>> params = TensorDict.from_module(module)
            >>> print(params["layers", "0", "linear1"])
            TensorDict(
                fields={
                    bias: Parameter(shape=torch.Size([2048]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Parameter(shape=torch.Size([2048, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        ...

    @abc.abstractmethod
    def to_module(self, module: "nn.Module", return_swap: bool = False, swap_dest=None):
        """Writes the content of a TensorDictBase instance onto a given nn.Module attributes, recursively.

        Args:
            module (nn.Module): a module to write the parameters into.
            return_swap (bool, optional): if ``True``, the old parameter configuration
                will be returned. Defaults to ``False``.
            swap_dest (TensorDictBase, optional): if ``return_swap`` is ``True``,
                the tensordict where the swap should be written.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1)
            >>> params = TensorDict.from_module(module)
            >>> params.zero_()
            >>> params.to_module(module)
            >>> assert (module.layers[0].linear1.weight == 0).all()
        """
        ...

    # Shape functionality
    @property
    def shape(self) -> torch.Size:
        """See :obj:`~torch.dict.TensorDictBase.batch_size`."""
        return self.batch_size

    @property
    @abc.abstractmethod
    def batch_size(self) -> torch.Size:
        """Shape (or batch_size) of a TensorDict.

        The shape of a tensordict corresponds to the common first ``N``
        dimensions of the tensors it contains, where ``N`` is an arbitrary
        number.
        The ``TensorDict`` shape is controlled by the user upon
        initialization (ie, it is not inferred from the tensor shapes).

        The ``batch_size`` can be edited dynamically if the new size is compatible
        with the TensorDict content. For instance, setting the batch size to
        an empty value is always allowed.

        Returns:
            a :obj:`~torch.Size` object describing the TensorDict batch size.

        Examples:
            >>> data = TensorDict({
            ...     "key 0": torch.randn(3, 4),
            ...     "key 1": torch.randn(3, 5),
            ...     "nested": TensorDict({"key 0": torch.randn(3, 4)}, batch_size=[3, 4])},
            ...     batch_size=[3])
            >>> data.batch_size = () # resets the batch-size to an empty value
        """
        ...

    def size(self, dim: int | None = None) -> torch.Size | int:
        """Returns the size of the dimension indicated by ``dim``.

        If ``dim`` is not specified, returns the ``batch_size`` attribute of the TensorDict.

        """
        if dim is None:
            return self.batch_size
        return self.batch_size[dim]

    def _batch_size_setter(self, new_batch_size: torch.Size) -> None:
        if new_batch_size == self.batch_size:
            return
        if self._lazy:
            raise RuntimeError(
                "modifying the batch size of a lazy representation of a "
                "tensordict is not permitted. Consider instantiating the "
                "tensordict first by calling `td = td.to_tensordict()` before "
                "resetting the batch size."
            )
        if not isinstance(new_batch_size, torch.Size):
            new_batch_size = torch.Size(new_batch_size)
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                tensordict = self.get(key)
                if len(tensordict.batch_size) < len(new_batch_size):
                    # document as edge case
                    tensordict.batch_size = new_batch_size
                    self._set_str(key, tensordict, inplace=True, validated=True)
        self._check_new_batch_size(new_batch_size)
        self._change_batch_size(new_batch_size)
        if self._has_names():
            # if the tensordict has dim names and the new batch-size has more dims,
            # we can simply add empty names after the current ones.
            # Otherwise, we discard the extra existing names.
            names = self.names
            if len(names) < len(new_batch_size):
                self.names = names + [None] * (len(new_batch_size) - len(names))
            else:
                self.names = names[: self.batch_dims]

    @property
    def batch_dims(self) -> int:
        """Length of the tensordict batch size.

        Returns:
            int describing the number of dimensions of the tensordict.

        """
        return len(self.batch_size)

    def ndimension(self) -> int:
        """See :meth:`~.batch_dims`."""
        return self.batch_dims

    @property
    def ndim(self) -> int:
        """See :meth:`~.batch_dims`."""
        return self.batch_dims

    def dim(self) -> int:
        """See :meth:`~.batch_dims`."""
        return self.batch_dims

    def numel(self) -> int:
        """Total number of elements in the batch.

        Lower-bounded to 1, as a stack of two tensordict with empty shape will
        have two elements, therefore we consider that a tensordict is at least
        1-element big.
        """
        return max(1, self.batch_size.numel())

    @overload
    def expand(self, *shape: int) -> T:
        ...

    @overload
    def expand(self, shape: torch.Size) -> T:
        ...

    @abc.abstractmethod
    def expand(self, *args: int | torch.Size) -> T:
        """Expands each tensor of the tensordict according to the :func:`~torch.expand` function, ignoring the feature dimensions.

        Supports iterables to specify the shape.

        Examples:
            >>> td = TensorDict({
            ...     'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10, 3, 4)
            >>> assert td_expand.shape == torch.Size([10, 3, 4])
            >>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])

        """
        ...

    @abc.abstractmethod
    def unbind(self, dim: int) -> tuple[T, ...]:
        """Returns a tuple of indexed tensordicts, unbound along the indicated dimension.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td0, td1, td2 = td.unbind(0)
            >>> td0['x']
            tensor([0, 1, 2, 3])
            >>> td1['x']
            tensor([4, 5, 6, 7])

        """
        ...

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        """Splits a tensordict into the specified number of chunks, if possible.

        Each chunk is a view of the input tensordict.

        Args:
            chunks (int): number of chunks to return
            dim (int, optional): dimension along which to split the
                tensordict. Default is 0.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.chunk(dim=-1, chunks=2)
            >>> td0['x']
            tensor([[[ 0,  1],
                     [ 2,  3]],
                    [[ 8,  9],
                     [10, 11]],
                    [[16, 17],
                     [18, 19]]])

        """
        if chunks < 1:
            raise ValueError(
                f"chunks must be a strictly positive integer, got {chunks}."
            )
        indices = []
        _idx_start = 0
        if chunks > 1:
            interval = _idx_end = self.batch_size[dim] // chunks
        else:
            interval = _idx_end = self.batch_size[dim]
        for c in range(chunks):
            indices.append(slice(_idx_start, _idx_end))
            _idx_start = _idx_end
            _idx_end = _idx_end + interval if c < chunks - 2 else self.batch_size[dim]
        if dim < 0:
            dim = len(self.batch_size) + dim
        return tuple(self[(*[slice(None) for _ in range(dim)], idx)] for idx in indices)

    def unsqueeze(self, dim: int) -> T:
        """Unsqueezes all tensors for a dimension comprised in between `-td.batch_dims` and `td.batch_dims` and returns them in a new tensordict.

        Args:
            dim (int): dimension along which to unsqueeze

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 4, 2),
            ... }, batch_size=[3, 4])
            >>> td = td.unsqueeze(-2)
            >>> td.shape
            torch.Size([3, 1, 4])
            >>> td.get("x").shape
            torch.Size([3, 1, 4, 2])
        """
        # make the dim positive
        if dim < 0:
            newdim = self.batch_dims + dim + 1
        else:
            newdim = dim

        if (newdim > self.batch_dims) or (newdim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims - 1` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        batch_size = list(self.batch_size)
        batch_size.insert(newdim, 1)
        batch_size = torch.Size(batch_size)

        names = copy(self.names)
        names.insert(dim, None)

        def _unsqueeze(tensor):
            return tensor.unsqueeze(newdim)

        return self._fast_apply(
            _unsqueeze,
            batch_size=batch_size,
            names=names,
            inplace=False,
            call_on_nested=True,
        )

    def squeeze(self, dim: int | None = None) -> T:
        """Squeezes all tensors for a dimension in between `-self.batch_dims+1` and `self.batch_dims-1` and returns them in a new tensordict.

        Args:
            dim (Optional[int]): dimension along which to squeeze. If dim is
                ``None``, all singleton dimensions will be squeezed.
                Defaults to ``None``.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(24).reshape(3, 1, 4, 2),
            ... }, batch_size=[3, 1, 4])
            >>> td = td.unsqueeze()
            >>> td.shape
            torch.Size([3, 4])
            >>> td.get("x").shape
            torch.Size([3, 4, 2])

        """
        batch_size = self.batch_size
        if dim is None:
            names = list(self.names)
            batch_size, names = zip(
                *[(size, name) for size, name in zip(batch_size, names) if size != 1]
            )
            batch_size = torch.Size(batch_size)
            if batch_size == self.batch_size:
                return self

            # we only want to squeeze dimensions lower than the batch dim, and view
            # is the perfect op for this
            def _squeeze(tensor):
                return tensor.view(*batch_size, *tensor.shape[self.batch_dims :])

            return self._fast_apply(
                _squeeze,
                batch_size=batch_size,
                names=names,
                inplace=False,
                call_on_nested=True,
            )
        # make the dim positive
        if dim < 0:
            newdim = self.batch_dims + dim
        else:
            newdim = dim

        if (newdim >= self.batch_dims) or (newdim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims - 1` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        if batch_size[dim] != 1:
            return self
        batch_size = list(batch_size)
        batch_size.pop(dim)
        batch_size = list(batch_size)
        names = list(self.names)
        names.pop(dim)

        return self._fast_apply(
            lambda x: x.squeeze(newdim),
            batch_size=batch_size,
            names=names,
            inplace=False,
            call_on_nested=True,
        )

    @overload
    def reshape(self, *shape: int):
        ...

    @overload
    def reshape(self, shape: list | tuple):
        ...

    @abc.abstractmethod
    def reshape(
        self,
        *args,
        **kwargs,
    ) -> T:
        """Returns a contiguous, reshaped tensor of the desired shape.

        Args:
            *shape (int): new shape of the resulting tensordict.

        Returns:
            A TensorDict with reshaped keys

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td = td.reshape(12)
            >>> print(td['x'])
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        """
        ...

    @abc.abstractmethod
    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        """Splits each tensor in the TensorDict with the specified size in the given dimension, like `torch.split`.

        Returns a list of ``TensorDict`` instances with the view of split chunks of items.

        Args:
            split_size (int or List(int)): size of a single chunk or list of sizes for each chunk.
            dim (int): dimension along which to split the tensor.

        Returns:
            A list of TensorDict with specified size in given dimension.

        Examples:
            >>> td = TensorDict({
            ...     'x': torch.arange(12).reshape(3, 4),
            ... }, batch_size=[3, 4])
            >>> td0, td1 = td.split([1, 2], dim=0)
            >>> print(td0['x'])
            torch.Tensor([[0, 1, 2, 3]])
        """
        ...

    def gather(self, dim: int, index: Tensor, out: T | None = None) -> T:
        """Gathers values along an axis specified by `dim`.

        Args:
            dim (int): the dimension along which collect the elements
            index (torch.Tensor): a long tensor which number of dimension matches
                the one of the tensordict with only one dimension differring between
                the two (the gathering dimension). Its elements refer to the
                index to be gathered along the required dimension.
            out (TensorDictBase, optional): a destination tensordict. It must
                have the same shape as the index.

        Examples:
            >>> td = TensorDict(
            ...     {"a": torch.randn(3, 4, 5),
            ...      "b": TensorDict({"c": torch.zeros(3, 4, 5)}, [3, 4, 5])},
            ...     [3, 4])
            >>> index = torch.randint(4, (3, 2))
            >>> td_gather = td.gather(dim=1, index=index)
            >>> print(td_gather)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 2, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 2]),
                device=None,
                is_shared=False)

        Gather keeps the dimension names.

        Examples:
            >>> td.names = ["a", "b"]
            >>> td_gather = td.gather(dim=1, index=index)
            >>> td_gather.names
            ["a", "b"]
        """
        return torch.gather(self, dim, index, out=out)

    @overload
    def view(self, *shape: int):
        ...

    @overload
    def view(self, shape):
        ...

    def view(
        self,
        *args,
        **kwargs,
    ) -> T:
        """Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict batch_size.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns:
            a new tensordict with the desired batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5),
            ...    'b': torch.zeros(3,4,10,1)}, batch_size=torch.Size([3, 4]))
            >>> td_view = td.view(12)
            >>> print(td_view.get("a").shape)  # torch.Size([12, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([12, 10, 1])
            >>> td_view = td.view(-1, 4, 3)
            >>> print(td_view.get("a").shape)  # torch.Size([1, 4, 3, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([1, 4, 3, 10, 1])

        """
        ...

    def transpose(self, dim0, dim1):
        """Returns a tensordit that is a transposed version of input. The given dimensions ``dim0`` and ``dim1`` are swapped.

        In-place or out-place modifications of the transposed tensordict will
        impact the original tensordict too as the memory is shared and the operations
        are mapped back on the original tensordict.

        Examples:
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> tensordict_transpose = tensordict.transpose(0, 1)
            >>> print(tensordict_transpose.shape)
            torch.Size([4, 3])
            >>> tensordict_transpose.set("b",, torch.randn(4, 3))
            >>> print(tensordict.get("b").shape)
            torch.Size([3, 4])
        """
        ...

    @overload
    def permute(self, *dims: int):
        ...

    @overload
    def permute(self, dims: list | tuple):
        ...

    def permute(
        self,
        *args,
        **kwargs,
    ) -> T:
        """Returns a view of a tensordict with the batch dimensions permuted according to dims.

        Args:
            *dims_list (int): the new ordering of the batch dims of the tensordict. Alternatively,
                a single iterable of integers can be provided.
            dims (list of int): alternative way of calling permute(...).

        Returns:
            a new tensordict with the batch dimensions in the desired order.

        Examples:
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> print(tensordict.permute([1, 0]))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
            >>> print(tensordict.permute(1, 0))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
            >>> print(tensordict.permute(dims=[1, 0]))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
        """
        ...

    # Cache functionality
    def _erase_cache(self):
        self._cache = None

    # Dim names functionality
    @property
    @abc.abstractmethod
    def names(self):
        ...

    @abc.abstractmethod
    def _erase_names(self):
        """Erases the dimension names from a tensordict."""
        ...

    @abc.abstractmethod
    def _rename_subtds(self, value):
        """Renames all the sub-tensordicts dimension according to value.
        If value has less dimensions than the TD, the rest is just assumed to be None.
        """
        ...

    def _check_dim_name(self, name):
        if name is None:
            return False
        if self._has_names() and name in self.names:
            return True
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                if self._get_str(key, NO_DEFAULT)._check_dim_name(name):
                    return True
        else:
            return False

    def refine_names(self, *names):
        """Refines the dimension names of self according to names.

        Refining is a special case of renaming that “lifts” unnamed dimensions.
        A None dim can be refined to have any name; a named dim can only be
        refined to have the same name.

        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.

        names may contain up to one Ellipsis (...). The Ellipsis is expanded
        greedily; it is expanded in-place to fill names to the same length as
        self.dim() using names from the corresponding indices of self.names.

        Returns: the same tensordict with dimensions named according to the input.

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4, 5, 6])
            >>> tdr = td.refine_names(None, None, None, "d")
            >>> assert tdr.names == [None, None, None, "d"]
            >>> tdr = td.refine_names("a", None, None, "d")
            >>> assert tdr.names == ["a", None, None, "d"]

        """
        # replace ellipsis if any
        names_copy = copy(names)
        if any(name is Ellipsis for name in names):
            ellipsis_name = [NO_DEFAULT for _ in range(self.ndim - len(names) + 1)]
            names = []
            for name in names_copy:
                if name is Ellipsis:
                    names += ellipsis_name
                else:
                    names.append(name)
        # check that the names that are set are either None or identical
        curr_names = self.names
        for i, name in enumerate(names):
            if name is NO_DEFAULT:
                # whatever value is ok
                names[i] = curr_names[i]
                continue
            else:
                if curr_names[i] is None:
                    continue
                if self.names[i] == name:
                    continue
                else:
                    raise RuntimeError(
                        f"refine_names: cannot coerce TensorDict names {self.names} with {names_copy}."
                    )
        self.names = names
        # we also need to rename the sub-tensordicts
        # self._rename_subtds(self.names)
        return self

    def rename(self, *names, **rename_map):
        """Returns a clone of the tensordict with dimensions renamed.

        Examples:
            >>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
            >>> td.names = list("abcd")
            >>> td_rename = td.rename(c="g")
            >>> assert td_rename.names == list("abgd")

        """
        clone = self.clone(recurse=False)
        if len(names) == 1 and names[0] is None:
            clone.names = None
        if rename_map and names:
            raise ValueError(
                "Passed both a name map and a name list. Only one is accepted."
            )
        elif not rename_map and not names:
            raise ValueError(
                "Neither a name map nor a name list was passed. "
                "Only one is accepted."
            )
        elif rename_map:
            cnames = list(clone.names)
            for i, name in enumerate(cnames):
                new_name = rename_map.pop(name, NO_DEFAULT)
                if new_name is not NO_DEFAULT:
                    cnames[i] = new_name
            clone.names = cnames
            if rename_map:
                raise ValueError(
                    f"Some names to be renamed were not part of the tensordict names: {rename_map.keys()} vs {self.names}."
                )
        else:
            clone.names = names
        return clone

    def rename_(self, *names, **rename_map):
        """Same as :meth:`~.rename`, but executes the renaming in-place.

        Examples:
            >>> td = TensorDict({}, batch_size=[1, 2, 3 ,4])
            >>> td.names = list("abcd")
            >>> assert td.rename_(c="g")
            >>> assert td.names == list("abgd")
        """
        if len(names) == 1 and names[0] is None:
            self.names = None
        if rename_map and names:
            raise ValueError(
                "Passed both a name map and a name list. " "Only one is accepted."
            )
        elif not rename_map and not names and self.batch_dims:
            raise ValueError(
                "Neither a name map nor a name list was passed. "
                "Only one is accepted."
            )
        elif rename_map:
            cnames = list(self.names)
            for i, name in enumerate(cnames):
                new_name = rename_map.pop(name, NO_DEFAULT)
                if new_name is not NO_DEFAULT:
                    cnames[i] = new_name
            if rename_map:
                raise ValueError(
                    f"Some names to be renamed were not part of the tensordict names: {rename_map.keys()} vs {self.names}."
                )
            self.names = cnames
        else:
            self.names = names
        return self

    @abc.abstractmethod
    def _has_names(self):
        ...

    # Device functionality: device is optional. If provided, it will enforce
    # all data is on the same device
    @property
    @abc.abstractmethod
    def device(self) -> torch.device | None:
        """Device of a TensorDict.

        If the TensorDict has a specified device, all
        its tensors (incl. nested ones) must live on the same device.
        If the TensorDict device is ``None``, different values can be located
        on different devices.

        Returns:
            torch.device object indicating the device where the tensors
            are placed, or None if TensorDict does not have a device.

        Examples:
            >>> td = TensorDict({
            ...     "cpu": torch.randn(3, device='cpu'),
            ...     "cuda": torch.randn(3, device='cuda'),
            ... }, batch_size=[], device=None)
            >>> td['cpu'].device
            device(type='cpu')
            >>> td['cuda'].device
            device(type='cuda')
            >>> td = TensorDict({
            ...     "x": torch.randn(3, device='cpu'),
            ...     "y": torch.randn(3, device='cuda'),
            ... }, batch_size=[], device='cuda')
            >>> td['x'].device
            device(type='cuda')
            >>> td['y'].device
            device(type='cuda')
            >>> td = TensorDict({
            ...     "x": torch.randn(3, device='cpu'),
            ...     "y": TensorDict({'z': torch.randn(3, device='cpu')}, batch_size=[], device=None),
            ... }, batch_size=[], device='cuda')
            >>> td['x'].device
            device(type='cuda')
            >>> td['y'].device # nested tensordicts are also mapped onto the appropriate device.
            device(type='cuda')
            >>> td['y', 'x'].device
            device(type='cuda')

        """
        ...

    @device.setter
    @abc.abstractmethod
    def device(self, value: DeviceType) -> None:
        ...

    def clear_device_(self) -> T:
        """Clears the device of the tensordict.

        Returns: self

        """
        self._device = None
        for value in self.values():
            if _is_tensor_collection(value.__class__):
                value.clear_device_()
        return self

    @abc.abstractmethod
    def pin_memory(self) -> T:
        """Calls :meth:`~torch.Tensor.pin_memory` on the stored tensors."""
        ...

    def cpu(self) -> T:
        """Casts a tensordict to CPU."""
        return self.to("cpu")

    def cuda(self, device: int = None) -> T:
        """Casts a tensordict to a cuda device (if not already on it).

        Args:
            device (int, optional): if provided, the cuda device on which the
                tensor should be cast.

        """
        if device is None:
            return self.to(torch.device("cuda"))
        return self.to(f"cuda:{device}")

    # Serialization functionality
    def state_dict(
        self,
        destination=None,
        prefix="",
        keep_vars=False,
        flatten=False,
    ) -> OrderedDict[str, Any]:
        """Produces a state_dict from the tensordict.

        The structure of the state-dict will still be nested, unless ``flatten`` is set to ``True``.

        A tensordict state-dict contains all the tensors and meta-data needed
        to rebuild the tensordict (names are currently not supported).

        Args:
            destination (dict, optional): If provided, the state of tensordict will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to tensor
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            flatten (bool, optional): whether the structure should be flattened
                with the ``"."`` character or not.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> sd = data.state_dict()
            >>> print(sd)
            OrderedDict([('1', tensor(1)), ('2', tensor(2)), ('3', OrderedDict([('3', tensor(3)), ('__batch_size', torch.Size([])), ('__device', None)])), ('__batch_size', torch.Size([])), ('__device', None)])
            >>> sd = data.state_dict(flatten=True)
            OrderedDict([('1', tensor(1)), ('2', tensor(2)), ('3.3', tensor(3)), ('__batch_size', torch.Size([])), ('__device', None)])

        """
        out = collections.OrderedDict()
        source = self
        if flatten:
            source = source.flatten_keys(".")
        for key, item in source.items():
            if not _is_tensor_collection(item.__class__):
                if not keep_vars:
                    out[prefix + key] = item.detach().clone()
                else:
                    out[prefix + key] = item
            else:
                out[prefix + key] = item.state_dict(keep_vars=keep_vars)
        if "__batch_size" in out:
            raise KeyError(
                "Cannot retrieve the state_dict of a TensorDict with `'__batch_size'` key"
            )
        if "__device" in out:
            raise KeyError(
                "Cannot retrieve the state_dict of a TensorDict with `'__batch_size'` key"
            )
        out[prefix + "__batch_size"] = source.batch_size
        out[prefix + "__device"] = source.device
        if destination is not None:
            destination.update(out)
            return destination
        return out

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict=True,
        assign=False,
        from_flatten=False,
    ) -> T:
        """Loads a state-dict, formatted as in :meth:`~.state_dict`, into the tensordict.

        Args:
            state_dict (OrderedDict): the state_dict of to be copied.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this tensordict's
                :meth:`torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): whether to assign items in the state
                dictionary to their corresponding keys in the tensordict instead
                of copying them inplace into the tensordict's current tensors.
                When ``False``, the properties of the tensors in the current
                module are preserved while when ``True``, the properties of the
                Tensors in the state dict are preserved.
                Default: ``False``
            from_flatten (bool, optional): if ``True``, the input state_dict is
                assumed to be flattened.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
            >>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
            >>> sd = data.state_dict()
            >>> data_zeroed.load_state_dict(sd)
            >>> print(data_zeroed["3", "3"])
            tensor(3)
            >>> # with flattening
            >>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
            >>> data_zeroed.load_state_dict(data.state_dict(flatten=True), from_flatten=True)
            >>> print(data_zeroed["3", "3"])
            tensor(3)


        """
        if from_flatten:
            self_flatten = self.flatten_keys(".")
            self_flatten.load_state_dict(state_dict, strict=strict, assign=assign)
            if not assign:
                # modifications are done in-place so we should be fine returning self
                return self
            else:
                # run a check over keys, if we any key with a '.' in name we're doomed
                DOT_ERROR = "Cannot use load_state_dict(..., from_flatten=True, assign=True) when some keys contain a dot character."
                for key in self.keys(True, True):
                    if isinstance(key, tuple):
                        for subkey in key:
                            if "." in subkey:
                                raise RuntimeError(DOT_ERROR)
                    elif "." in key:
                        raise RuntimeError(DOT_ERROR)
                return self.update(self_flatten.unflatten_keys("."))
        # copy since we'll be using pop
        state_dict = copy(state_dict)
        self.batch_size = state_dict.pop("__batch_size")
        device = state_dict.pop("__device", None)
        if device is not None and self.device is not None and device != self.device:
            raise RuntimeError("Loading data from another device is not yet supported.")

        for key, item in state_dict.items():
            if isinstance(item, dict):
                dest = self.get(key, default=None)
                if dest is None:
                    dest = self.empty()
                dest.load_state_dict(item, assign=assign, strict=strict)
                self.set(
                    key,
                    dest,
                    inplace=not assign,
                )
            else:
                self.set(key, item, inplace=not assign)
        if strict and set(state_dict.keys()) != set(self.keys()):
            set_sd = set(state_dict.keys())
            set_td = set(self.keys())
            raise RuntimeError(
                "Cannot load state-dict because the key sets don't match: got "
                f"state_dict extra keys \n{set_sd - set_td}\n and tensordict extra keys\n{set_td - set_sd}\n"
            )
        return self

    def is_shared(self) -> bool:
        """Checks if tensordict is in shared memory.

        If a TensorDict instance is in shared memory, it is locked (entries cannot
        be renamed, removed or added). If a ``TensorDict`` is created with
        tensors that are all in shared memory, this does __not__ mean that ``is_shared``
        will return ``True`` (as a new tensor may or may not be in shared memory).
        Only if one calls `tensordict.share_memory_()` or places the tensordict
        on a device where the content is shared by default (eg, ``"cuda"``)
        will the tensordict be considered in shared memory.

        This is always ``True`` for tensordicts on a CUDA device.

        """
        if self.device and not self._is_memmap:
            return self.device.type == "cuda" or self._is_shared
        return self._is_shared

    def is_memmap(self) -> bool:
        """Checks if tensordict is memory-mapped.

        If a TensorDict instance is memory-mapped, it is locked (entries cannot
        be renamed, removed or added). If a ``TensorDict`` is created with
        tensors that are all memory-mapped, this does __not__ mean that ``is_memmap``
        will return ``True`` (as a new tensor may or may not be memory-mapped).
        Only if one calls `tensordict.memmap_()` will the tensordict be
        considered as memory-mapped.

        This is always ``True`` for tensordicts on a CUDA device.

        """
        return self._is_memmap

    @abc.abstractmethod
    def share_memory_(self) -> T:
        """Places all the tensors in shared memory.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Conversely, once the tensordict is unlocked, the share_memory attribute
        is turned to ``False``, because cross-process identity is not
        guaranteed anymore.

        Returns:
            self

        """
        ...

    @abc.abstractmethod
    def memmap_(self, prefix: str | None = None, copy_existing: bool = False) -> T:
        """Writes all tensors onto a corresponding memory-mapped Tensor.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.
            copy_existing (bool): If False (default), an exception will be raised if an
                entry in the tensordict is already a tensor stored on disk
                with an associated file, but is not saved in the correct
                location according to prefix.
                If ``True``, any existing Tensor will be copied to the new location.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            self.

        Note:
            Serialising in this fashion might be slow with deeply nested tensordicts, so
            it is not recommended to call this method inside a training loop.
        """
        ...

    @abc.abstractmethod
    def memmap_like(self, prefix: str | None = None) -> T:
        """Creates a contentless Memory-mapped tensordict with the same shapes as the original one.

        Args:
            prefix (str): directory prefix where the memory-mapped tensors will
                be stored. The directory tree structure will mimic the tensordict's.

        The TensorDict is then locked, meaning that any writing operations that
        isn't in-place will throw an exception (eg, rename, set or remove an
        entry).
        Once the tensordict is unlocked, the memory-mapped attribute is turned to ``False``,
        because cross-process identity is not guaranteed anymore.

        Returns:
            a new ``TensorDict`` instance with data stored as memory-mapped tensors.

        .. note:: This is the recommended method to write a set of large buffers
            on disk, as :meth:`~.memmap_()` will copy the information, which can
            be slow for large content.

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
            ...     "b": torch.zeros(1, dtype=torch.int64),
            ... }, batch_size=[]).expand(1_000_000)  # expand does not allocate new memory
            >>> buffer = td.memmap_like("/path/to/dataset")

        """
        ...

    # Key functionality: set, get, set_, set_at_, update, update_
    @abc.abstractmethod
    def entry_class(self, key: NestedKey) -> type:
        """Returns the class of an entry, possibly avoiding a call to `isinstance(td.get(key), type)`.

        This method should be preferred to ``tensordict.get(key).shape`` whenever
        :meth:`.get` can be expensive to execute.

        """
        ...

    def set(
        self, key: NestedKey, item: CompatibleType, inplace: bool = False, **kwargs: Any
    ) -> T:
        """Sets a new key-value pair.

        Args:
            key (str, tuple of str): name of the key to be set.
            item (torch.Tensor or equivalent, TensorDictBase instance): value
                to be stored in the tensordict.
            inplace (bool, optional): if ``True`` and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. If inplace is ``True`` and
                the entry cannot be found, it will be added. For a more restrictive
                in-place operation, use :meth:`~.set_` instead.
                Defaults to ``False``.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size[3, 4])
            >>> td.set("x", torch.randn(3, 4))
            >>> y = torch.randn(3, 4, 5)
            >>> td.set("y", y, inplace=True) # works, even if 'y' is not present yet
            >>> td.set("y", torch.zeros_like(y), inplace=True)
            >>> assert (y==0).all() # y values are overwritten
            >>> td.set("y", torch.ones(5), inplace=True) # raises an exception as shapes mismatch

        """
        key = _unravel_key_to_tuple(key)
        # inplace is loose here, but for set_ it is constraining. We translate it
        # to None to tell _set_str and others to drop it if the key isn't found
        inplace = BEST_ATTEMPT_INPLACE if inplace else False
        return self._set_tuple(key, item, inplace=inplace, validated=False)

    @abc.abstractmethod
    def _set_str(self, key, value, *, inplace, validated):
        ...

    @abc.abstractmethod
    def _set_tuple(self, key, value, *, inplace, validated):
        ...

    def _convert_inplace(self, inplace, key):
        if inplace is not False:
            has_key = key in self.keys()
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    _KEY_ERROR.format(key, self.__class__.__name__, sorted(self.keys()))
                )
            inplace = has_key
        return inplace

    def set_at_(self, key: NestedKey, value: CompatibleType, index: IndexType) -> T:
        """Sets the values in-place at the index indicated by ``index``.

        Args:
            key (str, tuple of str): key to be modified.
            value (torch.Tensor): value to be set at the index `index`
            index (int, tensor or tuple): index where to write the values.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size[3, 4])
            >>> x = torch.randn(3, 4)
            >>> td.set("x", x)
            >>> td.set_at_("x", value=torch.ones(1, 4), index=slice(1))
            >>> assert (x[0] == 1).all()
        """
        key = _unravel_key_to_tuple(key)
        return self._set_at_tuple(key, value, index, validated=False)

    @abc.abstractmethod
    def _set_at_str(self, key, value, idx, *, validated):
        ...

    @abc.abstractmethod
    def _set_at_tuple(self, key, value, idx, *, validated):
        ...

    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
    ) -> T:
        """Sets a value to an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            item (torch.Tensor or compatible type, TensorDictBase): value to
                be stored in the tensordict

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size[3, 4])
            >>> x = torch.randn(3, 4)
            >>> td.set("x", x)
            >>> td.set_("x", torch.zeros_like(x))
            >>> assert (x == 0).all()

        """
        key = _unravel_key_to_tuple(key)
        return self._set_tuple(key, item, inplace=True, validated=False)

    # Stack functionality
    @abc.abstractmethod
    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        """Stacks a list of values onto an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            list_item (list of torch.Tensor): value to be stacked and stored in the tensordict.
            dim (int): dimension along which the tensors should be stacked.

        Returns:
            self

        """
        ...

    def _stack_onto_at_(
        self,
        key: str,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> T:
        """Similar to _stack_onto_ but on a specific index. Only works with regular TensorDicts."""
        raise RuntimeError(
            f"Cannot call _stack_onto_at_ with {self.__class__.__name__}. "
            "Make sure your sub-classed tensordicts are turned into regular tensordicts by calling to_tensordict() "
            "before calling __getindex__ and stack."
        )

    def _default_get(
        self, key: str, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        if default is not NO_DEFAULT:
            return default
        else:
            # raise KeyError
            raise KeyError(
                _KEY_ERROR.format(key, self.__class__.__name__, sorted(self.keys()))
            )

    def get(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        """Gets the value stored with the input key.

        Args:
            key (str, tuple of str): key to be queried. If tuple of str it is
                equivalent to chained calls of getattr.
            default: default value if the key is not found in the tensordict.

        Examples:
            >>> td = TensorDict({"x": 1}, batch_size=[])
            >>> td.get("x")
            tensor(1)
            >>> td.get("y", default=None)
            None
        """
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        return self._get_tuple(key, default=default)

    @abc.abstractmethod
    def _get_str(self, key, default):
        ...

    @abc.abstractmethod
    def _get_tuple(self, key, default):
        ...

    def get_at(
        self, key: NestedKey, index: IndexType, default: CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        """Get the value of a tensordict from the key `key` at the index `idx`.

        Args:
            key (str, tuple of str): key to be retrieved.
            index (int, slice, torch.Tensor, iterable): index of the tensor.
            default (torch.Tensor): default value to return if the key is
                not present in the tensordict.

        Returns:
            indexed tensor.

        Examples:
            >>> td = TensorDict({"x": torch.arange(3)}, batch_size=[])
            >>> td.get_at("x", index=1)
            tensor(1)

        """
        # TODO: check that this works with masks, and add to docstring
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        # must be a tuple
        return self._get_at_tuple(key, index, default)

    def _get_at_str(self, key, idx, default):
        out = self._get_str(key, default)
        if out is default:
            return out
        return out[idx]

    def _get_at_tuple(self, key, idx, default):
        out = self._get_tuple(key, default)
        if out is default:
            return out
        return out[idx]

    def get_item_shape(self, key: NestedKey):
        """Returns the shape of the entry, possibly avoiding recurring to :meth:`~.get`."""
        return _shape(self.get(key))

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        inplace: bool = False,
    ) -> T:
        """Updates the TensorDict with values from either a dictionary or another TensorDict.

        Args:
            input_dict_or_td (TensorDictBase or dict): input data to be written
                in self.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set.
                Defaults to ``False``.
            inplace (bool, optional): if ``True`` and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. If the entry cannot be found, it will be
                added. Defaults to ``False``.

        Returns:
            self

        Examples:
            >>> td = TensorDict({}, batch_size=[3])
            >>> a = torch.randn(3)
            >>> b = torch.randn(3, 4)
            >>> other_td = TensorDict({"a": a, "b": b}, batch_size=[])
            >>> td.update(other_td, inplace=True) # writes "a" and "b" even though they can't be found
            >>> assert td['a'] is other_td['a']
            >>> other_td = other_td.clone().zero_()
            >>> td.update(other_td)
            >>> assert td['a'] is not other_td['a']

        """
        # TODO: allow to pass a list of keys to update if not everything needs to be updated
        if input_dict_or_td is self:
            # no op
            return self
        keys = set(self.keys(False))
        for key, value in list(input_dict_or_td.items()):
            if clone and hasattr(value, "clone"):
                value = value.clone()
            if isinstance(key, tuple):
                key, subkey = key[0], key[1:]
            else:
                subkey = []
            # the key must be a string by now. Let's check if it is present
            if key in keys:
                target_type = self.entry_class(key)
                if _is_tensor_collection(target_type):
                    target = self.get(key)
                    if len(subkey):
                        target.update({subkey: value}, inplace=inplace, clone=clone)
                        continue
                    elif isinstance(value, (dict,)) or _is_tensor_collection(
                        value.__class__
                    ):
                        target.update(value, inplace=inplace, clone=clone)
                        continue
            if len(subkey):
                self.set((key, *subkey), value, inplace=inplace)
            else:
                self.set(key, value, inplace=inplace)
        return self

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
    ) -> T:
        """Updates the TensorDict in-place with values from either a dictionary or another TensorDict.

        Unlike :meth:`~.update`, this function will throw an error if the key is unknown to ``self``.

        Args:
            input_dict_or_td (TensorDictBase or dict): input data to be written
                in self.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Defaults to ``False``.

        Returns:
            self

        Examples:
            >>> a = torch.randn(3)
            >>> b = torch.randn(3, 4)
            >>> td = TensorDict({"a": a, "b": b}, batch_size=[3])
            >>> other_td = TensorDict({"a": a*0, "b": b*0}, batch_size=[])
            >>> td.update_(other_td)
            >>> assert td['a'] is not other_td['a']
            >>> assert (td['a'] == other_td['a']).all()
            >>> assert (td['a'] == 0).all()

        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            # if not isinstance(value, _accepted_classes):
            #     raise TypeError(
            #         f"Expected value to be one of types {_accepted_classes} "
            #         f"but got {type(value)}"
            #     )
            if clone:
                value = value.clone()
            self.set_(key, value)
        return self

    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool = False,
    ) -> T:
        """Updates the TensorDict in-place at the specified index with values from either a dictionary or another TensorDict.

        Unlike  TensorDict.update, this function will throw an error if the key is unknown to the TensorDict.

        Args:
            input_dict_or_td (TensorDictBase or dict): input data to be written
                in self.
            idx (int, torch.Tensor, iterable, slice): index of the tensordict
                where the update should occur.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.

        Returns:
            self

        Examples:
            >>> td = TensorDict({
            ...     'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td.update_at_(
            ...     TensorDict({
            ...         'a': torch.ones(1, 4, 5),
            ...         'b': torch.ones(1, 4, 10)}, batch_size=[1, 4]),
            ...    slice(1, 2))
            TensorDict(
                fields={
                    a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32),
                    b: Tensor(torch.Size([3, 4, 10]), dtype=torch.float32)},
                batch_size=torch.Size([3, 4]),
                device=None,
                is_shared=False)
            >>> assert (td[1] == 1).all()

        """
        for key, value in input_dict_or_td.items():
            if not isinstance(value, tuple(_ACCEPTED_CLASSES)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(key, value, idx)
        return self

    @lock_blocked
    def create_nested(self, key):
        """Creates a nested tensordict of the same shape, device and dim names as the current tensordict.

        If the value already exists, it will be overwritten by this operation.
        This operation is blocked in locked tensordicts.

        Examples:
            >>> data = TensorDict({}, [3, 4, 5])
            >>> data.create_nested("root")
            >>> data.create_nested(("some", "nested", "value"))
            >>> print(data)
            TensorDict(
                fields={
                    root: TensorDict(
                        fields={
                        },
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False),
                    some: TensorDict(
                        fields={
                            nested: TensorDict(
                                fields={
                                    value: TensorDict(
                                        fields={
                                        },
                                        batch_size=torch.Size([3, 4, 5]),
                                        device=None,
                                        is_shared=False)},
                                batch_size=torch.Size([3, 4, 5]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([3, 4, 5]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3, 4, 5]),
                device=None,
                is_shared=False)
        """
        key = _unravel_key_to_tuple(key)
        self._create_nested_tuple(key)
        return self

    def _create_nested_str(self, key):
        self._set_str(key, self.select(), inplace=False, validated=True)

    def _create_nested_tuple(self, key):
        self._create_nested_str(key[0])
        if len(key) > 1:
            td = self._get_str(key[0], NO_DEFAULT)
            td._create_nested_tuple(key[1:])

    def copy_(self, tensordict: T) -> T:
        """See :obj:`TensorDictBase.update_`."""
        return self.update_(tensordict)

    def copy_at_(self, tensordict: T, idx: IndexType) -> T:
        """See :obj:`TensorDictBase.update_at_`."""
        return self.update_at_(tensordict, idx)

    def is_empty(self) -> bool:
        """Checks if the tensordict contains any leaf."""
        for _ in self.keys(True, True):
            return False
        return True

    # Dict features: setdefault, items, values, keys, ...
    def setdefault(
        self, key: NestedKey, default: CompatibleType, inplace: bool = False
    ) -> CompatibleType:
        """Insert the ``key`` entry with a value of ``default`` if ``key`` is not in the tensordict.

        Return the value for ``key`` if ``key`` is in the tensordict, else ``default``.

        Args:
            key (str or nested key): the name of the value.
            default (torch.Tensor or compatible type, TensorDictBase): value
                to be stored in the tensordict if the key is not already present.

        Returns:
            The value of key in the tensordict. Will be default if the key was not
            previously set.

        Examples:
            >>> td = TensorDict({}, batch_size=[3, 4])
            >>> val = td.setdefault("a", torch.zeros(3, 4))
            >>> assert (val == 0).all()
            >>> val = td.setdefault("a", torch.ones(3, 4))
            >>> assert (val == 0).all() # output is still 0

        """
        if key not in self.keys(include_nested=isinstance(key, tuple)):
            self.set(key, default, inplace=inplace)
        return self.get(key)

    def items(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[tuple[str, CompatibleType]]:
        """Returns a generator of key-value pairs for the tensordict."""
        # check the conditions once only
        if include_nested and leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if _is_tensor_collection(val.__class__):
                    yield from (
                        (_unravel_key_to_tuple((k, _key)), _val)
                        for _key, _val in val.items(
                            include_nested=include_nested, leaves_only=leaves_only
                        )
                    )
                else:
                    yield k, val
        elif include_nested:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                yield k, val
                if _is_tensor_collection(val.__class__):
                    yield from (
                        (_unravel_key_to_tuple((k, _key)), _val)
                        for _key, _val in val.items(
                            include_nested=include_nested, leaves_only=leaves_only
                        )
                    )
        elif leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if not _is_tensor_collection(val.__class__):
                    yield k, val
        else:
            for k in self.keys():
                yield k, self._get_str(k, NO_DEFAULT)

    def values(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[CompatibleType]:
        """Returns a generator representing the values for the tensordict."""
        # check the conditions once only
        if include_nested and leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if _is_tensor_collection(val.__class__):
                    yield from val.values(
                        include_nested=include_nested, leaves_only=leaves_only
                    )
                else:
                    yield val
        elif include_nested:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                yield val
                if _is_tensor_collection(val.__class__):
                    yield from val.values(
                        include_nested=include_nested, leaves_only=leaves_only
                    )
        elif leaves_only:
            for k in self.keys():
                val = self._get_str(k, NO_DEFAULT)
                if not _is_tensor_collection(val.__class__):
                    yield val
        else:
            for k in self.keys():
                yield self._get_str(k, NO_DEFAULT)

    @abc.abstractmethod
    def keys(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> "_TensorDictKeysView":
        """Returns a generator of tensordict keys."""
        ...

    def pop(self, key: NestedKey, default: Any = NO_DEFAULT) -> CompatibleType:
        """Removes and returns a value from a tensordict.

        If the value is not present and no default value is provided, a KeyError
        is thrown.

        Args:
            key (str or nested key): the entry to look for.
            default (Any, optional): the value to return if the key cannot be found.

        Examples:
            >>> td = TensorDict({"1": 1}, [])
            >>> one = td.pop("1")
            >>> assert one == 1
            >>> none = td.pop("1", default=None)
            >>> assert none is None
        """
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        try:
            # using try/except for get/del is suboptimal, but
            # this is faster that checkink if key in self keys
            out = self.get(key, default)
            self.del_(key)
        except KeyError as err:
            # if default provided, 'out' value will return, else raise error
            if default == NO_DEFAULT:
                raise KeyError(
                    f"You are trying to pop key `{key}` which is not in dict "
                    f"without providing default value."
                ) from err
        return out

    @property
    @cache  # noqa: B019
    def sorted_keys(self) -> list[NestedKey]:
        """Returns the keys sorted in alphabetical order.

        Does not support extra arguments.

        If the TensorDict is locked, the keys are cached until the tensordict
        is unlocked for faster execution.

        """
        return sorted(self.keys())

    def flatten(self, start_dim=0, end_dim=-1):
        """Flattens all the tensors of a tensordict.

        Args:
            start_dim (int): the first dim to flatten
            end_dim (int): the last dim to flatten

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)}, batch_size=[3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_flat.batch_size
            torch.Size([12])
            >>> td_flat["a"]
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  6,  7,  8,  9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 46, 47, 48, 49],
                    [50, 51, 52, 53, 54],
                    [55, 56, 57, 58, 59]])
            >>> td_flat["b"]
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

        """
        if end_dim < 0:
            end_dim = self.ndim + end_dim
            if end_dim < 0:
                raise ValueError(
                    f"Incompatible end_dim {end_dim} for tensordict with shape {self.shape}."
                )
        if end_dim <= start_dim:
            raise ValueError(
                "The end dimension must be strictly greater than the start dim."
            )

        def flatten(tensor):
            return torch.flatten(tensor, start_dim, end_dim)

        nelt = prod(self.batch_size[start_dim : end_dim + 1])
        if start_dim > 0:
            batch_size = (
                list(self.batch_size)[:start_dim]
                + [nelt]
                + list(self.batch_size[end_dim + 1 :])
            )
        else:
            batch_size = [nelt] + list(self.batch_size[end_dim + 1 :])
        # TODO: check that this works with nested tds of different batch size
        out = self._fast_apply(flatten, batch_size=batch_size)
        if self._has_names():
            names = [
                name
                for i, name in enumerate(self.names)
                if (i < start_dim or i > end_dim)
            ]
            names.insert(start_dim, None)
            out.names = names
        return out

    def unflatten(self, dim, unflattened_size):
        """Unflattens a tensordict dim expanding it to a desired shape.

        Args:
            dim (int): specifies the dimension of the input tensor to be
                unflattened.
            unflattened_size (shape): is the new shape of the unflattened
                dimension of the tensordict.

        Examples:
            >>> td = TensorDict({
            ...     "a": torch.arange(60).view(3, 4, 5),
            ...     "b": torch.arange(12).view(3, 4)},
            ...     batch_size=[3, 4])
            >>> td_flat = td.flatten(0, 1)
            >>> td_unflat = td_flat.unflatten(0, [3, 4])
            >>> assert (td == td_unflat).all()
        """
        if dim < 0:
            dim = self.ndim + dim
            if dim < 0:
                raise ValueError(
                    f"Incompatible dim {dim} for tensordict with shape {self.shape}."
                )

        def unflatten(tensor):
            return torch.unflatten(
                tensor,
                dim,
                unflattened_size,
            )

        if dim > 0:
            batch_size = (
                list(self.batch_size)[:dim]
                + list(unflattened_size)
                + list(self.batch_size[dim + 1 :])
            )
        else:
            batch_size = list(unflattened_size) + list(self.batch_size[1:])
        # TODO: check that this works with nested tds of different batch size
        out = self._fast_apply(unflatten, batch_size=batch_size)
        if self._has_names():
            names = copy(self.names)
            for _ in range(len(unflattened_size) - 1):
                names.insert(dim, None)
            out.names = names
        return out

    @abc.abstractmethod
    def rename_key_(self, old_key: str, new_key: str, safe: bool = False) -> T:
        """Renames a key with a new string and returns the same tensordict with the updated key name.

        Args:
            old_key (str or nested key): key to be renamed.
            new_key (str or nested key): new name of the entry.
            safe (bool, optional): if ``True``, an error is thrown when the new
                key is already present in the TensorDict.

        Returns:
            self

        """
        ...

    @abc.abstractmethod
    def del_(self, key: NestedKey) -> T:
        """Deletes a key of the tensordict.

        Args:
            key (NestedKey): key to be deleted

        Returns:
            self

        """
        ...

    # Distributed functionality
    def gather_and_stack(self, dst: int) -> T | None:
        """Gathers tensordicts from various workers and stacks them onto self in the destination worker.

        Args:
            dst (int): the rank of the destination worker where :func:`gather_and_stack` will be called.

        Example:
            >>> from torch import multiprocessing as mp
            >>> from torch.dict import TensorDict
            >>> import torch
            >>>
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     # Create a single tensordict to be sent to server
            ...     td = TensorDict(
            ...         {("a", "b"): torch.randn(2),
            ...          "c": torch.randn(2)}, [2]
            ...     )
            ...     td.gather_and_stack(0)
            ...
            >>> def server():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     # Creates the destination tensordict on server.
            ...     # The first dim must be equal to world_size-1
            ...     td = TensorDict(
            ...         {("a", "b"): torch.zeros(2),
            ...          "c": torch.zeros(2)}, [2]
            ...     ).expand(1, 2).contiguous()
            ...     td.gather_and_stack(0)
            ...     assert td["a", "b"] != 0
            ...     print("yuppie")
            ...
            >>> if __name__ == "__main__":
            ...     mp.set_start_method("spawn")
            ...
            ...     main_worker = mp.Process(target=server)
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...
            ...     main_worker.join()
            ...     secondary_worker.join()
        """
        output = (
            [None for _ in range(dist.get_world_size())]
            if dst == dist.get_rank()
            else None
        )
        dist.gather_object(self, output, dst=dst)
        if dst == dist.get_rank():
            # remove self from output
            output = [item for i, item in enumerate(output) if i != dst]
            self.update(torch.stack(output, 0), inplace=True)
            return self
        return None

    def send(self, dst: int, init_tag: int = 0, pseudo_rand: bool = False) -> None:
        """Sends the content of a tensordict to a distant worker.

        Args:
            dst (int): the rank of the destination worker where the content
                should be sent.
            init_tag (int): the initial tag to be used to mark the tensors.
                Note that this will be incremented by as much as the number of
                tensors contained in the TensorDict.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                Defaults to ``False``.

        Example:
            >>> from torch import multiprocessing as mp
            >>> from torch.dict import TensorDict
            >>> import torch
            >>>
            >>>
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.randn(2),
            ...             "c": torch.randn(2, 3),
            ...             "_": torch.ones(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.send(0)
            ...
            >>>
            >>> def server(queue):
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.zeros(2),
            ...             "c": torch.zeros(2, 3),
            ...             "_": torch.zeros(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.recv(1)
            ...     assert (td != 0).all()
            ...     queue.put("yuppie")
            ...
            >>>
            >>> if __name__=="__main__":
            ...     queue = mp.Queue(1)
            ...     main_worker = mp.Process(target=server, args=(queue,))
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...     out = queue.get(timeout=10)
            ...     assert out == "yuppie"
            ...     main_worker.join()
            ...     secondary_worker.join()

        """
        self._send(dst, _tag=init_tag - 1, pseudo_rand=pseudo_rand)

    def _send(self, dst: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(value.__class__):
                _tag = value._send(dst, _tag=_tag, pseudo_rand=pseudo_rand)
                continue
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            dist.send(value, dst=dst, tag=_tag)

        return _tag

    def recv(self, src: int, init_tag: int = 0, pseudo_rand: bool = False) -> int:
        """Receives the content of a tensordict and updates content with it.

        Check the example in the `send` method for context.

        Args:
            src (int): the rank of the source worker.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`send`.
                Defaults to ``False``.

        """
        return self._recv(src, _tag=init_tag - 1, pseudo_rand=pseudo_rand)

    def _recv(self, src: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if isinstance(value, Tensor):
                pass
            elif _is_tensor_collection(value.__class__):
                _tag = value._recv(src, _tag=_tag, pseudo_rand=pseudo_rand)
                continue
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            dist.recv(value, src=src, tag=_tag)
            self._set_str(key, value, inplace=True, validated=True)

        return _tag

    def isend(self, dst: int, init_tag: int = 0, pseudo_rand: bool = False) -> int:
        """Sends the content of the tensordict asynchronously.

        Args:
            dst (int): the rank of the destination worker where the content
                should be sent.
            init_tag (int): the initial tag to be used to mark the tensors.
                Note that this will be incremented by as much as the number of
                tensors contained in the TensorDict.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                Defaults to ``False``.

        Example:
            >>> import torch
            >>> from torch.dict import TensorDict
            >>> from torch import multiprocessing as mp
            >>> def client():
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=1,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.randn(2),
            ...             "c": torch.randn(2, 3),
            ...             "_": torch.ones(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     td.isend(0)
            ...
            >>>
            >>> def server(queue, return_premature=True):
            ...     torch.distributed.init_process_group(
            ...         "gloo",
            ...         rank=0,
            ...         world_size=2,
            ...         init_method=f"tcp://localhost:10003",
            ...     )
            ...     td = TensorDict(
            ...         {
            ...             ("a", "b"): torch.zeros(2),
            ...             "c": torch.zeros(2, 3),
            ...             "_": torch.zeros(2, 1, 5),
            ...         },
            ...         [2],
            ...     )
            ...     out = td.irecv(1, return_premature=return_premature)
            ...     if return_premature:
            ...         for fut in out:
            ...             fut.wait()
            ...     assert (td != 0).all()
            ...     queue.put("yuppie")
            ...
            >>>
            >>> if __name__ == "__main__":
            ...     queue = mp.Queue(1)
            ...     main_worker = mp.Process(
            ...         target=server,
            ...         args=(queue, )
            ...         )
            ...     secondary_worker = mp.Process(target=client)
            ...
            ...     main_worker.start()
            ...     secondary_worker.start()
            ...     out = queue.get(timeout=10)
            ...     assert out == "yuppie"
            ...     main_worker.join()
            ...     secondary_worker.join()

        """
        return self._isend(dst, init_tag - 1, pseudo_rand=pseudo_rand)

    def _isend(
        self,
        dst: int,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
    ) -> int:
        root = False
        if _futures is None:
            root = True
            _futures = []
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(value.__class__):
                _tag = value._isend(
                    dst, _tag=_tag, pseudo_rand=pseudo_rand, _futures=_futures
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            _future = dist.isend(value, dst=dst, tag=_tag)
            _futures.append(_future)
        if root:
            for _future in _futures:
                _future.wait()
        return _tag

    def irecv(
        self,
        src: int,
        return_premature: bool = False,
        init_tag: int = 0,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        """Receives the content of a tensordict and updates content with it asynchronously.

        Check the example in the :meth:`~.isend` method for context.

        Args:
            src (int): the rank of the source worker.
            return_premature (bool): if ``True``, returns a list of futures to wait
                upon until the tensordict is updated. Defaults to ``False``,
                i.e. waits until update is completed withing the call.
            init_tag (int): the ``init_tag`` used by the source worker.
            pseudo_rand (bool): if True, the sequence of tags will be pseudo-
                random, allowing to send multiple data from different nodes
                without overlap. Notice that the generation of these pseudo-random
                numbers is expensive (1e-5 sec/number), meaning that it could
                slow down the runtime of your algorithm.
                This value must match the one passed to :func:`isend`.
                Defaults to ``False``.

        Returns:
            if ``return_premature=True``, a list of futures to wait
                upon until the tensordict is updated.
        """
        return self._irecv(
            src, return_premature, _tag=init_tag - 1, pseudo_rand=pseudo_rand
        )

    def _irecv(
        self,
        src: int,
        return_premature: bool = False,
        _tag: int = -1,
        _future_list: list[torch.Future] = None,
        pseudo_rand: bool = False,
    ) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
        root = False
        if _future_list is None:
            _future_list = []
            root = True

        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(value.__class__):
                _tag, _future_list = value._irecv(
                    src,
                    _tag=_tag,
                    _future_list=_future_list,
                    pseudo_rand=pseudo_rand,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            if not pseudo_rand:
                _tag += 1
            else:
                _tag = int_generator(_tag + 1)
            _future_list.append(dist.irecv(value, src=src, tag=_tag))
        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    def reduce(self, dst, op=dist.ReduceOp.SUM, async_op=False, return_premature=False):
        """Reduces the tensordict across all machines.

        Only the process with ``rank`` dst is going to receive the final result.

        """
        return self._reduce(dst, op, async_op, return_premature)

    def _reduce(
        self,
        dst,
        op=dist.ReduceOp.SUM,
        async_op=False,
        return_premature=False,
        _future_list=None,
    ):
        root = False
        if _future_list is None:
            _future_list = []
            root = True
        for key in self.sorted_keys:
            value = self._get_str(key, NO_DEFAULT)
            if _is_tensor_collection(value.__class__):
                _future_list = value._reduce(
                    dst=dst,
                    op=op,
                    async_op=async_op,
                    _future_list=_future_list,
                )
                continue
            elif isinstance(value, Tensor):
                pass
            else:
                raise NotImplementedError(f"Type {type(value)} is not supported.")
            _future_list.append(dist.reduce(value, dst=dst, op=op, async_op=async_op))
        if not root:
            return _future_list
        elif async_op and return_premature:
            return _future_list
        elif async_op:
            for future in _future_list:
                future.wait()
            return

    # Apply and map functionality
    def apply_(self, fn: Callable, *others) -> T:
        """Applies a callable to all values stored in the tensordict and re-writes them in-place.

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            *others (sequence of TensorDictBase, optional): the other
                tensordicts to be used.

        Returns:
            self or a copy of self with the function applied

        """
        return self.apply(fn, *others, inplace=True)

    def apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        **constructor_kwargs,
    ) -> T:
        """Applies a callable to all values stored in the tensordict and sets them in a new tensordict.

        The apply method will return an :class:`~torch.dict.TensorDict` instance,
        regardless of the input type. To keep the same type, one can execute

          >>> out = td.clone(False).update(td.apply(...))

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            *others (TensorDictBase instances, optional): if provided, these
                tensordict instances should have a structure matching the one
                of self. The ``fn`` argument should receive as many
                unnamed inputs as the number of tensordicts, including self.
            batch_size (sequence of int, optional): if provided,
                the resulting TensorDict will have the desired batch_size.
                The :obj:`batch_size` argument should match the batch_size after
                the transformation. This is a keyword only argument.
            device (torch.device, optional): the resulting device, if any.
            names (list of str, optional): the new dimension names, in case the
                batch_size is modified.
            inplace (bool, optional): if True, changes are made in-place.
                Default is False. This is a keyword only argument.
            **constructor_kwargs: additional keyword arguments to be passed to the
                TensorDict constructor.

        Returns:
            a new tensordict with transformed_in tensors.

        Example:
            >>> td = TensorDict({
            ...     "a": -torch.ones(3),
            ...     "b": {"c": torch.ones(3)}},
            ...     batch_size=[3])
            >>> td_1 = td.apply(lambda x: x+1)
            >>> assert (td["a"] == 0).all()
            >>> assert (td["b", "c"] == 2).all()
            >>> td_2 = td.apply(lambda x, y: x+y, td)
            >>> assert (td_2["a"] == -2).all()
            >>> assert (td_2["b", "c"] == 2).all()

        """
        return self._apply_nest(
            fn,
            *others,
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=False,
            **constructor_kwargs,
        )

    @abc.abstractmethod
    def _apply_nest(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        checked: bool = False,
        call_on_nested: bool = False,
        **constructor_kwargs,
    ) -> T:
        ...

    def _fast_apply(
        self,
        fn: Callable,
        *others: T,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        call_on_nested: bool = False,
        **constructor_kwargs,
    ) -> T:
        """A faster apply method.

        This method does not run any check after performing the func. This
        means that one to make sure that the metadata of the resulting tensors
        (device, shape etc.) match the :meth:`~.apply` ones.

        """
        return self._apply_nest(
            fn,
            *others,
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=True,
            call_on_nested=call_on_nested,
            **constructor_kwargs,
        )

    def map(
        self,
        fn: Callable,
        dim: int = 0,
        num_workers: int = None,
        chunksize: int = None,
        num_chunks: int = None,
        pool: "mp.Pool" = None,
    ):
        """Maps a function to splits of the tensordict across one dimension.

        This method will apply a function to a tensordict instance by chunking
        it in tensordicts of equal size and dispatching the operations over the
        desired number of workers.

        The function signature should be ``Callabe[[TensorDict], Union[TensorDict, Tensor]]``.
        The output must support the :func:`torch.cat` operation. The function
        must be serializable.

        Args:
            fn (callable): function to apply to the tensordict.
                Signatures similar to ``Callabe[[TensorDict], Union[TensorDict, Tensor]]``
                are supported.
            dim (int, optional): the dim along which the tensordict will be chunked.
            num_workers (int, optional): the number of workers. Exclusive with ``pool``.
                If none is provided, the number of workers will be set to the
                number of cpus available.
            chunksize (int, optional): The size of each chunk of data. If none
                is provided, the number of chunks will equate the number
                of workers. For very large tensordicts, such large chunks
                may not fit in memory for the operation to be done and
                more chunks may be needed to make the operation practically
                doable. This argument is exclusive with num_chunks.
            num_chunks (int, optional): the number of chunks to split the tensordict
                into. If none is provided, the number of chunks will equate the number
                of workers. For very large tensordicts, such large chunks
                may not fit in memory for the operation to be done and
                more chunks may be needed to make the operation practically
                doable. This argument is exclusive with chunksize.
            pool (mp.Pool, optional): a multiprocess Pool instance to use
                to execute the job. If none is provided, a pool will be created
                within the ``map`` method.

        Examples:
            >>> import torch
            >>> from torch.dict import TensorDict
            >>>
            >>> def process_data(data):
            ...     data.set("y", data.get("x") + 1)
            ...     return data
            >>> if __name__ == "__main__":
            ...     data = TensorDict({"x": torch.zeros(1, 1_000_000)}, [1, 1_000_000]).memmap_()
            ...     data = data.map(process_data, dim=1)
            ...     print(data["y"][:, :10])
            ...
            tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        .. note:: This method is particularily useful when working with large
            datasets stored on disk (e.g. memory-mapped tensordicts) where
            chunks will be zero-copied slices of the original data which can
            be passed to the processes with virtually zero-cost. This allows
            to tread very large datasets (eg. over a Tb big) to be processed
            at little cost.

        """
        from torch import multiprocessing as mp

        if pool is None:
            if num_workers is None:
                num_workers = mp.cpu_count()  # Get the number of CPU cores
            with mp.Pool(num_workers) as pool:
                return self.map(
                    fn, dim=dim, chunksize=chunksize, num_chunks=num_chunks, pool=pool
                )
        num_workers = pool._processes
        dim_orig = dim
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Got incompatible dimension {dim_orig}")

        self_split = _split_tensordict(self, chunksize, num_chunks, num_workers, dim)
        chunksize = 1
        out = pool.imap(fn, self_split, chunksize)
        out = torch.cat(list(out), dim)
        return out

    # Functorch compatibility
    @abc.abstractmethod
    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        ...

    @abc.abstractmethod
    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        ...

    # Validation and checks
    def _convert_to_tensor(self, array: np.ndarray) -> Tensor:
        if isinstance(array, np.bool_):
            array = array.item()
        if isinstance(array, list):
            array = np.asarray(array)
        return torch.as_tensor(array, device=self.device)

    @abc.abstractmethod
    def _convert_to_tensordict(self, dict_value: dict[str, Any]) -> T:
        ...

    def _check_batch_size(self) -> None:
        batch_dims = self.batch_dims
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                value._check_batch_size()
            if _shape(value)[:batch_dims] != self.batch_size:
                raise RuntimeError(
                    f"batch_size are incongruent, got value with shape {_shape(value)}, "
                    f"-- expected {self.batch_size}"
                )

    @abc.abstractmethod
    def _check_is_shared(self) -> bool:
        ...

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        batch_dims = len(new_size)
        for key, tensor in self.items():
            if _shape(tensor)[:batch_dims] != new_size:
                raise RuntimeError(
                    f"the tensor {key} has shape {_shape(tensor)} which "
                    f"is incompatible with the batch-size {new_size}."
                )

    @abc.abstractmethod
    def _check_device(self) -> None:
        ...

    def _validate_key(self, key: NestedKey) -> NestedKey:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise KeyError(_GENERIC_NESTED_ERR.format(key))
        return key

    def _validate_value(
        self,
        value: CompatibleType | dict[str, CompatibleType],
        *,
        check_shape: bool = True,
    ) -> CompatibleType | dict[str, CompatibleType]:
        cls = type(value)
        is_tc = _is_tensor_collection(cls)
        if is_tc or issubclass(cls, tuple(_ACCEPTED_CLASSES)):
            pass
        elif issubclass(cls, dict):
            value = self._convert_to_tensordict(value)
            is_tc = True
        else:
            try:
                value = self._convert_to_tensor(value)
            except ValueError as err:
                raise ValueError(
                    f"TensorDict conversion only supports tensorclasses, tensordicts,"
                    f" numeric scalars and tensors. Got {type(value)}"
                ) from err
        batch_size = self.batch_size
        batch_dims = len(batch_size)
        if check_shape and batch_size and _shape(value)[:batch_dims] != batch_size:
            # if TensorDict, let's try to map it to the desired shape
            if is_tc:
                # we must clone the value before not to corrupt the data passed to set()
                value = value.clone(recurse=False)
                value.batch_size = self.batch_size
            else:
                raise RuntimeError(
                    f"batch dimension mismatch, got self.batch_size"
                    f"={self.batch_size} and value.shape={_shape(value)}."
                )
        device = self.device
        if device is not None and value.device != device:
            value = value.to(device, non_blocking=True)
        if is_tc and check_shape:
            has_names = self._has_names()
            # we do our best to match the dim names of the value and the
            # container.
            if has_names and value.names[:batch_dims] != self.names:
                # we clone not to corrupt the value
                value = value.clone(False).refine_names(*self.names)
            elif not has_names and value._has_names():
                self.names = value.names[: self.batch_dims]
        return value

    # Context manager functionality
    @property
    def _last_op_queue(self):
        # this is used to keep track of the last operation when using
        # the tensordict as a context manager.
        last_op_queue = self.__dict__.get("__last_op_queue", None)
        if last_op_queue is None:
            last_op_queue = collections.deque()
            self.__dict__["__last_op_queue"] = last_op_queue
        return last_op_queue

    def __enter__(self):
        self._last_op_queue.append(self._last_op)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, Exception):
            return False
        _last_op = self._last_op_queue.pop()
        if _last_op is not None:
            last_op, (args, kwargs) = _last_op
            if last_op == self.__class__.lock_.__name__:
                return self.unlock_()
            elif last_op == self.__class__.unlock_.__name__:
                return self.lock_()
            else:
                raise NotImplementedError(f"Unrecognised function {last_op}.")
        return self

    # Clone, select, exclude, empty
    @abc.abstractmethod
    def select(self, *keys: str, inplace: bool = False, strict: bool = True) -> T:
        """Selects the keys of the tensordict and returns an new tensordict with only the selected keys.

        The values are not copied: in-place modifications a tensor of either
        of the original or new tensordict will result in a change in both
        tensordicts.

        Args:
            *keys (str): keys to select
            inplace (bool): if True, the tensordict is pruned in place.
                Default is :obj:`False`.
            strict (bool, optional): whether selecting a key that is not present
                will return an error or not. Default: :obj:`True`.

        Returns:
            A new tensordict with the selected keys only.

        """
        ...

    def exclude(self, *keys: str, inplace: bool = False) -> T:
        target = self if inplace else self.clone(recurse=False)
        for key in keys:
            if key in self.keys(True):
                del target[key]
        return target

    def to_tensordict(self) -> "TensorDict":
        """Returns a regular TensorDict instance from the TensorDictBase.

        Returns:
            a new TensorDict object containing the same values.

        """
        from torch.dict import TensorDict

        return TensorDict(
            {
                key: value.clone()
                if not _is_tensor_collection(value.__class__)
                else value.to_tensordict()
                for key, value in self.items()
            },
            device=self.device,
            batch_size=self.batch_size,
            names=self.names if self._has_names() else None,
        )

    @abc.abstractmethod
    def clone(self, recurse: bool = True) -> T:
        """Clones a TensorDictBase subclass instance onto a new TensorDictBase subclass of the same type.

        To create a TensorDict instance from any other TensorDictBase subtype, call the :meth:`~.to_tensordict` method
        instead.

        Args:
            recurse (bool, optional): if ``True``, each tensor contained in the
                TensorDict will be copied too. Otherwise only the TensorDict
                tree structure will be copied. Defaults to ``True``.

        """
        ...

    def copy(self):
        """Return a shallow copy of the tensordict (ie, copies the structure but not the data).

        Equivalent to `TensorDictBase.clone(recurse=False)`
        """
        return self.clone(recurse=False)

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary with key-value pairs matching those of the tensordict."""
        return {
            key: value.to_dict() if _is_tensor_collection(type(value)) else value
            for key, value in self.items()
        }

    def empty(self, recurse=False) -> T:
        """Returns a new, empty tensordict with the same device and batch size.

        Args:
            recurse (bool, optional): if ``True``, the entire structure of the
                ``TensorDict`` will be reproduced without content.
                Otherwise, only the root will be duplicated.
                Defaults to ``False``.

        """
        if not recurse:
            return self.select()
        # simply exclude the leaves
        return self.exclude(*self.keys(True, True))

    # Filling
    def zero_(self) -> T:
        """Zeros all tensors in the tensordict in-place."""
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def fill_(self, key: NestedKey, value: float | bool) -> T:
        """Fills a tensor pointed by the key with a given scalar value.

        Args:
            key (str or nested key): entry to be filled.
            value (Number or bool): value to use for the filling.

        Returns:
            self

        """
        key = _unravel_key_to_tuple(key)
        data = self._get_tuple(key, NO_DEFAULT)
        if _is_tensor_collection(type(data)):
            data._fast_apply(lambda x: x.fill_(value), inplace=True)
        else:
            data = data.fill_(value)
            self._set_tuple(key, data, inplace=True, validated=True)
        return self

    # Masking
    @abc.abstractmethod
    def masked_fill_(self, mask: Tensor, value: float | bool) -> T:
        """Fills the values corresponding to the mask with the desired value.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match the tensordict batch-size.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td.masked_fill_(mask, 1.0)
            >>> td.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        ...

    @abc.abstractmethod
    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        """Out-of-place version of masked_fill.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match the tensordict batch-size.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td1 = td.masked_fill(mask, 1.0)
            >>> td1.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        ...

    def where(self, condition, other, *, out=None, pad=None):  # noqa: D417
        """Return a ``TensorDict`` of elements selected from either self or other, depending on condition.

        Args:
            condition (BoolTensor): When ``True`` (nonzero), yields ``self``,
                otherwise yields ``other``.
            other (TensorDictBase or Scalar): value (if ``other`` is a scalar)
                or values selected at indices where condition is ``False``.

        Keyword Args:
            out (TensorDictBase, optional): the output ``TensorDictBase`` instance.
            pad (scalar, optional): if provided, missing keys from the source
                or destination tensordict will be written as `torch.where(mask, self, pad)`
                or `torch.where(mask, pad, other)`. Defaults to ``None``, ie
                missing keys are not tolerated.

        """
        ...

    @abc.abstractmethod
    def masked_select(self, mask: Tensor) -> T:
        """Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values.

        Args:
            mask (torch.Tensor): boolean mask to be used for the tensors.
                Shape must match the TensorDict ``batch_size``.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...    batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td_mask = td.masked_select(mask)
            >>> td_mask.get("a")
            tensor([[0., 0., 0., 0.]])

        """
        ...

    @abc.abstractmethod
    def _change_batch_size(self, new_size: torch.Size) -> None:
        ...

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        """Returns a boolean indicating if all the tensors are contiguous."""
        ...

    @abc.abstractmethod
    def contiguous(self) -> T:
        """Returns a new tensordict of the same type with contiguous values (or self if values are already contiguous)."""
        ...

    @cache  # noqa: B019
    def flatten_keys(self, separator: str = ".", inplace: bool = False) -> T:
        """Converts a nested tensordict into a flat one, recursively.

        The TensorDict type will be lost and the result will be a simple TensorDict instance.

        Args:
            separator (str, optional): the separator between the nested items.
            inplace (bool, optional): if ``True``, the resulting tensordict will
                have the same identity as the one where the call has been made.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"a": 1, ("b", "c"): 2, ("e", "f", "g"): 3}, batch_size=[])
            >>> data.flatten_keys(separator=" - ")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b - c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    e - f - g: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        This method and :meth:`~.unflatten_keys` are particularily useful when
        handling state-dicts, as they make it possible to seamlessly convert
        flat dictionaries into data structures that mimic the structure of the
        model.

        Examples:
            >>> model = torch.nn.Sequential(torch.nn.Linear(3 ,4))
            >>> ddp_model = torch.ao.quantization.QuantWrapper(model)
            >>> state_dict = TensorDict(ddp_model.state_dict(), batch_size=[]).unflatten_keys(".")
            >>> print(state_dict)
            TensorDict(
                fields={
                    module: TensorDict(
                        fields={
                            0: TensorDict(
                                fields={
                                    bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                                    weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                                batch_size=torch.Size([]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model_state_dict = state_dict.get("module")
            >>> print(model_state_dict)
            TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                            weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model.load_state_dict(dict(model_state_dict.flatten_keys(".")))
        """
        all_leaves = list(self.keys(include_nested=True, leaves_only=True))
        all_leaves_flat = [
            separator.join(key) if isinstance(key, tuple) else key for key in all_leaves
        ]
        if len(set(all_leaves_flat)) < len(set(all_leaves)):
            # find duplicates
            seen = set()
            conflicts = []
            for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
                if leaf_flat in seen:
                    conflicts.append(leaf)
                else:
                    seen.add(leaf_flat)
            raise KeyError(
                f"Flattening keys in tensordict causes keys {conflicts} to collide."
            )
        if inplace:
            # we will need to remove the empty tensordicts later on
            root_keys = set(self.keys())
            for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
                self.rename_key_(leaf, leaf_flat)
                if isinstance(leaf, str):
                    root_keys.discard(leaf)
            self.exclude(*root_keys, inplace=True)
            return self
        else:
            result = self.empty()
            for leaf, leaf_flat in zip(all_leaves, all_leaves_flat):
                result._set_str(
                    leaf_flat, self.get(leaf), validated=True, inplace=False
                )
            return result

    @cache  # noqa: B019
    def unflatten_keys(self, separator: str = ".", inplace: bool = False) -> T:
        """Converts a flat tensordict into a nested one, recursively.

        The TensorDict type will be lost and the result will be a simple TensorDict instance.
        The metadata of the nested tensordicts will be inferred from the root:
        all instances across the data tree will share the same batch-size,
        dimension names and device.

        Args:
            separator (str, optional): the separator between the nested items.
            inplace (bool, optional): if ``True``, the resulting tensordict will
                have the same identity as the one where the call has been made.
                Defaults to ``False``.

        Examples:
            >>> data = TensorDict({"a": 1, "b - c": 2, "e - f - g": 3}, batch_size=[])
            >>> data.unflatten_keys(separator=" - ")
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False),
                    e: TensorDict(
                        fields={
                            f: TensorDict(
                                fields={
                                    g: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                                batch_size=torch.Size([]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        This method and :meth:`~.unflatten_keys` are particularily useful when
        handling state-dicts, as they make it possible to seamlessly convert
        flat dictionaries into data structures that mimic the structure of the
        model.

        Examples:
            >>> model = torch.nn.Sequential(torch.nn.Linear(3 ,4))
            >>> ddp_model = torch.ao.quantization.QuantWrapper(model)
            >>> state_dict = TensorDict(ddp_model.state_dict(), batch_size=[]).unflatten_keys(".")
            >>> print(state_dict)
            TensorDict(
                fields={
                    module: TensorDict(
                        fields={
                            0: TensorDict(
                                fields={
                                    bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                                    weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                                batch_size=torch.Size([]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model_state_dict = state_dict.get("module")
            >>> print(model_state_dict)
            TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                            weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> model.load_state_dict(dict(model_state_dict.flatten_keys(".")))

        """
        to_unflatten = defaultdict(list)
        for key in self.keys():
            if separator in key[1:-1]:
                split_key = key.split(separator)
                to_unflatten[split_key[0]].append((key, separator.join(split_key[1:])))

        if not inplace:
            out = self.empty()
            for key, value in self.items():
                if separator not in key[1:-1]:
                    out._set_str(key, value, validated=True, inplace=False)
        else:
            out = self

        keys = set(out.keys())
        for key, list_of_keys in to_unflatten.items():
            # if the key is present and either (1) it is not a tensor collection or (2) it is but it's not empty, then we raise an error.
            if key in keys and (
                not is_tensor_collection(out.get(key)) or not out.get(key).is_empty()
            ):
                raise KeyError(
                    "Unflattening key(s) in tensordict will override existing unflattened key"
                )

            tensordict = self.empty()
            if key in self.keys():
                tensordict.update(self[key])
            for old_key, new_key in list_of_keys:
                value = self.get(old_key)
                tensordict[new_key] = value
                if inplace:
                    del self[old_key]
            out._set_str(
                key,
                tensordict.unflatten_keys(separator=separator),
                inplace=False,
                validated=True,
            )
        return out

    @abc.abstractmethod
    def _index_tensordict(self, index: IndexType) -> T:
        ...

    # Locking functionality
    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    def _lock_propagate(self, lock_ids=None):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        is_root = lock_ids is None
        if is_root:
            lock_ids = set()
        self._lock_id = self._lock_id.union(lock_ids)
        lock_ids = lock_ids.union({id(self)})
        _locked_tensordicts = []
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                value._lock_propagate(lock_ids)
                _locked_tensordicts.append(value)
        if is_root:
            self._locked_tensordicts = _locked_tensordicts
        else:
            self._locked_tensordicts += _locked_tensordicts

    @as_decorator("is_locked")
    def lock_(self) -> T:
        if self.is_locked:
            return self
        self._lock_propagate()
        return self

    def _remove_lock(self, lock_id):
        self._lock_id = self._lock_id - {lock_id}
        if self._locked_tensordicts:
            for td in self._locked_tensordicts:
                td._remove_lock(lock_id)

    @erase_cache
    def _propagate_unlock(self, lock_ids=None):
        if lock_ids is not None:
            self._lock_id.difference_update(lock_ids)
        else:
            lock_ids = set()
        self._is_locked = False

        unlocked_tds = [self]
        lock_ids.add(id(self))
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                unlocked_tds.extend(value._propagate_unlock(lock_ids))
        self._locked_tensordicts = []

        self._is_shared = False
        self._is_memmap = False
        return unlocked_tds

    @as_decorator("is_locked")
    def unlock_(self) -> T:
        unlock_tds = self._propagate_unlock()
        for td in unlock_tds:
            if len(td._lock_id):
                self.lock_()
                raise RuntimeError(
                    "Cannot unlock a tensordict that is part of a locked graph. "
                    "Unlock the root tensordict first. If the tensordict is part of multiple graphs, "
                    "group the graphs under a common tensordict an unlock this root. "
                )
        return self

    # Conversion (device or dtype)
    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[torch.device, str]] = ...,
        non_blocking: bool = ...,
    ) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[torch.device, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, *, other: T, non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, *, batch_size: torch.Size) -> T:
        ...

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> T:
        """Maps a TensorDictBase subclass either on another device, dtype or to another TensorDictBase subclass (if permitted).

        Casting tensors to a new dtype is not allowed, as tensordicts are not bound to contain a single
        tensor dtype.

        Args:
            device (torch.device, optional): the desired device of the tensordict.
            dtype (torch.dtype, optional): the desired floating point or complex dtype of
                the tensordict.
            tensor (torch.Tensor, optional): Tensor whose dtype and device are the desired
                dtype and device for all tensors in this TensorDict.

        Keyword Args:
            non_blocking (bool, optional): whether the operations should be blocking.
            memory_format (torch.memory_format, optional): the desired memory
                format for 4D parameters and buffers in this tensordict.
            batch_size (torch.Size, optional): resulting batch-size of the
                output tensordict.
            other (TensorDictBase, optional): TensorDict instance whose dtype
                and device are the desired dtype and device for all tensors
                in this TensorDict.
                .. note:: Since :class:`~tensordict.TensorDictBase` instances do not have
                    a dtype, the dtype is gathered from the example leaves.
                    If there are more than one dtype, then no dtype
                    casting is undertook.

        Returns:
            a new tensordict instance if the device differs from the tensordict
            device and/or if the dtype is passed. The same tensordict otherwise.
            ``batch_size`` only modifications are done in-place.

        Examples:
            >>> data = TensorDict({"a": 1.0}, [], device=None)
            >>> data_cuda = data.to("cuda:0")  # casts to cuda
            >>> data_int = data.to(torch.int)  # casts to int
            >>> data_cuda_int = data.to("cuda:0", torch.int)  # multiple casting
            >>> data_cuda = data.to(torch.randn(3, device="cuda:0"))  # using an example tensor
            >>> data_cuda = data.to(other=TensorDict({}, [], device="cuda:0"))  # using a tensordict example
        """
        ...

    def is_floating_point(self):
        for item in self.values(include_nested=True, leaves_only=True):
            if not item.is_floating_point():
                return False
        else:
            return True

    def double(self):
        r"""Casts all tensors to ``torch.bool``."""
        return self._fast_apply(lambda x: x.double())

    def float(self):
        r"""Casts all tensors to ``torch.float``."""
        return self._fast_apply(lambda x: x.float())

    def int(self):
        r"""Casts all tensors to ``torch.int``."""
        return self._fast_apply(lambda x: x.int())

    def bool(self):
        r"""Casts all tensors to ``torch.bool``."""
        return self._fast_apply(lambda x: x.bool())

    def half(self):
        r"""Casts all tensors to ``torch.half``."""
        return self._fast_apply(lambda x: x.half())

    def bfloat16(self):
        r"""Casts all tensors to ``torch.bfloat16``."""
        return self._fast_apply(lambda x: x.bfloat16())

    def type(self, dst_type):
        r"""Casts all tensors to :attr:`dst_type`.

        Args:
            dst_type (type or string): the desired type

        """
        return self._fast_apply(lambda x: x.type(dst_type))

    # Gradient compatibility
    @property
    def requires_grad(self) -> bool:
        return any(v.requires_grad for v in self.values())

    @abc.abstractmethod
    def detach_(self) -> T:
        """Detach the tensors in the tensordict in-place.

        Returns:
            self.

        """
        ...

    def detach(self) -> T:
        """Detach the tensors in the tensordict.

        Returns:
            a new tensordict with no tensor requiring gradient.

        """
        return self._fast_apply(lambda x: x.detach())


_ACCEPTED_CLASSES = [
    Tensor,
    TensorDictBase,
]
_ACCEPTED_CLASSES = set(_ACCEPTED_CLASSES)


def _register_tensor_class(cls):
    global _ACCEPTED_CLASSES
    _ACCEPTED_CLASSES.add(cls)


def _is_tensor_collection(datatype):
    out = _TENSOR_COLLECTION_MEMO.get(datatype, None)
    if out is None:
        if issubclass(datatype, TensorDictBase):
            out = True
        elif _is_tensorclass(datatype):
            out = True
        else:
            out = False
        _TENSOR_COLLECTION_MEMO[datatype] = out
    return out


def is_tensor_collection(datatype: type | Any) -> bool:
    """Checks if a data object or a type is a tensor container from the tensordict lib.

    Returns:
        ``True`` if the input is a TensorDictBase subclass, a tensorclass or an istance of these.
        ``False`` otherwise.

    Examples:
        >>> is_tensor_collection(TensorDictBase)  # True
        >>> is_tensor_collection(TensorDict({}, []))  # True
        >>> @tensorclass
        ... class MyClass:
        ...     pass
        ...
        >>> is_tensor_collection(MyClass)  # True
        >>> is_tensor_collection(MyClass(batch_size=[]))  # True

    """
    # memoizing is 2x faster
    if not isinstance(datatype, type):
        datatype = type(datatype)
    return _is_tensor_collection(datatype)
