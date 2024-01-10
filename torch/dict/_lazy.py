from __future__ import annotations

import json
import numbers
import os
import re
import weakref
from collections import defaultdict
from copy import copy
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, Iterator, Sequence, Type

import numpy as np
from functorch import dim as ftdim

import torch
from torch import Tensor
from torch.dict._memmap import MemoryMappedTensor as MemmapTensor
from torch.dict.base import (
    _ACCEPTED_CLASSES,
    _is_tensor_collection,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    T,
    TensorDictBase,
)
from torch.dict.tensordict import _SubTensorDict, _TensorDictKeysView, TensorDict
from torch.utils._pytree import tree_map
from .utils import (
    _broadcast_tensors,
    _check_keys,
    _getitem_batch_size,
    _is_number,
    _parse_to,
    _prune_selected_keys,
    _renamed_inplace_method,
    _shape,
    _td_fields,
    _unravel_key_to_tuple,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    erase_cache,
    expand_right,
    IndexType,
    is_tensorclass,
    lock_blocked,
    NestedKey,
    unravel_key_list,
)

_has_functorch = False
try:
    try:
        from torch._C._functorch import (
            _add_batch_dim,
            _remove_batch_dim,
            is_batchedtensor,
        )
    except ImportError:
        from functorch._C import is_batchedtensor

    _has_functorch = True
except ImportError:
    _has_functorch = False

    def is_batchedtensor(tensor: Tensor) -> bool:
        """Placeholder for the functorch function."""
        return False


class _LazyStackedTensorDictKeysView(_TensorDictKeysView):
    tensordict: LazyStackedTensorDict

    def __len__(self) -> int:
        return len(self._keys())

    def _keys(self) -> list[str]:
        return self.tensordict._key_list()

    def __contains__(self, item):
        item = _unravel_key_to_tuple(item)
        if item[0] in self.tensordict._iterate_over_keys():
            if self.leaves_only:
                return not _is_tensor_collection(self.tensordict.entry_class(item[0]))
            has_first_key = True
        else:
            has_first_key = False
        if not has_first_key or len(item) == 1:
            return has_first_key
        # otherwise take the long way
        return all(
            item[1:]
            in tensordict.get(item[0]).keys(self.include_nested, self.leaves_only)
            for tensordict in self.tensordict.tensordicts
        )


class LazyStackedTensorDict(TensorDictBase):
    """A Lazy stack of TensorDicts.

    When stacking TensorDicts together, the default behaviour is to put them
    in a stack that is not instantiated.
    This allows to seamlessly work with stacks of tensordicts with operations
    that will affect the original tensordicts.

    Args:
         *tensordicts (TensorDict instances): a list of tensordict with
            same batch size.
         stack_dim (int): a dimension (between `-td.ndimension()` and
            `td.ndimension()-1` along which the stack should be performed.
         hook_out (callable, optional): a callable to execute after :meth:`~.get`.
         hook_in (callable, optional): a callable to execute before :meth:`~.set`.

    Examples:
        >>> from torch.dict import TensorDict
        >>> import torch
        >>> tds = [TensorDict({'a': torch.randn(3, 4)}, batch_size=[3])
        ...     for _ in range(10)]
        >>> td_stack = torch.stack(tds, -1)
        >>> print(td_stack.shape)
        torch.Size([3, 10])
        >>> print(td_stack.get("a").shape)
        torch.Size([3, 10, 4])
        >>> print(td_stack[:, 0] is tds[0])
        True

    """

    _is_vmapped: bool = False

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        from ._torch_func import LAZY_TD_HANDLED_FUNCTIONS

        if func in LAZY_TD_HANDLED_FUNCTIONS:
            if kwargs is None:
                kwargs = {}
            if func not in LAZY_TD_HANDLED_FUNCTIONS or not all(
                issubclass(t, (Tensor, TensorDictBase)) for t in types
            ):
                return NotImplemented
            return LAZY_TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            return super().__torch_function__(func, types, args, kwargs)

    _td_dim_name = None
    _safe = False
    _lazy = True

    def __init__(
        self,
        *tensordicts: T,
        stack_dim: int = 0,
        hook_out: callable | None = None,
        hook_in: callable | None = None,
        batch_size: Sequence[int] | None = None,  # TODO: remove
    ) -> None:
        self._is_shared = False
        self._is_memmap = False
        self._is_locked = None

        # sanity check
        N = len(tensordicts)
        if not N:
            raise RuntimeError(
                "at least one tensordict must be provided to "
                "StackedTensorDict to be instantiated"
            )
        if stack_dim < 0:
            raise RuntimeError(
                f"stack_dim must be non negative, got stack_dim={stack_dim}"
            )
        _batch_size = tensordicts[0].batch_size
        device = tensordicts[0].device

        for td in tensordicts[1:]:
            if not is_tensor_collection(td):
                raise TypeError(
                    "Expected all inputs to be TensorDictBase instances but got "
                    f"{type(td)} instead."
                )
            _bs = td.batch_size
            _device = td.device
            if device != _device:
                raise RuntimeError(f"devices differ, got {device} and {_device}")
            if _bs != _batch_size:
                raise RuntimeError(
                    f"batch sizes in tensordicts differs, StackedTensorDict "
                    f"cannot be created. Got td[0].batch_size={_batch_size} "
                    f"and td[i].batch_size={_bs} "
                )
        self.tensordicts: list[TensorDictBase] = list(tensordicts)
        self.stack_dim = stack_dim
        self._batch_size = self._compute_batch_size(_batch_size, stack_dim, N)
        self.hook_out = hook_out
        self.hook_in = hook_in
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @property
    def device(self) -> torch.device | None:
        # devices might have changed, so we check that they're all the same
        device_set = {td.device for td in self.tensordicts}
        if len(device_set) != 1:
            return None
        device = self.tensordicts[0].device
        return device

    @device.setter
    def device(self, value: DeviceType) -> None:
        for t in self.tensordicts:
            t.device = value

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        return self._batch_size_setter(new_size)

    @property
    @cache  # noqa
    def names(self):
        names = list(self.tensordicts[0].names)
        for td in self.tensordicts[1:]:
            if names != td.names:
                raise ValueError(
                    f"Not all dim names match, got {names} and {td.names}."
                )
        names.insert(self.stack_dim, self._td_dim_name)
        return names

    @names.setter
    @erase_cache  # a nested lazy stacked tensordict is not apparent to the root
    def names(self, value):
        if value is None:
            for td in self.tensordicts:
                td.names = None
            self._td_dim_name = None
        else:
            names_c = list(value)
            name = names_c[self.stack_dim]
            self._td_dim_name = name
            del names_c[self.stack_dim]
            for td in self.tensordicts:
                if td._check_dim_name(name):
                    # TODO: should reset names here
                    raise ValueError(f"The dimension name {name} is already taken.")
                td.rename_(*names_c)

    def _rename_subtds(self, names):
        # remove the name of the stack dim
        names = list(names)
        del names[self.stack_dim]
        for td in self.tensordicts:
            td.names = names

    def _has_names(self):
        return all(td._has_names() for td in self.tensordicts)

    def _erase_names(self):
        self._td_dim_name = None
        for td in self.tensordicts:
            td._erase_names()

    def get_item_shape(self, key):
        """Gets the shape of an item in the lazy stack.

        Heterogeneous dimensions are returned as -1.

        This implementation is inefficient as it will attempt to stack the items
        to compute their shape, and should only be used for printing.
        """
        try:
            item = self.get(key)
            return item.shape
        except RuntimeError as err:
            if re.match(r"Found more than one unique shape in the tensors", str(err)):
                shape = None
                for td in self.tensordicts:
                    if shape is None:
                        shape = list(td.get_item_shape(key))
                    else:
                        _shape = td.get_item_shape(key)
                        if len(shape) != len(_shape):
                            shape = [-1]
                            return torch.Size(shape)
                        shape = [
                            s1 if s1 == s2 else -1 for (s1, s2) in zip(shape, _shape)
                        ]
                shape.insert(self.stack_dim, len(self.tensordicts))
                return torch.Size(shape)
            else:
                raise err

    def is_shared(self) -> bool:
        are_shared = [td.is_shared() for td in self.tensordicts]
        are_shared = [value for value in are_shared if value is not None]
        if not len(are_shared):
            return None
        if any(are_shared) and not all(are_shared):
            raise RuntimeError(
                f"tensordicts shared status mismatch, got {sum(are_shared)} "
                f"shared tensordicts and "
                f"{len(are_shared) - sum(are_shared)} non shared tensordict "
            )
        return all(are_shared)

    def is_memmap(self) -> bool:
        are_memmap = [td.is_memmap() for td in self.tensordicts]
        if any(are_memmap) and not all(are_memmap):
            raise RuntimeError(
                f"tensordicts memmap status mismatch, got {sum(are_memmap)} "
                f"memmap tensordicts and "
                f"{len(are_memmap) - sum(are_memmap)} non memmap tensordict "
            )
        return all(are_memmap)

    @staticmethod
    def _compute_batch_size(
        batch_size: torch.Size, stack_dim: int, N: int
    ) -> torch.Size:
        s = list(batch_size)
        s.insert(stack_dim, N)
        return torch.Size(s)

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> T:
        try:
            inplace = self._convert_inplace(inplace, key)
        except KeyError as e:
            raise KeyError(
                "setting a value in-place on a stack of TensorDict is only "
                "permitted if all members of the stack have this key in "
                "their register."
            ) from e
        if not validated:
            value = self._validate_value(value)
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        values = value.unbind(self.stack_dim)
        for tensordict, item in zip(self.tensordicts, values):
            tensordict._set_str(key, item, inplace=inplace, validated=validated)
        return self

    def _set_tuple(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> T:
        if len(key) == 1:
            return self._set_str(key[0], value, inplace=inplace, validated=validated)
        # if inplace is not False:  # inplace could be None
        #     # we don't want to end up in the situation where one tensordict has
        #     # inplace=True and another one inplace=False because inplace was loose.
        #     # Worse could be writing with inplace=True up until some level then to
        #     # realize the key is missing in one td, raising an exception and having
        #     # messed up the data. Hence we must start by checking if the key
        #     # is present.
        #     has_key = key in self.keys(True)
        #     if inplace is True and not has_key:  # inplace could be None
        #         raise KeyError(
        #             TensorDictBase.KEY_ERROR.format(
        #                 key, self.__class__.__name__, sorted(self.keys())
        #             )
        #         )
        #     inplace = has_key
        if not validated:
            value = self._validate_value(value)
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        values = value.unbind(self.stack_dim)
        for tensordict, item in zip(self.tensordicts, values):
            tensordict._set_tuple(key, item, inplace=inplace, validated=validated)
        return self

    def _split_index(self, index):
        """Given a tuple index, split it in as many indices as the number of tensordicts.

        Returns:
            a dictionary with {index-of-td: index-within-td}
            the number of single dim indices until stack dim
            a boolean indicating if the index along the stack dim is an integer
        """
        if not isinstance(index, tuple):
            index = (index,)
        index = convert_ellipsis_to_idx(index, self.batch_size)
        index = _broadcast_tensors(index)
        out = []
        num_single = 0
        num_none = 0
        isinteger = False
        is_nd_tensor = False
        cursor = 0  # the dimension cursor
        selected_td_idx = range(len(self.tensordicts))
        has_bool = False
        num_squash = 0
        for i, idx in enumerate(index):  # noqa: B007
            cursor_incr = 1
            if idx is None:
                out.append(None)
                num_none += cursor <= self.stack_dim
                continue
            if cursor == self.stack_dim:
                # we need to check which tds need to be indexed
                if isinstance(idx, ftdim.Dim):
                    raise ValueError(
                        "Cannot index a lazy stacked tensordict along the stack dimension with "
                        "a first-class dimension index. Consider consolidating the tensordict first "
                        "using `tensordict.contiguous()`."
                    )
                elif isinstance(idx, slice) or _is_number(idx):
                    selected_td_idx = range(len(self.tensordicts))[idx]
                    if not isinstance(selected_td_idx, range):
                        isinteger = True
                        selected_td_idx = [selected_td_idx]
                elif isinstance(idx, (list, range)):
                    selected_td_idx = idx
                elif isinstance(idx, (torch.Tensor, np.ndarray)):
                    if idx.dtype in (np.dtype("bool"), torch.bool):
                        # we mark that we need to dispatch the indices across stack idx
                        has_bool = True
                        # split mask along dim
                        individual_masks = idx = idx.unbind(0)
                        selected_td_idx = range(len(self.tensordicts))
                        out.append(idx)
                        split_dim = self.stack_dim - num_single
                        mask_loc = i
                    else:
                        if isinstance(idx, np.ndarray):
                            idx = torch.tensor(idx)
                        is_nd_tensor = True
                        selected_td_idx = range(len(idx))
                        out.append(idx.unbind(0))
                else:
                    raise TypeError(f"Invalid index type: {type(idx)}.")
            else:
                if _is_number(idx) and cursor < self.stack_dim:
                    num_single += 1
                if isinstance(
                    idx,
                    (
                        int,
                        ftdim.Dim,
                        slice,
                        list,
                        range,
                    ),
                ):
                    out.append(idx)
                elif isinstance(idx, (np.ndarray, torch.Tensor)):
                    if idx.dtype in (np.dtype("bool"), torch.bool):
                        cursor_incr = idx.ndim
                        if cursor < self.stack_dim:
                            num_squash += cursor_incr - 1
                        if (
                            cursor < self.stack_dim
                            and cursor + cursor_incr > self.stack_dim
                        ):
                            # we mark that we need to dispatch the indices across stack idx
                            has_bool = True
                            # split mask along dim
                            # relative_stack_dim = self.stack_dim - cursor - cursor_incr
                            individual_masks = idx = idx.unbind(0)
                            selected_td_idx = range(self.shape[i])
                            split_dim = cursor - num_single
                            mask_loc = i
                    out.append(idx)
                else:
                    raise TypeError(f"Invalid index type: {type(idx)}.")
            cursor += cursor_incr
        if has_bool:
            out = tuple(
                tuple(idx if not isinstance(idx, tuple) else idx[i] for idx in out)
                for i in selected_td_idx
            )
            return {
                "index_dict": {i: out[i] for i in selected_td_idx},
                "num_single": num_single,
                "isinteger": isinteger,
                "has_bool": has_bool,
                "individual_masks": individual_masks,
                "split_dim": split_dim,
                "mask_loc": mask_loc,
                "is_nd_tensor": is_nd_tensor,
                "num_none": num_none,
                "num_squash": num_squash,
            }
        elif is_nd_tensor:

            def isindexable(idx):
                if isinstance(idx, (torch.Tensor, np.ndarray)):
                    if idx.dtype in (torch.bool, np.dtype("bool")):
                        return False
                    return True
                if isinstance(idx, (tuple, list, range)):
                    return True
                return False

            out = tuple(
                tuple(idx if not isindexable(idx) else idx[i] for idx in out)
                for i in selected_td_idx
            )
            return {
                "index_dict": dict(enumerate(out)),
                "num_single": num_single,
                "isinteger": isinteger,
                "has_bool": has_bool,
                "is_nd_tensor": is_nd_tensor,
                "num_none": num_none,
                "num_squash": num_squash,
            }
        return {
            "index_dict": {i: tuple(out) for i in selected_td_idx},
            "num_single": num_single,
            "isinteger": isinteger,
            "has_bool": has_bool,
            "is_nd_tensor": is_nd_tensor,
            "num_none": num_none,
            "num_squash": num_squash,
        }

    def _set_at_str(self, key, value, index, *, validated):
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        split_index = self._split_index(index)
        converted_idx = split_index["index_dict"]
        num_single = split_index["num_single"]
        isinteger = split_index["isinteger"]
        has_bool = split_index["has_bool"]
        num_squash = split_index.get("num_squash", 0)
        num_none = split_index.get("num_none", 0)
        is_nd_tensor = split_index.get("is_nd_tensor", False)
        if isinteger:
            # this will break if the index along the stack dim is [0] or :1 or smth
            for i, _idx in converted_idx.items():
                self.tensordicts[i]._set_at_str(key, value, _idx, validated=validated)
            return self
        if is_nd_tensor:
            unbind_dim = self.stack_dim - num_single + num_none - num_squash
            value_unbind = value.unbind(unbind_dim)
            for idx, _value in zip(converted_idx.values(), value_unbind):
                self._set_at_str(key, _value, idx, validated=validated)
            return self
        elif not has_bool:
            unbind_dim = self.stack_dim - num_single + num_none - num_squash
            value_unbind = value.unbind(unbind_dim)
            for (i, _idx), _value in zip(
                converted_idx.items(),
                value_unbind,
            ):
                self.tensordicts[i]._set_at_str(key, _value, _idx, validated=validated)
        else:
            # we must split, not unbind
            mask_unbind = split_index["individual_masks"]
            split_dim = split_index["split_dim"]
            splits = [_mask_unbind.sum().item() for _mask_unbind in mask_unbind]
            value_unbind = value.split(splits, split_dim)
            if mask_unbind[0].ndim == 0:
                # we can return a stack
                for (i, _idx), mask, _value in zip(
                    converted_idx.items(),
                    mask_unbind,
                    value_unbind,
                ):
                    if mask.any():
                        self.tensordicts[i]._set_at_str(
                            key, _value, _idx, validated=validated
                        )
            else:
                for (i, _idx), _value in zip(converted_idx.items(), value_unbind):
                    self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                    self[self_idx]._set_at_str(key, _value, _idx, validated=validated)

    def _set_at_tuple(self, key, value, idx, *, validated):
        if len(key) == 1:
            return self._set_at_str(key[0], value, idx, validated=validated)
        # get the "last" tds
        tds = []
        for td in self.tensordicts:
            tds.append(td.get(key[:-1]))
        # build only a single lazy stack from it
        # (if the stack is a stack of stacks this won't be awesomely efficient
        # but then we'd need to splut the value (which we can do) and recompute
        # the sub-index for each td, which is a different story!
        td = LazyStackedTensorDict(
            *tds, stack_dim=self.stack_dim, hook_out=self.hook_out, hook_in=self.hook_in
        )
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        if self._is_vmapped:
            value = self.hook_in(value)
        item = td._get_str(key, NO_DEFAULT)
        item[idx] = value
        td._set_str(key, item, inplace=True, validated=True)
        return self

    def unsqueeze(self, dim: int | None = None) -> T:
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        if dim <= self.stack_dim:
            stack_dim = self.stack_dim + 1
        else:
            dim = dim - 1
            stack_dim = self.stack_dim
        return LazyStackedTensorDict(
            *(tensordict.unsqueeze(dim) for tensordict in self.tensordicts),
            stack_dim=stack_dim,
        )

    def squeeze(self, dim: int | None = None) -> T:
        """Squeezes all tensors for a dimension comprised in between `-td.batch_dims+1` and `td.batch_dims-1` and returns them in a new tensordict.

        Args:
            dim (Optional[int]): dimension along which to squeeze. If dim is None, all singleton dimensions will be squeezed. dim is None by default.

        """
        if dim is None:
            size = self.size()
            if len(self.size()) == 1 or size.count(1) == 0:
                return self
            first_singleton_dim = size.index(1)
            return self.squeeze(first_singleton_dim).squeeze()

        if dim < 0:
            dim = self.batch_dims + dim

        if self.batch_dims and (dim >= self.batch_dims or dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and "
                f"td.batch_dims only. Got dim={dim} and batch_size"
                f"={self.batch_size}."
            )

        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self
        if dim == self.stack_dim:
            return self.tensordicts[0]
        elif dim < self.stack_dim:
            stack_dim = self.stack_dim - 1
        else:
            dim = dim - 1
            stack_dim = self.stack_dim
        return LazyStackedTensorDict(
            *(tensordict.squeeze(dim) for tensordict in self.tensordicts),
            stack_dim=stack_dim,
        )

    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        if dim < 0:
            dim = self.batch_dims + dim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Cannot unbind along dimension {dim} with batch size {self.batch_size}."
            )
        if dim == self.stack_dim:
            return tuple(self.tensordicts)
        else:
            # return a stack of unbound tensordicts
            out = []
            new_dim = dim if dim < self.stack_dim else dim - 1
            new_stack_dim = (
                self.stack_dim if dim > self.stack_dim else self.stack_dim - 1
            )
            for td in self.tensordicts:
                out.append(td.unbind(new_dim))
            from ._torch_func import _stack

            return tuple(_stack(vals, new_stack_dim) for vals in zip(*out))

    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> T:
        if dim == self.stack_dim:
            for source, tensordict_dest in zip(list_item, self.tensordicts):
                tensordict_dest.update_(source)
        else:
            for i, td in enumerate(list_item):
                idx = (slice(None),) * dim + (i,)
                self.update_at_(td, idx)
        return self

    @cache  # noqa: B019
    def _get_str(
        self,
        key: NestedKey,
        default: str | CompatibleType = NO_DEFAULT,
    ) -> CompatibleType:
        # we can handle the case where the key is a tuple of length 1
        tensors = []
        for td in self.tensordicts:
            tensors.append(td._get_str(key, default=default))
            if (
                tensors[-1] is default
                and not isinstance(default, (MemmapTensor, torch.Tensor))
                and not is_tensor_collection(default)
            ):
                # then we consider this default as non-stackable and return prematurly
                return default
        try:
            out = self.lazy_stack(tensors, self.stack_dim)
            if _is_tensor_collection(out.__class__):
                if isinstance(out, LazyStackedTensorDict):
                    # then it's a LazyStackedTD
                    out.hook_out = self.hook_out
                    out.hook_in = self.hook_in
                    out._is_vmapped = self._is_vmapped
                    incr = 0 if not self._is_vmapped else 1
                    out._batch_size = (
                        self._batch_size
                        + out.batch_size[(len(self._batch_size) + incr) :]
                    )
                    if self._td_dim_name is not None:
                        out._td_dim_name = self._td_dim_name
                elif is_tensorclass(out):
                    # then it's a tensorclass
                    out._tensordict.hook_out = self.hook_out
                    out._tensordict.hook_in = self.hook_in
                    out._tensordict._is_vmapped = self._is_vmapped
                    incr = 0 if not self._is_vmapped else 1
                    out._tensordict._batch_size = (
                        self._batch_size
                        + out._tensordict.batch_size[(len(self._batch_size) + incr) :]
                    )
                    if self._td_dim_name is not None:
                        out._tensordict._td_dim_name = self._td_dim_name
                else:
                    raise RuntimeError
                    # # it must be a TensorDict
                    # incr = 0 if not self._is_vmapped else 1
                    # out._batch_size = (
                    #     self._batch_size
                    #     + out.batch_size[(len(self._batch_size) + incr) :]
                    # )
                    # if self._has_names():
                    #     out.names = self.names + [None] * (out.ndim - self.ndim)
            elif self.hook_out is not None:
                out = self.hook_out(out)
            return out
        except RuntimeError as err:
            if "stack expects each tensor to be equal size" in str(err):
                shapes = {_shape(tensor) for tensor in tensors}
                raise RuntimeError(
                    f"Found more than one unique shape in the tensors to be "
                    f"stacked ({shapes}). This is likely due to a modification "
                    f"of one of the stacked TensorDicts, where a key has been "
                    f"updated/created with an uncompatible shape. If the entries "
                    f"are intended to have a different shape, use the get_nestedtensor "
                    f"method instead."
                )
            else:
                raise err

    def _get_tuple(self, key, default):
        first = self._get_str(key[0], None)
        if first is None:
            return self._default_get(key[0], default)
        if len(key) == 1:
            return first
        try:
            return first._get_tuple(key[1:], default=default)
        except AttributeError as err:
            if "has no attribute" in str(err):
                raise ValueError(
                    f"Expected a TensorDictBase instance but got {type(first)} instead"
                    f" for key '{key[1:]}' in tensordict:\n{self}."
                )

    @classmethod
    def lazy_stack(
        cls,
        items: Sequence[TensorDictBase],
        dim: int = 0,
        device: DeviceType | None = None,
        out: T | None = None,
    ) -> T:
        """Stacks tensordicts in a LazyStackedTensorDict."""
        if not items:
            raise RuntimeError("items cannot be empty")

        from .tensorclass import NonTensorData

        if all(isinstance(item, torch.Tensor) for item in items):
            return torch.stack(items, dim=dim, out=out)
        if all(is_tensorclass(item) and type(item) == type(items[0]) for item in items):
            if all(isinstance(tensordict, NonTensorData) for tensordict in items):
                return NonTensorData._stack_non_tensor(items, dim=dim)
            lazy_stack = cls.lazy_stack(
                [item._tensordict for item in items], dim=dim, out=out
            )
            non_tensordict = [item._non_tensordict for item in items]
            return type(items[0])._from_tensordict(
                tensorict=lazy_stack, non_tensordict=non_tensordict
            )

        batch_size = items[0].batch_size
        if dim < 0:
            dim = len(batch_size) + dim + 1

        for td in items[1:]:
            if td.batch_size != items[0].batch_size:
                raise RuntimeError(
                    "stacking tensordicts requires them to have congruent batch sizes, "
                    f"got td1.batch_size={td.batch_size} and td2.batch_size="
                    f"{items[0].batch_size}"
                )

        if out is None:
            # We need to handle tensordicts with exclusive keys and tensordicts with
            # mismatching shapes.
            # The first case is handled within _check_keys which fails if keys
            # don't match exactly.
            # The second requires a check over the tensor shapes.
            return LazyStackedTensorDict(*items, stack_dim=dim)
        else:
            batch_size = list(batch_size)
            batch_size.insert(dim, len(items))
            batch_size = torch.Size(batch_size)

            if out.batch_size != batch_size:
                raise RuntimeError(
                    "out.batch_size and stacked batch size must match, "
                    f"got out.batch_size={out.batch_size} and batch_size"
                    f"={batch_size}"
                )

            try:
                out._stack_onto_(items, dim)
            except KeyError as err:
                raise err
        return out

    @classmethod
    def maybe_dense_stack(
        cls,
        items: Sequence[TensorDictBase],
        dim: int = 0,
        out: T | None = None,
        strict: bool = False,
        contiguous: bool = False,
    ) -> T:
        """Stacks tensors or tensordicts densly if possible, or onto a LazyStackedTensorDict otherwise.

        Examples:
            >>> td0 = TensorDict({"a": 0}, [])
            >>> td1 = TensorDict({"b": 0}, [])
            >>> LazyStackedTensorDict.maybe_dense_stack([td0, td0])  # returns a TensorDict with shape [2]
            >>> LazyStackedTensorDict.maybe_dense_stack([td0, td1])  # returns a LazyStackedTensorDict with shape [2]
            >>> LazyStackedTensorDict.maybe_dense_stack(list(torch.randn(2)))  # returns a torch.Tensor with shape [2]
        """
        if not items:
            raise RuntimeError("items cannot be empty")

        from .tensorclass import NonTensorData

        if all(isinstance(item, torch.Tensor) for item in items):
            return torch.stack(items, dim=dim, out=out)

        if all(isinstance(tensordict, NonTensorData) for tensordict in items):
            return NonTensorData._stack_non_tensor(items, dim=dim)

        batch_size = items[0].batch_size
        if dim < 0:
            dim = len(batch_size) + dim + 1

        for td in items[1:]:
            if td.batch_size != items[0].batch_size:
                raise RuntimeError(
                    "stacking tensordicts requires them to have congruent batch sizes, "
                    f"got td1.batch_size={td.batch_size} and td2.batch_size="
                    f"{items[0].batch_size}"
                )

        if out is None:
            # We need to handle tensordicts with exclusive keys and tensordicts with
            # mismatching shapes.
            # The first case is handled within _check_keys which fails if keys
            # don't match exactly.
            # The second requires a check over the tensor shapes.
            device = items[0].device
            if any(device != item.device for item in items[1:]):
                device = None
            try:
                keys = _check_keys(items, strict=True)
            except KeyError:
                if not contiguous:
                    return LazyStackedTensorDict(*items, stack_dim=dim)
                raise

            out = {}
            for key in keys:
                out[key] = []
                tensor_shape = None
                for _tensordict in items:
                    tensor = _tensordict._get_str(key, default=NO_DEFAULT)
                    if tensor_shape is None:
                        tensor_shape = tensor.shape
                    elif tensor.shape != tensor_shape:
                        return LazyStackedTensorDict(*items, stack_dim=dim, out=out)
                    out[key].append(tensor)

            def stack_fn(key_values):
                key, values = key_values
                return cls.maybe_dense_stack(values, dim)

            out = {key: stack_fn((key, value)) for key, value in out.items()}

            is_locked = any(item.is_locked for item in items)
            result = TensorDict(
                out,
                batch_size=LazyStackedTensorDict._compute_batch_size(
                    batch_size, dim, len(items)
                ),
                device=device,
                _run_checks=False,
            )
            if is_locked:
                return result.lock_()
            return result
        else:
            keys = _check_keys(items)
            batch_size = list(batch_size)
            batch_size.insert(dim, len(items))
            batch_size = torch.Size(batch_size)

            if out.batch_size != batch_size:
                raise RuntimeError(
                    "out.batch_size and stacked batch size must match, "
                    f"got out.batch_size={out.batch_size} and batch_size"
                    f"={batch_size}"
                )

            out_keys = set(out.keys())
            if strict:
                in_keys = set(keys)
                if len(out_keys - in_keys) > 0:
                    raise RuntimeError(
                        "The output tensordict has keys that are missing in the "
                        "tensordict that has to be written: {out_keys - in_keys}. "
                        "As per the call to `stack(..., strict=True)`, this "
                        "is not permitted."
                    )
                elif len(in_keys - out_keys) > 0:
                    raise RuntimeError(
                        "The resulting tensordict has keys that are missing in "
                        f"its destination: {in_keys - out_keys}. As per the call "
                        "to `stack(..., strict=True)`, this is not permitted."
                    )

            try:
                out._stack_onto_(items, dim)
            except KeyError as err:
                raise err
        return out

    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        if self.is_memmap():
            td = LazyStackedTensorDict.maybe_dense_stack(
                [td.cpu().as_tensor() for td in self.tensordicts], 0
            )
        else:
            td = self
        if in_dim < 0:
            in_dim = self.ndim + in_dim
        if in_dim == self.stack_dim:
            return self._cached_add_batch_dims(td, in_dim=in_dim, vmap_level=vmap_level)
        if in_dim < td.stack_dim:
            # then we'll stack along a dim before
            stack_dim = td.stack_dim - 1
        else:
            in_dim = in_dim - 1
            stack_dim = td.stack_dim
        tds = [
            td._fast_apply(
                lambda _arg: _add_batch_dim(_arg, in_dim, vmap_level),
                batch_size=[b for i, b in enumerate(td.batch_size) if i != in_dim],
                names=[name for i, name in enumerate(td.names) if i != in_dim],
            )
            for td in td.tensordicts
        ]
        return LazyStackedTensorDict(*tds, stack_dim=stack_dim)

    @classmethod
    def _cached_add_batch_dims(cls, td, in_dim, vmap_level):
        # we return a stack with hook_out, and hack the batch_size and names
        # Per se it is still a LazyStack but the stacking dim is "hidden" from
        # the outside
        out = td.copy()

        def hook_out(tensor, in_dim=in_dim, vmap_level=vmap_level):
            return _add_batch_dim(tensor, in_dim, vmap_level)

        n = len(td.tensordicts)

        def hook_in(
            tensor,
            out_dim=in_dim,
            batch_size=n,
            vmap_level=vmap_level,
        ):
            if _is_tensor_collection(type(tensor)):
                return tensor._remove_batch_dim(vmap_level, batch_size, out_dim)
            return _remove_batch_dim(tensor, vmap_level, batch_size, out_dim)

        out.hook_out = hook_out
        out.hook_in = hook_in
        out._is_vmapped = True
        out._batch_size = torch.Size(
            [dim for i, dim in enumerate(out._batch_size) if i != out.stack_dim]
        )
        return out

    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        if self.hook_out is not None:
            # this is the hacked version. We just need to remove the hook_out and
            # reset a proper batch size
            return LazyStackedTensorDict(
                *self.tensordicts,
                stack_dim=out_dim,
            )
            # return self._cache_remove_batch_dim(vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim)
        else:
            # we must call _remove_batch_dim on all tensordicts
            # batch_size: size of the batch when we unhide it.
            # out_dim: dimension where the output will be found
            new_batch_size = list(self.batch_size)
            new_batch_size.insert(out_dim, batch_size)
            new_names = list(self.names)
            new_names.insert(out_dim, None)
            # rebuild the lazy stack
            # the stack dim is the same if the out_dim is past it, but it
            # must be incremented by one otherwise.
            # In the first case, the out_dim must be decremented by one
            if out_dim > self.stack_dim:
                stack_dim = self.stack_dim
                out_dim = out_dim - 1
            else:
                stack_dim = self.stack_dim + 1
            out = LazyStackedTensorDict(
                *[
                    td._remove_batch_dim(
                        vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                    )
                    for td in self.tensordicts
                ],
                stack_dim=stack_dim,
            )
        return out

    def get_nestedtensor(
        self,
        key: NestedKey,
        default: str | CompatibleType = NO_DEFAULT,
    ) -> CompatibleType:
        """Returns a nested tensor when stacking cannot be achieved.

        Args:
            key (NestedKey): the entry to nest.
            default (Any, optiona): the default value to return in case the key
                isn't in all sub-tensordicts.

                .. note:: In case the default is a tensor, this method will attempt
                  the construction of a nestedtensor with it. Otherwise, the default
                  value will be returned.

        Examples:
            >>> td0 = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4)}, [])
            >>> td1 = TensorDict({"a": torch.ones(5)}, [])
            >>> td = torch.stack([td0, td1], 0)
            >>> a = td.get_nestedtensor("a")
            >>> # using a tensor as default uses this default to build the nested tensor
            >>> b = td.get_nestedtensor("b", default=torch.ones(4))
            >>> assert (a == b).all()
            >>> # using anything else as default returns the default
            >>> b2 = td.get_nestedtensor("b", None)
            >>> assert b2 is None

        """
        # disallow getting nested tensor if the stacking dimension is not 0
        if self.stack_dim != 0:
            raise RuntimeError(
                "Because nested tensors can only be stacked along their first "
                "dimension, LazyStackedTensorDict.get_nestedtensor can only be called "
                "when the stack_dim is 0."
            )

        # we can handle the case where the key is a tuple of length 1
        key = _unravel_key_to_tuple(key)
        subkey = key[0]
        if len(key) > 1:
            tensordict = self.get(subkey, default)
            if tensordict is default:
                return default
            return tensordict.get_nestedtensor(key[1:], default=default)
        tensors = [td.get(subkey, default=default) for td in self.tensordicts]
        if not isinstance(default, torch.Tensor) and any(
            tensor is default for tensor in tensors
        ):
            # we don't stack but return the default
            return default
        return torch.nested.nested_tensor(tensors)

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> T:
        source = {key: value.contiguous() for key, value in self.items()}
        batch_size = self.batch_size
        device = self.device
        out = TensorDict(
            source=source,
            batch_size=batch_size,
            device=device,
            names=self.names,
            _run_checks=False,
        )
        return out

    def empty(self, recurse=False) -> T:
        return LazyStackedTensorDict(
            *[td.empty(recurse=recurse) for td in self.tensordicts],
            stack_dim=self.stack_dim,
        )

    def clone(self, recurse: bool = True) -> T:
        if recurse:
            # This could be optimized using copy but we must be careful with
            # metadata (_is_shared etc)
            out = LazyStackedTensorDict(
                *[td.clone() for td in self.tensordicts],
                stack_dim=self.stack_dim,
            )
        else:
            out = LazyStackedTensorDict(
                *[td.clone(recurse=False) for td in self.tensordicts],
                stack_dim=self.stack_dim,
            )
        if self._td_dim_name is not None:
            out._td_dim_name = self._td_dim_name
        return out

    def pin_memory(self) -> T:
        for td in self.tensordicts:
            td.pin_memory()
        return self

    def to(self, *args, **kwargs) -> T:
        device, dtype, non_blocking, convert_to_format, batch_size = _parse_to(
            *args, **kwargs
        )
        if batch_size is not None:
            raise TypeError("Cannot pass batch-size to a LazyStackedTensorDict.")
        result = self

        if device is not None and dtype is None and device == self.device:
            return result

        return LazyStackedTensorDict(
            *[td.to(*args, **kwargs) for td in self.tensordicts],
            stack_dim=self.stack_dim,
            hook_out=self.hook_out,
            hook_in=self.hook_in,
        )

    def _check_new_batch_size(self, new_size: torch.Size) -> None:
        if len(new_size) <= self.stack_dim:
            raise RuntimeError(
                "Changing the batch_size of a LazyStackedTensorDicts can only "
                "be done with sizes that are at least as long as the "
                "stacking dimension."
            )
        super()._check_new_batch_size(new_size)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _LazyStackedTensorDictKeysView:
        keys = _LazyStackedTensorDictKeysView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )
        return keys

    valid_keys = keys

    # def _iterate_over_keys(self) -> None:
    #     for key in self.tensordicts[0].keys():
    #         if all(key in td.keys() for td in self.tensordicts):
    #             yield key
    def _iterate_over_keys(self) -> None:
        # this is about 20x faster than the version above
        yield from self._key_list()

    @cache  # noqa: B019
    def _key_list(self):
        keys = set(self.tensordicts[0].keys())
        for td in self.tensordicts[1:]:
            keys = keys.intersection(td.keys())
        return sorted(keys, key=str)

    def entry_class(self, key: NestedKey) -> type:
        data_type = type(self.tensordicts[0].get(key))
        if _is_tensor_collection(data_type):
            return LazyStackedTensorDict
        return data_type

    def apply_(self, fn: Callable, *others, **kwargs):
        for i, td in enumerate(self.tensordicts):
            idx = (slice(None),) * self.stack_dim + (i,)
            td._fast_apply(
                fn, *[other[idx] for other in others], inplace=True, **kwargs
            )
        return self

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
        default: Any = NO_DEFAULT,
        named: bool = False,
        **constructor_kwargs,
    ) -> T:
        if inplace:
            if any(arg for arg in (batch_size, device, names, constructor_kwargs)):
                raise ValueError(
                    "Cannot pass other arguments to LazyStackedTensorDict.apply when inplace=True."
                )
            return self.apply_(fn, *others, named=named, default=default)
        else:
            if batch_size is not None:
                # any op that modifies the batch-size will result in a regular TensorDict
                return TensorDict._apply_nest(
                    self,
                    fn,
                    *others,
                    batch_size=batch_size,
                    device=device,
                    names=names,
                    checked=checked,
                    call_on_nested=call_on_nested,
                    default=default,
                    named=named,
                    **constructor_kwargs,
                )
            others = (other.unbind(self.stack_dim) for other in others)
            out = LazyStackedTensorDict(
                *(
                    td._apply_nest(
                        fn,
                        *oth,
                        checked=checked,
                        device=device,
                        call_on_nested=call_on_nested,
                        default=default,
                        named=named,
                    )
                    for td, *oth in zip(self.tensordicts, *others)
                ),
                stack_dim=self.stack_dim,
            )
            if names is not None:
                out.names = names
            return out

    def select(
        self, *keys: str, inplace: bool = False, strict: bool = False
    ) -> LazyStackedTensorDict:
        # the following implementation keeps the hidden keys in the tensordicts
        tensordicts = [
            td.select(*keys, inplace=inplace, strict=strict) for td in self.tensordicts
        ]
        if inplace:
            return self
        return LazyStackedTensorDict.maybe_dense_stack(tensordicts, dim=self.stack_dim)

    def exclude(self, *keys: str, inplace: bool = False) -> LazyStackedTensorDict:
        tensordicts = [
            tensordict.exclude(*keys, inplace=inplace)
            for tensordict in self.tensordicts
        ]
        if inplace:
            self.tensordicts = tensordicts
            return self
        return LazyStackedTensorDict.maybe_dense_stack(tensordicts, dim=self.stack_dim)

    def __setitem__(self, index: IndexType, value: T) -> T:
        if isinstance(index, (tuple, str)):
            # try:
            index_unravel = _unravel_key_to_tuple(index)
            if index_unravel:
                self._set_tuple(
                    index_unravel,
                    value,
                    inplace=BEST_ATTEMPT_INPLACE
                    if isinstance(self, _SubTensorDict)
                    else False,
                    validated=False,
                )
                return

            if any(isinstance(sub_index, (list, range)) for sub_index in index):
                index = tuple(
                    torch.tensor(sub_index, device=self.device)
                    if isinstance(sub_index, (list, range))
                    else sub_index
                    for sub_index in index
                )

        if index is Ellipsis or (isinstance(index, tuple) and Ellipsis in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        elif isinstance(index, (list, range)):
            index = torch.tensor(index, device=self.device)

        if isinstance(value, (TensorDictBase, dict)):
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            if isinstance(value, dict):
                value = TensorDict(
                    value, batch_size=indexed_bs, device=self.device, _run_checks=False
                )
            if value.batch_size != indexed_bs:
                # try to expand
                try:
                    value = value.expand(indexed_bs)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"indexed destination TensorDict batch size is {indexed_bs} "
                        f"(batch_size = {self.batch_size}, index={index}), "
                        f"which differs from the source batch size {value.batch_size}"
                    ) from err
            split_index = self._split_index(index)
            converted_idx = split_index["index_dict"]
            num_single = split_index["num_single"]
            isinteger = split_index["isinteger"]
            has_bool = split_index["has_bool"]
            num_squash = split_index.get("num_squash", 0)
            num_none = split_index.get("num_none", 0)
            is_nd_tensor = split_index.get("is_nd_tensor", False)
            if isinteger:
                # this will break if the index along the stack dim is [0] or :1 or smth
                for i, _idx in converted_idx.items():
                    self.tensordicts[i][_idx] = value
                return self
            if is_nd_tensor:
                raise RuntimeError(
                    "Indexing along stack dim with a non-boolean tensor is not supported yet. "
                    "Use SubTensorDict instead."
                )
            if not has_bool:
                unbind_dim = self.stack_dim - num_single + num_none - num_squash
                value_unbind = value.unbind(unbind_dim)
                for (i, _idx), _value in zip(
                    converted_idx.items(),
                    value_unbind,
                ):
                    self.tensordicts[i][_idx] = _value
            else:
                # we must split, not unbind
                mask_unbind = split_index["individual_masks"]
                split_dim = split_index["split_dim"]
                splits = [_mask_unbind.sum().item() for _mask_unbind in mask_unbind]
                value_unbind = value.split(splits, split_dim)
                if mask_unbind[0].ndim == 0:
                    # we can return a stack
                    for (i, _idx), mask, _value in zip(
                        converted_idx.items(),
                        mask_unbind,
                        value_unbind,
                    ):
                        if mask.any():
                            self.tensordicts[i][_idx] = _value
                else:
                    for (i, _idx), _value in zip(converted_idx.items(), value_unbind):
                        self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                        self[self_idx][_idx] = _value
        else:
            for key in self.keys():
                self.set_at_(key, value, index)

    def __contains__(self, item: IndexType) -> bool:
        if isinstance(item, TensorDictBase):
            return any(item is td for td in self.tensordicts)
        return super().__contains__(item)

    def __getitem__(self, index: IndexType) -> T:
        if isinstance(index, (tuple, str)):
            index_key = _unravel_key_to_tuple(index)
            if index_key:
                return self._get_tuple(index_key, NO_DEFAULT)
        split_index = self._split_index(index)
        converted_idx = split_index["index_dict"]
        isinteger = split_index["isinteger"]
        has_bool = split_index["has_bool"]
        is_nd_tensor = split_index["is_nd_tensor"]
        num_single = split_index.get("num_single", 0)
        num_none = split_index.get("num_none", 0)
        num_squash = split_index.get("num_squash", 0)
        if has_bool:
            mask_unbind = split_index["individual_masks"]
            cat_dim = split_index["mask_loc"] - num_single
            out = []
            if mask_unbind[0].ndim == 0:
                # we can return a stack
                for (i, _idx), mask in zip(converted_idx.items(), mask_unbind):
                    if mask.any():
                        if mask.all() and self.tensordicts[i].ndim == 0:
                            out.append(self.tensordicts[i])
                        else:
                            out.append(self.tensordicts[i][_idx])
                            out[-1] = out[-1].squeeze(cat_dim)
                return LazyStackedTensorDict.lazy_stack(out, cat_dim)
            else:
                for i, _idx in converted_idx.items():
                    self_idx = (slice(None),) * split_index["mask_loc"] + (i,)
                    out.append(self[self_idx][_idx])
                return torch.cat(out, cat_dim)
        elif is_nd_tensor:
            new_stack_dim = self.stack_dim - num_single + num_none
            return LazyStackedTensorDict.lazy_stack(
                [self[idx] for idx in converted_idx.values()], new_stack_dim
            )
        else:
            if isinteger:
                for (
                    i,
                    _idx,
                ) in (
                    converted_idx.items()
                ):  # for convenience but there's only one element
                    out = self.tensordicts[i]
                    if _idx is not None and _idx != ():
                        out = out[_idx]
                    return out
            else:
                out = []
                new_stack_dim = self.stack_dim - num_single + num_none - num_squash
                for i, _idx in converted_idx.items():
                    out.append(self.tensordicts[i][_idx])
                out = LazyStackedTensorDict.lazy_stack(out, new_stack_dim)
                out._td_dim_name = self._td_dim_name
                return out

    def __eq__(self, other):
        if is_tensorclass(other):
            return other == self
        if isinstance(other, (dict,)):
            other = TensorDict.from_dict(other)
        if _is_tensor_collection(other.__class__):
            out = []
            for i, td in enumerate(self.tensordicts):
                idx = (slice(None),) * self.stack_dim + (i,)
                out.append(other[idx] == td)
            return LazyStackedTensorDict.maybe_dense_stack(out, self.stack_dim)
        if isinstance(other, (numbers.Number, Tensor)):
            return LazyStackedTensorDict.maybe_dense_stack(
                [td == other for td in self.tensordicts],
                self.stack_dim,
            )
        return False

    def __ne__(self, other):
        if is_tensorclass(other):
            return other != self
        if isinstance(other, (dict,)):
            other = TensorDict.from_dict(other)
        if _is_tensor_collection(other.__class__):
            out = []
            for i, td in enumerate(self.tensordicts):
                idx = (slice(None),) * self.stack_dim + (i,)
                out.append(other[idx] != td)
            return LazyStackedTensorDict.maybe_dense_stack(out, self.stack_dim)
        if isinstance(other, (numbers.Number, Tensor)):
            return LazyStackedTensorDict.maybe_dense_stack(
                [td != other for td in self.tensordicts],
                self.stack_dim,
            )
        return True

    def all(self, dim: int = None) -> bool | TensorDictBase:
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            # TODO: we need to adapt this to LazyStackedTensorDict too
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
            )
        return all(value.all() for value in self.tensordicts)

    def any(self, dim: int = None) -> bool | TensorDictBase:
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            # TODO: we need to adapt this to LazyStackedTensorDict too
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
            )
        return any(value.any() for value in self.tensordicts)

    def _send(self, dst: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for td in self.tensordicts:
            _tag = td._send(dst, _tag=_tag, pseudo_rand=pseudo_rand)
        return _tag

    def _isend(
        self,
        dst: int,
        _tag: int = -1,
        _futures: list[torch.Future] | None = None,
        pseudo_rand: bool = False,
    ) -> int:
        if _futures is None:
            is_root = True
            _futures = []
        else:
            is_root = False
        for td in self.tensordicts:
            _tag = td._isend(dst, _tag=_tag, pseudo_rand=pseudo_rand, _futures=_futures)
        if is_root:
            for future in _futures:
                future.wait()
        return _tag

    def _recv(self, src: int, _tag: int = -1, pseudo_rand: bool = False) -> int:
        for td in self.tensordicts:
            _tag = td._recv(src, _tag=_tag, pseudo_rand=pseudo_rand)
        return _tag

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
        for td in self.tensordicts:
            _tag, _future_list = td._irecv(
                src=src,
                return_premature=return_premature,
                _tag=_tag,
                _future_list=_future_list,
                pseudo_rand=pseudo_rand,
            )

        if not root:
            return _tag, _future_list
        elif return_premature:
            return _future_list
        else:
            for future in _future_list:
                future.wait()
            return

    @lock_blocked
    def del_(self, key: NestedKey, **kwargs: Any) -> T:
        ids = set()
        cur_len = len(ids)
        is_deleted = False
        error = None
        for td in self.tensordicts:
            # checking that the td has not been processed yet.
            # It could be that not all sub-tensordicts have the appropriate
            # entry but one must have it (or an error is thrown).
            tdid = id(td)
            ids.add(tdid)
            new_cur_len = len(ids)
            if new_cur_len == cur_len:
                continue
            cur_len = new_cur_len
            try:
                td.del_(key, **kwargs)
                is_deleted = True
            except KeyError as err:
                error = err
                continue
        if not is_deleted:
            # we know err is defined because LazyStackedTensorDict cannot be empty
            raise error
        return self

    def pop(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        # using try/except for get/del is suboptimal, but
        # this is faster that checkink if key in self keys
        key = _unravel_key_to_tuple(key)
        if len(key) == 1:
            key = key[0]
        present = False
        if isinstance(key, tuple):
            if key in self.keys(True):
                present = True
                value = self._get_tuple(key, NO_DEFAULT)
        elif key in self.keys():
            present = True
            value = self._get_str(key, NO_DEFAULT)
        if present:
            self.del_(key)
        elif default is not NO_DEFAULT:
            value = default
        else:
            raise KeyError(
                f"You are trying to pop key `{key}` which is not in dict "
                f"without providing default value."
            )
        return value

    def share_memory_(self) -> T:
        for td in self.tensordicts:
            td.share_memory_()
        self._is_shared = True
        self.lock_()
        return self

    def detach_(self) -> T:
        for td in self.tensordicts:
            td.detach_()
        return self

    def _memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        executor=None,
        futures=None,
        inplace=True,
        like=False,
    ) -> T:
        if prefix is not None:
            prefix = Path(prefix)

            def save_metadata(prefix=prefix, self=self):
                prefix = Path(prefix)
                if not prefix.exists():
                    os.makedirs(prefix, exist_ok=True)
                with open(prefix / "meta.json", "w") as f:
                    json.dump(
                        {"_type": str(self.__class__), "stack_dim": self.stack_dim}, f
                    )

            if executor is None:
                save_metadata()
            else:
                futures.append(executor.submit(save_metadata))

        results = []
        for i, td in enumerate(self.tensordicts):
            results.append(
                td._memmap_(
                    prefix=(prefix / str(i)) if prefix is not None else None,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=inplace,
                    like=like,
                )
            )
        if not inplace:
            results = LazyStackedTensorDict.maybe_dense_stack(
                results, dim=self.stack_dim
            )
        else:
            results = self
        results._is_memmap = True
        results._is_shared = False
        results._device = torch.device("cpu")
        return results

    @classmethod
    def _load_memmap(cls, prefix: str, metadata: dict) -> LazyStackedTensorDict:
        tensordicts = []
        i = 0
        while (prefix / str(i)).exists():
            tensordicts.append(TensorDict.load_memmap(prefix / str(i)))
            i += 1

        return cls(*tensordicts, stack_dim=metadata["stack_dim"])

    def expand(self, *args: int, inplace: bool = False) -> T:
        if len(args) == 1 and isinstance(args[0], Sequence):
            shape = tuple(args[0])
        else:
            shape = args
        stack_dim = len(shape) + self.stack_dim - self.ndimension()
        new_shape_tensordicts = [v for i, v in enumerate(shape) if i != stack_dim]
        tensordicts = [td.expand(new_shape_tensordicts) for td in self.tensordicts]
        if inplace:
            self.tensordicts = tensordicts
            self.stack_dim = stack_dim
            return self
        return LazyStackedTensorDict.maybe_dense_stack(tensordicts, dim=stack_dim)

    def update(
        self,
        input_dict_or_td: T,
        clone: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs: Any,
    ) -> T:
        if input_dict_or_td is self:
            # no op
            return self
        if keys_to_update is not None:
            keys_to_update = unravel_key_list(keys_to_update)
            if len(keys_to_update) == 0:
                return self

        if (
            isinstance(input_dict_or_td, LazyStackedTensorDict)
            and input_dict_or_td.stack_dim == self.stack_dim
        ):
            if len(input_dict_or_td.tensordicts) != len(self.tensordicts):
                raise ValueError(
                    "cannot update stacked tensordicts with different shapes."
                )
            for td_dest, td_source in zip(
                self.tensordicts, input_dict_or_td.tensordicts
            ):
                td_dest.update(
                    td_source, clone=clone, keys_to_update=keys_to_update, **kwargs
                )
            return self

        inplace = kwargs.get("inplace", False)
        for key, value in input_dict_or_td.items():
            if clone and hasattr(value, "clone"):
                value = value.clone()
            elif clone:
                value = tree_map(torch.clone, value)
            key = _unravel_key_to_tuple(key)
            firstkey, subkey = key[0], key[1:]
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue

            if subkey:
                # we must check that the target is not a leaf
                target = self._get_str(firstkey, default=None)
                if is_tensor_collection(target):
                    sub_keys_to_update = _prune_selected_keys(keys_to_update, firstkey)
                    target.update(
                        {subkey: value},
                        inplace=inplace,
                        clone=clone,
                        keys_to_update=sub_keys_to_update,
                    )
                elif target is None:
                    self._set_tuple(key, value, inplace=inplace, validated=False)
                else:
                    raise TypeError(
                        f"Type mismatch: self.get(key[0]) is {type(target)} but expected a tensor collection."
                    )
            else:
                target = self._get_str(firstkey, default=None)
                if is_tensor_collection(target) and (
                    is_tensor_collection(value) or isinstance(value, dict)
                ):
                    sub_keys_to_update = _prune_selected_keys(keys_to_update, firstkey)
                    target.update(
                        value,
                        inplace=inplace,
                        clone=clone,
                        keys_to_update=sub_keys_to_update,
                    )
                elif target is None or not is_tensor_collection(value):
                    self._set_str(firstkey, value, inplace=inplace, validated=False)
                else:
                    raise TypeError(
                        f"Type mismatch: self.get(key) is {type(target)} but value is of type {type(value)}."
                    )

        return self

    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        **kwargs: Any,
    ) -> T:
        if input_dict_or_td is self:
            # no op
            return self
        if isinstance(input_dict_or_td, LazyStackedTensorDict):
            if input_dict_or_td.stack_dim == self.stack_dim:
                if not input_dict_or_td.shape[self.stack_dim] == len(self.tensordicts):
                    raise ValueError(
                        "cannot update stacked tensordicts with different shapes."
                    )
                for td_dest, td_source in zip(
                    self.tensordicts, input_dict_or_td.tensordicts
                ):
                    td_dest.update_(td_source)
                return self
            else:
                for i, td in enumerate(input_dict_or_td.tensordicts):
                    idx = (slice(None),) * input_dict_or_td.stack_dim + (i,)
                    self.update_at_(td, idx)
        for key, value in input_dict_or_td.items():
            if not isinstance(value, tuple(_ACCEPTED_CLASSES)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_(key, value, **kwargs)
        return self

    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        index: IndexType,
        clone: bool = False,
    ) -> T:
        if isinstance(input_dict_or_td, TensorDictBase):
            split_index = self._split_index(index)
            converted_idx = split_index["index_dict"]
            num_single = split_index["num_single"]
            isinteger = split_index["isinteger"]
            if isinteger:
                # this will break if the index along the stack dim is [0] or :1 or smth
                for i, _idx in converted_idx.items():
                    self.tensordicts[i].update_at_(
                        input_dict_or_td,
                        _idx,
                    )
                return self
            unbind_dim = self.stack_dim - num_single
            for (i, _idx), _value in zip(
                converted_idx.items(),
                input_dict_or_td.unbind(unbind_dim),
            ):
                self.tensordicts[i].update_at_(
                    _value,
                    _idx,
                )
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _ACCEPTED_CLASSES):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(key, value, index)
        return self

    def rename_key_(self, old_key: str, new_key: str, safe: bool = False) -> T:
        def sort_keys(element):
            if isinstance(element, tuple):
                return "_-|-_".join(element)
            return element

        for td in self.tensordicts:
            td.rename_key_(old_key, new_key, safe=safe)
        return self

    rename_key = _renamed_inplace_method(rename_key_)

    def where(self, condition, other, *, out=None, pad=None):
        if condition.ndim < self.ndim:
            condition = expand_right(condition, self.batch_size)
        condition = condition.unbind(self.stack_dim)
        if _is_tensor_collection(other.__class__) or (
            isinstance(other, Tensor)
            and other.shape[: self.stack_dim] == self.shape[: self.stack_dim]
        ):
            other = other.unbind(self.stack_dim)
            result = LazyStackedTensorDict.maybe_dense_stack(
                [
                    td.where(cond, _other, pad=pad)
                    for td, cond, _other in zip(self.tensordicts, condition, other)
                ],
                self.stack_dim,
            )
        else:
            result = LazyStackedTensorDict.maybe_dense_stack(
                [
                    td.where(cond, other, pad=pad)
                    for td, cond in zip(self.tensordicts, condition)
                ],
                self.stack_dim,
            )
        # We should not pass out to stack because this will overwrite the tensors in-place, but
        # we don't want that
        if out is not None:
            out.update(result)
            return out
        return result

    def masked_fill_(self, mask: Tensor, value: float | bool) -> T:
        mask_unbind = mask.unbind(dim=self.stack_dim)
        for _mask, td in zip(mask_unbind, self.tensordicts):
            td.masked_fill_(_mask, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    @lock_blocked
    def insert(self, index: int, tensordict: T) -> None:
        """Insert a TensorDict into the stack at the specified index.

        Analogous to list.insert. The inserted TensorDict must have compatible
        batch_size and device. Insertion is in-place, nothing is returned.

        Args:
            index (int): The index at which the new TensorDict should be inserted.
            tensordict (TensorDictBase): The TensorDict to be inserted into the stack.

        """
        if not isinstance(tensordict, TensorDictBase):
            raise TypeError(
                "Expected new value to be TensorDictBase instance but got "
                f"{type(tensordict)} instead."
            )

        batch_size = self.tensordicts[0].batch_size
        device = self.tensordicts[0].device

        _batch_size = tensordict.batch_size
        _device = tensordict.device

        if device != _device:
            raise ValueError(
                f"Devices differ: stack has device={device}, new value has "
                f"device={_device}."
            )
        if _batch_size != batch_size:
            raise ValueError(
                f"Batch sizes in tensordicts differs: stack has "
                f"batch_size={batch_size}, new_value has batch_size={_batch_size}."
            )

        self.tensordicts.insert(index, tensordict)

        N = len(self.tensordicts)
        self._batch_size = self._compute_batch_size(batch_size, self.stack_dim, N)

    @lock_blocked
    def append(self, tensordict: T) -> None:
        """Append a TensorDict onto the stack.

        Analogous to list.append. The appended TensorDict must have compatible
        batch_size and device. The append operation is in-place, nothing is returned.

        Args:
            tensordict (TensorDictBase): The TensorDict to be appended onto the stack.

        """
        self.insert(len(self.tensordicts), tensordict)

    @property
    def is_locked(self) -> bool:
        if self._is_locked is not None:
            # if tensordicts have been locked through this Lazy stack, then we can
            # trust this lazy stack to contain the info.
            # In all other cases we must check
            return self._is_locked
        # If any of the tensordicts is not locked, we assume that the lazy stack
        # is not locked either. Caching is then disabled and
        for td in self.tensordicts:
            if not td.is_locked:
                return False
        else:
            # In this case, all tensordicts were locked before the lazy stack
            # was created and they were not locked through the lazy stack.
            # This means we cannot cache the value because this lazy stack
            # if not part of the graph. We don't want it to be part of the graph
            # because this object being locked is only a side-effect.
            # Calling self.lock_() here could however speed things up.
            return True

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    @property
    def _lock_parents_weakrefs(self):
        """Weakrefs of all tensordicts that need to be unlocked for this to be unlocked."""
        _lock_parents_weakrefs = []
        for tensordict in self.tensordicts:
            _lock_parents_weakrefs = (
                _lock_parents_weakrefs + tensordict._lock_parents_weakrefs
            )
        _lock_parents_weakrefs = [
            item for item in _lock_parents_weakrefs if item is not weakref.ref(self)
        ]
        return _lock_parents_weakrefs

    def _propagate_lock(self, lock_parents_weakrefs=None):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        is_root = lock_parents_weakrefs is None
        if is_root:
            lock_parents_weakrefs = []

        lock_parents_weakrefs = copy(lock_parents_weakrefs) + [weakref.ref(self)]
        for dest in self.tensordicts:
            dest._propagate_lock(lock_parents_weakrefs)

    @erase_cache
    def _propagate_unlock(self):
        # we can't set _is_locked to False because after it's unlocked, anything
        # can happen to a child tensordict.
        self._is_locked = None
        sub_tds = defaultdict()
        for child in self.tensordicts:
            # we want to make sure that if the same child is present twice in the
            # stack we won't iterate multiple times over it
            sub_tds[id(child)] = child._propagate_unlock() + [child]
        sub_tds = [item for value in sub_tds.values() for item in value]
        self._is_shared = False
        self._is_memmap = False
        return sub_tds

    def __repr__(self):
        fields = _td_fields(self)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        exclusive_fields_str = indent(
            f"exclusive_fields={{{self._repr_exclusive_fields()}}}", 4 * " "
        )
        batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
        device_str = indent(f"device={self.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
        stack_dim = indent(f"stack_dim={self.stack_dim}", 4 * " ")
        string = ",\n".join(
            [
                field_str,
                exclusive_fields_str,
                batch_size_str,
                device_str,
                is_shared_str,
                stack_dim,
            ]
        )
        return f"{type(self).__name__}(\n{string})"

    def _repr_exclusive_fields(self):
        keys = set(self.keys())
        exclusive_keys = [
            _td_fields(td, [k for k in td.keys() if k not in keys])
            for td in self.tensordicts
        ]
        exclusive_key_str = ",\n".join(
            [
                indent(f"{i} ->{line}", 4 * " ")
                for i, line in enumerate(exclusive_keys)
                if line != "\n"
            ]
        )

        return "\n" + exclusive_key_str

    lock_ = TensorDictBase.lock_
    lock = _renamed_inplace_method(lock_)

    unlock_ = TensorDictBase.unlock_
    unlock = _renamed_inplace_method(unlock_)

    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    _convert_to_tensordict = TensorDict._convert_to_tensordict
    _index_tensordict = TensorDict._index_tensordict
    masked_select = TensorDict.masked_select
    reshape = TensorDict.reshape
    split = TensorDict.split
    to_module = TensorDict.to_module
    permute = TensorDict.permute
    transpose = TensorDict.transpose
    view = TensorDict.view


def _iter_items_lazystack(
    tensordict: LazyStackedTensorDict, return_none_for_het_values: bool = False
) -> Iterator[tuple[str, CompatibleType]]:
    for key in tensordict.keys():
        try:
            value = tensordict.get(key)
        except RuntimeError as err:
            if return_none_for_het_values and re.match(
                r"Found more than one unique shape in the tensors", str(err)
            ):
                # this is a het key
                value = None
            else:
                raise err
        yield key, value


_register_tensor_class(LazyStackedTensorDict)
