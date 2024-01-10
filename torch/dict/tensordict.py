from __future__ import annotations

import abc

import json
import numbers
import os
from collections import defaultdict
from copy import copy
from numbers import Number
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Type
from warnings import warn

import numpy as np

from functorch import dim as ftdim

import torch
from torch import Tensor
from torch._C._functorch import _add_batch_dim, _remove_batch_dim
from torch.jit._shape_functions import infer_size_impl
from torch.utils._pytree import tree_map
from ._memmap import MemoryMappedTensor
from .base import (
    _ACCEPTED_CLASSES,
    _default_is_leaf,
    _is_tensor_collection,
    _register_tensor_class,
    BEST_ATTEMPT_INPLACE,
    CompatibleType,
    is_tensor_collection,
    NO_DEFAULT,
    T,
    TensorDictBase,
)
from .utils import (
    _clone_value,
    _expand_to_match_shape,
    _get_item,
    _get_leaf_tensordict,
    _get_shape_from_args,
    _getitem_batch_size,
    _is_number,
    _is_shared,
    _is_tensorclass,
    _KEY_ERROR,
    _LOCK_ERROR,
    _NON_STR_KEY_ERR,
    _NON_STR_KEY_TUPLE_ERR,
    _parse_to,
    _prune_selected_keys,
    _set_item,
    _set_max_batch_size,
    _shape,
    _STRDTYPE2DTYPE,
    _StringOnlyDict,
    _sub_index,
    _unravel_key_to_tuple,
    as_decorator,
    Buffer,
    cache,
    convert_ellipsis_to_idx,
    DeviceType,
    expand_as_right,
    IndexType,
    is_tensorclass,
    lock_blocked,
    NestedKey,
    unravel_key_list,
)

_register_tensor_class(ftdim.Tensor)

__base__setattr__ = torch.nn.Module.__setattr__


class TensorDict(TensorDictBase):
    """A batched dictionary of tensors.

    TensorDict is a tensor container where all tensors are stored in a
    key-value pair fashion and where each element shares the same first ``N``
    leading dimensions shape, where is an arbitrary number with ``N >= 0``.

    Additionally, if the tensordict has a specified device, then each element
    must share that device.

    TensorDict instances support many regular tensor operations with the notable
    exception of algebraic operations:

    - operations on shape: when a shape operation is called (indexing,
      reshape, view, expand, transpose, permute,
      unsqueeze, squeeze, masking etc), the operations is done as if it
      was executed on a tensor of the same shape as the batch size then
      expended to the right, e.g.:

        >>> td = TensorDict({'a': torch.zeros(3, 4, 5)}, batch_size=[3, 4])
        >>> # returns a TensorDict of batch size [3, 4, 1]:
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> # returns a TensorDict of batch size [12]
        >>> td_view = td.view(-1)
        >>> # returns a tensor of batch size [12, 4]
        >>> a_view = td.view(-1).get("a")

    - casting operations: a TensorDict can be cast on a different device using

        >>> td_cpu = td.to("cpu")
        >>> dictionary = td.to_dict()

      A call of the `.to()` method with a dtype will return an error.

    - Cloning (:meth:`~TensorDictBase.clone`), contiguous (:meth:`~TensorDictBase.contiguous`);

    - Reading: `td.get(key)`, `td.get_at(key, index)`

    - Content modification: :obj:`td.set(key, value)`, :obj:`td.set_(key, value)`,
      :obj:`td.update(td_or_dict)`, :obj:`td.update_(td_or_dict)`, :obj:`td.fill_(key,
      value)`, :obj:`td.rename_key_(old_name, new_name)`, etc.

    - Operations on multiple tensordicts: `torch.cat(tensordict_list, dim)`,
      `torch.stack(tensordict_list, dim)`, `td1 == td2`, `td.apply(lambda x+y, other_td)` etc.

    Args:
        source (TensorDict or Dict[NestedKey, Union[Tensor, TensorDictBase]]): a
            data source. If empty, the tensordict can be populated subsequently.
        batch_size (iterable of int): a batch size for the
            tensordict. The batch size can be modified subsequently as long
            as it is compatible with its content. Unless the
            source is another TensorDict, the batch_size argument must be
            provided as it won't be inferred from the data.
        device (torch.device or compatible type, optional): a device for the
            TensorDict. If provided, all tensors will be stored on that device.
            If not, tensors on different devices are allowed.
        names (lsit of str, optional): the names of the dimensions of the
            tensordict. If provided, its length must match the one of the
            ``batch_size``. Defaults to ``None`` (no dimension name, or ``None``
            for every dimension).

    Examples:
        >>> import torch
        >>> from torch.dict import TensorDict
        >>> source = {'random': torch.randn(3, 4),
        ...     'zeros': torch.zeros(3, 4, 5)}
        >>> batch_size = [3]
        >>> td = TensorDict(source, batch_size=batch_size)
        >>> print(td.shape)  # equivalent to td.batch_size
        torch.Size([3])
        >>> td_unqueeze = td.unsqueeze(-1)
        >>> print(td_unqueeze.get("zeros").shape)
        torch.Size([3, 1, 4, 5])
        >>> print(td_unqueeze[0].shape)
        torch.Size([1])
        >>> print(td_unqueeze.view(-1).shape)
        torch.Size([3])
        >>> print((td.clone()==td).all())
        True

    """

    _td_dim_names = None

    def __init__(
        self,
        source: T | dict[str, CompatibleType],
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
        names: Sequence[str] | None = None,
        _run_checks: bool = True,
        _is_shared: bool | None = False,
        _is_memmap: bool | None = False,
    ) -> None:
        self._is_shared = _is_shared
        self._is_memmap = _is_memmap
        if device is not None and isinstance(device, (int, str)):
            device = torch.device(device)
        self._device = device

        self._tensordict = _tensordict = _StringOnlyDict()
        if not _run_checks:
            self._batch_size = batch_size
            if source:  # faster than calling items
                for key, value in source.items():
                    if isinstance(value, dict):
                        value = TensorDict(
                            value,
                            batch_size=self._batch_size,
                            device=self._device,
                            _run_checks=_run_checks,
                            _is_shared=_is_shared,
                            _is_memmap=_is_memmap,
                        )
                    _tensordict[key] = value
            self._td_dim_names = names
        else:
            if not isinstance(source, (TensorDictBase, dict)):
                raise ValueError(
                    "A TensorDict source is expected to be a TensorDictBase "
                    f"sub-type or a dictionary, found type(source)={type(source)}."
                )
            self._batch_size = self._parse_batch_size(source, batch_size)
            self.names = names

            if source is not None:
                for key, value in source.items():
                    self.set(key, value)

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        as_module: bool = False,
        lock: bool = False,
        use_state_dict: bool = False,
    ):
        result = cls._from_module(
            module=module, as_module=as_module, use_state_dict=use_state_dict
        )
        if lock:
            result.lock_()
        return result

    @classmethod
    def _from_module(
        cls,
        module: torch.nn.Module,
        as_module: bool = False,
        use_state_dict: bool = False,
        prefix="",
    ):
        destination = {}
        if use_state_dict:
            keep_vars = False
            # do we need this feature atm?
            local_metadata = {}
            # if hasattr(destination, "_metadata"):
            #     destination._metadata[prefix[:-1]] = local_metadata
            for hook in module._state_dict_pre_hooks.values():
                hook(module, prefix, keep_vars)
            module._save_to_state_dict(destination, "", keep_vars)
        else:
            for name, param in module._parameters.items():
                if param is None:
                    continue
                destination[name] = param
            for name, buffer in module._buffers.items():
                if buffer is None:
                    continue
                destination[name] = buffer

        if use_state_dict:
            for hook in module._state_dict_hooks.values():
                hook_result = hook(module, destination, prefix, local_metadata)
                if hook_result is not None:
                    destination = hook_result
        destination = TensorDict(destination, batch_size=[])
        for name, submodule in module._modules.items():
            if submodule is not None:
                subtd = cls._from_module(
                    module=submodule,
                    as_module=as_module,
                    use_state_dict=use_state_dict,
                    prefix=prefix + name + ".",
                )
                destination._set_str(name, subtd, validated=True, inplace=False)

        if as_module:
            from .params import TensorDictParams

            return TensorDictParams(destination, no_convert=True)
        return destination

    def is_empty(self):
        for _ in self._tensordict:
            return False
        return True

    @as_decorator()
    def to_module(
        self,
        module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest=None,
        memo=None,
        use_state_dict: bool = False,
    ):
        # we use __dict__ directly to avoid the getattr/setattr overhead whenever we can
        __dict__ = module.__dict__

        swap = None
        has_set_device = False
        if memo is None:
            hooks = getattr(
                torch.nn.modules.module, "_global_parameter_registration_hooks", {}
            )
            memo = {"hooks": tuple(hooks.values())}
        else:
            hooks = memo["hooks"]
        if return_swap:
            # this could break if the device and batch-size are not congruent.
            # For batch-size it is a minor issue (unlikely that a td with batch-size
            # is passed with to_module) but for the device it could be a problem.
            if swap_dest is None:
                swap = TensorDict({}, batch_size=torch.Size(()), _run_checks=False)
            else:
                swap = swap_dest
            memo[id(module)] = swap
            _swap = {}
        if use_state_dict:
            if inplace is not None:
                raise RuntimeError(
                    "inplace argument cannot be passed when use_state_dict=True."
                )
            # execute module's pre-hooks
            state_dict = self.flatten_keys(".")
            prefix = ""
            strict = True
            local_metadata = {}
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            for hook in module._load_state_dict_pre_hooks.values():
                hook(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )

            def convert_type(x, y):
                if isinstance(y, torch.nn.Parameter):
                    return torch.nn.Parameter(x)
                if isinstance(y, Buffer):
                    return Buffer(x)
                return x

            input = state_dict.unflatten_keys(".").apply(convert_type, self)
        else:
            input = self
            inplace = bool(inplace)

        for key, value in input.items():
            from functorch import dim as ftdim

            if isinstance(value, (Tensor, ftdim.Tensor)):
                if module.__class__.__setattr__ is __base__setattr__:
                    # if setattr is the native nn.Module.setattr, we can rely on _set_tensor_dict
                    local_out = _set_tensor_dict(
                        __dict__, hooks, module, key, value, inplace
                    )
                else:
                    if return_swap:
                        local_out = getattr(module, key)
                    if not inplace:
                        # use specialized __setattr__ if needed
                        delattr(module, key)
                        setattr(module, key, value)
                    else:
                        new_val = local_out
                        if return_swap:
                            local_out = local_out.clone()
                        new_val.data.copy_(value.data)
            else:
                if value.is_empty():
                    # if there is at least one key, we must populate the module.
                    # Otherwise we just go to the next key
                    continue
                if swap_dest is not None:
                    local_dest = swap_dest._get_str(key, default=NO_DEFAULT)
                else:
                    local_dest = None
                child = __dict__["_modules"][key]
                if id(child) in memo:
                    local_out = memo[id(child)]
                else:
                    local_out = value.to_module(
                        child,
                        inplace=inplace,
                        return_swap=return_swap,
                        swap_dest=local_dest,
                        memo=memo,
                        use_state_dict=use_state_dict,
                    )
                # we don't want to do this op more than once
                if return_swap and (
                    not has_set_device
                    and swap.device is not None
                    and local_out.device is not None
                    and local_out.device != swap.device
                ):
                    has_set_device = True
                    # map out to the local_out device
                    swap = swap.to(device=local_out.device)

            if return_swap:
                _swap[key] = local_out
        if return_swap:
            if isinstance(swap, TensorDict):
                # this is very ad-hoc but faster than calling _set_str every time
                swap._tensordict.update(_swap)
            else:
                swap.update(_swap)
        return swap

    def __ne__(self, other: object) -> T | bool:
        if _is_tensorclass(other):
            return other != self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 != other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value != other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return True

    def __xor__(self, other: object) -> T | bool:
        if _is_tensorclass(other):
            return other ^ self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 ^ other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value ^ other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return True

    def __or__(self, other: object) -> T | bool:
        if _is_tensorclass(other):
            return other | self
        if isinstance(other, (dict,)) or _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(
                    f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
                )
            d = {}
            for key, item1 in self.items():
                d[key] = item1 | other.get(key)
            return TensorDict(batch_size=self.batch_size, source=d, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value | other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __eq__(self, other: object) -> T | bool:
        if is_tensorclass(other):
            return other == self
        if isinstance(other, (dict,)):
            other = self.empty(recurse=True).update(other)
        if _is_tensor_collection(other.__class__):
            keys1 = set(self.keys())
            keys2 = set(other.keys())
            if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
                raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
            d = {}
            for key, item1 in self.items():
                d[key] = item1 == other.get(key)
            return TensorDict(source=d, batch_size=self.batch_size, device=self.device)
        if isinstance(other, (numbers.Number, Tensor)):
            return TensorDict(
                {key: value == other for key, value in self.items()},
                self.batch_size,
                device=self.device,
            )
        return False

    def __setitem__(
        self,
        index: IndexType,
        value: T | dict | numbers.Number | CompatibleType,
    ) -> None:
        istuple = isinstance(index, tuple)
        if istuple or isinstance(index, str):
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

        # we must use any and because using Ellipsis in index can break with some indices
        if index is Ellipsis or (
            isinstance(index, tuple) and any(idx is Ellipsis for idx in index)
        ):
            index = convert_ellipsis_to_idx(index, self.batch_size)

        if isinstance(value, (TensorDictBase, dict)):
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            if isinstance(value, dict):
                value = self.empty(recurse=True)[index].update(value)
            if value.batch_size != indexed_bs:
                # try to expand on the left (broadcasting)
                try:
                    value = value.expand(indexed_bs)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"indexed destination TensorDict batch size is {indexed_bs} "
                        f"(batch_size = {self.batch_size}, index={index}), "
                        f"which differs from the source batch size {value.batch_size}"
                    ) from err

            keys = set(self.keys())
            if any(key not in keys for key in value.keys()):
                subtd = self._get_sub_tensordict(index)
            for key, item in value.items():
                if key in keys:
                    self._set_at_str(key, item, index, validated=False)
                else:
                    subtd.set(key, item, inplace=True)
        else:
            for key in self.keys():
                self.set_at_(key, value, index)

    def all(self, dim: int = None) -> bool | TensorDictBase:
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim

            names = None
            if self._has_names():
                names = copy(self.names)
                names = [name for i, name in enumerate(names) if i != dim]

            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=names,
            )
        return all(value.all() for value in self.values())

    def any(self, dim: int = None) -> bool | TensorDictBase:
        if dim is not None and (dim >= self.batch_dims or dim < -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than or equal to -tensordict.batch_dims and "
                "smaller than tensordict.batch_dims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim

            names = None
            if self._has_names():
                names = copy(self.names)
                names = [name for i, name in enumerate(names) if i != dim]

            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self.device,
                names=names,
            )
        return any([value.any() for value in self.values()])

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
            out = self
        elif batch_size is not None:
            out = TensorDict(
                {},
                batch_size=torch.Size(batch_size),
                names=names,
                device=self.device if not device else device,
                _run_checks=False,
                **constructor_kwargs,
            )
        else:
            out = TensorDict(
                {},
                batch_size=self.batch_size,
                device=self.device if not device else device,
                names=self.names if self._has_names() else None,
                _run_checks=False,
                **constructor_kwargs,
            )

        is_locked = out.is_locked
        if not inplace and is_locked:
            out.unlock_()

        for key, item in self.items():
            if not call_on_nested and _is_tensor_collection(item.__class__):
                if default is not NO_DEFAULT:
                    _others = [_other._get_str(key, default=None) for _other in others]
                    _others = [
                        self.empty() if _other is None else _other for _other in _others
                    ]
                else:
                    _others = [
                        _other._get_str(key, default=NO_DEFAULT) for _other in others
                    ]

                item_trsf = item._apply_nest(
                    fn,
                    *_others,
                    inplace=inplace,
                    batch_size=batch_size,
                    device=device,
                    checked=checked,
                    named=named,
                    default=default,
                    **constructor_kwargs,
                )
            else:
                _others = [_other._get_str(key, default=default) for _other in others]
                if named:
                    item_trsf = fn(key, item, *_others)
                else:
                    item_trsf = fn(item, *_others)
            if item_trsf is not None:
                if isinstance(self, _SubTensorDict):
                    out.set(key, item_trsf, inplace=inplace)
                else:
                    out._set_str(
                        key,
                        item_trsf,
                        inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                        validated=checked,
                    )

        if not inplace and is_locked:
            out.lock_()
        return out

    # Functorch compatibility
    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim, vmap_level):
        td = self
        out = TensorDict(
            {
                key: value._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)
                if is_tensor_collection(value)
                else _add_batch_dim(value, in_dim, vmap_level)
                for key, value in td.items()
            },
            batch_size=[b for i, b in enumerate(td.batch_size) if i != in_dim],
            names=[name for i, name in enumerate(td.names) if i != in_dim],
        )
        return out

    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        new_batch_size = list(self.batch_size)
        new_batch_size.insert(out_dim, batch_size)
        new_names = list(self.names)
        new_names.insert(out_dim, None)
        out = TensorDict(
            {
                key: value._remove_batch_dim(
                    vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                )
                if is_tensor_collection(value)
                else _remove_batch_dim(value, vmap_level, batch_size, out_dim)
                for key, value in self.items()
            },
            batch_size=new_batch_size,
            names=new_names,
        )
        return out

    def _convert_to_tensordict(self, dict_value: dict[str, Any]) -> T:
        return TensorDict(
            dict_value,
            batch_size=self.batch_size,
            device=self.device,
            _is_shared=self._is_shared,
            _is_memmap=self._is_memmap,
        )

    def _index_tensordict(
        self,
        index: IndexType,
        new_batch_size: torch.Size | None = None,
        names: List[str] | None = None,
    ) -> T:
        batch_size = self.batch_size
        batch_dims = len(batch_size)
        if (
            not batch_size
            and index is not None
            and (not isinstance(index, tuple) or any(idx is not None for idx in index))
        ):
            raise RuntimeError(
                f"indexing a tensordict with td.batch_dims==0 is not permitted. Got index {index}."
            )
        if names is None:
            names = self._get_names_idx(index)
        if new_batch_size is not None:
            batch_size = new_batch_size
        else:
            batch_size = _getitem_batch_size(batch_size, index)
        source = {}
        for key, item in self.items():
            if isinstance(item, TensorDict):
                # this is the simplest case, we can pre-compute the batch size easily
                new_batch_size = batch_size + item.batch_size[batch_dims:]
                source[key] = item._index_tensordict(
                    index, new_batch_size=new_batch_size
                )
            else:
                source[key] = _get_item(item, index)
        return TensorDict(
            source=source,
            batch_size=batch_size,
            device=self.device,
            names=names,
            _run_checks=False,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
        )

    def expand(self, *args, **kwargs) -> T:
        tensordict_dims = self.batch_dims
        shape = _get_shape_from_args(*args, **kwargs)

        # new shape dim check
        if len(shape) < len(self.shape):
            raise RuntimeError(
                f"the number of sizes provided ({len(shape)}) must be greater or equal to the number of "
                f"dimensions in the TensorDict ({tensordict_dims})"
            )

        # new shape compatability check
        for old_dim, new_dim in zip(self.batch_size, shape[-tensordict_dims:]):
            if old_dim != 1 and new_dim != old_dim:
                raise RuntimeError(
                    "Incompatible expanded shape: The expanded shape length at non-singleton dimension should be same "
                    f"as the original length. target_shape = {shape}, existing_shape = {self.batch_size}"
                )

        def _expand(tensor):
            tensor_shape = tensor.shape
            tensor_dims = len(tensor_shape)
            last_n_dims = tensor_dims - tensordict_dims
            if last_n_dims > 0:
                new_shape = (*shape, *tensor_shape[-last_n_dims:])
            else:
                new_shape = shape
            return tensor.expand(new_shape)

        names = [None] * (len(shape) - tensordict_dims) + self.names
        return self._fast_apply(
            _expand, batch_size=shape, call_on_nested=True, names=names
        )

    def unbind(self, dim: int) -> tuple[T, ...]:
        if dim < 0:
            dim = self.batch_dims + dim
        batch_size = torch.Size([s for i, s in enumerate(self.batch_size) if i != dim])
        names = None
        if self._has_names():
            names = copy(self.names)
            names = [name for i, name in enumerate(names) if i != dim]
        out = []
        # unbind_self_dict = {key: tensor.unbind(dim) for key, tensor in self.items()}
        prefix = (slice(None),) * dim
        for _idx in range(self.batch_size[dim]):
            _idx = prefix + (_idx,)
            td = self._index_tensordict(_idx, new_batch_size=batch_size, names=names)
            out.append(td)
            if self.is_shared():
                td._is_shared = True
            elif self.is_memmap():
                td._is_memmap = True
        return tuple(out)

    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        # we must use slices to keep the storage of the tensors
        WRONG_TYPE = "split(): argument 'split_size' must be int or list of ints"
        batch_size = self.batch_size
        batch_sizes = []
        batch_dims = len(batch_size)
        if dim < 0:
            dim = len(batch_size) + dim
        if dim >= batch_dims or dim < 0:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{self.batch_dims}, {self.batch_dims - 1}], but got {dim})"
            )
        max_size = batch_size[dim]
        if isinstance(split_size, int):
            idx0 = 0
            idx1 = min(max_size, split_size)
            split_sizes = [slice(idx0, idx1)]
            batch_sizes.append(
                torch.Size(
                    tuple(
                        d if i != dim else idx1 - idx0 for i, d in enumerate(batch_size)
                    )
                )
            )
            while idx1 < max_size:
                idx0 = idx1
                idx1 = min(max_size, idx1 + split_size)
                split_sizes.append(slice(idx0, idx1))
                batch_sizes.append(
                    torch.Size(
                        tuple(
                            d if i != dim else idx1 - idx0
                            for i, d in enumerate(batch_size)
                        )
                    )
                )
        elif isinstance(split_size, (list, tuple)):
            if len(split_size) == 0:
                raise RuntimeError("Insufficient number of elements in split_size.")
            try:
                idx0 = 0
                idx1 = split_size[0]
                split_sizes = [slice(idx0, idx1)]
                batch_sizes.append(
                    torch.Size(
                        tuple(
                            d if i != dim else idx1 - idx0
                            for i, d in enumerate(batch_size)
                        )
                    )
                )
                for idx in split_size[1:]:
                    idx0 = idx1
                    idx1 = min(max_size, idx1 + idx)
                    split_sizes.append(slice(idx0, idx1))
                    batch_sizes.append(
                        torch.Size(
                            tuple(
                                d if i != dim else idx1 - idx0
                                for i, d in enumerate(batch_size)
                            )
                        )
                    )
            except TypeError:
                raise TypeError(WRONG_TYPE)

            if idx1 < batch_size[dim]:
                raise RuntimeError(
                    f"Split method expects split_size to sum exactly to {self.batch_size[dim]} (tensor's size at dimension {dim}), but got split_size={split_size}"
                )
        else:
            raise TypeError(WRONG_TYPE)
        index = (slice(None),) * dim
        names = self.names
        return tuple(
            self._index_tensordict(index + (ss,), new_batch_size=bs, names=names)
            for ss, bs in zip(split_sizes, batch_sizes)
        )

    def masked_select(self, mask: Tensor) -> T:
        d = {}
        mask_expand = mask
        while mask_expand.ndimension() > self.batch_dims:
            mndim = mask_expand.ndimension()
            mask_expand = mask_expand.squeeze(-1)
            if mndim == mask_expand.ndimension():  # no more squeeze
                break
        for key, value in self.items():
            d[key] = value[mask_expand]
        dim = int(mask.sum().item())
        other_dim = self.shape[mask.ndim :]
        return TensorDict(
            device=self.device, source=d, batch_size=torch.Size([dim, *other_dim])
        )

    def _view(
        self,
        *args,
        **kwargs,
    ) -> T:
        shape = _get_shape_from_args(*args, **kwargs)
        if any(dim < 0 for dim in shape):
            shape = infer_size_impl(shape, self.numel())
        if torch.Size(shape) == self.shape:
            return self
        batch_dims = self.batch_dims

        def _view(tensor):
            return tensor.view((*shape, *tensor.shape[batch_dims:]))

        return self._fast_apply(_view, batch_size=shape, call_on_nested=True)

    def reshape(
        self,
        *args,
        **kwargs,
    ) -> T:
        shape = _get_shape_from_args(*args, **kwargs)
        if any(dim < 0 for dim in shape):
            shape = infer_size_impl(shape, self.numel())
            shape = torch.Size(shape)
        if torch.Size(shape) == self.shape:
            return self
        batch_dims = self.batch_dims

        def _reshape(tensor):
            return tensor.reshape((*shape, *tensor.shape[batch_dims:]))

        return self._fast_apply(_reshape, batch_size=shape, call_on_nested=True)

    def _transpose(self, dim0, dim1):
        if dim0 < 0:
            dim0 = self.ndim + dim0
        if dim1 < 0:
            dim1 = self.ndim + dim1
        if dim0 < 0 or dim1 < 0:
            raise ValueError(
                "The provided dimensions are incompatible with the tensordict batch-size."
            )
        if dim0 == dim1:
            return self

        def _transpose(tensor):
            return tensor.transpose(dim0, dim1)

        batch_size = list(self.batch_size)
        v0 = batch_size[dim0]
        v1 = batch_size[dim1]
        batch_size[dim1] = v0
        batch_size[dim0] = v1
        return self._fast_apply(
            _transpose, batch_size=torch.Size(batch_size), call_on_nested=True
        )

    def _permute(self, *args, **kwargs):
        dims_list = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
        dims_list = [dim if dim >= 0 else self.ndim + dim for dim in dims_list]
        if any(dim < 0 or dim >= self.ndim for dim in dims_list):
            raise ValueError(
                "Received an permutation order incompatible with the tensordict shape."
            )
        # note: to allow this to work recursively, we must allow permutation order with fewer elements than dims,
        # as long as this list is complete.
        if not np.array_equal(sorted(dims_list), range(len(dims_list))):
            raise ValueError(
                f"Cannot compute the permutation, got dims={dims_list} but expected a permutation of {list(range(len(dims_list)))}."
            )
        if not len(dims_list) and not self.batch_dims:
            return self
        if np.array_equal(dims_list, range(len(dims_list))):
            return self

        # min_dim, max_dim = -self.batch_dims, self.batch_dims - 1
        # seen = [False for dim in range(max_dim + 1)]
        # for idx in dims_list:
        #     if idx < min_dim or idx > max_dim:
        #         raise IndexError(
        #             f"dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {idx})"
        #         )
        #     if seen[idx]:
        #         raise RuntimeError("repeated dim in permute")
        #     seen[idx] = True
        def _permute(tensor):
            return tensor.permute(*dims_list, *range(len(dims_list), tensor.ndim))

        batch_size = self.batch_size
        batch_size = [batch_size[p] for p in dims_list] + list(
            batch_size[len(dims_list) :]
        )
        return self._fast_apply(_permute, batch_size=batch_size, call_on_nested=True)

    @classmethod
    def from_dict(cls, input_dict, batch_size=None, device=None, batch_dims=None):
        """Returns a TensorDict created from a dictionary or another :class:`~.tensordict.TensorDict`.

        If ``batch_size`` is not specified, returns the maximum batch size possible.

        This function works on nested dictionaries too, or can be used to determine the
        batch-size of a nested tensordict.

        Args:
            input_dict (dictionary, optional): a dictionary to use as a data source
                (nested keys compatible).
            batch_size (iterable of int, optional): a batch size for the tensordict.
            device (torch.device or compatible type, optional): a device for the TensorDict.
            batch_dims (int, optional): the ``batch_dims`` (ie number of leading dimensions
                to be considered for ``batch_size``). Exclusinve with ``batch_size``.
                Note that this is the __maximum__ number of batch dims of the tensordict,
                a smaller number is tolerated.

        Examples:
            >>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
            >>> print(TensorDict.from_dict(input_dict))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
            >>> # nested dict: the nested TensorDict can have a different batch-size
            >>> # as long as its leading dims match.
            >>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
            >>> print(TensorDict.from_dict(input_dict))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
            >>> # we can also use this to work out the batch sie of a tensordict
            >>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
            >>> print(TensorDict.from_dict(input_td))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3, 4]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)

        """
        if batch_dims is not None and batch_size is not None:
            raise ValueError(
                "Cannot pass both batch_size and batch_dims to `from_dict`."
            )

        batch_size_set = torch.Size(()) if batch_size is None else batch_size
        for key, value in list(input_dict.items()):
            if isinstance(value, (dict,)):
                # we don't know if another tensor of smaller size is coming
                # so we can't be sure that the batch-size will still be valid later
                input_dict[key] = TensorDict.from_dict(
                    value, batch_size=[], device=device, batch_dims=None
                )
        # _run_checks=False breaks because a tensor may have the same batch-size as the tensordict
        out = cls(
            input_dict,
            batch_size=batch_size_set,
            device=device,
        )
        if batch_size is None:
            _set_max_batch_size(out, batch_dims)
        else:
            out.batch_size = batch_size
        return out

    @staticmethod
    def _parse_batch_size(
        source: T | dict,
        batch_size: Sequence[int] | torch.Size | int | None = None,
    ) -> torch.Size:
        try:
            return torch.Size(batch_size)
        except Exception as err:
            if isinstance(batch_size, Number):
                return torch.Size([batch_size])
            elif isinstance(source, TensorDictBase):
                return source.batch_size
            raise ValueError(
                "batch size was not specified when creating the TensorDict "
                "instance and it could not be retrieved from source."
            ) from err

    @property
    def batch_dims(self) -> int:
        return len(self.batch_size)

    @batch_dims.setter
    def batch_dims(self, value: int) -> None:
        raise RuntimeError(
            f"Setting batch dims on {self.__class__.__name__} instances is "
            f"not allowed."
        )

    def _has_names(self):
        return self._td_dim_names is not None

    def _erase_names(self):
        self._td_dim_names = None

    @property
    def names(self):
        names = self._td_dim_names
        if names is None:
            return [None for _ in range(self.batch_dims)]
        return names

    def _get_names_idx(self, idx):
        if not self._has_names():
            names = None
        else:

            def is_boolean(idx):
                from functorch import dim as ftdim

                if isinstance(idx, ftdim.Dim):
                    return None
                if isinstance(idx, tuple) and len(idx) == 1:
                    return is_boolean(idx[0])
                if hasattr(idx, "dtype") and idx.dtype is torch.bool:
                    return idx.ndim
                return None

            num_boolean_dim = is_boolean(idx)
            names = self.names
            if num_boolean_dim:
                names = [None] + names[num_boolean_dim:]
            else:
                if not isinstance(idx, tuple):
                    idx = (idx,)
                if len(idx) < self.ndim:
                    idx = (*idx, Ellipsis)
                idx_names = convert_ellipsis_to_idx(idx, self.batch_size)
                # this will convert a [None, :, :, 0, None, 0] in [None, 0, 1, None, 3]
                count = 0
                idx_to_take = []
                for _idx in idx_names:
                    if _idx is None:
                        idx_to_take.append(None)
                    elif _is_number(_idx):
                        count += 1
                    else:
                        idx_to_take.append(count)
                        count += 1
                names = [names[i] if i is not None else None for i in idx_to_take]
        return names

    @names.setter
    def names(self, value):
        # we don't run checks on types for efficiency purposes
        if value is None:
            self._rename_subtds(value)
            self._erase_names()
            return
        num_none = sum(v is None for v in value)
        if num_none:
            num_none -= 1
        if len(set(value)) != len(value) - num_none:
            raise ValueError(f"Some dimension names are non-unique: {value}.")
        if len(value) != self.batch_dims:
            raise ValueError(
                "the length of the dimension names must equate the tensordict batch_dims attribute. "
                f"Got {value} for batch_dims {self.batch_dims}."
            )
        self._rename_subtds(value)
        self._td_dim_names = list(value)

    def _rename_subtds(self, names):
        if names is None:
            for item in self._tensordict.values():
                if _is_tensor_collection(type(item)):
                    item._erase_names()
            return
        for item in self._tensordict.values():
            if _is_tensor_collection(item.__class__):
                item_names = item.names
                td_names = list(names) + item_names[len(names) :]
                item.rename_(*td_names)

    @property
    def device(self) -> torch.device | None:
        """Device of the tensordict.

        Returns `None` if device hasn't been provided in the constructor or set via `tensordict.to(device)`.

        """
        return self._device

    @device.setter
    def device(self, value: DeviceType) -> None:
        raise RuntimeError(
            "device cannot be set using tensordict.device = device, "
            "because device cannot be updated in-place. To update device, use "
            "tensordict.to(new_device), which will return a new tensordict "
            "on the new device."
        )

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._batch_size_setter(new_size)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    # Checks
    def _check_is_shared(self) -> bool:
        share_list = [_is_shared(value) for value in self.values()]
        if any(share_list) and not all(share_list):
            shared_str = ", ".join(
                [f"{key}: {_is_shared(value)}" for key, value in self.items()]
            )
            raise RuntimeError(
                f"tensors must be either all shared or not, but mixed "
                f"features is not allowed. "
                f"Found: {shared_str}"
            )
        return all(share_list) and len(share_list) > 0

    def _check_device(self) -> None:
        devices = {value.device for value in self.values()}
        if self.device is not None and len(devices) >= 1 and devices != {self.device}:
            raise RuntimeError(
                f"TensorDict.device is {self._device}, but elements have "
                f"device values {devices}. If TensorDict.device is set then "
                "all elements must share that device."
            )

    def pin_memory(self) -> T:
        def pin_mem(tensor):
            return tensor.pin_memory()

        return self._fast_apply(pin_mem)

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> T:
        if inplace is not False:
            best_attempt = inplace is BEST_ATTEMPT_INPLACE
            inplace = self._convert_inplace(inplace, key)
        if not validated:
            value = self._validate_value(value, check_shape=True)
        if not inplace:
            if self._is_locked:
                raise RuntimeError(_LOCK_ERROR)
            self._tensordict[key] = value
        else:
            try:
                dest = self._get_str(key, default=NO_DEFAULT)
                if best_attempt and _is_tensor_collection(dest.__class__):
                    dest.update(value, inplace=True)
                else:
                    if dest is not value:
                        dest.copy_(value, non_blocking=True)
            except KeyError as err:
                raise err
            except Exception as err:
                raise ValueError(
                    f"Failed to update '{key}' in tensordict {self}"
                ) from err
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
        td = self._get_str(key[0], None)
        if td is None:
            td = self._create_nested_str(key[0])
            inplace = False
        elif not _is_tensor_collection(td.__class__):
            raise KeyError(
                f"The entry {key[0]} is already present in tensordict {self}."
            )
        td._set_tuple(key[1:], value, inplace=inplace, validated=validated)
        return self

    def _set_at_str(self, key, value, idx, *, validated):
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        tensor_in = self._get_str(key, NO_DEFAULT)

        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
        else:
            _set_item(tensor_in, idx, value, validated=validated)

        return self

    def _set_at_tuple(self, key, value, idx, *, validated):
        if len(key) == 1:
            return self._set_at_str(key[0], value, idx, validated=validated)
        if key[0] not in self.keys():
            # this won't work
            raise KeyError(f"key {key} not found in set_at_ with tensordict {self}.")
        else:
            td = self._get_str(key[0], NO_DEFAULT)
        td._set_at_tuple(key[1:], value, idx, validated=validated)
        return self

    @lock_blocked
    def del_(self, key: NestedKey) -> T:
        key = _unravel_key_to_tuple(key)
        if len(key) > 1:
            td, subkey = _get_leaf_tensordict(self, key)
            td.del_(subkey)
            return self

        del self._tensordict[key[0]]
        return self

    @lock_blocked
    def rename_key_(self, old_key: str, new_key: str, safe: bool = False) -> T:
        # these checks are not perfect, tuples that are not tuples of strings or empty
        # tuples could go through but (1) it will raise an error anyway and (2)
        # those checks are expensive when repeated often.
        if old_key == new_key:
            return self
        if not isinstance(old_key, (str, tuple)):
            raise TypeError(
                f"Expected old_name to be a string or a tuple of strings but found {type(old_key)}"
            )
        if not isinstance(new_key, (str, tuple)):
            raise TypeError(
                f"Expected new_name to be a string or a tuple of strings but found {type(new_key)}"
            )
        if safe and (new_key in self.keys(include_nested=True)):
            raise KeyError(f"key {new_key} already present in TensorDict.")

        if isinstance(new_key, str):
            self._set_str(new_key, self.get(old_key), inplace=False, validated=True)
        else:
            self._set_tuple(new_key, self.get(old_key), inplace=False, validated=True)
        self.del_(old_key)
        return self

    def _stack_onto_(self, list_item: list[CompatibleType], dim: int) -> TensorDict:
        # if not isinstance(key, str):
        #     raise ValueError("_stack_onto_ expects string keys.")
        for key in self.keys():
            vals = [item._get_str(key, None) for item in list_item]
            if all(v is None for v in vals):
                continue
            dest = self._get_str(key, NO_DEFAULT)
            torch.stack(
                vals,
                dim=dim,
                out=dest,
            )
        return self

    def entry_class(self, key: NestedKey) -> type:
        return type(self.get(key))

    def _stack_onto_at_(
        self,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> TensorDict:
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = convert_ellipsis_to_idx(idx, self.batch_size)
        for key in self.keys():
            vals = [td._get_str(key, NO_DEFAULT) for td in list_item]
            if all(v is None for v in vals):
                continue
            v = self._get_str(key, NO_DEFAULT)
            v_idx = v[idx]
            if v.data_ptr() != v_idx.data_ptr():
                raise IndexError(
                    f"Index {idx} is incompatible with stack(..., out=data) as the storages of the indexed tensors differ."
                )
            torch.stack(vals, dim=dim, out=v_idx)
            # raise ValueError(
            #     f"Cannot stack onto an indexed tensor with index {idx} "
            #     f"as its storage differs."
            # )
        return self

    def _get_str(self, key, default):
        first_key = key
        out = self._tensordict.get(first_key, None)
        if out is None:
            return self._default_get(first_key, default)
        return out

    def _get_tuple(self, key, default):
        first = self._get_str(key[0], default)
        if len(key) == 1 or first is default:
            return first
        try:
            return first._get_tuple(key[1:], default=default)
        except AttributeError as err:
            if "has no attribute" in str(err):
                raise ValueError(
                    f"Expected a TensorDictBase instance but got {type(first)} instead"
                    f" for key '{key[1:]}' in tensordict:\n{self}."
                )

    def share_memory_(self) -> T:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if self.device is not None and self.device.type == "cuda":
            # cuda tensors are shared by default
            return self
        for value in self.values():
            if (
                isinstance(value, Tensor)
                and value.device.type == "cpu"
                or _is_tensor_collection(value.__class__)
            ):
                value.share_memory_()
        self._is_shared = True
        self.lock_()
        return self

    def detach_(self) -> T:
        for value in self.values():
            value.detach_()
        return self

    def _memmap_(
        self,
        prefix: str | None,
        copy_existing: bool,
        executor,
        futures,
        inplace,
        like,
    ) -> T:
        def save_metadata(data: TensorDictBase, filepath, metadata=None):
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "shape": list(data.shape),
                    "device": str(data.device),
                    "_type": str(data.__class__),
                }
            )
            with open(filepath, "w") as json_metadata:
                json.dump(metadata, json_metadata)

        if prefix is not None:
            prefix = Path(prefix)
            if not prefix.exists():
                os.makedirs(prefix, exist_ok=True)
            metadata = {}
        if inplace and self.is_shared() and self.device.type == "cpu":
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        dest = (
            self
            if inplace
            else TensorDict(
                {},
                batch_size=self.batch_size,
                _is_memmap=True,
                _is_shared=False,
                names=self.names if self._has_names() else None,
                device=torch.device("cpu"),
            )
        )
        for key, value in self.items():
            if _is_tensor_collection(value.__class__):
                dest._tensordict[key] = value._memmap_(
                    prefix=prefix / key if prefix is not None else None,
                    copy_existing=copy_existing,
                    executor=executor,
                    futures=futures,
                    inplace=inplace,
                    like=like,
                )
                continue
            else:
                # user did specify location and memmap is in wrong place, so we copy
                def _populate(
                    dest=dest, value=value, key=key, copy_existing=copy_existing
                ):
                    filename = None if prefix is None else str(prefix / f"{key}.memmap")
                    dest._tensordict[key] = MemoryMappedTensor.from_tensor(
                        value.data if value.requires_grad else value,
                        filename=filename,
                        copy_existing=copy_existing,
                        existsok=True,
                        copy_data=not like,
                    )

                if executor is None:
                    _populate()
                else:
                    futures.append(executor.submit(_populate))
                if prefix is not None:
                    metadata[key] = {
                        "device": str(value.device),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }

        if prefix is not None:
            if executor is None:
                save_metadata(
                    dest,
                    prefix / "meta.json",
                    metadata=metadata,
                )
            else:
                futures.append(
                    executor.submit(save_metadata, dest, prefix / "meta.json", metadata)
                )
        if inplace:
            self._is_memmap = True
            self._is_shared = False  # since they are mutually exclusive
            self._device = torch.device("cpu")
        dest._is_locked = True
        return dest

    @classmethod
    def _load_memmap(cls, prefix: str, metadata: dict) -> T:
        if metadata["device"] == "None":
            metadata["device"] = None
        else:
            metadata["device"] = torch.device(metadata["device"])
        metadata["shape"] = torch.Size(metadata["shape"])

        out = cls({}, batch_size=metadata.pop("shape"), device=metadata.pop("device"))

        for key, entry_metadata in metadata.items():
            if not isinstance(entry_metadata, dict):
                # there can be other metadata
                continue
            dtype = entry_metadata.get("dtype", None)
            shape = entry_metadata.get("shape", None)
            if (
                not (prefix / f"{key}.memmap").exists()
                or dtype is None
                or shape is None
            ):
                # invalid dict means
                continue
            out._set_str(
                key,
                MemoryMappedTensor.from_filename(
                    dtype=_STRDTYPE2DTYPE[dtype],
                    shape=torch.Size(entry_metadata["shape"]),
                    filename=str(prefix / f"{key}.memmap"),
                ),
                validated=True,
                inplace=False,
            )
        # iterate over folders and load them
        for path in prefix.iterdir():
            if path.is_dir():
                key = path.parts[len(prefix.parts) :]
                out.set(key, TensorDict.load_memmap(path))
        return out

    def to(self, *args, **kwargs: Any) -> T:
        device, dtype, non_blocking, convert_to_format, batch_size = _parse_to(
            *args, **kwargs
        )
        result = self

        if device is not None and dtype is None and device == self.device:
            return result

        if convert_to_format is not None:

            def to(tensor):
                return tensor.to(device, dtype, non_blocking, convert_to_format)

        else:

            def to(tensor):
                return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)

        apply_kwargs = {}
        if device is not None or dtype is not None:
            apply_kwargs["device"] = device if device is not None else self.device
            apply_kwargs["batch_size"] = batch_size
            result = result._fast_apply(to, **apply_kwargs)
        elif batch_size is not None:
            result.batch_size = batch_size
        return result

    def where(self, condition, other, *, out=None, pad=None):
        if _is_tensor_collection(other.__class__):

            def func(tensor, _other, key):
                if tensor is None:
                    if pad is not None:
                        tensor = _other
                        _other = pad
                    else:
                        raise KeyError(
                            f"Key {key} not found and no pad value provided."
                        )
                    cond = expand_as_right(~condition, tensor)
                elif _other is None:
                    if pad is not None:
                        _other = pad
                    else:
                        raise KeyError(
                            f"Key {key} not found and no pad value provided."
                        )
                    cond = expand_as_right(condition, tensor)
                else:
                    cond = expand_as_right(condition, tensor)
                return torch.where(
                    condition=cond,
                    input=tensor,
                    other=_other,
                )

            result = self.empty() if out is None else out
            other_keys = set(other.keys())
            # we turn into a list because out could be = to self!
            for key in list(self.keys()):
                tensor = self._get_str(key, default=NO_DEFAULT)
                _other = other._get_str(key, default=None)
                if _is_tensor_collection(type(tensor)):
                    _out = None if out is None else out._get_str(key, None)
                    if _other is None:
                        _other = tensor.empty()
                    val = tensor.where(
                        condition=condition, other=_other, out=_out, pad=pad
                    )
                else:
                    val = func(tensor, _other, key)
                result._set_str(key, val, inplace=False, validated=True)
                other_keys.discard(key)
            for key in other_keys:
                tensor = None
                _other = other._get_str(key, default=NO_DEFAULT)
                if _is_tensor_collection(type(_other)):
                    try:
                        tensor = _other.empty()
                    except NotImplementedError:
                        # H5 tensordicts do not support select()
                        tensor = _other.to_tensordict().empty()
                    val = _other.where(
                        condition=~condition, other=tensor, out=None, pad=pad
                    )
                else:
                    val = func(tensor, _other, key)
                result._set_str(key, val, inplace=False, validated=True)
            return result
        else:
            if out is None:

                def func(tensor):
                    return torch.where(
                        condition=expand_as_right(condition, tensor),
                        input=tensor,
                        other=other,
                    )

                return self._fast_apply(func)
            else:

                def func(tensor, _out):
                    return torch.where(
                        condition=expand_as_right(condition, tensor),
                        input=tensor,
                        other=other,
                        out=_out,
                    )

                return self._fast_apply(func, out)

    def masked_fill_(self, mask: Tensor, value: float | int | bool) -> T:
        for item in self.values():
            mask_expand = expand_as_right(mask, item)
            item.masked_fill_(mask_expand, value)
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def clone(self, recurse: bool = True) -> T:
        return TensorDict(
            source={key: _clone_value(value, recurse) for key, value in self.items()},
            batch_size=self.batch_size,
            device=self.device,
            names=copy(self._td_dim_names),
            _run_checks=False,
            _is_shared=self.is_shared() if not recurse else False,
            _is_memmap=self.is_memmap() if not recurse else False,
        )

    def contiguous(self) -> T:
        if not self.is_contiguous():
            return self.clone()
        return self

    def empty(self, recurse=False) -> T:
        if not recurse:
            return TensorDict(
                device=self._device,
                batch_size=self._batch_size,
                source={},
                names=self._td_dim_names,
                _run_checks=False,
                _is_memmap=False,
                _is_shared=False,
            )
        return super().empty(recurse=recurse)

    def select(self, *keys: NestedKey, inplace: bool = False, strict: bool = True) -> T:
        if inplace and self.is_locked:
            raise RuntimeError(_LOCK_ERROR)

        source = {}
        if len(keys):
            keys_to_select = None
            for key in keys:
                if isinstance(key, str):
                    subkey = []
                else:
                    key, subkey = key[0], key[1:]
                try:
                    source[key] = self.get(key)
                    if len(subkey):
                        if keys_to_select is None:
                            # delay creation of defaultdict
                            keys_to_select = defaultdict(list)
                        keys_to_select[key].append(subkey)
                except KeyError as err:
                    if not strict:
                        continue
                    else:
                        raise KeyError(f"select failed to get key {key}") from err
            if keys_to_select is not None:
                for key, val in keys_to_select.items():
                    source[key] = source[key].select(
                        *val, strict=strict, inplace=inplace
                    )

        out = TensorDict(
            device=self.device,
            batch_size=self.batch_size,
            source=source,
            # names=self.names if self._has_names() else None,
            names=self._td_dim_names,
            _run_checks=False,
            _is_memmap=self._is_memmap,
            _is_shared=self._is_shared,
        )
        if inplace:
            self._tensordict = out._tensordict
            return self
        return out

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _TensorDictKeysView:
        if not include_nested and not leaves_only:
            return self._tensordict.keys()
        else:
            return self._nested_keys(
                include_nested=include_nested, leaves_only=leaves_only, is_leaf=is_leaf
            )

    # @cache  # noqa: B019
    def _nested_keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _TensorDictKeysView:
        return _TensorDictKeysView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )

    def __getstate__(self):
        result = {
            key: val
            for key, val in self.__dict__.items()
            if key
            not in ("_last_op", "_cache", "__last_op_queue", "__lock_parents_weakrefs")
        }
        return result

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._cache = None
        self.__last_op_queue = None
        self._last_op = None
        if self._is_locked:
            # this can cause avoidable overhead, as we will be locking the leaves
            # then locking their parent, and the parent of the parent, every
            # time re-locking tensordicts that have already been locked.
            # To avoid this, we should lock only at the root, but it isn't easy
            # to spot what the root is...
            self._is_locked = False
            self.lock_()

    # some custom methods for efficiency
    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            return self._tensordict.items()
        else:
            return super().items(
                include_nested=include_nested, leaves_only=leaves_only, is_leaf=is_leaf
            )

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> Iterator[tuple[str, CompatibleType]]:
        if not include_nested and not leaves_only:
            return self._tensordict.values()
        else:
            return super().values(
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
            )


class _SubTensorDict(TensorDictBase):
    """A TensorDict that only sees an index of the stored tensors."""

    _is_shared = False
    _is_memmap = False
    _lazy = True
    _inplace_set = True
    _safe = False

    def __init__(
        self,
        source: T,
        idx: IndexType,
        batch_size: Sequence[int] | None = None,
    ) -> None:
        if not _is_tensor_collection(source.__class__):
            raise TypeError(
                f"Expected source to be a subclass of TensorDictBase, "
                f"got {type(source)}"
            )
        self._source = source
        idx = (
            (idx,)
            if not isinstance(
                idx,
                (
                    tuple,
                    list,
                ),
            )
            else tuple(idx)
        )
        if any(item is Ellipsis for item in idx):
            idx = convert_ellipsis_to_idx(idx, self._source.batch_size)
        self._batch_size = _getitem_batch_size(self._source.batch_size, idx)
        self.idx = idx

        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @staticmethod
    def _convert_ellipsis(idx, shape):
        if any(_idx is Ellipsis for _idx in idx):
            new_idx = []
            cursor = -1
            for _idx in idx:
                if _idx is Ellipsis:
                    if cursor == len(idx) - 1:
                        # then we can just skip
                        continue
                    n_upcoming = len(idx) - cursor - 1
                    while cursor < len(shape) - n_upcoming:
                        cursor += 1
                        new_idx.append(slice(None))
                else:
                    new_idx.append(_idx)
            return tuple(new_idx)
        return idx

    def exclude(self, *keys: str, inplace: bool = False) -> T:
        if inplace:
            return super().exclude(*keys, inplace=True)
        return self.to_tensordict().exclude(*keys, inplace=True)

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._batch_size_setter(new_size)

    @property
    def names(self):
        names = self._source._get_names_idx(self.idx)
        if names is None:
            return [None] * self.batch_dims
        return names

    @names.setter
    def names(self, value):
        raise RuntimeError(
            "Names of a subtensordict cannot be modified. Instantiate it as a TensorDict first."
        )

    def _has_names(self):
        return self._source._has_names()

    def _erase_names(self):
        raise RuntimeError(
            "Cannot erase names of a _SubTensorDict. Erase source TensorDict's names instead."
        )

    def _rename_subtds(self, names):
        for key in self.keys():
            if _is_tensor_collection(self.entry_class(key)):
                raise RuntimeError("Cannot rename nested sub-tensordict dimensions.")

    @property
    def device(self) -> None | torch.device:
        return self._source.device

    @device.setter
    def device(self, value: DeviceType) -> None:
        self._source.device = value

    def _preallocate(self, key: str, value: CompatibleType) -> T:
        return self._source.set(key, value)

    def _convert_inplace(self, inplace, key):
        has_key = key in self.keys()
        if inplace is not False:
            if inplace is True and not has_key:  # inplace could be None
                raise KeyError(
                    _KEY_ERROR.format(key, self.__class__.__name__, sorted(self.keys()))
                )
            inplace = has_key
        if not inplace and has_key:
            raise RuntimeError(
                "Calling `_SubTensorDict.set(key, value, inplace=False)` is "
                "prohibited for existing tensors. Consider calling "
                "_SubTensorDict.set_(...) or cloning your tensordict first."
            )
        elif not inplace and self.is_locked:
            raise RuntimeError(_LOCK_ERROR)
        return inplace

    def _set_str(
        self,
        key: NestedKey,
        value: dict[str, CompatibleType] | CompatibleType,
        *,
        inplace: bool,
        validated: bool,
    ) -> T:
        inplace = self._convert_inplace(inplace, key)
        # it is assumed that if inplace=False then the key doesn't exist. This is
        # checked in set method, but not here. responsibility lies with the caller
        # so that this method can have minimal overhead from runtime checks
        parent = self._source
        if not validated:
            value = self._validate_value(value, check_shape=True)
            validated = True
        if not inplace:
            if _is_tensor_collection(value.__class__):
                value_expand = _expand_to_match_shape(
                    parent.batch_size, value, self.batch_dims, self.device
                )
                for _key, _tensor in value.items():
                    value_expand._set_str(
                        _key,
                        _expand_to_match_shape(
                            parent.batch_size, _tensor, self.batch_dims, self.device
                        ),
                        inplace=inplace,
                        validated=validated,
                    )
            else:
                value_expand = torch.zeros(
                    (
                        *parent.batch_size,
                        *_shape(value)[self.batch_dims :],
                    ),
                    dtype=value.dtype,
                    device=self.device,
                )
                if self.is_shared() and self.device.type == "cpu":
                    value_expand.share_memory_()
                elif self.is_memmap():
                    value_expand = MemoryMappedTensor.from_tensor(value_expand)
            parent._set_str(key, value_expand, inplace=False, validated=validated)

        parent._set_at_str(key, value, self.idx, validated=validated)
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
        parent = self._source
        td = parent._get_str(key[0], None)
        if td is None:
            td = parent.select()
            parent._set_str(key[0], td, inplace=False, validated=True)
        _SubTensorDict(td, self.idx)._set_tuple(
            key[1:], value, inplace=inplace, validated=validated
        )
        return self

    def _set_at_str(self, key, value, idx, *, validated):
        tensor_in = self._get_str(key, NO_DEFAULT)
        if not validated:
            value = self._validate_value(value, check_shape=False)
            validated = True
        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
        else:
            _set_item(tensor_in, idx, value, validated=validated)
        # make sure that the value is updated
        self._source._set_at_str(key, tensor_in, self.idx, validated=validated)
        return self

    def _set_at_tuple(self, key, value, idx, *, validated):
        if len(key) == 1:
            return self._set_at_str(key[0], value, idx, validated=validated)
        if key[0] not in self.keys():
            # this won't work
            raise KeyError(f"key {key} not found in set_at_ with tensordict {self}.")
        else:
            td = self._get_str(key[0], NO_DEFAULT)
        td._set_at_tuple(key[1:], value, idx, validated=validated)
        return self

    # @cache  # noqa: B019
    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> _TensorDictKeysView:
        return self._source.keys(
            include_nested=include_nested, leaves_only=leaves_only, is_leaf=is_leaf
        )

    def entry_class(self, key: NestedKey) -> type:
        source_type = type(self._source.get(key))
        if _is_tensor_collection(source_type):
            return self.__class__
        return source_type

    def _stack_onto_(self, list_item: list[CompatibleType], dim: int) -> _SubTensorDict:
        self._source._stack_onto_at_(list_item, dim=dim, idx=self.idx)
        return self

    def to(self, *args, **kwargs: Any) -> T:
        device, dtype, non_blocking, convert_to_format, batch_size = _parse_to(
            *args, **kwargs
        )
        result = self

        if device is not None and dtype is None and device == self.device:
            return result
        return self.to_tensordict().to(*args, **kwargs)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def get(
        self,
        key: NestedKey,
        default: Tensor | str | None = NO_DEFAULT,
    ) -> CompatibleType:
        return self._source.get_at(key, self.idx, default=default)

    def _get_non_tensor(self, key: NestedKey, default=NO_DEFAULT):
        out = super()._get_non_tensor(key, default=default)
        from tensordict.tensorclass import NonTensorData

        if isinstance(out, _SubTensorDict) and isinstance(out._source, NonTensorData):
            return out._source.data
        return out

    def _get_str(self, key, default):
        if key in self.keys() and _is_tensor_collection(self.entry_class(key)):
            return _SubTensorDict(self._source._get_str(key, NO_DEFAULT), self.idx)
        return self._source._get_at_str(key, self.idx, default=default)

    def _get_tuple(self, key, default):
        return self._source._get_at_tuple(key, self.idx, default=default)

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> _SubTensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        keys = set(self.keys(False))
        for key, value in input_dict_or_td.items():
            key = _unravel_key_to_tuple(key)
            firstkey, subkey = key[0], key[1:]
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue
            if clone and hasattr(value, "clone"):
                value = value.clone()
            elif clone:
                value = tree_map(torch.clone, value)
            # the key must be a string by now. Let's check if it is present
            if firstkey in keys:
                target_class = self.entry_class(firstkey)
                if _is_tensor_collection(target_class):
                    target = self._source.get(firstkey)._get_sub_tensordict(self.idx)
                    if len(subkey):
                        sub_keys_to_update = _prune_selected_keys(
                            keys_to_update, firstkey
                        )
                        target.update(
                            {subkey: value},
                            inplace=False,
                            keys_to_update=sub_keys_to_update,
                        )
                        continue
                    elif isinstance(value, dict) or _is_tensor_collection(
                        value.__class__
                    ):
                        sub_keys_to_update = _prune_selected_keys(
                            keys_to_update, firstkey
                        )
                        target.update(value, keys_to_update=sub_keys_to_update)
                        continue
                    raise ValueError(
                        f"Tried to replace a tensordict with an incompatible object of type {type(value)}"
                    )
                else:
                    self._set_tuple(key, value, inplace=True, validated=False)
            else:
                self._set_tuple(
                    key,
                    value,
                    inplace=BEST_ATTEMPT_INPLACE if inplace else False,
                    validated=False,
                )
        return self

    def update_(
        self,
        input_dict: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        *,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> _SubTensorDict:
        return self.update_at_(
            input_dict,
            idx=self.idx,
            discard_idx_attr=True,
            clone=clone,
            keys_to_update=keys_to_update,
        )

    def update_at_(
        self,
        input_dict: dict[str, CompatibleType] | TensorDictBase,
        idx: IndexType,
        *,
        discard_idx_attr: bool = False,
        clone: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> _SubTensorDict:
        if keys_to_update is not None:
            if len(keys_to_update) == 0:
                return self
            keys_to_update = unravel_key_list(keys_to_update)
        for key, value in input_dict.items():
            key = _unravel_key_to_tuple(key)
            firstkey, _ = key[0], key[1:]
            if keys_to_update and not any(
                firstkey == ktu if isinstance(ktu, str) else firstkey == ktu[0]
                for ktu in keys_to_update
            ):
                continue
            if not isinstance(value, tuple(_ACCEPTED_CLASSES)):
                raise TypeError(
                    f"Expected value to be one of types {_ACCEPTED_CLASSES} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            if discard_idx_attr:
                self._source._set_at_tuple(
                    key,
                    value,
                    idx,
                    validated=False,
                )
            else:
                self._set_at_tuple(key, value, idx, validated=False)
        return self

    def get_parent_tensordict(self) -> T:
        if not isinstance(self._source, TensorDictBase):
            raise TypeError(
                f"_SubTensorDict was initialized with a source of type"
                f" {self._source.__class__.__name__}, "
                "parent tensordict not accessible"
            )
        return self._source

    @lock_blocked
    def del_(self, key: NestedKey) -> T:
        self._source = self._source.del_(key)
        return self

    def clone(self, recurse: bool = True) -> _SubTensorDict:
        """Clones the _SubTensorDict.

        Args:
            recurse (bool, optional): if ``True`` (default), a regular
                :class:`~.tensordict.TensorDict` instance will be created from the :class:`~.tensordict._SubTensorDict`.
                Otherwise, another :class:`~.tensordict._SubTensorDict` with identical content
                will be returned.

        Examples:
            >>> data = TensorDict({"a": torch.arange(4).reshape(2, 2,)}, batch_size=[2, 2])
            >>> sub_data = data._get_sub_tensordict([0,])
            >>> print(sub_data)
            _SubTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> # the data of both subtensordict is the same
            >>> print(data.get("a").data_ptr(), sub_data.get("a").data_ptr())
            140183705558208 140183705558208
            >>> sub_data_clone = sub_data.clone(recurse=True)
            >>> print(sub_data_clone)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>. print(sub_data.get("a").data_ptr())
            140183705558208
            >>> sub_data_clone = sub_data.clone(recurse=False)
            >>> print(sub_data_clone)
            _SubTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> print(sub_data.get("a").data_ptr())
            140183705558208
        """
        if not recurse:
            return _SubTensorDict(
                source=self._source.clone(recurse=False), idx=self.idx
            )
        return self.to_tensordict()

    def is_contiguous(self) -> bool:
        return all(value.is_contiguous() for value in self.values())

    def contiguous(self) -> T:
        if self.is_contiguous():
            return self
        return TensorDict(
            batch_size=self.batch_size,
            source={key: value for key, value in self.items()},
            device=self.device,
            names=self.names,
            _run_checks=False,
        )

    def select(self, *keys: str, inplace: bool = False, strict: bool = True) -> T:
        if inplace:
            self._source = self._source.select(*keys, strict=strict)
            return self
        return self._source.select(*keys, strict=strict)[self.idx]

    def expand(self, *args: int, inplace: bool = False) -> T:
        if len(args) == 1 and isinstance(args[0], Sequence):
            shape = tuple(args[0])
        else:
            shape = args
        return self._fast_apply(
            lambda x: x.expand((*shape, *x.shape[self.ndim :])), batch_size=shape
        )

    def is_shared(self) -> bool:
        return self._source.is_shared()

    def is_memmap(self) -> bool:
        return self._source.is_memmap()

    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> _SubTensorDict:
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    def pin_memory(self) -> T:
        self._source.pin_memory()
        return self

    def detach_(self) -> T:
        raise RuntimeError("Detaching a sub-tensordict in-place cannot be done.")

    def where(self, condition, other, *, out=None, pad=None):
        return self.to_tensordict().where(
            condition=condition, other=other, out=out, pad=pad
        )

    def masked_fill_(self, mask: Tensor, value: float | bool) -> T:
        for key, item in self.items():
            self.set_(key, torch.full_like(item, value))
        return self

    def masked_fill(self, mask: Tensor, value: float | bool) -> T:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        num_threads: int = 0,
    ) -> T:
        raise RuntimeError(
            "Converting a sub-tensordict values to memmap cannot be done."
        )

    def _memmap_(
        self,
        *,
        prefix: str | None,
        copy_existing: bool,
        executor,
        futures,
        inplace,
        like,
    ) -> T:
        if prefix is not None:

            def save_metadata(prefix=prefix, self=self):
                prefix = Path(prefix)
                if not prefix.exists():
                    os.makedirs(prefix, exist_ok=True)
                with open(prefix / "meta.json", "w") as f:
                    json.dump(
                        {
                            "_type": str(self.__class__),
                            "index": _index_to_str(self.idx),
                        },
                        f,
                    )

            if executor is None:
                save_metadata()
            else:
                futures.append(executor.submit(save_metadata))

        _source = self._source._memmap_(
            prefix=prefix / "_source" if prefix is not None else None,
            copy_existing=copy_existing,
            executor=executor,
            futures=futures,
            inplace=inplace,
            like=like,
        )
        if not inplace:
            result = _SubTensorDict(_source, idx=self.idx)
        else:
            result = self
        return result

    @classmethod
    def _load_memmap(cls, prefix: Path, metadata: dict):
        index = metadata["index"]
        return _SubTensorDict(
            TensorDict.load_memmap(prefix / "_source"), _str_to_index(index)
        )

    def share_memory_(self) -> T:
        raise RuntimeError(
            "Casting a sub-tensordict values to shared memory cannot be done."
        )

    @property
    def is_locked(self) -> bool:
        return self._source.is_locked

    @is_locked.setter
    def is_locked(self, value) -> bool:
        if value:
            self.lock_()
        else:
            self.unlock_()

    @as_decorator("is_locked")
    def lock_(self) -> T:
        # we can't lock sub-tensordicts because that would mean that the
        # parent tensordict cannot be modified either.
        if not self.is_locked:
            raise RuntimeError(
                "Cannot lock a _SubTensorDict. Lock the parent tensordict instead."
            )
        return self

    @as_decorator("is_locked")
    def unlock_(self) -> T:
        if self.is_locked:
            raise RuntimeError(
                "Cannot unlock a _SubTensorDict. Unlock the parent tensordict instead."
            )
        return self

    def _remove_lock(self, lock_id):
        raise RuntimeError(
            "Cannot unlock a _SubTensorDict. Unlock the parent tensordict instead."
        )

    def _propagate_lock(self, lock_ids=None):
        raise RuntimeError(
            "Cannot lock a _SubTensorDict. Lock the parent tensordict instead."
        )

    def __del__(self):
        pass

    def _create_nested_str(self, key):
        out = self.empty()
        self._set_str(key, out, inplace=False, validated=True)
        # the id of out changes
        return self._get_str(key, default=NO_DEFAULT)

    # TODO: check these implementations
    __eq__ = TensorDict.__eq__
    __ne__ = TensorDict.__ne__
    __setitem__ = TensorDict.__setitem__
    __xor__ = TensorDict.__xor__
    __or__ = TensorDict.__or__
    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared
    all = TensorDict.all
    any = TensorDict.any
    masked_select = TensorDict.masked_select
    memmap_like = TensorDict.memmap_like
    reshape = TensorDict.reshape
    split = TensorDict.split
    to_module = TensorDict.to_module
    unbind = TensorDict.unbind
    _permute = TensorDict._permute
    _transpose = TensorDict._transpose
    _view = TensorDict._view

    _add_batch_dim = TensorDict._add_batch_dim

    _apply_nest = TensorDict._apply_nest
    # def _apply_nest(self, *args, **kwargs):
    #     return self.to_tensordict()._apply_nest(*args, **kwargs)
    _convert_to_tensordict = TensorDict._convert_to_tensordict

    _get_names_idx = TensorDict._get_names_idx

    def _index_tensordict(self, index, new_batch_size=None, names=None):
        # we ignore the names and new_batch_size which are only provided for
        # efficiency purposes
        return self._get_sub_tensordict(index)

    def _remove_batch_dim(self, *args, **kwargs):
        raise NotImplementedError


###########################
# Keys utils


class _TensorDictKeysView:
    """A Key view for TensorDictBase instance.

    _TensorDictKeysView is returned when accessing tensordict.keys() and holds a
    reference to the original TensorDict. This class enables us to support nested keys
    when performing membership checks and when iterating over keys.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict

        >>> td = TensorDict(
        >>>     {"a": TensorDict({"b": torch.rand(1, 2)}, [1, 2]), "c": torch.rand(1)},
        >>>     [1],
        >>> )

        >>> assert "a" in td.keys()
        >>> assert ("a",) in td.keys()
        >>> assert ("a", "b") in td.keys()
        >>> assert ("a", "c") not in td.keys()

        >>> assert set(td.keys()) == {("a", "b"), "c"}
    """

    def __init__(
        self,
        tensordict: T,
        include_nested: bool,
        leaves_only: bool,
        is_leaf: Callable[[Type], bool] = None,
    ) -> None:
        self.tensordict = tensordict
        self.include_nested = include_nested
        self.leaves_only = leaves_only
        if is_leaf is None:
            is_leaf = _default_is_leaf
        self.is_leaf = is_leaf

    def __iter__(self) -> Iterable[str] | Iterable[tuple[str, ...]]:
        if not self.include_nested:
            if self.leaves_only:
                for key in self._keys():
                    target_class = self.tensordict.entry_class(key)
                    if _is_tensor_collection(target_class):
                        continue
                    yield key
            else:
                yield from self._keys()
        else:
            yield from (
                key if len(key) > 1 else key[0]
                for key in self._iter_helper(self.tensordict)
            )

    def _iter_helper(
        self, tensordict: T, prefix: str | None = None
    ) -> Iterable[str] | Iterable[tuple[str, ...]]:
        for key, value in self._items(tensordict):
            full_key = self._combine_keys(prefix, key)
            cls = value.__class__
            is_leaf = self.is_leaf(cls)
            if self.include_nested and not is_leaf:
                subkeys = tuple(self._iter_helper(value, prefix=full_key))
                yield from subkeys
            if not self.leaves_only or is_leaf:
                yield full_key

    def _combine_keys(self, prefix: tuple | None, key: str) -> tuple:
        if prefix is not None:
            return prefix + (key,)
        return (key,)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _items(
        self, tensordict: TensorDictBase | None = None
    ) -> Iterable[tuple[NestedKey, CompatibleType]]:
        if tensordict is None:
            tensordict = self.tensordict
        if isinstance(tensordict, TensorDict) or is_tensorclass(tensordict):
            return tensordict._tensordict.items()
        from .params import TensorDictParams

        if isinstance(tensordict, TensorDictParams):
            return tensordict._param_td.items()
        from ._lazy import _iter_items_lazystack, LazyStackedTensorDict

        if isinstance(tensordict, LazyStackedTensorDict):
            return _iter_items_lazystack(tensordict, return_none_for_het_values=True)
        raise NotImplementedError(type(tensordict))

    def _keys(self) -> _TensorDictKeysView:
        return self.tensordict._tensordict.keys()

    def __contains__(self, key: NestedKey) -> bool:
        key = _unravel_key_to_tuple(key)
        if not key:
            raise TypeError(_NON_STR_KEY_ERR)

        if isinstance(key, str):
            if key in self._keys():
                if self.leaves_only:
                    return not _is_tensor_collection(self.tensordict.entry_class(key))
                return True
            return False
        else:
            # thanks to _unravel_key_to_tuple we know the key is a tuple
            if len(key) == 1:
                return key[0] in self._keys()
            elif self.include_nested:
                if key[0] in self._keys():
                    entry_type = self.tensordict.entry_class(key[0])
                    if entry_type in (Tensor,):
                        return False
                    _is_tensordict = _is_tensor_collection(entry_type)
                    if _is_tensordict:
                        # # this will call _unravel_key_to_tuple many times
                        # return key[1:] in self.tensordict._get_str(key[0], NO_DEFAULT).keys(include_nested=self.include_nested)
                        # this won't call _unravel_key_to_tuple but requires to get the default which can be suboptimal
                        leaf_td = self.tensordict._get_tuple(key[:-1], None)
                        if leaf_td is None or (
                            not _is_tensor_collection(leaf_td.__class__)
                        ):
                            return False
                        return key[-1] in leaf_td.keys()
                return False
            # this is reached whenever there is more than one key but include_nested is False
            if all(isinstance(subkey, str) for subkey in key):
                raise TypeError(_NON_STR_KEY_TUPLE_ERR)

    def __repr__(self):
        include_nested = f"include_nested={self.include_nested}"
        leaves_only = f"leaves_only={self.leaves_only}"
        return f"{self.__class__.__name__}({list(self)},\n{indent(include_nested, 4 * ' ')},\n{indent(leaves_only, 4 * ' ')})"


def _set_tensor_dict(  # noqa: F811
    module_dict,
    hooks,
    module,
    name: str,
    tensor: torch.Tensor,
    inplace: bool,
) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    was_buffer = False
    out = module_dict["_parameters"].pop(name, None)  # type: ignore[assignment]
    if out is None:
        out = module_dict["_buffers"].pop(name, None)
        was_buffer = out is not None
    if out is None:
        out = module_dict.pop(name)
    if inplace:
        # swap tensor and out after updating out
        out_tmp = out.clone()
        out.data.copy_(tensor.data)
        tensor = out
        out = out_tmp

    if isinstance(tensor, torch.nn.Parameter):
        for hook in hooks:
            output = hook(module, name, tensor)
            if output is not None:
                tensor = output
        module_dict["_parameters"][name] = tensor
    elif was_buffer and isinstance(tensor, torch.Tensor):
        module_dict["_buffers"][name] = tensor
    else:
        module_dict[name] = tensor
    return out


class _subtd_meta_deprec(abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        warn(
            "SubTensorDict will become a private feature in v0.4. Please refrain from using it directly."
        )
        instance = _SubTensorDict(*args, **kwargs)
        return instance


def _index_to_str(index):
    if isinstance(index, tuple):
        return tuple(_index_to_str(elt) for elt in index)
    if isinstance(index, slice):
        return ("slice", {"start": index.start, "stop": index.stop, "step": index.step})
    if isinstance(index, range):
        return ("range", {"start": index.start, "stop": index.stop, "step": index.step})
    if isinstance(index, Tensor):
        return ("tensor", index.tolist(), str(index.device))
    return index


def _str_to_index(index):
    if isinstance(index, tuple):
        if not len(index):
            return index
        if index[0] == "slice":
            index = index[1]
            return slice(index["start"], index["stop"], index["step"])
        if index[0] == "range":
            index = index[1]
            return range(index["start"], index["stop"], index["step"])
        if index[0] == "tensor":
            index, device = index[1:]
            return torch.tensor(index, device=device)
        return tuple(_index_to_str(elt) for elt in index)
    return index


_register_tensor_class(TensorDict)
_register_tensor_class(_SubTensorDict)
