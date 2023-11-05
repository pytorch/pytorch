from __future__ import annotations

import functools
import inspect
import numbers
import re
from copy import copy
from functools import wraps
from typing import Any, Callable, Iterator, OrderedDict, Sequence

from functorch import dim as ftdim

from tensordict.utils import erase_cache, IndexType, NestedKey

import torch
from torch import multiprocessing as mp, nn, Tensor
from torch.utils._pytree import tree_map
from ._torch_func import TD_HANDLED_FUNCTIONS
from .base import _is_tensor_collection, CompatibleType, NO_DEFAULT, TensorDictBase
from .tensordict import _SubTensorDict, TensorDict
from .utils import _LOCK_ERROR, Buffer, lock_blocked


def _apply_leaves(data, fn):
    if isinstance(data, _SubTensorDict):
        raise RuntimeError(
            "Using a _SubTensorDict within a TensorDictParams isn't permitted."
        )
    elif isinstance(data, TensorDictBase):
        with data.unlock_():
            for key, val in list(data.items()):
                data._set_str(
                    key, _apply_leaves(val, fn), validated=True, inplace=False
                )
        return data
    else:
        return fn(data)


def _get_args_dict(func, args, kwargs):
    signature = inspect.signature(func)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    args_dict = dict(bound_arguments.arguments)
    return args_dict


def _maybe_make_param(tensor):
    if (
        isinstance(tensor, (Tensor, ftdim.Tensor))
        and not isinstance(tensor, nn.Parameter)
        and tensor.dtype in (torch.float, torch.double, torch.half)
    ):
        tensor = nn.Parameter(tensor)
    return tensor


def _maybe_make_param_or_buffer(tensor):
    if (
        isinstance(tensor, (Tensor, ftdim.Tensor))
        and not isinstance(tensor, nn.Parameter)
        and tensor.dtype in (torch.float, torch.double, torch.half)
    ):
        # convert all non-parameters to buffers
        tensor = Buffer(tensor)
    return tensor


class _unlock_and_set:
    # temporarily unlocks the nested tensordict to execute a function
    def __new__(cls, *args, **kwargs):
        if len(args) and callable(args[0]):
            return cls(**kwargs)(args[0])
        return super().__new__(cls)

    def __init__(self, **only_for_kwargs):
        self.only_for_kwargs = only_for_kwargs

    def __call__(self, func):
        name = func.__name__

        @wraps(func)
        def new_func(_self, *args, **kwargs):
            if self.only_for_kwargs:
                arg_dict = _get_args_dict(func, (_self, *args), kwargs)
                for kwarg, exp_value in self.only_for_kwargs.items():
                    cur_val = arg_dict.get(kwarg, NO_DEFAULT)
                    if cur_val != exp_value:
                        # escape
                        meth = getattr(_self._param_td, name)
                        out = meth(*args, **kwargs)
                        return out
            if not _self.no_convert:
                args = tree_map(_maybe_make_param, args)
                kwargs = tree_map(_maybe_make_param, kwargs)
            else:
                args = tree_map(_maybe_make_param_or_buffer, args)
                kwargs = tree_map(_maybe_make_param_or_buffer, kwargs)
            if _self.is_locked:
                # if the root (TensorDictParams) is locked, we still want to raise an exception
                raise RuntimeError(_LOCK_ERROR)
            with _self._param_td.unlock_():
                meth = getattr(_self._param_td, name)
                out = meth(*args, **kwargs)
            _self._reset_params()
            if out is _self._param_td:
                return _self
            return out

        return new_func


def _get_post_hook(func):
    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        return self._apply_get_post_hook(out)

    return new_func


def _fallback(func):
    """Calls the method on the nested tensordict."""
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            # if the output does not change, return the wrapper
            return self
        return out

    return new_func


def _fallback_property(func):
    name = func.__name__

    @wraps(func)
    def new_func(self):
        out = getattr(self._param_td, name)
        if out is self._param_td:
            return self
        return out

    def setter(self, value):
        return getattr(type(self._param_td), name).fset(self._param_td, value)

    return property(new_func, setter)


def _replace(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            return self
        self._param_td = out
        return self

    return new_func


def _carry_over(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):
        out = getattr(self._param_td, name)(*args, **kwargs)
        out = TensorDictParams(out, no_convert=True)
        out.no_convert = self.no_convert
        return out

    return new_func


class TensorDictParams(TensorDictBase, nn.Module):
    r"""Holds a TensorDictBase instance full of parameters.

    This class exposes the contained parameters to a parent nn.Module
    such that iterating over the parameters of the module also iterates over
    the leaves of the tensordict.

    Indexing works exactly as the indexing of the wrapped tensordict.
    The parameter names will be registered within this module using :meth:`~.TensorDict.flatten_keys("_")`.
    Therefore, the result of :meth:`~.named_parameters()` and the content of the
    tensordict will differ slightly in term of key names.

    Any operation that sets a tensor in the tensordict will be augmented by
    a :class:`torch.nn.Parameter` conversion.

    Args:
        parameters (TensorDictBase): a tensordict to represent as parameters.
            Values will be converted to parameters unless ``no_convert=True``.

    Keyword Args:
        no_convert (bool): if ``True``, no conversion to ``nn.Parameter`` will
            occur at construction and after (unless the ``no_convert`` attribute is changed).
            If ``no_convert`` is ``True`` and if non-parameters are present, they
            will be registered as buffers.
            Defaults to ``False``.

    Examples:
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
        >>> params = TensorDict.from_module(module)
        >>> params.lock_()
        >>> p = TensorDictParams(params)
        >>> print(p)
        TensorDictParams(params=TensorDict(
            fields={
                0: TensorDict(
                    fields={
                        bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight: Parameter(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                1: TensorDict(
                    fields={
                        bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight: Parameter(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False))
        >>> class CustomModule(nn.Module):
        ...     def __init__(self, params):
        ...         super().__init__()
        ...         self.params = params
        >>> m = CustomModule(p)
        >>> # the wrapper supports assignment and values are turned in Parameter
        >>> m.params['other'] = torch.randn(3)
        >>> assert isinstance(m.params['other'], nn.Parameter)

    """

    def __init__(self, parameters: TensorDictBase, *, no_convert=False):
        super().__init__()
        if isinstance(parameters, TensorDictParams):
            parameters = parameters._param_td
        self._param_td = parameters
        self.no_convert = no_convert
        if not no_convert:
            func = _maybe_make_param
        else:
            func = _maybe_make_param_or_buffer
        self._param_td = _apply_leaves(self._param_td, lambda x: func(x)).lock_()
        self._reset_params()
        self._is_locked = False
        self._locked_tensordicts = []
        self.__last_op_queue = None
        self._get_post_hook = []

    def register_get_post_hook(self, hook):
        """Register a hook to be called after any get operation on leaf tensors."""
        if not callable(hook):
            raise ValueError("Hooks must be callables.")
        self._get_post_hook.append(hook)

    def _apply_get_post_hook(self, val):
        if not _is_tensor_collection(type(val)):
            for hook in self._get_post_hook:
                new_val = hook(self, val)
                if new_val is not None:
                    val = new_val
        return val

    def _reset_params(self):
        parameters = self._param_td
        param_keys = []
        buffer_keys = []
        for key, value in parameters.items(True, True):
            if isinstance(value, nn.Parameter):
                param_keys.append(key)
            else:
                buffer_keys.append(key)
        self.__dict__["_parameters"] = (
            parameters.select(*param_keys).flatten_keys("_").to_dict()
        )
        self.__dict__["_buffers"] = (
            parameters.select(*buffer_keys).flatten_keys("_").to_dict()
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in TDPARAM_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, ftdim.Tensor, TensorDictBase)) for t in types
        ):
            return NotImplemented
        return TDPARAM_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def _flatten_key(cls, key):
        def make_valid_identifier(s):
            # Replace invalid characters with underscores
            s = re.sub(r"\W|^(?=\d)", "_", s)

            # Ensure the string starts with a letter or underscore
            if not s[0].isalpha() and s[0] != "_":
                s = "_" + s

            return s

        key_flat = "_".join(key)
        if not key_flat.isidentifier():
            key_flat = make_valid_identifier(key_flat)
        return key_flat

    @lock_blocked
    @_unlock_and_set
    def __setitem__(
        self,
        index: IndexType,
        value: TensorDictBase | dict | numbers.Number | CompatibleType,
    ) -> None:
        ...

    @lock_blocked
    @_unlock_and_set
    def set(
        self, key: NestedKey, item: CompatibleType, inplace: bool = False, **kwargs: Any
    ) -> TensorDictBase:
        ...

    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
    ) -> TensorDictBase:
        if not self.no_convert:
            func = _maybe_make_param
        else:
            func = _maybe_make_param_or_buffer
        if isinstance(input_dict_or_td, TensorDictBase):
            input_dict_or_td = input_dict_or_td.apply(func)
        else:
            input_dict_or_td = tree_map(func, input_dict_or_td)
        with self._param_td.unlock_():
            TensorDictBase.update(self, input_dict_or_td, clone=clone, inplace=inplace)
            self._reset_params()
        return self

    @lock_blocked
    @_unlock_and_set
    def pop(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        ...

    @lock_blocked
    @_unlock_and_set
    def rename_key_(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> TensorDictBase:
        ...

    @_unlock_and_set
    def apply_(self, fn: Callable, *others) -> TensorDictBase:
        ...

    def map(
        self,
        fn: Callable,
        dim: int = 0,
        num_workers: int = None,
        chunksize: int = None,
        num_chunks: int = None,
        pool: mp.Pool = None,
    ):
        raise RuntimeError(
            "Cannot call map on a TensorDictParams object. Convert it "
            "to a detached tensordict first (``tensordict.data``) and call "
            "map in a second time."
        )

    @_unlock_and_set(inplace=True)
    def apply(
        self,
        fn: Callable,
        *others: TensorDictBase,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = None,
        names: Sequence[str] | None = None,
        inplace: bool = False,
        **constructor_kwargs,
    ) -> TensorDictBase:
        ...

    @_unlock_and_set(inplace=True)
    def _apply_nest(*args, **kwargs):
        ...

    @_get_post_hook
    @_fallback
    def get(
        self, key: NestedKey, default: str | CompatibleType = NO_DEFAULT
    ) -> CompatibleType:
        ...

    @_get_post_hook
    @_fallback
    def __getitem__(self, index: IndexType) -> TensorDictBase:
        ...

    __getitems__ = __getitem__

    def to(self, *args, **kwargs) -> TensorDictBase:
        params = self._param_td.to(*args, **kwargs)
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def cpu(self):
        params = self._param_td.cpu()
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def cuda(self, device=None):
        params = self._param_td.cuda(device=device)
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def clone(self, recurse: bool = True) -> TensorDictBase:
        """Clones the TensorDictParams.

        The effect of this call is different from a regular torch.Tensor.clone call
        in that it will create a TensorDictParams instance with a new copy of the
        parameters and buffers __detached__ from the current graph.

        See :meth:`tensordict.TensorDictBase.clone` for more info on the clone
        method.

        """
        if not recurse:
            return TensorDictParams(self._param_td.clone(False), no_convert=True)

        def _clone(tensor):
            if isinstance(tensor, nn.Parameter):
                tensor = nn.Parameter(
                    tensor.data.clone(), requires_grad=tensor.requires_grad
                )
            else:
                tensor = Buffer(tensor.data.clone(), requires_grad=tensor.requires_grad)
            return tensor

        return TensorDictParams(self._param_td.apply(_clone), no_convert=True)

    @_fallback
    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        ...

    @_fallback
    def unbind(self, dim: int) -> tuple[TensorDictBase, ...]:
        ...

    @_fallback
    def to_tensordict(self):
        ...

    @_fallback
    def to_h5(
        self,
        filename,
        **kwargs,
    ):
        ...

    def __hash__(self):
        return hash((id(self), id(self._param_td)))

    @_fallback
    def __eq__(self, other: object) -> TensorDictBase:
        ...

    @_fallback
    def __ne__(self, other: object) -> TensorDictBase:
        ...

    def __getattr__(self, item: str) -> Any:
        try:
            return getattr(self._param_td, item)
        except AttributeError:
            return super().__getattr__(item)

    @_fallback
    def _change_batch_size(self, *args, **kwargs):
        ...

    @_fallback
    def _erase_names(self, *args, **kwargs):
        ...

    @_get_post_hook
    @_fallback
    def _get_str(self, *args, **kwargs):
        ...

    @_get_post_hook
    @_fallback
    def _get_tuple(self, *args, **kwargs):
        ...

    @_get_post_hook
    @_fallback
    def _get_at_str(self, key, idx, default):
        ...

    @_get_post_hook
    @_fallback
    def _get_at_tuple(self, key, idx, default):
        ...

    @_fallback
    def _add_batch_dim(self, *args, **kwargs):
        ...

    @_fallback
    def _convert_to_tensordict(self, *args, **kwargs):
        ...

    @_fallback
    def _get_names_idx(self, *args, **kwargs):
        ...

    @_fallback
    def _index_tensordict(self, *args, **kwargs):
        ...

    @_fallback
    def _remove_batch_dim(self, *args, **kwargs):
        ...

    @_fallback
    def _has_names(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _rename_subtds(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _set_at_str(self, *args, **kwargs):
        ...

    @_fallback
    def _set_at_tuple(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _set_str(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _set_tuple(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def _create_nested_str(self, *args, **kwargs):
        ...

    @_fallback
    def _stack_onto_(self, *args, **kwargs):
        ...

    @_fallback_property
    def batch_size(self) -> torch.Size:
        ...

    @_fallback
    def contiguous(self, *args, **kwargs):
        ...

    @lock_blocked
    @_unlock_and_set
    def del_(self, *args, **kwargs):
        ...

    @_fallback
    def detach_(self, *args, **kwargs):
        ...

    @_fallback_property
    def device(self):
        ...

    @_fallback
    def entry_class(self, *args, **kwargs):
        ...

    @_fallback
    def is_contiguous(self, *args, **kwargs):
        ...

    @_fallback
    def keys(self, *args, **kwargs):
        ...

    @_fallback
    def masked_fill(self, *args, **kwargs):
        ...

    @_fallback
    def masked_fill_(self, *args, **kwargs):
        ...

    def memmap_(
        self, prefix: str | None = None, copy_existing: bool = False
    ) -> TensorDictBase:
        raise RuntimeError("Cannot build a memmap TensorDict in-place.")

    @_fallback_property
    def names(self):
        ...

    @_fallback
    def pin_memory(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def select(self, *args, **kwargs):
        ...

    @_fallback
    def share_memory_(self, *args, **kwargs):
        ...

    @property
    def is_locked(self) -> bool:
        # Cannot be locked
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value):
        self._is_locked = bool(value)

    @_fallback_property
    def is_shared(self) -> bool:
        ...

    @_fallback_property
    def is_memmap(self) -> bool:
        ...

    @_fallback_property
    def shape(self) -> torch.Size:
        ...

    def _lock_propagate(self, lock_ids=None):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        is_root = lock_ids is None
        if is_root:
            lock_ids = set()
        self._lock_id = self._lock_id.union(lock_ids)
        lock_ids = lock_ids.union({id(self)})
        _locked_tensordicts = []
        # we don't want to double-lock the _param_td attrbute which is locked by default
        if not self._param_td.is_locked:
            for key, value in self._param_td.items():
                if _is_tensor_collection(type(value)):
                    value._lock_propagate(lock_ids)
                    _locked_tensordicts.append(value)
        if is_root:
            self._locked_tensordicts = _locked_tensordicts
        else:
            self._locked_tensordicts += _locked_tensordicts

    @erase_cache
    def _propagate_unlock(self, lock_ids=None):
        if lock_ids is not None:
            self._lock_id.difference_update(lock_ids)
        else:
            lock_ids = set()
        self._is_locked = False

        unlocked_tds = [self]
        lock_ids.add(id(self))
        self._locked_tensordicts = []

        self._is_shared = False
        self._is_memmap = False
        return unlocked_tds

    unlock_ = TensorDict.unlock_
    lock_ = TensorDict.lock_

    @property
    def data(self):
        return self.apply(lambda x: x.data)

    @_unlock_and_set(inplace=True)
    def flatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase:
        ...

    @_unlock_and_set(inplace=True)
    def unflatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase:
        ...

    @_unlock_and_set(inplace=True)
    def exclude(self, *keys: str, inplace: bool = False) -> TensorDictBase:
        ...

    @_fallback
    def transpose(self, dim0, dim1):
        ...

    @_fallback
    def where(self, condition, other, *, out=None, pad=None):
        ...

    @_fallback
    def permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> TensorDictBase:
        ...

    @_fallback
    def squeeze(self, dim: int | None = None) -> TensorDictBase:
        ...

    @_fallback
    def unsqueeze(self, dim: int) -> TensorDictBase:
        ...

    @_fallback
    def __xor__(self, other):
        ...

    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared

    @_fallback
    def all(self, dim: int = None) -> bool | TensorDictBase:
        ...

    @_fallback
    def any(self, dim: int = None) -> bool | TensorDictBase:
        ...

    @_fallback
    def expand(self, *args, **kwargs) -> T:
        ...

    @_fallback
    def masked_select(self, mask: Tensor) -> T:
        ...

    @_fallback
    def memmap_like(self, prefix: str | None = None) -> T:
        ...

    @_fallback
    def reshape(self, *shape: int):
        ...

    @_fallback
    def split(self, split_size: int | list[int], dim: int = 0) -> list[TensorDictBase]:
        ...

    @_fallback
    def to_module(self, module: nn.Module, return_swap: bool = False):
        ...

    @_fallback
    def view(self, *args, **kwargs):
        ...

    @_unlock_and_set
    def create_nested(self, key):
        ...

    def __repr__(self):
        return f"TensorDictParams(params={self._param_td})"

    def values(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[CompatibleType]:
        for v in self._param_td.values(include_nested, leaves_only):
            if _is_tensor_collection(type(v)):
                yield v
                continue
            yield self._apply_get_post_hook(v)

    def state_dict(
        self, *args, destination=None, prefix="", keep_vars=False, flatten=False
    ):
        return self._param_td.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
            flatten=flatten,
        )

    def load_state_dict(
        self, state_dict: OrderedDict[str, Any], strict=True, assign=False
    ):
        state_dict_tensors = {}
        state_dict = dict(state_dict)
        for k, v in list(state_dict.items()):
            if isinstance(v, torch.Tensor):
                del state_dict[k]
                state_dict_tensors[k] = v
        state_dict_tensors = dict(
            TensorDict(state_dict_tensors, []).unflatten_keys(".")
        )
        state_dict.update(state_dict_tensors)
        self.data.load_state_dict(state_dict, strict=True, assign=False)
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        data = (
            TensorDict(
                {
                    key: val
                    for key, val in state_dict.items()
                    if key.startswith(prefix) and val is not None
                },
                [],
            )
            .unflatten_keys(".")
            .get(prefix[:-1])
        )
        self.data.load_state_dict(data)

    def items(
        self, include_nested: bool = False, leaves_only: bool = False
    ) -> Iterator[CompatibleType]:
        for k, v in self._param_td.items(include_nested, leaves_only):
            if _is_tensor_collection(type(v)):
                yield k, v
                continue
            yield k, self._apply_get_post_hook(v)

    def _apply(self, fn, recurse=True):
        """Modifies torch.nn.Module._apply to work with Buffer class."""
        if recurse:
            for module in self.children():
                module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            should_use_set_data = compute_should_use_set_data(param, param_applied)
            if should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, nn.Parameter)
                assert param.is_leaf
                out_param = nn.Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                should_use_set_data = compute_should_use_set_data(
                    param.grad, grad_applied
                )
                if should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param.grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(
                        param.grad.requires_grad
                    )

        for key, buffer in self._buffers.items():
            if buffer is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `buffer_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                buffer_applied = fn(buffer)
            should_use_set_data = compute_should_use_set_data(buffer, buffer_applied)
            if should_use_set_data:
                buffer.data = buffer_applied
                out_buffer = buffer
            else:
                assert isinstance(buffer, Buffer)
                assert buffer.is_leaf
                out_buffer = Buffer(buffer_applied, buffer.requires_grad)
                self._buffers[key] = out_buffer

            if buffer.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(buffer.grad)
                should_use_set_data = compute_should_use_set_data(
                    buffer.grad, grad_applied
                )
                if should_use_set_data:
                    assert out_buffer.grad is not None
                    out_buffer.grad.data = grad_applied
                else:
                    assert buffer.grad.is_leaf
                    out_buffer.grad = grad_applied.requires_grad_(
                        buffer.grad.requires_grad
                    )

        return self


TDPARAM_HANDLED_FUNCTIONS = copy(TD_HANDLED_FUNCTIONS)


def implements_for_tdparam(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDictParams."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        TDPARAM_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for_tdparam(torch.empty_like)
def _empty_like(td: TensorDictBase, *args, **kwargs) -> TensorDictBase:
    try:
        tdclone = td.clone()
    except Exception as err:
        raise RuntimeError(
            "The tensordict passed to torch.empty_like cannot be "
            "cloned, preventing empty_like to be called. "
            "Consider calling tensordict.to_tensordict() first."
        ) from err
    return tdclone.data.apply_(lambda x: torch.empty_like(x, *args, **kwargs))
