from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import numbers
import os
import pickle
import re
import sys
import warnings
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import Any, Callable, List, Sequence, TypeVar

import torch

import torch.dict as tensordict_lib
from torch import Tensor

from torch.dict._torch_func import TD_HANDLED_FUNCTIONS
from torch.dict.base import (
    _register_tensor_class,
    _unravel_key_to_tuple,
    is_tensor_collection,
    NO_DEFAULT,
    TensorDictBase,
)
from torch.dict.tensordict import TensorDict

from torch.dict.utils import (
    _get_repr,
    _is_json_serializable,
    _LOCK_ERROR,
    DeviceType,
    IndexType,
    is_tensorclass,
    NestedKey,
)

T = TypeVar("T", bound=TensorDictBase)
PY37 = sys.version_info < (3, 8)

# Regex precompiled patterns
OPTIONAL_PATTERN = re.compile(r"Optional\[(.*?)\]")
UNION_PATTERN = re.compile(r"Union\[(.*?)\]")

# methods where non_tensordict data should be cleared in the return value
_CLEAR_METADATA = {"all", "any"}
# torch functions where we can wrap the corresponding TensorDict version
_TD_PASS_THROUGH = {
    torch.unbind,
    torch.full_like,
    torch.zeros_like,
    torch.ones_like,
    torch.clone,
    torch.squeeze,
    torch.unsqueeze,
    torch.split,
    torch.permute,
    torch.split,
    torch.stack,
    torch.cat,
    torch.gather,
}


def tensorclass(cls: T) -> T:
    """A decorator to create :obj:`tensorclass` classes.

    :obj:`tensorclass` classes are specialized :obj:`dataclass` instances that
    can execute some pre-defined tensor operations out of the box, such as
    indexing, item assignment, reshaping, casting to device or storage and many
    others.

    Examples:
        >>> from torch.dict import tensorclass
        >>> import torch
        >>> from typing import Optional
        >>>
        >>> @tensorclass
        ... class MyData:
        ...     X: torch.Tensor
        ...     y: torch.Tensor
        ...     z: str
        ...     def expand_and_mask(self):
        ...         X = self.X.unsqueeze(-1).expand_as(self.y)
        ...         X = X[self.y]
        ...         return X
        ...
        >>> data = MyData(
        ...     X=torch.ones(3, 4, 1),
        ...     y=torch.zeros(3, 4, 2, 2, dtype=torch.bool),
        ...     z="test"
        ...     batch_size=[3, 4])
        >>> print(data)
        MyData(
            X=Tensor(torch.Size([3, 4, 1]), dtype=torch.float32),
            y=Tensor(torch.Size([3, 4, 2, 2]), dtype=torch.bool),
            z="test"
            batch_size=[3, 4],
            device=None,
            is_shared=False)
        >>> print(data.expand_and_mask())
        tensor([])

    It is also possible to nest tensorclasses instances within each other:
        Examples:
        >>> from torch.dict import tensorclass
        >>> import torch
        >>> from typing import Optional
        >>>
        >>> @tensorclass
        ... class NestingMyData:
        ...     nested: MyData
        ...
        >>> nesting_data = NestingMyData(nested=data, batch_size=[3, 4])
        >>> # although the data is stored as a TensorDict, the type hint helps us
        >>> # to appropriately cast the data to the right type
        >>> assert isinstance(nesting_data.nested, type(data))


    """

    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if func not in _TD_PASS_THROUGH or not all(
            issubclass(t, (Tensor, cls)) for t in types
        ):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        # get the output type from the arguments / keyword arguments
        if len(args) > 0:
            tensorclass_instance = args[0]
        else:
            tensorclass_instance = kwargs.get("input", kwargs["tensors"])
        if isinstance(tensorclass_instance, (tuple, list)):
            tensorclass_instance = tensorclass_instance[0]
        args = tuple(_arg_to_tensordict(arg) for arg in args)
        kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}

        result = TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return result.__class__(
                _from_tensordict_with_copy(tensorclass_instance, tensordict_result)
                for tensordict_result in result
            )
        return _from_tensordict_with_copy(tensorclass_instance, result)

    cls = dataclass(cls)
    expected_keys = set(cls.__dataclass_fields__)

    for attr in cls.__dataclass_fields__:
        if attr in dir(TensorDict):
            raise AttributeError(
                f"Attribute name {attr} can't be used with @tensorclass"
            )

    cls.__init__ = _init_wrapper(cls.__init__)
    cls._from_tensordict = classmethod(_from_tensordict_wrapper(expected_keys))
    cls.from_tensordict = cls._from_tensordict
    if not hasattr(cls, "__torch_function__"):
        cls.__torch_function__ = classmethod(__torch_function__)
    cls.__getstate__ = _getstate
    cls.__setstate__ = _setstate
    cls.__getattribute__ = _getattribute_wrapper(cls.__getattribute__)
    cls.__setattr__ = _setattr_wrapper(cls.__setattr__, expected_keys)
    cls.__getattr__ = _getattr
    cls.__getitem__ = _getitem
    cls.__getitems__ = _getitem
    cls.__setitem__ = _setitem
    cls.__repr__ = _repr
    cls.__len__ = _len
    cls.__eq__ = _eq
    cls.__ne__ = _ne
    cls.__or__ = _or
    cls.__xor__ = _xor
    cls.set = _set
    cls.set_at_ = _set_at_
    cls.del_ = _del_
    cls.get = _get
    cls.get_at = _get_at
    cls.unbind = _unbind
    cls.state_dict = _state_dict
    cls.load_state_dict = _load_state_dict
    cls._memmap_ = _memmap_

    # Memmap
    cls.memmap_like = TensorDictBase.memmap_like
    cls.memmap_ = TensorDictBase.memmap_
    cls.memmap = TensorDictBase.memmap
    cls.load_memmap = TensorDictBase.load_memmap
    cls._load_memmap = classmethod(_load_memmap)

    for attr in TensorDict.__dict__.keys():
        func = getattr(TensorDict, attr)
        if inspect.ismethod(func) and attr not in cls.__dict__:
            tdcls = func.__self__
            if issubclass(tdcls, TensorDictBase):  # detects classmethods
                setattr(cls, attr, _wrap_classmethod(tdcls, cls, func))

    cls.to_tensordict = _to_tensordict
    cls.device = property(_device, _device_setter)
    cls.batch_size = property(_batch_size, _batch_size_setter)

    cls.__doc__ = f"{cls.__name__}{inspect.signature(cls)}"

    _register_tensor_class(cls)
    return cls


def _arg_to_tensordict(arg):
    # if arg is a tensorclass or sequence of tensorclasses, extract the underlying
    # tensordicts and return those instead
    if is_tensorclass(arg):
        return arg._tensordict
    elif isinstance(arg, (tuple, list)) and all(is_tensorclass(item) for item in arg):
        return arg.__class__(item._tensordict for item in arg)
    return arg


def _from_tensordict_with_copy(tc, tensordict):
    # creates a new tensorclass with the same type as tc, and a copy of the
    # non_tensordict data
    return tc._from_tensordict(
        tensordict=tensordict, non_tensordict=copy(tc._non_tensordict)
    )


def _from_tensordict_with_none(tc, tensordict):
    # creates a new tensorclass with the same type as tc, and all non_tensordict entries
    # set to None
    return tc._from_tensordict(
        tensordict=tensordict,
        non_tensordict={key: None for key in tc._non_tensordict},
    )


def _init_wrapper(init: Callable) -> Callable:
    init_sig = inspect.signature(init)
    params = list(init_sig.parameters.values())
    # drop first entry of params which corresponds to self and isn't passed by the user
    required_params = [p.name for p in params[1:] if p.default is inspect._empty]

    @functools.wraps(init)
    def wrapper(
        self,
        *args: Any,
        batch_size: Sequence[int] | torch.Size | int,
        device: DeviceType | None = None,
        names: List[str] | None = None,
        **kwargs,
    ):
        for value, key in zip(args, self.__dataclass_fields__):
            if key in kwargs:
                raise ValueError(f"The key {key} is already set in kwargs")
            kwargs[key] = value

        for key, field in self.__dataclass_fields__.items():
            if field.default_factory is not dataclasses.MISSING:
                default = field.default_factory()
            else:
                default = field.default
            if default not in (None, dataclasses.MISSING):
                kwargs.setdefault(key, default)

        missing_params = [p for p in required_params if p not in kwargs]
        if missing_params:
            n_missing = len(missing_params)
            raise TypeError(
                f"{self.__class__.__name__}.__init__() missing {n_missing} "
                f"required positional argument{'' if n_missing == 1 else 's'}: "
                f"""{", ".join(f"'{name}'" for name in missing_params)}"""
            )

        self._tensordict = TensorDict(
            {},
            batch_size=torch.Size(batch_size),
            device=device,
            names=names,
            _run_checks=False,
        )
        # To save non tensor data (Nested tensor classes also go here)
        self._non_tensordict = {}
        init(self, **kwargs)

    new_params = [
        inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("names", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
    wrapper.__signature__ = init_sig.replace(parameters=params + new_params)

    return wrapper


def _from_tensordict_wrapper(expected_keys):
    def wrapper(cls, tensordict, non_tensordict=None):  # noqa: D417
        """Tensor class wrapper to instantiate a new tensor class object.

        Args:
            tensordict (TensorDict): Dictionary of tensor types
            non_tensordict (dict): Dictionary with non-tensor and nested tensor class objects

        """
        if not isinstance(tensordict, TensorDictBase):
            raise RuntimeError(
                f"Expected a TensorDictBase instance but got {type(tensordict)}"
            )
        # Validating keys of tensordict
        for key in tensordict.keys():
            if key not in expected_keys:
                raise ValueError(
                    f"Keys from the tensordict ({set(tensordict.keys())}) must "
                    f"correspond to the class attributes ({expected_keys})."
                )

        # Validating non-tensor keys and for key clash
        tensor_keys = set(tensordict.keys())
        if non_tensordict is not None:
            for key in non_tensordict.keys():
                if key not in expected_keys:
                    raise ValueError(
                        f"Keys from the non-tensor data ({set(non_tensordict.keys())}) must "
                        f"correspond to the class attributes ({expected_keys})."
                    )
                if key in tensor_keys:
                    raise KeyError(
                        f"{key} is present in both tensor and non-tensor dicts"
                    )
        # bypass initialisation. this means we don't incur any overhead creating an
        # empty tensordict and writing values to it. we can skip this because we already
        # have a tensordict to use as the underlying tensordict
        tc = cls.__new__(cls)
        tc.__dict__["_tensordict"] = tensordict

        tc.__dict__["_non_tensordict"] = (
            non_tensordict if non_tensordict is not None else {}
        )
        # since we aren't calling the dataclass init method, we need to manually check
        # whether a __post_init__ method has been defined and invoke it if so
        if hasattr(tc, "__post_init__"):
            tc.__post_init__()
        return tc

    return wrapper


def _memmap_(
    self,
    prefix: str | None = None,
    copy_existing: bool = False,
    executor=None,
    futures=None,
    inplace=True,
    like=False,
):
    _non_tensordict = self._non_tensordict
    cls = self.__class__

    if prefix is not None:
        prefix = Path(prefix)
        if not prefix.exists():
            os.makedirs(prefix, exist_ok=True)

        def save_metadata(cls=cls, _non_tensordict=_non_tensordict, prefix=prefix):
            with open(prefix / "meta.json", "w") as f:
                metadata = {"_type": str(cls)}
                to_pickle = {}
                for key, value in _non_tensordict.items():
                    if _is_json_serializable(value):
                        metadata[key] = value
                    else:
                        to_pickle[key] = value
                json.dump(metadata, f)
                if to_pickle:
                    with open(prefix / "other.pickle", "wb") as pickle_file:
                        pickle.dump(to_pickle, pickle_file)

        if executor is None:
            save_metadata()
        else:
            futures.append(executor.submit(save_metadata))

        prefix = prefix / "_tensordict"

    td = self._tensordict._memmap_(
        prefix=prefix,
        executor=executor,
        futures=futures,
        inplace=inplace,
        like=like,
        copy_existing=copy_existing,
    )

    if not inplace:
        result = cls._from_tensordict(td, _non_tensordict)
    else:
        result = self
    return result


def _load_memmap(cls, prefix: Path, metadata: dict):
    non_tensordict = copy(metadata)
    del non_tensordict["_type"]
    if os.path.exists(prefix / "other.pickle"):
        with open(prefix / "other.pickle", "rb") as pickle_file:
            non_tensordict.update(pickle.load(pickle_file))
    td = TensorDict.load_memmap(prefix / "_tensordict")
    return cls._from_tensordict(td, non_tensordict)


def _getstate(self) -> dict[str, Any]:
    """Returns a state dict which consists of tensor and non_tensor dicts for serialization.

    Returns:
        dictionary of state of tensor class

    """
    return {"tensordict": self._tensordict, "non_tensordict": self._non_tensordict}


def _setstate(self, state: dict[str, Any]) -> None:  # noqa: D417
    """Used to set the state of an object using state parameter.

    Args:
        state (dict): State parameter to set the object
    """
    self._tensordict = state.get("tensordict", None)
    self._non_tensordict = state.get("non_tensordict", None)


def _getattribute_wrapper(getattribute: Callable) -> Callable:
    """Retrieve the value of an object's attribute or raise AttributeError.

    Args:
        item (str) : name of the attribute to retrieve

    Returns:
        value of the attribute

    """

    @functools.wraps(getattribute)
    def wrapper(self, item: str) -> Any:
        if not item.startswith("__"):
            if (
                "_tensordict" in self.__dict__
                and item in self.__dict__["_tensordict"].keys()
            ):
                out = self._tensordict.get(item)
                return out
            elif (
                "_non_tensordict" in self.__dict__
                and item in self.__dict__["_non_tensordict"]
            ):
                out = self._non_tensordict[item]
                return out
        return getattribute(self, item)

    return wrapper


SET_ATTRIBUTES = ("batch_size", "device", "_locked_tensordicts")


def _setattr_wrapper(setattr_: Callable, expected_keys: set[str]) -> Callable:
    @functools.wraps(setattr_)
    def wrapper(self, key: str, value: Any) -> None:  # noqa: D417
        """Set the value of an attribute for the tensor class object.

        Args:
            key (str): the name of the attribute to set
            value (any): the value to set for the attribute

        """
        __dict__ = self.__dict__
        if (
            "_tensordict" not in __dict__
            or "_non_tensordict" not in __dict__
            or key in SET_ATTRIBUTES
        ):
            return setattr_(self, key, value)

        out = self.set(key, value)
        if out is not self:
            raise RuntimeError(
                "Cannot set attribute on a locked tensorclass, even if "
                "clone_on_set is set to True. Use my_obj.set(...) instead."
            )

    return wrapper


def _wrap_method(self, attr, func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        args = tuple(_arg_to_tensordict(arg) for arg in args)
        kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}
        res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            if attr.endswith("_"):
                # in-place operation, return the current object
                return self
            elif attr in _CLEAR_METADATA:
                # this is an attribute where copying the metadata makes no sense, e.g.
                # .all or .any, so we replace all values with None
                return self._from_tensordict(
                    res, {k: None for k in self._non_tensordict}
                )
            # create a new tensorclass from res and copy the metadata from self
            return self._from_tensordict(res, copy(self._non_tensordict))
        return res

    return wrapped_func


def _wrap_classmethod(td_cls, cls, func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        res = func.__get__(td_cls)(*args, **kwargs)
        # res = func(*args, **kwargs)
        if isinstance(res, TensorDictBase):
            # create a new tensorclass from res and copy the metadata from self
            return cls._from_tensordict(res)
        return res

    return wrapped_func


def _getattr(self, attr: str) -> Any:
    """Retrieve the value of an object's attribute, or a method output if attr is callable.

    Args:
        attr: name of the attribute to retrieve or function to compute

    Returns:
        value of the attribute, or a method output applied on the instance

    """
    res = getattr(self._tensordict, attr)
    if not callable(res):
        return res
    func = res
    return _wrap_method(self, attr, func)


def _getitem(self, item: NestedKey) -> Any:
    """Retrieve the class object at the given index. Indexing will happen for nested tensors as well.

    Args:
       item (int or any other valid index type): index of the object to retrieve

    Returns:
        Tensor class object at the given index

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError(f"Invalid indexing arguments: {item}.")
    tensor_res = self._tensordict[item]
    return _from_tensordict_with_copy(self, tensor_res)  # device=res.device)


def _setitem(self, item: NestedKey, value: Any) -> None:  # noqa: D417
    """Set the value of the Tensor class object at the given index. Note that there is no strict validation on non-tensor values.

    Args:
        item (int or any other valid index type): index of the object to set
        value (any): value to set for the item

    """
    if isinstance(item, str) or (
        isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
    ):
        raise ValueError(f"Invalid indexing arguments: {item}.")

    if not is_tensorclass(value) and not isinstance(
        value, (TensorDictBase, numbers.Number, Tensor)
    ):
        raise ValueError(
            f"__setitem__ only supports tensorclasses, tensordicts,"
            f" numeric scalars and tensors. Got {type(value)}"
        )

    if is_tensorclass(value):
        if not isinstance(value, self.__class__):
            self_keys = set().union(self._non_tensordict, self._tensordict.keys())
            value_keys = set().union(value._non_tensordict, value._tensordict.keys())
            if self_keys != value_keys:
                # if tensorclass but different class ensure that all keys are equal
                raise ValueError(
                    "__setitem__ is only allowed for same-class or "
                    "compatible class (i.e. same members) assignment"
                )

        # Validating the non-tensor data before setting the item
        for key, val in value._non_tensordict.items():
            # Raise a warning if non_tensor data doesn't match
            if (
                key in self._non_tensordict.keys()
                and val is not self._non_tensordict[key]
            ):
                warnings.warn(
                    f"Meta data at {repr(key)} may or may not be equal, "
                    f"this may result in undefined behaviours",
                    category=UserWarning,
                    stacklevel=2,
                )

        for key in value._tensordict.keys():
            # Making sure that the key-clashes won't happen, if the key is present
            # in tensor data in value we will honor that and remove the key-value
            # pair from non-tensor data
            if key in self._non_tensordict.keys():
                del self._non_tensordict[key]

        self._tensordict[item] = value._tensordict
    else:  # it is one of accepted "broadcast" types
        # attempt broadcast on all tensordata and nested tensorclasses
        self._tensordict[item] = value
        for key, val in self._non_tensordict.items():
            if is_tensorclass(val):
                _setitem(self._non_tensordict[key], item, value)


def _repr(self) -> str:
    """Return a string representation of Tensor class object."""
    fields = _all_td_fields_as_str(self._tensordict)
    field_str = [fields] if fields else []
    non_tensor_fields = _all_non_td_fields_as_str(self._non_tensordict)
    batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
    device_str = indent(f"device={self.device}", 4 * " ")
    is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
    if len(non_tensor_fields) > 0:
        non_tensor_field_str = indent(
            ",\n".join(non_tensor_fields),
            4 * " ",
        )
        string = ",\n".join(
            field_str
            + [non_tensor_field_str, batch_size_str, device_str, is_shared_str]
        )
    else:
        string = ",\n".join(field_str + [batch_size_str, device_str, is_shared_str])
    return f"{self.__class__.__name__}(\n{string})"


def _len(self) -> int:
    """Returns the length of first dimension, if there is, otherwise 0."""
    return len(self._tensordict)


def _to_tensordict(self) -> TensorDict:
    """Convert the tensorclass into a regular TensorDict.

    Makes a copy of all entries. Memmap and shared memory tensors are converted to
    regular tensors.

    Returns:
        A new TensorDict object containing the same values as the tensorclass.

    """
    td = self._tensordict.to_tensordict()
    return td


def _device(self) -> torch.device:
    """Retrieves the device type of tensor class."""
    return self._tensordict.device


def _device_setter(self, value: DeviceType) -> None:
    raise RuntimeError(
        "device cannot be set using tensorclass.device = device, "
        "because device cannot be updated in-place. To update device, use "
        "tensorclass.to(new_device), which will return a new tensorclass "
        "on the new device."
    )


def _set(self, key: NestedKey, value: Any, inplace: bool = False):
    """Sets a new key-value pair.

    Args:
        key (str, tuple of str): name of the key to be set.
           If tuple of str it is equivalent to chained calls of getattr
           followed by a final setattr.
        value (Any): value to be stored in the tensorclass
        inplace (bool, optional): if ``True``, set will tentatively try to
            update the value in-place. If ``False`` or if the key isn't present,
            the value will be simply written at its destination.

    Returns:
        self

    """
    if isinstance(key, str):
        __dict__ = self.__dict__
        if __dict__["_tensordict"].is_locked:
            raise RuntimeError(_LOCK_ERROR)
        expected_keys = self.__dataclass_fields__
        if key not in expected_keys:
            raise AttributeError(
                f"Cannot set the attribute '{key}', expected attributes are {expected_keys}."
            )

        if isinstance(value, tuple(tensordict_lib.base._ACCEPTED_CLASSES)):
            # Avoiding key clash, honoring the user input to assign tensor type data to the key
            if key in self._non_tensordict.keys():
                if inplace:
                    raise RuntimeError(
                        f"Cannot update an existing entry of type {type(self._non_tensordict.get(key))} with a value of type {type(value)}."
                    )
                del self._non_tensordict[key]
            self._tensordict.set(key, value, inplace=inplace)
        else:
            # Avoiding key clash, honoring the user input to assign non-tensor data to the key
            if key in self._tensordict.keys():
                if inplace:
                    raise RuntimeError(
                        f"Cannot update an existing entry of type {type(self._tensordict.get(key))} with a value of type {type(value)}."
                    )
                self._tensordict.del_(key)
            # Saving all non-tensor attributes
            self._non_tensordict[key] = value
        return self

    if isinstance(key, tuple) and len(key):
        key = _unravel_key_to_tuple(key)
        if len(key) > 1:
            return self.set(key[0], getattr(self, key[0]).set(key[1:], value))
        out = self.set(key[0], value)
        return out
    raise ValueError(
        f"Supported type for key are str and tuple, got {key} of type {type(key)}"
    )


def _del_(self, key):
    key = _unravel_key_to_tuple(key)
    if len(key) > 1:
        td = self.get(key[0])
        td.del_(key[1:])
        return
    if key[0] in self._tensordict.keys():
        self._tensordict.del_(key[0])
        # self.set(key[0], None)
    elif key[0] in self._non_tensordict.keys():
        self._non_tensordict[key[0]] = None
    else:
        raise KeyError(f"Key {key} could not be found in tensorclass {self}.")
    return


def _set_at_(self, key: NestedKey, value: Any, idx: IndexType):
    if key in self._non_tensordict:
        del self._non_tensordict[key]
    return self._tensordict.set_at_(key, value, idx)


def _get(self, key: NestedKey, default: Any = NO_DEFAULT):
    """Gets the value stored with the input key.

    Args:
        key (str, tuple of str): key to be queried. If tuple of str it is
            equivalent to chained calls of getattr.
        default: default value if the key is not found in the tensorclass.

    Returns:
        value stored with the input key

    """
    if isinstance(key, str):
        key = (key,)

    if isinstance(key, tuple):
        try:
            if len(key) > 1:
                return getattr(self, key[0]).get(key[1:])
            return getattr(self, key[0])
        except AttributeError:
            if default is NO_DEFAULT:
                raise
            return default
    raise ValueError(f"Supported type for key are str and tuple, got {type(key)}")


def _get_at(self, key: NestedKey, idx, default: Any = NO_DEFAULT):
    try:
        return self.get(key, NO_DEFAULT)[idx]
    except AttributeError:
        if default is NO_DEFAULT:
            raise
        return default


def _batch_size(self) -> torch.Size:
    """Retrieves the batch size for the tensor class.

    Returns:
        batch size (torch.Size)

    """
    return self._tensordict.batch_size


def _batch_size_setter(self, new_size: torch.Size) -> None:  # noqa: D417
    """Set the value of batch_size.

    Args:
        new_size (torch.Size): new_batch size to be set

    """
    self._tensordict._batch_size_setter(new_size)


def _state_dict(
    self, destination=None, prefix="", keep_vars=False, flatten=False
) -> dict[str, Any]:
    """Returns a state_dict dictionary that can be used to save and load data from a tensorclass."""
    state_dict = {
        "_tensordict": self._tensordict.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, flatten=flatten
        )
    }
    state_dict["_non_tensordict"] = copy(self._non_tensordict)
    return state_dict


def _load_state_dict(
    self, state_dict: dict[str, Any], strict=True, assign=False, from_flatten=False
):
    """Loads a state_dict attemptedly in-place on the destination tensorclass."""
    for key, item in state_dict.items():
        # keys will never be nested which facilitates everything, but let's
        # double check in case someone does something nasty
        if not isinstance(key, str):
            raise TypeError("Only str keys are allowed when calling load_state_dict.")
        if key == "_non_tensordict":
            for sub_key, sub_item in item.items():
                # sub_item is the state dict of a tensorclass
                if isinstance(sub_item, dict) and "_non_tensordict" in sub_item:
                    raise RuntimeError(
                        "Loading a saved tensorclass on a uninitialized tensorclass is not allowed"
                    )
                else:
                    # check that sub_key is part of the tensorclass
                    if sub_key not in self.__class__.__dataclass_fields__:
                        raise KeyError(
                            f"Key '{sub_key}' wasn't expected in the state-dict."
                        )
                    self._non_tensordict[sub_key] = sub_item
        elif key == "_tensordict":
            for sub_key in item.keys():
                if (
                    sub_key not in self.__class__.__dataclass_fields__
                    and sub_key not in ("__batch_size", "__device")
                ):
                    raise KeyError(
                        f"Key '{sub_key}' wasn't expected in the state-dict."
                    )

            self._tensordict.load_state_dict(
                item, strict=strict, assign=assign, from_flatten=from_flatten
            )
        else:
            raise KeyError(f"Key '{key}' wasn't expected in the state-dict.")

    return self


def _eq(self, other: object) -> bool:
    """Compares the Tensor class object to another object for equality. However, the equality check for non-tensor data is not performed.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        False if the objects are of different class types, Tensorclass of boolean
        values for tensor attributes and None for non-tensor attributes

    Examples:
        >>> @tensorclass
        ... class MyClass:
        ...     x: Tensor
        ...     y: "MyClass"
        ...     z: str
        ...
        >>> c1 = MyClass(
        ...     x=torch.randn(3, 4),
        ...     y=MyClass(
        ...         x=torch.randn(3, 4, 1),
        ...         y=None,
        ...         z="bar",
        ...         batch_size=[3, 4, 1],
        ...     ),
        ...     z="foo",
        ...     batch_size=[3, 4],
        ... )
        >>> c2 = c1.clone()
        >>> print(c1 == c2)
        MyClass(
            x=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
            y=MyClass(
                x=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                y=None,
                z=None,
                batch_size=torch.Size([3, 4, 1]),
                device=None,
                is_shared=False),
            z=None,
            batch_size=torch.Size([3, 4]),
            device=None,
            is_shared=False)
        >>> assert (c1 == c2).all()
        >>> assert (c1[:2] == c2[:2]).all()
        >>> assert not (c1 == c2.apply(lambda x: x+1)).all()

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor, _MemmapTensor)
    ):
        return False
    if is_tensorclass(other):
        tensor = self._tensordict == other._tensordict
    else:
        tensor = self._tensordict == other
    return _from_tensordict_with_none(self, tensor)


def _ne(self, other: object) -> bool:
    """Compare the Tensor class object to another object for inequality. However, the equality check for non-tensor data is not performed.

    Args:
        other: object to compare to this object

    Returns:
        False if the objects are of different class types, Tensorclass of boolean values for tensor attributes and None for non-tensor attributes

    Examples:
        >>> @tensorclass
        ... class MyClass:
        ...     x: Tensor
        ...     y: "MyClass"
        ...     z: str
        ...
        >>> c1 = MyClass(
        ...     x=torch.randn(3, 4),
        ...     y=MyClass(
        ...         x=torch.randn(3, 4, 1),
        ...         y=None,
        ...         z="bar",
        ...         batch_size=[3, 4, 1],
        ...     ),
        ...     z="foo",
        ...     batch_size=[3, 4],
        ... )
        >>> c2 = c1.clone()
        >>> print(c1 != c2)
        MyClass(
            x=Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
            y=MyClass(
                x=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                y=None,
                z=None,
                batch_size=torch.Size([3, 4, 1]),
                device=None,
                is_shared=False),
            z=None,
            batch_size=torch.Size([3, 4]),
            device=None,
            is_shared=False)
        >>> c2 = c2.apply(lambda x: x+1)
        >>> assert (c1 != c2).all()

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor, _MemmapTensor)
    ):
        return True
    if is_tensorclass(other):
        tensor = self._tensordict != other._tensordict
    else:
        tensor = self._tensordict != other
    return _from_tensordict_with_none(self, tensor)


def _or(self, other: object) -> bool:
    """Compares the Tensor class object to another object for logical OR. However, the logical OR check for non-tensor data is not performed.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        False if the objects are of different class types, Tensorclass of boolean
        values for tensor attributes and None for non-tensor attributes

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor, _MemmapTensor)
    ):
        return False
    if is_tensorclass(other):
        tensor = self._tensordict | other._tensordict
    else:
        tensor = self._tensordict | other
    return _from_tensordict_with_none(self, tensor)


def _xor(self, other: object) -> bool:
    """Compares the Tensor class object to another object for exclusive OR. However, the exclusive OR check for non-tensor data is not performed.

    Args:
        other: object to compare to this object. Can be a tensorclass, a
            tensordict or any compatible type (int, float or tensor), in
            which case the equality check will be propagated to the leaves.

    Returns:
        False if the objects are of different class types, Tensorclass of boolean
        values for tensor attributes and None for non-tensor attributes

    """
    if not is_tensor_collection(other) and not isinstance(
        other, (dict, numbers.Number, Tensor, _MemmapTensor)
    ):
        return False
    if is_tensorclass(other):
        tensor = self._tensordict ^ other._tensordict
    else:
        tensor = self._tensordict ^ other
    return _from_tensordict_with_none(self, tensor)


def _single_td_field_as_str(key, item, tensordict):
    """Returns a string as a  key-value pair of tensordict.

    Args:
        key (str): key of tensor dict item
        item (tensor type): value to be returned for key
        tensordict (Tensordict): Tensordict object

    Returns:
        String representation of a key-value pair

    """
    if is_tensor_collection(type(item)):
        return f"{key}={repr(tensordict[key])}"
    return f"{key}={_get_repr(item)}"


def _all_td_fields_as_str(td: TensorDictBase) -> str:
    """Returns indented representation of tensor dict values as a key-value pairs.

    Args:
        td (TensorDict) : Tensordict object

    Returns:
        String representation of all tensor data

    """
    return indent(
        ",\n".join(
            sorted([_single_td_field_as_str(key, item, td) for key, item in td.items()])
        ),
        4 * " ",
    )


def _all_non_td_fields_as_str(src_dict) -> list:
    """Returns a list of string representation of non-tensor key-value pairs.

    Args:
        src_dict (dict): non_tensor_dict

    Returns:
        result (list): list of strings with key-value representation

    """
    result = []
    for key, val in src_dict.items():
        if not is_tensor_collection(val):
            result.append(f"{key}={repr(val)}")

    return result


def _unbind(self, dim: int):
    """Returns a tuple of indexed tensorclass instances unbound along the indicated dimension.

    Resulting tensorclass instances will share the storage of the initial tensorclass instance.

    """
    return tuple(
        self._from_tensordict(td, non_tensordict=copy(self._non_tensordict))
        for td in self._tensordict.unbind(dim)
    )


################
# Custom classes
# --------------

NONTENSOR_HANDLED_FUNCTIONS = []


@tensorclass
class NonTensorData:
    """A carrier for non-tensordict data.

    This class can be used whenever non-tensor data needs to be carrier at
    any level of a tensordict instance.

    :class:`~tensordict.tensorclass.NonTensorData` instances can be created
    explicitely or using :meth:`~tensordict.TensorDictBase.set_non_tensor`.

    This class is serializable using :meth:`tensordict.TensorDictBase.memmap`
    and related methods, and can be loaded through :meth:`~tensordict.TensorDictBase.load_memmap`.
    If the content of the object is JSON-serializable, it will be serializsed in
    the `meta.json` file in the directory pointed by the parent key of the `NoneTensorData`
    object. If it isn't, serialization will fall back on pickle. This implies
    that we assume that the content of this class is either json-serializable or
    pickable, and it is the user responsibility to make sure that one of these
    holds. We try to avoid pickling/unpickling objects for performance and security
    reasons (as pickle can execute arbitrary code during loading).

    .. note:: if the data passed to :class:`NonTensorData` is a :class:`NonTensorData`
        itself, the data from the nested object will be gathered.

        >>> non_tensor = NonTensorData("a string!")
        >>> non_tensor = NonTensorData(non_tensor)
        >>> assert non_tensor.data == "a string!"

    .. note:: Unlike other tensorclass classes, :class:`NonTensorData` supports
        comparisons of two non-tensor data through :meth:`~.__eq__`, :meth:`~.__ne__`,
        :meth:`~.__xor__` or :meth:`~.__or__`. These operations return a tensor
        of shape `batch_size`. For compatibility with `<a tensordict> == <float_number>`,
        comparison with non-:class:`NonTensorData` will always return an empty
        :class:`NonTensorData`.

        >>> a = NonTensorData(True, batch_size=[])
        >>> b = NonTensorData(True, batch_size=[])
        >>> assert a == b
        >>> assert not (a != b)
        >>> assert not (a ^ b)
        >>> assert a | b
        >>> # The output is a tensor of shape batch-size
        >>> a = NonTensorData(True, batch_size=[3])
        >>> b = NonTensorData(True, batch_size=[3])
        >>> print(a == b)
        tensor([True, True, True])

    .. note:: Stacking :class:`NonTensorData` instances results in either
        a single :class:`NonTensorData` instance if all shapes match, or a
        :class:`~tensordict.LazyStackedTensorDict` object if the content
        mismatch. To get to this result, the content of the :class:`NonTensorData`
        instances must be compared, which can be computationally intensive
        depending on what this content is.

        >>> data = torch.stack([NonTensorData(1, batch_size=[]) for _ in range(10)])
        >>> data
        NonTensorData(
            data=1,
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False)
        >>> data = torch.stack([NonTensorData(i, batch_size=[3,]) for i in range(10)], 1)
        >>> data[:, 0]
        NonTensorData(
            data=0,
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    .. note:: Non-tensor data can be filtered out from a tensordict using
        :meth:`~tensordict.TensorDictBase.filter_non_tensor`.

    Examples:
        >>> # create an instance explicitly
        >>> non_tensor = NonTensorData("a string!", batch_size=[]) # batch-size can be anything
        >>> data = TensorDict({}, batch_size=[3])
        >>> data.set_non_tensor(("nested", "key"), "a string!")
        >>> assert isinstance(data.get(("nested", "key")), NonTensorData)
        >>> assert data.get_non_tensor(("nested", "key")) == "a string!"
        >>> # serialization
        >>> class MyPickableClass:
        ...     value = 10
        >>> data.set_non_tensor("pickable", MyPickableClass())
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     data.memmap(tmpdir)
        ...     loaded = TensorDict.load_memmap(tmpdir)
        ...     # print directory path
        ...     print_directory_tree(tmpdir)
        Directory size: 511.00 B
        tmp2cso9og_/
            pickable/
                _tensordict/
                    meta.json
                other.pickle
                meta.json
            nested/
                key/
                    _tensordict/
                        meta.json
                    meta.json
                meta.json
            meta.json
        >>> assert loaded.get_non_tensor("pickable").value == 10

    """

    # Used to carry non-tensor data in a tensordict.
    # The advantage of storing this in a tensorclass is that we don't need
    # to patch tensordict with additional checks that will encur unwanted overhead
    # and all the overhead falls back on this class.
    data: Any

    def __post_init__(self):
        if isinstance(self.data, NonTensorData):
            self.data = self.data.data

        old_eq = self.__class__.__eq__
        if old_eq is _eq:
            global NONTENSOR_HANDLED_FUNCTIONS
            NONTENSOR_HANDLED_FUNCTIONS.extend(TD_HANDLED_FUNCTIONS)

            # Patch only the first time a class is created

            @functools.wraps(_eq)
            def __eq__(self, other):
                if isinstance(other, NonTensorData):
                    return torch.full(
                        self.batch_size, self.data == other.data, device=self.device
                    )
                return old_eq(self, other)

            self.__class__.__eq__ = __eq__

            _ne = self.__class__.__ne__

            @functools.wraps(_ne)
            def __ne__(self, other):
                if isinstance(other, NonTensorData):
                    return torch.full(
                        self.batch_size, self.data != other.data, device=self.device
                    )
                return _ne(self, other)

            self.__class__.__ne__ = __ne__

            _xor = self.__class__.__xor__

            @functools.wraps(_xor)
            def __xor__(self, other):
                if isinstance(other, NonTensorData):
                    return torch.full(
                        self.batch_size, self.data ^ other.data, device=self.device
                    )
                return _xor(self, other)

            self.__class__.__xor__ = __xor__

            _or = self.__class__.__or__

            @functools.wraps(_or)
            def __or__(self, other):
                if isinstance(other, NonTensorData):
                    return torch.full(
                        self.batch_size, self.data | other.data, device=self.device
                    )
                return _or(self, other)

            self.__class__.__or__ = __or__

    def empty(self, recurse=False):
        return NonTensorData(
            data=self.data,
            batch_size=self.batch_size,
            names=self.names if self._has_names() else None,
            device=self.device,
        )

    def to_dict(self):
        # override to_dict to return just the data
        return self.data

    @classmethod
    def _stack_non_tensor(cls, list_of_non_tensor, dim=0):
        # checks have been performed previously, so we're sure the list is non-empty
        first = list_of_non_tensor[0]
        if all(data.data == first.data for data in list_of_non_tensor[1:]):
            batch_size = list(first.batch_size)
            batch_size.insert(dim, len(list_of_non_tensor))
            return NonTensorData(
                data=first.data,
                batch_size=batch_size,
                names=first.names if first._has_names() else None,
                device=first.device,
            )

        from tensordict._lazy import LazyStackedTensorDict

        return LazyStackedTensorDict(*list_of_non_tensor, stack_dim=dim)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        # A modified version of __torch_function__ to account for the different behaviour
        # of stack, which should return lazy stacks of data of data does not match.
        if func not in _TD_PASS_THROUGH or not all(
            issubclass(t, (Tensor, cls)) for t in types
        ):
            return NotImplemented

        escape_conversion = func in (torch.stack,)

        if kwargs is None:
            kwargs = {}

        # get the output type from the arguments / keyword arguments
        if len(args) > 0:
            tensorclass_instance = args[0]
        else:
            tensorclass_instance = kwargs.get("input", kwargs["tensors"])
        if isinstance(tensorclass_instance, (tuple, list)):
            tensorclass_instance = tensorclass_instance[0]
        if not escape_conversion:
            args = tuple(_arg_to_tensordict(arg) for arg in args)
            kwargs = {key: _arg_to_tensordict(value) for key, value in kwargs.items()}

        result = TD_HANDLED_FUNCTIONS[func](*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return result.__class__(
                _from_tensordict_with_copy(tensorclass_instance, tensordict_result)
                for tensordict_result in result
            )
        if not escape_conversion:
            return _from_tensordict_with_copy(tensorclass_instance, result)
        return result
