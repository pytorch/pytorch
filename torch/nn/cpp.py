# mypy: allow-untyped-defs
"""Functionality for Python <-> C++ frontend inter-op."""

from collections.abc import Iterator, MutableMapping
from typing import Any, overload, TypeVar

from torch import Tensor
from torch.nn import Module, Parameter


_T = TypeVar("_T")


class OrderedDictWrapper(MutableMapping[str, _T]):
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr: str) -> None:
        self.cpp_module = cpp_module
        self.attr = attr

    @property
    def cpp_dict(self):
        return getattr(self.cpp_module, self.attr)

    # Magic methods cannot be assigned dynamically and bypass ``getattr``, so we
    # must manually override them.

    def items(self) -> list[tuple[str, _T]]:  # type: ignore[override]
        return self.cpp_dict.items()

    def keys(self) -> list[str]:  # type: ignore[override]
        return self.cpp_dict.keys()

    def values(self) -> list[_T]:  # type: ignore[override]
        return self.cpp_dict.values()

    # This should return an Iterator[str], but OrderedDict::item is not currently
    # designed to let us iterate over only the keys.
    def __iter__(self) -> Iterator[tuple[str, _T]]:  # type: ignore[override]
        return self.cpp_dict.__iter__()

    def __len__(self) -> int:
        return self.cpp_dict.__len__()

    def __contains__(self, key: Any) -> bool:
        if not isinstance(key, str):
            return False
        return self.cpp_dict.__contains__(key)

    @overload
    def __getitem__(self, arg0: str) -> _T: ...

    @overload
    def __getitem__(self, arg0: int) -> tuple[str, _T]: ...

    def __getitem__(self, arg0: int | str) -> tuple[str, _T] | _T:
        return self.cpp_dict.__getitem__(arg0)

    def __setitem__(self, arg0: str, arg1: _T) -> None:
        self.cpp_dict.__setitem__(arg0, arg1)

    def __delitem__(self, arg0: str) -> None:
        self.cpp_dict.__delitem__(arg0)


class ModuleWrapper(Module):
    """A subclass of ``torch.nn.Module`` that wraps a C++ frontend module and delegates all access."""

    def __init__(self, cpp_module):
        # Assign before the super class constructor so ``self.training`` can be
        # assigned to in the super class constructor.
        self.cpp_module = cpp_module
        super().__init__()

        # In all three of these cases, None is not actually possible, but MutableMapping
        # enforces an invariant on the value type.
        self._parameters = OrderedDictWrapper[Parameter | None](
            cpp_module, "_parameters"
        )
        self._buffers = OrderedDictWrapper[Tensor | None](cpp_module, "_buffers")
        self._modules = OrderedDictWrapper[Module | None](cpp_module, "_modules")

        for attr in dir(cpp_module):
            # Skip magic methods and the three attributes above.
            if not attr.startswith("_"):
                setattr(self, attr, getattr(self.cpp_module, attr))

    def _apply(self, fn, recurse=True):
        for param in self.parameters():
            # Tensors stored in modules are graph leaves, and we don't
            # want to create copy nodes, so we have to unpack the data.
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad.data = fn(param._grad.data)

        for buf in self.buffers():
            buf.data = fn(buf.data)

        return self

    # nn.Module defines training as a boolean
    @property  # type: ignore[override]
    def training(self):
        return self.cpp_module.training

    @training.setter
    def training(self, mode):
        self.cpp_module.train(mode)

    def __repr__(self):
        return self.cpp_module.__repr__()


__all__ = ("OrderedDictWrapper", "ModuleWrapper")
