# mypy: allow-untyped-defs
"""Functionality for Python <-> C++ frontend inter-op."""

from torch import nn


class OrderedDictWrapper:
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr) -> None:
        self.cpp_module = cpp_module
        self.attr = attr

    @property
    def cpp_dict(self):
        return getattr(self.cpp_module, self.attr)

    # Magic methods cannot be assigned dynamically and bypass ``getattr``, so we
    # must manually override them.

    def items(self):
        return self.cpp_dict.items()

    def keys(self):
        return self.cpp_dict.keys()

    def values(self):
        return self.cpp_dict.values()

    def __iter__(self):
        return self.cpp_dict.__iter__()

    def __len__(self) -> int:
        return self.cpp_dict.__len__()

    def __contains__(self, key) -> bool:
        return self.cpp_dict.__contains__(key)

    def __getitem__(self, key):
        return self.cpp_dict.__getitem__(key)


class ModuleWrapper(nn.Module):
    """A subclass of ``torch.nn.Module`` that wraps a C++ frontend module and delegates all access."""

    def __init__(self, cpp_module) -> None:
        # Assign before the super class constructor so ``self.training`` can be
        # assigned to in the super class constructor.
        self.cpp_module = cpp_module
        super().__init__()
        self._parameters = OrderedDictWrapper(cpp_module, "_parameters")  # type: ignore[assignment]
        self._buffers: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_buffers")  # type: ignore[assignment]
        self._modules: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_modules")  # type: ignore[assignment]
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
    # pyrefly: ignore [bad-override]
    def training(self):
        return self.cpp_module.training

    @training.setter
    def training(self, mode) -> None:
        self.cpp_module.train(mode)

    def __repr__(self) -> str:
        return self.cpp_module.__repr__()
