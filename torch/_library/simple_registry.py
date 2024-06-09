# mypy: allow-untyped-defs
from .abstract_impl import AbstractImplHolder

__all__ = ["SimpleLibraryRegistry", "SimpleOperatorEntry", "singleton"]


class SimpleLibraryRegistry:
    """Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - fake impl

    Registrations for these APIs do not go into the PyTorch dispatcher's
    table because they may not directly involve a DispatchKey. For example,
    the fake impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    """

    def __init__(self):
        self._data = {}

    def find(self, qualname: str) -> "SimpleOperatorEntry":
        if qualname not in self._data:
            self._data[qualname] = SimpleOperatorEntry(qualname)
        return self._data[qualname]


singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()


class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self.abstract_impl: AbstractImplHolder = AbstractImplHolder(qualname)
