# mypy: allow-untyped-defs
from typing import Callable, Optional

from .fake_impl import FakeImplHolder
from .utils import RegistrationHandle


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
        res = self._data.get(qualname, None)
        if res is None:
            self._data[qualname] = res = SimpleOperatorEntry(qualname)
        return res


singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()


class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self.fake_impl: FakeImplHolder = FakeImplHolder(qualname)
        self.torch_dispatch_rules: GenericTorchDispatchRuleHolder = (
            GenericTorchDispatchRuleHolder(qualname)
        )

    # For compatibility reasons. We can delete this soon.
    @property
    def abstract_impl(self):
        return self.fake_impl


class GenericTorchDispatchRuleHolder:
    def __init__(self, qualname):
        self._data = {}
        self.qualname = qualname

    def register(
        self, torch_dispatch_class: type, func: Callable
    ) -> RegistrationHandle:
        if self.find(torch_dispatch_class):
            raise RuntimeError(
                f"{torch_dispatch_class} already has a `__torch_dispatch__` rule registered for {self.qualname}"
            )
        self._data[torch_dispatch_class] = func

        def deregister():
            del self._data[torch_dispatch_class]

        return RegistrationHandle(deregister)

    def find(self, torch_dispatch_class):
        return self._data.get(torch_dispatch_class, None)


def find_torch_dispatch_rule(op, torch_dispatch_class: type) -> Optional[Callable]:
    return singleton.find(op.__qualname__).torch_dispatch_rules.find(
        torch_dispatch_class
    )
