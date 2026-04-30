from collections.abc import Callable
from typing import Any

from torch.utils._ordered_set import OrderedSet

from .effects import EffectHolder
from .fake_impl import FakeImplHolder
from .utils import RegistrationHandle


__all__ = [
    "SimpleLibraryRegistry",
    "SimpleOperatorEntry",
    "singleton",
    "SymmMemArgsHolder",
]


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

    def __init__(self) -> None:
        self._data: dict[str, SimpleOperatorEntry] = {}
        # Raw C++ symm_mem registrations, applied lazily to avoid circular imports.
        self._cpp_symm_mem_args: dict[str, list[str]] = {}

    def find(self, qualname: str) -> "SimpleOperatorEntry":
        res = self._data.get(qualname, None)
        if res is None:
            self._data[qualname] = res = SimpleOperatorEntry(qualname)
        self._apply_cpp_symm_mem(qualname, res)
        return res

    def get(self, qualname: str) -> "SimpleOperatorEntry | None":
        res = self._data.get(qualname, None)
        if res is None and qualname in self._cpp_symm_mem_args:
            return self.find(qualname)
        if res is not None:
            self._apply_cpp_symm_mem(qualname, res)
        return res

    def _apply_cpp_symm_mem(self, qualname: str, entry: "SimpleOperatorEntry") -> None:
        if entry.symm_mem_args._symm_mem_args is not None:
            return

        cpp_args = self._cpp_symm_mem_args.pop(qualname, None)

        # If not found in cache, check C++ registry directly (for late-loaded extensions)
        if cpp_args is None:
            try:
                import torch._C

                cpp_registry = (
                    torch._C._get_cpp_symm_mem_args_registry()  # pyrefly: ignore [missing-attribute]
                )
                cpp_args = cpp_registry.get(qualname, None)
            except ImportError:
                pass

        if cpp_args is not None:
            entry.symm_mem_args._symm_mem_args = OrderedSet(cpp_args)

    def _load_cpp_symm_mem_registrations(self) -> None:
        """Load C++ symm_mem registrations as raw data (applied lazily via find/get)."""
        try:
            import torch._C

            self._cpp_symm_mem_args = dict(
                torch._C._get_cpp_symm_mem_args_registry()  # pyrefly: ignore [missing-attribute]
            )
        except ImportError:
            pass


singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()


class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str) -> None:
        self.qualname: str = qualname
        self.fake_impl: FakeImplHolder = FakeImplHolder(qualname)
        self.torch_dispatch_rules: GenericTorchDispatchRuleHolder = (
            GenericTorchDispatchRuleHolder(qualname)
        )

        self.effect: EffectHolder = EffectHolder(qualname)
        self.symm_mem_args: SymmMemArgsHolder = SymmMemArgsHolder(qualname)

    # For compatibility reasons. We can delete this soon.
    @property
    def abstract_impl(self) -> FakeImplHolder:
        return self.fake_impl


class GenericTorchDispatchRuleHolder:
    def __init__(self, qualname: str) -> None:
        self._data: dict[type, Callable[..., Any]] = {}
        self.qualname: str = qualname

    def register(
        self, torch_dispatch_class: type, func: Callable[..., Any]
    ) -> RegistrationHandle:
        if self.find(torch_dispatch_class):
            raise RuntimeError(
                f"{torch_dispatch_class} already has a `__torch_dispatch__` rule registered for {self.qualname}"
            )
        self._data[torch_dispatch_class] = func

        def deregister() -> None:
            del self._data[torch_dispatch_class]

        return RegistrationHandle(deregister)

    def find(self, torch_dispatch_class: type) -> Callable[..., Any] | None:
        return self._data.get(torch_dispatch_class, None)


class SymmMemArgsHolder:
    """Tracks which arguments of an operator require symmetric memory allocation.

    Used by Inductor during lowering to automatically realize tensors as
    symmetric memory buffers.
    """

    def __init__(self, qualname: str) -> None:
        self._symm_mem_args: OrderedSet[str] | None = None
        self.qualname: str = qualname

    def register(self, arg_names: list[str], *, op_overload: Any | None = None) -> None:
        if not arg_names:
            raise ValueError(
                f"Cannot register empty arg_names list for {self.qualname}"
            )

        if op_overload is not None:
            self._validate_arg_names(arg_names, op_overload)

        if self._symm_mem_args is not None:
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                "Overwriting symm_mem arg registration for %s. Old: %s, New: %s",
                self.qualname,
                self._symm_mem_args,
                arg_names,
            )

        self._symm_mem_args = OrderedSet(arg_names)

    def get(self) -> OrderedSet[str] | None:
        return self._symm_mem_args

    def is_registered(self) -> bool:
        return self._symm_mem_args is not None

    def is_symm_mem_arg(self, arg_name: str) -> bool:
        return self._symm_mem_args is not None and arg_name in self._symm_mem_args

    def _validate_arg_names(self, arg_names: list[str], op_overload: Any) -> None:
        try:
            schema = op_overload._schema
            schema_arg_names = {arg.name for arg in schema.arguments}

            invalid_args = [name for name in arg_names if name not in schema_arg_names]
            if invalid_args:
                raise ValueError(
                    f"Invalid argument names for {self.qualname}: {invalid_args}. "
                    f"Valid arguments are: {sorted(schema_arg_names)}"
                )

            # Warn if group_name is missing (it's needed for automatic realization)
            if "group_name" not in schema_arg_names:
                import logging

                log = logging.getLogger(__name__)
                log.warning(
                    "Operator %s is missing 'group_name' argument. "
                    "Automatic symmetric memory realization will not work at compile time. "
                    "Available arguments: %s",
                    self.qualname,
                    sorted(schema_arg_names),
                )
        except AttributeError:
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                "Could not validate arg names for %s: OpOverload missing _schema attribute",
                self.qualname,
            )


def find_torch_dispatch_rule(
    op: Any, torch_dispatch_class: type
) -> Callable[..., Any] | None:
    return singleton.find(op.__qualname__).torch_dispatch_rules.find(
        torch_dispatch_class
    )


# load C++ registrations into the singleton registry after all classes are defined
singleton._load_cpp_symm_mem_registrations()
