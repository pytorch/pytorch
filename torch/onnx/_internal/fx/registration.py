"""Module for handling ATen to ONNX functions registration."""

from __future__ import annotations

import warnings
from typing import (
    Callable,
    Collection,
    Dict,
    Generic,
    Optional,
    Set,
    TYPE_CHECKING,
    TypeVar,
)

from torch.onnx._internal import _beartype

# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    from onnxscript.function_libs.torch_lib import registration  # type: ignore[import]

OpsetVersion = int
_K = TypeVar("_K")
_V = TypeVar("_V")


class MergeDict(Generic[_K, _V], Collection[_K]):
    """
    A dictionary that merges built-in and custom symbolic functions.

    It supports adding and removing built-in symbolic functions with custom ones.

    Attributes:
        _torchlib (Dict[_K, List[_V]]): A dictionary to hold built-in symbolic functions.
        _customs (Dict[_K, List[_V]]): A dictionary to hold custom symbolic functions.
        _merged (Dict[_K, List[_V]]): A dictionary that merges built-in and custom symbolic functions.
    """

    def __init__(self):
        self._torchlib: Dict[_K, Set[_V]] = {}
        self._merged: Dict[_K, Set[_V]] = {}
        self._customs: Dict[_K, Set[_V]] = {}

    def set_base(self, key: _K, value: _V) -> None:
        self._torchlib.setdefault(key, set()).add(value)
        if key not in self._customs:
            self._merged.setdefault(key, set()).add(value)

    def in_base(self, key: _K) -> bool:
        """Checks if a key is in the base dictionary."""
        return key in self._torchlib

    def add_custom(self, key: _K, value: _V) -> None:
        """Add a base key-value with a new pair."""
        self._customs.setdefault(key, set()).add(value)
        self._merged.setdefault(key, set()).add(value)

    def remove_custom(self, key: _K) -> None:
        """Remove a key-value pair."""
        # FIXME(titaiwang): How to remove a specific function instead of whole overloads?
        self._customs.pop(key, None)  # type: ignore[arg-type]
        self._merged.pop(key, None)  # type: ignore[arg-type]
        if key in self._torchlib:
            self._merged[key] = self._torchlib[key]

    def custom_added(self, key: _K) -> bool:
        """Checks if a key-value pair is overridden."""
        return key in self._customs

    def __getitem__(self, key: _K) -> Set[_V]:
        return self._merged[key]

    def get(self, key: _K, default: Optional[Set[_V]] = None) -> Optional[Set[_V]]:
        return self._merged.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._merged

    def __iter__(self):
        return iter(self._merged)

    def __len__(self) -> int:
        return len(self._merged)

    def __repr__(self) -> str:
        return f"MergeDict(torchlib={self._torchlib}, customs={self._customs})"

    def __bool__(self) -> bool:
        return bool(self._merged)


class _SymbolicFunctionGroup:
    """A group of overloaded functions registered to the same name.

    This class stores a collection of overloaded functions that share the same name.
    Each function is associated with a specific opset version, and multiple versions
    of the same function can be registered to support different opset versions.

    Attributes:
        _name: The name of the function group.
        _functions: A dictionary of functions, keyed by the opset version.

    """

    def __init__(self, name: str) -> None:
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: MergeDict[OpsetVersion, Callable] = MergeDict()

    def __repr__(self) -> str:
        return f"_SymbolicFunctionGroup({self._name}, registered={self._functions})"

    def __getitem__(self, key: OpsetVersion) -> Set[Callable]:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, opset: OpsetVersion) -> Optional[Set[Callable]]:
        """Find the most recent version of the overloaded functions."""
        return self._functions.get(opset)

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a symbolic function.

        Args:
            func: The function to add.
            opset: The opset version of the function to add.
        """
        # FIXME(titaiwang): Check if the "function" is ducplicated.
        self._functions.set_base(opset, func)

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
        self._functions.add_custom(opset, func)

    def remove_custom(self, opset: OpsetVersion) -> None:
        """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
        if not self._functions.custom_added(opset):
            warnings.warn(
                f"No custom function registered for '{self._name}' opset {opset}"
            )
            return
        self._functions.remove_custom(opset)

    def support_opset(self) -> Collection[OpsetVersion]:
        """Returns a list of supported opset versions."""
        return list(self._functions)


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions,
    which can be overloaded to support multiple opset versions. It is used to
    register new symbolic functions and to dispatch calls to the appropriate function.

    Attributes:
        _registry: A dictionary mapping qualified names to _SymbolicFunctionGroup objects.
    """

    def __init__(self, opset_version: int = 18) -> None:
        self._registry: Dict[str, _SymbolicFunctionGroup] = {}
        # FIXME: Avoid importing onnxscript into torch
        from onnxscript.function_libs.torch_lib import (  # type: ignore[import]  # noqa: F401
            ops,  # TODO(titaiwang): get rid of this import
            registration,
        )

        self._opset_version = opset_version
        self._initiate_registry_from_torchlib(registration.default_registry)

    # TODO(titaiwang): subject to change if multiple opset_version is supported in torchlib
    def _initiate_registry_from_torchlib(
        self, torchlib_registry: registration.Registry
    ):
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        for aten_name, aten_overloads_func in torchlib_registry.items():
            for func in aten_overloads_func.overloads:
                self.register(
                    aten_name,
                    self._opset_version,
                    func,
                    custom=False,
                )

    @_beartype.beartype
    def register(
        self, name: str, opset: OpsetVersion, func: Callable, custom: bool = True
    ) -> None:
        """Registers overloaded functions to an ATen operator (overload).

        Args:
            name: The qualified name of the function to register. In the form of 'domain::op'.
                E.g. 'aten::add' or 'aten::pow.int'.
            opset: The opset version of the function to register.
            func: The symbolic function to register.
            custom: Whether the function is a custom function that overrides existing ones.

        Raises:
            ValueError: If the separator '::' is not in the name.
        """
        if "::" not in name:
            raise ValueError(
                f"The name must be in the form of 'domain::op', not '{name}'"
            )
        symbolic_functions = self._registry.setdefault(
            name, _SymbolicFunctionGroup(name)
        )
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    @_beartype.beartype
    def unregister(self, name: str, opset: OpsetVersion) -> None:
        """Unregisters overloaded functions to an ATen operator (overload).

        Args:
            name: The qualified name of the function to unregister.
            opset: The opset version of the function to unregister.
        """
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    @_beartype.beartype
    def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
        """Returns the _SymbolicFunctionGroup object for the given name.

        Args:
            name: The qualified name of the function group to retrieve.

        Returns:
            The _SymbolicFunctionGroup object corresponding to the given name, or None if the name is not in the registry.
        """
        return self._registry.get(name)

    @_beartype.beartype
    def is_registered_op(self, name: str, version: int) -> bool:
        """Returns whether the given op is registered for the given opset version.

        Args:
            name: The qualified name of the function to check.
            version: The opset version to check.

        Returns:
            True if the given op is registered for the given opset version, otherwise False.
        """
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    @_beartype.beartype
    def all_functions(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)
