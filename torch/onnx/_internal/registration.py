"""Module for handling symbolic function registration."""

import functools
import importlib
import inspect
import itertools
import warnings
from typing import Callable, Collection, Dict, Optional, Set

from torch.onnx import _constants

OpsetVersion = int
_BASE_OPSET_VERSION = 9


def _dispatch_opset_version(
    target: OpsetVersion, available_opsets: Collection[OpsetVersion]
) -> Optional[OpsetVersion]:
    """Finds the registered opset given a target opset version and the available opsets.

    Args:
        target: The target opset version.
        available_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    available_versions_set = set(available_opsets)
    if target in available_versions_set:
        # An exact match
        return target

    available_versions = sorted(available_opsets)
    # Linear search for the opset version, which is fine since the number of opset
    # versions is small.

    # Always round toward opset 9 (_BASE_OPSET_VERSION).
    # Count down until opset 9 is reached.
    for version in reversed(available_versions):
        if _BASE_OPSET_VERSION <= version <= target:
            return version

    for version in available_versions:
        # Count back up until _BASE_OPSET_VERSION
        if target < version <= _BASE_OPSET_VERSION:
            return version

    print(available_versions, target)
    assert (
        not available_versions
        or _BASE_OPSET_VERSION <= target < available_versions[0]
        or available_versions[-1] < _BASE_OPSET_VERSION < target
    )
    return None


class _SymbolicFunctionGroup:
    """Different versions of symbolic functions registered to the same name.

    O(n) search is performed to find the most recent version of the op.
    The results are cached for faster lookup.

    The registration is delayed until op is used to improve startup time.

    Function overloads with different arguments are not allowed.
    Custom op overrides are supported.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: Dict[OpsetVersion, Callable] = {}
        self._overrides: Dict[OpsetVersion, Callable] = {}
        # Symbolic functions and overrides combined.
        self._merged: Dict[OpsetVersion, Callable] = {}

    def __repr__(self) -> str:
        return f"_SymbolicFunctionGroup({self._name}, registered={self._merged})"

    def __getitem__(self, key: OpsetVersion) -> Callable:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def _update_merged(self) -> None:
        """Updates the merged dictionary of functions."""
        self._merged.clear()
        self._merged.update(self._functions)
        self._merged.update(self._overrides)
        self.get.cache_clear()

    @functools.lru_cache(maxsize=None)
    def get(self, opset: OpsetVersion) -> Optional[Callable]:
        """Find the most recent version of the function."""
        # Remember to clear the cache when the merged dictionary is updated.
        if not self._merged:
            return None

        # Linear search across the merged dictionary. This is OK because the
        # number of opsets is small and the result is cached.

        version = _dispatch_opset_version(opset, self._merged)
        if version is None:
            return None

        return self._merged[version]

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a symbolic function.

        Args:
            func: The function to add.
            opset: The opset version of the function to add.
        """
        if opset in self._functions:
            raise ValueError(
                f"Symbolic function '{self._name}' already registered for opset {opset}"
            )
        self._functions[opset] = func
        self._update_merged()

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
        self._overrides[opset] = func
        self._update_merged()

    def remove_custom(self, opset: OpsetVersion) -> None:
        """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
        if opset not in self._overrides:
            warnings.warn(
                f"No custom function registered for '{self._name}' opset {opset}"
            )
            return
        del self._overrides[opset]
        self._update_merged()

    def get_min_supported(self) -> OpsetVersion:
        """Returns the lowest built-in opset version supported by the function."""
        return min(self._functions)


class SymbolicRegistry:
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, _SymbolicFunctionGroup] = {}
        # Whether the registry has not been initialized with builtin symbolic functions.
        self._uninitialized = True

    @property
    def uninitialized(self) -> bool:
        """Whether the registry has not been initialized with builtin symbolic functions."""
        return self._uninitialized

    def register(
        self, name: str, opset: OpsetVersion, func: Callable, custom=False
    ) -> None:
        """Registers a symbolic function.

        Args:
            name: the qualified name of the function to register.
            opset: the opset version of the function to register.
            func: the symbolic function to register.
            custom: whether the function is a custom function that overrides existing ones.
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
            self._uninitialized = False
            symbolic_functions.add(func, opset)

    def unregister(self, name: str, opset: OpsetVersion) -> None:
        """Unregisters a symbolic function.

        Args:
            name: the qualified name of the function to unregister.
            opset: the opset version of the function to unregister.
        """
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
        """Returns the function group for the given name."""
        return self._registry.get(name)

    def is_registered_op(self, name: str, version: int) -> bool:
        """Returns whether the given op is registered for the given opset version."""
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    def all_functions(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)


def discover_and_register_all_symbolic_opsets() -> None:
    """Discover all symbolic functions.

    Opset 9 is the base version. It is selected as the base version because
        1. It is the first opset version supported by PyTorch export.
        2. opset 9 is more robust than previous opset versions. Opset versions like 7/8 have limitations
            that certain basic operators cannot be expressed in ONNX. Instead of basing on these limitations,
            we chose to handle them as special cases separately.

    Backward support for opset versions beyond opset 7 is not in our roadmap.

    For opset versions other than 9, by default they will inherit the symbolic functions defined in
    symbolic_opset9.py.

    To extend support for updated operators in different opset versions on top of opset 9,
    simply add the updated symbolic functions in the respective symbolic_opset{version}.py file.
    Checkout topk in symbolic_opset10.py, and upsample_nearest2d in symbolic_opset8.py for example.
    """
    global registry
    if not registry.uninitialized:
        return

    for opset in itertools.chain(
        _constants.onnx_stable_opsets, [_constants.onnx_main_opset]
    ):
        module = importlib.import_module(f"torch.onnx.symbolic_opset{opset}")
        _register_module(module, opset)


def _register_module(module, opset: OpsetVersion) -> None:
    """Registers all functions in the given module.

    Args:
        module: the module to register.
        opset: the opset version to register.
    """
    global registry
    members = inspect.getmembers(module)
    for name, obj in members:
        if isinstance(obj, type) and hasattr(obj, "domain"):
            # Symbolic functions in domains other than aten
            ops = inspect.getmembers(obj, predicate=inspect.isfunction)
            for op in ops:
                registry.register(f"{obj.domain}::{op[0]}", opset, op[1])  # type: ignore[attr-defined]

        elif inspect.isfunction(obj):
            if name in {"_len", "_list", "_any", "_all"}:
                name = name[1:]
            registry.register(f"aten::{name}", opset, obj)


# The registry for all symbolic functions.
registry = SymbolicRegistry()
