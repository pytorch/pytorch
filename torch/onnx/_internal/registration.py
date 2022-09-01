"""Module for handling symbolic function registration.


O(opset_count) search is performed to find the most recent version of the op.
the results are cached in a dictionary for faster lookup.

The registration is delayed until op is used to improve startup time.

Function overloads are not allowed. Custom op overrides are supported.
"""

import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from torch.onnx import errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype


# class _SymbolicFunctionSignature(Hashable):
#     """A hashable class for storing the argument signature of a symbolic function."""

#     def __init__(self, name: str, opset: int, arg_count: Optional[int]) -> None:
#         self.name = name
#         self.opset = int
#         self.arg_count = arg_count

#     def __hash__(self) -> int:
#         return hash((self.opset, self.arg_count))

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, _SymbolicFunctionSignature):
#             return False
#         return self.opset == other.opset and self.arg_count == other.arg_count

#     def __repr__(self) -> str:
#         return f"SymbolicFunctionSignature({self.name}, opset={self.opset}, arg_count={self.arg_count})"

OpsetVersion = int


class _SymbolicFunctionGroup:
    """Overloads of symbolic functions registered to the same name."""

    def __init__(self, name: str) -> None:
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: Dict[OpsetVersion, Callable] = {}
        self._overrides: Dict[OpsetVersion, Callable] = {}
        # Symbolic functions and overrides combined.
        self._merged: Dict[OpsetVersion, Callable] = {}

    def __repr__(self) -> str:
        return f"_SymbolicFunctionGroup({self._name})"

    def __getitem__(self, key: OpsetVersion) -> Callable:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the current opset version of the function."""
        func = self.get(GLOBALS.export_onnx_opset_version)
        if func is None:
            domain, op = self._name.split("::")
            raise errors.UnsupportedOperatorError(
                domain, op, GLOBALS.export_onnx_opset_version, min(self._functions)
            )
        return func(*args, **kwargs)

    def _update_merged(self) -> None:
        """Updates the merged dictionary of functions."""
        self._merged.clear()
        self._merged.update(self._functions)
        self._merged.update(self._overrides)
        self.get.cache_clear()

    @functools.cache
    def get(self, opset: OpsetVersion) -> Optional[Callable]:
        """Find the most recent version of the function."""
        # Remember to clear the cache when the merged dictionary is updated.

        # Linear search across the merged dictionary. This is OK because the
        # number of opsets is small and the result is cached.
        for version in reversed(sorted(self._merged.keys())):
            if version <= opset:
                return self._merged[version]
        return None

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


class SymbolicRegistry:
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, _SymbolicFunctionGroup] = {}

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
        symbolic_functions = self._registry.setdefault(
            name, _SymbolicFunctionGroup(name)
        )
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
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

    def get_function_group(self, name: str) -> _SymbolicFunctionGroup:
        """Returns the function group for the given name."""
        if name not in self._registry:
            raise ValueError(f"No symbolic function registered for '{name}'")
        return self._registry[name]


@_beartype.beartype
def onnx_symbolic(
    name: str,
    opset: OpsetVersion,
    decorate: Optional[Sequence[Callable]] = None,
    custom: bool = False,
) -> Callable:
    """Registers a symbolic function.

    Usage::

    ```
    @onnx_symbolic("aten::symbolic_b", opset=10, decorate=[quantized_aten_handler(scale=1/128, zero_point=0)])
    @symbolic_helper.parse_args("v", "v", "b")
    def symbolic_b(g: _C.Graph, x: _C.Value, y: _C.Value, arg1: bool) -> _C.Value:
        ...
    ```

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.
        custom: whether the function is a custom symbolic function.
    """

    def wrapper(func: Callable) -> Callable:
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        if custom:
            global registry
            registry.register(name, opset, decorated, custom=custom)
        else:
            # Store all torch.onnx built-in functions for delayed registration.
            global collected_symbolic_functions
            collected_symbolic_functions.append((name, opset, decorated, custom))

        return decorated

    return wrapper


@_beartype.beartype
def custom_onnx_symbolic(
    name: str, opset: OpsetVersion, decorate: Optional[Sequence[Callable]] = None
) -> Callable:
    """Registers a custom symbolic function.

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.

    Returns:
        The decorator.
    """
    return onnx_symbolic(name, opset, decorate, custom=True)


# The registry for all symbolic functions.
registry = SymbolicRegistry()
# Store the discovered functions for delayed registration.
collected_symbolic_functions: List[Tuple[str, OpsetVersion, Callable, bool]] = []
