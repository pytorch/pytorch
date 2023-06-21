"""Module for handling ATen to ONNX functions registration."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, Optional, Set, TYPE_CHECKING, Union

from torch.onnx._internal import _beartype

# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import onnxscript  # type: ignore[import]
    from onnxscript.function_libs.torch_lib import registration  # type: ignore[import]

OpsetVersion = int


@dataclasses.dataclass(frozen=True, eq=True)
class SymbolicFunction:
    """A wrapper of symbolic function from torchlib.

    onnx_function: The symbolic function from torchlib.
    is_complex: Whether the function is a complex function.
    is_custom: Whether the function is a custom function.

    """

    onnx_function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]
    is_complex: bool = False
    is_custom: bool = False


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It is used to register custom symbolic functions and to dispatch
    calls to the appropriate function.

    Attributes:
        _registry: A dictionary mapping qualified names to _SymbolicFunction objects.
    """

    def __init__(self, opset_version: int = 18) -> None:
        self._registry: Dict[str, Set[SymbolicFunction]] = defaultdict(set)
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
            for overload_func in aten_overloads_func.overloads:
                self.register(aten_name, overload_func)
            for complex_func in aten_overloads_func.complex:
                self.register(aten_name, complex_func)

    @_beartype.beartype
    def register(self, name: str, symbolic_function: SymbolicFunction) -> None:
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
        self._registry[name].add(symbolic_function)

    @_beartype.beartype
    def get_functions(
        self, name: str, complex: bool = False
    ) -> Optional[Set[SymbolicFunction]]:
        """Returns the _SymbolicFunctionGroup object for the given name.

        Args:
            name: The qualified name of the functions to retrieve.
            complex: Whether to return complex functions.

        Returns:
            The SymbolicFunction object corresponding to the given name, or None if the name is not in the registry.
        """
        functions = self._registry.get(name)
        if functions is None:
            return None
        if complex:
            return self._get_complex_functions(functions)
        return functions

    # Do we need get default functions?
    @_beartype.beartype
    def get_custom_functions(
        self, name: str, complex: bool = False
    ) -> Optional[Set[SymbolicFunction]]:
        functions = self.get_functions(name, complex=complex)
        if functions is None:
            return None

        custom_functions = {func for func in functions if func.is_custom}
        if not custom_functions:
            return None
        return custom_functions

    @_beartype.beartype
    def _get_complex_functions(
        self, functions: Set[SymbolicFunction]
    ) -> Optional[Set[SymbolicFunction]]:
        complex_functions = {func for func in functions if func.is_complex}
        if not complex_functions:
            return None
        return complex_functions

    @_beartype.beartype
    def is_registered_op(self, name: str) -> bool:
        """Returns whether the given op is registered.

        Args:
            name: The qualified name of the function to check.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_functions(name)
        return functions is not None

    @_beartype.beartype
    def all_functions(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)
