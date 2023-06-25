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
    """A wrapper of onnx-script function.

    op_name: The qualified name of the function. In the form of 'domain::op'.
    onnx_function: The symbolic function from torchlib.
    is_custom: Whether the function is a custom function.

    """

    onnx_function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]
    op_name: str
    is_custom: bool = False


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    Attributes:
        _registry: A dictionary mapping qualified names to a set of SymbolicFunctions.

    Public Methods:
        register_custom_op: Registers a custom operator.
        get_functions: Returns the set of SymbolicFunctions for the given op.
        is_registered_op: Returns whether the given op is registered.
        all_registered_ops: Returns the set of all registered op names.

    Private Methods:
        _register: Registers a SymbolicFunction to an operator.
        _initiate_registry_from_torchlib: Populates the registry with ATen functions from torchlib.
        _get_custom_functions: Returns the set of custom functions for the given name.

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
                symbolic_function = SymbolicFunction(
                    onnx_function=overload_func, op_name=aten_name, is_custom=False
                )
                self._register(symbolic_function)

    @_beartype.beartype
    def _register(self, symbolic_function: SymbolicFunction) -> None:
        """Registers a SymbolicFunction to an operator.

        Args:
            symbolic_function: The SymbolicFunction to register.
        """
        self._registry[symbolic_function.op_name].add(symbolic_function)

    @_beartype.beartype
    def register_custom_op(
        self,
        op_name: str,
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
    ) -> None:
        """Registers a custom operator.

        Args:
            op_name: The qualified name of the operator to register. In the form of 'domain::op'.
                E.g. 'aten::add' or 'aten::pow.int'.
            function: The onnx-sctip function to register.

        Raises:
            ValueError: If the name is not in the form of 'domain::op'.
        """
        if "::" not in op_name:
            raise ValueError(
                f"The name must be in the form of 'domain::op', not '{op_name}'"
            )
        symbolic_function = SymbolicFunction(
            onnx_function=function, op_name=op_name, is_custom=True
        )
        self._register(symbolic_function)

    @_beartype.beartype
    def get_functions(self, name: str) -> Optional[Set[SymbolicFunction]]:
        """Returns the set of SymbolicFunctions for the given op.

        Args:
            name: The qualified op name of the functions to retrieve.

        Returns:
            Thethe set of SymbolicFunctions corresponding to the given name, or None if
            the name is not in the registry.
        """
        if (functions := self._registry.get(name)) is not None:
            return functions
        return None

    @_beartype.beartype
    def _get_custom_functions(self, op_name: str) -> Optional[Set[SymbolicFunction]]:
        """Returns the set of custom functions for the given name.

        Args:
            op_name: The qualified op name of the functions to retrieve.

        Returns:
            The set of custom SymbolicFunctions corresponding to the given name, or None
            if the name is not in the registry.
        """
        if (functions := self.get_functions(op_name)) is not None:
            custom_functions = {func for func in functions if func.is_custom}
            if custom_functions:
                return custom_functions
        return None

    @_beartype.beartype
    def is_registered_op(self, op_name: str) -> bool:
        """Returns whether the given op is registered.

        Args:
            op_name: The qualified op name of the function to check.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_functions(op_name)
        return functions is not None

    @_beartype.beartype
    def all_registered_ops(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)
