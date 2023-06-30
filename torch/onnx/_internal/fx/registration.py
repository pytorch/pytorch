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
        _registry: The registry maps <domain>::<op_name>.<overload>(e.g: aten::add.Tensor)
            to a set of SymbolicFunctions. It is important not to directly modify this variable.
            Instead, access to it should be done through the public methods: register_custom_op,
            get_functions, and is_registered_op.

    Public Methods:
        register_custom_op: Registers a custom operator.
        get_functions: Returns the set of SymbolicFunctions for the given op.
        is_registered_op: Returns whether the given op is registered.
    """

    def __init__(self, opset_version: int = 18) -> None:
        """Initializes the registry.

        Args:
            opset_version: The opset version to use for the registry.

        """
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
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
        domain: str,
        op_name: str,
        overload: Optional[str] = None,
    ) -> None:
        """Registers a custom operator: torch.ops.<domain>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
            domain: The domain of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.

        Raises:
            ValueError: If the name is not in the form of 'domain::op'.
        """
        if overload is None:
            # NOTE: "default" overload
            name = f"{domain}::{op_name}"
        else:
            name = f"{domain}::{op_name}.{overload}"
        symbolic_function = SymbolicFunction(
            onnx_function=function, op_name=name, is_custom=True
        )
        self._register(symbolic_function)

    @_beartype.beartype
    def get_functions(
        self, domain: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[Set[SymbolicFunction]]:
        """Returns the set of SymbolicFunctions for the given op: torch.ops.<domain>.<op_name>.<overload>.

        Args:
            domain: The domain of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            Thethe set of SymbolicFunctions corresponding to the given name, or None if
            the name is not in the registry.
        """
        if overload is None:
            # NOTE: "default" overload
            name = f"{domain}::{op_name}"
        else:
            name = f"{domain}::{op_name}.{overload}"
        if (functions := self._registry.get(name)) is not None:
            return functions
        return None

    @_beartype.beartype
    def _get_custom_functions(
        self, domain: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[Set[SymbolicFunction]]:
        """Returns the set of custom functions for the given name: torch.ops.<domain>.<op_name>.<overload>.

        Args:
            domain: The domain of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.

        Returns:
            The set of custom SymbolicFunctions corresponding to the given name, or None
            if the name is not in the registry.
        """
        if (
            functions := self.get_functions(
                domain=domain, op_name=op_name, overload=overload
            )
        ) is not None:
            custom_functions = {func for func in functions if func.is_custom}
            if custom_functions:
                return custom_functions
        return None

    @_beartype.beartype
    def is_registered_op(
        self, domain: str, op_name: str, overload: Optional[str] = None
    ) -> bool:
        """Returns whether the given op is registered: torch.ops.<domain>.<op_name>.<overload>.

        Args:
            domain: The domain of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_functions(
            domain=domain, op_name=op_name, overload=overload
        )
        return functions is not None

    @_beartype.beartype
    def _all_registered_ops(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)
