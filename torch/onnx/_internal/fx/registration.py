"""Module for handling ATen to ONNX functions registration."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Union

import torch._ops
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

    op_full_name: The qualified name of the function. In the form of '<namespace>::<op_name>.<overload>'.
    onnx_function: The symbolic function from torchlib.
    is_custom: Whether the function is a custom function.

    """

    onnx_function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"]
    op_full_name: str
    is_custom: bool = False


class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    Attributes:
        _registry: The registry maps OpNameto a list of SymbolicFunctions. It is important
            not to directly modify this variable. Instead, access to it should be done through
            the public methods: register_custom_op, get_functions, and is_registered_op.

    """

    def __init__(self, opset_version: int) -> None:
        """Initializes the registry.

        Args:
            opset_version: The opset version to use for the registry.

        """
        self._registry: Dict[OpName, List[SymbolicFunction]] = defaultdict(list)
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
            internal_name_instance = OpName.from_qualified_name(aten_name)
            for overload_func in aten_overloads_func.overloads:
                symbolic_function = SymbolicFunction(
                    onnx_function=overload_func,
                    op_full_name=internal_name_instance.qualified_name(),
                    is_custom=False,
                )
                self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def _register(
        self, internal_qualified_name: OpName, symbolic_function: SymbolicFunction
    ) -> None:
        """Registers a SymbolicFunction to an operator.

        Args:
            internal_qualified_name: The qualified name of the operator to register: OpName.
            symbolic_function: The SymbolicFunction to register.
        """
        self._registry[internal_qualified_name].append(symbolic_function)

    @_beartype.beartype
    def register_custom_op(
        self,
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
        namespace: str,
        op_name: str,
        overload: Optional[str] = None,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
            namespace: The namespace of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
        """
        internal_name_instance = OpName.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        symbolic_function = SymbolicFunction(
            onnx_function=function,
            op_full_name=internal_name_instance.qualified_name(),
            is_custom=True,
        )
        self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def get_functions(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[List[SymbolicFunction]]:
        """Returns a list of SymbolicFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            A list of SymbolicFunctions corresponding to the given name, or None if
            the name is not in the registry.
        """
        internal_name_instance = OpName.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return self._registry.get(internal_name_instance)

    @_beartype.beartype
    def is_registered_op(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_functions(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return functions is not None

    @_beartype.beartype
    def _all_registered_ops(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return {
            op_name_class.qualified_name() for op_name_class in self._registry.keys()
        }


@dataclasses.dataclass(frozen=True, eq=True)
class OpName:
    """A class representing an operator name in internal ONNX converter."""

    namespace: str
    op_name: str
    overload: str

    @classmethod
    @_beartype.beartype
    def from_name_parts(
        cls, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> OpName:
        # NOTE: in PyTorch, the overload could be unprovided to indicate the
        # default overload
        # TODO: This is slightly unsafe that dev could accidentally create illegal
        # OpName by using initializer directly
        # https://github.com/pytorch/pytorch/pull/103943#discussion_r1256511069
        if overload is None or overload == "":
            overload = "default"
        return cls(namespace, op_name, overload)

    @classmethod
    @_beartype.beartype
    def from_qualified_name(cls, qualified_name: str) -> OpName:
        """When the name is <namespace>::<op_name>[.<overload>]"""
        namespace, opname_overload = qualified_name.split("::")
        op_name, *overload = opname_overload.split(".", 1)
        overload = overload[0] if overload else "default"
        return cls(namespace, op_name, overload)

    @classmethod
    @_beartype.beartype
    def from_op_overload(cls, op_overload: torch._ops.OpOverload) -> OpName:
        return cls.from_qualified_name(op_overload.name())

    @_beartype.beartype
    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.op_name}.{self.overload}"
