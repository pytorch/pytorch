"""Module for handling ATen to ONNX functions registration."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING


# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import types

    import onnxscript  # type: ignore[import]

    import torch._ops


@dataclasses.dataclass(frozen=True, eq=True)
class ONNXFunction:
    """A wrapper of onnx-script function.

    op_full_name: The qualified name of the function. In the form of '<namespace>::<op_name>.<overload>'.
    onnx_function: The onnx-script function from torchlib.
    is_custom: Whether the function is a custom function.
    is_complex: Whether the function is a function that handles complex valued inputs.

    """

    onnx_function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction
    op_full_name: str
    is_custom: bool = False
    is_complex: bool = False


@dataclasses.dataclass(frozen=True, eq=True)
class OpName:
    """A class representing an operator name in internal ONNX converter."""

    namespace: str
    op_name: str
    overload: str

    @classmethod
    def from_name_parts(
        cls, namespace: str, op_name: str, overload: str | None = None
    ) -> OpName:
        # NOTE: in PyTorch, the overload could be unprovided to indicate the
        # default overload
        if overload is None or overload == "":
            overload = "default"
        return cls(namespace, op_name, overload)

    @classmethod
    def from_qualified_name(cls, qualified_name: str) -> OpName:
        """When the name is <namespace>::<op_name>[.<overload>]"""
        namespace, opname_overload = qualified_name.split("::")
        op_name, *overload = opname_overload.split(".", 1)
        overload = overload[0] if overload else "default"
        return cls(namespace, op_name, overload)

    @classmethod
    def from_op_overload(cls, op_overload: torch._ops.OpOverload) -> OpName:
        return cls.from_qualified_name(op_overload.name())

    @classmethod
    def from_builtin_function(
        cls, builtin_function: types.BuiltinFunctionType
    ) -> OpName:
        """From a builtin function, e.g. operator.add, math.ceil, etc, get the OpName.

        FX graph uses built-in functions to caculate sympy expression. This function
        is used to get the OpName from a builtin function.

        Args:
            builtin_function (types.BuiltinFunctionType): operator.add, math.ceil, etc.

        Returns:
            OpName: _description_
        """
        op = builtin_function.__name__  # add, sub, etc.
        module = builtin_function.__module__  # _operators or math
        return cls.from_qualified_name(module + "::" + op)

    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.op_name}.{self.overload}"
