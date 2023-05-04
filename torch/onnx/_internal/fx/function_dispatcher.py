"""Dispatcher for AtenLib functions from onnx-script."""

from __future__ import annotations

import operator
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Protocol,
    runtime_checkable,
    Sequence,
    Union,
)

import onnx

import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
    graph_building,
    registration,
)

import torch
import torch.fx
from torch.onnx._internal import _beartype


_TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS = {
    torch.bfloat16: {"tensor(bfloat16)"},
    torch.bool: {"tensor(bool)"},
    torch.float64: {"tensor(double)"},
    torch.float32: {"tensor(float)"},
    torch.float16: {"tensor(float16)"},
    torch.int16: {"tensor(int16)"},
    torch.int32: {"tensor(int32)"},
    torch.int64: {"tensor(int64)"},
    torch.int8: {"tensor(int8)"},
    str: {"tensor(string)"},
    int: {"tensor(int16)", "tensor(int32)", "tensor(int64)"},
    float: {"tensor(float16)", "tensor(float)", "tensor(double)"},
    bool: {"tensor(int32)", "tensor(int64)", "tensor(bool)"},
}


@runtime_checkable
class WithDtype(Protocol):
    @property
    def dtype(self) -> torch.dtype:
        ...


class Dispatcher:
    """
    FX exporter dispatcher finds the best matched function for a given aten op.

    # 1. Use torch.ops name to find the function
        1a. Check aten:stft.center
        2a. Check aten:stft (Fall back to default)

    # 2. Find the best match among all overloaded functions.
        2a. Type matched -> selected
        2b. Type mismatched -> Nearest upcast/downcast
            Rules:
                - Float to Float / Int to Int
                - Int to Float
    """

    def __init__(self, registry: registration.Registry):
        self._registry = registry

    @property
    def registry(self) -> registration.Registry:
        return self._registry

    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence[graph_building.TorchScriptTensor],
        onnx_kwargs: Mapping[str, onnxscript.values.Value],
    ) -> Union[onnxscript.values.ONNXFunction, onnxscript.values.TracedONNXFunction]:
        """Dispatch a function.

        Args:
            node: torch.fx.Node.
            args: Arguments of the function.
            kwargs: Keyword arguments of the function.

        Returns:
            The result of the function.
        """
        aten_op = node.target

        if aten_op == operator.getitem:
            aten_name = "aten::getitem"
        elif isinstance(aten_op, torch._ops.OpOverloadPacket):
            # aten::sym_size is the only OverloadPacket that we support.
            # schema: aten::sym_size(Tensor self, int dim) -> Tensor
            if aten_op != torch.ops.aten.sym_size:
                raise ValueError(
                    f"Unsupported OverloadPacket: {aten_op}, aten.sym_size is the only allowed OverloadPacket!"
                )
            # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
            # overloadpacket for some reasons.
            # https://github.com/pytorch/pytorch/issues/97201
            aten_op_default = aten_op.default
            aten_name = aten_op_default.name()
        elif aten_op in _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE:
            # Make sure it's symint/symfloat consuming builtin ops.
            for node_arg in node.args:
                if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                    isinstance(node_arg, torch.fx.Node)
                    and not isinstance(
                        node_arg.meta["val"], (torch.SymInt, torch.SymFloat)
                    )
                ):
                    raise ValueError(
                        f"Unsupported node arg: {node_arg} with builtin function: {aten_op},"
                        " only int/float/SymInt/SymFloat is supported with built-in ops!"
                    )
            aten_op = _SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE[aten_op]
            aten_name = aten_op.name()
        elif isinstance(aten_op, torch._ops.OpOverload):
            aten_name = aten_op.name()
        else:
            raise RuntimeError(f"Unknown call_function target: {node.target}")

        func = self.registry.get(aten_name, None)
        if func is None:
            # Fall back default if there is no overload
            if isinstance(aten_op, torch._ops.OpOverload) and hasattr(
                aten_op.overloadpacket, "default"
            ):
                func = self.registry.get(aten_op.overloadpacket.default.name(), None)
            else:
                raise RuntimeError(
                    f"aten name: {aten_name} is not registered in the torchlib registry!"
                )

        # opschema algo
        # TODO: Cache the OpSchemaWrapper so we don't need to run the init logic everytime
        function_opschema = OpSchemaWrapper(func.op_schema)
        for f in func.overloads:
            if function_opschema.match_inputs(onnx_args, onnx_kwargs):
                return f
            if function_opschema.nearest_match_inputs(onnx_args, onnx_kwargs):
                return f
        raise RuntimeError(f"There are no overloaded function for {aten_name}")


fx_dispatcher = Dispatcher(registration.default_registry)


class OpSchemaWrapper:
    """A wrapper for ONNX OpSchema."""

    def __init__(self, op_schema: onnx.defs.OpSchema):
        self.schema = op_schema
        self.type_constraints = {
            # "T": {"tensor(int64)"}
            constraint.type_param_str: set(constraint.allowed_type_strs)
            for constraint in op_schema.type_constraints
        }

    def nearest_match_inputs(self, args, kwargs) -> bool:
        return False

    def match_inputs(
        self, args: Sequence[Union[WithDtype, str, int, float, bool]], kwargs
    ) -> bool:
        # TODO: Refine the logic for it to be more robust
        # TODO: Handle attributes
        for schema_input, torch_input in zip(self.schema.inputs, args):
            data_type = (
                torch_input.dtype
                if isinstance(torch_input, WithDtype)
                else type(torch_input)
            )
            torch_input_compatible_types = _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
                data_type
            ]
            allowed_types = self.type_constraints[schema_input]
            if not allowed_types.intersection(torch_input_compatible_types):
                # If none of the types in torch_input_compatible_types is in
                # allowed_types of this input defined in the OpSchema, we know
                # the function and the input are not compatible
                return False
        return True


_SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE: dict[
    Union[Callable[..., Any], str], torch._ops.OpOverload
] = {
    operator.mul: torch.ops.aten.mul.default,
    operator.add: torch.ops.aten.add.default,
    operator.pow: torch.ops.aten.pow.int,
    operator.sub: torch.ops.aten.sub.default,
}


@_beartype.beartype
def _create_onnx_friendly_decomposition_table() -> (
    Dict[torch._ops.OpOverload, Callable]
):
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():
        # Skip decomposition into "prim::*" ops (defined in 'torch._refs'), because they
        # are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload.name() in fx_dispatcher.registry
        ):
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table


# This is a subset of PyTorch's built-in aten-to-aten decomposition. If an aten
# op (e.g., torch.ops.aten.add.Tensor) has exporter, we exclude the op's decomposition
# function in the DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE.
DEFAULT_ONNX_EXPORTER_DECOMPOSITION_TABLE: Dict[
    torch._ops.OpOverload, Callable
] = _create_onnx_friendly_decomposition_table()
