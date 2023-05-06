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
    ops,  # noqa: F401
    registration,  # type: ignore[import]
)

import torch
import torch.fx
from torch.onnx import _type_utils
from torch.onnx._internal import _beartype


_SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE: dict[
    Union[Callable[..., Any], str], torch._ops.OpOverload
] = {
    operator.mul: torch.ops.aten.mul.default,
    operator.add: torch.ops.aten.add.default,
    operator.pow: torch.ops.aten.pow.int,
    operator.sub: torch.ops.aten.sub.default,
}


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> torch.dtype:
        ...


class Dispatcher:
    """
    The Dispatcher class is responsible for finding the best matched function for a given aten op in the FX exporter.

    Steps for function dispatch:

    1. Use the torch.ops name to find the function:
        a. Check if aten:stft.center exists.
        b. If not found, check aten:stft (Fallback to default).

    2. Find the best match among all overloaded functions:
        a. If the types match, select the function.
        b. If the types don't match, find the nearest upcast/downcast:
            - Float to Float / Int to Int
            - Int to Float
    """

    def __init__(self, registry: registration.Registry):
        """Initialize the Dispatcher.

        Args:
            registry: The registration registry.
        """
        self._registry = registry

    @property
    def registry(self) -> registration.Registry:
        """Get the registration registry.

        Returns:
            The registration registry.
        """
        return self._registry

    def dispatch(
        self,
        node: torch.fx.Node,
        onnx_args: Sequence[graph_building.TorchScriptTensor],
        onnx_kwargs: Mapping[str, onnxscript.values.Value],
    ) -> Union[onnxscript.values.ONNXFunction, onnxscript.values.TracedONNXFunction]:
        """Dispatch a function based on the given node, arguments, and keyword arguments.

        Args:
            node: The torch.fx.Node.
            onnx_args: The arguments of the function.
            onnx_kwargs: The keyword arguments of the function.

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

        if self.registry.is_registered(aten_name):
            func = self.registry.get_functions(aten_name)
        elif hasattr(aten_op, "overloadpacket") and self.registry.is_registered(
            aten_op.overloadpacket._qualified_op_name
        ):
            # Fall back to overloadpacket name: eg: aten.add.Tensor -> aten::add
            func = self.registry.get_functions(
                aten_op.overloadpacket._qualified_op_name
            )
        else:
            raise RuntimeError(
                f"aten name: {aten_name} is not registered in the torchlib registry!"
            )

        # TODO(titaiwang): OrderedDict to record function matching score.
        if len(func.overloads) == 1:
            return func.overloads[0]
        # TODO: Cache the OpSchemaWrapper so we don't need to run the init logic everytime
        for overload_func in func.overloads:
            function_opschema = OpSchemaWrapper(overload_func.op_schema)
            if function_opschema.match_inputs(onnx_args, onnx_kwargs):
                return overload_func
            if function_opschema.nearest_match_inputs(onnx_args, onnx_kwargs):
                return overload_func
        raise RuntimeError(f"There are no overloaded function for {aten_name}")


# TODO(titaiwang): Need converter registry
fx_dispatcher = Dispatcher(registration.default_registry)


class OpSchemaWrapper:
    """
    The OpSchemaWrapper class is a wrapper for ONNX OpSchema.

    It provides methods to check for input compatibility based on the OpSchema.

    Attributes:
        schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.
    """

    def __init__(self, op_schema: onnx.defs.OpSchema):
        """Initialize the OpSchemaWrapper.

        Args:
            op_schema: The ONNX OpSchema.
        """
        self.schema = op_schema
        self.type_constraints = {
            # "T": {"tensor(int64)"}
            constraint.type_param_str: set(constraint.allowed_type_strs)
            for constraint in op_schema.type_constraints
        }

    def nearest_match_inputs(self, args, kwargs) -> bool:
        """Check if there is a nearest match for the inputs.

        This method is not implemented yet and always returns False.

        The function should be able to find the nearest matched overload
        for following cases:

        1. onnx_args with no dtype information
        2. onnx_args with dtype information but not match the OpSchema (autocast)

        Args:
            args: The arguments.
            kwargs: The keyword arguments.

        Returns:
            False.
        """
        return False

    def match_inputs(
        self, args: Sequence[Union[TensorLike, str, int, float, bool]], kwargs
    ) -> bool:
        """Check if the inputs match the OpSchema requirements.

        Args:
            args: The input arguments.
            kwargs: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        # TODO: Refine the logic for it to be more robust
        # TODO: Handle attributes
        for schema_input, torch_input in zip(self.schema.inputs, args):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if not allowed_types.intersection(torch_input_compatible_types):
                # If none of the types in torch_input_compatible_types is in
                # allowed_types of this input defined in the OpSchema, we know
                # the function and the input are not compatible
                return False
        return True


def _find_onnx_data_type(
    torch_input: Union[TensorLike, str, int, float, bool]
) -> set[str]:
    """Find the data type of the input."""
    if isinstance(torch_input, TensorLike):
        dtype = torch_input.dtype
        if isinstance(dtype, set):
            # NOTE: dtype is a sequence_dtype_string, eg: seq(tensor(int64))
            # torch doesn't support, but onnx does
            return dtype
        return _type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
            torch_input.dtype
        ]
    if isinstance(torch_input, (int, float, bool, str)):
        return _type_utils.TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[
            type(torch_input)
        ]
    if isinstance(torch_input, (list, tuple)) and torch_input:
        set_dtype = _find_onnx_data_type(torch_input[0])
        if any(isinstance(input, TensorLike) for input in torch_input):
            # NOTE: Any Tensor involved in a list would make it a seq(tensor(onnx_type))
            return {f"seq({dtype})" for dtype in set_dtype}
        else:
            # constant list of non-tensor type
            return set_dtype
    raise RuntimeError(f"Unknown input type from input: {torch_input}")


def _create_op_overload_to_exporter_key_table() -> (
    Mapping[Union[torch._ops.OpOverload, Callable], str]
):
    # TODO(justinchuby): Improve how the table is constructed.
    table: Dict[Union[torch._ops.OpOverload, Callable], str] = {}

    # Some ops in `torch.ops.aten` are not discoverable through `dir(torch.ops.aten)`,
    # but retrievable via explicit lookup.
    # https://github.com/pytorch/pytorch/issues/99681
    # This is a workaround to make sure we register ONNX symbolic functions for these.
    onnx_supported_aten_lookup_table = [
        k.split("::")[1].split(".")[0]
        for k in registration.default_registry
        if k.startswith("aten::")
    ]

    for op_namespace in (torch.ops.aten, torch.ops.prims):
        attr_names = dir(op_namespace)
        if op_namespace is torch.ops.aten:
            attr_names += onnx_supported_aten_lookup_table
        for attr_name in attr_names:
            if not hasattr(op_namespace, attr_name):
                # torchlib owns some attributes that are not aten ops.
                continue
            op_overload_packet = getattr(op_namespace, attr_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            exporter_look_up_key = op_overload_packet._qualified_op_name
            # TODO(titaiwang): registry API
            if (
                registration.default_registry.get_functions(exporter_look_up_key)
                is None
            ):
                # This aten op doesn't have ONNX exporter.
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                # This line maps torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar, torch.ops.aten.add.out, etc
                # to "aten::add". This means the exporter for "aten::add" is used for all overloads of "aten::add".
                # This is applied to all ops under torch.ops.aten.
                #
                # TODO(wechi): in the future, we might want to write individual exporter for each overload, if,
                # for example, they have different type promotion rules. If so, just map different overloads to
                # different exporter keys.
                table[op_overload] = op_overload_packet._qualified_op_name
    return table


# Dictionary that maps torch.ops.aten.* to exporter look up key; e.g.,
# _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.add.Tensor] is "aten::add".
_OP_OVERLOAD_TO_EXPORTER_KEY_TABLE = _create_op_overload_to_exporter_key_table()


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
            or op_overload in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
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
