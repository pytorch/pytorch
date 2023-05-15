from __future__ import annotations

import inspect

import operator

import re
import types
import warnings

from typing import Callable, Dict, List, Optional, Tuple, Union

import onnxscript  # type: ignore[import]
from onnxscript import evaluator, opset18  # type: ignore[import]
from onnxscript.function_libs.torch_lib import graph_building  # type: ignore[import]

import torch
import torch.fx
from torch._subclasses import fake_tensor

from torch.onnx import _type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.exporter import ResolvedExportOptions
from torch.onnx._internal.fx import diagnostics, function_dispatcher, op_validation
from torch.utils import _pytree


@_beartype.beartype
def _fx_node_to_onnx_message_formatter(
    fn: Callable,
    diagnostic_context: diagnostics.DiagnosticContext,
    node: torch.fx.Node,
    *args,
    **kwargs,
) -> str:
    return f"FX Node: {node.op}:{node.target}[name={node.name}]. "


def _location_from_fx_stack_trace(
    node_stack_trace: str,
) -> Optional[diagnostics.infra.Location]:
    """Extract location from FX node stack trace.

    TODO(bowbao): Create fx utils module and move this function there.

    Args:
        node_stack_trace: The stack trace of the FX node. Example:

            File "path/file.py", line 311, in <function>
                <code>
            |   File "path/file2.py", line 389, in <function>
                <code>

    Returns:
        location: The location of the FX node.
    """
    if "File" not in node_stack_trace:
        return None

    lines = node_stack_trace.strip().split("\n")
    idx = 0
    while idx < len(lines) and "File" not in lines[idx]:
        idx += 1
    if idx + 1 >= len(lines):
        return None

    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    matches = pattern.match(lines[idx].strip())
    if matches:
        uri = matches.group(1)
        line_number = int(matches.group(2))
        snippet = lines[idx + 1].strip()
        return diagnostics.infra.Location(uri=uri, line=line_number, snippet=snippet)
    return None


@_beartype.beartype
def _retrieve_or_adapt_input_to_graph_set(
    fx_node_arg: _type_utils.Argument,
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            graph_building.TorchScriptTensor,
            Tuple[graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: graph_building.TorchScriptTracingEvaluator,
):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """

    onnx_tensor = fx_node_arg
    if isinstance(onnx_tensor, torch.fx.Node):
        # 1. fx_node_arg is a torch.fx.Node, which means
        #    fx_node_arg stands for the output of that torch.fx.Node.
        # 2. fx_node_arg (variable in torch.fx.Graph) is be mapped to
        #    torch.jit.Value, fx_name_to_onnxscript_value[fx_node_arg.name],
        #    in TorchScript graph.
        return fx_name_to_onnxscript_value[onnx_tensor.name]
    if isinstance(onnx_tensor, (tuple, list)) and any(
        isinstance(node, torch.fx.Node) and isinstance(node.meta["val"], torch.SymInt)
        for node in onnx_tensor
    ):
        # This intends to handle dynamic axes. for example, if the input size of op.Expand
        # is dynamic, each dimension would be variable (i.e., sym variable in Pytorch
        # FX graph. Note that sym variable is mapped to tensor in ONNX Script world)
        # calculated by other operators.
        sequence_mixed_elements: List[
            Union[graph_building.TorchScriptTensor, List[int]]
        ] = []
        for tensor in onnx_tensor:
            if isinstance(tensor, torch.fx.Node) and isinstance(
                tensor.meta["val"], torch.SymInt
            ):
                sequence_mixed_elements.append(fx_name_to_onnxscript_value[tensor.name])
            elif isinstance(tensor, int):
                # NOTE: op.Concat doesn't support scalar, so we need to wrap it with
                # dim, and onnx-script will promote it to tensot(int64)
                sequence_mixed_elements.append([tensor])
        # Concat all the elements in the sequence.
        # shapes are mapped to tensors in ONNX graph (TorchScriptGraph),
        # so list of sym_ints is concatenated to a tensor before calling ONNX op.

        # For example:
        #    inputs: [[2], [4], fx.Node(SymIntA), [1], fx.Node(SymIntB)]
        #    outputs: op.Concat([op.Constant(2), op.Constant(4), TorchScriptTensor(A), op.Constant(1), TorchScriptTensor(B)])

        # onnx-script auto wraps python number with op.Constants,
        # so we don't need to specifically process them.
        with evaluator.default_as(tracer):
            return opset18.Concat(*sequence_mixed_elements, axis=0)
    elif isinstance(onnx_tensor, (tuple, list)) and all(
        isinstance(node, torch.fx.Node) for node in onnx_tensor
    ):
        sequence_elements: List[
            Union[
                graph_building.TorchScriptTensor,
                Tuple[graph_building.TorchScriptTensor, ...],
            ]
        ] = []
        for tensor in onnx_tensor:
            sequence_elements.append(fx_name_to_onnxscript_value[tensor.name])
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type())

    # all other cases, we do nothing.
    return onnx_tensor


def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
    filtered = {}
    for key, value in kwargs.items():
        if key in {
            "layout",
            "device",
            "requires_grad",
            "pin_memory",
            "memory_format",
            "implicit",
        }:
            continue
        if key == "dtype":
            if value is None:
                # We omit if dtype is not provided, because onnxscript handles the
                # default case.
                continue
            else:
                filtered["dtype"] = int(
                    _type_utils.JitScalarType.from_dtype(value).onnx_type()
                )
            continue
        filtered[key] = value
    return filtered


@_beartype.beartype
def _fill_tensor_meta(
    onnxscript_values: Union[
        graph_building.TorchScriptTensor, Tuple[graph_building.TorchScriptTensor, ...]
    ],
    name: str,
    expected_values: Union[
        fake_tensor.FakeTensor,
        torch.SymInt,
        torch.SymFloat,
        List[fake_tensor.FakeTensor],
        Tuple[fake_tensor.FakeTensor, ...],
    ],
):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""

    # NOTE(titaiwang): Type of expected_values is showing what we support right now.
    # Currently, we only expect FakeTensor and SymInt in the graph.

    # aten::sym_size output is a int, not a tensor, which stands
    # for the size of one dim. We treat it as 0-D tensor.
    if isinstance(expected_values, (torch.SymInt, torch.SymFloat)):
        return

    if isinstance(expected_values, (list, tuple)) and not isinstance(
        onnxscript_values, (list, tuple)
    ):
        return

    flat_onnxscript_values, _ = _pytree.tree_flatten(onnxscript_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for i, (onnxscript_value, expected_value) in enumerate(
        zip(flat_onnxscript_values, flat_expected_values)
    ):
        # We set node output sizes to be dynamic to continue the model conversion,
        # and inputs are also set to be dynamic in add_input().
        onnxscript_value.shape = tuple(
            [dim if isinstance(dim, int) else None for dim in expected_value.size()]
        )
        onnxscript_value.dtype = expected_value.dtype
        if i > 0:
            onnxscript_value.name = f"{name}_{i}"
        else:
            onnxscript_value.name = name


@_beartype.beartype
def _fill_in_default_kwargs(
    node: torch.fx.Node,
) -> Tuple[List[_type_utils.Argument], Dict[str, _type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""

    # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
    # overloadpacket for some reasons.
    # https://github.com/pytorch/pytorch/issues/97201
    # We manually assigned overload for aten::sym_size.
    if hasattr(node.target, "_schema"):
        node_schema = node.target._schema  # type: ignore[union-attr]
    else:
        node_schema = torch.ops.aten.sym_size.int._schema  # type: ignore[union-attr]

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args: List[_type_utils.Argument] = []
    complete_kwargs: Dict[str, _type_utils.Argument] = {}

    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for i, expected_arg in enumerate(node_schema.arguments):
            if i < len(node.args):
                complete_args.append(node.args[i])
            elif expected_arg.name in node.kwargs:
                complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
            else:
                # Get default from schema.
                complete_kwargs[expected_arg.name] = expected_arg.default_value

    return complete_args, complete_kwargs


@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(
    complete_args: List[_type_utils.Argument],
    complete_kwargs: Dict[str, _type_utils.Argument],
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            graph_building.TorchScriptTensor,
            Tuple[graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: graph_building.TorchScriptTracingEvaluator,
) -> Tuple[tuple, dict]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscript_value, tracer)
        for arg in complete_args
    )
    onnxscript_kwargs = filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    return onnxscript_args, onnxscript_kwargs


@_beartype.beartype
@diagnostics.diagnose_call(
    diagnostics.rules.fx_node_to_onnx,
    diagnostic_message_formatter=_fx_node_to_onnx_message_formatter,
)
def _export_fx_node_to_onnxscript(
    diagnostic_context: diagnostics.DiagnosticContext,
    node: torch.fx.Node,
    onnxscript_graph: graph_building.TorchScriptGraph,
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            graph_building.TorchScriptTensor,
            Tuple[graph_building.TorchScriptTensor, ...],
        ],
    ],
    tracer: graph_building.TorchScriptTracingEvaluator,
    fx_module_with_metadata: torch.fx.GraphModule,
    options: ResolvedExportOptions,
):
    # Record stack trace of node in diagnostic.
    node_stack_trace = node.stack_trace
    if node_stack_trace:
        diagnostic = diagnostic_context.inflight_diagnostic(
            rule=diagnostics.rules.fx_node_to_onnx
        )
        diagnostic.with_additional_message(
            f"### PyTorch source information\n```\n{node_stack_trace}\n```"
        )
        location = _location_from_fx_stack_trace(node_stack_trace)
        if location is not None:
            diagnostic.with_location(location)

    if node.op == "placeholder":
        # Input of graph.
        # The node.meta["val"] is generated by FakeTensorProp.
        # NOTE: add_input() intends to create nodes with shape/type
        fake_tensor = node.meta.get("val", None)
        if fake_tensor is None:
            output = onnxscript_graph.add_input(
                input_name=None,
            )
        else:
            output = onnxscript_graph.add_input(
                input_name=node.name,
                shape=fake_tensor.shape,
                dtype=fake_tensor.dtype,
            )
        assert (
            output is not None
        ), f"Node creates None with target={node.target} and name={node.name}"
        assert isinstance(output, graph_building.TorchScriptTensor)
        assert isinstance(output, onnxscript.tensor.Tensor)

        fx_name_to_onnxscript_value[node.name] = output
    elif node.op == "call_function":
        # aten ops and other stateless functions.
        if node.target == operator.getitem and isinstance(
            fx_name_to_onnxscript_value[node.args[0].name], tuple  # type: ignore[union-attr]
        ):
            onnx_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]  # type: ignore[union-attr]
            index = node.args[1]
            output = onnx_tensor_tuple[index]  # type: ignore[index]
            assert (
                output is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            assert isinstance(output, (graph_building.TorchScriptTensor, tuple)), type(
                output
            )

            fx_name_to_onnxscript_value[node.name] = output
            return
        if node.target == operator.getitem:
            # __getitem__ on Tensor or Sequence of tensors. Not tuple.
            exporter_key = "getitem"
        elif (
            isinstance(node.target, types.BuiltinFunctionType)
            and node.target
            in function_dispatcher._SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE
        ):
            for node_arg in node.args:
                if (not isinstance(node_arg, (torch.fx.Node, int, float))) or (
                    isinstance(node_arg, torch.fx.Node)
                    and not isinstance(
                        node_arg.meta["val"], (torch.SymInt, torch.SymFloat)
                    )
                ):
                    raise ValueError(
                        f"Unsupported node arg: {node_arg} with builtin function: {node.target},"
                        " only int/float/SymInt/SymFloat is supported with built-in ops!"
                    )

            # symbolic fx.graph contains built-in functions to calculate python values.
            exporter_key = function_dispatcher._SYMINT_SYMFLOAT_BUILTIN_TO_EXPORTER_KEY_TABLE[
                node.target  # type: ignore[index]
            ]
        elif (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target in function_dispatcher._OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
        ):
            exporter_key = function_dispatcher._OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[
                node.target
            ]
        elif isinstance(node.target, torch._ops.OpOverloadPacket):
            # aten::sym_size is the only OverloadPacket that we support.
            # schema: aten::sym_size(Tensor self, int dim) -> Tensor
            if node.target != torch.ops.aten.sym_size:
                raise ValueError(
                    f"Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!"
                )
            # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
            # overloadpacket for some reasons.
            # https://github.com/pytorch/pytorch/issues/97201
            # We manually assigned overload for aten::sym_size.
            exporter_key = function_dispatcher._OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[
                torch.ops.aten.sym_size.int
            ]
        else:
            raise RuntimeError(f"Unknown call_function target: {node.target}")
        # Only the latest opset version is only supported in atenlib for now
        symbolic_fn = function_dispatcher._ATENLIB_FUNCTIONS.get(exporter_key)
        if symbolic_fn is None:
            raise RuntimeError(f"Cannot find function for {exporter_key}")
        # Map FX inputs to ONNX inputs and fill optional inputs with default values.
        # torch_args and torch_kwargs are for op-level validation
        complete_args, complete_kwargs = _fill_in_default_kwargs(node)
        onnx_args, onnx_kwargs = _wrap_fx_args_as_onnxscript_args(
            complete_args, complete_kwargs, fx_name_to_onnxscript_value, tracer
        )
        with evaluator.default_as(tracer):
            output: Union[  # type: ignore[no-redef]
                graph_building.TorchScriptTensor,
                Tuple[graph_building.TorchScriptTensor, ...],
            ] = symbolic_fn(*onnx_args, **onnx_kwargs)
        assert (
            output is not None
        ), f"Node creates None with target={node.target}, name={node.name}, args={onnx_args}, kwargs={onnx_kwargs}"
        # Assign type and shape from fx graph.
        _fill_tensor_meta(output, node.name, node.meta["val"])
        # One fx node could produce multiple outputs (e.g., tuple of tensors); in
        # that case, v is a tuple of TorchScriptTensors.
        assert isinstance(output, (graph_building.TorchScriptTensor, tuple)), type(
            output
        )
        # NOTE(titaiwang): We bypass two kinds of ops as it's not meaningful to
        # validate them with op level debug.
        # 1. aten::sym_size: The op is simply get item from a list of tensors.
        # 2. BuiltinFunction: It doesn't supported tensor
        if (
            options.op_level_debug
            and node.target != torch.ops.aten.sym_size
            and not isinstance(node.target, types.BuiltinFunctionType)
        ):
            (
                node_with_fixed_shape_args,
                node_with_fixed_shape_kwargs,
            ) = _fill_in_default_kwargs(node)
            try:
                torch_args, torch_kwargs = op_validation.wrap_fx_args_as_torch_args(
                    node_with_fixed_shape_args, node_with_fixed_shape_kwargs
                )
            except ValueError as value_error:
                warnings.warn(
                    f"\nFound unsupported input types on PyTorch Op {node.target} with "
                    f"ValueError: \n{value_error}.\n"
                )
                diagnostic = diagnostic_context.inflight_diagnostic()
                diagnostic.with_additional_message(
                    f"### Op level debug fails due to unsupported input types\n"
                    f"{diagnostics.decorator.format_exception_in_markdown(value_error)}"
                )
                diagnostic.level = diagnostics.levels.ERROR
            else:
                op_validation.validate_op_between_ort_torch(
                    diagnostic_context, node, symbolic_fn, torch_args, torch_kwargs
                )
        fx_name_to_onnxscript_value[node.name] = output
    elif node.op == "output":
        if isinstance(node.args[0], torch.fx.Node):
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
            # tensor, etc), we flatten the collection and register each element as output.
            flat_args, _ = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(
                    arg, torch.fx.Node
                ), f"arg must be a torch.fx.Node, not {type(arg)}"
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscript_value[arg.name]
                onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
    elif node.op == "call_method":
        # TODO(wechi): Support call_method.
        raise RuntimeError("call_method is not supported yet.")
    elif node.op == "call_module":
        # TODO(wechi): Support call_module.
        raise RuntimeError("call_module is not supported yet.")
    elif node.op == "get_attr":
        current_attr = fx_module_with_metadata
        sub_attr_names = node.target.split(".")  # type: ignore[union-attr]
        # If node.targe is "conv.weight", the following loop first
        # assigns fx_module_with_metadata.conv to current_attr, and then
        # fx_module_with_metadata.conv.weight to current_attr.
        while sub_attr_names:
            sub_attr_name = sub_attr_names.pop(0)
            if not hasattr(current_attr, sub_attr_name):
                raise AttributeError(
                    f"Attribute {sub_attr_name} is not found in {current_attr}."
                )
            current_attr = getattr(current_attr, sub_attr_name)

        input_ = onnxscript_graph.add_initializer(node.name, current_attr)

        assert isinstance(input_, graph_building.TorchScriptTensor)
        assert isinstance(input_, onnxscript.tensor.Tensor)
        fx_name_to_onnxscript_value[node.name] = input_
        # FIXME: Refactor logic getting 'current_attr'.
        assert isinstance(current_attr, torch.Tensor)
    else:
        # TODO(wechi): Support get_attr, call_module, call_method.
        raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")


@_beartype.beartype
@diagnostics.diagnose_call(diagnostics.rules.atenlib_fx_to_onnx)
def export_fx_to_onnxscript(
    diagnostic_context: diagnostics.DiagnosticContext,
    fx_module_with_metadata: torch.fx.GraphModule,
    options: ResolvedExportOptions,
):
    """
    Export a torch.fx.GraphModule to a TorchScript graph with ONNX symbols.

    Args:
        fx_module_with_metadata: A torch.fx.GraphModule with metadata.
        export_options: Export options.

    Returns:
        A TorchScript graph with ONNX symbols.
    """
    # Initialize the ONNX graph
    onnxscript_graph = graph_building.TorchScriptGraph()
    tracer = graph_building.TorchScriptTracingEvaluator(onnxscript_graph)

    # In the following loop, a TorchScript graph is created to
    # represent the input FX graph with ONNX symbols (e.g., onnx::add).
    # To connect the values to nodes in the TorchScript graph, we maintian
    # fx_name_to_onnxscript_value. Basically, we want to translate
    #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
    # to
    #   fx_name_to_onnxscript_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_onnxscript_value[fx_tensor_y.name]
    fx_name_to_onnxscript_value: Dict[
        str,
        Union[
            graph_building.TorchScriptTensor,
            Tuple[graph_building.TorchScriptTensor, ...],
        ],
    ] = {}
    # node_fixed_shape is only used on op_level_debug purpose.
    for node in fx_module_with_metadata.graph.nodes:
        _export_fx_node_to_onnxscript(
            diagnostic_context,
            node,
            onnxscript_graph,
            fx_name_to_onnxscript_value,
            tracer,
            fx_module_with_metadata,
            options,
        )

    return onnxscript_graph
