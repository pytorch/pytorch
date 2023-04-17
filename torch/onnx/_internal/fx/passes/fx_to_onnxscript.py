from __future__ import annotations

import inspect

import operator

import re
import types
import warnings
from types import FunctionType

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import onnxscript  # type: ignore[import]
from onnxscript import evaluator, opset18  # type: ignore[import]
from onnxscript.function_libs.torch_aten import graph_building  # type: ignore[import]

import torch
import torch.fx
from torch._subclasses import fake_tensor

from torch.onnx import _type_utils
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.onnx._internal.fx import diagnostics, function_dispatcher, options
from torch.utils import _pytree


def _onnx_function_diagnose_call_message_formatter(
    fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> str:
    if len(args) > 0 and isinstance(args[0], onnxscript.OnnxFunction):
        onnx_function: onnxscript.OnnxFunction = args[0]  # self
        return f"{onnx_function.name}: {onnxscript.OnnxFunction}"
    return f"{fn.__name__}: {fn}"


def _onnx_function_diagnose_call_append_symbolic_source_location(
    diagnostic: diagnostics.infra.Diagnostic,
    fn: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    return_values: Any,
) -> None:
    # TODO(bowbao): Record source location of symbolic.
    # Need this separate step because normally only the source location of
    # class `onnxscript.OnnxFunction.__call__` is recorded.
    pass


# TODO(bowbao): Delete this once diagnostics is introduced in onnxscript.
_diagnose_onnx_function = diagnostics.diagnose_call(
    rule=diagnostics.rules.atenlib_symbolic_function,
    diagnostic_message_formatter=_onnx_function_diagnose_call_message_formatter,
    diagnostic_modifier=_onnx_function_diagnose_call_append_symbolic_source_location,
)
for key, onnx_function in function_dispatcher._ATENLIB_FUNCTIONS.items():
    if isinstance(onnx_function, FunctionType):
        function_dispatcher._ATENLIB_FUNCTIONS[key] = _diagnose_onnx_function(
            onnx_function
        )
onnxscript.OnnxFunction.__call__ = _diagnose_onnx_function(
    onnxscript.OnnxFunction.__call__
)


@_beartype.beartype
def _fx_node_to_onnx_message_formatter(
    fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> str:
    assert len(args) > 0
    node = args[0]
    assert isinstance(node, torch.fx.Node)
    return f"FX Node: {node.op}:{node.target}[name={node.name}]"


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
    fx_name_to_onnxscipt_value: Dict[
        str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]
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
        #    torch.jit.Value, fx_name_to_onnxscipt_value[fx_node_arg.name],
        #    in TorchScript graph.
        return fx_name_to_onnxscipt_value[onnx_tensor.name]
    if isinstance(onnx_tensor, (tuple, list)) and any(
        isinstance(node, torch.fx.Node) and isinstance(node.meta["val"], torch.SymInt)
        for node in onnx_tensor
    ):
        # This intends to handle dynamic axes. for example, if the input size of op.Expand
        # is dynamic, each dimension would be variable (i.e., sym variable in Pytorch
        # FX graph. Note that sym variable is mapped to tensor in ONNX Script world)
        # calculated by other operators.
        sequence_elements: List[Union[torch.Value, Tuple[torch._C.Value, ...]]] = []
        for tensor in onnx_tensor:
            if isinstance(tensor, torch.fx.Node) and isinstance(
                tensor.meta["val"], torch.SymInt
            ):
                sequence_elements.append(fx_name_to_onnxscipt_value[tensor.name])
            else:
                sequence_elements.append(tensor)
        # Concat all the elements in the sequence.
        # shapes are mapped to tensors in ONNX graph (TorchScriptGraph),
        # so list of sym_ints is concatenated to a tensor before calling ONNX op.

        # For example:
        #    inputs: [2, 4, fx.Node(SymIntA), 1, fx.Node(SymIntB)]
        #    outputs: op.Concat([op.Constant(2), op.Constant(4), TorchScriptTensor(A), op.Constant(1), TorchScriptTensor(B)])

        # onnx-script auto wraps python number with op.Constants,
        # so we don't need to specifically process them.
        with evaluator.default_as(tracer):
            return opset18.Concat(*sequence_elements, axis=0)
    elif isinstance(onnx_tensor, (tuple, list)) and all(
        isinstance(node, torch.fx.Node) for node in onnx_tensor
    ):
        sequence_elements: List[Union[torch.Value, Tuple[torch._C.Value, ...]]] = []  # type: ignore[no-redef]
        for tensor in onnx_tensor:
            sequence_elements.append(fx_name_to_onnxscipt_value[tensor.name])
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type())

    # all other cases, we do nothing.
    return onnx_tensor


def _filter_incompatible_and_dtype_convert_kwargs(kwargs):
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
                filtered["dtype"] = -1
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
        # TODO(titaiwang): How to annotate type from sequence_type of ONNX?
        # eg: aten_split
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
        # TODO(titaiwang): This could break in terms of lacking of type_promotion now.
        onnxscript_value.dtype = expected_value.dtype
        if i > 0:
            onnxscript_value.name = f"{name}_{i}"
        else:
            onnxscript_value.name = name


@_beartype.beartype
def _wrap_fx_args_as_torch_args(
    complete_args: List[_type_utils.Argument],
    complete_kwargs: Dict[str, _type_utils.Argument],
) -> Tuple[tuple, dict]:
    """Prepare torch format args and kwargs for op-level validation Use fake tensor to create real tensor to feed in ops"""

    # NOTE: This function only supports FakeTensor with concrete shapes
    torch_args: List[_type_utils.Argument] = []
    for arg in complete_args:
        if isinstance(arg, torch.fx.Node):
            with torch.utils._mode_utils.no_dispatch():
                # eg: aten_where needs BOOL in input_args
                # fx_name_to_onnxscipt_value could help?
                fake_tensor = arg.meta["val"]
                if isinstance(fake_tensor, list):
                    for meta_value in fake_tensor:
                        shape = meta_value.shape
                        dtype = (
                            meta_value.dtype
                            if meta_value.dtype
                            not in {torch.int64, torch.int32, torch.uint8, torch.bool}
                            else torch.float32
                        )
                        torch_args.append(torch.randn(meta_value.shape, dtype=dtype))
                elif isinstance(fake_tensor, torch.Tensor):
                    shape = fake_tensor.shape
                    dtype = (
                        fake_tensor.dtype
                        if fake_tensor.dtype
                        not in {torch.int64, torch.int32, torch.uint8, torch.bool}
                        else torch.float32
                    )
                    if not isinstance(fake_tensor, torch.Tensor):
                        # aten_sym_size output is a int, not a tensor,
                        # which stands for the size of one dim.
                        torch_args.append(fake_tensor)
                    else:
                        torch_args.append(torch.randn(shape, dtype=dtype))
        else:
            torch_args.append(arg)
    torch_kwargs = complete_kwargs
    return (tuple(torch_args), torch_kwargs)


@_beartype.beartype
def _wrap_fx_args_as_onnxscript_args(
    node: torch.fx.Node,
    fx_name_to_onnxscipt_value: Dict[
        str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]
    ],
    tracer: graph_building.TorchScriptTracingEvaluator,
) -> Tuple[tuple, dict, list, dict]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    # (1) Complete the arguments with default values.
    complete_args: List[_type_utils.Argument] = []
    complete_kwargs: Dict[str, _type_utils.Argument] = {}

    # TODO(titaiwang): aten::sym_size has overload, but fx graph is using
    # overloadpacket for some reasons.
    # https://github.com/pytorch/pytorch/issues/97201
    # We manually assigned overload for aten::sym_size.
    if hasattr(node.target, "_schema"):
        node_schema = node.target._schema  # type: ignore[union-attr]
    else:
        node_schema = torch.ops.aten.sym_size.int._schema  # type: ignore[union-attr]

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
    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscipt_value, tracer)
        for arg in complete_args
    )
    onnxscript_kwargs = _filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    return (onnxscript_args, onnxscript_kwargs, complete_args, complete_kwargs)


@_beartype.beartype
@diagnostics.diagnose_call(
    rule=diagnostics.rules.fx_node_to_onnx,
    exception_report_level=diagnostics.levels.ERROR,
    diagnostic_message_formatter=_fx_node_to_onnx_message_formatter,
)
def _export_fx_node_to_onnxscript(
    node: torch.fx.Node,
    onnxscript_graph: graph_building.TorchScriptGraph,
    fx_name_to_onnxscipt_value: Dict[
        str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]
    ],
    tracer: graph_building.TorchScriptTracingEvaluator,
    fx_module_with_metadata: torch.fx.GraphModule,
    options: options.ExportOptions,
):
    # Record stack trace of node in diagnostic.
    node_stack_trace = node.stack_trace
    if node_stack_trace:
        diagnostic = diagnostics.export_context().inflight_diagnostic(
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

        fx_name_to_onnxscipt_value[node.name] = output
    elif node.op == "call_function":
        # aten ops and other stateless functions.
        if node.target == operator.getitem and isinstance(
            fx_name_to_onnxscipt_value[node.args[0].name], tuple  # type: ignore[union-attr]
        ):
            onnx_tensor_tuple = fx_name_to_onnxscipt_value[node.args[0].name]  # type: ignore[union-attr]
            index = node.args[1]
            output = onnx_tensor_tuple[index]  # type: ignore[index]
            assert (
                output is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            assert isinstance(output, (graph_building.TorchScriptTensor, tuple)), type(
                output
            )

            fx_name_to_onnxscipt_value[node.name] = output
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
        (
            onnx_args,
            onnx_kwargs,
            complete_args,
            complete_kwargs,
        ) = _wrap_fx_args_as_onnxscript_args(node, fx_name_to_onnxscipt_value, tracer)
        with evaluator.default_as(tracer):
            output: Union[  # type: ignore[no-redef]
                graph_building.TorchScriptTensor,
                Tuple[graph_building.TorchScriptTensor, ...],
            ] = symbolic_fn(*onnx_args, **onnx_kwargs)
        assert (
            output is not None
        ), f"Node creates None with target={node.target}, name={node.name}, args={onnx_args}, kwargs={onnx_kwargs}"
        # TODO(justinchuby): Add diagnostic information.
        # Assign type and shape obtained from FakeTensorProp.
        _fill_tensor_meta(output, node.name, node.meta["val"])
        # One fx node could produce multiple outputs (e.g., tuple of tensors); in
        # that case, v is a tuple of TorchScriptTensors.
        assert isinstance(output, (graph_building.TorchScriptTensor, tuple)), type(
            output
        )
        if options.op_level_debug:
            # FIXME(titaiwang): Refactor me with
            # https://github.com/microsoft/onnx-script/issues/393
            (torch_args, torch_kwargs) = _wrap_fx_args_as_torch_args(
                complete_args, complete_kwargs
            )
            _validate_op_between_ort_torch(node, symbolic_fn, torch_args, torch_kwargs)
        fx_name_to_onnxscipt_value[node.name] = output
    elif node.op == "output":
        if isinstance(node.args[0], torch.fx.Node):
            onnx_tensor_or_tensor_tuple = fx_name_to_onnxscipt_value[node.args[0].name]
            onnxscript_graph.register_outputs(onnx_tensor_or_tensor_tuple)
        else:
            # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
            # tensor, etc), we flatten the collection and register each element as output.
            flat_args, _ = _pytree.tree_flatten(node.args[0])
            for arg in flat_args:
                assert isinstance(
                    arg, torch.fx.Node
                ), f"arg must be a torch.fx.Node, not {type(arg)}"
                onnx_tensor_or_tensor_tuple = fx_name_to_onnxscipt_value[arg.name]
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
        fx_name_to_onnxscipt_value[node.name] = input_
        # FIXME: Refactor logic getting 'current_attr'.
        assert isinstance(current_attr, torch.Tensor)
    else:
        # TODO(wechi): Support get_attr, call_module, call_method.
        raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")


@diagnostics.diagnose_call(diagnostics.rules.atenlib_fx_to_onnx)
def export_fx_to_onnxscript(
    fx_module_with_metadata: torch.fx.GraphModule, options: options.ExportOptions
):
    # Initialize the ONNX graph
    onnxscript_graph = graph_building.TorchScriptGraph()
    tracer = graph_building.TorchScriptTracingEvaluator(onnxscript_graph)

    # In the following loop, a TorchScript graph is created to
    # represent the input FX graph with ONNX symbols (e.g., onnx::add).
    # To connect the values to nodes in the TorchScript graph, we maintian
    # fx_name_to_onnxscipt_value. Basically, we want to translate
    #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
    # to
    #   fx_name_to_onnxscipt_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_onnxscipt_value[fx_tensor_y.name]
    fx_name_to_onnxscipt_value: Dict[
        str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]
    ] = {}
    for node in fx_module_with_metadata.graph.nodes:
        _export_fx_node_to_onnxscript(
            node,
            onnxscript_graph,
            fx_name_to_onnxscipt_value,
            tracer,
            fx_module_with_metadata,
            options,
        )

    return onnxscript_graph


@_beartype.beartype
def _validate_op_between_ort_torch(
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, Callable],
    torch_args: tuple,
    torch_kwargs: dict,
):
    """Validate the op between ONNX Runtime and PyTorch."""
    # op-level validation
    # Symbolic_fn should have the same output as node.target (torch ops)
    # trace_only function is regular python function
    function_name = (
        symbolic_fn.name
        if isinstance(symbolic_fn, onnxscript.OnnxFunction)
        else symbolic_fn.__name__
    )
    try:
        with evaluator.default_as(evaluator.ort_evaluator):
            expected_outputs = node.target(*torch_args, **torch_kwargs)  # type: ignore[operator]
            # TODO(titaiwang): Expose _convert_tensor_to_numpy and _convert_kwargs_for_onnx?
            input_onnx = [
                onnx_proto_utils._convert_tensor_to_numpy(x) for x in torch_args
            ]
            kwargs_onnx = _filter_incompatible_and_dtype_convert_kwargs(torch_kwargs)
            ort_outputs = symbolic_fn(*input_onnx, **kwargs_onnx)

            # TODO: add pytree structure comparison.
            flattened_torch_outputs, _ = _pytree.tree_flatten(expected_outputs)
            flattened_function_outputs, _ = _pytree.tree_flatten(ort_outputs)

            assert flattened_torch_outputs
            assert len(flattened_torch_outputs) == len(flattened_function_outputs)

            for torch_output, function_output in zip(
                flattened_torch_outputs, flattened_function_outputs
            ):
                try:
                    if not isinstance(function_output, np.ndarray):
                        # An onnxscript tensor
                        function_output = function_output.value

                    # Use torch.testing as opposed to np.testing to ensure dtypes and shapes match
                    torch.testing.assert_close(
                        torch.tensor(function_output).cpu(),
                        torch_output.cpu()
                        if isinstance(torch_output, torch.Tensor)
                        else torch.tensor(torch_output).cpu(),
                        rtol=1e-4,
                        atol=1e-3,
                    )

                except AssertionError as e:
                    warnings.warn(
                        f"\nSuppressed AssertionError:\n{e}.\n"
                        f"Op {node.target} has mismatch outputs. "
                        f"Please check the implementation of {function_name}.\n"
                    )
                    diagnostic = diagnostics.export_context().inflight_diagnostic()
                    diagnostic.with_additional_message(
                        f"### Validation failed\n"
                        f"{diagnostics.decorator.format_exception_in_markdown(e)}"
                    )
                    diagnostic.level = diagnostics.levels.ERROR
    except Exception as e:
        warnings.warn(
            f"\nORT fails to run on Op {node.target} with error: \n{e}.\n"
            f"Please check the implementation of {function_name}.\n"
        )
        diagnostic = diagnostics.export_context().inflight_diagnostic()
        diagnostic.with_additional_message(
            f"### Validation failed\n"
            f"{diagnostics.decorator.format_exception_in_markdown(e)}"
        )
        diagnostic.level = diagnostics.levels.WARNING
