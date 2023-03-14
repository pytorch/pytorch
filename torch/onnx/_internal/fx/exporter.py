from __future__ import annotations

import copy
import inspect
import itertools
import operator
import re
import warnings
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
from onnxscript.function_libs.torch_aten import graph_building  # type: ignore[import]

import torch
import torch._C
import torch._decomp
import torch._dynamo
import torch._ops
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.passes import fake_tensor_prop
from torch.nn.utils import stateless
from torch.onnx import _constants, _type_utils

from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.onnx._internal.fx import diagnostics, function_dispatcher, options
from torch.utils import _pytree

# TODO: Separate into individual components.
# TODO: make_fx lose stack info https://github.com/pytorch/pytorch/issues/90276


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


def _retrieve_or_adapt_input_to_graph_set(fx_node_arg, fx_name_to_onnxscipt_value):
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
        onnx_tensor = fx_name_to_onnxscipt_value[onnx_tensor.name]
    elif isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type())

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


def _wrap_fx_args_as_onnxscript_args(
    node: torch.fx.Node,
    fx_name_to_onnxscipt_value: Dict[
        str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]
    ],
) -> Tuple[tuple, dict, tuple, dict]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    # (1) Complete the arguments with default values.
    complete_args: List[Any] = []
    complete_kwargs: Dict[str, Any] = {}
    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for i, expected_arg in enumerate(node.target._schema.arguments):  # type: ignore[union-attr]
            if i < len(node.args):
                complete_args.append(node.args[i])
            else:
                if expected_arg.name in node.kwargs:
                    complete_kwargs[expected_arg.name] = node.kwargs[expected_arg.name]
                else:
                    # Get default from schema.
                    complete_kwargs[expected_arg.name] = expected_arg.default_value

    onnxscript_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscipt_value)
        for arg in complete_args
    )
    onnxscript_kwargs = _filter_incompatible_and_dtype_convert_kwargs(complete_kwargs)

    # prepare torch format args and kwargs for op-level validation
    # Use fake tensor to create real tensor to feed in ops
    torch_args = []
    for arg in complete_args:
        if isinstance(arg, torch.fx.Node):
            # Create a concreate test tensor based on the fake tensor
            with torch.utils._mode_utils.no_dispatch():
                # TODO(titaiwang): The assumption of torch.float might not be true, eg: aten_where needs BOOL in input_args
                # fx_name_to_onnxscipt_value could help?
                if isinstance(arg.meta["val"], list):
                    for meta_value in arg.meta["val"]:
                        torch_args.append(
                            torch.randn_like(meta_value, dtype=torch.float)
                        )
                else:
                    torch_args.append(
                        torch.randn_like(arg.meta["val"], dtype=torch.float)
                    )
        else:
            torch_args.append(arg)
    torch_kwargs = complete_kwargs
    return (onnxscript_args, onnxscript_kwargs, tuple(torch_args), torch_kwargs)


def _fill_tensor_meta(
    onnxscript_values,
    name: str,
    expected_values: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""
    flat_onnxscript_values, _ = _pytree.tree_flatten(onnxscript_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for i, (onnxscript_value, expected_value) in enumerate(
        zip(flat_onnxscript_values, flat_expected_values)
    ):
        # Only set shape for now as we don't need type information.
        onnxscript_value.shape = tuple(expected_value.size())
        if i > 0:
            onnxscript_value.name = f"{name}_{i}"
        else:
            onnxscript_value.name = name


def _location_from_fx_stack_trace(
    node_stack_trace: str,
) -> Optional[diagnostics.infra.Location]:
    """Extract location from FX node stack trace.

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
def _fx_node_to_onnx_message_formatter(
    fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> str:
    assert len(args) > 0
    node = args[0]
    assert isinstance(node, torch.fx.Node)
    return f"FX Node: {node.op}:{node.target}[name={node.name}]"


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
        output = onnxscript_graph.add_input(
            input_name=node.name,
            # The node.meta["val"] is generated by FakeTensorProp.
            input_value=node.meta["val"],
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
            isinstance(node.target, torch._ops.OpOverload)
            and node.target in function_dispatcher._OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
        ):
            exporter_key = function_dispatcher._OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[
                node.target
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
            torch_args,
            torch_kwargs,
        ) = _wrap_fx_args_as_onnxscript_args(node, fx_name_to_onnxscipt_value)
        with evaluator.default_as(tracer):
            output: Union[  # type: ignore[no-redef]
                graph_building.TorchScriptTensor,
                Tuple[graph_building.TorchScriptTensor],
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

        input_ = onnxscript_graph.add_input(
            input_name=node.name, input_value=current_attr
        )
        assert isinstance(input_, graph_building.TorchScriptTensor)
        assert isinstance(input_, onnxscript.tensor.Tensor)
        fx_name_to_onnxscipt_value[node.name] = input_
        # FIXME: Refactor logic getting 'current_attr'.
        assert isinstance(current_attr, torch.Tensor)
        onnxscript_graph.add_initializer(input_.name, current_attr)
    else:
        # TODO(wechi): Support get_attr, call_module, call_method.
        raise RuntimeError(f"Found node type not defined in torch.fx: {node.op}")


@diagnostics.diagnose_call(diagnostics.rules.atenlib_fx_to_onnx)
def _export_fx_to_onnxscript(
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

    # Apply TorchScript's type promotion code.
    # Ideally, we should implement our type promotion but
    # to save time, we just reuse.
    onnxscript_graph.apply(
        torch._C._jit_pass_onnx_scalar_type_analysis,
        lowprecision_cast=True,
        opset_version=options.opset_version,
    )

    return onnxscript_graph


@_beartype.beartype
def _shape_inference_with_fake_tensor(decomposed_module: "torch.fx.GraphModule", *args):
    # Use this FakeTensorMode to
    # 1. convert nn.Parameter's in nn.Module to FakeTensor
    # 2. run FakeTensorProp
    # If (1) and (2) are done with difference FakeTensorMode's, undefined behavior may
    # happen.
    fake_tensor_mode = fake_tensor.FakeTensorMode()

    def to_fake_tensor(x):
        if isinstance(x, torch.Tensor) and not isinstance(x, fake_tensor.FakeTensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    # "args" are FakeTensor in FakeTensorProp so the parameters and buffers
    # in model must be converted to FakeTensor as well.
    fake_parameters_and_buffers = {
        k: to_fake_tensor(v)
        for k, v in itertools.chain(
            decomposed_module.named_parameters(), decomposed_module.named_buffers()
        )
    }

    # Shape inference via FakeTensorProp
    with stateless._reparametrize_module(
        decomposed_module, fake_parameters_and_buffers
    ):
        # Assign output types and shapes to each node.
        # TODO(wechi): It's possible to get symbolic types (and shapes)
        # for each node's output. Consider to set "tracing_mode=symbolic"
        # when calling make_fx and then remove FakeTensorProp below.
        fake_tensor_prop.FakeTensorProp(decomposed_module, fake_tensor_mode).propagate(
            *args
        )

    return decomposed_module


@_beartype.beartype
def _rename_placeholder_targets(
    module: "torch.fx.GraphModule", reference_module: "torch.fx.GraphModule"
):
    """Align the argument names in module with those in reference_module.
    After calling this function, the two forward(...) in module and reference_module should have
    the same signature.
    """
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    reference_placeholders = [
        node for node in reference_module.graph.nodes if node.op == "placeholder"
    ]

    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        placeholder.name = reference_placeholder.name

    module.recompile()


@_beartype.beartype
def _export(
    module: torch.fx.GraphModule,
    args,
    **kwargs,
) -> Union["onnx.ModelProto", bytes]:

    export_options = options.ExportOptions()
    export_options.update(**kwargs)
    # Apply decomposition table to the input graph.
    # Make sure the feed-in "module" is stateless.
    decomposed_module = proxy_tensor.make_fx(
        module,
        decomposition_table=export_options.decomposition_table,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(*args)
    # Rename placeholder targets to match the original module's signature since
    # We don't want to map forward(x, y, z) to forward(arg0, arg1, arg2).
    _rename_placeholder_targets(decomposed_module, module)
    # Run FakeTensorProp on decomposed_module.
    # Symbolic output of the i-th node can be accessed via
    # decomposed_module.graph.nodes[i].meta["val"]
    decomposed_module = _shape_inference_with_fake_tensor(decomposed_module, *args)

    # We want to pass list of ints and floats to TorchScript graph correctly
    # in _export_fx_to_ts, so we must disable FakeTensorMode. Otherwise, graph may
    # receive FakeTensor and results runtime error. In addition, TorchScript-based
    # ONNX exporter used in _ts_graph_to_onnx_model_in_protobuf is not compatible
    # with FakeTensorMode.
    with torch.utils._mode_utils.no_dispatch():
        onnxscript_graph = _export_fx_to_onnxscript(decomposed_module, export_options)
    # Export TorchScript graph to ONNX ModelProto.
    onnx_model = onnxscript_graph.to_model_proto(export_options.opset_version)

    if export_options.use_binary_format:
        # Return ModelProto in binary format.
        return onnx_model.SerializeToString()
    # Return ModelProto
    return onnx_model


@_beartype.beartype
def export(
    fn: Union[torch.nn.Module, Callable],
    *args,
    use_binary_format: bool = True,
    opset_version: int = _constants.ONNX_DEFAULT_OPSET,
    op_level_debug: bool = False,
) -> Union["onnx.ModelProto", bytes]:
    # args will be converted to symbolic tensor. Let's copy to avoid side effects.
    args = copy.deepcopy(args)
    # Translate callable to FX graph.
    #
    # TODO(wechi): There are several symbolic tracing mechanisms to convert
    # nn.Module to FX graph. We should choose the right one after they are
    # matured.
    graph_module, graph_guard = torch._dynamo.export(fn, *args, aten_graph=True)
    del graph_guard  # Unused
    # Export FX graph to ONNX ModelProto.
    #
    # Note that ALL kwargs are folded into constants in graph_module, so we don't pass kwargs
    # to _export.
    return _export(
        graph_module,
        args,
        opset_version=opset_version,
        decomposition_table=function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
        op_level_debug=op_level_debug,
    )


@_beartype.beartype
def export_after_normalizing_args_and_kwargs(
    fn: Union[torch.nn.Module, Callable],
    *args,
    use_binary_format: bool = True,
    opset_version: int = _constants.ONNX_DEFAULT_OPSET,
    op_level_debug: bool = False,
    **kwargs,
) -> Union["onnx.ModelProto", bytes]:
    """Export an nn.Module or a callable to ONNX.

    This traces the given nn.Module or a callable into FX graph and then
    and exports it to ONNX by calling `_export`. Notice that ONNX does
    not represent keyword arguments, so `args` and `kwargs` are normalized by
    calling `inspect.Signature.bind` and `inspect.BoundArgument.apply_defaults`
    in the beginning.

    Args:
        fn: nn.Module or a callable to be exported to ONNX.
        opset_version: the opset version to export the model to. E.g., 14.
        args: the positional arguments to pass to `fn`.
        use_binary_format: whether to return the ONNX model in binary format.
            If False, `onnx.ModelProto` will be returned. If False, the byte array
            generated by `onnx.ModelProto.SerializeToString` is returned.
        kwargs: the keyword arguments to pass to `fn`.

    Returns:
        ONNX model in binary format or `onnx.ModelProto`. To select return type,
        use `use_binary_format` argument.
    """

    if isinstance(fn, torch.nn.Module):
        signature = inspect.signature(fn.forward)
    else:
        signature = inspect.signature(fn)

    # We hope the input kwargs will be mapped to bound.args after binding.
    # If not, we will raise an error.
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # keyword-only arguments are not handled.
    # bound.kwargs only contains keyword-only arguments after calling
    # bind & apply_defaults, so we throw if it's not empty.
    assert not bound.kwargs

    class Wrapper(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, *args):
            result, _ = _pytree.tree_flatten(self.fn(*args))
            return result

    # args will be converted to symbolic tensor. Let's copy to avoid side effects.
    bound_args = copy.deepcopy(bound.args)
    # Translate callable to FX graph.
    #
    # TODO(wechi): There are several symbolic tracing mechanisms to convert
    # nn.Module to FX graph. We should choose the right one after they are
    # matured.

    class GraphCaptureCompiler:
        def __init__(self):
            self.captured_graph: Optional["torch.fx.GraphModule"] = None
            self.captured_graph_count = 0

        def compile(self, graph_module: "torch.fx.GraphModule", _):
            assert self.captured_graph_count == 0
            self.captured_graph = graph_module
            self.captured_graph_count += 1
            return graph_module

    compiler = GraphCaptureCompiler()
    torch._dynamo.reset()
    torch._dynamo.optimize(compiler.compile, nopython=True)(Wrapper(fn))(*bound_args)
    torch._dynamo.reset()
    assert compiler.captured_graph
    # Export FX graph to ONNX ModelProto.
    return _export(
        compiler.captured_graph,
        # Function optimized by _dynamo doesn't have None in args.
        tuple(arg for arg in bound_args if arg is not None),
        opset_version=opset_version,
        decomposition_table=function_dispatcher._ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
        op_level_debug=op_level_debug,
    )


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


# Register a few argument formatter
