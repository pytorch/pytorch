from __future__ import annotations

import copy
import functools
import inspect
import itertools
import operator
import os
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

from torch.onnx._internal import _beartype
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


class ModuleExpansionTracer(torch.fx._symbolic_trace.Tracer):
    """Tracer to create ONNX-exporting friendly FX graph.

    This tracer traces models into operators. That is,
    the traced graph mostly contains call_function nodes and
    has no call_module nodes. The call_module nodes
    are problematic to the use of make_fx(...) in ONNX
    exporter.
    """

    @_beartype.beartype
    def is_leaf_module(
        self, module: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        # This returns False so that all sub-modules are considered as not leaves
        # and therefore expanded into operators in
        # torch.fx._symbolic_trace.Tracer.call_module.
        return False

    @_beartype.beartype
    def to_bool(self, obj: "torch.fx.Proxy") -> bool:
        # This is a hack to tracing through if-else Python blocks.
        # It may generate incorrect ONNX graphs if the if-else block
        return False


# Functions directly wrapped to produce torch.fx.Proxy so that symbolic
# data can flow through those functions. Python functions (e.g., `torch.arange`)
# not defined by pybind11 in C++ do not go though Python dispatcher, so
# they are not automatically patched by FX's Python dispatcher.
# The list below means `torch.arange`, `torch.tensor`, and so on will be
# patched.
_TORCH_METHODS_TO_PATCH: Tuple[str, ...] = (
    "arange",
    "tensor",
    "finfo",
    "full",
    "empty",
)


def _wrap_for_symbolic_trace(target: Callable) -> Tuple[Callable, Callable]:
    """This function wraps ```target`` for symbolic tracing.

    This function wraps ```target``` so that its wrapper produces
    torch.fx.Proxy in symbolic computation. The returned values are
    the wrapper and then the original function. Per `_TORCH_METHODS_TO_PATCH`,
    this function shall receive `torch.arange`, `torch.tensor`, etc. as inputs.
    """

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


@_beartype.beartype
def _module_expansion_symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> "torch.fx.GraphModule":
    """Trace a callable into FX graph.

    When "root" is torch.nn.Module, calls to its submodule (type: torch.nn.Module) will be
    expanded into operators (e.g., torch.matmul, torch.add, +, and -) to simplify graph
    structure.
    """
    # For functions doesn't support symbolic tracing, create wrappers
    # which produce symbolic results during tracing.
    patched_torch_methods = {
        target_name: _wrap_for_symbolic_trace(getattr(torch, target_name))
        for target_name in _TORCH_METHODS_TO_PATCH
    }

    # Set the symbolic-tracing friendly functions so that `tracer.trace` below
    # can work.
    for name, (wrapper, _) in patched_torch_methods.items():
        setattr(torch, name, wrapper)

    try:
        # Set up a tracer.
        tracer = ModuleExpansionTracer()
        # Trace the model.
        graph = tracer.trace(root, concrete_args)
        name = (
            root.__class__.__name__
            if isinstance(root, torch.nn.Module)
            else root.__name__
        )
        return torch.fx.GraphModule(tracer.root, graph, name)
    finally:
        # Revert the patches for symbolic tracing.
        for name, (_, wrapped) in patched_torch_methods.items():
            # wrapped is the original version of `torch.name`.
            setattr(torch, name, wrapped)


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


def _filter_incompatible_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
    filtered = {}
    for key, value in kwargs.items():
        if key in {
            "layout",
            "device",
            "requires_grad",
            "pin_memory",
            "memory_format",
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

    graph_args = tuple(
        _retrieve_or_adapt_input_to_graph_set(arg, fx_name_to_onnxscipt_value)
        for arg in complete_args
    )
    graph_kwargs = _filter_incompatible_kwargs(complete_kwargs)

    # prepare torch format args and kwargs for op-level validation
    # Use fake tensor to create real tensor to feed in ops
    torch_args = []
    for arg in complete_args:
        if isinstance(arg, torch.fx.Node):
            # Create a concreate test tensor based on the fake tensor
            with torch.utils._mode_utils.no_dispatch():
                # TODO(titaiwang): improve engineering
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
    return (graph_args, graph_kwargs, tuple(torch_args), torch_kwargs)


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
    onnxscript_value_name_to_real_tensor: Dict[
        str, Union[torch.Tensor, Tuple[torch._C.Value, ...]]
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
        onnxscript_value_name_to_real_tensor[input_.name] = current_attr  # type: ignore[assignment]
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
    # Similar to fx_name_to_onnxscipt_value, we need a mapping fo real tensors (usually tensor parameters
    # in nn.Module). Note that TorchScript's cannot store real tensors; TorchScript values are all
    # symbolic. This is passed into ONNX ModelProto as the initializers.
    onnxscript_value_name_to_real_tensor: Dict[
        str, Union[torch.Tensor, Tuple[torch._C.Value, ...]]
    ] = {}
    for node in fx_module_with_metadata.graph.nodes:
        _export_fx_node_to_onnxscript(
            node,
            onnxscript_graph,
            fx_name_to_onnxscipt_value,
            onnxscript_value_name_to_real_tensor,
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

    return onnxscript_graph, onnxscript_value_name_to_real_tensor


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
        onnxscript_graph, initializers = _export_fx_to_onnxscript(
            decomposed_module, export_options
        )
    # Export TorchScript graph to ONNX ModelProto.
    onnx_model = onnxscript_graph.to_model_proto(
        initializers, export_options.opset_version
    )

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
def export_without_kwargs(
    fn: Union[torch.nn.Module, Callable],
    *args,
    use_binary_format: bool = True,
    opset_version: int = _constants.ONNX_DEFAULT_OPSET,
    op_level_debug: bool = False,
    **kwargs,
) -> Union["onnx.ModelProto", bytes]:
    if isinstance(fn, torch.nn.Module):
        signature = inspect.signature(fn.forward)
    else:
        signature = inspect.signature(fn)

    # We hope the input kwargs will be mapped to bound.args after binding.
    # If not, we will raise an error.
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # kwargs are not handled.
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
def _move_placeholder_to_front(graph_module: "torch.fx.GraphModule") -> None:
    """
    This function move all placeholder nodes to the front of the graph node list.
    In torch.fx.Graph, placeholder is a special assignment node. If it's not
    executed in the beginning, it could overwrite values computed by upstream
    nodes.
    """

    graph = graph_module.graph
    placeholders = []
    first_not_placeholder = None
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)
        if first_not_placeholder is None and node.op != "placeholder":
            first_not_placeholder = node
    if first_not_placeholder is None:
        return
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)


@_beartype.beartype
def _replace_get_attr_with_placeholder(
    graph_module: "torch.fx.GraphModule",
) -> Tuple[torch.Tensor, ...]:
    """
    Replace get_attr with placeholder.
    The parameters and buffers accessed by the original get_attr are returned;
    they are useful when creating random inputs for the modified graph_module.
    """
    graph = graph_module.graph
    replaced_attrs: List[torch.Tensor] = []
    for node in graph.nodes:
        if node.op == "get_attr":
            replaced_attr: Optional[torch.Tensor] = None
            # get_attr could retrieve either parameter or buffer, so
            # we need to try both.
            try:
                replaced_attr = graph_module.get_parameter(node.target)
            except AttributeError:
                # It's possible that model author use buffer instead of
                # parameter to store trainable weights. In this case,
                # 1. get_parameter will throw something like
                #    AttributeError: `bias` is not an nn.Parameter.
                # 2. get_buffer should work.
                replaced_attr = graph_module.get_buffer(node.target)

            # Reassign op type so that get_attr node becomes placeholder node.
            node.op = "placeholder"
            # The target name in placeholder must be a valid Python identifier.
            # Thus, we replace, e.g., "module.submodule.weight" with
            # "module_submodule_weight".
            node.target = node.target.replace(".", "_")
            # Default value is None. This is needed as long as the "graph_module"
            # has optional inputs. Assume the original forward signature is
            #  def forward(self, x, y=None)
            # and the replaced get_attr node has target "z". Then, the modified
            # signature should be
            #  def forward(self, x, y=None, z=None)
            # Without the following line, the signature will be
            #  def forward(self, x, y=None, z)
            # , which is not valid Python code.
            node.args = (None,)

            replaced_attrs.append(replaced_attr)

    return tuple(replaced_attrs)


@_beartype.beartype
def _trace_into_fx_graph_via_fx_symbolic_trace(
    module: torch.nn.Module,
    *args,
    # kwargs are the keyword arguments to call "module"; that is,
    # module(*args, **kwargs) must run.
    **kwargs,
) -> Tuple["torch.fx.GraphModule", Tuple[Any, ...]]:
    signature = inspect.signature(module.forward)

    # We hope the input kwargs will be mapped to bound.args after binding.
    # If not, we will raise an error.
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    # After apply_defaults, all non keyword-only arguments are in bound.args.
    # Because below code do not support keyword-word arguments, bound.kwargs
    # must be empty.
    assert len(bound.kwargs) == 0, bound.kwargs

    # Create inputs to call symbolic trace (torch.fx.symbolic_trace)
    # Example content of concrete_args:
    #  concrete_args["x"] = torch.fx._symbolic_trace.PH
    #  concrete_args["b"] = 1
    # where "x" and "b" are argument names in "signature".
    concrete_args = {}
    for param_name, param_value in bound.arguments.items():
        if isinstance(param_value, torch.Tensor):
            # param_value can be, e.g., a real tensor or a fake tensor.
            # param_value is treated as substitutable tensor symbol (aka placeholder).
            concrete_args[param_name] = torch.fx._symbolic_trace.PH
        else:
            concrete_args[param_name] = param_value

    return (
        _module_expansion_symbolic_trace(module, concrete_args=concrete_args),
        bound.args,
    )


@_beartype.beartype
def export_without_parameters_and_buffers(
    module: torch.nn.Module,
    *args,
    decomposition_table: Optional[Dict[torch._ops.OpOverload, Callable]] = None,
    use_binary_format: bool = True,
    opset_version: int = _constants.ONNX_DEFAULT_OPSET,
    op_level_debug: bool = False,
    # kwargs are the keyword arguments to call "module"; that is,
    # module(*args, **kwargs) must run.
    **kwargs,
) -> Tuple[
    Union["onnx.ModelProto", bytes],
    "torch.fx.GraphModule",
    Tuple[Any, ...],
    Tuple[Any, ...],
]:

    graph_module, bound_args = _trace_into_fx_graph_via_fx_symbolic_trace(
        module, *args, **kwargs
    )

    # Make sure all placeholder nodes are executed before get_attr nodes.
    # Otherwise, inputs can interleave with initializers in the final ModeoProto.graph.input.
    # Basically, we want
    #  ModeoProto.graph.input =
    #   [input_0, input_1, ..., input_n, weight_0, weight_1, ..., weight_m]
    # and we don't want
    #  ModeoProto.graph.input =
    #   [input_0, weight_0, input_1, weight_1, ..., input_n, weight_0, weight_1, ..., weight_m]
    _move_placeholder_to_front(graph_module)
    # To save memory, move get_attr to input so that the generated model doesn't
    # have weigh tensors. "replaced_attrs" are the list of replaced weight tensors.
    replaced_attrs = _replace_get_attr_with_placeholder(graph_module)
    # Move all newly created placeholder nodes to the front of the graph.
    _move_placeholder_to_front(graph_module)
    # Finalize the graph editing.
    graph_module.recompile()

    return (
        _export(
            graph_module,
            (*bound_args, *replaced_attrs),
            opset_version=opset_version,
            decomposition_table=decomposition_table,
            use_binary_format=use_binary_format,
            op_level_debug=op_level_debug,
        ),
        graph_module,
        bound_args,
        replaced_attrs,
    )


@_beartype.beartype
def _create_tensor_proto_with_external_data(
    tensor: torch.Tensor, name: str, location: str, basepath: str
) -> "onnx.TensorProto":
    """Create a TensorProto with external data from a PyTorch tensor.
    The external data is saved to os.path.join(basepath, location).

    Args:
        tensor: Tensor to be saved.
        name: Name of the tensor (i.e., initializer name in ONNX graph).
        location: Relative location of the external data file
            (e.g., "/tmp/initializers/weight_0" when model is "/tmp/model_name.onnx").
        basepath: Base path of the external data file (e.g., "/tmp/external_data" while model must be in "/tmp").


    Reference for ONNX's external data format:
        How to load?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L187
        How to save?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L43
        How to set ONNX fields?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L88
    """
    tensor_proto = onnx.TensorProto()
    tensor_proto.name = name
    tensor_proto.data_type = torch.onnx._type_utils._SCALAR_TYPE_TO_ONNX[  # type: ignore[assignment]
        torch.onnx._type_utils._DTYPE_TO_SCALAR_TYPE[tensor.dtype]
    ]
    tensor_proto.dims.extend(tensor.shape)
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL

    # Settings for saving one tensor per file.
    # Offset is zero because there is no other tensor in the same file.
    key_value_pairs = {
        "location": location,
        "offset": 0,
        "length": tensor.untyped_storage().nbytes(),
    }
    for k, v in key_value_pairs.items():
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)

    # Actual path to write content of tensor.
    external_data_file_path = os.path.join(basepath, location)
    if os.path.exists(external_data_file_path):
        os.remove(external_data_file_path)

    # Create external data's folder if not exists.
    external_data_dir_path = os.path.dirname(external_data_file_path)
    if not os.path.exists(external_data_dir_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(external_data_dir_path)

    # Create a fresh file.
    with open(external_data_file_path, "xb") as data_file:
        # No need to call "seek" because offset is 0.
        # data_file.seek(0)
        # Write tensor content to the file.
        data_file.write(tensor.numpy().tobytes())

    return tensor_proto


@_beartype.beartype
def save_model_with_external_data(
    basepath: str,
    model_location: str,
    initializer_location: str,
    torch_load_paths: Tuple[str, ...],
    onnx_model: "onnx.ModelProto",
) -> None:
    """Load PyTorch tensors from files and add to "onnx_model" as external initializers.

    Output files:
        ONNX model file path:
        ONNX initializer folder: os.path.join(basepath, initializer_location)

    After running this function, you can do
        ort_sess = onnxruntime.InferenceSession(os.path.join(basepath, model_location))
    to execute the model.

    Arguments:
        basepath: Base path of the external data file (e.g., "/tmp/large-onnx-model").
        model_location: Relative location of the ONNX model file.
            E.g., "model.onnx" so that the model file is saved to
            "/tmp/large-onnx-model/model.onnx".
        initializer_location: Relative location of the ONNX initializer folder.
            E.g., "initializers" so that the initializers are saved to
            "/tmp/large-onnx-model/initializers".
        torch_load_paths: Files which containing serialized PyTorch tensors to be saved
            as ONNX initializers. They are loaded by torch.load.
        onnx_model: ONNX model to be saved with external initializers.
            If an input name matches a tensor loaded from "torch_load_paths",
            the tensor will be saved as that input's external initializer.
    """
    onnx_model_with_initializers = onnx.ModelProto()
    onnx_model_with_initializers.CopyFrom(onnx_model)
    onnx_input_names = [input.name for input in onnx_model.graph.input]

    for path in torch_load_paths:
        state_ditc = torch.load(path)
        for name, tensor in state_ditc.items():
            # Basically, "transformer.attention.self.query.weight" is mapped
            # to "transformer_attention_self_query_weight" for mimicking the
            # name-modifying code in FX-to-ONNX exporter.
            # See function _replace_get_attr_with_placeholder for details.
            refined_name = name.replace(".", "_")

            # For each refined PyTorch tensor name loaded by torch.load,
            #  1.  Search its best match in ONNX model. E.g., the match of
            #       "transformer_attention_weight" could be "attention_weight".
            #  2.  Set "tensor" as the initializer of the matched ONNX input.
            #      E.g., "tensor" is stored as the initializer of "attention_weight".
            # Step 1 is required because sometimes, tensor names are stored with prefix the dictionary
            # loaded by torch.load.
            for onnx_input_name in onnx_input_names:
                if onnx_input_name.endswith(refined_name) or refined_name.endswith(
                    onnx_input_name
                ):
                    # Find a match. Change refined_name to the matched ONNX input name, so that we
                    # create initializer with the right ONNX name.
                    refined_name = onnx_input_name
                    break

            relative_tensor_file_path = os.path.join(initializer_location, refined_name)
            # Create one file per tensor.
            # tensor_proto.raw_data is stored to external file at
            # os.path.join(basepath, relative_tensor_file_path).
            tensor_proto = _create_tensor_proto_with_external_data(
                tensor, refined_name, relative_tensor_file_path, basepath
            )
            # Add the tensor_proto to the ONNX model as an initializer with external data.
            onnx_model_with_initializers.graph.initializer.append(tensor_proto)

    # model_location should be a pure file name such as "file_name.onnx", not "folder/file_name.onnx".
    onnx.save(onnx_model_with_initializers, os.path.join(basepath, model_location))


# TODO(titaiwang): copied from ops_correctness_test.py, should have a common place?
TORCH_TYPE_TO_ONNX = {
    torch.bool: onnx.TensorProto.BOOL,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.int8: onnx.TensorProto.INT8,
    torch.int16: onnx.TensorProto.INT16,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.complex64: onnx.TensorProto.COMPLEX64,
    torch.complex128: onnx.TensorProto.COMPLEX128,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}

# TODO(titaiwang): copied from ops_correctness_test.py, should have a common place?
def _convert_tensor_to_numpy(input: Any) -> Any:
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    if isinstance(input, (tuple, list)):
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        if isinstance(input[0], torch.Tensor):
            return [_convert_tensor_to_numpy(x) for x in input]
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # Just a sequence of numbers
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        if isinstance(input[0], float):
            return np.array(input)

    return input


# TODO(titaiwang): copied from ops_correctness_test.py, should have a common place?
def _convert_kwargs_for_onnx(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Converts kwargs to be compatible with ONNX Runtime.

    ONNX Runtime doesn't support torch.bool, so we convert them to torch.uint8.
    """
    new_kwargs = {}
    for key, value in kwargs.items():
        if key == "device":
            continue
        if key == "dtype":
            value = TORCH_TYPE_TO_ONNX[value]
        new_kwargs[key] = value
    return new_kwargs


@_beartype.beartype
def _validate_op_between_ort_torch(
    node: torch.fx.Node,
    symbolic_fn: onnxscript.OnnxFunction,
    torch_args: tuple,
    torch_kwargs: dict,
):
    """Validate the op between ONNX Runtime and PyTorch."""
    # op-level validation
    # Symbolic_fn should have the same output as node.target (torch ops)
    try:
        with evaluator.default_as(evaluator.ort_evaluator):
            expected_outputs = node.target(*torch_args, **torch_kwargs)  # type: ignore[operator]
            # TODO(titaiwang): Expose _convert_tensor_to_numpy and _convert_kwargs_for_onnx?
            input_onnx = [_convert_tensor_to_numpy(x) for x in torch_args]
            # deal with dtype and device
            kwargs_onnx = _convert_kwargs_for_onnx(torch_kwargs)
            ort_outputs = symbolic_fn(*input_onnx, **kwargs_onnx)

            for ort_output, expected_output in zip(ort_outputs, expected_outputs):
                try:
                    torch.testing.assert_close(
                        expected_output.numpy(),
                        ort_output,
                        check_device=False,
                        atol=10e-4,
                        rtol=10e-3,
                    )
                except AssertionError as e:
                    warnings.warn(
                        f"Suppressed AssertionError:\n{e}.\n"
                        f"Op {node.target} has mismatch outputs. "
                        f"Please check the implementation of {symbolic_fn}."
                    )
                    diagnostic = diagnostics.export_context().inflight_diagnostic()
                    diagnostic.with_additional_message(
                        f"### Validation failed\n"
                        f"{diagnostics.decorator.format_exception_in_markdown(e)}"
                    )
                    diagnostic.level = diagnostics.levels.ERROR
    except Exception as e:
        warnings.warn(f"ORT fails to run with error: {e}.")
        diagnostic = diagnostics.export_context().inflight_diagnostic()
        diagnostic.with_additional_message(
            f"### Validation failed\n"
            f"{diagnostics.decorator.format_exception_in_markdown(e)}"
        )
        diagnostic.level = diagnostics.levels.WARNING


# Register a few argument formatter
