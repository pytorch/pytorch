from __future__ import annotations

import copy
import functools
import inspect
import itertools
import operator
import os
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import onnx
except ImportError:
    onnx = ModuleType("onnx")

    def _onnx_not_available(name):
        raise RuntimeError(
            "ONNX is not available. Please install ONNX to use this feature."
        )

    onnx.__getattr__ = _onnx_not_available  # type: ignore[assignment]

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
from torch.onnx._globals import GLOBALS as ONNX_GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.utils import _pytree


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


@_beartype.beartype
def _create_op_overload_to_exporter_key_table() -> Dict[torch._ops.OpOverload, str]:
    table: Dict[torch._ops.OpOverload, str] = {}

    for attr_name in dir(torch.ops.aten):
        op_overload_packet = getattr(torch.ops.aten, attr_name)
        if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
            continue

        exporter_look_up_key = op_overload_packet._qualified_op_name
        if registration.registry.get_function_group(exporter_look_up_key) is None:
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

    table[torch.ops.prims.convert_element_type.default] = "prim::convert_element_type"
    table[torch.ops.aten.baddbmm.default] = "aten::baddbmm"
    return table


# Dictionary that maps torch.ops.aten.* to exporter look up key; e.g.,
# _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[torch.add.Tensor] is "aten::add".
# In subsequent code, torch.ops.aten.add.Tensor's exporter is found by
# registration.registry.get_function_group("aten::add").
_OP_OVERLOAD_TO_EXPORTER_KEY_TABLE = _create_op_overload_to_exporter_key_table()


@_beartype.beartype
def _create_onnx_friendly_decomposition_table() -> Dict[
    torch._ops.OpOverload, Callable
]:
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = {}
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():
        # Skip decomposition into "prim::*" ops, because they are not generally supported by ONNX.
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX exporter.
        if (
            "torch._refs" in decomp_fn.__module__
            or op_overload in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
        ):
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table


# This is a subset of PyTorch's built-in aten-to-aten decomposition. If an aten
# op (e.g., torch.ops.aten.add.Tensor) has exporter, we exclude the op's decomposition
# function in the _ONNX_FRIENDLY_DECOMPOSITION_TABLE.
_ONNX_FRIENDLY_DECOMPOSITION_TABLE = _create_onnx_friendly_decomposition_table()


@_beartype.beartype
def _retrieve_or_wrap_scalar_as_constant(
    g: "torch.onnx._internal.jit_utils.GraphContext",
    fx_node_arg: Any,
    fx_name_to_ts_value: Dict[str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]],
    example_output: Any,
):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """

    ts_value = fx_node_arg
    if isinstance(ts_value, torch.fx.Node):
        # 1. fx_node_arg is a torch.fx.Node, which means
        #    fx_node_arg stands for the output of that torch.fx.Node.
        # 2. fx_node_arg (variable in torch.fx.Graph) is be mapped to
        #    torch.jit.Value, fx_name_to_ts_value[fx_node_arg.name],
        #    in TorchScript graph.
        ts_value = fx_name_to_ts_value[ts_value.name]
    elif isinstance(ts_value, float):
        # Always promote scalar to tensor with element type "dtype."
        # Usually, "dtype" is extracted from the expected output tensor of the node.
        # If this assumption is broken, we probably need to
        #  1. add "scalar" type in ONNX  and extend all exporters to support it, or
        #  2. write type promotion logic for each operator.
        # TODO(wechi): the called exporting function should tell all allowed input and output types.
        # Then, here we can try type-casting if type-mismatch happens.
        ts_value = g.op("Constant", value_t=torch.tensor(ts_value, dtype=torch.float))
    elif isinstance(ts_value, int):
        # Always promote scalar to tensor with element type "dtype."
        # Usually, "dtype" is extracted from the expected output tensor of the node.
        # If this assumption is broken, we probably need to
        #  1. add "scalar" type in ONNX  and extend all exporters to support it, or
        #  2. write type promotion logic for each operator.
        # TODO(wechi): the called exporting function should tell all allowed input and output types.
        # Then, here we can try type-casting if type-mismatch happens.
        ts_value = g.op("Constant", value_t=torch.tensor(ts_value, dtype=torch.int64))
    elif ts_value is None:
        ts_value = g.op("prim::Constant")
        ts_value.setType(torch._C.OptionalType.ofTensor())
    elif isinstance(ts_value, list) and all(isinstance(val, int) for val in ts_value):
        ts_value = g.op("Constant", value_t=torch.tensor(ts_value, dtype=torch.int64))
    elif isinstance(ts_value, list) and all(isinstance(val, float) for val in ts_value):
        ts_value = g.op("Constant", value_t=torch.tensor(ts_value, dtype=torch.float))
    elif isinstance(ts_value, list) and all(
        isinstance(val, torch.fx.Node) for val in ts_value
    ):
        # A list of torch.fx.Node's (aka ts_value) should be mapped to a list of TorchScript values
        # in TorchScript graph.
        ts_list = [fx_name_to_ts_value[val.name] for val in ts_value]
        ts_value = g.op("prim::ListConstruct", *ts_list)
    elif isinstance(ts_value, torch.dtype):
        from torch.onnx import _type_utils

        ts_value = _type_utils.JitScalarType.from_dtype(ts_value)
    else:
        raise RuntimeError(f"Unexpected type of fx_node_arg: {type(fx_node_arg)}")
    return ts_value


@_beartype.beartype
def _wrap_fx_args_as_ts_args(
    g: "torch.onnx._internal.jit_utils.GraphContext",
    root: torch.nn.Module,
    node: torch.fx.Node,
    fx_name_to_ts_value: Dict[str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]],
):
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args: List[Any] = []
    if inspect.isbuiltin(node.target):
        complete_args = list(node.args)
    else:
        for i, expected_arg in enumerate(node.target._schema.arguments):  # type: ignore[union-attr]
            if i < len(node.args):
                complete_args.append(node.args[i])
            else:
                # Get default from schema.
                complete_args.append(expected_arg.default_value)
    return tuple(
        _retrieve_or_wrap_scalar_as_constant(
            # The node.meta["val"] is generated by FakeTensorProp.
            g,
            arg,
            fx_name_to_ts_value,
            node.meta["val"],
        )
        for arg in complete_args
    )


@_beartype.beartype
def _fill_tensor_types(
    ts_values: Union[torch._C.Value, Tuple[torch._C.Value, ...]],
    expected_values: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
):
    flat_ts_values, _ = _pytree.tree_flatten(ts_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for ts_value, expected_value in zip(flat_ts_values, flat_expected_values):
        ts_value.setType(torch._C.TensorType.create_from_tensor(expected_value))


@_beartype.beartype
def _export_fx_to_ts(fx_module_with_metadata, opset_version):
    # TODO(wechi): To get rid of TorchScript dependency,
    # "g" should just be onnx.GraphProto or an equivalent
    # data structure in ONNXScript.
    g = torch._C.Graph()
    # In the following loop, a TorchScript graph is created to
    # represent the input FX graph with ONNX symbols (e.g., onnx::add).
    # To connect the values to nodes in the TorchScript graph, we maintain
    # fx_name_to_ts_value. Basically, we want to translate
    #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
    # to
    #   fx_name_to_ts_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_ts_value[fx_tensor_y.name]
    fx_name_to_ts_value: Dict[
        str, Union[torch._C.Value, Tuple[torch._C.Value, ...]]
    ] = {}
    # Similar to fx_name_to_ts_value, we need a mapping fo real tensors (usually tensor parameters
    # in nn.Module). Note that TorchScript's cannot store real tensors; TorchScript values are all
    # symbolic. This is passed into ONNX ModelProto as the initializers.
    ts_name_to_real_tensor: Dict[
        str, Union[torch.Tensor, Tuple[torch._C.Value, ...]]
    ] = {}
    for node in fx_module_with_metadata.graph.nodes:
        if node.op == "placeholder":
            if node.meta["val"] is None:
                # This input argument is None, which is mapped
                # to a NULL value in TorchScript type system.
                v = g.op("prim::Constant")  # type: ignore[attr-defined]
                v.setType(torch._C.OptionalType.ofTensor())
            else:
                # Input of graph.
                v = g.addInput(node.name)
                v.setType(torch._C.TensorType.create_from_tensor(node.meta["val"]))
                assert (
                    v is not None
                ), f"Node creates None with target={node.target} and name={node.name}"
            fx_name_to_ts_value[node.name] = v
        elif node.op == "call_function":
            # aten ops and other statless functions.
            if (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target in _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE
            ):
                exporter_key = _OP_OVERLOAD_TO_EXPORTER_KEY_TABLE[node.target]
                symbolic_function_group = registration.registry.get_function_group(
                    exporter_key
                )
                assert symbolic_function_group is not None
                symbolic_fn = symbolic_function_group.get(opset_version)
                assert symbolic_fn is not None
                # TODO(wechi): current type checking throws when feeding torch._C.Graph
                # to symbolic_opset*.py functions, so we need the following wrapper.
                # After we get rid of TorchScript, we can remove this wrapper.
                graph_context = jit_utils.GraphContext(
                    graph=g,
                    block=g.block(),  # Pointless. Just make linter happy.
                    opset=opset_version,
                    original_node=g.insertPoint(),  # Pointless. Just make linter happy.
                    params_dict={},  # Pointless. Just make linter happy.
                    env={},  # Pointless. Just make linter happy.
                )
                # Map FX inputs to ONNX inputs and fill optional inputs with default values.
                ts_args = _wrap_fx_args_as_ts_args(
                    graph_context, fx_module_with_metadata, node, fx_name_to_ts_value
                )
                # The returned value could be a value of a tuple of values.
                v = symbolic_fn(graph_context, *ts_args)
                assert (
                    v is not None
                ), f"Node creates None with target={node.target}, name={node.name}, args={ts_args}"
                # Assign type and shape obtained from FakeTensorProp.
                # _fill_tensor_types(v, node.meta["val"])
                # One fx node could produce multiple outputs (e.g., tuple of tensors); in
                # that case, v is a tuple of TorchScript values.
                fx_name_to_ts_value[node.name] = v
            elif node.target == operator.getitem and isinstance(node.args, tuple):
                ts_value_tuple = fx_name_to_ts_value[node.args[0].name]
                if isinstance(ts_value_tuple, tuple):
                    v = ts_value_tuple[node.args[1]]
                    assert (
                        v is not None
                    ), f"Node creates None with target={node.target} and name={node.name}"
                    fx_name_to_ts_value[node.name] = v
                else:
                    # TODO: lots of repeated code from above, remove this hack.
                    symbolic_function_group = registration.registry.get_function_group(
                        "aten::__getitem_"
                    )
                    assert symbolic_function_group is not None
                    symbolic_fn = symbolic_function_group.get(opset_version)
                    assert symbolic_fn is not None
                    graph_context = jit_utils.GraphContext(
                        graph=g,
                        block=g.block(),  # Pointless. Just make linter happy.
                        opset=opset_version,
                        original_node=g.insertPoint(),  # Pointless. Just make linter happy.
                        params_dict={},  # Pointless. Just make linter happy.
                        env={},  # Pointless. Just make linter happy.
                    )
                    # Map FX inputs to ONNX inputs and fill optional inputs with default values.
                    ts_args = _wrap_fx_args_as_ts_args(
                        graph_context,
                        fx_module_with_metadata,
                        node,
                        fx_name_to_ts_value,
                    )
                    v = symbolic_fn(graph_context, *ts_args)
                    assert (
                        v is not None
                    ), f"Node creates None with target={node.target}, name={node.name}, args={ts_args}"
                    # One fx node could produce multiple outputs (e.g., tuple of tensors); in
                    # that case, v is a tuple of TorchScript values.
                    fx_name_to_ts_value[node.name] = v
            elif node.target == torch.fx._symbolic_trace._assert_is_none:
                # Skip the assert_is_none node because it is isolated from other computation and
                # ONNX doesn't have a corresponding ASSERT op.
                pass
            else:
                raise RuntimeError(
                    "Unknown call_function target: {}".format(node.target)
                )
        elif node.op == "output":

            def register_outputs(
                ts_outputs: Union[torch._C.Value, Tuple[torch._C.Value, ...]]
            ):
                if isinstance(ts_outputs, torch._C.Value):
                    g.registerOutput(ts_outputs)
                else:
                    for ts_output in ts_outputs:
                        assert isinstance(
                            ts_output, torch._C.Value
                        ), f"ts_output must be a torch._C.Value, not {type(ts_output)}"
                        g.registerOutput(ts_output)

            if isinstance(node.args[0], torch.fx.Node):
                ts_value_or_ts_value_tuple = fx_name_to_ts_value[node.args[0].name]
                register_outputs(ts_value_or_ts_value_tuple)
            else:
                # ONNX can't represent collection types (e.g., dictionary, tuple of tuple of
                # tensor, etc), we flatten the collection and register each element as output.
                flat_args, _ = _pytree.tree_flatten(node.args[0])
                for arg in flat_args:
                    assert isinstance(
                        arg, torch.fx.Node
                    ), f"ts_output must be a torch.fx.Node, not {type(arg)}"
                    ts_value_or_ts_value_tuple = fx_name_to_ts_value[arg.name]
                    register_outputs(ts_value_or_ts_value_tuple)
        elif node.op == "call_method":
            # TODO(wechi): Support call_method.
            raise RuntimeError("call_method is not supported yet.")
        elif node.op == "call_module":
            # TODO(wechi): Support call_module.
            raise RuntimeError("call_module is not supported yet.")
        elif node.op == "get_attr":
            current_attr = fx_module_with_metadata
            sub_attr_names = node.target.split(".")
            # If node.targe is "conv.weight", the following loop first
            # assigns fx_module_with_metadata.conv to current_attr, and then
            # fx_module_with_metadata.conv.weight to current_attr.
            while sub_attr_names:
                sub_attr_name = sub_attr_names.pop(0)
                if not hasattr(current_attr, sub_attr_name):
                    raise ValueError(
                        f"Attribute {sub_attr_name} is not found in {current_attr}."
                    )
                current_attr = getattr(current_attr, sub_attr_name)

            v = g.addInput(node.name)
            v.setType(torch._C.TensorType.create_from_tensor(current_attr))  # type: ignore[assignment]
            assert (
                v is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            fx_name_to_ts_value[node.name] = v
            ts_name_to_real_tensor[v.debugName()] = current_attr  # type: ignore[assignment]
        else:
            # TODO(wechi): Support get_attr, call_module, call_method.
            raise RuntimeError("Found node type not defined in torch.fx: " + node.op)

    torch._C._jit_pass_onnx_scalar_type_analysis(
        g, lowprecision_cast=True, opset_version=opset_version
    )

    # When replace aten with onnx ops, the node-level shape type inference uses
    # ConstantValueMap which will not be cleared up until graph-level
    # shape type inference, and could be a bug. node/graph level inference should be both applied.
    # TODO(titaiwang): If onnx shape type inference is intended to be deprecated in converter.
    # node-level shape type inference should be also deprecated as well in g.op
    if ONNX_GLOBALS.onnx_shape_inference:
        torch._C._jit_pass_onnx_graph_shape_type_inference(
            g, params_dict={}, opset_version=opset_version
        )

    return g, ts_name_to_real_tensor


@_beartype.beartype
def _ts_graph_to_onnx_model_in_protobuf(
    ts_graph: torch._C.Graph,
    ts_name_to_real_tensor: Dict[str, torch.Tensor],
    opset_version: int,
) -> Union["onnx.ModelProto", bytes]:
    proto, _, _, _ = ts_graph._export_onnx(  # type: ignore[attr-defined]
        initializers=ts_name_to_real_tensor,
        onnx_opset_version=opset_version,
        dynamic_axes={},
        defer_weight_export=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        strip_doc_string=False,
        keep_initializers_as_inputs=False,
        custom_opsets={},
        add_node_names=True,
        onnx_file_path="",
        node_attr_to_name={},
    )

    return proto


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
    opset_version: int = None,
    *args,
    decomposition_table: Optional[Dict[torch._ops.OpOverload, Callable]] = None,
    use_binary_format: bool = True,
) -> Union["onnx.ModelProto", bytes]:
    # Export FX graph to ONNX ModelProto.
    if decomposition_table is None:
        # Use default decomposition table.
        decomposition_table = _ONNX_FRIENDLY_DECOMPOSITION_TABLE
    # Apply decomposition table to the input graph.
    # Make sure the feed-in "module" is stateless.
    decomposed_module = proxy_tensor.make_fx(
        module,
        decomposition_table=decomposition_table,
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
        ts_graph, ts_initializers = _export_fx_to_ts(decomposed_module, opset_version)
        # Export TorchScript graph to ONNX ModelProto.
        onnx_model = _ts_graph_to_onnx_model_in_protobuf(
            ts_graph, ts_initializers, opset_version
        )
    if use_binary_format:
        # Return ModelProto in binary format.
        return onnx_model
    # Return ModelProto in readable format (printable).
    model_proto: "onnx.ModelProto" = onnx.ModelProto.FromString(onnx_model)
    return model_proto


@_beartype.beartype
def export(
    fn: Union[torch.nn.Module, Callable],
    opset_version,
    *args,
    use_binary_format: bool = True,
) -> Union["onnx.ModelProto", bytes]:
    # args will be converted to symbolic tensor. Let's copy to avoid side effects.
    args = copy.deepcopy(args)
    # Translate callable to FX graph.
    #
    # TODO(wechi): There are several symbolic tracing mechanisms to convert
    # nn.Module to FX graph. We should choose the right one after they are
    # matured.
    graph_module, graph_guard = torch._dynamo.export(fn, *args, aten_graph=True)
    # Export FX graph to ONNX ModelProto.
    #
    # Note that ALL kwargs are folded into constants in graph_module, so we don't pass kwargs
    # to _export.
    return _export(
        graph_module,
        opset_version,
        *args,
        decomposition_table=_ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
    )


@_beartype.beartype
def export_without_kwargs(
    fn: Union[torch.nn.Module, Callable],
    opset_version,
    *args,
    use_binary_format: bool = True,
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
    torch._dynamo.optimize(compiler.compile, nopython=True)(Wrapper(fn))(*bound_args)
    torch._dynamo.reset()
    assert compiler.captured_graph
    # Export FX graph to ONNX ModelProto.
    return _export(
        compiler.captured_graph,
        opset_version,
        # Function optimized by _dynamo doesn't have None in args.
        *tuple(arg for arg in bound_args if arg is not None),
        decomposition_table=_ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
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
    opset_version: int = None,
    # kwargs are the keyword arguments to call "module"; that is,
    # module(*args, **kwargs) must run.
    **kwargs,
) -> Tuple[
    Union["onnx.ModelProto", bytes],
    "torch.fx.GraphModule",
    Tuple[Any, ...],
    Tuple[Any, ...],
]:
    if opset_version is None:
        opset_version = torch.onnx._constants.ONNX_DEFAULT_OPSET

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
            opset_version,
            *bound_args,
            *replaced_attrs,
            decomposition_table=decomposition_table,
            use_binary_format=use_binary_format,
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
