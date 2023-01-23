import copy
import inspect
import itertools
import operator
from typing import Callable, Dict, Optional, Tuple, Union

import onnx

import torch
import torch._C
import torch._decomp
import torch._dynamo
import torch._ops
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.passes import fake_tensor_prop
from torch.nn.utils import stateless
from torch.onnx._globals import GLOBALS as ONNX_GLOBALS
from torch.onnx._internal import jit_utils, registration
from torch.utils import _pytree


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


def _retrieve_or_wrap_scalar_as_constant(
    g, fx_node_arg, fx_name_to_ts_value, example_output
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
    elif isinstance(ts_value, list) and all(isinstance(val, torch.fx.Node) for val in ts_value):
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


def _wrap_fx_args_as_ts_args(g, root, node, fx_name_to_ts_value):
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args = []
    if inspect.isbuiltin(node.target):
        complete_args = node.args
    else:
        for i, expected_arg in enumerate(node.target._schema.arguments):
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


def _fill_tensor_types(ts_values, expected_values):
    flat_ts_values, _ = _pytree.tree_flatten(ts_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for ts_value, expected_value in zip(flat_ts_values, flat_expected_values):
        ts_value.setType(torch._C.TensorType.create_from_tensor(expected_value))


def _export_fx_to_ts(fx_module_with_metadata, opset_version):
    # TODO(wechi): To get rid of TorchScript dependency,
    # "g" should just be onnx.GraphProto or an equivalent
    # data structure in ONNXScript.
    g = torch._C.Graph()
    # In the following loop, a TorchScript graph is created to
    # represent the input FX graph with ONNX symbols (e.g., onnx::add).
    # To connect the values to nodes in the TorchScript graph, we maintian
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
    # fx_module_with_metadata.print_readable()
    for node in fx_module_with_metadata.graph.nodes:
        # print(f"Export {node}, {node.target}:")
        # print(g)
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
            v.setType(torch._C.TensorType.create_from_tensor(current_attr))
            assert (
                v is not None
            ), f"Node creates None with target={node.target} and name={node.name}"
            fx_name_to_ts_value[node.name] = v
            ts_name_to_real_tensor[v.debugName()] = current_attr
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


def _ts_graph_to_onnx_model_in_protobuf(
    ts_graph, ts_name_to_real_tensor, opset_version
):
    proto, _, _, _ = ts_graph._export_onnx(
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


def get_innermost_fake_tensor_mode():
    """
    This function inspects Pytorch's mode stack found by
    _get_current_dispatch_mode_stack(...) and return the innermost
    FakeTensorMode (or None if no FakeTensorMode is found).
    It also ensures the uniqueness of FakeTensorMode in that mode stack.
    """
    number_of_fake_tensor_modes = 0
    # The innermost FakeTensorMode.
    fake_tensor_mode = None
    for mode in torch.utils._python_dispatch._get_current_dispatch_mode_stack():
        if isinstance(mode, fake_tensor.FakeTensorMode):
            number_of_fake_tensor_modes += 1
            fake_tensor_mode = mode
    # Recursive FakeTensorMode's easily leads to runtime error.
    assert number_of_fake_tensor_modes <= 1
    # Return the innermost FakeTensorMode found. Otherwise, reture None.
    return fake_tensor_mode


def shape_inference_with_fake_tensor(decomposed_module: torch.fx.GraphModule, *args):
    # Use this FakeTensorMode to
    # 1. convert nn.Parameter's in nn.Module to FakeTensor
    # 2. run FakeTensorProp
    # If (1) and (2) are done with difference FakeTensorMode's, undefined behavior may
    # happen.
    fake_tensor_mode = get_innermost_fake_tensor_mode()
    if fake_tensor_mode is None:
        # Create a temporary FakeTensorMode for FakeTensorProp.
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


def _export(
    module: torch.fx.GraphModule,
    opset_version=None,
    *args,
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = None,
    use_binary_format: bool = True,
):
    # Export FX graph to ONNX ModelProto.
    if decomposition_table is None:
        # Use default decomposition table.
        decomposition_table = _ONNX_FRIENDLY_DECOMPOSITION_TABLE
    # Apply decomposition table to the input graph.
    decomposed_module = proxy_tensor.make_fx(
        module, decomposition_table=decomposition_table, tracing_mode="fake", _allow_non_fake_inputs=True)(*args)

    decomposed_module = shape_inference_with_fake_tensor(decomposed_module, *args)

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
    model_proto = onnx.ModelProto.FromString(onnx_model)
    return model_proto


def export(
    fn: Union[torch.nn.Module, Callable],
    opset_version,
    *args,
    use_binary_format: bool = True,
):
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


def export_without_kwargs(
    fn: Union[torch.nn.Module, Callable],
    opset_version,
    *args,
    use_binary_format: bool = True,
    **kwargs,
):
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
            self.captured_graph: Optional[torch.fx.GraphModule] = None
            self.captured_graph_count = 0

        def compile(self, gm: torch.fx.GraphModule, _):
            assert self.captured_graph_count == 0
            self.captured_graph = gm
            self.captured_graph_count += 1
            return gm

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


def _move_placeholder_to_front(graph_module: torch.fx.GraphModule) -> None:
    """
    This function move all placeholder nodes to the front of the graph node list.
    In torch.fx.Graph, placehoder is a special assignment node. If it's not
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


def _replace_get_attr_with_placeholder(graph_module: torch.fx.GraphModule):
    """
    Replace get_attr with placeholder.
    """
    graph = graph_module.graph
    replaced_attrs = []
    for node in graph.nodes:
        if node.op == "get_attr":
            replaced_attr = None
            try:
                replaced_attr = graph_module.get_parameter(node.target)
            except:
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

    return replaced_attrs


def export_without_parameters_and_buffers(
    module: Union[torch.nn.Module, Callable],
    *args,
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = None,
    use_binary_format: bool = True,
    opset_version: int = None,
    # kwargs are the keyword arguments to call "module"; that is,
    # module(*args, **kwargs) must run.
    **kwargs,
):
    """
    This function export the input "module" into a stateless ONNX model.
    All parameters and buffer in "module" will be inputs of the generated
    ONNX model.
    """
    if opset_version is None:
        opset_version = torch.onnx._constants.ONNX_DEFAULT_OPSET
    if isinstance(module, torch.nn.Module):
        signature = inspect.signature(module.forward)
    else:
        signature = inspect.signature(module)

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
    #  concrete_args["x"] = torch.fx.PH
    #  concrete_args["b"] = 1
    # where "x" and "b" are argument names in "signature".
    concrete_args = {}
    for param_name, param_value in bound.arguments.items():
        if isinstance(param_value, torch.Tensor):
            # param_value can be, e.g., a real tensor or a fake tensor.
            # param_value is treated as substitable tensor symbol (aka placeholder).
            concrete_args[param_name] = torch.fx.PH
        else:
            concrete_args[param_name] = param_value

    graph_module = torch.fx.symbolic_trace(module, concrete_args=concrete_args)

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
    # Finalize the graph editting. This new graph_module is stateless now (i.e., contains no
    # parameters and buffers). To call it, run
    #  graph_module(*bound.args, *replaced_attrs)
    # Note that the original module (contains parameters and buffers) is called by
    #  module(*bound.args)
    graph_module.recompile()

    return _export(
        graph_module,
        opset_version,
        *bound.args,
        *replaced_attrs,
        decomposition_table=decomposition_table,
        use_binary_format=use_binary_format), graph_module, bound.args, replaced_attrs
