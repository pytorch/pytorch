import copy
import itertools
import operator
from typing import Callable, Dict

import functorch
import onnx

import torch
import torch._C
import torch._decomp
import torch._dynamo
import torch._ops
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.nn.utils import stateless
from torch.onnx._globals import GLOBALS as ONNX_GLOBALS
from torch.onnx._internal import registration


class GraphWrapper(torch._C.Graph):
    """A graph replacement of torch.onnx.utils.jit_utils.GraphContext

    Some symbolic_opset*.py functions requires extra information in
    addition to torch._C.Graph itself. GraphContext contains those
    information. GraphContext is not reused here because it introduces
    extra TorchScript dependency and we want to move away from it.
    """

    def __init__(self):
        super().__init__()
        self.opset = ONNX_GLOBALS.export_onnx_opset_version


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
        ts_value = g.op("Constant", value_t=torch.tensor(ts_value, dtype=torch.float))
    elif ts_value is None:
        ts_value = g.op("prim::Constant")
        ts_value.setType(torch._C.OptionalType.ofTensor())
    elif isinstance(ts_value, list) and all(
        isinstance(val, (float, int)) for val in ts_value
    ):
        pass
    else:
        raise RuntimeError(f"Unexpected type of fx_node_arg: {type(fx_node_arg)}")
    return ts_value


def _wrap_fx_args_as_ts_args(g, root, node, fx_name_to_ts_value):
    """Map all FX arguments of a node to arguments in TorchScript graph."""

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    complete_args = []
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


def _export_fx_to_ts(fx_module_with_metadata):
    # TODO(wechi): To get rid of TorchScript dependency,
    # "g" should just be onnx.GraphProto or an equivalent
    # data structure in ONNXScript.
    g = GraphWrapper()
    # In the following loop, a TorchScript graph is created to
    # represent the input FX graph with ONNX symbols (e.g., onnx::add).
    # To connect the values to nodes in the TorchScript graph, we maintian
    # fx_name_to_ts_value. Basically, we want to translate
    #   fx_tensor_x (type: torch.fx.Node) -> fx_node_1 -> fx_tensor_y (type: torch.fx.Node)
    # to
    #   fx_name_to_ts_value[fx_tensor_x.name] -> onnx_node_1 -> fx_name_to_ts_value[fx_tensor_y.name]
    fx_name_to_ts_value: Dict[str, torch._C.Value] = {}
    # Similar to fx_name_to_ts_value, we need a mapping fo real tensors (usually tensor parameters
    # in nn.Module). Note that TorchScript's cannot store real tensors; TorchScript values are all
    # symbolic. This is passed into ONNX ModelProto as the initializers.
    ts_name_to_real_tensor: Dict[str, torch.Tensor] = {}
    for node in fx_module_with_metadata.graph.nodes:
        if node.op == "placeholder":
            # Input of graph.
            v = g.addInput(node.name)
            v.setType(torch._C.TensorType.create_from_tensor(node.meta["val"]))
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
                symbolic_fn = symbolic_function_group.get(14)
                assert symbolic_fn is not None
                ts_args = _wrap_fx_args_as_ts_args(
                    g, fx_module_with_metadata, node, fx_name_to_ts_value
                )
                v = symbolic_fn(g, *ts_args)
                fx_name_to_ts_value[node.name] = v
            elif node.target == operator.getitem and isinstance(node.args, tuple):
                v = fx_name_to_ts_value[node.args[0].name][node.args[1]]
                fx_name_to_ts_value[node.name] = v
            else:
                raise RuntimeError(
                    "Unknown call_function target: {}".format(node.target)
                )
        elif node.op == "output":
            if isinstance(node.args[0], torch.fx.Node):
                g.registerOutput(fx_name_to_ts_value[node.args[0].name])
            else:
                for arg in node.args[0]:
                    g.registerOutput(fx_name_to_ts_value[arg.name])
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
            fx_name_to_ts_value[node.name] = v
            ts_name_to_real_tensor[v.debugName()] = current_attr
        else:
            # TODO(wechi): Support get_attr, call_module, call_method.
            raise RuntimeError("Found node type not defined in torch.fx: " + node.op)
    return g, ts_name_to_real_tensor


def _ts_graph_to_onnx_model_in_protobuf(ts_graph, ts_name_to_real_tensor):
    proto, _, _, _ = ts_graph._export_onnx(
        ts_name_to_real_tensor,
        ONNX_GLOBALS.export_onnx_opset_version,
        {},
        False,
        torch.onnx.OperatorExportTypes.ONNX,
        False,
        False,
        {},
        True,
        "",
        {},
    )

    return proto


def _export(
    module: torch.fx.GraphModule,
    *args,
    decomposition_table: Dict[torch._ops.OpOverload, Callable] = None,
    use_binary_format: bool = True,
):
    # Export FX graph to ONNX ModelProto.
    if decomposition_table is None:
        # Use default decomposition table.
        decomposition_table = torch._decomp.decomposition_table
    # Apply decomposition table to the input graph.
    decomposed_module = functorch.make_fx(module, decomposition_table)(*args)
    decomposed_module.print_readable()

    fake_tensor_mode = FakeTensorMode()

    def to_fake_tensor(x):
        if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor):
            return fake_tensor_mode.from_tensor(x)
        return x

    fake_parameters_and_buffers = {
        k: to_fake_tensor(v)
        for k, v in itertools.chain(module.named_parameters(), module.named_buffers())
    }
    decomposed_module.print_readable()

    with stateless._reparametrize_module(
        decomposed_module, fake_parameters_and_buffers
    ):
        # Assign output types and shapes to each node.
        # TODO(wechi): It's possible to get symbolic types (and shapes)
        # for each node's output. Consider to set "tracing_mode=symbolic"
        # when calling make_fx and then remove FakeTensorProp below.
        if isinstance(args, tuple):
            FakeTensorProp(decomposed_module).propagate(*args)
        else:
            FakeTensorProp(decomposed_module).propagate(*args)

    ts_graph, ts_initializers = _export_fx_to_ts(decomposed_module)
    # Export TorchScript graph to ONNX ModelProto.
    onnx_model = _ts_graph_to_onnx_model_in_protobuf(ts_graph, ts_initializers)
    if use_binary_format:
        # Return ModelProto in binary format.
        return onnx_model
    # Return ModelProto in readable format (printable).
    model_proto = onnx.ModelProto.FromString(onnx_model)
    return model_proto


def _export_function(fn: Callable, *args, use_binary_format: bool = True):
    # args will be converted to symbolic tensor. Let's copy to avoid side effects.
    args = copy.deepcopy(args)
    # Translate callable to FX graph.
    graph_module = functorch.make_fx(fn)(*args)
    # Export FX graph to ONNX ModelProto.
    return _export(
        graph_module,
        *args,
        decomposition_table=_ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
    )


def _export_module(module: torch.nn.Module, *args, use_binary_format: bool = True):
    # args will be converted to symbolic tensor. Let's copy to avoid side effects.
    args = copy.deepcopy(args)
    # Convert nn.Module to FX graph
    # TODO(wechi): There are several symbolic tracing mechanisms to convert
    # nn.Module to FX graph. We should choose the right one after they are
    # matured.
    graph_module, graph_guard = torch._dynamo.export(module, *args, aten_graph=True)
    # Export FX graph to ONNX ModelProto.
    return _export(
        graph_module,
        *args,
        decomposition_table=_ONNX_FRIENDLY_DECOMPOSITION_TABLE,
        use_binary_format=use_binary_format,
    )
