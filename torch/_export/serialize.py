import io
import logging
import operator
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv, is_concrete_int
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
import torch._export.exported_program as ep
from .serde.schema import (   # type: ignore[attr-defined]
    Argument,
    BackwardSignature,
    CallSpec,
    Device,
    Graph,
    GraphModule,
    GraphSignature,
    Layout,
    MemoryFormat,
    NamedArgument,
    Node,
    Operator,
    ScalarType,
    SymInt,
    SymIntArgument,
    TensorArgument,
    TensorMeta,
    TensorValue,
)


__all__ = ["convert_fake_tensor_to_tensor_meta", "convert_tensor_meta_to_fake_tensor"]


log = logging.getLogger(__name__)


class SerializeError(RuntimeError):
    pass

def _reverse_map(d):
    return {v: k for k, v in d.items()}


_TORCH_TO_SERIALIZE_DTYPE = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.int16: ScalarType.SHORT,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.float16: ScalarType.HALF,
    torch.float32: ScalarType.FLOAT,
    torch.float64: ScalarType.DOUBLE,
    torch.complex32: ScalarType.COMPLEXHALF,
    torch.complex64: ScalarType.COMPLEXFLOAT,
    torch.complex128: ScalarType.COMPLEXDOUBLE,
    torch.bool: ScalarType.BOOL,
    torch.bfloat16: ScalarType.BFLOAT16
}


_SERIALIZE_TO_TORCH_DTYPE = _reverse_map(_TORCH_TO_SERIALIZE_DTYPE)


_TORCH_TO_SERIALIZE_LAYOUT = {
    torch.sparse_coo: Layout.SparseCoo,
    torch.sparse_csr: Layout.SparseCsr,
    torch.sparse_csc: Layout.SparseCsc,
    torch.sparse_bsr: Layout.SparseBsr,
    torch.sparse_bsc: Layout.SparseBsc,
    torch._mkldnn: Layout._mkldnn,  # type: ignore[attr-defined]
    torch.strided: Layout.Strided,
}


_SERIALIZE_TO_TORCH_LAYOUT = _reverse_map(_TORCH_TO_SERIALIZE_LAYOUT)


_TORCH_TO_SERIALIZE_MEMORY_FORMAT = {
    torch.contiguous_format: MemoryFormat.ContiguousFormat,
    torch.channels_last: MemoryFormat.ChannelsLast,
    torch.channels_last_3d: MemoryFormat.ChannelsLast3d,
    torch.preserve_format: MemoryFormat.PreserveFormat,
}


_SERIALIZE_TO_TORCH_MEMORY_FORMAT = _reverse_map(_TORCH_TO_SERIALIZE_MEMORY_FORMAT)


_SYM_INT_OPS = {
    operator.mul,
    operator.add,
    operator.sub,
    operator.floordiv,
    operator.mod,
}


def import_device(d: Device) -> torch.device:
    if d.index is None:
        return torch.device(type=d.type)  # type: ignore[call-overload]
    return torch.device(type=d.type, index=d.index)


def export_sym_int(s: Union[int, torch.SymInt]) -> SymInt:
    if isinstance(s, int):
        return SymInt.create(as_int=s)
    elif isinstance(s, torch.SymInt):
        if is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            return SymInt.create(as_symbol=str(s))
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )


def export_tensor_meta(t: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `t`.
    """
    return TensorMeta(
        dtype=_TORCH_TO_SERIALIZE_DTYPE[t.dtype],
        sizes=[export_sym_int(s) for s in t.shape],
        requires_grad=t.requires_grad,
        device=Device(type=t.device.type, index=t.device.index),
        strides=[export_sym_int(s) for s in t.stride()],
        storage_offset=0,
        layout=_TORCH_TO_SERIALIZE_LAYOUT[t.layout],
    )


def import_tensor_meta(tensor_meta: TensorMeta, fake_tensor_mode: FakeTensorMode) -> FakeTensor:
    with fake_tensor_mode:
        return cast(
            FakeTensor,
            torch.empty_strided(
                tuple([val.as_int for val in tensor_meta.sizes]),
                tuple([val.as_int for val in tensor_meta.strides]),
                device=import_device(tensor_meta.device),
                dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
            ),
        )


def export_metadata(node: torch.fx.Node) -> Dict[str, str]:
    ret = {}
    if stack_trace := node.meta.get("stack_trace"):
        ret["stack_trace"] = stack_trace
    module_fqn = node.meta.get("module_fqn")
    # Need an explicit None check instead of walrus operator, because
    # module_fqn can be the empty string if the node belongs to the root.
    # The walrus operator returns False on an empty string :(
    if module_fqn is not None:
        ret["module_fqn"] = module_fqn
    # TODO(angelayi) add nn_module_stack and source_fn
    return ret


def import_metadata(metadata) -> Dict[str, str]:
    ret = {}
    if stack_trace := metadata.get("stack_trace"):
        ret["stack_trace"] = stack_trace
    # Need an explicit None check instead of walrus operator, because
    # module_fqn can be the empty string if the node belongs to the root.
    # The walrus operator returns False on an empty string :(
    module_fqn = metadata.get("module_fqn")
    if module_fqn is not None:
        ret["module_fqn"] = module_fqn
    # TODO(angelayi) add nn_module_stack and source_fn
    return ret


def export_operator(target, version) -> Operator:
    if isinstance(target, str):
        return Operator(name=target, version=version)
    elif target in _SYM_INT_OPS:
        return Operator(name=f"{target.__module__}.{target.__name__}", version=version)
    elif isinstance(target, torch._ops.HigherOrderOperator):
        return Operator(name=target.__name__, version=version)
    else:
        return Operator(name=str(target), version=version)


def import_operator(serialized_target, op_version):
    if serialized_target.version != op_version:
        raise SerializeError(
            f"Target {serialized_target.name} had op version {serialized_target.version} "
            f"but the existing op version is now {op_version}"
        )
    target = torch.ops
    for name in serialized_target.name.split("."):
        if not hasattr(target, name):
            log.warning(f"Could not find operator {serialized_target}. Returning target as string.")  # noqa: G004
            return serialized_target
        else:
            target = getattr(target, name)
    return target


def export_call_spec(call_spec: Optional[ep.CallSpec]) -> CallSpec:
    if call_spec is None:
        return CallSpec(in_spec="", out_spec="")
    # TODO(angelayi): spec
    return CallSpec(in_spec="", out_spec="")


def import_call_spec(call_spec: CallSpec) -> Optional[ep.CallSpec]:
    # TODO(angelayi): spec
    return None


def export_signature(sig: Optional[ep.GraphSignature]) -> GraphSignature:
    if sig is None:
        return GraphSignature(
            inputs_to_parameters={},
            inputs_to_buffers={},
            user_inputs=[],
            user_outputs=[],
            buffers_to_mutate={},
            backward_signature=None,
        )

    if bw_sig := sig.backward_signature:
        backward_signature = BackwardSignature(
            gradients_to_parameters=bw_sig.gradients_to_parameters,
            gradients_to_user_inputs=bw_sig.gradients_to_user_inputs,
            loss_output=bw_sig.loss_output,
        )
    else:
        backward_signature = None

    graph_signature = GraphSignature(
        inputs_to_parameters=sig.inputs_to_parameters,
        inputs_to_buffers=sig.inputs_to_buffers,
        user_inputs=sig.user_inputs,
        user_outputs=sig.user_outputs,
        buffers_to_mutate=sig.buffers_to_mutate,
        backward_signature=backward_signature,
    )
    return graph_signature


def import_signature(sig: GraphSignature) -> ep.GraphSignature:
    backward_signature = None
    if bw_sig := sig.backward_signature:
        backward_signature = ep.BackwardSignature(
            gradients_to_parameters=dict(bw_sig.gradients_to_parameters),
            gradients_to_user_inputs=dict(bw_sig.gradients_to_user_inputs),
            loss_output=bw_sig.loss_output,
        )
    return ep.GraphSignature(
        parameters=list(sig.inputs_to_parameters.values()),
        buffers=list(sig.inputs_to_buffers.values()),
        user_inputs=list(sig.user_inputs),
        user_outputs=list(sig.user_outputs),
        inputs_to_buffers=dict(sig.inputs_to_buffers),
        inputs_to_parameters=dict(sig.inputs_to_parameters),
        buffers_to_mutate=dict(sig.buffers_to_mutate),
        backward_signature=backward_signature,
    )


def export_state_dict(state_dict: Optional[Dict[str, Any]]) -> bytes:
    if state_dict is None:
        return bytes("", encoding="utf8")

    buffer = io.BytesIO()
    state_dict = dict(state_dict)
    for name in state_dict:
        # This is a workaround for backend's tensor deserialization problem:
        # unpickleTensor() always create a tensor on the device where it was originally saved
        # This behavior is bad for multi-gpu training, as we wish to directly load the tensor
        # on the designated device.
        # For now, we simply move the tensor to cpu before saving.
        # TODO: this should be fixed by deserialization instead.
        state_dict[name] = state_dict[name].cpu()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def import_state_dict(serialized: bytes) -> Dict[str, torch.Tensor]:
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    return torch.load(buffer)


def _is_single_tensor_return(target: torch._ops.OpOverload) -> bool:
    returns = target._schema.returns
    return len(returns) == 1 and isinstance(returns[0].real_type, torch.TensorType)


class Serializer:
    def __init__(self, op_version: int = 0):
        self.inputs: List[Argument] = []
        self.outputs: List[Argument] = []
        self.nodes: List[Node] = []
        self.tensor_values: Dict[str, TensorValue] = {}
        self.sym_int_values: Dict[str, SymInt] = {}
        self.op_version: int = op_version

    def handle_placeholder(self, node: torch.fx.Node):
        assert node.op == "placeholder"
        self.inputs.append(Argument.create(as_tensor=TensorArgument(name=node.name)))

        self.tensor_values[node.name] = TensorValue(
            meta=export_tensor_meta(node.meta["val"])
        )

    def handle_output(self, node: torch.fx.Node):
        assert node.op == "output"
        assert len(node.args) == 1, "FX.Node's args should have one arg"
        node_args = node.args[0]
        assert isinstance(node_args, list)
        self.outputs = [self.export_input(arg) for arg in node_args]

    def handle_call_function(self, node: torch.fx.Node):
        assert node.op == "call_function"

        # getitem has been handled in the producer node, skip it here
        if node.target is operator.getitem:
            return

        if node.target is torch.set_grad_enabled:
            # Hack for torch.no_grad support. In the long run this should become
            # a higher order op but this is fine for now. See [NOTE: nograd support]
            ex_node = Node(
                target=export_operator("torch.set_grad_enabled", self.op_version),
                inputs=[NamedArgument(name="arg", arg=self.export_input(node.args[0]))],
                outputs=[],
                metadata=export_metadata(node),
            )
        elif node.target in _SYM_INT_OPS:
            assert len(node.kwargs) == 0
            meta_val = node.meta["val"]
            ex_node = Node(
                target=export_operator(node.target, self.op_version),
                inputs=self.export_sym_int_op_inputs(node.args),
                outputs=[Argument.create(as_sym_int=self.export_sym_int_output(node.name, meta_val))],
                metadata=export_metadata(node),
            )
        elif isinstance(node.target, torch._ops.OpOverload):
            ex_node = Node(
                target=export_operator(node.target, self.op_version),
                inputs=self.export_inputs(node.target, node.args, node.kwargs),
                outputs=self.export_outputs(node),
                # TODO: create a new tensor_values here, meta might have faketensor info
                metadata=export_metadata(node),
            )
        else:
            # TODO(angelayi) Higher order ops
            raise SerializeError(f"Serializing {node.target} is not supported")

        self.nodes.append(ex_node)

    def handle_get_attr(self, node):
        pass

    def export_sym_int_op_inputs(self, args) -> List[NamedArgument]:
        serialized_args = []
        args_names = ["a", "b"]
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(name=args_name, arg=self.export_input(arg))
            )
        return serialized_args

    def export_inputs(
        self, target: torch._ops.OpOverload, args, kwargs
    ) -> List[NamedArgument]:
        assert isinstance(target, torch._ops.OpOverload)
        serialized_args = []
        for i, schema_arg in enumerate(target._schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.export_input(kwargs[schema_arg.name]),
                    )
                )
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.export_input(args[i]),
                    )
                )
            else:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.export_input(schema_arg.default_value),
                    )
                )

        return serialized_args

    def is_sym_int_arg(self, arg) -> bool:
        return isinstance(arg, int) or (
            isinstance(arg, torch.fx.Node) and arg.name in self.sym_int_values
        )

    def export_input(self, arg) -> Argument:
        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                return Argument.create(as_tensor=TensorArgument(name=str(arg.target)))
            elif self.is_sym_int_arg(arg):
                return Argument.create(as_sym_int=SymIntArgument.create(asName=arg.name))
            else:
                return Argument.create(as_tensor=TensorArgument(name=arg.name))
        elif isinstance(arg, bool):
            return Argument.create(as_bool=arg)
        elif isinstance(arg, str):
            return Argument.create(as_string=arg)
        elif isinstance(arg, int):
            return Argument.create(as_int=arg)
        elif isinstance(arg, float):
            return Argument.create(as_float=arg)
        elif arg is None:
            return Argument.create(as_none=())
        elif isinstance(arg, (list, tuple)):
            # Must check bool first, as bool is also treated as int
            if all(isinstance(a, bool) for a in arg):
                return Argument.create(as_bools=list(arg))
            elif all(isinstance(a, int) for a in arg):
                return Argument.create(as_ints=list(arg))
            elif all(isinstance(a, float) for a in arg):
                return Argument.create(as_floats=list(arg))
            elif all(self.is_sym_int_arg(a) for a in arg):
                # list of sym_ints
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymIntArgument.create(as_name=a.name))
                    elif isinstance(a, int):
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(isinstance(a, torch.fx.Node) for a in arg):
                # list of tensors
                return Argument.create(
                    as_tensors=[TensorArgument(name=a.name) for a in arg],
                )
            else:
                raise SerializeError(f"Unsupported list/tuple argument type: {type(arg)}")
        elif isinstance(arg, torch.dtype):
            return Argument.create(as_scalar_type=_TORCH_TO_SERIALIZE_DTYPE[arg])
        elif isinstance(arg, torch.device):
            return Argument.create(as_device=Device(type=arg.type, index=arg.index))
        elif isinstance(arg, torch.memory_format):
            return Argument.create(as_memory_format=_TORCH_TO_SERIALIZE_MEMORY_FORMAT[arg])
        elif isinstance(arg, torch.layout):
            return Argument.create(as_layout=_TORCH_TO_SERIALIZE_LAYOUT[arg])
        else:
            raise SerializeError(f"Unsupported argument type: {type(arg)}")

    def export_tensor_output(self, name, meta_val) -> TensorArgument:
        assert name not in self.tensor_values
        self.tensor_values[name] = TensorValue(meta=export_tensor_meta(meta_val))
        return TensorArgument(name=name)

    def export_sym_int_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.sym_int_values
        self.sym_int_values[name] = export_sym_int(meta_val)
        return SymIntArgument.create(as_name=name)

    def export_outputs(self, node: torch.fx.Node) -> List[Argument]:
        """For a given node, return the dataclass representing its output values.

        [NOTE: Multiple outputs] We handle aggregates differently than FX. For
        FX, it looks like:

            x = call_function("multiple_return", ...)
            element0 = call_function(getitem, x, 0)
            foo = call_function("use_output", element0)

        We do not want the intermediate `getitem` call, so our serialized thing looks like:

            element0, element1, element2 = call_function("multiple_return", ...)
            foo = call_function("use_output", element0)

        We want names to be consistent across these two schemes, so that we can
        mostly reuse the names coming from FX. This function computes a mapping from
        the FX representation to our representation, preserving the names.
        """
        assert node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)

        meta_val = node.meta["val"]

        assert isinstance(node.target, torch._ops.OpOverload)
        returns = node.target._schema.returns

        # Check single value return
        if _is_single_tensor_return(node.target):
            return [Argument.create(as_tensor=self.export_tensor_output(node.name, meta_val))]
        elif len(returns) == 1 and isinstance(returns[0].real_type, torch.SymIntType):  # type: ignore[attr-defined]
            return [Argument.create(as_sym_int=self.export_sym_int_output(node.name, meta_val))]

        # There are a two possibilities at this point:
        # - This operator returns a list of Tensors.
        # - This operator returns multiple Tensors.
        #
        # Either way, start by gathering a list of TensorArguments with the correct names.
        # For consistent naming with FX, consult the downstream `getitem` node and
        # make sure our outputs have the same name.
        idx_to_name = {}
        for user in node.users:
            assert user.target is operator.getitem, f"User node {user} of {node} is incorrect"
            idx_to_name[user.args[1]] = user.name

        for idx, _ in enumerate(meta_val):
            # FX does not emit a getitem node for any outputs that are unused.
            # However, we need a name for them so that the number of outputs will
            # correctly match the schema. Just assign a dummy name.
            if idx not in idx_to_name:
                idx_to_name[idx] = f"{node.name}_unused_{idx}"

        arg_list = []
        for i, element_meta_val in enumerate(meta_val):
            arg_list.append(
                self.export_tensor_output(idx_to_name[i], element_meta_val)
            )

        # Then, pack the return value differently depending on what the return type is.
        if len(returns) == 1:
            return_type = returns[0].real_type
            assert isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            ), "Only tensors and lists of tensors supported"

            return [Argument.create(as_tensors=arg_list)]
        else:
            assert all(
                isinstance(ret.real_type, torch.TensorType) for ret in returns
            ), f"Multiple returns can only have tensor returns, got: {[ret.real_type for ret in returns]}"

            return [Argument.create(as_tensor=arg) for arg in arg_list]

    def serialize(self, exported_program: ep.ExportedProgram) -> Tuple[GraphModule, bytes]:
        for node in exported_program.graph.nodes:
            try:
                self.node = node
                getattr(self, f"handle_{node.op}")(node)
            except Exception as e:
                if not isinstance(e, SerializeError):
                    raise SerializeError(f"Failed serializing node {node}") from e

        graph = Graph(
            inputs=self.inputs,
            nodes=self.nodes,
            tensor_values=self.tensor_values,
            sym_int_values=self.sym_int_values,
            outputs=self.outputs,
        )

        # TODO(angelayi): I forgot where this belongs
        buffers = {}
        parameters = {}
        for name, buffer in exported_program.graph_module.named_buffers():
            buffers[name] = export_tensor_meta(buffer)
        for name, parameter in exported_program.graph_module.named_parameters():
            parameters[name] = export_tensor_meta(parameter)


        # TODO(angelayi): Graph Module metadata?
        metadata: Dict[str, str] = {}

        return (
            GraphModule(
                graph=graph,
                buffers=buffers,
                parameters=parameters,
                metadata=metadata,
                signature=export_signature(exported_program.graph_signature),
                call_spec=export_call_spec(exported_program.call_spec),
            ),
            export_state_dict(exported_program.state_dict),
        )


class Deserializer:
    def __init__(self, op_version: int = 0):
        self.serialized_name_to_node: Dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: Dict[str, FakeTensor] = {}
        self.graph = torch.fx.Graph()
        self.fake_tensor_mode = FakeTensorMode()
        self.op_version = op_version

    def deserialize(
        self, serialized_graph_module: GraphModule, serialized_state_dict: bytes
    ) -> ep.ExportedProgram:
        graph = self.graph
        serialized_graph = serialized_graph_module.graph

        # Handle the tensor metas.
        for name, tensor_value in serialized_graph.tensor_values.items():
            meta_val = import_tensor_meta(tensor_value.meta, self.fake_tensor_mode)
            self.serialized_name_to_meta[name] = meta_val

        # Inputs: convert to placeholder nodes in FX.
        for input in serialized_graph.inputs:
            placeholder_node = graph.placeholder(input.as_tensor.name)
            self.sync_serialized_node(input.as_tensor.name, placeholder_node)

        # Nodes: convert to call_function nodes.
        for serialized_node in serialized_graph.nodes:
            if serialized_node.target.name == "torch.set_grad_enabled":
                # Hack for torch.no_grad support. In the long run this should become
                # a higher order op but this is fine for now. See [NOTE: nograd support]
                fx_node = graph.call_function(
                    torch.set_grad_enabled,
                    (self.import_input(serialized_node.inputs[0].arg),),
                )
                fx_node.meta.update(import_metadata(serialized_node.metadata))
                continue

            target = import_operator(serialized_node.target, self.op_version)

            # For convenience: if this node returns a single tensor, name the
            # newly-created node after it. This ensures that these tensor values
            # have names that are consistent with serialized.
            name = (
                serialized_node.outputs[0].value.name
                if _is_single_tensor_return(target)
                else None  # FX will generate a name for us.
            )
            args, kwargs = self.import_inputs(target, serialized_node)

            fx_node = graph.create_node("call_function", target, args, kwargs, name)

            self.import_outputs(serialized_node, fx_node)

            fx_node.meta.update(import_metadata(serialized_node.metadata))

        # Outputs: convert to a single `output` node.
        outputs = []
        for output in serialized_graph.outputs:
            assert isinstance(output.value, TensorArgument)
            outputs.append(self.serialized_name_to_node[output.value.name])

        graph.output(tuple(outputs) if len(outputs) > 1 else outputs[0])

        sig = import_signature(serialized_graph_module.signature)
        call_spec = import_call_spec(serialized_graph_module.call_spec)
        state_dict = import_state_dict(serialized_state_dict)

        return ep.ExportedProgram(state_dict, graph, sig, call_spec, state_dict)

    def sync_serialized_node(self, name: str, fx_node: torch.fx.Node):
        self.serialized_name_to_node[name] = fx_node
        fx_node.meta["val"] = self.serialized_name_to_meta[name]

    def import_inputs(self, target: torch._ops.OpOverload, serialized_node: Node):
        schema_args = target._schema.arguments
        actual_args = {
            input.name: self.import_input(input.arg) for input in serialized_node.inputs
        }
        args = []
        kwargs = {}
        for schema_arg in schema_args:
            is_positional = not schema_arg.has_default_value()
            if is_positional:
                args.append(actual_args[schema_arg.name])
            else:
                if schema_arg.name in actual_args:
                    kwargs[schema_arg.name] = actual_args[schema_arg.name]
        return tuple(args), kwargs

    def import_input(self, value: Argument) -> Any:
        type_ = value.type
        if type_ == Argument.fields().as_none:
            # None should converted as None, but is encoded as bool in serialized
            # Convert serialized object to torch equivalent
            return None
        elif type_ == Argument.fields().as_tensor:
            return self.serialized_name_to_node[value.as_tensor.name]
        elif type_ == Argument.fields().as_tensors:
            return [self.serialized_name_to_node[arg.name] for arg in value.as_tensors]
        elif type_ == Argument.fields().as_int:
            return value.as_int
        elif type_ == Argument.fields().as_ints:
            # convert from serialized.python.types.List to python list
            return list(value.as_ints)
        elif type_ == Argument.fields().as_float:
            return value.as_float
        elif type_ == Argument.fields().as_floats:
            # convert from serialized.python.types.List to python list
            return list(value.as_floats)
        elif type_ == Argument.fields().as_string:
            return str(value.as_string)
        elif type_ == Argument.fields().as_sym_int or type_ == Argument.fields().as_sym_ints:
            raise ValueError("Symints not yet supported")
        elif type_ == Argument.fields().as_scalar_type:
            return _SERIALIZE_TO_TORCH_DTYPE[value.as_scalar_type]
        elif type_ == Argument.fields().as_memory_format:
            return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[value.as_memory_format]
        elif type_ == Argument.fields().as_layout:
            return _SERIALIZE_TO_TORCH_LAYOUT[value.as_layout]
        elif type_ == Argument.fields().as_device:
            return import_device(value.as_device),
        elif type_ == Argument.fields().as_bool:
            return value.as_bool
        elif type_ == Argument.fields().as_bools:
            # convert from serialized.python.types.List to python list
            return list(value.as_bools)
        else:
            raise SerializeError("Unhandled argument type:", type_)

    def import_outputs(self, serialized_node: Node, fx_node: torch.fx.Node) -> None:
        # Simple case for single tensor return.
        assert isinstance(fx_node.target, torch._ops.OpOverload)
        if _is_single_tensor_return(fx_node.target):
            return self.sync_serialized_node(serialized_node.outputs[0].as_tensor.name, fx_node)

        # Convert multiple return types to FX format.
        # In FX, each node only returns one value. So in order to represent
        # multiple return values, we have to emit a `getitem` node for each
        # return value.
        # This performs the inverse mapping of the `export_outputs` call in
        # serialization, see [NOTE: Multiple outputs]
        output_names = []
        if len(serialized_node.outputs) == 1:
            assert serialized_node.outputs[0].type == Argument.fields().as_tensors
            output_names = [arg.name for arg in serialized_node.outputs[0].as_tensors]
        else:
            for output in serialized_node.outputs:
                assert output.type == Argument.fields().as_tensor
                output_names.append(output.as_tensor.name)

        for idx, name in enumerate(output_names):
            individual_output = self.graph.create_node(
                "call_function",
                operator.getitem,
                (fx_node, idx),
                name=name,
            )
            self.sync_serialized_node(name, individual_output)
            # The derived `getitem` nodes should have the same stacktrace as the
            # original `fx_node`
            individual_output.meta.update(import_metadata(serialized_node.metadata))

        # also update the metaval for `fx_node` to be a list(meta)
        fx_node.meta["val"] = [self.serialized_name_to_meta[name] for name in output_names]


def serialize(exported_program: ep.ExportedProgram) -> Tuple[GraphModule, bytes]:
    return Serializer().serialize(exported_program)


def deserialize(serialized_graph_module: GraphModule, state_dict: bytes) -> ep.ExportedProgram:
    return Deserializer().deserialize(serialized_graph_module, state_dict)

###################################################################################################################


def convert_fake_tensor_to_tensor_meta(
    ep: ep.ExportedProgram
) -> Tuple[ep.ExportedProgram, Optional[ShapeEnv]]:
    """
    Replace the faketensor metadata with the tensor metadata dataclass since we
    cannot serialize faketensors
    """
    shape_env = None
    for node in ep.graph.nodes:
        def get_shape_env(val) -> Optional[ShapeEnv]:
            val_flat, _ = pytree.tree_flatten(val)
            curr_shape_env = None
            for v in val_flat:
                if not isinstance(v, FakeTensor):
                    continue
                if curr_shape_env is None:
                    curr_shape_env = v.fake_mode.shape_env
                else:
                    assert (
                        curr_shape_env is v.fake_mode.shape_env
                    ), "Multiple shape envs detected."
            return curr_shape_env

        if (val := node.meta.get("val", None)) is not None:
            if shape_env is None:
                shape_env = get_shape_env(val)
            elif (new_shape_env := get_shape_env(val)) is not None:
                assert (
                    shape_env is new_shape_env
                ), "Multiple shape envs detected."

            node.meta["tensor_meta"] = pytree.tree_map_only(
                torch.Tensor, export_tensor_meta, val
            )
            del node.meta["val"]

    return ep, shape_env


def convert_tensor_meta_to_fake_tensor(ep: ep.ExportedProgram, shape_env: ShapeEnv = None) -> ep.ExportedProgram:
    """
    Replace (inplace) the tensor metadata with faketensor
    """
    fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)
    for node in ep.graph.nodes:
        if (val := node.meta.get("tensor_meta", None)) is not None:
            node.meta["val"] = pytree.tree_map_only(
                TensorMeta, lambda v: import_tensor_meta(v, fake_tensor_mode), val
            )
    return ep
