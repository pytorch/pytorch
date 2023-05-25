import io
import operator
from typing import Any, Dict, List, Optional, Tuple, Union

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


def export_operator(target, version) -> Operator:
    if isinstance(target, str):
        return Operator(name=target, version=version)
    elif target in _SYM_INT_OPS:
        return Operator(name=f"{target.__module__}.{target.__name__}", version=version)
    elif isinstance(target, torch._ops.HigherOrderOperator):
        return Operator(name=target.__name__, version=version)
    else:
        return Operator(name=str(target), version=version)


def export_call_spec(call_spec: Optional[ep.CallSpec]) -> CallSpec:
    if call_spec is None:
        return CallSpec(in_spec="", out_spec="")
    # TODO(angelayi): spec
    return CallSpec(in_spec="", out_spec="")


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


def serialize(exported_program: ep.ExportedProgram) -> Tuple[GraphModule, bytes]:
    return Serializer().serialize(exported_program)


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

            def _extract_faketensor(tensor_meta: TensorMeta):
                return FakeTensor(
                    fake_tensor_mode,
                    torch.empty(
                        # TODO Support dynamic shape.
                        tuple(s.as_int for s in tensor_meta.sizes),
                        dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                    ),
                    torch.device("cpu"),
                )

            node.meta["val"] = pytree.tree_map_only(
                TensorMeta, _extract_faketensor, val
            )
    return ep
