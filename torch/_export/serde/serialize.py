import base64
import dataclasses
import io
import json
import logging
import math
import operator
import pickle
import typing

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple, Union

import sympy

import torch
import torch._export.exported_program as ep
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads, tree_map_only

from .schema import (  # type: ignore[attr-defined]
    _Union,
    Argument,
    BackwardSignature,
    CallSpec,
    CustomObjArgument,
    Device,
    ExportedProgram,
    Graph,
    GraphArgument,
    GraphModule,
    GraphSignature,
    Layout,
    MemoryFormat,
    ModuleCallEntry,
    ModuleCallSignature,
    NamedArgument,
    Node,
    OptionalTensorArgument,
    RangeConstraint,
    ScalarType,
    SCHEMA_VERSION,
    SymBool,
    SymBoolArgument,
    SymExpr,
    SymInt,
    SymIntArgument,
    TensorArgument,
    TensorMeta,
    TensorValue,
    TREESPEC_VERSION,
)


__all__ = [
    "serialize",
    "GraphModuleSerializer",
    "ExportedProgramSerializer",
    "GraphModuleDeserializer",
    "ExportedProgramDeserializer",
]

from .upgrade import GraphModuleOpUpgrader

log = logging.getLogger(__name__)


class SerializeError(RuntimeError):
    pass


def _reverse_map(d: Dict[Any, Enum]):
    return {v.value: k for k, v in d.items()}


MetaType = Union[FakeTensor, int, torch.SymInt, bool, torch.SymBool]


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


_SERIALIZE_TO_TORCH_DTYPE = _reverse_map(_TORCH_TO_SERIALIZE_DTYPE)  # type: ignore[arg-type]


_TORCH_TO_SERIALIZE_LAYOUT = {
    torch.sparse_coo: Layout.SparseCoo,
    torch.sparse_csr: Layout.SparseCsr,
    torch.sparse_csc: Layout.SparseCsc,
    torch.sparse_bsr: Layout.SparseBsr,
    torch.sparse_bsc: Layout.SparseBsc,
    torch._mkldnn: Layout._mkldnn,  # type: ignore[attr-defined]
    torch.strided: Layout.Strided,
}


_SERIALIZE_TO_TORCH_LAYOUT = _reverse_map(_TORCH_TO_SERIALIZE_LAYOUT)  # type: ignore[arg-type]


_TORCH_TO_SERIALIZE_MEMORY_FORMAT = {
    torch.contiguous_format: MemoryFormat.ContiguousFormat,
    torch.channels_last: MemoryFormat.ChannelsLast,
    torch.channels_last_3d: MemoryFormat.ChannelsLast3d,
    torch.preserve_format: MemoryFormat.PreserveFormat,
}


_SERIALIZE_TO_TORCH_MEMORY_FORMAT = _reverse_map(_TORCH_TO_SERIALIZE_MEMORY_FORMAT)  # type: ignore[arg-type]


_SYM_INT_OPS = {
    operator.mul,
    operator.add,
    operator.sub,
    operator.floordiv,
    operator.mod,
}


_SYM_BOOL_OPS = {
    operator.eq,
    operator.ne,
    operator.le,
    operator.ge,
    operator.lt,
    operator.gt,
}


def deserialize_device(d: Device) -> torch.device:
    if d.index is None:
        return torch.device(type=d.type)  # type: ignore[call-overload]
    return torch.device(type=d.type, index=d.index)


def serialize_sym_int(s: Union[int, torch.SymInt]) -> SymInt:
    if isinstance(s, (torch.SymInt, int)):
        if symbolic_shapes.is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            assert isinstance(s, torch.SymInt)
            return SymInt.create(as_expr=SymExpr(str(s), s.node.hint))
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_bool(s: Union[bool, torch.SymBool]) -> SymBool:
    if isinstance(s, (torch.SymBool, bool)):
        if symbolic_shapes.is_concrete_bool(s):
            return SymBool.create(as_bool=bool(s))
        else:
            return SymBool.create(as_expr=str(s))
    else:
        raise SerializeError(
            f"SymBool should be either symbol or bool, got `{s}` of type `{type(s)}`"
        )


def serialize_tensor_meta(t: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `t`.
    """
    return TensorMeta(
        dtype=_TORCH_TO_SERIALIZE_DTYPE[t.dtype],
        sizes=[serialize_sym_int(s) for s in t.shape],
        requires_grad=t.requires_grad,
        device=Device(type=t.device.type, index=t.device.index),
        strides=[serialize_sym_int(s) for s in t.stride()],
        storage_offset=0,
        layout=_TORCH_TO_SERIALIZE_LAYOUT[t.layout],
    )

def serialize_call_spec(call_spec: ep.CallSpec) -> CallSpec:
    return CallSpec(
        in_spec=treespec_dumps(call_spec.in_spec, TREESPEC_VERSION) if call_spec.in_spec else "",
        out_spec=treespec_dumps(call_spec.out_spec, TREESPEC_VERSION) if call_spec.out_spec else "",
    )


def deserialize_call_spec(call_spec: CallSpec) -> ep.CallSpec:
    return ep.CallSpec(
        in_spec=treespec_loads(call_spec.in_spec) if call_spec.in_spec else None,
        out_spec=treespec_loads(call_spec.out_spec) if call_spec.out_spec else None,
    )


def serialize_signature(sig: ep.ExportGraphSignature) -> GraphSignature:
    if bw_sig := sig.backward_signature:
        backward_signature = BackwardSignature(
            gradients_to_parameters=bw_sig.gradients_to_parameters,
            gradients_to_user_inputs=bw_sig.gradients_to_user_inputs,
            loss_output=bw_sig.loss_output,
        )
    else:
        backward_signature = None

    graph_signature = GraphSignature(
        inputs_to_parameters=sig.inputs_to_parameters,  # type: ignore[arg-type]
        inputs_to_buffers=sig.inputs_to_buffers,  # type: ignore[arg-type]
        user_inputs=sig.user_inputs,  # type: ignore[arg-type]
        user_outputs=sig.user_outputs,  # type: ignore[arg-type]
        buffers_to_mutate=sig.buffers_to_mutate,  # type: ignore[arg-type]
        backward_signature=backward_signature,
    )
    return graph_signature


def deserialize_signature(sig: GraphSignature) -> ep.ExportGraphSignature:
    backward_signature = None
    if bw_sig := sig.backward_signature:
        backward_signature = ep.ExportBackwardSignature(
            gradients_to_parameters=dict(bw_sig.gradients_to_parameters),
            gradients_to_user_inputs=dict(bw_sig.gradients_to_user_inputs),
            loss_output=bw_sig.loss_output,
        )
    return ep.ExportGraphSignature(
        parameters=list(sig.inputs_to_parameters.values()),  # type: ignore[arg-type]
        buffers=list(sig.inputs_to_buffers.values()),  # type: ignore[arg-type]
        user_inputs=list(sig.user_inputs),  # type: ignore[arg-type]
        user_outputs=list(sig.user_outputs),  # type: ignore[arg-type]
        inputs_to_buffers=dict(sig.inputs_to_buffers),  # type: ignore[arg-type]
        inputs_to_parameters=dict(sig.inputs_to_parameters),  # type: ignore[arg-type]
        buffers_to_mutate=dict(sig.buffers_to_mutate),  # type: ignore[arg-type]
        backward_signature=backward_signature,
    )


def serialize_torch_artifact(artifact) -> bytes:
    buffer = io.BytesIO()
    # This is a workaround for backend's tensor deserialization problem:
    # unpickleTensor() always create a tensor on the device where it was originally saved
    # This behavior is bad for multi-gpu training, as we wish to directly load the tensor
    # on the designated device.
    # For now, we simply move the tensor to cpu before saving.
    # TODO: this should be fixed by deserialization instead.
    artifact = tree_map_only(torch.Tensor, lambda t: t.cpu(), artifact)
    torch.save(artifact, buffer)
    return buffer.getvalue()


def deserialize_torch_artifact(serialized: bytes):
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    return torch.load(buffer)


def _sympy_int_to_int(val: sympy.Expr):
    # Convert simple sympy Integers into concrete int
    if val == sympy.oo:
        return math.inf
    if val == -sympy.oo:
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError(
        "Export constraints cannot be non-integer expressions"
    )


def _int_to_sympy_int(val) -> sympy.Expr:
    # Convert concrete int into simple sympy Integers
    if val == math.inf:
        return sympy.oo
    if val == -math.inf:
        return -sympy.oo
    return sympy.Integer(val)


def serialize_range_constraints(
    range_constraints: Dict[sympy.Symbol, ep.RangeConstraint]
) -> Dict[str, RangeConstraint]:
    return {
        str(k): RangeConstraint(
            _sympy_int_to_int(v.min_val),
            _sympy_int_to_int(v.max_val),
        )
        for k, v in range_constraints.items()
    }


def serialize_equality_constraints(
    equality_constraints: List[Tuple[ep.InputDim, ep.InputDim]]
) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    return [
        ((v1.input_name, v1.dim), (v2.input_name, v2.dim))
        for (v1, v2) in equality_constraints
    ]


def deserialize_equality_constraints(
    equality_constraints: List[Tuple[Tuple[str, int], Tuple[str, int]]]
) -> List[Tuple[ep.InputDim, ep.InputDim]]:
    return [
        (ep.InputDim(v1[0], v1[1]), ep.InputDim(v2[0], v2[1]))
        for (v1, v2) in equality_constraints
    ]


def _is_single_tensor_return(target: torch._ops.OpOverload) -> bool:
    returns = target._schema.returns
    return len(returns) == 1 and isinstance(returns[0].real_type, torch.TensorType)


@dataclass
class GraphState:
    inputs: List[Argument] = field(default_factory=list)
    outputs: List[Argument] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    tensor_values: Dict[str, TensorValue] = field(default_factory=dict)
    sym_int_values: Dict[str, SymInt] = field(default_factory=dict)
    sym_bool_values: Dict[str, SymBool] = field(default_factory=dict)
    is_single_tensor_return: bool = False
    constants: Dict[str, torch.Tensor] = field(default_factory=dict)


class GraphModuleSerializer:
    def __init__(
        self,
        graph_signature: ep.ExportGraphSignature,
        call_spec: ep.CallSpec,
        module_call_graph: List[ep.ModuleCallEntry]
    ):
        self.graph_state = GraphState()
        self.graph_signature = graph_signature
        self.call_spec = call_spec
        self.module_call_graph = module_call_graph

    @contextmanager
    def save_graph_state(self):
        saved = self.graph_state
        self.graph_state = GraphState()
        try:
            yield
        finally:
            self.graph_state = saved

    def handle_placeholder(self, node: torch.fx.Node):
        assert node.op == "placeholder"
        self.graph_state.inputs.append(Argument.create(as_tensor=TensorArgument(name=node.name)))

        self.graph_state.tensor_values[node.name] = TensorValue(
            meta=serialize_tensor_meta(node.meta["val"])
        )

    def handle_output(self, node: torch.fx.Node):
        assert node.op == "output"
        assert len(node.args) == 1, "FX.Node's args should have one arg"
        node_args = node.args[0]
        if isinstance(node_args, torch.fx.Node):
            # For singleton tensor returns
            self.graph_state.is_single_tensor_return = True
            self.graph_state.outputs = [self.serialize_input(node_args)]
        else:
            assert isinstance(node_args, (tuple, list))
            self.graph_state.outputs = [self.serialize_input(arg) for arg in node_args]

    def serialize_operator(self, target) -> str:
        if isinstance(target, str):
            return target
        elif target.__module__.startswith("torch._ops"):
            # TODO(zhxchen17) Maybe provide a function name helper in FX.
            # From torch.fx.node._get_qualified_name
            module = target.__module__.replace("torch._ops", "torch.ops")
            return f"{module}.{target.__name__}"
        else:  # TODO(zhxchen17) Don't catch all here.
            return f"{target.__module__}.{target.__name__}"

    def handle_call_function(self, node: torch.fx.Node):
        assert node.op == "call_function"

        # getitem has been handled in the producer node, skip it here
        if node.target is operator.getitem:
            return

        if node.target in _SYM_INT_OPS:
            assert len(node.kwargs) == 0
            meta_val = node.meta["val"]
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_sym_op_inputs(node.args),
                outputs=[Argument.create(as_sym_int=self.serialize_sym_int_output(node.name, meta_val))],
                metadata=self.serialize_metadata(node),
            )
        elif node.target in _SYM_BOOL_OPS:
            assert len(node.kwargs) == 0
            meta_val = node.meta["val"]
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_sym_op_inputs(node.args),
                outputs=[Argument.create(as_sym_bool=self.serialize_sym_bool_output(node.name, meta_val))],
                metadata=self.serialize_metadata(node),
            )
        elif isinstance(node.target, torch._ops.OpOverload):
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                # TODO: create a new tensor_values here, meta might have faketensor info
                metadata=self.serialize_metadata(node),
            )
        elif isinstance(node.target, torch._ops.HigherOrderOperator):

            inputs = [
                NamedArgument(
                    name="",  # TODO(zhxchen17) This is sad, should be improved when HOO has schema arg names.
                    arg=self.serialize_input(a),
                ) for a in node.args
            ]

            meta_val = node.meta["val"]

            if isinstance(meta_val, torch.Tensor):
                outputs = [Argument.create(as_tensor=self.serialize_tensor_output(node.name, meta_val))]
            elif isinstance(meta_val, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in meta_val):
                arg_list = self._handle_getitem_users(node)
                outputs = [Argument.create(as_tensors=arg_list)]
            else:
                raise SerializeError(
                    "Only single tensor output or list of tensor output "
                    "is supported for HigherOrderOperator serialization"
                )

            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=inputs,
                outputs=outputs,
                metadata=self.serialize_metadata(node),
            )
        else:
            raise SerializeError(f"Serializing {node.target} is not supported")

        self.graph_state.nodes.append(ex_node)

    def handle_get_attr(self, node):
        pass

    def serialize_metadata(self, node: torch.fx.Node) -> Dict[str, str]:
        ret = {}
        if stack_trace := node.meta.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        if nn_module_stack := node.meta.get("nn_module_stack"):
            # Serialize to "fx_node_name:(orig_ref,type_str)"
            nn_module_list = [
                f"{k}:({v[0]},{self.serialize_operator(v[1])})"
                for k, v in nn_module_stack.items()
            ]
            ret["nn_module_stack"] = ";".join(nn_module_list)

        if source_fn := node.meta.get("source_fn"):
            # Serialize to "fx_node_name,op_str"
            op = self.serialize_operator(source_fn[1])
            ret["source_fn"] = f"{source_fn[0]},{op}"

        return ret

    def serialize_sym_op_inputs(self, args) -> List[NamedArgument]:
        serialized_args = []
        args_names = ["a", "b"]
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(name=args_name, arg=self.serialize_input(arg))
            )
        return serialized_args

    def serialize_inputs(
        self, target: torch._ops.OpOverload, args, kwargs=None
    ) -> List[NamedArgument]:
        assert isinstance(target, torch._ops.OpOverload)
        kwargs = kwargs or {}
        serialized_args = []
        for i, schema_arg in enumerate(target._schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(kwargs[schema_arg.name]),
                    )
                )
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(args[i]),
                    )
                )
            else:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(schema_arg.default_value),
                    )
                )

        return serialized_args

    def is_sym_int_arg(self, arg) -> bool:
        return isinstance(arg, int) or (
            isinstance(arg, torch.fx.Node) and arg.name in self.graph_state.sym_int_values
        )

    def is_sym_bool_arg(self, arg) -> bool:
        return isinstance(arg, bool) or (
            isinstance(arg, torch.fx.Node) and arg.name in self.graph_state.sym_bool_values
        )

    def serialize_input(self, arg) -> Argument:
        import torch._inductor.ir as inductor_ir
        inductor_tensor_buffers = (inductor_ir.InputBuffer, inductor_ir.ComputedBuffer, inductor_ir.ConcatKernel)

        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                assert isinstance(arg.target, str)
                attr = getattr(arg.graph.owning_module, arg.target)

                if isinstance(attr, torch.Tensor):
                    self.graph_state.constants[arg.name] = attr
                    return Argument.create(as_tensor=TensorArgument(name=arg.name))
                elif isinstance(attr, torch.fx.GraphModule):
                    with self.save_graph_state():
                        graph = self.serialize_graph(attr)
                    return Argument.create(as_graph=GraphArgument(name=arg.target, graph=graph))
                else:
                    raise SerializeError(f"Unsupported getattr attribute {arg.target} with type: {type(attr)}")
            elif self.is_sym_int_arg(arg):
                return Argument.create(as_sym_int=SymIntArgument.create(as_name=arg.name))
            elif self.is_sym_bool_arg(arg):
                return Argument.create(as_sym_bool=SymBoolArgument.create(as_name=arg.name))
            else:
                return Argument.create(as_tensor=TensorArgument(name=arg.name))
        elif isinstance(arg, inductor_tensor_buffers):
            # Other branches are for arguments in fx node.
            # This is a special branch for handling buffers (representing tensor arguments)
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            assert arg.name is not None, "Input buffer must have valid name"
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
            elif all(isinstance(a, str) for a in arg):
                return Argument.create(as_strings=list(arg))
            elif all(self.is_sym_int_arg(a) for a in arg):
                # list of sym_ints
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymIntArgument.create(as_name=a.name))
                    elif isinstance(a, int):
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(self.is_sym_bool_arg(a) for a in arg):
                # list of sym_bools
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymBoolArgument.create(as_name=a.name))
                    elif isinstance(a, bool):
                        values.append(SymBoolArgument.create(as_bool=a))
                return Argument.create(as_sym_bools=values)
            elif all(isinstance(a, torch.fx.Node) for a in arg):
                # list of tensors
                arguments = []
                for a in arg:
                    if a.op == "get_attr":
                        assert isinstance(a.target, str)
                        attr = getattr(a.graph.owning_module, a.target)
                        assert isinstance(attr, torch.Tensor)
                        self.graph_state.constants[a.name] = attr
                    arguments.append(TensorArgument(name=a.name))
                return Argument.create(as_tensors=arguments)
            elif all(isinstance(a, (torch.fx.Node, type(None))) for a in arg):
                # list of optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=())
                    elif isinstance(a, torch.fx.Node):
                        return OptionalTensorArgument.create(as_tensor=a.name)
                    else:
                        raise SerializeError(f"Unsupported list/tuple argument: {a}")
                return Argument.create(
                    as_optional_tensors=list(map(serialize_optional_tensor_args, arg))
                )
            elif all(isinstance(a, inductor_tensor_buffers) for a in arg):
                # list of tensors
                return Argument.create(
                    as_tensors=[TensorArgument(name=a.name) for a in arg],
                )
            elif all(isinstance(a, (*inductor_tensor_buffers, type(None))) for a in arg):
                # list of optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=())
                    elif isinstance(a, inductor_tensor_buffers):
                        return OptionalTensorArgument.create(as_tensor=a.name)
                    else:
                        raise SerializeError(f"Unsupported list/tuple argument: {a}")
                return Argument.create(
                    as_optional_tensors=list(map(serialize_optional_tensor_args, arg))
                )
            else:
                raise SerializeError(f"Unsupported list/tuple argument type: {[type(a) for a in arg]}")
        elif isinstance(arg, torch.dtype):
            return Argument.create(as_scalar_type=_TORCH_TO_SERIALIZE_DTYPE[arg])
        elif isinstance(arg, torch.device):
            return Argument.create(as_device=Device(type=arg.type, index=arg.index))
        elif isinstance(arg, torch.memory_format):
            return Argument.create(as_memory_format=_TORCH_TO_SERIALIZE_MEMORY_FORMAT[arg])
        elif isinstance(arg, torch.layout):
            return Argument.create(as_layout=_TORCH_TO_SERIALIZE_LAYOUT[arg])
        elif isinstance(arg, torch._C.ScriptObject):
            if not (
                hasattr(type(arg), "__getstate__") and
                hasattr(type(arg), "__setstate__")
            ):
                raise SerializeError(
                    f"Unable to serialize ScriptObject {arg}. Please define "
                    "serialization methods via def_pickle()."
                )
            # Custom objects through torchind are serializable with pickle,
            # through implementing the .def_pickle function.  This should result
            # in the object containing a __getstate__ and __setstate__
            # serialize/deserialize function.
            blob = pickle.dumps(arg)
            return Argument.create(as_custom_obj=CustomObjArgument(blob))
        else:
            raise SerializeError(f"Unsupported argument type: {type(arg)}")

    def serialize_tensor_output(self, name, meta_val) -> TensorArgument:
        assert name not in self.graph_state.tensor_values
        self.graph_state.tensor_values[name] = TensorValue(meta=serialize_tensor_meta(meta_val))
        return TensorArgument(name=name)

    def serialize_sym_int_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_int_values
        self.graph_state.sym_int_values[name] = serialize_sym_int(meta_val)
        return SymIntArgument.create(as_name=name)

    def serialize_sym_bool_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_bool_values
        self.graph_state.sym_bool_values[name] = serialize_sym_bool(meta_val)
        return SymBoolArgument.create(as_name=name)

    def serialize_module_call_signature(self, module_call_signature: ep.ModuleCallSignature) -> ModuleCallSignature:
        def serialize_argument(x: ep.ArgumentSpec) -> Argument:
            if x.kind == ep.ArgumentKind.Tensor:
                return Argument.create(as_tensor=TensorArgument(name=x.value))
            elif x.kind == ep.ArgumentKind.SymInt:
                return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.value))
            else:
                assert x.kind == ep.ArgumentKind.Constant
                return self.serialize_input(x.value)
        return ModuleCallSignature(
            inputs=[serialize_argument(x) for x in module_call_signature.inputs],
            outputs=[serialize_argument(x) for x in module_call_signature.outputs],
            in_spec=treespec_dumps(module_call_signature.in_spec, TREESPEC_VERSION),
            out_spec=treespec_dumps(module_call_signature.out_spec, TREESPEC_VERSION),
        )

    def serialize_module_call_graph(self, module_call_graph: List[ep.ModuleCallEntry]) -> List[ModuleCallEntry]:
        return [
            ModuleCallEntry(
                fqn=entry.fqn,
                signature=self.serialize_module_call_signature(entry.signature) if entry.signature else None,
            ) for entry in module_call_graph
        ]

    def serialize_outputs(self, node: torch.fx.Node) -> List[Argument]:
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

        assert isinstance(node.target, torch._ops.OpOverload)
        returns = node.target._schema.returns

        if len(returns) == 0:
            return []

        meta_val = node.meta["val"]

        # Check single value return
        if _is_single_tensor_return(node.target):
            return [Argument.create(as_tensor=self.serialize_tensor_output(node.name, meta_val))]
        elif len(returns) == 1 and isinstance(meta_val, torch.SymInt):
            return [Argument.create(as_sym_int=self.serialize_sym_int_output(node.name, meta_val))]
        elif len(returns) == 1 and isinstance(meta_val, torch.SymBool):
            return [Argument.create(as_sym_bool=self.serialize_sym_bool_output(node.name, meta_val))]

        # There are a two possibilities at this point:
        # - This operator returns a list of Tensors.
        # - This operator returns multiple Tensors.
        #
        # Either way, start by gathering a list of TensorArguments with the correct names.
        # For consistent naming with FX, consult the downstream `getitem` node and
        # make sure our outputs have the same name.

        arg_list = self._handle_getitem_users(node)

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

    def _handle_getitem_users(self, node: torch.fx.Node) -> List[TensorArgument]:
        meta_val = node.meta["val"]

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
                self.serialize_tensor_output(idx_to_name[i], element_meta_val)
            )

        return arg_list

    def serialize_graph(self, graph_module: torch.fx.GraphModule) -> Graph:
        assert isinstance(graph_module, torch.fx.GraphModule)
        for node in graph_module.graph.nodes:
            try:
                getattr(self, f"handle_{node.op}")(node)
            except Exception as e:
                raise SerializeError(f"Failed serializing node {node} in graph:\n{graph_module.graph}") from e

        serialized_constants = {
            k: serialize_torch_artifact(v)
            for k, v in self.graph_state.constants.items()
        }

        return Graph(
            inputs=self.graph_state.inputs,
            nodes=self.graph_state.nodes,
            tensor_values=self.graph_state.tensor_values,
            sym_int_values=self.graph_state.sym_int_values,
            sym_bool_values=self.graph_state.sym_bool_values,
            outputs=self.graph_state.outputs,
            is_single_tensor_return=self.graph_state.is_single_tensor_return,
            constants=serialized_constants,
        )

    def serialize(self, graph_module: torch.fx.GraphModule) -> GraphModule:
        graph = self.serialize_graph(graph_module)

        return GraphModule(
            graph=graph,
            signature=serialize_signature(self.graph_signature),
            call_spec=serialize_call_spec(self.call_spec),
            module_call_graph=self.serialize_module_call_graph(self.module_call_graph),
        )


class ExportedProgramSerializer:
    def __init__(self, opset_version: Optional[Dict[str, int]] = None):
        self.opset_version: Dict[str, int] = {}
        if opset_version:
            self.opset_version.update(opset_version)
        if "aten" not in self.opset_version:
            self.opset_version["aten"] = torch._C._get_max_operator_version()

    def serialize(self, exported_program: ep.ExportedProgram) -> Tuple[ExportedProgram, bytes]:
        serialized_graph_module = (
            GraphModuleSerializer(
                exported_program.graph_signature,
                exported_program.call_spec,
                exported_program.module_call_graph
            ).serialize(exported_program.graph_module)
        )
        serialized_range_constraints = serialize_range_constraints(exported_program.range_constraints)
        serialized_equality_constraints = serialize_equality_constraints(exported_program.equality_constraints)

        return (
            ExportedProgram(
                graph_module=serialized_graph_module,
                opset_version=self.opset_version,
                range_constraints=serialized_range_constraints,
                equality_constraints=serialized_equality_constraints,
                schema_version=SCHEMA_VERSION,
                example_inputs=None,
            ),
            serialize_torch_artifact(exported_program.state_dict),
        )


class GraphModuleDeserializer:
    def __init__(self):
        self.serialized_name_to_node: Dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: Dict[str, MetaType] = {}
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()

    @contextmanager
    def save_graph_module(self) -> Iterator[None]:
        saved = self.graph, self.module, self.serialized_name_to_node, self.serialized_name_to_meta
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()
        self.serialized_name_to_node = {}
        self.serialized_name_to_meta = {}
        try:
            yield
        finally:
            self.graph, self.module, self.serialized_name_to_node, self.serialized_name_to_meta = saved

    def deserialize_operator(self, serialized_target: str):
        if serialized_target.startswith("_operator"):  # TODO(zhxchen17) Follow up on this.
            module = operator
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("torch.ops"):
            module = torch.ops
            serialized_target_names = serialized_target.split(".")[2:]
        else:  # TODO(zhxchen17) Don't catch all here.
            return serialized_target

        target = module
        for name in serialized_target_names:
            if not hasattr(target, name):
                return serialized_target
            else:
                target = getattr(target, name)
        return target

    def deserialize_sym_int(self, s: SymInt) -> Union[int, torch.SymInt]:
        val = s.value
        if s.type == "as_expr":
            if val.expr_str in self.symbol_name_to_symbol:
                sym = self.symbol_name_to_symbol[val.expr_str]
            else:
                sym = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
                if isinstance(sym, sympy.Symbol):
                    self.symbol_name_to_symbol[val.expr_str] = sym

                    if vr := self.symbol_name_to_range.get(val.expr_str):
                        symbolic_shapes._constrain_symbol_range(
                            self.shape_env,
                            sym,
                            compiler_min=vr.lower,  # type: ignore[arg-type]
                            compiler_max=vr.upper,  # type: ignore[arg-type]
                            runtime_min=vr.lower,  # type: ignore[arg-type]
                            runtime_max=vr.upper  # type: ignore[arg-type]
                        )

            return self.shape_env.create_symintnode(sym, hint=val.hint)
        elif s.type == "as_int":
            assert isinstance(val, int)
            return val
        else:
            raise SerializeError(
                f"SymInt has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_sym_bool(self, s: SymBool) -> Union[bool, torch.SymBool]:
        val = s.value
        if s.type == "as_expr":
            expr = sympy.sympify(val, locals=self.symbol_name_to_symbol)
            return self.shape_env.create_symboolnode(expr)
        elif s.type == "as_bool":
            assert isinstance(val, bool)
            return val
        else:
            raise SerializeError(
                f"SymBool has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_tensor_meta(
        self,
        tensor_meta: TensorMeta,
        fake_tensor_mode: FakeTensorMode,
    ) -> FakeTensor:
        with fake_tensor_mode:
            return cast(
                FakeTensor,
                torch.empty_strided(
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.sizes),  # type: ignore[misc]
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.strides),  # type: ignore[misc]
                    device=deserialize_device(tensor_meta.device),
                    dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
                ),
            )

    def deserialize_graph_output(self, output) -> torch.fx.Node:
        if isinstance(output.value, TensorArgument):
            if output.value.name in self.constants:
                val = self.constants[output.value.name]
                setattr(self.module, output.value.name, val)
                node = self.graph.create_node(
                    "get_attr",
                    output.value.name,
                    name=output.value.name,
                )
                return node
            return self.serialized_name_to_node[output.value.name]
        elif isinstance(output.value, (SymIntArgument, SymBoolArgument)):
            return self.serialized_name_to_node[output.value.as_name]
        else:
            raise SerializeError(f"Unable to deserialize output node {output}")

    def deserialize_graph(self, serialized_graph: Graph) -> torch.fx.Graph:
        self.constants: Dict[str, torch.Tensor] = {
            k: deserialize_torch_artifact(base64.b64decode(v))
            for k, v in serialized_graph.constants.items()
        }

        # Handle the tensor metas.
        for name, tensor_value in serialized_graph.tensor_values.items():
            meta_val = self.deserialize_tensor_meta(tensor_value.meta, self.fake_tensor_mode)
            self.serialized_name_to_meta[name] = meta_val

        for name, sym_int_value in serialized_graph.sym_int_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_sym_int(sym_int_value)

        for name, sym_bool_value in serialized_graph.sym_bool_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_sym_bool(sym_bool_value)

        # Inputs: convert to placeholder nodes in FX.
        for input in serialized_graph.inputs:
            placeholder_node = self.graph.placeholder(input.as_tensor.name)
            self.sync_fx_node(input.as_tensor.name, placeholder_node)

        # Nodes: convert to call_function nodes.
        for serialized_node in serialized_graph.nodes:
            try:
                target = self.deserialize_operator(serialized_node.target)
                self.deserialize_node(serialized_node, target)

            except Exception as e:
                raise SerializeError(f"Failed deserializing node {serialized_node}") from e

        # Outputs: convert to a single `output` node.
        outputs = []
        for output in serialized_graph.outputs:
            outputs.append(self.deserialize_graph_output(output))

        if serialized_graph.is_single_tensor_return:
            assert len(outputs) == 1
            outputs = outputs[0]  # type: ignore[assignment]
        else:
            outputs = tuple(outputs)  # type: ignore[assignment]

        output_node = self.graph.output(outputs)

        if serialized_graph.is_single_tensor_return:
            output_node.meta["val"] = output_node.args[0].meta["val"]
        else:
            output_node.meta["val"] = tuple(
                arg.meta["val"] for arg in output_node.args[0]
            )

        return self.graph

    def deserialize_node(self, serialized_node: Node, target: Callable) -> None:
        if target.__module__ == "_operator":  # TODO(zhxchen17) Follow up on this.
            name = serialized_node.outputs[0].value.as_name
            args = self.deserialize_sym_op_inputs(serialized_node.inputs)

            fx_node = self.graph.create_node("call_function", target, args, {}, name)
            self.deserialize_sym_op_outputs(serialized_node, fx_node)
        elif isinstance(target, torch._ops.HigherOrderOperator):
            assert (
                len(serialized_node.outputs) == 1
                and serialized_node.outputs[0].type in ("as_tensors", "as_tensor")
            ), "Only single tensor output or list of tensor output is supported for higher order operators."

            output = serialized_node.outputs[0]

            name = (
                output.value.name
                if output.type == "as_tensor"
                else None  # FX will generate a name for us.
            )
            args = tuple(self.deserialize_input(input.arg) for input in serialized_node.inputs)
            fx_node = self.graph.create_node("call_function", target, args, {}, name)

            if output.type == "as_tensor":
                self.sync_fx_node(name, fx_node)
            if output.type == "as_tensors":
                self.deserialize_multiple_outputs(serialized_node, fx_node)

        elif isinstance(target, torch._ops.OpOverload):
            # For convenience: if this node returns a single tensor, name the
            # newly-created node after it. This ensures that these tensor values
            # have names that are consistent with serialized.
            name = (
                serialized_node.outputs[0].value.name
                if _is_single_tensor_return(target)
                else None  # FX will generate a name for us.
            )
            args, kwargs = self.deserialize_inputs(target, serialized_node)
            fx_node = self.graph.create_node("call_function", target, args, kwargs, name)
            self.deserialize_outputs(serialized_node, fx_node)
        else:
            raise SerializeError(f"Unsupported target type for node {serialized_node}: {target}")

        fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))

    def deserialize(
        self,
        serialized_graph_module: GraphModule,
        symbol_name_to_range: Optional[Dict[str, symbolic_shapes.ValueRanges]] = None,
    ) -> Tuple[torch.fx.GraphModule, ep.ExportGraphSignature, ep.CallSpec, List[ep.ModuleCallEntry], Dict[str, sympy.Symbol]]:
        self.shape_env = symbolic_shapes.ShapeEnv(assume_static_by_default=True)
        self.fake_tensor_mode = FakeTensorMode(
            allow_fallback_kernels=False,
            allow_non_fake_inputs=True,
            shape_env=self.shape_env,
        )
        self.symbol_name_to_symbol: Dict[str, sympy.Symbol] = {}
        self.symbol_name_to_range = {} if symbol_name_to_range is None else symbol_name_to_range

        self.deserialize_graph(serialized_graph_module.graph)

        sig = deserialize_signature(serialized_graph_module.signature)
        call_spec = deserialize_call_spec(serialized_graph_module.call_spec)
        module_call_graph = self.deserialize_module_call_graph(serialized_graph_module.module_call_graph)
        return (
            ep._create_graph_module_for_export(self.module, self.graph),
            sig,
            call_spec,
            module_call_graph,
            self.symbol_name_to_symbol,
        )

    def sync_fx_node(self, name: str, fx_node: torch.fx.Node):
        if name in self.serialized_name_to_node:
            raise SerializeError(f"Node {name} has already been deserialized before.")
        self.serialized_name_to_node[name] = fx_node
        assert "val" not in fx_node.meta
        fx_node.meta["val"] = self.serialized_name_to_meta[name]

    def deserialize_sym_op_inputs(self, inputs):
        return tuple(self.deserialize_input(input.arg) for input in inputs)

    def deserialize_inputs(self, target: torch._ops.OpOverload, serialized_node: Node):
        schema_args = target._schema.arguments
        actual_args = {
            input.name: self.deserialize_input(input.arg) for input in serialized_node.inputs
        }
        args = []
        kwargs = {}
        for schema_arg in schema_args:
            is_positional = not schema_arg.has_default_value() and not schema_arg.kwarg_only
            if is_positional:
                args.append(actual_args[schema_arg.name])
            else:
                if schema_arg.name in actual_args:
                    kwargs[schema_arg.name] = actual_args[schema_arg.name]
        return tuple(args), kwargs

    def deserialize_input(self, inp: Argument) -> Any:
        value = inp.value
        typ_ = inp.type
        if typ_ == "as_none":
            # None should converted as None, but is encoded as bool in serialized
            # Convert serialized object to torch equivalent
            return None
        elif typ_ == "as_scalar_type":
            return _SERIALIZE_TO_TORCH_DTYPE[value]
        elif typ_ == "as_memory_format":
            return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[value]
        elif typ_ == "as_layout":
            return _SERIALIZE_TO_TORCH_LAYOUT[value]
        elif typ_ == "as_graph":
            assert isinstance(value, GraphArgument)
            with self.save_graph_module():
                self.deserialize_graph(value.graph)
                submodule = ep._create_graph_module_for_export(self.module, self.graph)
            self.module.register_module(value.name, submodule)
            return self.graph.create_node(
                "get_attr",
                value.name,
                name=value.name,
            )
        elif isinstance(value, Device):
            return deserialize_device(value)
        elif isinstance(value, TensorArgument):
            if value.name in self.constants:
                val = self.constants[value.name]
                setattr(self.module, value.name, val)
                return self.graph.create_node(
                    "get_attr",
                    value.name,
                    name=value.name,
                )
            return self.serialized_name_to_node[value.name]
        elif isinstance(value, (int, float, bool)):
            return value
        elif isinstance(value, str):
            return str(value)
        elif isinstance(value, (SymIntArgument, SymBoolArgument)):
            return self.deserialize_sym_argument(value)
        elif isinstance(value, list):
            if len(value) == 0:
                return []
            elif isinstance(value[0], TensorArgument):
                result = []
                for arg in value:
                    if arg.name in self.constants:
                        val = self.constants[arg.name]
                        setattr(self.module, arg.name, val)
                        result.append(
                            self.graph.create_node(
                                "get_attr",
                                arg.name,
                                name=arg.name,
                            )
                        )
                    else:
                        result.append(self.serialized_name_to_node[arg.name])
                return result
            elif isinstance(value[0], (int, float, bool)):
                # convert from serialized.python.types.List to python list
                return list(value)
            elif isinstance(value[0], (SymIntArgument, SymBoolArgument)):
                return [self.deserialize_sym_argument(arg) for arg in value]
            elif isinstance(value[0], OptionalTensorArgument):
                def deserialize_optional_tensor_args(a):
                    if a.type == "as_none":
                        return None
                    elif a.type == "as_tensor":
                        return self.serialized_name_to_node[a.value]
                    else:
                        raise SerializeError(f"Unhandled argument {inp}")
                return list(map(deserialize_optional_tensor_args, value))
            else:
                raise SerializeError(f"Unhandled argument {inp}")
        elif isinstance(value, CustomObjArgument):
            # Custom objects through torchind are deserializable with pickle,
            # through implementing the .def_pickle function.
            blob = base64.b64decode(value.blob)
            return pickle.loads(blob)
        else:
            raise SerializeError(f"Unhandled argument {inp}")

    def deserialize_sym_argument(self, sym_int_arg):
        if sym_int_arg.type == "as_int":
            return sym_int_arg.as_int
        else:
            assert sym_int_arg.type == "as_name"
            return self.serialized_name_to_node[sym_int_arg.as_name]

    def deserialize_sym_op_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)

    def deserialize_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        # Simple case for single tensor return.
        assert isinstance(fx_node.target, torch._ops.OpOverload)
        returns = fx_node.target._schema.returns

        # Check single value return
        if len(returns) == 0:
            return
        if _is_single_tensor_return(fx_node.target):
            self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
            return
        elif len(returns) == 1 and isinstance(serialized_node.outputs[0].value, (SymIntArgument, SymBoolArgument)):
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return

        self.deserialize_multiple_outputs(serialized_node, fx_node)

    def deserialize_multiple_outputs(self, serialized_node: Node, fx_node: torch.fx.Node) -> None:
        # Convert multiple return types to FX format.
        # In FX, each node only returns one value. So in order to represent
        # multiple return values, we have to emit a `getitem` node for each
        # return value.
        # This performs the inverse mapping of the `serialize_outputs` call in
        # serialization, see [NOTE: Multiple outputs]
        output_names = []
        if len(serialized_node.outputs) == 1:
            assert isinstance(serialized_node.outputs[0].value, list)
            assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
            output_names = [arg.name for arg in serialized_node.outputs[0].as_tensors]
        else:
            for output in serialized_node.outputs:
                assert isinstance(output.value, TensorArgument)
                output_names.append(output.as_tensor.name)

        for idx, name in enumerate(output_names):
            individual_output = self.graph.create_node(
                "call_function",
                operator.getitem,
                (fx_node, idx),
                name=name,
            )
            self.sync_fx_node(name, individual_output)
            # The derived `getitem` nodes should have the same stacktrace as the
            # original `fx_node`
            individual_output.meta.update(self.deserialize_metadata(serialized_node.metadata))

        # also update the metaval for `fx_node` to be a list(meta)
        fx_node.meta["val"] = tuple(self.serialized_name_to_meta[name] for name in output_names)
        self.serialized_name_to_node[fx_node.name] = fx_node

    def deserialize_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        ret: Dict[str, Any] = {}
        if stack_trace := metadata.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        def deserialize_meta_func(serialized_target: str):
            module = None
            if serialized_target.startswith("torch.nn"):
                module = torch.nn
                serialized_target_names = serialized_target.split(".")[2:]
            elif serialized_target.startswith("torch"):
                module = torch
                serialized_target_names = serialized_target.split(".")[1:]
            else:
                return self.deserialize_operator(serialized_target)

            target = module
            for name in serialized_target_names:
                if not hasattr(target, name):
                    return serialized_target
                else:
                    target = getattr(target, name)
            return target

        if nn_module_stack_str := metadata.get("nn_module_stack"):
            # Originally serialized to "fx_node_name:(orig_ref,type_str)"
            nn_module_stack_list = nn_module_stack_str.split(";")
            nn_module_stack = {}
            for kv in nn_module_stack_list:
                key_idx = kv.find(":")
                key = kv[:key_idx]
                assert kv[key_idx + 1] == "("
                assert kv[-1] == ")"

                paren = 0
                comma_idx = None
                for i, c in enumerate(kv[key_idx + 2:-1]):
                    if c == "," and paren == 0:
                        assert comma_idx is None
                        comma_idx = i + key_idx + 2
                    elif c == "(":
                        paren += 1
                    elif c == ")":
                        paren -= 1

                assert comma_idx is not None
                module = deserialize_meta_func(kv[comma_idx + 1:-1])
                nn_module_stack[key] = (kv[key_idx + 2:comma_idx], module)
            ret["nn_module_stack"] = nn_module_stack

        if source_fn_str := metadata.get("source_fn"):
            # Originally serializes to "fx_node_name,op_str"
            source_fn = source_fn_str.split(",")
            op = deserialize_meta_func(source_fn[1])
            ret["source_fn"] = (source_fn[0], op)

        return ret

    def deserialize_module_call_signature(self, module_call_signature: ModuleCallSignature) -> ep.ModuleCallSignature:
        def deserialize_argument(x: Argument) -> ep.ArgumentSpec:
            if x.as_tensor is not None:
                return ep.ArgumentSpec(kind=ep.ArgumentKind.Tensor, value=x.as_tensor.name)
            elif x.as_symint is not None:
                return ep.ArgumentSpec(kind=ep.ArgumentKind.SymInt, value=x.as_symint.as_name)
            else:
                return ep.ArgumentSpec(
                    kind=ep.ArgumentKind.Constant, value=self.deserialize_input(x)
                )

        return ep.ModuleCallSignature(
            inputs=[deserialize_argument(x) for x in module_call_signature.inputs],
            outputs=[deserialize_argument(x) for x in module_call_signature.outputs],
            in_spec=treespec_loads(module_call_signature.in_spec),
            out_spec=treespec_loads(module_call_signature.out_spec),
        )

    def deserialize_module_call_graph(self, module_call_graph: List[ModuleCallEntry]) -> List[ep.ModuleCallEntry]:
        return [
            ep.ModuleCallEntry(
                fqn=entry.fqn,
                signature=self.deserialize_module_call_signature(entry.signature) if entry.signature else None,
            ) for entry in module_call_graph
        ]


class ExportedProgramDeserializer:
    def __init__(self, expected_opset_version: Optional[Dict[str, int]] = None):
        self.expected_opset_version: Dict[str, int] = {}
        if expected_opset_version:
            self.expected_opset_version.update(expected_opset_version)
        if "aten" not in self.expected_opset_version:
            self.expected_opset_version["aten"] = torch._C._get_max_operator_version()

    def deserialize_range_constraints(
        self,
        symbol_name_to_range: Dict[str, symbolic_shapes.ValueRanges],
        symbol_name_to_symbol: Dict[str, sympy.Symbol],
    ) -> Dict[sympy.Symbol, ep.RangeConstraint]:
        range_constraints = {}
        for k, v in symbol_name_to_range.items():
            if symbol := symbol_name_to_symbol.get(k):
                range_constraints[symbol] = ep.RangeConstraint(v.lower, v.upper)  # type: ignore[arg-type]
            else:
                log.warning(f"Symbol {k} did not appear in the graph that was deserialized")  # noqa: G004
        return range_constraints

    def deserialize(
        self, serialized_exported_program: ExportedProgram, serialized_state_dict: bytes
    ) -> ep.ExportedProgram:
        if serialized_exported_program.schema_version != SCHEMA_VERSION:
            raise SerializeError(
                f"Serialized schema version {serialized_exported_program.schema_version} "
                f"does not match our current schema version {SCHEMA_VERSION}."
            )

        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(_int_to_sympy_int(v.min_val), _int_to_sympy_int(v.max_val))
            for k, v in serialized_exported_program.range_constraints.items()
        }

        graph_module, sig, call_spec, module_call_graph, symbol_name_to_symbol = (
            GraphModuleDeserializer()
            .deserialize(
                serialized_exported_program.graph_module,
                symbol_name_to_range,
            )
        )
        range_constraints = self.deserialize_range_constraints(
            symbol_name_to_range, symbol_name_to_symbol,
        )
        model_opset_version: Optional[Dict[str, int]] = serialized_exported_program.opset_version
        self._validate_model_opset_version(model_opset_version)

        upgrader = GraphModuleOpUpgrader(self.expected_opset_version, model_opset_version)

        state_dict = deserialize_torch_artifact(serialized_state_dict)
        equality_constraints = deserialize_equality_constraints(serialized_exported_program.equality_constraints)

        exported_program = ep.ExportedProgram(
            graph_module,
            graph_module.graph,
            sig,
            call_spec,
            state_dict,  # type: ignore[arg-type]
            range_constraints,
            equality_constraints,
            module_call_graph,
            None,  # type: ignore[arg-type]
        )
        return upgrader.upgrade(exported_program)

    def _validate_model_opset_version(self, model_opset_version: Optional[Dict[str, int]]):
        """Compare model_opset_version with expected_opset_version and raise error if we can't resolve the version
        difference.
        E.g., model_opset_version = {"aten": 3, "custom": 4}
        expected_opset_version = {"aten": 4, "custom": 4}
        This means we can use an upgrader for ATen to reconcile the deserialized model.

        The logic of this method:

        For common op namespaces:
        1. if model version < expected version, this case can be handled by upgraders.
        2. if model version > expected version, we need downgraders but not implemented yet.
        3. if model version == expected version, we don't need extra handling.

        For op namespace only in model_opset_version, we should give a warning because it is missing from
        expected_opset_version.
        """
        if not model_opset_version:
            raise RuntimeError("Serialized model should have opset version.")
        common_namespaces = {key for key in model_opset_version if key in self.expected_opset_version}
        for namespace in common_namespaces:
            assert (
                isinstance(model_version := model_opset_version[namespace], int)
            ), f"model_opset_version value should be int, got {model_opset_version[namespace]}"

            assert (
                isinstance(compiler_version := self.expected_opset_version[namespace], int)
            ), f"expected_opset_version value should be int, got {self.expected_opset_version[namespace]}"

            # TODO(larryliu0820): Add support for upgrader & downgrader
            if model_version != compiler_version:
                raise NotImplementedError(
                    f"Model opset version {model_opset_version} doesn't match to compiler opset version "
                    f"{self.expected_opset_version}! Upgrader/downgrader is not implemented yet."
                )
        for namespace in model_opset_version:
            if namespace in common_namespaces:
                continue
            log.warning("Compiler doesn't have a version table for op namespace: {ns}. ", extra={"ns": namespace})


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        return super().default(obj)


def serialize(
    exported_program: ep.ExportedProgram,
    opset_version: Optional[Dict[str, int]] = None,
) -> Tuple[bytes, bytes]:
    serialized_exported_program, serialized_state_dict = (
        ExportedProgramSerializer(opset_version).serialize(exported_program)
    )
    json_program = json.dumps(
        dataclasses.asdict(serialized_exported_program), cls=EnumEncoder
    )
    json_bytes = json_program.encode('utf-8')
    return json_bytes, serialized_state_dict


def _dict_to_dataclass(cls, data):
    assert not isinstance(cls, str), f"Unresolved class type: '{cls}'."
    if typing.get_origin(cls) == typing.Union and type(None) in typing.get_args(cls):
        if data is None:
            return None
        ty_args = typing.get_args(cls)
        assert len(ty_args) == 2
        return _dict_to_dataclass(ty_args[0], data)
    elif isinstance(cls, type) and issubclass(cls, _Union):
        obj = cls(**data)
        field_type = cls.__annotations__[obj.type]
        setattr(obj, obj.type, _dict_to_dataclass(field_type, obj.value))
        return obj
    elif dataclasses.is_dataclass(cls):
        obj = cls(**data)  # type: ignore[assignment]
        type_hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            name = f.name
            new_field_obj = _dict_to_dataclass(type_hints[name], getattr(obj, name))
            setattr(obj, name, new_field_obj)
        return obj
    elif isinstance(data, list):
        if len(data) == 0:
            return data
        d_type = typing.get_args(cls)[0]
        return [
            _dict_to_dataclass(d_type, d)
            for d in data
        ]
    elif isinstance(data, dict):
        v_type = typing.get_args(cls)[1]
        return {
            k: _dict_to_dataclass(v_type, v)
            for k, v in data.items()
        }
    return data


def deserialize(
    exported_program_bytes: bytes,
    state_dict: bytes,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ep.ExportedProgram:
    exported_program_str = exported_program_bytes.decode('utf-8')
    exported_program_dict = json.loads(exported_program_str)
    serialized_exported_program = _dict_to_dataclass(ExportedProgram, exported_program_dict)
    return (
        ExportedProgramDeserializer(expected_opset_version)
        .deserialize(serialized_exported_program, state_dict)
    )
