# mypy: allow-untyped-defs
import base64
import copy
import copyreg
import dataclasses
import heapq
import inspect
import io
import json
import logging
import math
import operator
import re
import typing

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    final,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Type,
)

import sympy

import torch
import torch.export.exported_program as ep
from torch._export.serde.schema import SchemaVersion
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils import _pytree as pytree
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges

from .schema import (  # type: ignore[attr-defined]
    Argument,
    BufferMutationSpec,
    ConstantInputSpec,
    ConstantValue,
    CustomObjArgument,
    Device,
    ExportedProgram,
    GradientToParameterSpec,
    GradientToUserInputSpec,
    Graph,
    GraphArgument,
    GraphModule,
    GraphSignature,
    InputSpec,
    InputToBufferSpec,
    InputToCustomObjSpec,
    InputTokenSpec,
    InputToParameterSpec,
    InputToTensorConstantSpec,
    Layout,
    LossOutputSpec,
    MemoryFormat,
    ModuleCallEntry,
    ModuleCallSignature,
    NamedArgument,
    Node,
    OptionalTensorArgument,
    OutputSpec,
    OutputTokenSpec,
    RangeConstraint,
    ScalarType,
    SCHEMA_VERSION,
    SymBool,
    SymBoolArgument,
    SymExpr,
    SymExprHint,
    SymInt,
    SymIntArgument,
    TensorArgument,
    TensorMeta,
    TokenArgument,
    TREESPEC_VERSION,
    UserInputMutationSpec,
    UserInputSpec,
    UserOutputSpec,
)
from .union import _Union


__all__ = [
    "serialize",
    "GraphModuleSerializer",
    "ExportedProgramSerializer",
    "GraphModuleDeserializer",
    "ExportedProgramDeserializer",
]

log = logging.getLogger(__name__)


class SerializeError(RuntimeError):
    pass


def _reverse_map(d: Dict[Any, Enum]):
    return {v.value: k for k, v in d.items()}


MetaType = Union[
    FakeTensor, int, torch.SymInt, bool, torch.SymBool, ep.CustomObjArgument
]


ST_DELIMITER = ";"

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
    torch.bfloat16: ScalarType.BFLOAT16,
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
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_sqrt,
}


_SYM_BOOL_OPS = {
    operator.eq,
    operator.ne,
    operator.le,
    operator.ge,
    operator.lt,
    operator.gt,
    torch.sym_not,
}


@dataclass
class SerializedArtifact:
    exported_program: bytes
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


@dataclass
class _SerializedProgram:
    exported_program: ExportedProgram
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


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
            if s.node.hint is None:
                return SymInt.create(as_expr=SymExpr(str(s)))
            else:
                return SymInt.create(
                    as_expr=SymExpr(str(s), hint=SymExprHint.create(as_int=s.node.hint))
                )
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_bool(s: Union[bool, torch.SymBool]) -> SymBool:
    if isinstance(s, (torch.SymBool, bool)):
        if symbolic_shapes.is_concrete_bool(s):
            return SymBool.create(as_bool=bool(s))
        else:
            return SymBool.create(as_expr=SymExpr(expr_str=str(s)))
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
        storage_offset=serialize_sym_int(0),  # TODO needs to be fixed.
        layout=_TORCH_TO_SERIALIZE_LAYOUT[t.layout],
    )


_CURRENT_DESERIALIZER: Optional["GraphModuleDeserializer"] = None


def _reduce_fake_tensor(fake_tensor: FakeTensor):
    is_parameter = isinstance(fake_tensor, torch.nn.Parameter)
    tensor_meta = serialize_tensor_meta(fake_tensor)
    tensor_meta_bytes = json.dumps(
        _dataclass_to_dict(tensor_meta), cls=EnumEncoder
    ).encode("utf-8")
    return _reconstruct_fake_tensor, (tensor_meta_bytes, is_parameter)


def _reconstruct_fake_tensor(
    serialized_tensor_meta: bytes, is_parameter: bool
) -> FakeTensor:
    # Deserialize the bytes into a TensorMeta
    json_tensor_meta = json.loads(serialized_tensor_meta.decode("utf-8"))
    tensor_meta = _dict_to_dataclass(TensorMeta, json_tensor_meta)
    # Find the current fake mode
    assert (
        _CURRENT_DESERIALIZER is not None
    ), "Need access to current deserializer state"
    fake_tensor = _CURRENT_DESERIALIZER.deserialize_tensor_meta(tensor_meta)
    if is_parameter:
        fake_tensor = torch.nn.Parameter(fake_tensor)  # type: ignore[assignment]
    return fake_tensor


def serialize_torch_artifact(artifact: Optional[Any]) -> bytes:
    if artifact is None:
        return b""

    assert (
        FakeTensor not in copyreg.dispatch_table
    ), "Refusing to stomp on existing FakeTensor reducer"
    try:
        copyreg.pickle(FakeTensor, _reduce_fake_tensor)
        buffer = io.BytesIO()
        # This is a workaround for backend's tensor deserialization problem:
        # unpickleTensor() always create a tensor on the device where it was originally saved
        # This behavior is bad for multi-gpu training, as we wish to directly load the tensor
        # on the designated device.
        # For now, we simply move the tensor to cpu before saving.
        # TODO: this should be fixed by deserialization instead.
        torch.save(artifact, buffer)
        return buffer.getvalue()
    finally:
        del copyreg.dispatch_table[FakeTensor]


def deserialize_torch_artifact(serialized: Union[Dict[str, Any], Tuple[Any, ...], bytes]):
    if isinstance(serialized, (dict, tuple)):
        return serialized
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    artifact = torch.load(buffer)
    assert isinstance(artifact, (tuple, dict))
    return artifact


def _sympy_int_to_int(val: sympy.Expr, adjust: str):
    # Convert simple sympy Integers into concrete int
    if val == sympy.oo:
        return math.inf
    if val == -sympy.oo:
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)

    # TODO: Remove this adjustment when Ed gets rid of fractional ranges
    log.warning(
        "Export constraints cannot be non-integer expressions. Found "
        "type %s, and value %s. We will attempt to %s "
        "this value.", type(val), val, adjust
    )

    if adjust == "floor":
        return math.floor(val)
    elif adjust == "ceil":
        return math.ceil(val)
    else:
        raise RuntimeError(f"Got invalid adjustment {adjust}")


def _int_to_sympy_int(val) -> sympy.Expr:
    # Convert concrete int into simple sympy Integers
    if val == math.inf:
        return sympy.oo
    if val == -math.inf:
        return -sympy.oo
    return sympy.Integer(val)


def serialize_range_constraints(
    range_constraints: Dict[sympy.Symbol, ValueRanges]
) -> Dict[str, RangeConstraint]:
    return {
        str(k): RangeConstraint(
            _sympy_int_to_int(v.lower, "ceil"),  # type: ignore[arg-type]
            _sympy_int_to_int(v.upper, "floor"),  # type: ignore[arg-type]
        )
        for k, v in range_constraints.items()
    }


def _get_schema_from_target(target):
    if isinstance(target, torch._ops.OpOverload):
        return target._schema
    elif type(target) in _serialization_registry:
        return _serialization_registry[type(target)].op_schema(type(target))
    raise RuntimeError(f"Cannot find schema for {type(target)}")


def _is_single_tensor_return(target: torch._ops.OpOverload) -> bool:
    schema = _get_schema_from_target(target)
    returns = schema.returns
    return len(returns) == 1 and isinstance(returns[0].real_type, torch.TensorType)


def _is_single_tensor_list_return(target: Any) -> bool:
    schema = _get_schema_from_target(target)
    returns = schema.returns

    if len(returns) != 1:
        return False
    return_type = returns[0].real_type
    return isinstance(return_type, torch.ListType) and isinstance(
        return_type.getElementType(), torch.TensorType
    )


@dataclass
class GraphState:
    inputs: List[Argument] = field(default_factory=list)
    outputs: List[Argument] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    tensor_values: Dict[str, TensorMeta] = field(default_factory=dict)
    sym_int_values: Dict[str, SymInt] = field(default_factory=dict)
    sym_bool_values: Dict[str, SymBool] = field(default_factory=dict)
    is_single_tensor_return: bool = False
    custom_obj_values: Dict[str, CustomObjArgument] = field(default_factory=dict)


class Final(type):
    def __new__(metacls, name, bases, classdict):
        for b in bases:
            if isinstance(b, Final):
                raise TypeError(f"type '{b.__name__}' is not an acceptable base type")
        return type.__new__(metacls, name, bases, dict(classdict))


@final
class GraphModuleSerializer(metaclass=Final):
    def __init__(
        self,
        graph_signature: ep.ExportGraphSignature,
        module_call_graph: List[ep.ModuleCallEntry],
    ):
        self.graph_state = GraphState()
        self.graph_signature = graph_signature
        self.module_call_graph = module_call_graph
        self.custom_objs: Dict[str, torch._C.ScriptObject] = {}
        self.duplicate_getitem_nodes: Dict[str, str] = {}

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
        if isinstance(node.meta["val"], torch.Tensor):
            graph_input = Argument.create(as_tensor=TensorArgument(name=node.name))
            self.graph_state.tensor_values[node.name] = serialize_tensor_meta(
                node.meta["val"]
            )
        elif isinstance(node.meta["val"], torch.SymInt):
            raise AssertionError("SymInt graph input is not implemented yet.")
        elif isinstance(node.meta["val"], (int, bool, str, float, type(None))):
            graph_input = self.serialize_input(node.meta["val"])
        elif isinstance(node.meta["val"], ep.CustomObjArgument):
            class_fqn = node.meta["val"].class_fqn
            graph_input = Argument.create(
                as_custom_obj=CustomObjArgument(name=node.name, class_fqn=class_fqn)
            )
            self.graph_state.custom_obj_values[node.name] = (
                self.serialize_script_obj_meta(node.meta["val"])
            )
        else:
            raise AssertionError(f"Unimplemented graph input type: {node.meta['val']}")
        self.graph_state.inputs.append(graph_input)

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
                inputs=self.serialize_sym_op_inputs(node.target, node.args),
                outputs=[
                    Argument.create(
                        as_sym_int=self.serialize_sym_int_output(node.name, meta_val)
                    )
                ],
                metadata=self.serialize_metadata(node),
            )
        elif node.target in _SYM_BOOL_OPS:
            assert len(node.kwargs) == 0
            meta_val = node.meta["val"]
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_sym_op_inputs(node.target, node.args),
                outputs=[
                    Argument.create(
                        as_sym_bool=self.serialize_sym_bool_output(node.name, meta_val)
                    )
                ],
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
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_hoo_inputs(node.args, node.kwargs),
                outputs=self.serialize_hoo_outputs(node),
                metadata=self.serialize_metadata(node),
            )
        elif type(node.target) in _serialization_registry:
            custom_op_handler = node.target

            # Sanity check for unhandled serialization.
            assert type(node.target) in _serialization_registry, f"Miss {type(node.target)} CustomOpHandler"

            handler = _serialization_registry[type(node.target)]
            ex_node = Node(
                target=f"${handler.namespace()}:{handler.op_name(node.target)}",
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                metadata=self.serialize_metadata(node),
            )
        else:
            raise SerializeError(f"Serializing {node.target} is not supported")

        self.graph_state.nodes.append(ex_node)

    def handle_get_attr(self, node):
        pass

    def _output_node_at_index(self, node, index):
        user_node = None
        for user in node.users:
            assert user.target is operator.getitem, f"{user} is not a getitem node"
            if index == user.args[1]:
                if user_node is None:
                    user_node = user
                else:
                    # We want to deduplicate getitem nodes that are trying to
                    # index to the same index
                    self.duplicate_getitem_nodes[user.name] = user_node.name
        return user_node

    def serialize_metadata(self, node: torch.fx.Node) -> Dict[str, str]:
        ret = {}
        if stack_trace := node.meta.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        if nn_module_stack := node.meta.get("nn_module_stack"):

            def export_nn_module_stack(val):
                assert isinstance(val, tuple) and len(val) == 2
                path, ty = val

                assert isinstance(path, str)
                assert isinstance(ty, str)

                return path + "," + ty

            # Serialize to "key,orig_path,type_str"
            nn_module_list = [
                f"{k},{export_nn_module_stack(v)}" for k, v in nn_module_stack.items()
            ]
            ret["nn_module_stack"] = ST_DELIMITER.join(nn_module_list)

        if source_fn_st := node.meta.get("source_fn_stack"):
            source_fn_list = [
                f"{source_fn[0]},{self.serialize_operator(source_fn[1])}"
                for source_fn in source_fn_st
            ]
            ret["source_fn_stack"] = ST_DELIMITER.join(source_fn_list)

        if torch_fn := node.meta.get("torch_fn"):
            ret["torch_fn"] = ST_DELIMITER.join(list(torch_fn))

        return ret

    def serialize_script_obj_meta(
        self, script_obj_meta: ep.CustomObjArgument
    ) -> CustomObjArgument:
        return CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    def serialize_sym_op_inputs(self, op, args) -> List[NamedArgument]:
        serialized_args = []
        args_names = inspect.signature(op).parameters.keys()
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(name=args_name, arg=self.serialize_input(arg))
            )
        return serialized_args

    def serialize_inputs(
        self,
        target: Any,  # torch._ops.OpOverload and other custom operator types.
        args,
        kwargs=None
    ) -> List[NamedArgument]:
        assert isinstance(target, (torch._ops.OpOverload, *allowed_registered_op_types()))
        kwargs = kwargs or {}
        serialized_args = []

        schema = _get_schema_from_target(target)

        for i, schema_arg in enumerate(schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(kwargs[schema_arg.name], schema_arg.type),
                    )
                )
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(args[i], schema_arg.type),
                    )
                )
            else:
                # We intentionally don't serialize the missing arguments
                # with default values
                pass

        return serialized_args

    def serialize_hoo_inputs(self, args, kwargs) -> List[NamedArgument]:
        """
        For serializing HOO inputs since HOOs do not have a schema.
        """
        inputs = [
            NamedArgument(
                name="",
                arg=self.serialize_input(a),
            )
            for a in args
        ]
        inputs.extend(
            [
                NamedArgument(name=name, arg=self.serialize_input(a))
                for name, a in kwargs.items()
            ]
        )
        return inputs

    def is_sym_int_arg(self, arg) -> bool:
        return isinstance(arg, int) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_int_values
        )

    def is_sym_bool_arg(self, arg) -> bool:
        return isinstance(arg, bool) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_bool_values
        )

    def serialize_input(
        self, arg, arg_type: Optional[torch._C.Argument] = None
    ) -> Argument:
        import torch._inductor.ir as inductor_ir

        inductor_tensor_buffers = (
            inductor_ir.Buffer,
            inductor_ir.ReinterpretView,
        )

        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                assert isinstance(arg.target, str)
                attr = getattr(arg.graph.owning_module, arg.target)

                if isinstance(attr, torch.Tensor):
                    raise SerializeError(
                        "getattr nodes containing tensors should not appear in the graph"
                    )
                elif isinstance(attr, torch.fx.GraphModule):
                    with self.save_graph_state():
                        graph = self.serialize_graph(attr)
                    return Argument.create(
                        as_graph=GraphArgument(name=arg.target, graph=graph)
                    )
                else:
                    raise SerializeError(
                        f"Unsupported getattr attribute {arg.target} with type: {type(attr)}"
                    )
            elif self.is_sym_int_arg(arg):
                return Argument.create(
                    as_sym_int=SymIntArgument.create(as_name=arg.name)
                )
            elif self.is_sym_bool_arg(arg):
                return Argument.create(
                    as_sym_bool=SymBoolArgument.create(as_name=arg.name)
                )
            elif isinstance(arg.meta["val"], ep.CustomObjArgument):
                return Argument.create(
                    as_custom_obj=CustomObjArgument(
                        name=arg.name, class_fqn=arg.meta["val"].class_fqn
                    )
                )
            elif arg.name in self.duplicate_getitem_nodes:
                dedup_name = self.duplicate_getitem_nodes[arg.name]
                return Argument.create(as_tensor=TensorArgument(name=dedup_name))
            else:
                return Argument.create(as_tensor=TensorArgument(name=arg.name))
        elif isinstance(arg, inductor_tensor_buffers):
            # Other branches are for arguments in fx node.
            # This is a special branch for handling buffers (representing tensor arguments)
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            arg_name = arg.get_name()
            assert arg_name is not None, "Buffer must have valid name"
            return Argument.create(as_tensor=TensorArgument(name=arg_name))
        elif isinstance(arg, torch.SymInt):
            # This is a special branch for handling SymInt args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_int_arg(arg) being true
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=str(arg)))
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
            if len(arg) == 0:
                if arg_type is not None:
                    if isinstance(arg_type, torch.OptionalType):
                        arg_type = arg_type.getElementType()  # type: ignore[assignment]
                    assert isinstance(arg_type, torch.ListType)
                    elem_type = arg_type.getElementType()
                    if isinstance(elem_type, torch.OptionalType):
                        elem_type = elem_type.getElementType()

                    if isinstance(elem_type, torch.BoolType):
                        return Argument.create(as_bools=[])
                    elif isinstance(elem_type, torch.IntType):
                        return Argument.create(as_ints=[])
                    elif isinstance(elem_type, torch.FloatType):
                        return Argument.create(as_floats=[])
                    elif isinstance(elem_type, torch.StringType):
                        return Argument.create(as_strings=[])
                    elif isinstance(elem_type, torch.TensorType):
                        return Argument.create(as_tensors=[])
                    else:
                        # I believe empty symint lists default to ints, but
                        # please file an issue if this is not the case
                        raise SerializeError(f"Empty list with type {elem_type} nyi.")
                else:
                    # We could serialize this by default to a tensor list. This
                    # is needed in the HOO case
                    log.warning(
                        "Unsure how to serialize the given empty list, "
                        "as we don't know what is the type of this argument. "
                        "Serializing it as a tensor list by default."
                    )
                    return Argument.create(as_tensors=[])

            # Must check bool first, as bool is also treated as int
            if all(isinstance(a, bool) for a in arg):
                return Argument.create(as_bools=list(arg))
            elif all(isinstance(a, int) for a in arg):
                return Argument.create(as_ints=list(arg))
            elif all(isinstance(a, float) for a in arg):
                return Argument.create(as_floats=list(arg))
            elif all(isinstance(a, str) for a in arg):
                return Argument.create(as_strings=list(arg))
            elif all(isinstance(a, torch.SymInt) for a in arg):
                # This is a special branch for handling SymInt args in inductor's
                # ExternalFallbackNode.
                # For regular FX graph, SymInt arg should be a fx.Node with
                # self.is_sym_int_arg(arg) being true
                return Argument.create(
                    as_sym_ints=[SymIntArgument.create(as_name=str(a)) for a in arg]
                )
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
                        raise SerializeError(
                            "getattr nodes containing tensors should not appear in the graph"
                        )
                    arguments.append(TensorArgument(name=a.name))
                return Argument.create(as_tensors=arguments)
            elif all(isinstance(a, (torch.fx.Node, type(None))) for a in arg):
                # list of optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=())
                    elif isinstance(a, torch.fx.Node):
                        return OptionalTensorArgument.create(
                            as_tensor=TensorArgument(name=a.name)
                        )
                    else:
                        raise SerializeError(f"Unsupported list/tuple argument: {a}")

                return Argument.create(
                    as_optional_tensors=list(map(serialize_optional_tensor_args, arg))
                )
            elif all(isinstance(a, inductor_tensor_buffers) for a in arg):
                # list of inductor buffers
                return Argument.create(
                    as_tensors=[TensorArgument(name=a.get_name()) for a in arg],
                )
            elif all(
                isinstance(a, (*inductor_tensor_buffers, type(None))) for a in arg
            ):
                # list of inductor buffers as optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=())
                    elif isinstance(a, inductor_tensor_buffers):
                        return OptionalTensorArgument.create(
                            as_tensor=TensorArgument(name=a.get_name())
                        )
                    else:
                        raise SerializeError(f"Unsupported list/tuple argument: {a}")

                return Argument.create(
                    as_optional_tensors=list(map(serialize_optional_tensor_args, arg))
                )
            else:
                raise SerializeError(
                    f"Unsupported list/tuple argument type: {[type(a) for a in arg]}"
                )
        elif isinstance(arg, torch.dtype):
            return Argument.create(as_scalar_type=_TORCH_TO_SERIALIZE_DTYPE[arg])
        elif isinstance(arg, torch.device):
            return Argument.create(as_device=Device(type=arg.type, index=arg.index))
        elif isinstance(arg, torch.memory_format):
            return Argument.create(
                as_memory_format=_TORCH_TO_SERIALIZE_MEMORY_FORMAT[arg]
            )
        elif isinstance(arg, torch.layout):
            return Argument.create(as_layout=_TORCH_TO_SERIALIZE_LAYOUT[arg])
        elif isinstance(arg, torch._C.ScriptObject):
            if not (
                arg._has_method("__getstate__")  # type: ignore[attr-defined]
                and arg._has_method("__setstate__")  # type: ignore[attr-defined]
            ):
                raise SerializeError(
                    f"Unable to serialize custom class {arg}. Please define "
                    "serialization methods via def_pickle()."
                )
            # Custom objects through torchind are serializable with pickle,
            # through implementing the .def_pickle function.  This should result
            # in the object containing a __getstate__ and __setstate__
            # serialize/deserialize function.
            custom_obj_name = f"_custom_obj_{len(self.custom_objs)}"
            self.custom_objs[custom_obj_name] = arg
            class_fqn = arg._type().qualified_name()  # type: ignore[attr-defined]
            return Argument.create(
                as_custom_obj=CustomObjArgument(custom_obj_name, class_fqn)
            )
        elif isinstance(arg, torch._ops.OpOverload):
            return Argument.create(as_operator=self.serialize_operator(arg))
        else:
            raise SerializeError(f"Unsupported argument type: {type(arg)}")

    def serialize_tensor_output(self, name, meta_val) -> TensorArgument:
        assert name not in self.graph_state.tensor_values
        self.graph_state.tensor_values[name] = serialize_tensor_meta(meta_val)
        return TensorArgument(name=name)

    def serialize_sym_int_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_int_values
        self.graph_state.sym_int_values[name] = serialize_sym_int(meta_val)
        return SymIntArgument.create(as_name=name)

    def serialize_sym_bool_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_bool_values
        self.graph_state.sym_bool_values[name] = serialize_sym_bool(meta_val)
        return SymBoolArgument.create(as_name=name)

    def serialize_input_spec(self, spec: ep.InputSpec) -> InputSpec:
        if spec.kind == ep.InputKind.USER_INPUT:
            if isinstance(spec.arg, ep.ConstantArgument):
                if isinstance(spec.arg.value, int):
                    constant_spec = ConstantValue.create(as_int=spec.arg.value)
                elif isinstance(spec.arg.value, bool):
                    constant_spec = ConstantValue.create(as_bool=spec.arg.value)
                elif isinstance(spec.arg.value, str):
                    constant_spec = ConstantValue.create(as_string=spec.arg.value)
                elif isinstance(spec.arg.value, float):
                    constant_spec = ConstantValue.create(as_float=spec.arg.value)
                elif spec.arg.value is None:
                    constant_spec = ConstantValue.create(as_none=())
                else:
                    raise SerializeError(f"Unhandled constant input {spec.arg.value} to serialize")
                return InputSpec.create(
                    constant_input=ConstantInputSpec(
                        name=spec.arg.name, value=constant_spec
                    )
                )
            else:
                return InputSpec.create(
                    user_input=UserInputSpec(
                        arg=self.serialize_argument_spec(spec.arg)
                    )
                )
        elif spec.kind == ep.InputKind.PARAMETER:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return InputSpec.create(
                parameter=InputToParameterSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    parameter_name=spec.target,
                )
            )
        elif spec.kind == ep.InputKind.BUFFER:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            assert spec.persistent is not None
            return InputSpec.create(
                buffer=InputToBufferSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    buffer_name=spec.target,
                    persistent=spec.persistent,
                )
            )
        elif spec.kind == ep.InputKind.CONSTANT_TENSOR:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return InputSpec.create(
                tensor_constant=InputToTensorConstantSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    tensor_constant_name=spec.target,
                )
            )
        elif spec.kind == ep.InputKind.CUSTOM_OBJ:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.CustomObjArgument)
            return InputSpec.create(
                custom_obj=InputToCustomObjSpec(
                    arg=CustomObjArgument(
                        name=spec.arg.name, class_fqn=spec.arg.class_fqn
                    ),
                    custom_obj_name=spec.target,
                )
            )
        elif spec.kind == ep.InputKind.TOKEN:
            assert isinstance(spec.arg, ep.TokenArgument)
            return InputSpec.create(
                token=InputTokenSpec(
                    arg=TokenArgument(name=spec.arg.name),
                )
            )
        else:
            raise AssertionError(f"Unknown argument kind: {spec}")

    def serialize_output_spec(self, spec: ep.OutputSpec) -> OutputSpec:
        if spec.kind == ep.OutputKind.USER_OUTPUT:
            return OutputSpec.create(
                user_output=UserOutputSpec(arg=self.serialize_argument_spec(spec.arg))
            )
        elif spec.kind == ep.OutputKind.LOSS_OUTPUT:
            assert isinstance(spec.arg, ep.TensorArgument)
            return OutputSpec.create(
                loss_output=LossOutputSpec(arg=TensorArgument(name=spec.arg.name))
            )
        elif spec.kind == ep.OutputKind.BUFFER_MUTATION:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return OutputSpec.create(
                buffer_mutation=BufferMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    buffer_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.GRADIENT_TO_PARAMETER:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return OutputSpec.create(
                gradient_to_parameter=GradientToParameterSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    parameter_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.GRADIENT_TO_USER_INPUT:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return OutputSpec.create(
                gradient_to_user_input=GradientToUserInputSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    user_input_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.USER_INPUT_MUTATION:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return OutputSpec.create(
                user_input_mutation=UserInputMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    user_input_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.TOKEN:
            assert isinstance(spec.arg, ep.TokenArgument)
            return OutputSpec.create(
                token=OutputTokenSpec(
                    arg=TokenArgument(name=spec.arg.name),
                )
            )
        else:
            raise AssertionError(f"Unknown argument kind: {spec}")

    def serialize_signature(self, sig: ep.ExportGraphSignature) -> GraphSignature:
        return GraphSignature(
            input_specs=[self.serialize_input_spec(s) for s in sig.input_specs],
            output_specs=[self.serialize_output_spec(s) for s in sig.output_specs],
        )

    def serialize_argument_spec(self, x: ep.ArgumentSpec) -> Argument:
        if isinstance(x, ep.TensorArgument):
            return Argument.create(as_tensor=TensorArgument(name=x.name))
        elif isinstance(x, ep.SymIntArgument):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.name))
        elif isinstance(x, ep.ConstantArgument):
            return self.serialize_input(x.value)
        elif isinstance(x, ep.CustomObjArgument):
            return Argument.create(
                as_custom_obj=CustomObjArgument(name=x.name, class_fqn=x.class_fqn)
            )
        else:
            raise AssertionError("TODO")

    def serialize_module_call_signature(
        self, module_call_signature: ep.ModuleCallSignature
    ) -> ModuleCallSignature:
        return ModuleCallSignature(
            inputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=treespec_dumps(module_call_signature.in_spec, TREESPEC_VERSION),
            out_spec=treespec_dumps(module_call_signature.out_spec, TREESPEC_VERSION),
        )

    def serialize_module_call_graph(
        self, module_call_graph: List[ep.ModuleCallEntry]
    ) -> List[ModuleCallEntry]:
        return [
            ModuleCallEntry(
                fqn=entry.fqn,
                signature=(
                    self.serialize_module_call_signature(entry.signature)
                    if entry.signature
                    else None
                ),
            )
            for entry in module_call_graph
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
        assert node.op == "call_function" and isinstance(node.target, (torch._ops.OpOverload, *allowed_registered_op_types()))

        schema = _get_schema_from_target(node.target)
        returns = schema.returns

        if len(returns) == 0:
            return []

        meta_val = node.meta["val"]

        # Check single value return
        if _is_single_tensor_list_return(node.target):
            # e.g "-> Tensor[]"
            tensor_args = []
            for idx, meta in enumerate(meta_val):
                user_node = self._output_node_at_index(node, idx)
                name = (
                    user_node.name
                    if user_node is not None
                    else f"{node.name}_unused_{idx}"
                )
                tensor_args.append(self.serialize_tensor_output(name, meta))
            return [Argument.create(as_tensors=tensor_args)]
        elif len(returns) == 1:
            return [self.serialize_output(node.name, meta_val)]

        # There are a two possibilities at this point:
        # - This operator returns a tuple of Tensors, e.g. "-> (Tensor, Tensor)"
        # - This operator returns a tuple of mixed of Tensor and Tensors, e.g. "-> (Tensor, Tensor[])"
        #
        # Either way, start by gathering a list of TensorArguments with the correct names.
        # For consistent naming with FX, consult the downstream `getitem` node and
        # make sure our outputs have the same name.

        output_arguments = []
        for idx, (meta, return_schema) in enumerate(zip(meta_val, returns)):
            if meta is None:
                assert isinstance(
                    return_schema.real_type, (torch.OptionalType, torch.TensorType)
                )
                # When the return type is annoated as Tensor type, the op can also return an
                # undefined Tensor which will be implicitly converted to None in Python.
                output_arguments.append(Argument.create(as_none=()))
            elif isinstance(meta, FakeTensor):
                assert isinstance(return_schema.real_type, (torch.OptionalType, torch.TensorType))
                user_node = self._output_node_at_index(node, idx)
                name = (
                    user_node.name
                    if user_node is not None
                    else f"{node.name}_unused_{idx}"
                )
                output_arguments.append(self.serialize_output(name, meta))
            elif isinstance(meta, list):
                # for List[Tensor] return type
                assert isinstance(
                    return_schema.real_type, torch.ListType
                ) and isinstance(
                    return_schema.real_type.getElementType(), torch.TensorType
                )
                user_node = self._output_node_at_index(node, idx)
                assert user_node is not None

                args = []
                for i, m in enumerate(meta):
                    if m is None:
                        continue
                    sub_user_node = self._output_node_at_index(user_node, i)
                    assert sub_user_node is not None, f"No user found at index {i}"

                    args.append(self.serialize_tensor_output(sub_user_node.name, m))
                output_arguments.append(Argument.create(as_tensors=args))
            elif isinstance(meta, (int, SymInt)):
                user_node = self._output_node_at_index(node, idx)
                name = (
                    user_node.name
                    if user_node is not None
                    else f"{node.name}_unused_{idx}"
                )
                output_arguments.append(self.serialize_output(name, meta))
            else:
                raise ValueError(
                    f"Unhandled output type {type(meta)} from node {node.format_node()}"
                )

        return output_arguments

    def serialize_hoo_outputs(self, node: torch.fx.Node) -> List[Argument]:
        """
        For serializing HOO outputs since HOOs do not have a schema.
        """
        meta_val = node.meta["val"]

        if isinstance(meta_val, tuple):
            # Note: Since we don't have a schema, we just serialize all tuple
            # outputs to be a list of values. Even if the output is supposed to
            # be a tensor list (Tensor[]), we will serialize it to be a list of
            # tensors (Tensor, Tensor, Tensor). An exception is that if there's
            # a singleton tensor, we will serialize this to be a singleton
            # tensor list so that the deserializer knows to insert getitem nodes.

            if len(meta_val) == 1:
                assert isinstance(meta_val[0], torch.Tensor)
                user_node = self._output_node_at_index(node, 0)
                name = (
                    user_node.name
                    if user_node is not None
                    else f"{node.name}_unused_0"
                )
                return [Argument.create(as_tensors=[self.serialize_tensor_output(name, meta_val[0])])]

            outputs = []
            for i, element_meta_val in enumerate(meta_val):
                user_node = self._output_node_at_index(node, i)
                if isinstance(element_meta_val, list):
                    # e.g "-> Tensor[]"
                    assert user_node is not None

                    tensors = []
                    for j, m in enumerate(element_meta_val):
                        if not isinstance(m, torch.Tensor):
                            raise SerializeError(f"Serialize list output with type {type(m)} nyi")

                        sub_user_node = self._output_node_at_index(user_node, j)
                        name = (
                            sub_user_node.name
                            if sub_user_node is not None
                            else f"{user_node.name}_unused_{j}"
                        )
                        tensors.append(self.serialize_tensor_output(name, m))
                    outputs.append(Argument.create(as_tensors=tensors))

                else:
                    name = (
                        user_node.name
                        if user_node is not None
                        else f"{node.name}_unused_{i}"
                    )

                    outputs.append(self.serialize_output(name, element_meta_val))

            return outputs
        else:
            return [self.serialize_output(node.name, meta_val)]

    def serialize_output(self, name: str, meta_val: Any) -> Argument:
        # Check single value return
        if meta_val is None:
            return Argument.create(as_none=())
        if isinstance(meta_val, torch.Tensor):
            # e.g "-> Tensor"
            return Argument.create(
                as_tensor=self.serialize_tensor_output(name, meta_val)
            )
        elif isinstance(meta_val, (int, torch.SymInt)):
            # e.g "-> SymInt"
            return Argument.create(
                as_sym_int=self.serialize_sym_int_output(name, meta_val)
            )
        elif isinstance(meta_val, torch.SymBool):
            # e.g "-> SymBool"
            return Argument.create(
                as_sym_bool=self.serialize_sym_bool_output(name, meta_val)
            )

        # list outputs should've been handled earlier
        raise SerializeError(f"Unable to serialize output {meta_val}")

    def _handle_getitem_users(self, node: torch.fx.Node) -> List[TensorArgument]:
        meta_val = node.meta["val"]

        idx_to_name = {}
        for user in node.users:
            assert (
                user.target is operator.getitem
            ), f"User node {user} of {node} is incorrect"
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
                raise SerializeError(
                    f"Failed serializing node {node} in graph: {node.format_node()}"
                ) from e

        return Graph(
            inputs=self.graph_state.inputs,
            nodes=self.graph_state.nodes,
            tensor_values=self.graph_state.tensor_values,
            sym_int_values=self.graph_state.sym_int_values,
            sym_bool_values=self.graph_state.sym_bool_values,
            custom_obj_values=self.graph_state.custom_obj_values,
            outputs=self.graph_state.outputs,
            is_single_tensor_return=self.graph_state.is_single_tensor_return,
        )

    def serialize(self, graph_module: torch.fx.GraphModule) -> GraphModule:
        graph = self.serialize_graph(graph_module)

        return GraphModule(
            graph=graph,
            signature=self.serialize_signature(self.graph_signature),
            module_call_graph=self.serialize_module_call_graph(self.module_call_graph),
        )


@final
class ExportedProgramSerializer(metaclass=Final):
    def __init__(self, opset_version: Optional[Dict[str, int]] = None):
        self.opset_version: Dict[str, int] = {}
        if opset_version:
            self.opset_version.update(opset_version)
        if "aten" not in self.opset_version:
            self.opset_version["aten"] = torch._C._get_max_operator_version()

    def serialize(self, exported_program: ep.ExportedProgram) -> _SerializedProgram:
        """
        Args:
            exported_program: Exported Program to serialize
        """
        exported_program._validate()

        gm_serializer = GraphModuleSerializer(
            exported_program.graph_signature, exported_program.module_call_graph
        )
        serialized_graph_module = gm_serializer.serialize(exported_program.graph_module)
        serialized_range_constraints = serialize_range_constraints(
            exported_program.range_constraints
        )

        # TODO: Directly serialize exported_program.constants once
        # CustomClassHolders get stored in the ExportedProgram rather than in
        # the graph
        constants = {}
        for n, c in gm_serializer.custom_objs.items():
            constants[n] = c
        for n, t in exported_program.constants.items():
            assert n not in constants
            constants[n] = t

        serialized_ep = ExportedProgram(
            graph_module=serialized_graph_module,
            opset_version=self.opset_version,
            range_constraints=serialized_range_constraints,
            schema_version=SchemaVersion(
                major=SCHEMA_VERSION[0],
                minor=SCHEMA_VERSION[1],
            ),
            dialect=exported_program.dialect
        )

        # Test canonical form is well defined.
        canonicalize(serialized_ep)

        return _SerializedProgram(
            serialized_ep,
            serialize_torch_artifact(exported_program.state_dict),
            serialize_torch_artifact(constants),
            serialize_torch_artifact(exported_program.example_inputs),
        )


@final
class GraphModuleDeserializer(metaclass=Final):
    @dataclasses.dataclass
    class Result:
        graph_module: torch.fx.GraphModule
        signature: ep.ExportGraphSignature
        module_call_graph: List[ep.ModuleCallEntry]
        names_to_symbols: Dict[str, sympy.Symbol]
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]]
        constants: Dict[str, Union[torch.Tensor, torch.ScriptObject]]
        example_inputs: Optional[Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]]

    def __init__(self):
        self.serialized_name_to_node: Dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: Dict[str, MetaType] = {}
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()

    @contextmanager
    def save_graph_module(self) -> Iterator[None]:
        saved = (
            self.graph,
            self.module,
            self.serialized_name_to_node,
            self.serialized_name_to_meta,
        )
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()
        self.serialized_name_to_node = {}
        self.serialized_name_to_meta = {}
        try:
            yield
        finally:
            (
                self.graph,
                self.module,
                self.serialized_name_to_node,
                self.serialized_name_to_meta,
            ) = saved

    def deserialize_operator(self, serialized_target: str):
        if serialized_target.startswith(
            "_operator"
        ):  # TODO(zhxchen17) Follow up on this.
            module = operator
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("torch"):
            module = torch  # type: ignore[misc]
            serialized_target_names = serialized_target.split(".")[1:]
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
            if val.hint is None:
                hint = None
            else:
                assert val.hint.type == "as_int"
                hint = val.hint.value

            if val.expr_str in self.symbol_name_to_symbol:
                sym = self.symbol_name_to_symbol[val.expr_str]
            else:
                sym = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
                # NOTE(avik): Assumptions on symbols are not explicitly serialized.
                # This seems dangerous: it might cause unknown differences in shape env behavior
                # on deserialization? Probably deserves a follow-up.

                # Here we force symbols corresponding to SymInts to be at least integers.
                # Otherwise some expressions that the shape env would otherwise evaluate to False,
                # e.g., 2*s = 9, can have rational solutions, e.g., 9/2.
                # TODO: This is HIGHLY SUSPICIOUS ezyang(May 2024)
                sym = sym.subs(
                    {s: sympy.Symbol(s.name, integer=True) for s in sym.free_symbols}
                )
                # We need to check if the symbol has already been allocated,
                # self.symbol_name_to_symbol is not enough because the
                # integer-ification of symbols can induce simplification;
                # e.g., (2**s0 + 1) // 2  -->  s0 when we know s0 is integral
                if isinstance(sym, sympy.Symbol) and sym not in self.shape_env.var_to_val:
                    self.symbol_name_to_symbol[val.expr_str] = sym
                    if hint is not None:
                        self.shape_env.add_var_to_val(sym, hint)

                    if vr := self.symbol_name_to_range.get(val.expr_str):
                        self.shape_env.constrain_symbol_range(
                            sym,
                            compiler_min=vr.lower,  # type: ignore[arg-type]
                            compiler_max=vr.upper,  # type: ignore[arg-type]
                        )
                else:
                    # Placeholders, in particular, can have shapes as symbolic expressions.
                    # We need to populate the shape env with the range constraints of their
                    # free symbols, otherwise evaluating such expressions will error.
                    self.symbol_name_to_symbol[val.expr_str] = sym
                    free_symbols = sym.free_symbols
                    for s in free_symbols:
                        if s.name not in self.symbol_name_to_symbol:
                            self.symbol_name_to_symbol[s.name] = s  # type: ignore[assignment]
                        if vr := self.symbol_name_to_range.get(s.name):
                            self.shape_env.constrain_symbol_range(
                                s,
                                compiler_min=vr.lower,  # type: ignore[arg-type]
                                compiler_max=vr.upper,  # type: ignore[arg-type]
                            )

            return self.shape_env.create_symintnode(sym, hint=hint)
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
            expr = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
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
    ) -> FakeTensor:
        with self.fake_tensor_mode:
            return cast(
                FakeTensor,
                torch.empty_strided(
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.sizes),  # type: ignore[misc]
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.strides),  # type: ignore[misc]
                    device=deserialize_device(tensor_meta.device),
                    dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
                ),
            )

    def deserialize_script_obj_meta(
        self, script_obj_meta: CustomObjArgument
    ) -> ep.CustomObjArgument:
        return ep.CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    def deserialize_graph_output(self, output) -> Optional[Union[torch.fx.Node, int]]:
        if output.type == "as_tensor":
            return self.serialized_name_to_node[output.as_tensor.name]
        elif output.type == "as_sym_int":
            return self.serialized_name_to_node[output.as_sym_int.as_name]
        elif output.type == "as_sym_bool":
            return self.serialized_name_to_node[output.as_sym_bool.as_name]
        elif output.type == "as_int":
            return output.as_int
        elif output.type == "as_none":
            return None
        else:
            raise SerializeError(f"Unable to deserialize output node {output}")

    def deserialize_graph(self, serialized_graph: Graph) -> torch.fx.Graph:
        # Handle the tensor metas.
        for name, tensor_value in serialized_graph.tensor_values.items():
            meta_val = self.deserialize_tensor_meta(tensor_value)
            self.serialized_name_to_meta[name] = meta_val

        for name, sym_int_value in serialized_graph.sym_int_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_sym_int(sym_int_value)

        for name, sym_bool_value in serialized_graph.sym_bool_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_sym_bool(
                sym_bool_value
            )

        for name, script_obj_meta in serialized_graph.custom_obj_values.items():
            self.serialized_name_to_meta[name] = self.deserialize_script_obj_meta(
                script_obj_meta
            )

        # Inputs: convert to placeholder nodes in FX.
        for i, input_ in enumerate(serialized_graph.inputs):
            if input_.type in ("as_tensor", "as_sym_int", "as_custom_obj"):
                node_name = input_.value.name
                placeholder_node = self.graph.placeholder(node_name)
                # FX might declare a name illegal (e.g. some nn.Modules use "input" as forward() arguments)
                # we will overwrite it
                placeholder_node.name = node_name
                self.sync_fx_node(node_name, placeholder_node)
            elif input_.type in (
                "as_int",
                "as_float",
                "as_bool",
                "as_none",
                "as_string",
            ):
                node_name = self.signature.input_specs[i].arg.name
                placeholder_node = self.graph.placeholder(node_name)
                placeholder_node.meta["val"] = self.deserialize_input(input_)
            else:
                raise SerializeError(f"Invalid input type {input_}")

        # Nodes: convert to call_function nodes.
        for serialized_node in serialized_graph.nodes:
            try:
                target = self.deserialize_operator(serialized_node.target)
                self.deserialize_node(serialized_node, target)

            except Exception as e:
                raise SerializeError(
                    f"Failed deserializing node {serialized_node}"
                ) from e

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
                arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
                for arg in output_node.args[0]
            )

        return self.graph

    def deserialize_node(self, serialized_node: Node, target: Callable) -> None:
        if target in _SYM_BOOL_OPS or target in _SYM_INT_OPS:
            name = serialized_node.outputs[0].value.as_name
            args = self.deserialize_sym_op_inputs(serialized_node.inputs)

            fx_node = self.graph.create_node("call_function", target, args, {}, name)
            self.deserialize_sym_op_outputs(serialized_node, fx_node)

        elif isinstance(target, torch._ops.HigherOrderOperator):
            args, kwargs = self.deserialize_hoo_inputs(serialized_node.inputs)
            # If HOP returns a single tensor, name the
            # newly-created node after it. This ensures that these tensor values
            # have names that are consistent with serialized.
            #
            # HOPs don't have schema yet, just check the output lengths and as_tensor attribute
            name = (
                serialized_node.outputs[0].as_tensor.name
                if len(serialized_node.outputs) == 1
                and hasattr(serialized_node.outputs[0], "as_tensor")
                else None
            )
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
            fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))

        elif isinstance(target, torch._ops.OpOverload):
            # For convenience: if this node returns a single tensor, name the
            # newly-created node after it. This ensures that these tensor values
            # have names that are consistent with serialized.
            name = (
                serialized_node.outputs[0].as_tensor.name
                if _is_single_tensor_return(target)
                else None  # FX will generate a name for us.
            )
            args, kwargs = self.deserialize_inputs(target, serialized_node)
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
        else:
            raise SerializeError(
                f"Unsupported target type for node {serialized_node}: {type(target)}"
            )

        fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))
        if fx_node.op not in ["placeholder", "output"] and "nn_module_stack" not in fx_node.meta:
            fx_node.meta["nn_module_stack"] = {}  # serialization throws away empty dicts

    def deserialize_input_spec(self, i: InputSpec) -> ep.InputSpec:
        if i.type == "user_input":
            return ep.InputSpec(
                kind=ep.InputKind.USER_INPUT,
                arg=self.deserialize_argument_spec(i.user_input.arg),
                target=None,
            )
        elif i.type == "parameter":
            return ep.InputSpec(
                kind=ep.InputKind.PARAMETER,
                arg=ep.TensorArgument(name=i.parameter.arg.name),
                target=i.parameter.parameter_name,
            )
        elif i.type == "buffer":
            return ep.InputSpec(
                kind=ep.InputKind.BUFFER,
                arg=ep.TensorArgument(name=i.buffer.arg.name),
                target=i.buffer.buffer_name,
                persistent=i.buffer.persistent,
            )
        elif i.type == "tensor_constant":
            return ep.InputSpec(
                kind=ep.InputKind.CONSTANT_TENSOR,
                arg=ep.TensorArgument(name=i.tensor_constant.arg.name),
                target=i.tensor_constant.tensor_constant_name,
            )
        elif i.type == "custom_obj":
            return ep.InputSpec(
                kind=ep.InputKind.CUSTOM_OBJ,
                arg=ep.CustomObjArgument(
                    name=i.custom_obj.arg.name, class_fqn=i.custom_obj.arg.class_fqn
                ),
                target=i.custom_obj.custom_obj_name,
            )
        elif i.type == "token":
            return ep.InputSpec(
                kind=ep.InputKind.TOKEN,
                arg=ep.TokenArgument(name=i.token.arg.name),
                target=None
            )
        elif i.type == "constant_input":
            return ep.InputSpec(
                kind=ep.InputKind.USER_INPUT,
                arg=ep.ConstantArgument(
                    name=i.constant_input.name,
                    value=self.deserialize_constant_input(i.constant_input.value)
                ),
                target=None,
            )
        else:
            raise AssertionError(f"Unknown input spec {i}")

    def deserialize_output_spec(self, o: OutputSpec) -> ep.OutputSpec:
        if o.type == "user_output":
            return ep.OutputSpec(
                kind=ep.OutputKind.USER_OUTPUT,
                arg=self.deserialize_argument_spec(o.user_output.arg),
                target=None,
            )
        elif o.type == "loss_output":
            return ep.OutputSpec(
                kind=ep.OutputKind.LOSS_OUTPUT,
                arg=ep.TensorArgument(name=o.loss_output.arg.name),
                target=None,
            )
        elif o.type == "buffer_mutation":
            return ep.OutputSpec(
                kind=ep.OutputKind.BUFFER_MUTATION,
                arg=ep.TensorArgument(name=o.buffer_mutation.arg.name),
                target=o.buffer_mutation.buffer_name,
            )
        elif o.type == "gradient_to_parameter":
            return ep.OutputSpec(
                kind=ep.OutputKind.GRADIENT_TO_PARAMETER,
                arg=ep.TensorArgument(name=o.gradient_to_parameter.arg.name),
                target=o.gradient_to_parameter.parameter_name,
            )
        elif o.type == "gradient_to_user_input":
            return ep.OutputSpec(
                kind=ep.OutputKind.GRADIENT_TO_USER_INPUT,
                arg=ep.TensorArgument(name=o.gradient_to_user_input.arg.name),
                target=o.gradient_to_user_input.user_input_name,
            )
        elif o.type == "user_input_mutation":
            return ep.OutputSpec(
                kind=ep.OutputKind.USER_INPUT_MUTATION,
                arg=ep.TensorArgument(name=o.user_input_mutation.arg.name),
                target=o.user_input_mutation.user_input_name,
            )
        elif o.type == "token":
            return ep.OutputSpec(
                kind=ep.OutputKind.TOKEN,
                arg=ep.TokenArgument(name=o.token.arg.name),
                target=None
            )
        else:
            raise AssertionError(f"Unknown output spec {o}")

    def deserialize_signature(self, sig: GraphSignature) -> ep.ExportGraphSignature:
        return ep.ExportGraphSignature(
            input_specs=[self.deserialize_input_spec(i) for i in sig.input_specs],
            output_specs=[self.deserialize_output_spec(o) for o in sig.output_specs],
        )

    def deserialize(
        self,
        serialized_graph_module: GraphModule,
        serialized_state_dict: Union[Dict[str, torch.Tensor], bytes],
        constants: Union[Dict[str, Any], bytes],
        example_inputs: Optional[Union[Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]], bytes]] = None,
        symbol_name_to_range: Optional[Dict[str, symbolic_shapes.ValueRanges]] = None,
    ) -> Result:
        global _CURRENT_DESERIALIZER
        assert _CURRENT_DESERIALIZER is None
        _CURRENT_DESERIALIZER = self
        try:
            self.shape_env = symbolic_shapes.ShapeEnv(assume_static_by_default=True)
            self.fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=True,
                shape_env=self.shape_env,
            )
            self.symbol_name_to_symbol: Dict[str, sympy.Symbol] = {}
            self.constants = deserialize_torch_artifact(constants)
            self.signature = self.deserialize_signature(serialized_graph_module.signature)

            # deserialization does analysis with checks on 0/1, so we create fake range constraints and
            # restore the original range constraints afterwards
            self.symbol_name_to_range = {}
            if symbol_name_to_range:
                for k, vr in symbol_name_to_range.items():
                    lower = int(vr.lower)
                    if vr.upper >= 2:  # max is >= 2, not sym bool range
                        lower = max(2, lower)
                    self.symbol_name_to_range[k] = symbolic_shapes.ValueRanges(_int_to_sympy_int(lower), vr.upper)

            if example_inputs is not None and len(example_inputs) > 0:
                self.example_inputs = deserialize_torch_artifact(example_inputs)
            else:
                self.example_inputs = None
            self.deserialize_graph(serialized_graph_module.graph)

            module_call_graph = self.deserialize_module_call_graph(
                serialized_graph_module.module_call_graph
            )
            return GraphModuleDeserializer.Result(
                graph_module=ep._create_graph_module_for_export(
                    self.module, self.graph
                ),
                signature=self.signature,
                module_call_graph=module_call_graph,
                names_to_symbols=self.symbol_name_to_symbol,
                state_dict=deserialize_torch_artifact(serialized_state_dict),
                constants=self.constants,
                example_inputs=self.example_inputs,
            )
        finally:
            _CURRENT_DESERIALIZER = None

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
            input.name: self.deserialize_input(input.arg)
            for input in serialized_node.inputs
        }
        args = []
        kwargs = {}
        for schema_arg in schema_args:
            is_positional = (
                not schema_arg.has_default_value() and not schema_arg.kwarg_only
            )
            if is_positional:
                args.append(actual_args[schema_arg.name])
            else:
                if schema_arg.name in actual_args:
                    kwargs[schema_arg.name] = actual_args[schema_arg.name]
        return tuple(args), kwargs

    def deserialize_hoo_inputs(self, inputs: List[NamedArgument]):
        """
        For deserializing HOO inputs since HOOs do not have a schema.
        """
        args = []
        kwargs = {}
        for input_ in inputs:
            if input_.name != "":
                kwargs[input_.name] = self.deserialize_input(input_.arg)
            else:
                args.append(self.deserialize_input(input_.arg))
        return (tuple(args), kwargs)

    def deserialize_input(self, inp: Argument) -> Any:
        value = inp.value
        typ_ = inp.type
        if typ_ == "as_none":
            # None should converted as None, but is encoded as bool in serialized
            # Convert serialized object to torch equivalent
            return None
        elif typ_ == "as_tensor":
            return self.serialized_name_to_node[inp.as_tensor.name]
        elif typ_ == "as_scalar_type":
            return _SERIALIZE_TO_TORCH_DTYPE[inp.as_scalar_type]
        elif typ_ == "as_memory_format":
            return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[inp.as_memory_format]
        elif typ_ == "as_layout":
            return _SERIALIZE_TO_TORCH_LAYOUT[inp.as_layout]
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
        elif typ_ == "as_device":
            return deserialize_device(inp.as_device)
        elif typ_ == "as_int":
            return inp.as_int
        elif typ_ == "as_float":
            return inp.as_float
        elif typ_ == "as_bool":
            return inp.as_bool
        elif typ_ == "as_string":
            return inp.as_string
        elif typ_ == "as_sym_int":
            return self.deserialize_sym_argument(inp.as_sym_int)
        elif typ_ == "as_sym_bool":
            return self.deserialize_sym_argument(inp.as_sym_bool)
        elif isinstance(value, list):
            if len(value) == 0:
                return []
            elif typ_ == "as_tensors":
                result = []
                for arg in value:
                    result.append(self.serialized_name_to_node[arg.name])
                return result
            elif typ_ in ("as_ints", "as_floats", "as_bools", "as_strings"):
                # convert from serialized.python.types.List to python list
                return list(value)
            elif typ_ in ("as_sym_ints", "as_sym_bools"):
                return [self.deserialize_sym_argument(arg) for arg in value]
            elif typ_ == "as_optional_tensors":

                def deserialize_optional_tensor_args(a):
                    if a.type == "as_none":
                        return None
                    elif a.type == "as_tensor":
                        return self.serialized_name_to_node[a.value.name]
                    else:
                        raise SerializeError(f"Unhandled argument {inp}")

                return list(map(deserialize_optional_tensor_args, value))
            else:
                raise SerializeError(f"Unhandled argument {inp}")
        elif typ_ == "as_custom_obj":
            if inp.as_custom_obj.name in self.serialized_name_to_node:
                # Custom object has been lifted as an input
                return self.serialized_name_to_node[inp.as_custom_obj.name]
            return self.constants[inp.as_custom_obj.name]
        elif typ_ == "as_operator":
            return self.deserialize_operator(inp.as_operator)
        else:
            raise SerializeError(f"Unhandled argument {inp}")

    def deserialize_constant_input(self, inp: ConstantValue) -> Any:
        if inp.type == "as_int":
            return int(inp.as_int)
        elif inp.type == "as_float":
            return float(inp.as_float)
        elif inp.type == "as_string":
            return str(inp.as_string)
        elif inp.type == "as_bool":
            return bool(inp.as_bool)
        elif inp.type == "as_none":
            return None
        else:
            raise SerializeError(f"Unhandled constant argument {inp} to deserialize")

    def deserialize_sym_argument(self, sym_arg):
        if isinstance(sym_arg, SymIntArgument):
            if sym_arg.type == "as_int":
                return sym_arg.as_int
            elif sym_arg.type == "as_name":
                return self.serialized_name_to_node[sym_arg.as_name]
        elif isinstance(sym_arg, SymBoolArgument):
            if sym_arg.type == "as_bool":
                return sym_arg.as_bool
            elif sym_arg.type == "as_name":
                return self.serialized_name_to_node[sym_arg.as_name]
        raise SerializeError(f"Unknown symbolic argument type: {sym_arg}")

    def deserialize_sym_op_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)

    def deserialize_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        # Check single value return
        if len(serialized_node.outputs) == 0:
            return
        if (
            len(serialized_node.outputs) == 1
            and serialized_node.outputs[0].type == "as_tensor"
        ):
            self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
            return
        elif len(serialized_node.outputs) == 1 and isinstance(
            serialized_node.outputs[0].value, (SymIntArgument, SymBoolArgument)
        ):
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return

        self.deserialize_multiple_outputs(serialized_node, fx_node)

    def deserialize_multiple_outputs(
        self, serialized_node: Node, fx_node: torch.fx.Node
    ) -> None:
        deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)

        def generate_getitem(
            meta_val,
            fx_node: torch.fx.Node,
            arg: Union[TensorArgument, SymIntArgument],
            idx: int,
        ):
            if isinstance(arg, TensorArgument):
                name = arg.name
            elif isinstance(arg, SymIntArgument):
                name = arg.as_name
            else:
                raise AssertionError(
                    f"generate_getitem got unknown argument type {type(arg)}"
                )
            individual_output = self.graph.create_node(
                "call_function",
                operator.getitem,
                (fx_node, idx),
                name=name,
            )
            self.sync_fx_node(name, individual_output)
            meta_val.append(self.serialized_name_to_meta[name])
            # The derived `getitem` nodes should have the same stacktrace as the
            # original `fx_node`
            individual_output.meta.update(deserialized_metadata)

        def generate_getitems(meta_val, fx_node: torch.fx.Node, args):
            for idx, arg in enumerate(args):
                if isinstance(arg, Argument):
                    arg = arg.value
                if isinstance(arg, (TensorArgument, SymIntArgument)):
                    generate_getitem(meta_val, fx_node, arg, idx)
                elif isinstance(arg, (list, tuple)):
                    list_output = self.graph.create_node(
                        "call_function",
                        operator.getitem,
                        (fx_node, idx),
                    )
                    meta_val.append([])
                    generate_getitems(meta_val[-1], list_output, arg)
                    list_output.meta.update(deserialized_metadata)
                    list_output.meta["val"] = meta_val[-1]
                else:
                    raise NotImplementedError(f"Unimplemented node output type: {arg}")

        # Convert multiple return types to FX format.
        # In FX, each node only returns one value. So in order to represent
        # multiple return values, we have to emit a `getitem` node for each
        # return value.
        # This performs the inverse mapping of the `serialize_outputs` call in
        # serialization, see [NOTE: Multiple outputs]
        meta_val: List[Any] = []
        if len(serialized_node.outputs) == 1:
            assert isinstance(serialized_node.outputs[0].value, list)
            assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
            generate_getitems(meta_val, fx_node, serialized_node.outputs[0].as_tensors)
        else:
            generate_getitems(meta_val, fx_node, serialized_node.outputs)

        # also update the metaval for `fx_node` to be a list(meta)
        fx_node.meta["val"] = tuple(meta_val)
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
            # Originally serialized to "key,orig_path,type_str"
            def import_nn_module_stack(key, path, ty):
                return key, (path, ty)

            # Helper function that splits strings by commas except for those
            # encapsulated by parens, which are valid traces.
            # TODO: Currently this is needed due to indexing Sequential
            # layers introducing names in the form "layer.slice(1, None, None)".
            # If that naming is improved, this fancier splitting can probably be
            # reverted to a simple split by comma.
            def metadata_split(metadata):
                # Remove the parentheses and commas inside them
                metadata = re.sub(r'\(.*?\)', '', metadata)
                # Split the string by comma, except for those inside parentheses
                return re.split(r'(?<!\()\s*,\s*(?!\()', metadata)

            nn_module_stack = dict(
                import_nn_module_stack(*metadata_split(item))
                for item in nn_module_stack_str.split(ST_DELIMITER)
            )
            ret["nn_module_stack"] = nn_module_stack

        if source_fn_st_str := metadata.get("source_fn_stack"):
            # Originally serializes to "fx_node_name,op_str"
            source_fn_st = []
            for source_fn_str in source_fn_st_str.split(ST_DELIMITER):
                name, target_str = source_fn_str.split(",")
                source_fn_st.append((name, deserialize_meta_func(target_str)))
            ret["source_fn_stack"] = source_fn_st

        if torch_fn_str := metadata.get("torch_fn"):
            ret["torch_fn"] = tuple(torch_fn_str.split(ST_DELIMITER))
        return ret

    def deserialize_argument_spec(self, x: Argument) -> ep.ArgumentSpec:
        if x.type == "as_tensor":
            return ep.TensorArgument(name=x.as_tensor.name)
        elif x.type == "as_sym_int":
            return ep.SymIntArgument(name=x.as_sym_int.as_name)
        elif x.type == "as_custom_obj":
            return ep.ConstantArgument(name=x.as_custom_obj.name, value=self.deserialize_input(x))
        else:
            return ep.ConstantArgument(name="", value=self.deserialize_input(x))

    def deserialize_module_call_signature(
        self, module_call_signature: ModuleCallSignature
    ) -> ep.ModuleCallSignature:
        return ep.ModuleCallSignature(
            inputs=[
                self.deserialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.deserialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=treespec_loads(module_call_signature.in_spec),
            out_spec=treespec_loads(module_call_signature.out_spec),
        )

    def deserialize_module_call_graph(
        self, module_call_graph: List[ModuleCallEntry]
    ) -> List[ep.ModuleCallEntry]:
        return [
            ep.ModuleCallEntry(
                fqn=entry.fqn,
                signature=(
                    self.deserialize_module_call_signature(entry.signature)
                    if entry.signature
                    else None
                ),
            )
            for entry in module_call_graph
        ]


@final
class ExportedProgramDeserializer(metaclass=Final):
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
    ) -> Dict[sympy.Symbol, ValueRanges]:
        range_constraints = {}
        for k, v in symbol_name_to_range.items():
            if symbol := symbol_name_to_symbol.get(k):
                range_constraints[symbol] = v  # type: ignore[arg-type]
            else:
                log.warning(f"Symbol {k} did not appear in the graph that was deserialized")  # noqa: G004
        return range_constraints

    def deserialize(
        self,
        exported_program: ExportedProgram,
        state_dict: Union[Dict[str, torch.Tensor], bytes],
        constants: Union[Dict[str, torch.Tensor], bytes],
        example_inputs: Optional[Union[Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]], bytes]] = None,
    ) -> ep.ExportedProgram:
        assert isinstance(exported_program, ExportedProgram)
        version = exported_program.schema_version

        # TODO(zhxchen17) blocked on thrift schema refactor
        if version.major != SCHEMA_VERSION[0] and not (version.major == 0 and version.minor == 0):
            raise SerializeError(
                f"Serialized schema version {exported_program.schema_version} "
                f"does not match our current schema version {SCHEMA_VERSION}."
            )

        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(
                _int_to_sympy_int(v.min_val), _int_to_sympy_int(v.max_val)
            )
            for k, v in exported_program.range_constraints.items()
        }
        res = (
            GraphModuleDeserializer()
            .deserialize(
                exported_program.graph_module,
                state_dict,
                constants,
                example_inputs,
                symbol_name_to_range,
            )
        )
        range_constraints = self.deserialize_range_constraints(
            symbol_name_to_range,
            res.names_to_symbols,
        )
        model_opset_version: Optional[Dict[str, int]] = exported_program.opset_version

        return ep.ExportedProgram(
            root=res.graph_module,
            graph=res.graph_module.graph,
            graph_signature=res.signature,
            state_dict=res.state_dict,  # type: ignore[arg-type]
            range_constraints=range_constraints,
            module_call_graph=res.module_call_graph,
            example_inputs=res.example_inputs,
            verifier=load_verifier(exported_program.dialect),
            constants=res.constants,
        )


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        return super().default(obj)


def _dataclass_to_dict(obj):
    if isinstance(obj, _Union):
        return {obj.type: _dataclass_to_dict(obj.value)}
    elif dataclasses.is_dataclass(obj):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not (f.default is None and getattr(obj, f.name) is None)
        }
    elif isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def serialize(
    exported_program: ep.ExportedProgram,
    opset_version: Optional[Dict[str, int]] = None,
) -> SerializedArtifact:
    serialized_program = ExportedProgramSerializer(opset_version).serialize(
        exported_program
    )
    assert isinstance(serialized_program.exported_program, ExportedProgram)

    json_program = json.dumps(
        _dataclass_to_dict(serialized_program.exported_program), cls=EnumEncoder
    )
    json_bytes = json_program.encode("utf-8")
    artifact = SerializedArtifact(
        json_bytes,
        serialized_program.state_dict,
        serialized_program.constants,
        serialized_program.example_inputs
    )
    return artifact


def _dict_to_dataclass(cls, data):
    assert not isinstance(cls, str), f"Unresolved class type: '{cls}'."
    if typing.get_origin(cls) == typing.Union and type(None) in typing.get_args(cls):
        if data is None:
            return None
        ty_args = typing.get_args(cls)
        assert len(ty_args) == 2
        return _dict_to_dataclass(ty_args[0], data)
    elif isinstance(cls, type) and issubclass(cls, _Union):
        assert isinstance(data, dict)
        assert len(data) == 1
        _type = next(iter(data.keys()))
        _value = next(iter(data.values()))
        assert isinstance(_type, str)
        field_type = cls.__annotations__[_type]
        return cls.create(**{_type: _dict_to_dataclass(field_type, _value)})
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
        return [_dict_to_dataclass(d_type, d) for d in data]
    elif isinstance(data, dict):
        v_type = typing.get_args(cls)[1]
        return {k: _dict_to_dataclass(v_type, v) for k, v in data.items()}
    return data


def deserialize(
    artifact: SerializedArtifact,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ep.ExportedProgram:
    assert isinstance(artifact.exported_program, bytes)
    exported_program_str = artifact.exported_program.decode("utf-8")
    exported_program_dict = json.loads(exported_program_str)
    serialized_exported_program = _dict_to_dataclass(ExportedProgram, exported_program_dict)
    return (
        ExportedProgramDeserializer(expected_opset_version)
        .deserialize(
            serialized_exported_program,
            artifact.state_dict,
            artifact.constants,
            artifact.example_inputs,
        )
    )


def _canonicalize_graph(
    sorted_inputs, sorted_outputs, graph
) -> Tuple[Graph, Dict[str, str]]:
    def _get_argument(a: Argument):
        if a.type == "as_none":
            return None
        elif a.type == "as_tensor":
            return a.as_tensor
        elif a.type == "as_tensors":
            return a.as_tensors
        elif a.type == "as_int":
            return None
        elif a.type == "as_ints":
            return None
        elif a.type == "as_float":
            return None
        elif a.type == "as_floats":
            return None
        elif a.type == "as_string":
            return None
        elif a.type == "as_strings":
            return None
        elif a.type == "as_sym_int":
            return a.as_sym_int
        elif a.type == "as_sym_ints":
            return a.as_sym_ints
        elif a.type == "as_scalar_type":
            return None
        elif a.type == "as_memory_format":
            return None
        elif a.type == "as_layout":
            return None
        elif a.type == "as_device":
            return None
        elif a.type == "as_bool":
            return None
        elif a.type == "as_bools":
            return None
        elif a.type == "as_sym_bool":
            return a.as_sym_bool
        elif a.type == "as_sym_bools":
            return a.as_sym_bools
        elif a.type == "as_graph":
            return None
        elif a.type == "as_optional_tensors":
            return a.as_optional_tensors
        elif a.type == "as_custom_obj":
            return None
        elif a.type == "as_operator":
            return None
        else:
            raise AssertionError(f"Unknown input type to the ExportedProgram: {a}")

    # Stage 1: Reorder named items.
    def for_args(f, a):
        assert isinstance(a, Argument)
        pytree.tree_map(f, _get_argument(a))

    def sort_nodes(nodes):
        @dataclass
        class Edges:
            outs: List[int]
            ins: int

        graph_inputs: Set[str] = set()
        def_table: Dict[str, int] = {}
        edges: Dict[int, Edges] = {}
        candidates: List[Tuple[str, List[Tuple[str, List[int]]], int]] = []
        rank: Dict[str, int] = {}
        ret: List[Node] = []

        def get_name(a) -> Optional[str]:
            if a is None:
                return None
            if isinstance(a, TensorArgument):
                return a.name
            elif isinstance(a, (SymIntArgument, SymBoolArgument)):
                if a.type == "as_name":
                    return a.as_name
                elif a.type in ("as_int", "as_bool"):
                    return None
                else:
                    raise AssertionError(f"Unknown argument type: {a}")
            elif isinstance(a, OptionalTensorArgument):
                if a.type == "as_tensor":
                    return a.as_tensor.name
                elif a.type == "as_none":
                    return None
                else:
                    raise AssertionError(f"Unknown optional tensor type: {a}")
            else:
                raise AssertionError(f"Unknown argument type: {a}")

        for i in sorted_inputs:

            def add_input(a):
                if s := get_name(a):
                    graph_inputs.add(s)

            for_args(add_input, i)

        for idx, node in enumerate(nodes):

            def add_def(a):
                if s := get_name(a):
                    assert s not in def_table
                    def_table[s] = idx

            for o in node.outputs:
                for_args(add_def, o)

            edges[idx] = Edges([], 0)

        for idx, user in enumerate(nodes):

            def add_edge(a):
                if s := get_name(a):
                    if s not in def_table:
                        assert s in graph_inputs
                        return
                    src = def_table[s]
                    edges[src].outs.append(idx)
                    edges[idx].ins += 1

            for i in user.inputs:
                for_args(add_edge, i.arg)

        def add_rank(a):
            if s := get_name(a):
                assert s not in rank
                rank[s] = len(rank)

        def get_rank(a):
            if s := get_name(a):
                return rank[s]
            else:
                return -1

        for i in sorted_inputs:
            for_args(add_rank, i)

        def add_candidate(idx: int):
            def get_ranks(i):
                ranks = []
                for_args(lambda x: ranks.append(get_rank(x)), i)
                return ranks

            node = nodes[idx]
            args_rank = [(a.name, get_ranks(a.arg)) for a in node.inputs]
            heapq.heappush(candidates, (node.target, args_rank, idx))

        for idx, e in edges.items():
            if e.ins == 0:
                add_candidate(idx)

        while len(candidates) > 0:
            _, _, idx = heapq.heappop(candidates)
            node = nodes[idx]
            for o in node.outputs:
                for_args(add_rank, o)
            ret.append(node)
            assert idx in edges
            for user in edges[idx].outs:
                e = edges[user]
                assert e.ins > 0
                e.ins -= 1
                if e.ins == 0:
                    add_candidate(user)
            edges[idx].outs.clear()

        return ret

    sorted_nodes = sort_nodes(graph.nodes)
    assert len(sorted_nodes) == len(graph.nodes)

    # Stage 2: Rename nodes.
    name_table: Dict[str, str] = {}

    def rename_def(a):
        def _rename(arg_name, values):
            new_name = f"_{len(name_table)}"
            assert arg_name not in name_table
            name_table[arg_name] = new_name
            assert arg_name in values
            values[new_name] = values.pop(arg_name)
            return new_name

        if a is None:
            return
        if isinstance(a, TensorArgument):
            a.name = _rename(a.name, graph.tensor_values)
        elif isinstance(a, SymIntArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_int_values)
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_bool_values)
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    def replace_use(a):
        if a is None:
            return
        if isinstance(a, TensorArgument):
            a.name = name_table.get(a.name, a.name)
        elif isinstance(a, SymIntArgument):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        elif isinstance(a, OptionalTensorArgument):
            if a.type == "as_tensor":
                a.as_tensor.name = name_table.get(a.as_tensor.name, a.as_tensor.name)
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    for i in sorted_inputs:
        for_args(rename_def, i)

    for n in sorted_nodes:
        for o in n.outputs:
            for_args(rename_def, o)

    for n in sorted_nodes:
        for i in n.inputs:
            for_args(replace_use, i.arg)

    for o in sorted_outputs:
        for_args(replace_use, o)

    # Stage 3: Remove unstable fields.
    for n in sorted_nodes:
        n.metadata.clear()

    # Stage 4: Aggregate values.
    sorted_tensor_values = dict(sorted(graph.tensor_values.items(), key=operator.itemgetter(0)))
    sorted_sym_int_values = dict(
        sorted(graph.sym_int_values.items(), key=operator.itemgetter(0))
    )
    sorted_sym_bool_values = dict(
        sorted(graph.sym_bool_values.items(), key=operator.itemgetter(0))
    )

    # Stage 5: Recurse in subgraphs.
    counter = 0
    for node in sorted_nodes:
        for i in node.inputs:
            a = i.arg
            if a.type == "as_graph":
                a.as_graph.graph = _canonicalize_graph(
                    a.as_graph.graph.inputs, a.as_graph.graph.outputs, a.as_graph.graph
                )
                a.as_graph.name = f"_g{counter}"
                counter += 1

    graph = Graph(
        inputs=sorted_inputs,
        outputs=sorted_outputs,
        nodes=sorted_nodes,
        tensor_values=sorted_tensor_values,
        sym_int_values=sorted_sym_int_values,
        sym_bool_values=sorted_sym_bool_values,
        is_single_tensor_return=graph.is_single_tensor_return,
    )
    return graph, name_table


def canonicalize(ep: ExportedProgram) -> ExportedProgram:
    """
    Normalize a serialized ExportedProgram, so that different eager program which
    shares the same semantics can get a single representation on disk.

    This function canonicalizes an ExportedProgram by:

    1. Sorting nodes in topological order.
    2. Rename nodes to have unique names.
    3. Remove unstable fields.
    4. Aggregate the above program fields.
    5. Recurse in subgraphs.

    Args:
        ep (ExportedProgram): The ExportedProgram to canonicalize.

    Returns:
        ExportedProgram: The canonicalized exported program.
    """
    ep = copy.deepcopy(ep)

    opset_version = dict(sorted(ep.opset_version.items(), key=operator.itemgetter(0)))
    range_constraints = dict(sorted(ep.range_constraints.items(), key=operator.itemgetter(0)))
    module_call_graph = sorted(ep.graph_module.module_call_graph, key=lambda x: x.fqn)
    signature = ep.graph_module.signature
    graph = ep.graph_module.graph

    assert len(graph.inputs) == len(signature.input_specs)
    assert len(graph.outputs) == len(signature.output_specs)

    def rank_input(inp) -> Tuple[int, Optional[str], int]:
        idx, (arg, spec) = inp
        assert isinstance(spec, InputSpec)
        if spec.type == "user_input":
            return 5, None, idx
        elif spec.type == "parameter":
            return 1, spec.parameter.parameter_name, idx
        elif spec.type == "buffer":
            return 2, spec.buffer.buffer_name, idx
        elif spec.type == "tensor_constant":
            return 3, spec.tensor_constant.tensor_constant_name, idx
        elif spec.type == "custom_obj":
            return 4, spec.custom_obj.custom_obj_name, idx
        elif spec.type == "token":
            return 0, None, idx
        elif spec.type == "constant_input":
            return 6, spec.constant_input.name, idx
        else:
            raise AssertionError(f"Unknown input type: {spec}")

    def rank_output(out) -> Tuple[int, Optional[str], int]:
        idx, (arg, spec) = out
        assert isinstance(spec, OutputSpec)
        if spec.type == "user_output":
            return 3, None, idx
        elif spec.type == "loss_output":
            return 3, None, idx
        elif spec.type == "buffer_mutation":
            return 1, spec.buffer_mutation.buffer_name, idx
        elif spec.type == "gradient_to_parameter":
            return 4, spec.gradient_to_parameter.parameter_name, idx
        elif spec.type == "gradient_to_user_input":
            return 5, None, idx
        elif spec.type == "user_input_mutation":
            return 2, None, idx
        elif spec.type == "token":
            return 0, None, idx
        else:
            raise AssertionError(f"Unknown output type: {spec}")

    sorted_ins = sorted(
        enumerate(zip(graph.inputs, signature.input_specs)), key=rank_input
    )
    sorted_inputs, input_specs = zip(*(i for idx, i in sorted_ins))  # type: ignore[assignment]

    sorted_outs = sorted(
        enumerate(zip(graph.outputs, signature.output_specs)), key=rank_output
    )
    sorted_outputs, output_specs = zip(*(i for idx, i in sorted_outs))  # type: ignore[assignment]

    sorted_graph, replace_table = _canonicalize_graph(
        sorted_inputs, sorted_outputs, graph
    )

    def replace_input(inp):
        assert isinstance(spec, InputSpec)
        if spec.type == "user_input":
            arg = spec.user_input.arg
            if arg.type == "as_tensor":
                t = arg.as_tensor
                t.name = replace_table[t.name]
            elif arg.type == "as_sym_int":
                s = arg.as_sym_int
                if s.type == "as_name":
                    s.as_name = replace_table[s.as_name]
                elif s.type == "as_int":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_int type: {s}")
            elif arg.type in (
                "as_none",
                "as_bool",
                "as_int",
                "as_float",
                "as_string",
                "as_custom_obj",
            ):
                return
            else:
                raise AssertionError(f"Unknown input type: {arg}")
        elif spec.type == "parameter":
            t = spec.parameter.arg
            t.name = replace_table[t.name]
        elif spec.type == "buffer":
            t = spec.buffer.arg
            t.name = replace_table[t.name]
        elif spec.type == "tensor_constant":
            t = spec.tensor_constant.arg
            t.name = replace_table[t.name]
        elif spec.type == "custom_obj":
            return
        elif spec.type == "token":
            tok = spec.token.arg
            tok.name = replace_table[tok.name]
        elif spec.type == "constant_input":
            return
        else:
            raise AssertionError(f"Unknown input type: {spec}")

    def replace_output(out):
        assert isinstance(spec, OutputSpec)
        if spec.type == "user_output":
            arg = spec.user_output.arg
            if arg.type == "as_tensor":
                t = arg.as_tensor
                t.name = replace_table[t.name]
            elif arg.type == "as_sym_int":
                s = arg.as_sym_int
                if s.type == "as_name":
                    s.as_name = replace_table[s.as_name]
                elif s.type == "as_int":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_int type: {s}")
            elif arg.type in ("as_none", "as_int", "as_float", "as_string"):
                return
            else:
                raise AssertionError(f"Unknown input type: {arg}")
        elif spec.type == "loss_output":
            t = spec.loss_output.arg
            t.name = replace_table[t.name]
        elif spec.type == "buffer_mutation":
            t = spec.buffer_mutation.arg
            t.name = replace_table[t.name]
        elif spec.type == "gradient_to_parameter":
            t = spec.gradient_to_parameter.arg
            t.name = replace_table[t.name]
        elif spec.type == "gradient_to_user_input":
            g = spec.gradient_to_user_input
            g.arg.name = replace_table[g.arg.name]
            g.user_input_name = replace_table[g.user_input_name]
        elif spec.type == "user_input_mutation":
            u = spec.user_input_mutation
            u.arg.name = replace_table[u.arg.name]
            u.user_input_name = replace_table[u.user_input_name]
        elif spec.type == "token":
            tok = spec.token.arg
            tok.name = replace_table[tok.name]
        else:
            raise AssertionError(f"Unknown output type: {spec}")

    for spec in input_specs:
        replace_input(spec)

    for spec in output_specs:
        replace_output(spec)

    return ExportedProgram(
        graph_module=GraphModule(
            graph=sorted_graph,
            signature=GraphSignature(
                input_specs=list(input_specs),
                output_specs=list(output_specs),
            ),
            module_call_graph=module_call_graph,
        ),
        opset_version=opset_version,
        range_constraints=range_constraints,
        schema_version=ep.schema_version,
        dialect=ep.dialect
    )


class CustomOpHandler:
    """
    Base class for handling custom operators.
    """
    @classmethod
    def namespace(cls):
        raise NotImplementedError(f"{cls.__class__} namespace() must be implemented")

    @classmethod
    def op_name(cls, op_type):
        raise NotImplementedError(f"{cls.__class__} op_name() must be implemented")

    @classmethod
    def op_type(cls, op_name):
        raise NotImplementedError(f"{cls.__class__} op_type() must be implemented")

    @classmethod
    def op_schema(cls, op_type):
        raise NotImplementedError(f"{cls.__class__} op_schema() must be implemented")


def register_custom_op_handler(
    op_handler: CustomOpHandler,
    op_type: Type[Any],
):
    """Register custom de/serialization method for a node."""
    assert isinstance(op_handler, CustomOpHandler), f"Expected CustomOpHandler, got {type(op_handler)}."
    _serialization_registry[op_type] = op_handler
    # FIXME: handles deserialization later.
    _deserialization_registry[op_handler.namespace()] = op_handler


def allowed_registered_op_types():
    return tuple(
        _serialization_registry.keys()
    )


# Registry to store all custom serialization implementations.
# The registry maps a operation to its serialization function (a callable), in their own
# namespace to avoid conflicts.
# Serialization: Op type --> custom handler.
# De-serialization: Namespace --> custom handler.
_serialization_registry: Dict[Type[Any], CustomOpHandler] = {}
_deserialization_registry: Dict[str, CustomOpHandler] = {}
