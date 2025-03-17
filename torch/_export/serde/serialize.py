# mypy: allow-untyped-defs
import base64
import copy
import copyreg
import dataclasses
import heapq
import inspect
import io
import json
import keyword
import logging
import math
import operator
import traceback
import typing

from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    cast,
    final,
    Optional,
    Union,
)
from collections.abc import Iterator

import sympy

import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._export.non_strict_utils import _enable_graph_inputs_of_type_nn_module
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils import _pytree as pytree
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.symbol import prefix_str, SymT
from torch.utils._sympy.value_ranges import ValueRanges

from ..utils import remove_proxy_from_state_dict

from .schema import (  # type: ignore[attr-defined]
    Argument,
    ArgumentKind,
    BufferMutationSpec,
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
    InputToConstantInputSpec,
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
    NamedTupleDef,
    Node,
    OptionalTensorArgument,
    OutputSpec,
    OutputTokenSpec,
    RangeConstraint,
    ScalarType,
    SCHEMA_VERSION,
    SchemaVersion,
    SymBool,
    SymBoolArgument,
    SymExpr,
    SymExprHint,
    SymFloat,
    SymFloatArgument,
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


def _reverse_map(d: dict[Any, Enum]):
    return {v.value: k for k, v in d.items()}


MetaType = Union[
    FakeTensor, int, torch.SymInt, float, torch.SymFloat, bool, torch.SymBool, ep.CustomObjArgument
]

DEFAULT_PICKLE_PROTOCOL = 2

ST_DELIMITER = ";"

_TORCH_TO_SERIALIZE_DTYPE = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.uint16: ScalarType.UINT16,
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
    torch.float8_e4m3fn: ScalarType.FLOAT8E4M3FN,
    torch.float8_e5m2: ScalarType.FLOAT8E5M2,
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

_SYM_OPS = {
    operator.eq,
    operator.ne,
    operator.le,
    operator.ge,
    operator.lt,
    operator.gt,
    operator.neg,
    operator.pos,
    math.trunc,
    torch.sym_not,
    operator.mul,
    operator.add,
    operator.sub,
    operator.floordiv,
    operator.mod,
    operator.pow,
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_sqrt,
    operator.truediv,
    operator.and_,
}


assert not any(isinstance(op, torch._ops.OpOverload) for op in _SYM_OPS)

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


def _print_sympy(s: Union[torch.SymInt, torch.SymBool, torch.SymFloat, sympy.Expr]):
    if isinstance(s, (torch.SymInt, torch.SymBool, torch.SymFloat)):
        s = s.node.expr
    return sympy.printing.repr.srepr(s)


def serialize_sym_int(s: Union[int, torch.SymInt]) -> SymInt:
    if isinstance(s, (torch.SymInt, sympy.Symbol, int)):
        if symbolic_shapes.is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            assert isinstance(s, (torch.SymInt, sympy.Symbol))
            if s.node.hint is None:
                return SymInt.create(as_expr=SymExpr(_print_sympy(s)))
            else:
                return SymInt.create(
                    as_expr=SymExpr(
                        _print_sympy(s),
                        hint=SymExprHint.create(as_int=s.node.hint),
                    )
                )
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )

def serialize_sym_float(s: Union[float, torch.SymFloat]) -> SymFloat:
    if isinstance(s, (torch.SymFloat, sympy.Symbol, float)):
        if symbolic_shapes.is_concrete_float(s):
            return SymFloat.create(as_float=float(s))
        else:
            assert isinstance(s, (torch.SymFloat, sympy.Symbol))
            if s.node.hint is None:
                return SymFloat.create(as_expr=SymExpr(_print_sympy(s)))
            else:
                return SymFloat.create(
                    as_expr=SymExpr(
                        _print_sympy(s),
                        hint=SymExprHint.create(as_float=s.node.hint),
                    )
                )
    else:
        raise SerializeError(
            f"SymFloat should be either symbol or float, got `{s}` of type `{type(s)}`"
        )

def serialize_sym_bool(s: Union[bool, torch.SymBool]) -> SymBool:
    if isinstance(s, (torch.SymBool, bool)):
        if symbolic_shapes.is_concrete_bool(s):
            return SymBool.create(as_bool=bool(s))
        else:
            return SymBool.create(
                as_expr=SymExpr(expr_str=_print_sympy(s))
            )
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


def serialize_torch_artifact(artifact: Optional[Any], pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL) -> bytes:
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
        torch.save(artifact, buffer, pickle_protocol=pickle_protocol)
        return buffer.getvalue()
    finally:
        del copyreg.dispatch_table[FakeTensor]


def deserialize_torch_artifact(serialized: Union[dict[str, Any], tuple[Any, ...], bytes]):
    if isinstance(serialized, (dict, tuple)):
        return serialized
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    # weights_only=False as we want to load custom objects here (e.g. ScriptObject)
    artifact = torch.load(buffer, weights_only=False)
    assert isinstance(artifact, (tuple, dict))
    return artifact


def _sympy_int_to_int(val: sympy.Expr, adjust: str) -> Optional[int]:
    # Convert simple sympy Integers into concrete int
    if val in (sympy.oo, int_oo):
        return None
    if val in (-sympy.oo, -int_oo):
        return None
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


def _int_to_sympy_int(val: Optional[int], default) -> sympy.Expr:
    # Convert concrete int into simple sympy Integers
    if val is None:
        return default
    if val == math.inf:
        return int_oo
    if val == -math.inf:
        return -int_oo
    return sympy.Integer(val)


def _symbol_index(sym: sympy.Symbol, sym_type: SymT):
    return int(str(sym)[len(prefix_str[sym_type]):])


def serialize_range_constraints(
    range_constraints: dict[sympy.Symbol, ValueRanges]
) -> dict[str, RangeConstraint]:
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
        return _serialization_registry[type(target)].op_schema(target)
    raise RuntimeError(f"Cannot find schema for {type(target)}")






@dataclass
class GraphState:
    inputs: list[Argument] = field(default_factory=list)
    outputs: list[Argument] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    tensor_values: dict[str, TensorMeta] = field(default_factory=dict)
    sym_int_values: dict[str, SymInt] = field(default_factory=dict)
    sym_bool_values: dict[str, SymBool] = field(default_factory=dict)
    sym_float_values: dict[str, SymFloat] = field(default_factory=dict)
    is_single_tensor_return: bool = False
    custom_obj_values: dict[str, CustomObjArgument] = field(default_factory=dict)


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
        module_call_graph: list[ep.ModuleCallEntry],
    ):
        self.graph_state = GraphState()
        self.graph_signature = graph_signature
        self.module_call_graph = module_call_graph
        self.custom_objs: dict[str, torch._C.ScriptObject] = {}
        self.duplicate_getitem_nodes: dict[str, str] = {}
        self.treespec_namedtuple_fields: dict[str, NamedTupleDef] = {}

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
        val = node.meta["val"]
        log.debug("[handle_placeholder] %s: %s", node.name, val)
        if isinstance(val, torch.Tensor):
            graph_input = Argument.create(as_tensor=self.serialize_tensor_output(node.name, val))
        elif isinstance(val, torch.SymInt):
            graph_input = Argument.create(as_sym_int=self.serialize_sym_int_output(node.name, val))
        elif isinstance(val, torch.SymFloat):
            raise AssertionError("SymFloat graph input is not implemented yet.")
        elif isinstance(val, (int, bool, str, float, type(None))):
            graph_input = self.serialize_input(val)
        elif isinstance(val, ep.CustomObjArgument):
            class_fqn = val.class_fqn
            graph_input = Argument.create(
                as_custom_obj=CustomObjArgument(name=node.name, class_fqn=class_fqn)
            )
            self.graph_state.custom_obj_values[node.name] = (
                self.serialize_script_obj_meta(val)
            )
        else:
            raise AssertionError(f"Unimplemented graph input type: {node.meta['val']}")
        self.graph_state.inputs.append(graph_input)

    def handle_output(self, node: torch.fx.Node):
        assert node.op == "output"
        assert len(node.args) == 1, "FX.Node's args should have one arg"
        node_args = node.args[0]
        log.debug("[handle_output] %s: %s", node.name, node_args)
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
        meta_val = node.meta.get("val")
        log.debug("[handle_call_function] %s: %s(%s, {%s}) -> %s", node.name, node.target, node.args, node.kwargs, meta_val)

        # getitem has been handled in the producer node, skip it here
        if node.target is operator.getitem:
            return

        if (
            node.target in _SYM_OPS
            or (meta_val is not None and isinstance(meta_val, (torch.SymInt, torch.SymBool, torch.SymFloat)))
        ):
            assert len(node.kwargs) == 0
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_sym_op_inputs(node.target, node.args),
                outputs=[self.serialize_output(node.name, meta_val)],
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
            def _is_hop_single_tensor_return(node) -> bool:
                assert isinstance(node.target, torch._ops.HigherOrderOperator)
                # HOP schema is not always available, so we look at node.meta["val"]
                meta_val = node.meta.get("val", None)
                return meta_val is not None and isinstance(meta_val, torch.Tensor)

            # Special handle serialization for aoti_call_delegate
            if node.target is torch._higher_order_ops.aoti_call_delegate:
                serializable_args = list(node.args)

                # AOTI lowered module is not serializable, serialize the aoti_path instead
                lowered_module_name: str = node.args[0].name  # type: ignore[assignment, no-untyped-def, union-attr]
                assert hasattr(node.graph.owning_module, lowered_module_name)
                lowered_module = getattr(node.graph.owning_module, lowered_module_name)  # type: ignore[no-untyped-def]
                serializable_args[0] = lowered_module.aoti_path

                # AOTI compiled graph module in node.args[0] is stateful, and will fail the verifier check
                # Skip serializing original_gm as a workaround
                serializable_args[1] = None

                def serialize_tensor_list_output(node):
                    meta_val = node.meta.get("val", None)
                    tensor_args = []
                    for idx, meta in enumerate(meta_val):
                        name = self._output_node_name_at_index(node, idx)
                        tensor_args.append(self.serialize_tensor_output(name, meta))
                    return [Argument.create(as_tensors=tensor_args)]


                ex_node = Node(
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(serializable_args, node.kwargs),
                    outputs=serialize_tensor_list_output(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=False,
                )
            else:
                ex_node = Node(
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(node.args, node.kwargs),
                    outputs=self.serialize_hoo_outputs(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=_is_hop_single_tensor_return(node),
                )
        elif type(node.target) in _serialization_registry:
            # Sanity check for unhandled serialization.
            assert type(node.target) in _serialization_registry, f"{type(node.target)} is not supported in export serialization."

            handler = _serialization_registry[type(node.target)]
            namespace = handler.namespace()
            op_name = handler.to_op_name(node.target)
            assert isinstance(namespace, str) and isinstance(op_name, str)
            assert ":" not in namespace and ":" not in op_name
            ex_node = Node(
                target=f"#{namespace}:{op_name}",
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                metadata=self.serialize_metadata(node),
            )
        else:
            raise SerializeError(f"Serializing {node.target} is not supported")

        self.graph_state.nodes.append(ex_node)

    def handle_get_attr(self, node):
        log.debug("[handle_get_attr] %s", node.name)

    def _output_node_at_index(self, node, index) -> Optional[torch.fx.Node]:
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

    def _output_node_name_at_index(self, node, index) -> str:
        user_node = self._output_node_at_index(node, index)
        if user_node is None:
            return f"{node.name}_unused_{index}"
        else:
            return user_node.name

    def serialize_metadata(self, node: torch.fx.Node) -> dict[str, str]:
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

        if custom := node.meta.get("custom"):
            try:
                ret["custom"] = json.dumps(custom)
            except Exception as e:
                raise SerializeError(
                    f"Failed to serialize custom metadata for node {node.name} with error {e}"
                ) from e

        return ret

    def serialize_script_obj_meta(
        self, script_obj_meta: ep.CustomObjArgument
    ) -> CustomObjArgument:
        log.debug("[serialize_script_obj_meta] %s", script_obj_meta)
        return CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    def serialize_sym_op_inputs(self, op, args) -> list[NamedArgument]:
        if isinstance(op, torch._ops.OpOverload):
            args_names = [arg.name for arg in op._schema.arguments]
        else:
            assert op in _SYM_OPS
            args_names = list(inspect.signature(op).parameters.keys())
        serialized_args = []
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(
                    name=args_name,
                    arg=self.serialize_input(arg),
                    kind=ArgumentKind.POSITIONAL,
                )
            )
        return serialized_args

    def serialize_inputs(
        self,
        target: Any,  # torch._ops.OpOverload and other custom operator types.
        args,
        kwargs=None
    ) -> list[NamedArgument]:
        schema = None
        serialized_args = []

        if isinstance(target, torch._higher_order_ops.torchbind.CallTorchBind):
            obj = args[0]
            method = args[1]
            schema = target.schema(obj, method)
        else:
            assert isinstance(target, (torch._ops.OpOverload, *_registered_extension_types()))
            schema = _get_schema_from_target(target)
        assert schema is not None
        kwargs = kwargs or {}

        for i, schema_arg in enumerate(schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(kwargs[schema_arg.name], schema_arg.type),
                        kind=ArgumentKind.KEYWORD,
                    )
                )
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(args[i], schema_arg.type),
                        kind=ArgumentKind.POSITIONAL,
                    )
                )
            else:
                # We intentionally don't serialize the missing arguments
                # with default values
                pass

        return serialized_args

    def serialize_hoo_inputs(self, args, kwargs) -> list[NamedArgument]:
        """
        For serializing HOO inputs since HOOs do not have a schema.
        """
        inputs = [
            NamedArgument(
                name="",
                arg=self.serialize_input(a),
                kind=ArgumentKind.POSITIONAL
            )
            for a in args
        ]
        inputs.extend(
            [
                NamedArgument(
                    name=name,
                    arg=self.serialize_input(a),
                    kind=ArgumentKind.KEYWORD,
                )
                for name, a in kwargs.items()
            ]
        )
        return inputs

    def is_inductor_sym_int_arg(self, arg) -> bool:
        # This is a special branch for handling SymInt args in inductor's
        # ExternalFallbackNode.
        # For regular FX graph, SymInt arg should be a fx.Node and should be
        # verified with is_sym_int_arg()
        return type(arg) is int or isinstance(arg, torch.SymInt)

    def is_sym_int_arg(self, arg) -> bool:
        return type(arg) is int or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_int_values
        )

    def is_sym_float_arg(self, arg) -> bool:
        return isinstance(arg, float) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_float_values
        )

    def is_sym_bool_arg(self, arg) -> bool:
        return isinstance(arg, bool) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_bool_values
        )

    # should be torch._C.JitType but that annotation is busted
    def serialize_input(
        self, arg, arg_type: Optional[Any] = None
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
            elif self.is_sym_float_arg(arg):
                return Argument.create(
                    as_sym_float=SymFloatArgument.create(as_name=arg.name)
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
        elif isinstance(arg, inductor_ir.TorchBindObject):
            # This is a special branch for handling TorchBindObject
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            arg_name = arg.get_name()
            assert arg_name is not None, "Buffer must have valid name"
            arg_val = arg.get_real_obj()
            class_fqn = arg_val._type().qualified_name()
            self.custom_objs[arg_name] = arg_val
            return Argument.create(
                as_custom_obj=CustomObjArgument(arg_name, class_fqn)
            )
        elif isinstance(arg, torch.SymInt):
            # This is a special branch for handling SymInt args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_int_arg(arg) being true
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=str(arg)))
        elif isinstance(arg, torch.SymFloat):
            # This is a special branch for handling SymFloat args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_float_arg(arg) being true
            return Argument.create(as_sym_float=SymFloatArgument.create(as_name=str(arg)))
        elif type(arg) is bool:
            return Argument.create(as_bool=arg)
        elif type(arg) is str:
            return Argument.create(as_string=arg)
        elif type(arg) is int:
            return Argument.create(as_int=arg)
        elif type(arg) is float:
            return Argument.create(as_float=arg)
        elif arg is None:
            return Argument.create(as_none=True)
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

            if all(type(a) is bool for a in arg):
                return Argument.create(as_bools=list(arg))
            elif all(type(a) is int for a in arg):
                return Argument.create(as_ints=list(arg))
            elif all(type(a) is float for a in arg):
                return Argument.create(as_floats=list(arg))
            elif all(type(a) is str for a in arg):
                return Argument.create(as_strings=list(arg))
            elif all(self.is_inductor_sym_int_arg(a) for a in arg):
                # This is a special branch for handling SymInt args in inductor's
                # ExternalFallbackNode.
                # For regular FX graph, SymInt arg should be a fx.Node
                values = []
                for a in arg:
                    if isinstance(a, torch.SymInt):
                        values.append(SymIntArgument.create(as_name=str(a)))
                    elif type(a) is int:
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(isinstance(a, torch.SymFloat) for a in arg):
                return Argument.create(
                    as_sym_floats=[SymFloatArgument.create(as_name=str(a)) for a in arg]
                )
            elif all(self.is_sym_int_arg(a) for a in arg):
                # list of sym_ints
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymIntArgument.create(as_name=a.name))
                    elif type(a) is int:
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(self.is_sym_float_arg(a) for a in arg):
                # list of sym_float
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymFloatArgument.create(as_name=a.name))
                    elif isinstance(a, float):
                        values.append(SymFloatArgument.create(as_float=a))
                return Argument.create(as_sym_floats=values)
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
                        return OptionalTensorArgument.create(as_none=True)
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
                        return OptionalTensorArgument.create(as_none=True)
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
        elif isinstance(arg, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            return Argument.create(as_operator=self.serialize_operator(arg))
        else:
            raise SerializeError(f"Unsupported argument type: {type(arg)} with schema arg_type {arg_type}")

    def serialize_tensor_output(self, name, meta_val) -> TensorArgument:
        assert name not in self.graph_state.tensor_values
        self.graph_state.tensor_values[name] = serialize_tensor_meta(meta_val)
        return TensorArgument(name=name)

    def serialize_sym_int_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_int_values
        self.graph_state.sym_int_values[name] = serialize_sym_int(meta_val)
        return SymIntArgument.create(as_name=name)

    def serialize_sym_float_output(self, name, meta_val) -> SymFloatArgument:
        assert name not in self.graph_state.sym_float_values
        self.graph_state.sym_float_values[name] = serialize_sym_float(meta_val)
        return SymFloatArgument.create(as_name=name)

    def serialize_sym_bool_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_bool_values
        self.graph_state.sym_bool_values[name] = serialize_sym_bool(meta_val)
        return SymBoolArgument.create(as_name=name)

    def serialize_input_spec(self, spec: ep.InputSpec) -> InputSpec:
        log.debug("[serialize_input_spec] %s", spec)
        if spec.kind == ep.InputKind.USER_INPUT:
            if isinstance(spec.arg, ep.ConstantArgument):
                if type(spec.arg.value) is int:
                    constant_spec = ConstantValue.create(as_int=spec.arg.value)
                elif type(spec.arg.value) is bool:
                    constant_spec = ConstantValue.create(as_bool=spec.arg.value)
                elif type(spec.arg.value) is str:
                    constant_spec = ConstantValue.create(as_string=spec.arg.value)
                elif type(spec.arg.value) is float:
                    constant_spec = ConstantValue.create(as_float=spec.arg.value)
                elif spec.arg.value is None:
                    constant_spec = ConstantValue.create(as_none=True)
                else:
                    raise SerializeError(f"Unhandled constant input {spec.arg.value} to serialize")
                return InputSpec.create(
                    constant_input=InputToConstantInputSpec(
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
        log.debug("[serialize_output_spec] %s", spec)
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
        log.debug("\n[serialize_signature]")
        return GraphSignature(
            input_specs=[self.serialize_input_spec(s) for s in sig.input_specs],
            output_specs=[self.serialize_output_spec(s) for s in sig.output_specs],
        )

    def serialize_argument_spec(self, x: ep.ArgumentSpec) -> Argument:
        if isinstance(x, ep.TensorArgument):
            return Argument.create(as_tensor=TensorArgument(name=x.name))
        elif isinstance(x, ep.SymIntArgument):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.name))
        elif isinstance(x, ep.SymFloatArgument):
            return Argument.create(as_sym_float=SymFloatArgument.create(as_name=x.name))
        elif isinstance(x, ep.ConstantArgument):
            return self.serialize_input(x.value)
        elif isinstance(x, ep.CustomObjArgument):
            return Argument.create(
                as_custom_obj=CustomObjArgument(name=x.name, class_fqn=x.class_fqn)
            )
        else:
            raise AssertionError("TODO")

    def serialize_treespec(self, treespec):
        # We want to additionally save all the field names of the namedtuples in
        # case users want to check that the treespec types are equivalent
        def store_namedtuple_fields(ts):
            if ts.type is None:
                return
            if ts.type == namedtuple:
                serialized_type_name = pytree.SUPPORTED_SERIALIZED_TYPES[ts.context].serialized_type_name
                if serialized_type_name in self.treespec_namedtuple_fields:
                    field_names = self.treespec_namedtuple_fields[serialized_type_name].field_names
                    if field_names != ts.context._fields:
                        raise SerializeError(
                            f"The given TreeSpec's namedtuple type {ts.context} "
                            f"was found to have field names {ts.context._fields} "
                            f"but somehow previously was found to have field names {field_names}."
                        )
                else:
                    self.treespec_namedtuple_fields[serialized_type_name] = NamedTupleDef(field_names=ts.context._fields)

            for child in ts.children_specs:
                store_namedtuple_fields(child)

        serialized_treespec = treespec_dumps(treespec, TREESPEC_VERSION)
        store_namedtuple_fields(treespec)
        return serialized_treespec

    def serialize_module_call_signature(
        self, module_call_signature: ep.ModuleCallSignature
    ) -> ModuleCallSignature:
        log.debug("[serialize_module_call_signature] %s", module_call_signature)
        return ModuleCallSignature(
            inputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=self.serialize_treespec(module_call_signature.in_spec),
            out_spec=self.serialize_treespec(module_call_signature.out_spec),
            forward_arg_names=names if (names := module_call_signature.forward_arg_names) else None
        )

    def serialize_module_call_graph(
        self, module_call_graph: list[ep.ModuleCallEntry]
    ) -> list[ModuleCallEntry]:
        log.debug("\n[serialize_module_call_graph]")
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

    def serialize_outputs(self, node: torch.fx.Node) -> list[Argument]:
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

        def _is_single_tensor_list_return(target: Any) -> bool:
            schema = _get_schema_from_target(target)
            returns = schema.returns

            if len(returns) != 1:
                return False
            return_type = returns[0].real_type
            return isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            )

        assert node.op == "call_function" and isinstance(node.target, (torch._ops.OpOverload, *_registered_extension_types()))

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
                name = self._output_node_name_at_index(node, idx)
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
                output_arguments.append(Argument.create(as_none=True))
            elif isinstance(meta, FakeTensor):
                assert isinstance(return_schema.real_type, (torch.OptionalType, torch.TensorType))
                name = self._output_node_name_at_index(node, idx)
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
                    sub_user_node_name = self._output_node_name_at_index(user_node, i)
                    args.append(self.serialize_tensor_output(sub_user_node_name, m))
                output_arguments.append(Argument.create(as_tensors=args))
            elif isinstance(meta, (int, SymInt, float, SymFloat)):
                user_node_name = self._output_node_name_at_index(node, idx)
                output_arguments.append(self.serialize_output(user_node_name, meta))
            else:
                raise ValueError(
                    f"Unhandled output type {type(meta)} from node {node.format_node()}"
                )

        return output_arguments

    def serialize_hoo_outputs(self, node: torch.fx.Node) -> list[Argument]:
        """
        For serializing HOO outputs since HOOs do not have a schema.
        """
        meta_val = node.meta["val"]

        if isinstance(meta_val, tuple):
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

                        name = self._output_node_name_at_index(user_node, j)
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
            return Argument.create(as_none=True)
        if isinstance(meta_val, torch.Tensor):
            # e.g "-> Tensor"
            return Argument.create(
                as_tensor=self.serialize_tensor_output(name, meta_val)
            )
        elif isinstance(meta_val, (bool, torch.SymBool)):
            # e.g "-> SymBool"
            return Argument.create(
                as_sym_bool=self.serialize_sym_bool_output(name, meta_val)
            )
        elif isinstance(meta_val, (int, torch.SymInt)):
            # e.g "-> SymInt"
            assert not isinstance(meta_val, bool)
            return Argument.create(
                as_sym_int=self.serialize_sym_int_output(name, meta_val)
            )
        elif isinstance(meta_val, (float, torch.SymFloat)):
            # e.g "-> SymFloat"
            return Argument.create(
                as_sym_float=self.serialize_sym_float_output(name, meta_val)
            )

        # list outputs should've been handled earlier
        raise SerializeError(f"Unable to serialize output {meta_val}")

    def _handle_getitem_users(self, node: torch.fx.Node) -> list[TensorArgument]:
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
        log.debug("[serialize_graph]\n\n%s", graph_module.print_readable(print_output=False))

        for node in graph_module.graph.nodes:
            try:
                getattr(self, f"handle_{node.op}")(node)
            except Exception as e:
                raise SerializeError(
                    f"Failed serializing node {node} in graph: {node.format_node()}\n Original exception {traceback.format_exc()}"
                ) from e

        return Graph(
            inputs=self.graph_state.inputs,
            nodes=self.graph_state.nodes,
            tensor_values=self.graph_state.tensor_values,
            sym_int_values=self.graph_state.sym_int_values,
            sym_float_values=self.graph_state.sym_float_values,
            sym_bool_values=self.graph_state.sym_bool_values,
            custom_obj_values=self.graph_state.custom_obj_values,
            outputs=self.graph_state.outputs,
            is_single_tensor_return=self.graph_state.is_single_tensor_return,
        )

    def serialize_graph_module_metadata(self, meta: dict[str, Any]):
        ret = {}
        if custom := meta.get("custom"):
            log.debug("\n[serialize_graph_module_metadata] %s", custom)
            try:
                ret["custom"] = json.dumps(custom)
            except Exception as e:
                raise SerializeError(
                    f"Failed to serialize custom metadata for graph with error {e}"
                ) from e

        return ret

    def serialize(self, graph_module: torch.fx.GraphModule) -> GraphModule:
        log.debug("\n[serialize]")
        graph = self.serialize_graph(graph_module)

        return GraphModule(
            graph=graph,
            signature=self.serialize_signature(self.graph_signature),
            module_call_graph=self.serialize_module_call_graph(self.module_call_graph),
            metadata=self.serialize_graph_module_metadata(graph_module.meta),
            treespec_namedtuple_fields=self.treespec_namedtuple_fields
        )


@final
class ExportedProgramSerializer(metaclass=Final):
    def __init__(self, opset_version: Optional[dict[str, int]] = None, pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL):
        self.opset_version: dict[str, int] = {}
        if opset_version:
            self.opset_version.update(opset_version)
        if "aten" not in self.opset_version:
            self.opset_version["aten"] = torch._C._get_max_operator_version()

        self.pickle_protocol = pickle_protocol

    def serialize(self, exported_program: ep.ExportedProgram) -> _SerializedProgram:
        """
        Args:
            exported_program: Exported Program to serialize
        """
        exported_program.validate()

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
        constants: dict[str, Any] = {}
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
            verifiers=[v.dialect for v in exported_program.verifiers],
            torch_version=torch.__version__,
        )

        # Test canonical form is well defined.
        canonicalize(serialized_ep, set(constants.keys()))

        # Proxy cannot be dumped, so we remove them.
        new_state_dict = remove_proxy_from_state_dict(
            exported_program.state_dict, in_place=False
        )
        return _SerializedProgram(
            serialized_ep,
            serialize_torch_artifact(new_state_dict, self.pickle_protocol),
            serialize_torch_artifact(constants, self.pickle_protocol),
            serialize_torch_artifact(exported_program.example_inputs, self.pickle_protocol),
        )


@final
class GraphModuleDeserializer(metaclass=Final):
    @dataclasses.dataclass
    class Result:
        graph_module: torch.fx.GraphModule
        signature: ep.ExportGraphSignature
        module_call_graph: list[ep.ModuleCallEntry]
        names_to_symbols: dict[str, sympy.Symbol]
        state_dict: dict[str, Union[torch.Tensor, torch.nn.Parameter]]
        constants: dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]]
        example_inputs: Optional[tuple[tuple[torch.Tensor, ...], dict[str, Any]]]

    def __init__(self) -> None:
        self.serialized_name_to_node: dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: dict[str, MetaType] = {}
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

    def deserialize_extension_operator(self, serialized_target: str):
        namespace, op_name = serialized_target.split(":")
        namespace = namespace[1:]  # starting with #
        handler = _deserialization_registry[namespace]
        return handler.from_op_name(op_name)

    def deserialize_operator(self, serialized_target: str):
        if serialized_target.startswith(
            "_operator"
        ):  # TODO(zhxchen17) Follow up on this.
            module = operator
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("torch"):
            module = torch  # type: ignore[misc]
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("#"):
            return self.deserialize_extension_operator(serialized_target)
        else:  # TODO(zhxchen17) Don't catch all here.
            return serialized_target

        target = module
        for name in serialized_target_names:
            if not hasattr(target, name):
                return serialized_target
            else:
                target = getattr(target, name)
        return target

    def _parse_sym_expr(self, expr_str: str, hint: Optional[Union[int, bool, float]] = None) -> sympy.Expr:
        """
        Parses and does bottom-up processing of sympy.Expr nodes,
        populating ShapeEnv & caching symbols as needed.
        """
        def _process_sym_expr(sym: sympy.Expr, hint: Optional[Union[int, bool, float]] = None) -> sympy.Expr:
            if sym.is_Integer or sym.is_Float or sym.is_Boolean:  # base case
                return sym
            else:  # recursive case
                # important to use str(expr) and not _print_sympy(),
                # str(expr) is key for self.symbol_name_to_range
                expr_str = str(sym)
                for arg in sym.args:
                    self._parse_sym_expr(arg)
                # symbol caching
                if expr_str in self.symbol_name_to_symbol:
                    sym = self.symbol_name_to_symbol[expr_str]
                else:
                    self.symbol_name_to_symbol[expr_str] = sym
                    if (
                        isinstance(sym, sympy.Symbol)
                        and symbolic_shapes.symbol_is_type(sym, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
                    ):
                        self.unbacked_symbols.add(sym)
                # hints
                if (
                    hint is not None
                    and sym not in self.shape_env.var_to_val
                ):
                    self.shape_env.add_var_to_val(sym, hint)  # type: ignore[arg-type]
                # ValueRanges
                if vr := self.symbol_name_to_range.get(expr_str):
                    self.shape_env.constrain_symbol_range(
                        sym,
                        compiler_min=vr.lower,  # type: ignore[arg-type]
                        compiler_max=vr.upper,  # type: ignore[arg-type]
                    )
            return sym

        expr = sympy.sympify(
            expr_str,
            locals={**self.sympy_functions, **self.symbol_name_to_symbol},
        )
        return _process_sym_expr(expr, hint)

    def deserialize_sym_int(self, s: SymInt) -> Union[int, torch.SymInt]:
        val = s.value
        if s.type == "as_expr":
            if val.hint is None:
                hint = None
            else:
                assert val.hint.type == "as_int"
                hint = val.hint.value

            sym = self._parse_sym_expr(val.expr_str, hint)
            return self.shape_env.create_symintnode(sym, hint=hint)
        elif s.type == "as_int":
            assert type(val) is int
            return val
        else:
            raise SerializeError(
                f"SymInt has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_sym_float(self, s: SymFloat) -> Union[float, torch.SymFloat]:
        val = s.value
        if s.type == "as_expr":
            hint = val.hint.as_float if val.hint else None
            sym = self._parse_sym_expr(val.expr_str, hint)
            return self.shape_env.create_symfloatnode(sym, hint=hint)
        elif s.type == "as_float":
            assert isinstance(val, float)
            return val
        else:
            raise SerializeError(
                f"SymFloat has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_sym_bool(self, s: SymBool) -> Union[bool, torch.SymBool]:
        val = s.value
        if s.type == "as_expr":
            expr = self._parse_sym_expr(val.expr_str)
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
                    requires_grad=tensor_meta.requires_grad,
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
        elif output.type == "as_sym_float":
            return self.serialized_name_to_node[output.as_sym_float.as_name]
        elif output.type == "as_int":
            return output.as_int
        elif output.type == "as_float":
            return output.as_float
        elif output.type == "as_bool":
            return output.as_bool
        elif output.type == "as_none":
            return None
        else:
            raise SerializeError(f"Unable to deserialize output node {output}")

    def deserialize_graph(self, serialized_graph: Graph) -> torch.fx.Graph:
        log.debug("\n[deserialize_graph]")

        # Handle the tensor metas.
        for name, tensor_value in serialized_graph.tensor_values.items():
            log.debug("[deserialize_tensor_meta] %s (input): %s", name, tensor_value)
            meta_val = self.deserialize_tensor_meta(tensor_value)
            log.debug("[deserialize_tensor_meta] %s (output): %s", name, meta_val)
            self.serialized_name_to_meta[name] = meta_val

        for name, sym_int_value in serialized_graph.sym_int_values.items():
            log.debug("[deserialize_sym_int] %s (input): %s", name, sym_int_value)
            int_val = self.deserialize_sym_int(sym_int_value)
            log.debug("[deserialize_sym_int] %s (output): %s", name, int_val)
            self.serialized_name_to_meta[name] = int_val

        for name, sym_float_value in serialized_graph.sym_float_values.items():
            log.debug("[deserialize_sym_float] %s (input): %s", name, sym_float_value)
            float_val = self.deserialize_sym_float(sym_float_value)
            log.debug("[deserialize_sym_float] %s (output): %s", name, float_val)
            self.serialized_name_to_meta[name] = float_val

        for name, sym_bool_value in serialized_graph.sym_bool_values.items():
            log.debug("[deserialize_sym_bool] %s (input): %s", name, sym_bool_value)
            bool_val = self.deserialize_sym_bool(sym_bool_value)
            log.debug("[deserialize_sym_bool] %s (output): %s", name, bool_val)
            self.serialized_name_to_meta[name] = bool_val

        for name, script_obj_meta in serialized_graph.custom_obj_values.items():
            log.debug("[deserialize_script_obj_meta] %s", script_obj_meta)
            self.serialized_name_to_meta[name] = self.deserialize_script_obj_meta(
                script_obj_meta
            )

        log.debug("\n[deserialize graph nodes]")
        # Inputs: convert to placeholder nodes in FX.
        for i, input_ in enumerate(serialized_graph.inputs):
            log.debug("[deserialize input] %s", input_)
            if input_.type in ("as_tensor", "as_custom_obj"):
                node_name = input_.value.name
                placeholder_node = self.graph.placeholder(node_name)
                # FX might declare a name illegal (e.g. some nn.Modules use "input" as forward() arguments)
                # we will overwrite it
                placeholder_node.name = node_name
                self.sync_fx_node(node_name, placeholder_node)
            elif input_.type == "as_sym_int":
                if input_.value.type == "as_name":
                    node_name = input_.value.as_name
                    placeholder_node = self.graph.placeholder(node_name)
                    # FX might declare a name illegal (e.g. some nn.Modules use "input" as forward() arguments)
                    # we will overwrite it
                    placeholder_node.name = node_name
                    self.sync_fx_node(node_name, placeholder_node)
                else:
                    raise SerializeError(f"Deserializing a constant symint {input_.value} as an input")
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
                    f"Failed deserializing node {serialized_node}\n Original exception {traceback.format_exc()}"
                ) from e

        # Outputs: convert to a single `output` node.
        outputs = []
        for output in serialized_graph.outputs:
            log.debug("[deserialize output] %s", output)
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

        # recompute unbacked bindings
        for node in self.graph.nodes:
            if (
                (val := node.meta.get("val")) is not None
                and (
                    unbacked_bindings := symbolic_shapes._free_unbacked_symbols_with_path(
                        val, (), shape_env=self.shape_env, pending=self.unbacked_symbols, simplify=True
                    )
                )
            ):
                node.meta["unbacked_bindings"] = unbacked_bindings

        assert len(self.unbacked_symbols) == 0
        return self.graph

    def deserialize_node(self, serialized_node: Node, target: Callable) -> None:

        def _is_single_tensor_return(target) -> bool:
            schema = _get_schema_from_target(target)
            returns = schema.returns
            return len(returns) == 1 and isinstance(returns[0].real_type, torch.TensorType)

        if (
            target in _SYM_OPS
            or target == torch.ops.aten.item.default  # this can produce either SymInt or SymBool
        ):
            name = serialized_node.outputs[0].value.as_name
            args = self.deserialize_sym_op_inputs(serialized_node.inputs)

            fx_node = self.graph.create_node("call_function", target, args, {}, name)
            self.deserialize_sym_op_outputs(serialized_node, fx_node)

        elif isinstance(target, torch._ops.HigherOrderOperator):
            args, kwargs = self.deserialize_hoo_inputs(serialized_node.inputs)
            metadata = self.deserialize_metadata(serialized_node.metadata)
            for x in (*args, *kwargs.values()):
                if isinstance(x, torch.fx.Node) and x.op == "get_attr":
                    # this means that we have deserialized a graph argument, but
                    # unfortunately the schema for it does not include metadata;
                    # so we reuse the metadata of the HOP call for such arguments
                    x.meta.update(metadata)
            # If a serialized HOP node has a length=1 outputs of type `as_tensor``.
            # There could be two cases:
            # (1) The HOP node returns a single tensor
            # (2) The HOP node returns a tuple containing a single tensor
            # We distinguish (1) and (2) by the `is_single_tensor_return`
            # field in the schema of Node
            # For BC, getattr() will return True if `is_single_tensor_return` doesn't
            # exist. This is because prior to adding `is_single_tensor_return`,
            # only (1) could happen as we handle (2) with type `as_tensors`
            name = (
                serialized_node.outputs[0].as_tensor.name
                if len(serialized_node.outputs) == 1
                and hasattr(serialized_node.outputs[0], "as_tensor")
                and getattr(serialized_node, "is_hop_single_tensor_return", True)
                else None
            )
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
            fx_node.meta.update(metadata)

        elif isinstance(target, (torch._ops.OpOverload, *_registered_extension_types())):
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
        log.debug(
            "[deserialize_node] %s: %s(%s, {%s}) -> %s",
            fx_node.name,
            fx_node.target,
            fx_node.args,
            fx_node.kwargs,
            fx_node.meta.get("val"),
        )
        if fx_node.op not in ["placeholder", "output"] and "nn_module_stack" not in fx_node.meta:
            fx_node.meta["nn_module_stack"] = {}  # serialization throws away empty dicts

    def deserialize_input_spec(self, i: InputSpec) -> ep.InputSpec:
        log.debug("[deserialize_input_spec] %s", i)
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
        log.debug("[deserialize_output_spec] %s", o)
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
        log.debug("\n[deserialize_signature]")
        return ep.ExportGraphSignature(
            input_specs=[self.deserialize_input_spec(i) for i in sig.input_specs],
            output_specs=[self.deserialize_output_spec(o) for o in sig.output_specs],
        )

    def deserialize(
        self,
        serialized_graph_module: GraphModule,
        serialized_state_dict: Union[dict[str, torch.Tensor], bytes],
        constants: Union[dict[str, Any], bytes],
        example_inputs: Optional[Union[tuple[tuple[torch.Tensor, ...], dict[str, Any]], bytes]] = None,
        symbol_name_to_range: Optional[dict[str, symbolic_shapes.ValueRanges]] = None,
    ) -> Result:
        global _CURRENT_DESERIALIZER
        assert _CURRENT_DESERIALIZER is None
        _CURRENT_DESERIALIZER = self
        try:
            log.debug("\n[deserialize]")
            self.shape_env = symbolic_shapes.ShapeEnv(assume_static_by_default=True)
            self.fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=True,
                shape_env=self.shape_env,
            )
            self.sympy_functions = {
                # all torch.utils._sympy.functions should go here
                # TODO(avik): find a better way to keep this collection in sync;
                # e.g.., `exec('from torch.utils._sympy.functions import *', ...)`
                # would work as long as the public API of that module is complete
                "FloorDiv": torch.utils._sympy.functions.FloorDiv,
                "ModularIndexing": torch.utils._sympy.functions.ModularIndexing,
                "Where": torch.utils._sympy.functions.Where,
                "PythonMod": torch.utils._sympy.functions.PythonMod,
                "Mod": torch.utils._sympy.functions.Mod,
                "CleanDiv": torch.utils._sympy.functions.CleanDiv,
                "CeilToInt": torch.utils._sympy.functions.CeilToInt,
                "FloorToInt": torch.utils._sympy.functions.FloorToInt,
                "CeilDiv": torch.utils._sympy.functions.CeilDiv,
                "LShift": torch.utils._sympy.functions.LShift,
                "RShift": torch.utils._sympy.functions.RShift,
                "PowByNatural": torch.utils._sympy.functions.PowByNatural,
                "FloatPow": torch.utils._sympy.functions.FloatPow,
                "FloatTrueDiv": torch.utils._sympy.functions.FloatTrueDiv,
                "IntTrueDiv": torch.utils._sympy.functions.IntTrueDiv,
                "IsNonOverlappingAndDenseIndicator": torch.utils._sympy.functions.IsNonOverlappingAndDenseIndicator,
                "TruncToFloat": torch.utils._sympy.functions.TruncToFloat,
                "TruncToInt": torch.utils._sympy.functions.TruncToInt,
                "RoundToInt": torch.utils._sympy.functions.RoundToInt,
                "RoundDecimal": torch.utils._sympy.functions.RoundDecimal,
                "ToFloat": torch.utils._sympy.functions.ToFloat,
                "Identity": torch.utils._sympy.functions.Identity,
            }
            self.symbol_name_to_symbol: dict[str, sympy.Symbol] = {}
            self.constants = deserialize_torch_artifact(constants)
            self.signature = self.deserialize_signature(serialized_graph_module.signature)

            # deserialization does analysis with checks on 0/1, so we create fake range constraints and
            # restore the original range constraints afterwards
            self.symbol_name_to_range = {}
            # we also need to bump unbacked sym[float,int] counters in the
            # shape env to accommodate unbacked symbols in the exported program
            self.unbacked_symbols: set[sympy.Symbol] = set()
            count_unbacked_symfloat, count_unbacked_symint = -1, -1
            unbacked_symfloat_prefix, unbacked_symint_prefix = (
                prefix_str[t] for t in [SymT.UNBACKED_FLOAT, SymT.UNBACKED_INT]
            )
            if symbol_name_to_range:
                for k, vr in symbol_name_to_range.items():
                    lower = vr.lower
                    if vr.upper >= 2:  # max is >= 2, not sym bool range
                        lower = max(2, lower)
                    self.symbol_name_to_range[k] = symbolic_shapes.ValueRanges(_int_to_sympy_int(lower, -int_oo), vr.upper)
                    if k.startswith(unbacked_symfloat_prefix):
                        i = int(k[len(unbacked_symfloat_prefix):])
                        count_unbacked_symfloat = max(count_unbacked_symfloat, i)
                    elif k.startswith(unbacked_symint_prefix):
                        i = int(k[len(unbacked_symint_prefix):])
                        count_unbacked_symint = max(count_unbacked_symint, i)

            # TODO(pianpwk): if we can clean up unused symbols in range_constraints,
            # then this logic can just be handled with self.unbacked_symbols alone
            for _ in range(count_unbacked_symfloat + 1):
                next(self.shape_env.unbacked_symfloat_counter)
            for _ in range(count_unbacked_symint + 1):
                next(self.shape_env.unbacked_symint_counter)

            if example_inputs is not None and len(example_inputs) > 0:
                self.example_inputs = deserialize_torch_artifact(example_inputs)
            else:
                self.example_inputs = None
            self.deserialize_graph(serialized_graph_module.graph)

            with _enable_graph_inputs_of_type_nn_module(self.example_inputs):
                module_call_graph = self.deserialize_module_call_graph(
                    serialized_graph_module.module_call_graph
                )
            graph_module = ep._create_graph_module_for_export(
                self.module, self.graph
            )
            meta = {}
            if custom := serialized_graph_module.metadata.get("custom"):
                meta["custom"] = json.loads(custom)
            if hasattr(serialized_graph_module, "treespec_namedtuple_fields"):
                meta["treespec_namedtuple_fields"] = {}
                for type_, fields in serialized_graph_module.treespec_namedtuple_fields.items():
                    meta["treespec_namedtuple_fields"][type_] = fields.field_names
            graph_module.meta = meta
            return GraphModuleDeserializer.Result(
                graph_module=graph_module,
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
        # overwrite name
        fx_node.name = name
        self.serialized_name_to_node[name] = fx_node
        assert "val" not in fx_node.meta
        fx_node.meta["val"] = self.serialized_name_to_meta[name]

    def deserialize_sym_op_inputs(self, inputs):
        return tuple(self.deserialize_input(input.arg) for input in inputs)

    def deserialize_inputs(self, target, serialized_node: Node):
        schema_args = _get_schema_from_target(target).arguments
        argument_kinds = {
            input.name: input.kind
            for input in serialized_node.inputs
        }
        actual_args = {
            input.name: self.deserialize_input(input.arg)
            for input in serialized_node.inputs
        }
        args = []
        kwargs: OrderedDict[str, Any] = OrderedDict()
        for schema_arg in schema_args:
            if schema_arg.name in actual_args:
                arg = actual_args[schema_arg.name]
                kind = argument_kinds[schema_arg.name]
                if kind == ArgumentKind.POSITIONAL:
                    args.append(arg)
                    continue
                elif kind == ArgumentKind.KEYWORD and not keyword.iskeyword(schema_arg.name):
                    kwargs[schema_arg.name] = arg
                    continue

            # If there's no ArgumentKind found, fallback to the old cases.
            is_positional = (
                not schema_arg.has_default_value() and not schema_arg.kwarg_only
            )
            if is_positional:
                args.append(actual_args[schema_arg.name])
            elif keyword.iskeyword(schema_arg.name):
                assert not schema_arg.kwarg_only
                if len(kwargs) > 0:
                    kwargs = OrderedDict()
                    args.extend(list(kwargs.values()))
                args.append(actual_args[schema_arg.name])
            else:
                if schema_arg.name in actual_args:
                    kwargs[schema_arg.name] = actual_args[schema_arg.name]
        return tuple(args), kwargs

    def deserialize_hoo_inputs(self, inputs: list[NamedArgument]):
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
        elif typ_ == "as_sym_float":
            return self.deserialize_sym_argument(inp.as_sym_float)
        elif typ_ == "as_sym_bool":
            return self.deserialize_sym_argument(inp.as_sym_bool)
        elif isinstance(value, list):
            if len(value) == 0:
                return []
            elif typ_ == "as_tensors":
                result = [self.serialized_name_to_node[arg.name] for arg in value]
                return result
            elif typ_ in ("as_ints", "as_floats", "as_bools", "as_strings"):
                # convert from serialized.python.types.List to python list
                return list(value)
            elif typ_ in ("as_sym_ints", "as_sym_bools", "as_sym_floats"):
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
        elif isinstance(sym_arg, SymFloatArgument):
            if sym_arg.type == "as_float":
                return sym_arg.as_float
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
            # If it is a HOP node and it returns a tuple containing a single element
            # we manually insert a getitem node to ensure the graph is consistent
            # For BC, getattr() will return True if `is_single_tensor_return` doens't exist
            # as prior to adding this field, it is guaranteed to have a single tensor return
            # when the serialized_node has length=1 outputs and of type `as_tensor`.
            if (
                "torch.ops.higher_order" in serialized_node.target
                and not getattr(serialized_node, "is_hop_single_tensor_return", True)
            ):
                meta_val: list[Any] = []
                arg = serialized_node.outputs[0].as_tensor
                deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)
                self.generate_getitem(meta_val, fx_node, arg, 0, deserialized_metadata)
                fx_node.meta["val"] = tuple(meta_val)
                self.serialized_name_to_node[fx_node.name] = fx_node
                return

            self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
            return
        elif len(serialized_node.outputs) == 1 and isinstance(
            serialized_node.outputs[0].value, (SymIntArgument, SymBoolArgument, SymFloatArgument)
        ):
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return
        elif len(serialized_node.outputs) == 1 and serialized_node.outputs[0].type == "as_none":
            # manually rename the node to a unused name to avoid naming conflicts
            fx_node.meta["val"] = None
            fx_node._rename(f"{self.graph._target_to_str(fx_node.target)}_unused")
            return

        self.deserialize_multiple_outputs(serialized_node, fx_node)

    def generate_getitem(
        self,
        meta_val,
        fx_node: torch.fx.Node,
        arg: Union[TensorArgument, SymIntArgument, SymFloatArgument],
        idx: int,
        deserialized_metadata: dict[str, Any],
    ):
        if isinstance(arg, TensorArgument):
            name = arg.name
        elif isinstance(arg, SymIntArgument):
            name = arg.as_name
        elif isinstance(arg, SymFloatArgument):
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

    def generate_getitems(
        self,
        meta_val,
        fx_node: torch.fx.Node,
        args,
        deserialized_metadata: dict[str, Any],
    ):
        for idx, arg in enumerate(args):
            if isinstance(arg, (TensorArgument, SymIntArgument, SymFloatArgument)):
                self.generate_getitem(meta_val, fx_node, arg, idx, deserialized_metadata)
                continue

            assert isinstance(arg, Argument)
            if arg.type in ("as_tensor", "as_sym_int", "as_sym_float"):
                self.generate_getitem(meta_val, fx_node, arg.value, idx, deserialized_metadata)
            elif arg.type in (
                "as_tensors",
                "as_sym_ints",
                "as_sym_floats",
                "as_ints",
                "as_floats",
                "as_strings",
                "as_bools",
                "as_sym_bools",
            ):
                list_output = self.graph.create_node(
                    "call_function",
                    operator.getitem,
                    (fx_node, idx),
                )
                meta_val.append([])
                self.generate_getitems(meta_val[-1], list_output, arg.value, deserialized_metadata)
                list_output.meta.update(deserialized_metadata)
                list_output.meta["val"] = meta_val[-1]
            elif arg.type == "as_none":
                individual_output = self.graph.create_node(
                    "call_function",
                    operator.getitem,
                    (fx_node, idx),
                    name="as_none",
                )
                meta_val.append(None)
                individual_output.meta['val'] = None
                individual_output.meta.update(deserialized_metadata)
            else:
                raise NotImplementedError(f"Unimplemented node output type: {arg}")

    def deserialize_multiple_outputs(
        self, serialized_node: Node, fx_node: torch.fx.Node
    ) -> None:
        deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)

        # Convert multiple return types to FX format.
        # In FX, each node only returns one value. So in order to represent
        # multiple return values, we have to emit a `getitem` node for each
        # return value.
        # This performs the inverse mapping of the `serialize_outputs` call in
        # serialization, see [NOTE: Multiple outputs]
        meta_val: list[Any] = []
        if len(serialized_node.outputs) == 1:
            assert isinstance(serialized_node.outputs[0].value, list)
            assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
            self.generate_getitems(meta_val, fx_node, serialized_node.outputs[0].as_tensors, deserialized_metadata)
        else:
            self.generate_getitems(meta_val, fx_node, serialized_node.outputs, deserialized_metadata)

        # also update the metaval for `fx_node` to be a list(meta)
        fx_node.meta["val"] = tuple(meta_val)
        self.serialized_name_to_node[fx_node.name] = fx_node

    def deserialize_metadata(self, metadata: dict[str, str]) -> dict[str, Any]:
        ret: dict[str, Any] = {}
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

            # Helper function to split string by commas, accounting for nested parentheses/brackets
            def metadata_split(metadata):
                out = []
                start, n = 0, 0
                a, b = "[(", ")]"
                for end, c in enumerate(metadata):
                    if c in a:
                        n += 1
                    elif c in b:
                        n -= 1
                    elif c == "," and n == 0:
                        out.append(metadata[start : end])
                        start = end + 1
                out.append(metadata[start:])
                assert len(out) == 3
                return out

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

        if custom_str := metadata.get("custom"):
            ret["custom"] = json.loads(custom_str)

        return ret

    def deserialize_argument_spec(self, x: Argument) -> ep.ArgumentSpec:
        log.debug("[deserialize_argument_spec] %s", x)
        if x.type == "as_tensor":
            return ep.TensorArgument(name=x.as_tensor.name)
        elif x.type == "as_sym_int":
            return ep.SymIntArgument(name=x.as_sym_int.as_name)
        elif x.type == "as_sym_float":
            return ep.SymFloatArgument(name=x.as_sym_float.as_name)
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
            forward_arg_names=names if (names := module_call_signature.forward_arg_names) else None,
        )

    def deserialize_module_call_graph(
        self, module_call_graph: list[ModuleCallEntry]
    ) -> list[ep.ModuleCallEntry]:
        log.debug("\n[deserialize_module_call_graph]")
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
    def __init__(self, expected_opset_version: Optional[dict[str, int]] = None):
        self.expected_opset_version: dict[str, int] = {}
        if expected_opset_version:
            self.expected_opset_version.update(expected_opset_version)
        if "aten" not in self.expected_opset_version:
            self.expected_opset_version["aten"] = torch._C._get_max_operator_version()

    def deserialize_range_constraints(
        self,
        symbol_name_to_range: dict[str, symbolic_shapes.ValueRanges],
        symbol_name_to_symbol: dict[str, sympy.Symbol],
    ) -> dict[sympy.Symbol, ValueRanges]:
        log.debug("\n[deserialize_range_constraints]")
        range_constraints = {}
        for k, v in symbol_name_to_range.items():
            if symbol := symbol_name_to_symbol.get(k):
                log.debug("[deserialize_range_constraints] %s -> %s", k, v)
                range_constraints[symbol] = v  # type: ignore[arg-type]
            else:
                log.warning("Symbol %s did not appear in the graph that was deserialized", k)
        return range_constraints

    def deserialize(
        self,
        exported_program: ExportedProgram,
        state_dict: Union[dict[str, torch.Tensor], bytes],
        constants: Union[dict[str, torch.Tensor], bytes],
        example_inputs: Optional[Union[tuple[tuple[torch.Tensor, ...], dict[str, Any]], bytes]] = None,
        *,
        _unsafe_skip_version_check=False,
    ) -> ep.ExportedProgram:
        assert isinstance(exported_program, ExportedProgram)
        version = exported_program.schema_version

        # TODO(zhxchen17) blocked on thrift schema refactor
        if version.major != SCHEMA_VERSION[0] and not (version.major == 0 and version.minor == 0):
            if not _unsafe_skip_version_check:
                raise SerializeError(
                    f"Serialized schema version {exported_program.schema_version} "
                    f"does not match our current schema version {SCHEMA_VERSION}."
                )

        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(
                _int_to_sympy_int(v.min_val, -int_oo), _int_to_sympy_int(v.max_val, int_oo)
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

        result = ep.ExportedProgram(
            root=res.graph_module,
            graph=res.graph_module.graph,
            graph_signature=res.signature,
            state_dict=res.state_dict,  # type: ignore[arg-type]
            range_constraints=range_constraints,
            module_call_graph=res.module_call_graph,
            example_inputs=res.example_inputs,
            constants=res.constants,
            verifiers=[load_verifier(v) for v in exported_program.verifiers],
        )
        log.debug("\n[deserialize]: %s", result)
        return result


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
        }
    elif isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if obj == math.inf:
            return "Infinity"
        elif obj == -math.inf:
            return "-Infinity"
        elif obj == math.nan:
            return "NaN"
        else:
            return obj
    else:
        return obj


def _to_json_bytes(obj: Any) -> bytes:
    return json.dumps(_dataclass_to_dict(obj), cls=EnumEncoder, allow_nan=False).encode("utf-8")


def serialize(
    exported_program: ep.ExportedProgram,
    opset_version: Optional[dict[str, int]] = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> SerializedArtifact:
    with _enable_graph_inputs_of_type_nn_module(exported_program.example_inputs):
        serialized_program = ExportedProgramSerializer(opset_version, pickle_protocol).serialize(
            exported_program
        )
    assert isinstance(serialized_program.exported_program, ExportedProgram)

    json_bytes = _to_json_bytes(serialized_program.exported_program)
    artifact = SerializedArtifact(
        json_bytes,
        serialized_program.state_dict,
        serialized_program.constants,
        serialized_program.example_inputs
    )
    return artifact


def _dict_to_dataclass(cls, data):
    assert not isinstance(cls, str), f"Unresolved class type: '{cls}'."
    if typing.get_origin(cls) == Annotated:
        return _dict_to_dataclass(cls.__origin__, data)
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
        obj = cls(**data)  # type: ignore[assignment,operator]
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
    elif cls == float:
        return float(data)
    return data


def deserialize(
    artifact: SerializedArtifact,
    expected_opset_version: Optional[dict[str, int]] = None,
    *,
    _unsafe_skip_version_check=False,
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
            _unsafe_skip_version_check=_unsafe_skip_version_check,
        )
    )


def _canonicalize_graph(
    sorted_inputs, sorted_outputs, graph, constants
) -> tuple[Graph, dict[str, str]]:
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
        elif a.type == "as_sym_float":
            return a.as_sym_float
        elif a.type == "as_sym_floats":
            return a.as_sym_floats
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
            return a.as_custom_obj
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
            outs: list[int]
            ins: int

        graph_inputs: set[str] = set()
        def_table: dict[str, int] = {}
        edges: dict[int, Edges] = {}
        candidates: list[tuple[str, list[tuple[str, list[int]]], int]] = []
        rank: dict[str, int] = {}
        ret: list[Node] = []

        def get_name(a) -> Optional[str]:
            if a is None:
                return None
            if isinstance(a, TensorArgument):
                return a.name
            elif isinstance(a, (SymIntArgument, SymBoolArgument, SymFloatArgument)):
                if a.type == "as_name":
                    return a.as_name
                elif a.type in ("as_int", "as_bool", "as_float"):
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
            elif isinstance(a, CustomObjArgument):
                return a.name
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
                    if s in constants:
                        return
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
            s = get_name(a)
            if s and s not in constants:
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
    name_table: dict[str, str] = {}

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
        elif isinstance(a, SymFloatArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_float_values)
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_bool_values)
        elif isinstance(a, CustomObjArgument):
            a.name = _rename(a.name, graph.custom_obj_values)
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    def replace_use(a):
        if a is None:
            return
        if isinstance(a, TensorArgument):
            a.name = name_table.get(a.name, a.name)
        elif isinstance(a, (SymIntArgument, SymFloatArgument)):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        elif isinstance(a, OptionalTensorArgument):
            if a.type == "as_tensor":
                a.as_tensor.name = name_table.get(a.as_tensor.name, a.as_tensor.name)
        elif isinstance(a, CustomObjArgument):
            a.name = name_table.get(a.name, a.name)
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
    sorted_sym_float_values = dict(
        sorted(graph.sym_float_values.items(), key=operator.itemgetter(0))
    )
    sorted_sym_bool_values = dict(
        sorted(graph.sym_bool_values.items(), key=operator.itemgetter(0))
    )
    sorted_custom_obj_values = dict(
        sorted(graph.custom_obj_values.items(), key=operator.itemgetter(0))
    )

    # Stage 5: Recurse in subgraphs.
    counter = 0
    for node in sorted_nodes:
        for i in node.inputs:
            a = i.arg
            if a.type == "as_graph":
                a.as_graph.graph, _ = _canonicalize_graph(
                    a.as_graph.graph.inputs, a.as_graph.graph.outputs, a.as_graph.graph, constants
                )
                a.as_graph.name = f"_g{counter}"
                counter += 1

    graph = Graph(
        inputs=sorted_inputs,
        outputs=sorted_outputs,
        nodes=sorted_nodes,
        tensor_values=sorted_tensor_values,
        sym_int_values=sorted_sym_int_values,
        sym_float_values=sorted_sym_float_values,
        sym_bool_values=sorted_sym_bool_values,
        is_single_tensor_return=graph.is_single_tensor_return,
        custom_obj_values=sorted_custom_obj_values,
    )
    return graph, name_table


def canonicalize(ep: ExportedProgram, constants: Optional[set[str]] = None) -> ExportedProgram:
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
        constants (Optional[set[str]]): Set of constants names

    Returns:
        ExportedProgram: The canonicalized exported program.
    """
    ep = copy.deepcopy(ep)
    constants: set[str] = constants or set()

    opset_version = dict(sorted(ep.opset_version.items(), key=operator.itemgetter(0)))
    range_constraints = dict(sorted(ep.range_constraints.items(), key=operator.itemgetter(0)))
    module_call_graph = sorted(ep.graph_module.module_call_graph, key=lambda x: x.fqn)
    signature = ep.graph_module.signature
    graph = ep.graph_module.graph

    assert len(graph.inputs) == len(signature.input_specs)
    assert len(graph.outputs) == len(signature.output_specs)

    def rank_input(inp) -> tuple[int, Optional[str], int]:
        idx, (_arg, spec) = inp
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

    def rank_output(out) -> tuple[int, Optional[str], int]:
        idx, (_arg, spec) = out
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

    if len(sorted_ins) > 0:
        sorted_inputs, input_specs = zip(*(i for idx, i in sorted_ins))  # type: ignore[assignment]
    else:
        sorted_inputs = ()
        input_specs = ()

    sorted_outs = sorted(
        enumerate(zip(graph.outputs, signature.output_specs)), key=rank_output
    )
    sorted_outputs, output_specs = zip(*(i for idx, i in sorted_outs))  # type: ignore[assignment]

    sorted_graph, replace_table = _canonicalize_graph(
        sorted_inputs, sorted_outputs, graph, constants
    )

    def replace_input(spec):
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
            elif arg.type == "as_sym_float":
                f = arg.as_sym_float
                if f.type == "as_name":
                    f.as_name = replace_table[f.as_name]
                elif f.type == "as_float":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_float type: {f}")
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
            t_custom_obj = spec.custom_obj.arg
            t_custom_obj.name = replace_table[t_custom_obj.name]
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
            elif arg.type == "as_sym_float":
                f = arg.as_sym_float
                if f.type == "as_name":
                    f.as_name = replace_table[f.as_name]
                elif f.type == "as_float":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_float type: {f}")
            elif arg.type in ("as_none", "as_bool", "as_int", "as_float", "as_string"):
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
        verifiers=ep.verifiers,
        torch_version=ep.torch_version,
    )


class ExtensionHandler:
    """
    Base class for handling extension operators.
    """
    @classmethod
    def namespace(cls) -> str:
        raise NotImplementedError(f"{cls.__class__} namespace() must be implemented")

    @classmethod
    def to_op_name(cls, op) -> str:
        raise NotImplementedError(f"{cls.__class__} op_name() must be implemented")

    @classmethod
    def from_op_name(cls, name: str):
        raise NotImplementedError(f"{cls.__class__} op_name() must be implemented")

    @classmethod
    def op_schema(cls, op) -> torch.FunctionSchema:
        raise NotImplementedError(f"{cls.__class__} op_schema() must be implemented")


def register_extension(
    op_type: type[Any],
    extension_handler: type[ExtensionHandler],
):
    """Register custom de/serialization method for a node with non-standard type."""
    assert issubclass(extension_handler, ExtensionHandler), f"Expected ExtensionHandler, got {extension_handler}."
    assert op_type not in _serialization_registry, f"{op_type} is already registered."
    assert isinstance(op_type, type)  # Maybe a good idea to enforce this first.
    assert not (op_type.__module__.startswith("torch") or op_type.__module__.startswith("builtins"))
    assert extension_handler.namespace() not in _deserialization_registry
    _serialization_registry[op_type] = extension_handler
    _deserialization_registry[extension_handler.namespace()] = extension_handler


def _registered_extension_types():
    return tuple(
        _serialization_registry.keys()
    )


# Registry to store all custom serialization implementations.
# The registry maps a operation to its serialization function (a callable), in their own
# namespace to avoid conflicts.
# Serialization: Op type --> custom handler.
# De-serialization: Namespace --> custom handler.
_serialization_registry: dict[type[Any], type[ExtensionHandler]] = {}
_deserialization_registry: dict[str, type[ExtensionHandler]] = {}
