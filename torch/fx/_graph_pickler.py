import dataclasses
import importlib
import io
import pickle
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, NewType, Optional, TypeVar, Union
from typing_extensions import override, Self

import torch
import torch.utils._pytree as pytree
from torch._guards import TracingContext
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, Tensor
from torch._subclasses.meta_utils import (
    MetaConverter,
    MetaTensorDesc,
    MetaTensorDescriber,
)
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._mode_utils import no_dispatch


_SymNodeT = TypeVar("_SymNodeT", torch.SymInt, torch.SymFloat)


def _is_process_group(obj: object) -> bool:
    """
    Check if an object is a ProcessGroup instance. This uses a deferred import
    to avoid importing distributed modules when they're not needed.
    """
    try:
        from torch._C._distributed_c10d import ProcessGroup

        return isinstance(obj, ProcessGroup)
    except ImportError:
        return False


def _ops_filter_safe(name: str) -> bool:
    """
    An ops filter which allows pickle-safe ops. Pickle-safe ops are built-in
    ones where it will be possible to unpickle on any machine which has PyTorch.
    """
    # TODO: This list is pretty pessimistic right now. What's the full list?
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
            "torch.ops.c10d",
            "torch.ops.device_mesh",
            "torch.ops.profiler",
            "torch.ops.higher_order",
        )
    )


def _node_metadata_key_filter_safe(key: str) -> bool:
    """
    A metadata filter which allows pickle-safe node metadata. These often times contain
    stacks with pointers to unserializable objects, so we clear them out.
    """
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


@dataclasses.dataclass
class Options:
    # A filter for which ops will cause the pickler to raise a
    # BypassFxGraphCache exception. If None then all ops are allowed.
    ops_filter: Optional[Callable[[str], bool]] = _ops_filter_safe
    node_metadata_key_filter: Optional[Callable[[str], bool]] = (
        _node_metadata_key_filter_safe
    )


class GraphPickler(pickle.Pickler):
    """
    GraphPickler is a Pickler which helps pickling fx graph - in particular
    GraphModule.
    """

    def __init__(self, file: io.BytesIO, options: Optional[Options] = None) -> None:
        super().__init__(file)
        self.options = options or Options()
        self._debug_pickled_types: set[type] = set()

        # This abomination is so we can pass external decoding state to the
        # unpickler functions. We serialize _unpickle_state as a persistent
        # external item and when we deserialize it we return the common state
        # object.
        self._unpickle_state = _UnpickleStateToken(object())

        # This is used to describe tensors. It needs to be common across the
        # pickle so that duplicates and views are properly handled.
        self._meta_tensor_describer = MetaTensorDescriber(copy_data=False)

    @override
    # pyrefly: ignore [bad-override]
    def reducer_override(
        self, obj: object
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # This function is supposed to return either NotImplemented (meaning to
        # do the default pickle behavior) or a pair of (unpickle callable, data
        # to pass to unpickle).

        # We could instead teach individual classes how to pickle themselves but
        # that has a few problems:
        #
        #   1. If we have some special needs (maybe for this use-case we don't
        #      want to fully serialize every field) then we're adding private
        #      details to a public interface.
        #
        #   2. If we need to have some common shared data (such as a
        #      FakeTensorMode) which is passed to each value it's harder to
        #      support.

        # These are the types that need special handling. See the individual
        # *PickleData classes for details on pickling that particular type.
        if isinstance(obj, FakeTensor):
            return _TensorPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.fx.GraphModule):
            return _GraphModulePickleData.reduce_helper(self, obj)
        elif isinstance(obj, (torch._ops.OperatorBase, torch._ops.OpOverloadPacket)):
            return _OpPickleData.reduce_helper(self, obj)
        elif isinstance(obj, ShapeEnv):
            return _ShapeEnvPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.SymInt):
            return _SymNodePickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch._guards.TracingContext):
            return _TracingContextPickleData.reduce_helper(self, obj)
        elif isinstance(obj, FakeScriptObject):
            return _FakeScriptObjectPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.ScriptObject):
            # Handle real ScriptObject instances (like _RecordFunction) that
            # end up in the graph during tracing. These cannot be pickled with
            # standard pickle since their C++ __getstate__ may not be implemented.
            return _ScriptObjectPickleData.reduce_helper(self, obj)
        elif _is_process_group(obj):
            # Handle ProcessGroup instances before the generic opaque type check.
            # ProcessGroups are stored by group name and resolved at load time.
            return _ProcessGroupPickleData.reduce_helper(self, obj)
        elif is_opaque_type(type(obj)):
            return _OpaqueObjectPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch._C.FunctionSchema):
            return _FunctionSchemaPickleData.reduce_helper(self, obj)
        else:
            # We should never get a raw Node!
            assert not isinstance(obj, torch.fx.Node)
            if reduce := _TorchNumpyPickleData.reduce_helper(self, obj):
                return reduce

            # DEBUG: Log unknown types for debugging serialization issues
            obj_type = type(obj)
            if obj_type not in self._debug_pickled_types:
                self._debug_pickled_types.add(obj_type)
                import logging
                logging.debug(
                    f"[GraphPickler] Falling back to default pickle for type: "
                    f"{obj_type.__module__}.{obj_type.__name__}, obj repr: {repr(obj)[:200]}"
                )

            # returning `NotImplemented` causes pickle to revert to the default
            # behavior for this object.
            return NotImplemented

    @override
    def persistent_id(self, obj: object) -> Optional[str]:
        if obj is self._unpickle_state:
            return "unpickle_state"
        else:
            return None

    @classmethod
    def dumps(cls, obj: object, options: Optional[Options] = None) -> bytes:
        """
        Pickle an object.
        """
        with io.BytesIO() as stream:
            pickler = cls(stream, options)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def loads(data: bytes, fake_mode: FakeTensorMode) -> object:
        """
        Unpickle an object.
        """
        state = _UnpickleState(fake_mode)
        with io.BytesIO(data) as stream:
            unpickler = _GraphUnpickler(stream, state)
            try:
                return unpickler.load()
            except Exception as e:
                import traceback
                print(f"[GraphPickler.loads] Exception type: {type(e).__name__}")
                print(f"[GraphPickler.loads] Exception message: {e}")
                print(f"[GraphPickler.loads] Exception args: {e.args}")
                print(f"[GraphPickler.loads] Full traceback:")
                traceback.print_exc()
                raise


class _UnpickleState:
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        self.fake_mode = fake_mode
        self.meta_converter: MetaConverter[FakeTensor] = MetaConverter()


# This token is passed when pickling to indicate that we want to use the
# unpickler's _UnpickleState as a parameter in that position.
_UnpickleStateToken = NewType("_UnpickleStateToken", object)


class _GraphUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, unpickle_state: _UnpickleState) -> None:
        super().__init__(stream)
        self._unpickle_state = unpickle_state

    @override
    def persistent_load(self, pid: object) -> object:
        if pid == "unpickle_state":
            return self._unpickle_state
        else:
            raise pickle.UnpicklingError("Invalid persistent ID")


class _ShapeEnvPickleData:
    data: dict[str, object]

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: ShapeEnv
    ) -> tuple[
        Callable[[Self, _UnpickleState], ShapeEnv], tuple[Self, _UnpickleStateToken]
    ]:
        return cls.unpickle, (cls(obj), pickler._unpickle_state)

    def __init__(self, env: ShapeEnv) -> None:
        # In theory pickle should recognize that a given ShapeEnv was already
        # pickled and reuse the resulting _ShapeEnvPickleData (so two objects
        # pointing at the same ShapeEnv get the same ShapeEnv out).
        assert not env._translation_validation_enabled
        self.data = env.__dict__.copy()
        del self.data["tracked_fakes"]
        del self.data["fake_tensor_cache"]

    def unpickle(self, unpickle_state: _UnpickleState) -> ShapeEnv:
        # Fill in the existing ShapeEnv rather than creating a new one
        assert unpickle_state.fake_mode
        assert unpickle_state.fake_mode.shape_env

        for k, v in self.data.items():
            setattr(unpickle_state.fake_mode.shape_env, k, v)

        return unpickle_state.fake_mode.shape_env


class _SymNodePickleData:
    @classmethod
    def reduce_helper(
        cls,
        pickler: GraphPickler,
        obj: _SymNodeT,
    ) -> tuple[
        Callable[[Self, _UnpickleState], _SymNodeT], tuple[Self, _UnpickleStateToken]
    ]:
        args = (cls(obj.node), pickler._unpickle_state)
        if isinstance(obj, torch.SymInt):
            # pyrefly: ignore [bad-return]
            return _SymNodePickleData.unpickle_sym_int, args
        else:
            raise NotImplementedError(f"Unhandled SymNode type {type(obj)}")

    def __init__(self, node: SymNode) -> None:
        self.expr = node._expr
        self.shape_env = node.shape_env
        self.pytype = node.pytype
        self.hint = node._hint

    def _to_sym_node(self) -> SymNode:
        assert self.shape_env is not None
        return SymNode(self.expr, self.shape_env, self.pytype, self.hint)

    def unpickle_sym_int(self, unpickle_state: _UnpickleState) -> torch.SymInt:
        return torch.SymInt(self._to_sym_node())


class _TensorPickleData:
    metadata: MetaTensorDesc[FakeTensor]

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: FakeTensor
    ) -> tuple[
        Callable[[Self, _UnpickleState], FakeTensor], tuple[Self, _UnpickleStateToken]
    ]:
        return cls.unpickle, (
            cls(pickler._meta_tensor_describer, obj),
            pickler._unpickle_state,
        )

    def __init__(self, describer: MetaTensorDescriber, t: Tensor) -> None:
        # THINGS TO WORRY ABOUT:
        # 1. Need to make sure that two tensors with the same id end up with the
        #    same id on the other side of the wire.

        metadata = describer.describe_tensor(t)

        # view_func is fine if it's either None or a _FakeTensorViewFunc. A
        # custom one (which is basically a lambda) can't be serialized.
        assert not metadata.view_func or isinstance(
            metadata.view_func, torch._subclasses.meta_utils._FakeTensorViewFunc
        )
        self.metadata = dataclasses.replace(metadata, fake_mode=None)

        # Some debugging/verification
        for k in MetaTensorDesc._UNSERIALIZABLE:
            if k in ("fake_mode", "view_func"):
                continue
            assert getattr(self.metadata, k) is None, (
                f"not None: {k}: {getattr(self.metadata, k)}"
            )

    def unpickle(self, unpickle_state: _UnpickleState) -> FakeTensor:
        # TODO: make common w/ _output_from_cache_entry() in fake_tensor.py?
        metadata = dataclasses.replace(
            self.metadata,
            fake_mode=unpickle_state.fake_mode,
        )

        # also need to set the fake_mode on the base of a tensor if it's a view
        if metadata.is_view and metadata.base is not None:
            new_base = dataclasses.replace(
                metadata.base,
                fake_mode=unpickle_state.fake_mode,
            )
            metadata = dataclasses.replace(metadata, base=new_base)

        def with_fake(
            make_meta_t: Callable[[], torch.Tensor], device: Union[torch.device, str]
        ) -> FakeTensor:
            with no_dispatch():
                return FakeTensor(
                    unpickle_state.fake_mode,
                    make_meta_t(),
                    # pyrefly: ignore [bad-argument-type]
                    device,
                )

        return unpickle_state.meta_converter.meta_tensor(
            metadata,
            unpickle_state.fake_mode.shape_env,
            with_fake,
            None,
            None,
        )


class _TorchNumpyPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: object
    ) -> Optional[
        tuple[
            Callable[[Self, _UnpickleState], object], tuple[Self, _UnpickleStateToken]
        ]
    ]:
        if data := cls.from_object(obj):
            return (cls.unpickle, (data, pickler._unpickle_state))
        else:
            return None

    def __init__(self, mod: str, name: str) -> None:
        self.mod = mod
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> Callable[..., object]:
        np = getattr(importlib.import_module(self.mod), self.name)
        return torch._dynamo.variables.misc.get_np_to_tnp_map()[np]

    @classmethod
    def from_object(cls, tnp: object) -> Optional[Self]:
        if not callable(tnp):
            return None

        tnp_to_np = torch._dynamo.variables.misc.get_tnp_to_np_map()
        try:
            if not (np := tnp_to_np.get(tnp)):
                return None
        except TypeError:
            return None

        if not (mod := getattr(np, "__module__", None)):
            mod = "numpy"

        if not (name := getattr(np, "__name__", None)):
            return None

        # pyrefly: ignore [unbound-name]
        assert np == getattr(importlib.import_module(mod), name)
        # pyrefly: ignore [unbound-name]
        return cls(mod, name)


class _GraphModulePickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch.fx.GraphModule
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch.fx.GraphModule],
        tuple[Self, _UnpickleStateToken],
    ]:
        return cls.unpickle, (
            cls(obj, pickler.options),
            pickler._unpickle_state,
        )

    def __init__(self, gm: torch.fx.GraphModule, options: Options) -> None:
        # Need to do this to ensure the code is created for later pickling.
        if isinstance(gm, torch.fx._lazy_graph_module._LazyGraphModule):
            _python_code = gm._real_recompile()
        else:
            _python_code = gm.recompile()
        self.gm_dict = gm.__dict__.copy()
        del self.gm_dict["_graph"]
        self.graph = _GraphPickleData(gm._graph, options)

    def unpickle(self, unpickle_state: _UnpickleState) -> torch.fx.GraphModule:
        gm = torch.fx.GraphModule.__new__(torch.fx.GraphModule)
        gm.__dict__ = self.gm_dict
        gm._graph = self.graph.unpickle(gm, unpickle_state)
        return gm


class _NodePickleData:
    def __init__(
        self,
        node: torch.fx.Node,
        mapping: dict[torch.fx.Node, "_NodePickleData"],
        options: Options,
    ) -> None:
        self.args = pytree.tree_map_only(torch.fx.Node, lambda n: mapping[n], node.args)
        self.kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda n: mapping[n], node.kwargs
        )
        # -- self.graph = node.graph
        self.name = node.name
        self.op = node.op
        self.target = _OpPickleData.pickle(node.target, options)
        # self.input_nodes = node._input_nodes
        # self.users = node.users
        self.type = node.type
        # self.sort_key = node._sort_key
        # self.repr_fn = node._repr_fn
        # self.meta = node.meta
        self.meta = {
            k: v
            for k, v in node.meta.items()
            if (
                not options.node_metadata_key_filter
                or options.node_metadata_key_filter(k)
            )
        }

    def unpickle(
        self,
        graph: torch.fx.Graph,
        mapping: dict["_NodePickleData", torch.fx.Node],
        unpickle_state: _UnpickleState,
    ) -> torch.fx.Node:
        args = pytree.tree_map_only(_NodePickleData, lambda n: mapping[n], self.args)
        kwargs = pytree.tree_map_only(
            _NodePickleData, lambda n: mapping[n], self.kwargs
        )
        target = self.target.unpickle(unpickle_state)
        assert callable(target) or isinstance(target, str)
        node = graph.create_node(self.op, target, args, kwargs, self.name, self.type)
        node.meta = self.meta
        return node


class _OpPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, op: object
    ) -> tuple[Callable[[_UnpickleState], object], tuple[_UnpickleStateToken]]:
        result = cls.pickle(op, pickler.options)
        return (result.unpickle, (pickler._unpickle_state,))

    @classmethod
    def pickle(cls, op: object, options: Options) -> "_OpPickleData":
        if isinstance(op, str):
            return _OpStrPickleData(op)

        if isinstance(getattr(op, "__wrapped__", None), AOTCompiledArtifact):
            assert hasattr(op, "__wrapped__")
            artifact = op.__wrapped__
            assert isinstance(artifact, AOTCompiledArtifact)
            return _OpPrecompiledPickleData(artifact)

        name = torch.fx.Node._pretty_print_target(op)

        if isinstance(op, torch._ops.OpOverload):
            return cls._pickle_op(name, _OpOverloadPickleData, options)
        elif isinstance(op, torch._ops.OpOverloadPacket):
            return cls._pickle_op(name, _OpOverloadPacketPickleData, options)
        elif name.startswith(_OpFunctionPickleData.SUPPORTED_ROOTS):
            root, detail = name.split(".", 1)
            return _OpFunctionPickleData(root, detail)
        else:
            # TODO: raise a BypassFxGraphCache so we will just bypass this one...
            raise NotImplementedError(f"TARGET: {type(op)} {op} {name}")

    @staticmethod
    def _pickle_op(
        name: str,
        datacls: Union[
            type["_OpOverloadPickleData"], type["_OpOverloadPacketPickleData"]
        ],
        options: Options,
    ) -> "_OpPickleData":
        if (ops_filter := options.ops_filter) and not ops_filter(name):
            from torch._inductor.codecache import BypassFxGraphCache

            raise BypassFxGraphCache(f"Unable to pickle non-standard op: {name}")
        return datacls(name)

    @abstractmethod
    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        pass

    @classmethod
    def _lookup_global_by_name(cls, name: str) -> object:
        """
        Like `globals()[name]` but supports dotted names.
        """
        if "." in name:
            mod, rest = name.split(".", 1)
            root = globals()[mod]
            return cls._getattr_by_name(root, rest)
        else:
            return globals()[name]

    @staticmethod
    def _getattr_by_name(root: object, name: str) -> object:
        """
        Like `getattr(root, name)` but supports dotted names.
        """
        while "." in name:
            mod, name = name.split(".", 1)
            root = getattr(root, mod)
        return getattr(root, name)


class _OpStrPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> str:
        return self.name


class _OpOverloadPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverload:
        obj = self._lookup_global_by_name(self.name)
        assert isinstance(obj, torch._ops.OpOverload)
        return obj


class _OpOverloadPacketPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverloadPacket:
        obj = self._lookup_global_by_name(self.name)
        assert isinstance(obj, torch._ops.OpOverloadPacket)
        return obj


class _OpPrecompiledPickleData(_OpPickleData):
    def __init__(self, artifact: AOTCompiledArtifact) -> None:
        self.contents = artifact.serialize()

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        precompiled_artifact = AOTCompiledArtifact.deserialize(self.contents)
        import functools

        @functools.wraps(precompiled_artifact)
        def wrapped(*args: Any) -> Any:
            return precompiled_artifact(*args)

        return wrapped


class _OpFunctionPickleData(_OpPickleData):
    """
    Supports pickling a set of standard/common functions
    These must be prefixed with the full namespace in order to properly
    be pickled (i.e `einops.rearrange` and not `from einops import rearrange`)
    """

    # Static variable listing supported root names
    SUPPORTED_ROOTS = ("builtins.", "math.", "torch.", "operator.", "einops.")

    def __init__(self, root: str, name: str) -> None:
        self.root = root
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        if self.root == "builtins":
            return __builtins__.get(self.name)  # type: ignore[attr-defined]
        elif self.root == "math":
            import math

            return self._getattr_by_name(math, self.name)
        elif self.root == "torch":
            return self._getattr_by_name(torch, self.name)
        elif self.root == "operator":
            import operator

            return self._getattr_by_name(operator, self.name)
        elif self.root == "einops":
            import einops

            return self._getattr_by_name(einops, self.name)
        else:
            raise NotImplementedError


class _GraphPickleData:
    def __init__(self, graph: torch.fx.Graph, options: Options) -> None:
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras

        nodes: dict[torch.fx.Node, _NodePickleData] = {}
        for node in graph.nodes:
            nodes[node] = _NodePickleData(node, nodes, options)
        self.nodes = tuple(nodes.values())
        self._codegen = graph._codegen

        # Unpickled variables:
        # self._used_names = graph._used_names
        # -- self._insert = self._root.prepend
        # self._len = graph._len
        # self._graph_namespace = graph._graph_namespace
        # self._owning_module = graph._owning_module
        # self._co_fields: Dict[str, Any] = graph._co_fields
        # -- self._find_nodes_lookup_table = _FindNodesLookupTable()

    def unpickle(
        self, gm: torch.fx.GraphModule, unpickle_state: _UnpickleState
    ) -> torch.fx.Graph:
        graph = torch.fx.Graph(gm, self.tracer_cls, self.tracer_extras)

        nodes: dict[_NodePickleData, torch.fx.Node] = {}
        for nd in self.nodes:
            nodes[nd] = nd.unpickle(graph, nodes, unpickle_state)
        if hasattr(self, "_codegen"):
            graph._codegen = self._codegen

        return graph


class _TracingContextPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch._guards.TracingContext
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch._guards.TracingContext],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, context: TracingContext) -> None:
        # TODO: Do we really need all of this?
        self.module_context = context.module_context
        self.frame_summary_stack = context.frame_summary_stack
        self.loc_in_frame = context.loc_in_frame
        self.aot_graph_name = context.aot_graph_name
        self.params_flat = context.params_flat
        self.params_flat_unwrap_subclasses = context.params_flat_unwrap_subclasses
        self.params_unwrapped_to_flat_index = context.params_unwrapped_to_flat_index
        self.output_strides = context.output_strides
        self.force_unspec_int_unbacked_size_like = (
            context.force_unspec_int_unbacked_size_like
        )
        # Not saved (because it's difficult and maybe not needed?):
        #   self.fw_metadata = context.fw_metadata
        #   self.guards_context = None
        #   self.global_context = None
        #   self.fake_mode = None
        #   self.fakify_first_call = None
        #   self.hop_dispatch_set_cache = None
        #   self.tensor_to_context = context.tensor_to_context

    def unpickle(self, unpickle_state: _UnpickleState) -> TracingContext:
        context = TracingContext(unpickle_state.fake_mode)
        context.module_context = self.module_context
        context.frame_summary_stack = self.frame_summary_stack
        context.loc_in_frame = self.loc_in_frame
        context.aot_graph_name = self.aot_graph_name
        context.params_flat = self.params_flat
        context.params_flat_unwrap_subclasses = self.params_flat_unwrap_subclasses
        context.params_unwrapped_to_flat_index = self.params_unwrapped_to_flat_index
        context.output_strides = self.output_strides
        context.force_unspec_int_unbacked_size_like = (
            self.force_unspec_int_unbacked_size_like
        )
        return context


class _FakeScriptObjectPickleData:
    """
    Handles pickling of FakeScriptObject instances. FakeScriptObjects wrap opaque
    objects like ProcessGroup that cannot be directly pickled. We store the
    script_class_name and wrapped_obj, but NOT the real_obj since that will be
    provided at runtime as a graph input.
    """

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: FakeScriptObject
    ) -> tuple[
        Callable[[Self, _UnpickleState], FakeScriptObject],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, fake_obj: FakeScriptObject) -> None:
        self.script_class_name = fake_obj.script_class_name
        self.wrapped_obj = fake_obj.wrapped_obj

    def unpickle(self, unpickle_state: _UnpickleState) -> FakeScriptObject:
        fake_obj = object.__new__(FakeScriptObject)
        object.__setattr__(fake_obj, "script_class_name", self.script_class_name)
        object.__setattr__(fake_obj, "wrapped_obj", self.wrapped_obj)
        object.__setattr__(fake_obj, "real_obj", None)
        return fake_obj


class _ScriptObjectPickleData:
    """
    Handles pickling of torch.ScriptObject instances that end up in FX graphs.

    ScriptObjects like _RecordFunction (from profiler) are C++ pybind11 objects
    that may not implement __getstate__. When these objects end up in the graph
    during tracing (e.g., profiler record functions), we need to handle them
    gracefully during serialization.

    On unpickle, we return None since these objects represent runtime-only state
    (like active profiler regions) that will be recreated during execution.
    """

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch.ScriptObject
    ) -> tuple[
        Callable[[Self, _UnpickleState], None],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, script_obj: torch.ScriptObject) -> None:
        # Store identifying information about the ScriptObject for debugging
        # We can't access most attributes since they may trigger the same
        # serialization error, so just store the type name.
        self.type_name = type(script_obj).__name__
        self.qualified_name = str(script_obj._type().qualified_name())

    def unpickle(self, unpickle_state: _UnpickleState) -> None:
        # Return None since ScriptObjects like _RecordFunction represent
        # runtime-only state that will be recreated during graph execution.
        # The profiler _record_function_exit call will handle None gracefully.
        return None


class _ProcessGroupPickleData:
    """
    Handles pickling of ProcessGroup instances. ProcessGroups are C++ pybind11
    objects that cannot be directly pickled. During precompilation with fake
    distributed mode, ProcessGroups may be captured in the graph. On unpickle,
    we resolve the group name to a real ProcessGroup using the c10d registry.

    This enables AOT precompilation to serialize graphs containing ProcessGroups
    and then load them in a real distributed environment where the ProcessGroups
    have been initialized.
    """

    @classmethod
    def reduce_helper(
        cls,
        pickler: GraphPickler,
        obj: "torch.distributed.ProcessGroup",  # type: ignore[name-defined]
    ) -> tuple[
        Callable[[Self, _UnpickleState], Any],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(
        self, pg: "torch.distributed.ProcessGroup"  # type: ignore[name-defined]
    ) -> None:
        # Store the group name which can be used to resolve the ProcessGroup
        # at load time. The group_name is a unique identifier for the group
        # that is registered in the c10d ProcessGroup registry.
        self.group_name = pg.group_name

    def unpickle(self, unpickle_state: _UnpickleState) -> Any:
        # Resolve the group name to a real ProcessGroup using the c10d registry.
        # This will return the ProcessGroup that was initialized in the current
        # distributed environment with the same group name.
        from torch._C._distributed_c10d import _resolve_process_group

        try:
            return _resolve_process_group(self.group_name)
        except Exception as e:
            # If resolution fails (e.g., distributed not initialized or group
            # not found), log a warning and return None. This allows the graph
            # to still be loaded, though collective ops will fail at runtime.
            import logging

            logging.warning(
                f"[_ProcessGroupPickleData] Failed to resolve ProcessGroup "
                f"with name '{self.group_name}': {e}. Returning None."
            )
            return None


class _OpaqueObjectPickleData:
    """
    Handles pickling of opaque objects. These objects are C++ pybind11 objects
    that cannot be directly pickled. We store just the type name, and on
    unpickle return None since the actual object will be provided at runtime
    as a graph input.

    Note: ProcessGroups are handled separately by _ProcessGroupPickleData which
    resolves them by group name at load time.
    """

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: object
    ) -> tuple[
        Callable[[Self, _UnpickleState], None],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, opaque_obj: object) -> None:
        self.type_name = type(opaque_obj).__name__
        self.module_name = type(opaque_obj).__module__

    def unpickle(self, unpickle_state: _UnpickleState) -> None:
        return None


class _FunctionSchemaPickleData:
    """
    Handles pickling of torch._C.FunctionSchema and HopSchema instances.

    FunctionSchema's default pickle behavior uses __getstate__ which returns
    the schema as a string, and __setstate__ which parses that string.
    However, the C++ schema parser cannot handle all type specifiers
    (e.g., PyObject, NoneType) that may appear in HopSchema for
    HigherOrderOperators.

    This class serializes the schema by storing its individual components
    (arguments, returns, etc.) and reconstructs them on unpickle, avoiding
    the string parsing entirely.
    """

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch._C.FunctionSchema
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch._C.FunctionSchema],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, schema: torch._C.FunctionSchema) -> None:
        self.name = schema.name
        self.overload_name = schema.overload_name

        # Serialize arguments - store the type as a string representation
        # along with other argument properties
        self.arguments = [self._serialize_argument(arg) for arg in schema.arguments]
        self.returns = [self._serialize_argument(ret) for ret in schema.returns]

        # Handle HopSchema's additional attributes
        # Note: torch._C.FunctionSchema doesn't expose is_vararg/is_varret as
        # readable attributes, but HopSchema stores them as instance attributes
        self.is_vararg = getattr(schema, "is_vararg", False)
        self.is_varret = getattr(schema, "is_varret", False)
        self.tree_spec = getattr(schema, "tree_spec", None)
        self.is_hop_schema = hasattr(schema, "tree_spec")

    def _serialize_argument(self, arg: torch._C.Argument) -> dict[str, Any]:
        """
        Serialize a torch._C.Argument to a dictionary.
        We store the type as a string that can be reconstructed.
        """
        return {
            "name": arg.name,
            "type_str": str(arg.type),
            "type_kind": arg.type.kind() if hasattr(arg.type, "kind") else None,
            "default_value": arg.default_value,
            "kwarg_only": arg.kwarg_only,
            "alias_info": self._serialize_alias_info(arg.alias_info),
        }

    def _serialize_alias_info(
        self, alias_info: Optional[torch._C._AliasInfo]
    ) -> Optional[dict[str, Any]]:
        """Serialize alias info if present."""
        if alias_info is None:
            return None
        return {
            "is_write": alias_info.is_write,
            "before_set": set(alias_info.before_set),
            "after_set": set(alias_info.after_set),
        }

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._C.FunctionSchema:
        args = [self._deserialize_argument(arg_data) for arg_data in self.arguments]
        rets = [self._deserialize_argument(ret_data) for ret_data in self.returns]

        if self.is_hop_schema:
            from torch._higher_order_ops.schema import HopSchema

            return HopSchema(
                self.name,
                self.overload_name,
                args,
                rets,
                self.is_vararg,
                self.is_varret,
                self.tree_spec,
            )
        else:
            return torch._C.FunctionSchema(
                self.name,
                self.overload_name,
                args,
                rets,
                self.is_vararg,
                self.is_varret,
            )

    def _deserialize_argument(self, arg_data: dict[str, Any]) -> torch._C.Argument:
        """Deserialize a torch._C.Argument from a dictionary."""
        ty = self._string_to_type(arg_data["type_str"])
        alias_info = self._deserialize_alias_info(arg_data["alias_info"])

        return torch._C.Argument(
            arg_data["name"],
            ty,
            None,  # N (dimension)
            arg_data["default_value"],
            arg_data["kwarg_only"],
            alias_info,
        )

    def _deserialize_alias_info(
        self, alias_data: Optional[dict[str, Any]]
    ) -> Optional[torch._C._AliasInfo]:
        """Deserialize alias info from a dictionary."""
        if alias_data is None:
            return None

        # The C++ _AliasInfo strips the "alias::" namespace prefix when exposing
        # before_set and after_set as Python properties, but requires it when
        # constructing. We need to add the prefix back for reconstruction.
        def add_namespace(symbol_set: set[str]) -> set[str]:
            return {f"alias::{s}" if "::" not in s else s for s in symbol_set}

        # pyrefly: ignore [attr-defined]
        return torch._C._AliasInfo(
            alias_data["is_write"],
            add_namespace(alias_data["before_set"]),
            add_namespace(alias_data["after_set"]),
        )

    def _string_to_type(self, type_str: str) -> Any:
        """
        Convert a type string back to a JIT type object.
        This handles both standard types and special ones like PyObject, NoneType.
        """
        # Handle simple well-known types
        type_map = {
            "int": torch._C.IntType.get(),
            "float": torch._C.FloatType.get(),
            "str": torch._C.StringType.get(),
            "bool": torch._C.BoolType.get(),
            "SymInt": torch._C.SymIntType.get(),
            "SymBool": torch._C.SymBoolType.get(),
            "Tensor": torch._C.TensorType.get(),
            "Any": torch._C.AnyType.get(),
            "None": torch._C.NoneType.get(),
            "NoneType": torch._C.NoneType.get(),
            "Device": torch._C.DeviceObjType.get(),
            "PyObject": torch._C.PyObjectType.get(),
        }

        if type_str in type_map:
            return type_map[type_str]

        # Handle optional types like "int?"
        if type_str.endswith("?"):
            inner = self._string_to_type(type_str[:-1])
            return torch._C.OptionalType(inner)

        # Handle list types like "int[]" or "Tensor[]"
        if type_str.endswith("[]"):
            inner = self._string_to_type(type_str[:-2])
            return torch._C.ListType(inner)

        # Handle tuple types like "(Tensor, Tensor)"
        if type_str.startswith("(") and type_str.endswith(")"):
            inner_str = type_str[1:-1]
            if not inner_str:
                return torch._C.TupleType([])
            inner_types = [
                self._string_to_type(t.strip()) for t in inner_str.split(",")
            ]
            return torch._C.TupleType(inner_types)

        # For unknown types, fall back to AnyType (this is a safe fallback
        # since these schemas are primarily used for metadata/debugging)
        import logging

        logging.debug(
            f"[_FunctionSchemaPickleData] Unknown type '{type_str}', "
            f"falling back to AnyType"
        )
        return torch._C.AnyType.get()
