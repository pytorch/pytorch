import dataclasses
import importlib
import io
import pickle
from abc import abstractmethod
from typing import Any, Callable, NewType, Optional, TypeVar, Union
from typing_extensions import override, Self

import torch
import torch.utils._pytree as pytree
from torch._guards import TracingContext
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
        )
    )


@dataclasses.dataclass
class Options:
    # A filter for which ops will cause the pickler to raise a
    # BypassFxGraphCache exception. If None then all ops are allowed.
    ops_filter: Optional[Callable[[str], bool]] = _ops_filter_safe


class GraphPickler(pickle.Pickler):
    """
    GraphPickler is a Pickler which helps pickling fx graph - in particular
    GraphModule.
    """

    def __init__(self, file: io.BytesIO, options: Optional[Options] = None) -> None:
        super().__init__(file)
        self.options = options or Options()

        # This abomination is so we can pass external decoding state to the
        # unpickler functions. We serialize _unpickle_state as a persistent
        # external item and when we deserialize it we return the common state
        # object.
        self._unpickle_state = _UnpickleStateToken(object())

        # This is used to describe tensors. It needs to be common across the
        # pickle so that duplicates and views are properly handled.
        self._meta_tensor_describer = MetaTensorDescriber(copy_data=False)

    @override
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
        else:
            # We should never get a raw Node!
            assert not isinstance(obj, torch.fx.Node)
            if reduce := _TorchNumpyPickleData.reduce_helper(self, obj):
                return reduce

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
            return unpickler.load()


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
            return _SymNodePickleData.unpickle_sym_int, args
        else:
            raise NotImplementedError(f"Unhandled SymNode type {type(obj)}")

    def __init__(self, node: SymNode) -> None:
        self.expr = node._expr
        self.shape_env = node.shape_env
        self.pytype = node.pytype
        self.hint = node._hint

    def _to_sym_node(self) -> SymNode:
        from torch.fx.experimental.sym_node import SymNode

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

        def with_fake(
            make_meta_t: Callable[[], torch.Tensor], device: Union[torch.device, str]
        ) -> FakeTensor:
            with no_dispatch():
                return FakeTensor(
                    unpickle_state.fake_mode,
                    make_meta_t(),
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

        assert np == getattr(importlib.import_module(mod), name)
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
        self.meta = node.meta

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

        name = torch.fx.Node._pretty_print_target(op)
        if isinstance(op, torch._ops.OpOverload):
            return cls._pickle_op(name, _OpOverloadPickleData, options)
        elif isinstance(op, torch._ops.OpOverloadPacket):
            return cls._pickle_op(name, _OpOverloadPacketPickleData, options)
        elif name.startswith(("builtins.", "math.", "torch.")):
            root, detail = name.split(".", 1)
            return _OpBuiltinPickleData(root, detail)
        elif name.startswith("operator."):
            _, detail = name.split(".", 1)
            return _OpOperatorPickleData(detail)
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


class _OpBuiltinPickleData(_OpPickleData):
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
        else:
            raise NotImplementedError


class _OpOperatorPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        import operator

        return self._getattr_by_name(operator, self.name)


class _GraphPickleData:
    def __init__(self, graph: torch.fx.Graph, options: Options) -> None:
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras

        nodes: dict[torch.fx.Node, _NodePickleData] = {}
        for node in graph.nodes:
            nodes[node] = _NodePickleData(node, nodes, options)
        self.nodes = tuple(nodes.values())

        # Unpickled variables:
        # self._used_names = graph._used_names
        # -- self._insert = self._root.prepend
        # self._len = graph._len
        # self._graph_namespace = graph._graph_namespace
        # self._owning_module = graph._owning_module
        # self._codegen = graph._codegen
        # self._co_fields: Dict[str, Any] = graph._co_fields
        # -- self._find_nodes_lookup_table = _FindNodesLookupTable()

    def unpickle(
        self, gm: torch.fx.GraphModule, unpickle_state: _UnpickleState
    ) -> torch.fx.Graph:
        graph = torch.fx.Graph(gm, self.tracer_cls, self.tracer_extras)

        nodes: dict[_NodePickleData, torch.fx.Node] = {}
        for nd in self.nodes:
            nodes[nd] = nd.unpickle(graph, nodes, unpickle_state)

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
