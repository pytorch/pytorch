import importlib
import io
import pickle
import typing
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from typing_extensions import override

import torch
import torch.utils._pytree as pytree
from torch._guards import TracingContext
from torch._subclasses.fake_tensor import (  # extract_tensor_metadata,
    FakeTensor,
    FakeTensorMode,
    Tensor,
)
from torch._subclasses.meta_utils import (
    MetaConverter,
    MetaTensorDesc,
    MetaTensorDescriber,
)
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


class _ShapeEnvPickleData:
    data: Dict[str, object]

    def __init__(self, env: ShapeEnv) -> None:
        # In theory pickle should recognize that a given ShapeEnv was already
        # pickled and reuse the resulting _ShapeEnvPickleData (so two objects
        # pointing at the same ShapeEnv get the same ShapeEnv out).
        assert not env._translation_validation_enabled
        self.data = env.__dict__.copy()
        del self.data["tracked_fakes"]
        del self.data["fake_tensor_cache"]

    def unpickle(self, unpickle_state: "_UnpickleState") -> ShapeEnv:
        # Fill in the existing ShapeEnv rather than creating a new one
        assert unpickle_state.fake_mode
        assert unpickle_state.fake_mode.shape_env

        for k, v in self.data.items():
            setattr(unpickle_state.fake_mode.shape_env, k, v)

        return unpickle_state.fake_mode.shape_env


class _SymNodePickleData:
    def __init__(self, node: SymNode) -> None:
        self.expr = node._expr
        self.shape_env = node.shape_env
        self.pytype = node.pytype
        self.hint = node._hint

    def _to_sym_node(self) -> SymNode:
        from torch.fx.experimental.sym_node import SymNode

        assert self.shape_env is not None
        return SymNode(self.expr, self.shape_env, self.pytype, self.hint)

    def unpickle_sym_int(self, unpickle_state: "_UnpickleState") -> torch.SymInt:
        return torch.SymInt(self._to_sym_node())


class _TensorPickleData:
    metadata: MetaTensorDesc[FakeTensor]

    def __init__(self, describer: MetaTensorDescriber, t: Tensor) -> None:
        # THINGS TO WORRY ABOUT:
        # 1. Need to make sure that two tensors with the same id end up with the
        #    same id on the other side of the wire.
        self.metadata = describer.describe_tensor(t)
        self.metadata.fake_mode = None
        self.metadata.view_func = bool(self.metadata.view_func)  # type: ignore[assignment]
        for k in MetaTensorDesc._UNSERIALIZABLE:
            if k in ("fake_mode", "view_func"):
                continue
            assert (
                getattr(self.metadata, k) is None
            ), f"not None: {k}: {getattr(self.metadata, k)}"

    def unpickle(self, unpickle_state: "_UnpickleState") -> Tensor:
        # TODO: make common w/ _output_from_cache_entry() in fake_tensor.py?
        self.metadata.fake_mode = unpickle_state.fake_mode
        if self.metadata.view_func:
            self.metadata.view_func = FakeTensor._view_func_unsafe  # type: ignore[assignment]
        else:
            self.metadata.view_func = None

        def with_fake(builder: Callable[[], torch.Tensor]) -> FakeTensor:
            with unpickle_state.fake_mode:
                return typing.cast(FakeTensor, builder())

        return unpickle_state.meta_converter.meta_tensor(
            self.metadata,
            unpickle_state.fake_mode.shape_env,
            with_fake,
            None,
            None,
            None,
        )

        with unpickle_state.fake_mode:
            empty = torch.empty_strided(
                self.metadata.shape,  # type: ignore[arg-type]
                self.metadata.stride,  # type: ignore[arg-type]
                dtype=self.metadata.dtype,
                layout=self.metadata.layout,
                device=self.metadata.device,
                requires_grad=self.metadata.requires_grad,
            )

        # TODO: Weird storage stuff?

        return empty


class _TorchNumpyPickleData:
    def __init__(self, mod: str, name: str) -> None:
        self.mod = mod
        self.name = name

    def unpickle(self, unpickle_state: "_UnpickleState") -> Callable[..., object]:
        np = getattr(importlib.import_module(self.mod), self.name)
        return torch._dynamo.variables.misc.get_np_to_tnp_map()[np]

    @staticmethod
    def from_object(tnp: object) -> Optional["_TorchNumpyPickleData"]:
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
        return _TorchNumpyPickleData(mod, name)


class _GraphModulePickleData:
    def __init__(self, gm: torch.fx.GraphModule) -> None:
        if isinstance(gm, torch.fx._lazy_graph_module._LazyGraphModule):
            python_code = gm._real_recompile()
        else:
            python_code = gm.recompile()
        self.gm_dict = gm.__dict__.copy()
        del self.gm_dict["_graph"]
        self.graph = _GraphPickleData(gm._graph)

    def unpickle(self, unpickle_state: "_UnpickleState") -> torch.fx.GraphModule:
        gm = torch.fx.GraphModule.__new__(torch.fx.GraphModule)
        gm.__dict__ = self.gm_dict
        gm._graph = self.graph.unpickle(gm, unpickle_state)
        return gm


class _NodePickleData:
    def __init__(
        self, node: torch.fx.Node, mapping: Dict[torch.fx.Node, "_NodePickleData"]
    ) -> None:
        self.args = pytree.tree_map_only(torch.fx.Node, lambda n: mapping[n], node.args)
        self.kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda n: mapping[n], node.kwargs
        )
        # -- self.graph = node.graph
        self.name = node.name
        self.op = node.op
        self.target = _OpPickleData.pickle(node.target)
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
        mapping: Dict["_NodePickleData", torch.fx.Node],
        unpickle_state: "_UnpickleState",
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
        cls, pickler: "_SubprocPickler", op: object
    ) -> Tuple[Callable[..., Any], Tuple[Any, ...]]:
        result = cls.pickle(op)
        return (result.unpickle, (pickler._unpickle_state,))

    @classmethod
    def pickle(cls, op: object) -> "_OpPickleData":
        if isinstance(op, str):
            return _OpStrPickleData(op)

        name = torch.fx.Node._pretty_print_target(op)
        if isinstance(op, torch._ops.OpOverload):
            return cls._pickle_op(name, _OpOverloadPickleData)
        elif isinstance(op, torch._ops.OpOverloadPacket):
            return cls._pickle_op(name, _OpOverloadPacketPickleData)
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
            Type["_OpOverloadPickleData"], Type["_OpOverloadPacketPickleData"]
        ],
    ) -> "_OpPickleData":
        if not name.startswith("torch.ops.aten"):  # TODO: What's the full list?
            from torch._inductor.codecache import BypassFxGraphCache

            raise BypassFxGraphCache(f"Unable to pickle non-standard op: {name}")
        return datacls(name)

    @abstractmethod
    def unpickle(self, unpickle_state: "_UnpickleState") -> object:
        pass

    @staticmethod
    def _lookup_by_name(name: str) -> object:
        mod, rest = name.split(".", 1)
        root = globals()[mod]
        while "." in rest:
            mod, rest = rest.split(".", 1)
            root = getattr(root, mod)
        return getattr(root, rest)


class _OpStrPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: "_UnpickleState") -> str:
        return self.name


class _OpOverloadPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: "_UnpickleState") -> torch._ops.OpOverload:
        obj = self._lookup_by_name(self.name)
        assert isinstance(obj, torch._ops.OpOverload)
        return obj


class _OpOverloadPacketPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: "_UnpickleState") -> torch._ops.OpOverloadPacket:
        obj = self._lookup_by_name(self.name)
        assert isinstance(obj, torch._ops.OpOverloadPacket)
        return obj


class _OpBuiltinPickleData(_OpPickleData):
    def __init__(self, root: str, name: str) -> None:
        self.root = root
        self.name = name

    def unpickle(self, unpickle_state: "_UnpickleState") -> object:
        if self.root == "builtins":
            return __builtins__.get(self.name)  # type: ignore[attr-defined]
        elif self.root == "math":
            import math

            return getattr(math, self.name)
        elif self.root == "torch":
            return getattr(torch, self.name)
        else:
            raise NotImplementedError


class _OpOperatorPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: "_UnpickleState") -> object:
        import operator

        return getattr(operator, self.name)


class _GraphPickleData:
    def __init__(self, graph: torch.fx.Graph) -> None:
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras

        nodes: Dict[torch.fx.Node, _NodePickleData] = {}
        for node in graph.nodes:
            nodes[node] = _NodePickleData(node, nodes)
        self.nodes = tuple(nodes.values())

        # self._used_names = graph._used_names
        # -- self._insert = self._root.prepend
        # self._len = graph._len
        # self._graph_namespace = graph._graph_namespace
        # self._owning_module = graph._owning_module
        # self._codegen = graph._codegen
        # self._co_fields: Dict[str, Any] = graph._co_fields
        # -- self._find_nodes_lookup_table = _FindNodesLookupTable()

    def unpickle(
        self, gm: torch.fx.GraphModule, unpickle_state: "_UnpickleState"
    ) -> torch.fx.Graph:
        graph = torch.fx.Graph(gm, self.tracer_cls, self.tracer_extras)

        nodes: Dict[_NodePickleData, torch.fx.Node] = {}
        for nd in self.nodes:
            nodes[nd] = nd.unpickle(graph, nodes, unpickle_state)

        return graph


class _TracingContextPickleData:
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

    def unpickle(self, unpickle_state: "_UnpickleState") -> TracingContext:
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


class _SubprocPickler(pickle.Pickler):
    def __init__(self, file: io.BytesIO) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)

        # This abomination is so we can pass external decoding state to the
        # unpickler functions. We serialize _unpickle_state as a persistent
        # external item and when we deserialize it we return the "known" state
        # object.
        self._unpickle_state = object()
        self._meta_tensor_describer = MetaTensorDescriber(copy_data=False)

    @override
    def reducer_override(
        self, obj: object
    ) -> Tuple[Callable[..., Any], Tuple[Any, ...]]:
        if isinstance(obj, FakeTensor):
            return (
                _TensorPickleData.unpickle,
                (
                    _TensorPickleData(self._meta_tensor_describer, obj),
                    self._unpickle_state,
                ),
            )
        elif isinstance(obj, torch.fx.GraphModule):
            return (
                _GraphModulePickleData.unpickle,
                (_GraphModulePickleData(obj), self._unpickle_state),
            )
        elif isinstance(obj, (torch._ops.OperatorBase, torch._ops.OpOverloadPacket)):
            return _OpPickleData.reduce_helper(self, obj)
        elif isinstance(obj, ShapeEnv):
            return (
                _ShapeEnvPickleData.unpickle,
                (_ShapeEnvPickleData(obj), self._unpickle_state),
            )
        elif isinstance(obj, torch.SymInt):
            return (
                _SymNodePickleData.unpickle_sym_int,
                (_SymNodePickleData(obj.node), self._unpickle_state),
            )
        elif isinstance(obj, torch._guards.TracingContext):
            return (
                _TracingContextPickleData.unpickle,
                (_TracingContextPickleData(obj), self._unpickle_state),
            )
        else:
            # We should never get a raw Node!
            assert not isinstance(obj, torch.fx.Node)

            if data := _TorchNumpyPickleData.from_object(obj):
                assert data.unpickle(self._unpickle_state) == obj  # type: ignore[arg-type]
                return (_TorchNumpyPickleData.unpickle, (data, self._unpickle_state))

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
    def dumps(cls, obj: object) -> bytes:
        """
        Pickle an object.
        """
        with io.BytesIO() as stream:
            pickler = cls(stream)
            pickler.dump(obj)
            return stream.getvalue()


class _UnpickleState:
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        self.fake_mode = fake_mode
        self.meta_converter: MetaConverter[FakeTensor] = MetaConverter()


class _SubprocUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, unpickle_state: _UnpickleState) -> None:
        super().__init__(stream)
        self._unpickle_state = unpickle_state

    @classmethod
    def loads(cls, data: bytes, unpickle_state: _UnpickleState) -> object:
        with io.BytesIO(data) as stream:
            unpickler = cls(stream, unpickle_state)
            return unpickler.load()

    @override
    def persistent_load(self, pid: object) -> object:
        if pid == "unpickle_state":
            return self._unpickle_state
        else:
            raise pickle.UnpicklingError("Invalid persistent ID")
