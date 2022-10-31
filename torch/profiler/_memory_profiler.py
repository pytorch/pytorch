import collections
import dataclasses
from typing import (
    Any,
    cast,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
    _EventType,
    _ExtraFields_Allocation,
    _ExtraFields_TorchOp,
    _ProfilerEvent,
    _TensorMetadata,
    RecordScope,
)
from torch.profiler import _utils


@dataclasses.dataclass
class _Storage:
    """Bundle storage pointer and id.

    All profiling logic should use `allocation_id`, however it is useful to
    print storage pointers for debugging and unit tests sometimes look up
    values using the storage data pointer of a live Tensor."""

    ptr: int
    allocation_id: int

    def __repr__(self) -> str:
        return f"{hex(self.ptr):>18} ({self.allocation_id})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Storage) and self.allocation_id == other.allocation_id

    def __hash__(self) -> int:
        return hash(self.allocation_id)


@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey:
    """Hashable identifier for a storage which has been asigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """

    id: int
    storage: _Storage
    device: torch.device

    def __repr__(self) -> str:
        return f"id={self.id}: {repr(self.storage):<24} ({self.device})"

    def __lt__(self, other: "TensorKey") -> bool:
        return self._as_sortable < other._as_sortable

    @staticmethod
    def _make(
        tensor_id: Optional[int],
        storage_ptr: Optional[int],
        allocation_id: Optional[int],
        device: torch.device,
    ) -> Optional["TensorKey"]:
        if (
            tensor_id is not None
            and storage_ptr is not None
            and allocation_id is not None
        ):
            return TensorKey(tensor_id, _Storage(storage_ptr, allocation_id), device)
        return None

    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> Optional["TensorKey"]:
        return cls._make(alloc.id, alloc.ptr, alloc.allocation_id, alloc.device)

    @classmethod
    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional["TensorKey"]:
        if t is not None:
            return cls._make(t.id, t.storage_data_ptr, t.allocation_id, t.device)
        return None

    @property
    def _as_sortable(self) -> Tuple[int, int, str, int]:
        return self.id, self.storage.allocation_id, self.device.type, self.device.index


def extract_gradients(
    node: _ProfilerEvent,
) -> Iterator[Tuple[Optional[TensorKey], TensorKey]]:
    children = node.children

    # AccumulateGrad is used in the Autograd engine to handle gradient updates.
    # There are two possible cases:
    # 1) This is a newly created gradient Tensor. In that case there is nothing
    #    to accumulate, so autograd simply detaches the Tensor.
    #
    # 2) There is a preexisting gradient Tensor and we need to add the newly
    #    computed update. This is done with an in-place add (aten::add_) op.
    #    (The underscore suffix denotes "in-place".)
    if (
        node.typed[0] == _EventType.TorchOp
        and node.typed[1].scope == RecordScope.BACKWARD_FUNCTION
        # TODO(robieta): Move away from load bearing names
        and node.name == "torch::autograd::AccumulateGrad"
        and children
        and children[0].typed[0] == _EventType.TorchOp
        and children[0].name in ("aten::detach", "aten::add_")
        and children[0].typed[1].inputs
        and isinstance(children[0].typed[1].inputs[0], _TensorMetadata)
    ):
        key = TensorKey.from_tensor(children[0].typed[1].inputs[0])
        if key:
            yield None, key

    # We directly instrument `torch.nn.Module` and `torch.optim.Optimizer`
    # NOTE: The values captured by the python tracer are cached; they can be
    #       used to build up labels but do not imply that a Tensor was live at
    #       a particular time.
    elif node.typed[0] == _EventType.PyCall:
        typed_fields = node.typed[1]
        assert typed_fields.module is None or typed_fields.optimizer is None
        if typed_fields.module is not None:
            for _, p, p_grad in typed_fields.module.parameters:
                p_grad_key = TensorKey.from_tensor(p_grad)
                if p_grad_key is not None:
                    yield TensorKey.from_tensor(p), p_grad_key

        if typed_fields.optimizer is not None:
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                p_grad_key = TensorKey.from_tensor(p_grad)
                if p_grad_key is not None:
                    yield TensorKey.from_tensor(p), p_grad_key


class SchemaMatcher:
    """Lookup operator schema based on profiled name.

    When profiling we record the operator's name but not the schema. However
    some analysis requires that information. Fortunately we can look up
    registered schema from the recorded name. We do not, however, record the
    overload and so we must compare the profiled arguments with all overloads
    to determine viable matches.

    Note: Once https://github.com/pytorch/pytorch/issues/78871 is completed
    this code will be obsolete.
    """

    @classmethod
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> Tuple[Optional[bool], ...]:
        """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
        mutable: Optional[List[bool]] = None
        for schema in cls.match_schemas(t):
            mutable = mutable or [False for _ in schema.arguments]
            for i, arg in enumerate(schema.arguments):
                mutable[i] |= getattr(arg.alias_info, "is_write", False)

        return tuple(mutable or (None for _ in t.inputs))

    @classmethod
    def match_schemas(cls, t: _ExtraFields_TorchOp) -> Tuple[FunctionSchema, ...]:
        signature = tuple(
            # Tensor
            TensorKey.from_tensor(i) if isinstance(i, _TensorMetadata)
            #
            # TensorList
            else [TensorKey.from_tensor(j) for j in i] if isinstance(i, list)
            #
            # Scalar and uncaptured inputs.
            else i
            for i in t.inputs
        )

        def matches(schema) -> bool:
            return len(schema.arguments) == len(signature) and all(
                cls._types_match(observed, schema_arg.type)
                for observed, schema_arg in zip(signature, schema.arguments)
            )

        return tuple(s for s in cls.lookup_schemas(t.name) or () if matches(s))

    @classmethod
    def _types_match(cls, observed, schema_type) -> bool:
        if isinstance(schema_type, torch._C.OptionalType):
            schema_type = schema_type.getElementType()
            return observed is None or cls._types_match(observed, schema_type)

        if isinstance(schema_type, torch._C.AnyType):
            return True

        if schema_type.isSubtypeOf(torch._C.ListType.ofTensors()):
            return isinstance(observed, list) and all(
                isinstance(i, TensorKey) for i in observed
            )

        type_map: Tuple[Tuple[Any, Union[type, Tuple[type, ...]]], ...] = (
            (torch._C.TensorType, TensorKey),
            (torch._C.NoneType, type(None)),
            (torch._C.BoolType, bool),
            (torch._C.IntType, int),
            (torch._C.FloatType, float),
            (torch._C.ComplexType, complex),
            (torch._C.NumberType, (bool, int, float, complex)),
        )

        for jit_type, py_types in type_map:
            if isinstance(schema_type, jit_type):
                return isinstance(observed, py_types)

        # Profiler only records a subset of possible argument types. If we
        # reach this point then the schema must call for a type that profiler
        # does not record. Thus, the schema can only be a match if `observed`
        # is also None.
        return observed is None

    @staticmethod
    def lookup_schemas(name: str) -> Optional[Tuple[FunctionSchema, ...]]:
        # TODO(robieta):
        #   _jit_get_schemas_for_operator is quite expensive. (~100us / call)
        #   Consider adding `functools.lru_cache` if that becomes an issue.

        try:
            # Schema lookup will throw if `name` is malformed. (For example,
            # schemas must be namespaced and schema lookup will fail if name
            # does not include "::".) We simply catch the exception and return
            # `None` to denote that `name` cannot be an operator name.
            #
            # Note that record_function annotations also go through this path,
            # so it is expected that some names will not correspond to PyTorch
            # operators.
            return tuple(torch._C._jit_get_schemas_for_operator(name))
        except RuntimeError:
            return None


class OpTree:
    def __init__(self, result: _ProfilerResult) -> None:
        self._root_nodes = result.experimental_event_tree()
        self._sorted_nodes = tuple(sorted(self.dfs(), key=lambda x: x.start_time_ns))

    def dfs(self, *args, **kwargs) -> Iterator[_ProfilerEvent]:
        yield from _utils.traverse_dfs(self._root_nodes, *args, **kwargs)

    @property
    def sorted_nodes(self) -> Tuple[_ProfilerEvent, ...]:
        return self._sorted_nodes


@dataclasses.dataclass()
class DataFlowEdge:
    input_version: Optional[int] = None
    mutated: Optional[bool] = False

    @property
    def is_allocation(self) -> bool:
        return self.input_version is None

    @property
    def is_deletion(self) -> bool:
        return self.mutated is None


class DataFlowNode:
    def __init__(self, event: _ProfilerEvent, graph: "DataFlowGraph") -> None:
        self._event = event
        self._graph = graph
        self._edges: Dict[TensorKey, DataFlowEdge] = self._determine_edges()

        for key, edge in self._edges.items():
            if edge.mutated and not edge.is_allocation:
                self._graph.bump(key)

        # Make sure the version bumping behavior matches what we expect.
        versions = {k: (v, self._graph.lookup(k)) for k, v in self.outputs.items()}
        assert all(i == j for i, j in versions.values()), f"{versions}, {self._edges}"

    def _determine_edges(self) -> Dict[TensorKey, DataFlowEdge]:
        subtree = tuple(_utils.traverse_dfs([self._event]))

        # Start by populating edges from op inputs and outputs.
        mutable_by_key: Dict[Optional[TensorKey], Set[Optional[bool]]] = {}
        for op in (i.typed[1] for i in subtree if i.typed[0] == _EventType.TorchOp):
            for op_input, mutable in zip(
                op.inputs, SchemaMatcher.inputs_are_mutable(op)
            ):
                # Tensor
                if isinstance(op_input, _TensorMetadata):
                    key = TensorKey.from_tensor(op_input)
                    mutable_by_key.setdefault(key, set()).add(mutable)

                # TensorList
                elif isinstance(op_input, list):
                    for op_input_i in op_input:
                        key = TensorKey.from_tensor(op_input_i)
                        mutable_by_key.setdefault(key, set()).add(mutable)

        edges: DefaultDict[Optional[TensorKey], DataFlowEdge]
        edges = collections.defaultdict(DataFlowEdge)
        for key, mutable_set in mutable_by_key.items():
            if key is not None:
                edges[key].input_version = self._graph.lookup(key) if key else -1

                # We consider an op to be mutated if we encounter a schema where it
                # is a mutable argument OR if it is ambiguous. (We never explicitly
                # see it in any schema.)
                mutated = (True in mutable_set) or (tuple(mutable_set) == (None,))
                edges[key].mutated = mutated

        # Then handle deletions. Note that deleting a Tensor implicitly adds
        # it as an input edge.
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size < 0:
                key = TensorKey.from_allocation(i.typed[1])
                edge = edges[key]
                assert key is None or edge.mutated is not None, f"Double delete: {key}"
                edge.mutated = None
                edge.input_version = self._graph.lookup(key) if key else -1

        # And finally handle allocations. This step must be last, because the
        # previous two steps optimistically add input edges.
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size > 0:
                edges[TensorKey.from_allocation(i.typed[1])].input_version = None

        # We don't need to sort the inputs, but it makes debugging and unit tests nicer.
        return dict(sorted((k, v) for k, v in edges.items() if k is not None))

    @property
    def inputs(self) -> Dict[TensorKey, Tuple[bool, int]]:
        return {
            # MyPy can't see through `is_allocation` to know that
            # `v.input_version` is not None.
            k: (bool(v.mutated), cast(int, v.input_version))
            for k, v in self._edges.items()
            if not v.is_allocation
        }

    @property
    def outputs(self) -> Dict[TensorKey, int]:
        return {
            k: 0 if v.input_version is None else v.input_version + 1
            for k, v in self._edges.items()
            if (v.is_allocation and not v.is_deletion) or v.mutated
        }

    @property
    def intermediates(self) -> Tuple[TensorKey, ...]:
        return tuple(
            k for k, v in self._edges.items() if v.is_allocation and v.is_deletion
        )

    @property
    def start_time(self) -> int:
        return self._event.start_time_ns


class DataFlowGraph:
    def __init__(self, op_tree: OpTree) -> None:
        self._op_tree = op_tree
        self._leaf_events = self._extract_leaf_events(op_tree)
        self._active_version: Dict[TensorKey, Optional[int]] = {}
        self._flow_nodes = [DataFlowNode(e, self) for e in self.leaf_events]
        self._flow_nodes.sort(key=lambda x: x.start_time)
        self.validate()

    @property
    def flow_nodes(self) -> Tuple[DataFlowNode, ...]:
        return tuple(self._flow_nodes)

    def validate(self):
        # Check that each (Tensor, version) pair has a unique creation node
        outputs: Set[Tuple[TensorKey, int]] = set()
        for node in self.flow_nodes:
            node_outputs = set(node.outputs.items())
            duplicates = outputs & node_outputs
            assert not duplicates, f"{node._event.name} {node._edges} {duplicates}"
            outputs |= node_outputs

        # And check that `self._nodes` forms a valid topologically sorted DAG.
        tensor_versions: Dict[TensorKey, int] = {}
        for node in self.flow_nodes:
            for key, (_, version) in node.inputs.items():
                expected = tensor_versions.get(key, 0)
                assert expected == version, (expected, version)

            for key, version in node.outputs.items():
                prior_version = tensor_versions.get(key, version)
                assert version >= prior_version, (version, prior_version)
                tensor_versions[key] = version

    @property
    def leaf_events(self) -> Tuple[_ProfilerEvent, ...]:
        return self._leaf_events

    @staticmethod
    def _extract_leaf_events(op_tree: OpTree) -> Tuple[_ProfilerEvent, ...]:
        """Partially traverse the op tree and extract top level ops.

        Consider the following code:
        ```
        with record_function("My annotation"):
            x.zero_()
            y.zero_()
        ```

        The op tree (assuming no Autograd) will look like:
          <Python context>
            TorchOp: "My annotation"
              TorchOp: zero_
                TorchOp: fill_
              TorchOp: zero_
                TorchOp: fill_

        The recursive structure of operator calls makes data flow unwieldy.
        In order to simplify analysis we would like to select the highest level
        ops to represent in the graph. In this case those are the `zero_` ops;
        the fact that `fill_` is called is an implementation detail. We also
        do not want to group everything under "My annotation" as this could
        create overly coarse bundles and lose critical semantics.

        To address this issue we walk over the graph and select the topmost
        torch ops ** which match at least one operator schema **. These form
        the leaves of the first pass through the op tree. (As well as any
        allocations or frees which do are not part of a kernel.) These events
        form the logical nodes in our data flow graph.
        """

        leaf_events: List[_ProfilerEvent] = []

        def leaf_op(e: _ProfilerEvent) -> bool:
            return e.typed[0] == _EventType.TorchOp and (
                e.typed[1].scope == RecordScope.BACKWARD_FUNCTION
                or bool(SchemaMatcher.match_schemas(e.typed[1]))
            )

        def children_fn(e: _ProfilerEvent):
            if leaf_op(e) or e.tag == _EventType.Allocation:
                leaf_events.append(e)
                return []

            return e.children

        for _ in op_tree.dfs(children_fn=children_fn):
            pass

        return tuple(sorted(leaf_events, key=lambda x: x.start_time_ns))

    def lookup(self, key: TensorKey) -> int:
        version = self._active_version.setdefault(key, 0)
        assert version is not None
        return version

    def bump(self, key: TensorKey) -> None:
        prior_version = self._active_version.get(key, None)
        assert prior_version is not None
        self._active_version[key] = prior_version + 1

    def delete(self, key: TensorKey) -> None:
        assert self._active_version.setdefault(key, 0) is not None
        self._active_version[key] = None


class MemoryProfile:
    def __init__(self, result: _ProfilerResult) -> None:
        self._op_tree = OpTree(result)
        self._data_flow_graph = DataFlowGraph(self._op_tree)
