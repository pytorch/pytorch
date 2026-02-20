# mypy: allow-untyped-defs
import collections
import dataclasses
import enum
import itertools as it
import logging
from collections.abc import Iterator
from typing import Any, cast, Literal, Optional

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
from torch._utils import _element_size
from torch.profiler import _utils


KeyAndID = tuple["Key", int]
TensorAndID = tuple["TensorKey", int]

log = logging.getLogger(__name__)


class Category(enum.Enum):
    INPUT = enum.auto()
    TEMPORARY = enum.auto()
    ACTIVATION = enum.auto()
    GRADIENT = enum.auto()
    AUTOGRAD_DETAIL = enum.auto()
    PARAMETER = enum.auto()
    OPTIMIZER_STATE = enum.auto()


_CATEGORY_TO_COLORS = {
    Category.PARAMETER: "darkgreen",
    Category.OPTIMIZER_STATE: "goldenrod",
    Category.INPUT: "black",
    Category.TEMPORARY: "mediumpurple",
    Category.ACTIVATION: "red",
    Category.GRADIENT: "mediumblue",
    Category.AUTOGRAD_DETAIL: "royalblue",
    None: "grey",
}

_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}


class Action(enum.Enum):
    PREEXISTING = enum.auto()
    CREATE = enum.auto()
    INCREMENT_VERSION = enum.auto()
    DESTROY = enum.auto()


_ACTION_TO_INDEX = {i: i.value for i in Action}


@dataclasses.dataclass(eq=True, unsafe_hash=False, frozen=True)
class Key:
    device: torch.device


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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Storage) and self.allocation_id == other.allocation_id

    def __hash__(self) -> int:
        return hash(self.allocation_id)


@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey(Key):
    """Hashable identifier for a storage which has been assigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """

    id: int
    storage: _Storage

    def __repr__(self) -> str:
        return f"id={self.id}: {repr(self.storage):<24} ({self.device})"

    def __lt__(self, other: "TensorKey") -> bool:
        return self._as_sortable < other._as_sortable

    @staticmethod
    def _make(
        tensor_id: int | None,
        storage_ptr: int | None,
        allocation_id: int | None,
        device: torch.device,
    ) -> Optional["TensorKey"]:
        if (
            tensor_id is not None
            and storage_ptr is not None
            and allocation_id is not None
        ):
            return TensorKey(device, tensor_id, _Storage(storage_ptr, allocation_id))
        return None

    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> Optional["TensorKey"]:
        return cls._make(alloc.id, alloc.ptr, alloc.allocation_id, alloc.device)

    @classmethod
    def from_tensor(cls, t: _TensorMetadata | None) -> Optional["TensorKey"]:
        if t is not None:
            return cls._make(t.id, t.storage_data_ptr, t.allocation_id, t.device)
        return None

    @property
    def _as_sortable(self) -> tuple[int, int, str, int]:
        return self.id, self.storage.allocation_id, self.device.type, self.device.index


def _extract_parameters_and_gradients(
    node: _ProfilerEvent,
) -> Iterator[tuple[TensorKey | None, TensorKey | None]]:
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
        yield None, TensorKey.from_tensor(children[0].typed[1].inputs[0])

    # We directly instrument `torch.nn.Module` and `torch.optim.Optimizer`
    # NOTE: The values captured by the python tracer are cached; they can be
    #       used to build up labels but do not imply that a Tensor was live at
    #       a particular time.
    elif node.typed[0] == _EventType.PyCall:
        typed_fields = node.typed[1]
        if typed_fields.module is not None and typed_fields.optimizer is not None:
            raise AssertionError("module and optimizer cannot both be set")
        if typed_fields.module is not None:
            for _, p, p_grad in typed_fields.module.parameters:
                yield TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad)

        if typed_fields.optimizer is not None:
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                yield TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad)


def extract_parameters(node: _ProfilerEvent) -> Iterator[TensorKey]:
    for p, _p_grad in _extract_parameters_and_gradients(node):
        if p is not None:
            yield p


def extract_gradients(
    node: _ProfilerEvent,
) -> Iterator[tuple[TensorKey | None, TensorKey]]:
    for p, p_grad in _extract_parameters_and_gradients(node):
        if p_grad is not None:
            yield p, p_grad


def get_scopes(event: _ProfilerEvent | None) -> tuple[RecordScope, ...]:
    scopes = []
    while event:
        if event.typed[0] == _EventType.TorchOp:
            scopes.append(event.typed[1].scope)
        event = event.parent
    return tuple(scopes)


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
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> tuple[bool | None, ...]:
        """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
        mutable: list[bool] | None = None
        for schema in cls.match_schemas(t):
            mutable = mutable or [False for _ in schema.arguments]
            for i, arg in enumerate(schema.arguments):
                # pyrefly: ignore [unsupported-operation]
                mutable[i] |= getattr(arg.alias_info, "is_write", False)

        return tuple(mutable or (None for _ in t.inputs))

    @classmethod
    def match_schemas(cls, t: _ExtraFields_TorchOp) -> tuple[FunctionSchema, ...]:
        signature = tuple(
            # Tensor
            TensorKey.from_tensor(i)
            if isinstance(i, _TensorMetadata)
            #
            # TensorList
            else [TensorKey.from_tensor(j) for j in i]
            if isinstance(i, list)
            #
            # Scalar and uncaptured inputs.
            else i
            for i in t.inputs
        )

        def matches(schema) -> bool:
            return len(schema.arguments) == len(signature) and all(
                cls._types_match(observed, schema_arg.type)
                for observed, schema_arg in zip(
                    signature, schema.arguments, strict=True
                )
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

        type_map: tuple[tuple[Any, type | tuple[type, ...]], ...] = (
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
    def lookup_schemas(name: str) -> tuple[FunctionSchema, ...] | None:
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
            if "::" not in name:
                return None
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
    def sorted_nodes(self) -> tuple[_ProfilerEvent, ...]:
        return self._sorted_nodes


class SizeMap:
    def __init__(self, op_tree: OpTree) -> None:
        self._values: dict[TensorKey, int] = {}

        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.TorchOp:
                for t in self._flat_tensor_inputs(node.typed[1]):
                    self._update_values(t)

            elif node.typed[0] == _EventType.PyCall:
                typed_fields = node.typed[1]
                if (
                    typed_fields.module is not None
                    and typed_fields.optimizer is not None
                ):
                    raise AssertionError("module and optimizer cannot both be set")
                if typed_fields.module is not None:
                    for _, p, p_grad in typed_fields.module.parameters:
                        self._update_values(p)
                        self._update_values(p_grad)

                if typed_fields.optimizer is not None:
                    for p, p_grad, state in typed_fields.optimizer.parameters:
                        self._update_values(p)
                        self._update_values(p_grad)
                        for _, t in state:
                            self._update_values(t)

        allocations: dict[TensorKey, int] = {}
        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.Allocation:
                alloc_fields = node.typed[1]
                key = TensorKey.from_allocation(alloc_fields)
                if key:
                    new_size = abs(alloc_fields.alloc_size)
                    prior_size = allocations.setdefault(key, new_size)

                    # It is possible to resize Storage in PyTorch, however we
                    # key on data pointer so most resizes will be treated as a
                    # change in storage. The one corner case that cannot be
                    # handled is `realloc` which successfully resizes the
                    # storage. At time of writing this is not done anywhere in
                    # the core PyTorch codebase.
                    if prior_size != new_size:
                        delta = f"{prior_size} vs. {new_size}"
                        log.warning("Mismatch between allocation and free: %s", delta)

        self._values.update(allocations)

    def _update_values(self, t: _TensorMetadata | None) -> None:
        key = TensorKey.from_tensor(t)
        if key is not None and t is not None and t.layout == torch.strided:
            # Scalars are represented as zero dim Tensors
            n = max(
                i[0] * i[1] for i in zip(t.sizes or [1], t.strides or [1], strict=True)
            )

            num_bytes = n * _element_size(t.dtype)
            if num_bytes < 0:
                raise AssertionError(f"num_bytes must be non-negative, got {num_bytes}")
            self._values[key] = max(self._values.get(key, 0), num_bytes)

    @staticmethod
    def _flat_tensor_inputs(op: _ExtraFields_TorchOp) -> Iterator[_TensorMetadata]:
        for i in op.inputs:
            if isinstance(i, _TensorMetadata):
                yield i
            elif isinstance(i, list):
                yield from i

    def __getitem__(self, key: TensorKey):
        return self._values[key]


@dataclasses.dataclass()
class DataFlowEdge:
    input_version: int | None = None
    mutated: bool | None = False

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
        self._edges: dict[TensorKey, DataFlowEdge] = self._determine_edges()

        for key, edge in self._edges.items():
            if edge.mutated and not edge.is_allocation:
                self._graph.bump(key)

        # Make sure the version bumping behavior matches what we expect.
        versions = {k: (v, self._graph.lookup(k)) for k, v in self.outputs.items()}
        if not all(i == j for i, j in versions.values()):
            raise AssertionError(f"version mismatch: {versions}, {self._edges}")

    def _determine_edges(self) -> dict[TensorKey, DataFlowEdge]:
        subtree = tuple(_utils.traverse_dfs([self._event]))

        # Start by populating edges from op inputs and outputs.
        mutable_by_key: dict[TensorKey | None, set[bool | None]] = {}
        for op in (i.typed[1] for i in subtree if i.typed[0] == _EventType.TorchOp):
            for op_input, mutable in zip(
                op.inputs, SchemaMatcher.inputs_are_mutable(op), strict=True
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

        edges: collections.defaultdict[TensorKey | None, DataFlowEdge]
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
                if key is not None and edge.mutated is None:
                    raise AssertionError(f"Double delete: {key}")
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
    def inputs(self) -> dict[TensorKey, tuple[bool, int]]:
        return {
            # MyPy can't see through `is_allocation` to know that
            # `v.input_version` is not None.
            k: (bool(v.mutated), cast(int, v.input_version))
            for k, v in self._edges.items()
            if not v.is_allocation
        }

    @property
    def outputs(self) -> dict[TensorKey, int]:
        return {
            k: 0 if v.input_version is None else v.input_version + 1
            for k, v in self._edges.items()
            if (v.is_allocation and not v.is_deletion) or v.mutated
        }

    @property
    def intermediates(self) -> tuple[TensorKey, ...]:
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
        self._active_version: dict[TensorKey, int | None] = {}
        self._flow_nodes = [DataFlowNode(e, self) for e in self.leaf_events]
        self._flow_nodes.sort(key=lambda x: x.start_time)
        self.validate()

    @property
    def flow_nodes(self) -> tuple[DataFlowNode, ...]:
        return tuple(self._flow_nodes)

    def validate(self) -> None:
        # Check that each (Tensor, version) pair has a unique creation node
        outputs: set[tuple[TensorKey, int]] = set()
        for node in self.flow_nodes:
            node_outputs = set(node.outputs.items())
            duplicates = outputs & node_outputs
            if duplicates:
                raise AssertionError(
                    f"duplicate outputs: {node._event.name} {node._edges} {duplicates}"
                )
            outputs |= node_outputs

        # And check that `self._nodes` forms a valid topologically sorted DAG.
        tensor_versions: dict[TensorKey, int] = {}
        for node in self.flow_nodes:
            for key, (_, version) in node.inputs.items():
                expected = tensor_versions.get(key, 0)
                if expected != version:
                    raise AssertionError(
                        f"version mismatch for input: expected {expected}, got {version}"
                    )

            for key, version in node.outputs.items():
                prior_version = tensor_versions.get(key, version)
                if version < prior_version:
                    raise AssertionError(
                        f"version regression: {version} < {prior_version}"
                    )
                tensor_versions[key] = version

    @property
    def leaf_events(self) -> tuple[_ProfilerEvent, ...]:
        return self._leaf_events

    @staticmethod
    def _extract_leaf_events(op_tree: OpTree) -> tuple[_ProfilerEvent, ...]:
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

        leaf_events: list[_ProfilerEvent] = []

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
        if version is None:
            raise AssertionError(f"version for key {key} is None")
        return version

    def bump(self, key: TensorKey) -> None:
        prior_version = self._active_version.get(key, None)
        if prior_version is None:
            raise AssertionError(f"prior_version for key {key} is None")
        self._active_version[key] = prior_version + 1

    def delete(self, key: TensorKey) -> None:
        if self._active_version.setdefault(key, 0) is None:
            raise AssertionError(f"cannot delete key {key}, already deleted")
        self._active_version[key] = None


@dataclasses.dataclass
class CategoryElement:
    by_id: Category | None = None
    by_key: dict[TensorKey, Category] = dataclasses.field(default_factory=dict)
    by_version: dict[TensorAndID, Category] = dataclasses.field(default_factory=dict)

    # Used by unit tests to check internals. (And consequently by
    # MemoryProfile.lookup) This should not be used in any other capacity.
    _by_id_keyset: set[TensorKey] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class CategoryDict:
    _values: collections.defaultdict[int, CategoryElement] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(CategoryElement)
    )

    def set_by_id(self, key: TensorKey, category: Category) -> None:
        self._values[key.id].by_id = category
        self._values[key.id]._by_id_keyset.add(key)

    def set_by_key(self, key: TensorKey, category: Category) -> None:
        self._values[key.id].by_key[key] = category

    def set_by_version(self, key: TensorKey, version: int, category: Category) -> None:
        self._values[key.id].by_version[(key, version)] = category

    def setdefault_by_version(
        self, key: TensorKey, version: int, category: Category
    ) -> None:
        self._values[key.id].by_version.setdefault((key, version), category)

    def get(self, key: Key, version: int) -> Category | None:
        if isinstance(key, Key) and not isinstance(key, TensorKey):
            return None
        element = self._values[key.id]
        return (
            element.by_id
            or element.by_key.get(key, None)
            or element.by_version.get((key, version), None)
        )


class MemoryProfile:
    def __init__(self, result: _ProfilerResult) -> None:
        self._op_tree = OpTree(result)
        self._data_flow_graph = DataFlowGraph(self._op_tree)
        self._size_map = SizeMap(self._op_tree)
        self._categories = CategoryDict()

        self._set_gradients_and_temporaries()
        self._set_parameters_using_python_tracer()
        self._set_inputs()
        self._set_parameters_using_data_flow()
        self._set_activations()
        self._set_optimizer_state()
        self._set_autograd_detail()

    @property
    def timeline(self) -> tuple[tuple[int, Action, KeyAndID, int], ...]:
        output: list[tuple[int, Action, KeyAndID, int]] = []
        allocation_times: dict[tuple[TensorKey, bool], int] = {}
        live_unknown: dict[tuple[int, torch.device], Literal[True]] = {}

        for event in self._op_tree.dfs():
            if event.typed[0] == _EventType.Allocation:
                alloc_fields = event.typed[1]
                alloc_size = alloc_fields.alloc_size
                is_allocation = alloc_size > 0
                t = event.start_time_ns

                tkey = TensorKey.from_allocation(alloc_fields)
                if tkey is not None:
                    allocation_times[(tkey, is_allocation)] = t

                else:
                    key = Key(alloc_fields.device)
                    ptr_and_device = (alloc_fields.ptr, key.device)
                    if is_allocation:
                        if ptr_and_device in live_unknown:
                            output.append(
                                (t, Action.INCREMENT_VERSION, (key, 0), alloc_size)
                            )
                        else:
                            live_unknown[ptr_and_device] = True
                            output.append((t, Action.CREATE, (key, 0), alloc_size))
                    else:
                        output.append((t, Action.DESTROY, (key, 0), -alloc_size))
                        if not live_unknown.pop(ptr_and_device, False):
                            output.append(
                                (-1, Action.PREEXISTING, (key, 0), -alloc_size)
                            )

        snapshot = self._category_snapshot()
        last_version = dict(sorted(snapshot.keys()))

        events: list[tuple[int, Action, TensorAndID]] = [
            (-1, Action.PREEXISTING, (key, version))
            for key, version in snapshot
            if (key, True) not in allocation_times and version == 0
        ]

        for node in self._data_flow_graph.flow_nodes:
            for key, edge in node._edges.items():
                if edge.is_allocation:
                    t = allocation_times[(key, True)]
                    events.append((t, Action.CREATE, (key, 0)))

                elif edge.mutated:
                    t = node._event.start_time_ns
                    version = edge.input_version
                    if version is None:
                        raise AssertionError(f"input_version is None for key {key}")
                    events.append((t, Action.INCREMENT_VERSION, (key, version)))

                if edge.is_deletion:
                    t = allocation_times[(key, False)]
                    events.append((t, Action.DESTROY, (key, last_version[key])))

        output.extend(
            (time, action, (key, version), self._size_map[key])
            for time, action, (key, version) in events
        )

        output.sort(key=lambda x: (x[0], x[1].value))
        return tuple(output)

    def _is_gradient(self, *args, **kwargs) -> bool:
        return self._categories.get(*args, **kwargs) == Category.GRADIENT

    def _category_snapshot(self) -> dict[TensorAndID, Category | None]:
        all_tensor_versions: set[TensorAndID] = set()

        for node in self._data_flow_graph.flow_nodes:
            all_tensor_versions.update(((k, v) for k, (_, v) in node.inputs.items()))
            all_tensor_versions.update((key, 0) for key in node.intermediates)
            all_tensor_versions.update(node.outputs.items())

        for i in self._categories._values.values():
            all_tensor_versions.update((key, 0) for key in i._by_id_keyset)

        return {
            (key, version): self._categories.get(key, version)
            for key, version in sorted(all_tensor_versions)
        }

    def _any_version_depends_on_gradient(self) -> set[int]:
        """Extract IDs of Tensors which depend or will depend on a gradient.

        Note that this weakened definition of "depends" requires us to loop
        over the data flow graph multiple times because it allows dependency
        information to flow backward through edges and removes the guarantee
        that nodes are topologically sorted. (Or indeed, even that a valid
        topological order exists.) Put another way, we have converted an
        acyclic data flow graph into a cyclic graph and we are attempting to
        partition cycles involving a gradient from the rest of the graph.
        """
        depends_on_gradient: set[int] = set()
        while True:
            start_size = len(depends_on_gradient)
            for node in self._data_flow_graph.flow_nodes:
                ids = tuple(
                    key.id
                    for key, (_, version) in node.inputs.items()
                    if self._categories.get(key, version)
                    in (Category.GRADIENT, Category.PARAMETER)
                    or key.id in depends_on_gradient
                )

                if ids:
                    depends_on_gradient.update(ids)

                    depends_on_gradient.update(key.id for key in node.outputs)

            # We are guaranteed to exit because there is a finite set of
            # TensorAndID pairs. In practice we do not expect to loop more than
            # three times: once to identify the core parameter update loop,
            # once to fold the first step into that loop, and a third time
            # where no new elements are added.
            if len(depends_on_gradient) == start_size:
                return depends_on_gradient

    def _set_gradients_and_temporaries(self) -> None:
        """Mark Tensors which are unambiguous and simple to reason about."""

        # Gradients are straightforward to detect. We directly check the
        # `.grad` property in the Python tracer, and we can detect any new
        # gradient Tensors from `AccumulateGrad` ops.
        for event in self._op_tree.dfs():
            for _, p_grad in extract_gradients(event):
                self._categories.set_by_id(p_grad, Category.GRADIENT)

        # Similarly, temporary Tensors are easy to identify and are useful to
        # flag since they can make memory use "spikier" than one would
        # otherwise expect.
        for node in self._data_flow_graph.flow_nodes:
            for i in node.intermediates:
                self._categories.set_by_key(i, Category.TEMPORARY)

    def _set_parameters_using_python_tracer(self) -> None:
        for event in self._op_tree.dfs():
            for p in extract_parameters(event):
                if p is not None:
                    self._categories.set_by_id(p, Category.PARAMETER)

    def _set_inputs(self) -> None:
        """Mark inputs based on which Tensors are updated using gradients.

        The process for differentiating between inputs and activations is more
        involved. Most Tensors in a training loop depend on at least one
        gradient: parameters depend on them through updates, and activations
        and optimizer state depend on them transitively through parameters.
        Critically, we do not need to know which Tensors are parameters to
        apply this method; we can simply walk the data flow graph to build the
        set of all values which depend on a gradient and then obtain the set
        of inputs from the conjugate set.

        There is, however, one hiccup. The first time we see a parameter is
        generally on the forward pass of the first step. We know from
        inspection of the data flow graph that v1 of that Tensor depends on
        a gradient (provided we profile an optimizer step), but not v0. To
        address this problem we weaken the definition of "depends on a
        gradient" to "any version of this Tensor depends on a gradient",
        which in turn strengthens the criteria for the input set enough to
        filter the activations in the forward pass of the first step."""

        # All of this analysis is predicated on using at least one training
        # step (or parameters from the python tracer) to partition the graph.
        # Absent that we cannot determine which Tensors are inputs and which
        # ones are part of the model.
        depends_on_gradient = self._any_version_depends_on_gradient()

        # We only want to annotate Tensors which actually contribute to the
        # model calculation.
        produces_gradient: set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            tensors = {(key, version) for key, (_, version) in node.inputs.items()}
            tensors |= node.outputs.items()
            if any(
                self._categories.get(*i) in (Category.GRADIENT, Category.PARAMETER)
                or i in produces_gradient
                for i in tensors
            ):
                produces_gradient |= tensors

        # Don't include Tensors created in the backward pass, as these are
        # generally Autograd implementation details rather than proper inputs.
        input_candidates = produces_gradient.copy()
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                input_candidates -= set(node.outputs.items())

        for key, version in input_candidates:
            if key.id not in depends_on_gradient:
                self._categories.setdefault_by_version(key, version, Category.INPUT)

    def _set_parameters_using_data_flow(self) -> None:
        """Deduce which Tensors are parameters.

        Consider the following code for the step of SGD with momentum
        (nesterov=False), where `d_p` is the gradient of `param` and `buf` is
        the momentum buffer.
        ```
          buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          d_p = buf
          param.add_(d_p, alpha=-lr)
        ```
        Both `param` and `buf` take a gradient and perform an in-place update.

        The python tracer will inspect calls to `nn.Module.forward` and
        `optim.Optimizer.step` to extract parameter and optimizer state
        respectively (including parameters), so this is generally a non-issue.

        However as a fallback we can also exploit several properties of
        parameters to distinguish them from other model state.

        First, they are directly used in the forward pass. (At this point we
        haven't established which parts of the graph correspond to the forward
        pass but we can deduce enough to suffice.) Some mutable state such as
        batch norm moving averages also contribute to the forward pass, but
        optimizer state does not.

        Second, a parameter is by definition used to compute at least one
        gradient and depends on at least one gradient.
        """
        snapshot = self._category_snapshot()

        # Determine which Tensors might be parameters based on forward pass
        # data flow. Note this these are only candidates; we filter nodes that
        # we know are part of the backward pass but that doesn't guarantee that
        # they are part of the forward pass.
        candidate_parameters: set[TensorAndID] = set()
        candidate_fwd_tensors: set[TensorAndID] = {
            i for i, category in snapshot.items() if category == Category.INPUT
        }

        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            if (
                # Don't check nodes in the backward pass.
                RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)
                and not any(self._is_gradient(*i) for i in inputs)
                and not any(self._is_gradient(*i) for i in node.outputs.items())
                #
                # and only check nodes which depend on an input.
                and candidate_fwd_tensors.intersection(inputs)
            ):
                candidate_fwd_tensors |= node.outputs.items()
                candidate_parameters |= inputs.difference(candidate_fwd_tensors)

        # Require that each parameter eventually contributes to the value of a gradient
        used_for_gradient: set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            if any(
                self._is_gradient(*i) or i in used_for_gradient
                for i in node.outputs.items()
            ):
                used_for_gradient.update(
                    (key, version) for key, (_, version) in node.inputs.items()
                )
        candidate_parameters.intersection_update(used_for_gradient)

        # and depends on a gradient.
        parameter_keys = {key.id for key, _ in candidate_parameters}
        parameter_keys &= self._any_version_depends_on_gradient()

        for key, _ in snapshot:
            if key.id in parameter_keys:
                self._categories.set_by_id(key, Category.PARAMETER)

    def _set_activations(self) -> None:
        """Flood the graph to identify activations."""

        required = {Category.INPUT, Category.ACTIVATION}
        also_allowed = {Category.PARAMETER, Category.TEMPORARY}
        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            input_categories = {self._categories.get(*i) for i in inputs}

            if (
                (input_categories & required)
                and not (input_categories - (required | also_allowed))
                #
                # Stop filling when we reach the backward pass.
                and RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)
            ):
                for i in node.outputs.items():
                    self._categories.setdefault_by_version(*i, Category.ACTIVATION)

    def _set_optimizer_state(self) -> None:
        for event in self._op_tree.dfs():
            if event.typed[0] == _EventType.PyCall and event.typed[1].optimizer:
                parameters = event.typed[1].optimizer.parameters
                for _, t in it.chain.from_iterable(
                    (state for _, _, state in parameters)
                ):
                    key = TensorKey.from_tensor(t)
                    if key is not None:
                        self._categories.set_by_id(key, Category.OPTIMIZER_STATE)

    def _set_autograd_detail(self) -> None:
        prior = {None, Category.AUTOGRAD_DETAIL}
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                for key, version in node.outputs.items():
                    if version == 0 or self._categories.get(key, version - 1) in prior:
                        self._categories.setdefault_by_version(
                            key, version, Category.AUTOGRAD_DETAIL
                        )


class MemoryProfileTimeline:
    def __init__(self, memory_profile) -> None:
        """The minimum representation of the memory profile timeline
        includes the memory timeline and categories. The timeline
        consists of [timestamp, action, (TensorKey, version), numbytes]
        elements, to denote any actions (pre-existing, create, destroy,
        or increment_version) that occurred to a specific Tensor for a
        chunk of memory. The categories help map each (TensorKey,
        version) pair into a category."""
        self.timeline = memory_profile.timeline
        self.categories = memory_profile._categories

    def _coalesce_timeline(self, device_str):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
        device = torch.device(device_str)
        times: list[int] = []
        sizes: list[list[int]] = []

        def update(key, version, delta) -> None:
            category = (
                self.categories.get(key, version)
                if isinstance(key, TensorKey)
                else None
            )
            index = _CATEGORY_TO_INDEX[category] + 1
            sizes[-1][index] += int(delta)

        t_min = -1
        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue

            # Convert timestamps from ns to us, to match trace events.
            if t != -1:
                t = int(t / 1000)

            # Save the smallest timestamp to populate pre-existing allocs.
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t

            # Handle timestep
            if len(times) == 0:
                times.append(t)
                sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])

            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())

            # Handle memory and categories
            if action in (Action.PREEXISTING, Action.CREATE):
                update(key, version, numbytes)

            elif action == Action.INCREMENT_VERSION:
                update(key, version, -numbytes)
                update(key, version + 1, numbytes)

            elif action == Action.DESTROY:
                update(key, version, -numbytes)

            else:
                raise ValueError(f"Unknown action: {action}")

        times = [t_min if t < 0 else t for t in times]
        return times, sizes

    def export_memory_timeline(self, path, device_str) -> None:
        """Saves the memory timeline as [times, sizes by category]
        as a JSON formatted file to the given path for the given
        device."""
        times, sizes = self._coalesce_timeline(device_str)
        # TODO: Write a faster serialize (orjson not available in CI)
        import json

        with open(path, "w") as f:
            json.dump([times, sizes], f)

    def export_memory_timeline_raw(self, path, device_str) -> None:
        """Saves the memory timeline as raw memory event tuples in the
        form of (timestamp, action, numbytes, category)
        as a JSON formatted file to the given path for the given
        device."""
        device = torch.device(device_str)
        raw_events: list[tuple[int, int, int, int]] = []

        def get_category_index(key, version):
            category = (
                self.categories.get(key, version)
                if isinstance(key, TensorKey)
                else None
            )
            return _CATEGORY_TO_INDEX[category]

        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue

            if action in (Action.PREEXISTING, Action.CREATE):
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        numbytes,
                        get_category_index(key, version),
                    )
                )

            elif action == Action.INCREMENT_VERSION:
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        -numbytes,
                        get_category_index(key, version),
                    )
                )
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        numbytes,
                        get_category_index(key, version + 1),
                    )
                )

            elif action == Action.DESTROY:
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        -numbytes,
                        get_category_index(key, version),
                    )
                )

            else:
                raise ValueError(f"Unknown action: {action}")

        import json

        with open(path, "w") as f:
            json.dump(raw_events, f)

    def export_memory_timeline_html(
        self, path, device_str, figsize=(20, 12), title=None
    ) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
        # Check if user has matplotlib installed, return gracefully if not.
        import importlib.util

        matplotlib_spec = importlib.util.find_spec("matplotlib")
        if matplotlib_spec is None:
            print(
                "export_memory_timeline_html failed because matplotlib was not found."
            )
            return

        from base64 import b64encode
        from tempfile import NamedTemporaryFile

        import matplotlib.pyplot as plt
        import numpy as np

        mt = self._coalesce_timeline(device_str)
        times, sizes = np.array(mt[0]), np.array(mt[1])
        # For this timeline, start at 0 to match Chrome traces.
        t_min = min(times)
        times -= t_min
        stacked = np.cumsum(sizes, axis=1) / 1024**3
        device = torch.device(device_str)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

        # Plot memory timeline as stacked data
        fig = plt.figure(figsize=figsize, dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(
                times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
            )
        fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
        # Usually training steps are in magnitude of ms.
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Memory (GB)")
        title = "\n\n".join(
            ([title] if title else [])
            + [
                f"Max memory allocated: {max_memory_allocated / (1024**3):.2f} GiB \n"
                f"Max memory reserved: {max_memory_reserved / (1024**3):.2f} GiB"
            ]
        )
        axes.set_title(title)

        # Embed the memory timeline image into the HTML file
        with NamedTemporaryFile("wb", suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")

            tmpfile.seek(0, 0)
            encoded = b64encode(tmpfile.read()).decode("utf-8")
            if not encoded:
                raise AssertionError("failed to encode image as base64")
            html = f"""<html>
<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
<body>
  <img src='data:image/png;base64,{encoded}'>
</body>
</html>"""

            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
