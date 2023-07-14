"""
CUDA graph trees are a safety abstraction over CUDAGraphs, similar to make_graph_callables,
which share the same memory pool.  Sharing a memory pool is an extremely
important optimization when chaining multiple CUDA graphs together, as it
prevents you from needing to copy intermediate tensors from one graph to the
next, and reduces overall memory usage by allowing dead memory from the first
pool to be reused in the second.

The standard graph/make_graph_callables support sharing memory pool, but
with a lot of caveats.  CUDA graph trees remove these restrictions:

* Previously, if you recorded graphs A, B, you had to replay A, B in that
  order.  With CUDA graph trees, after replaying A, you can change your
  mind and record/replay a different graph B'; we will support efficient
  execution of both A, B and A, B', using only max(mem(A, B), mem(A, B')).  In
  other words: we support arbitrary trees of CUDA graph operations, not just
  sequences (this is why this feature is called CUDA graph trees.)

* Previously, if you executed graph A, some non-CUDA graph code, and then
  graph B, after executing graph B, it was not safe to retain any references
  to intermediates produced by A.  With CUDA graph trees, we track if any
outputs of graph A are still live by the time graph B is run, and make
  sure graph B doesn't clobber there memory when reusing the CUDA graphs
  pool.  You'll get a separate recording of B depending on what tensors
  stay live or dead.

CUDA graph trees are flexible enough to be used in Dynamo across graph breaks,
which is their primary use case.

The ability to switch from replay to record is fairly nontrivial: remember that
when you replay a CUDA graph, you only replay CUDA operations; no CPU side state
is updated.  In particular, the CPU-side book-keeping for the allocator is not
reconstructed.  However, to record a new child CUDA graph, we must restore this
book-keeping.  This is what checkpoint pool state is used for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict

from enum import auto, Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    get_expanded_dims,
    get_input_idxs_to_check,
    index_expanded_dims,
    remove_unaligned_input_idxs,
    static_input,
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef

StorageWeakRefPointer = int
StorageDataPtr = int
NBytes = int

if torch.backends.cuda.is_built():
    from torch._C import (
        _cuda_CUDAAllocator_AllocatorState as AllocatorState,
        _set_cached_tensors_enabled as _set_cached_tensors_enabled,
    )
else:

    class AllocatorState:  # type: ignore[no-redef]
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        pass


log = logging.getLogger(__name__)

from . import config


@dataclasses.dataclass(frozen=True)
class GraphID:
    "Unique counter of a cuda graph recording"
    id: int


@dataclasses.dataclass(frozen=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"
    id: int


@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID


def clear_cublass_cache():
    """
    Cublas keeps a persistent workspace allocation for running matmuls. This poses a problem for
    doing warmup within a CUDAGraph private pool because we do not want persistent allocations from
    one one run to the next. When we begin a new run of a cudagraphs path (generation), all tensors
    from the previous generation are freed. This frees them the the memory pool, but not elsewhere.
    A tensor in the cublas workspace would continue to be in use the workspace but would also get allocated
    in the next run. The memory would be in use in two places.

    To solve this, we clear cublas caches before and after warming up or recording. If a workspace is required
    it will be allocated to the cudagraph private pool and accounted for in the allocator for the duration of the
    program. There is no overhead to this on replay since cudagraphs removes allocation overhead.
    """
    torch._C._cuda_clearCublasWorkspaces()


@contextlib.contextmanager
def clear_cublas_manager():
    "Context manager around clearing cublas caches that will clear on enter and exit"
    clear_cublass_cache()
    try:
        yield
    finally:
        clear_cublass_cache()


@contextlib.contextmanager
def disable_conv_cache_emptying():
    prev = torch._C._cuda_get_conv_benchmark_empty_cache()
    torch._C._cudnn_set_conv_benchmark_empty_cache(False)
    try:
        yield
    finally:
        torch._C._cudnn_set_conv_benchmark_empty_cache(prev)


@contextlib.contextmanager
def enable_history_recording():
    "Turns on history recording in the CUDA Caching Allocator"
    enabled = torch._C._cuda_isHistoryEnabled()
    try:
        if not enabled:
            torch.cuda.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.cuda.memory._record_memory_history(None)


def get_history_recording():
    # TODO - remove, prevents cleanup
    if not config.triton.cudagraph_trees_history_recording:
        return contextlib.nullcontext()
    return enable_history_recording()


class TreeManagerContainer:
    """
    Manages the lifetime of the tree manager. Like `PrivatePool` in cuda caching allocator,
    the tree and its corresponding memory pool should be kept alive as long as any outstanding
    graph or tensor which is an output of a graph remains alive.

    There is a single tree manager container per device.

    The lifecycle of a tree_manager is:
    -  Is constructed, no graph, no fns, no tensors
    -  Tree manager is fetched, resulting in tree manager being allocated
    -  We generate a bunch of functions, calling add_strong_reference
    -  These functions die, calling finalize_reference
    -  When all the functions die, we finalize_tree_manager.

    TODO: in the future, we would like to do the following once storage weak refs land
    -  We look for all the live storages and add references to THOSE
    -  We count as storages die
    -  All the storages are dead, we deallocate the tree manager
    """

    def __init__(self, device_index):
        # This class keeps a strong reference to tree_manager,
        # but upon all other strong references to the tree_manager will reset it to None.
        # We need a strong reference so that we can still access its attributes upon cleanup.
        self.tree_manager: Optional[CUDAGraphTreeManager] = None

        # Number of outstanding references to the current tree manager
        self.live_cudagraphify_fns = 0

        self.device_index = device_index

        # Following two objects are only set in the case that Tensor outputs outlive
        # the cudagraphify_fns. Reference to the Graph is needed to keep the private pool from
        # deallocation.
        self.live_storages_count = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        self.lock = threading.Lock()

    def _finalize_tensor(self):
        with self.lock:
            self.live_storages_count -= 1
            if self.live_storages_count == 0:
                self.graph = None

                # manager was used again after existing cleanup,
                # we shouldnt set it to None
                if self.live_cudagraphify_fns == 0:
                    self.tree_manager = None

    def finalize_cudagraphify_fn(self):
        with self.lock:
            self.live_cudagraphify_fns -= 1
            if self.live_cudagraphify_fns == 0:
                self._finalize_tree_manager()

    def _finalize_tree_manager(self):
        assert self.lock.locked()
        self.tree_manager = None

        # TODO - when issue #91395 is landed, we can set a weakref on
        # storages and trigger a deallocation when all outputs of the
        # cudagraph are dead.

        # live_storages = list(
        #     tree_manager.live_cudagraph_pool_storages_in_curr_execution()
        # )

        # # Maintain reference to graph to keep tensors alive
        # assert len(tree_manager.roots) > 0, "expected at least one use"
        # root = next(tree_manager.get_roots())
        # self.graph = root.graph
        # seen_storages = set()
        # for stor in live_storages:
        #     if stor in seen_storages:
        #         continue
        #     seen_storages.add(stor)
        #     self.live_storages_count += 1
        # .   weakref.finalize(stor, self._finalize_tensor)

    def add_strong_reference(self, fn: Callable[..., Any]):
        with self.lock:
            self.live_cudagraphify_fns += 1

        weakref.finalize(fn, self.finalize_cudagraphify_fn)

    def get_tree_manager(self) -> CUDAGraphTreeManager:
        with self.lock:
            if self.tree_manager is None:
                self.tree_manager = CUDAGraphTreeManager(self.device_index)
            return self.tree_manager


local = threading.local()

# one tree manager per device
local.tree_manager_containers = {}
local.tree_manager_locks = defaultdict(threading.Lock)


# only incremented by user call of mark_step_begin
class MarkStepBox:
    mark_step_counter = 0


# We need to register this as an object that will be copied over as TLS when new
# threads are created in autograd
torch._C._stash_obj_in_tls("tree_manager_containers", local.tree_manager_containers)
torch._C._stash_obj_in_tls("tree_manager_locks", local.tree_manager_locks)


def mark_step_begin():
    "Indicates that a new iteration of inference or training is about to begin."

    # iterate down to distinguish from GenerationTracking counter
    MarkStepBox.mark_step_counter -= 1


def reset_cudagraph_trees():
    "Clear all cudagraph trees"
    # see shutdown below for why this is necessary
    container_dict = get_obj(local, "tree_manager_containers")
    locks_dict = get_obj(local, "tree_manager_locks")
    for device, lock in locks_dict.items():
        with lock:
            container = container_dict.get(device)
            if not container or not container.tree_manager:
                continue

            container.tree_manager.shutdown()

    _set_cached_tensors_enabled(False)
    container_dict.clear()

    MarkStepBox.mark_step_counter = 0


def get_obj(local, attr_name):
    if hasattr(local, attr_name):
        return getattr(local, attr_name)
    else:
        assert torch._C._is_key_in_tls(attr_name)
        return torch._C._get_obj_in_tls(attr_name)


def get_container(device_index: int):
    container_dict = get_obj(local, "tree_manager_containers")
    lock = get_obj(local, "tree_manager_locks")[device_index]

    with lock:
        if device_index not in container_dict:
            container_dict[device_index] = TreeManagerContainer(device_index)

        return container_dict[device_index]


def get_manager(
    device_index: int, create_if_none_exists=True
) -> Optional[CUDAGraphTreeManager]:
    if create_if_none_exists:
        return get_container(device_index).get_tree_manager()
    return get_container(device_index).tree_manager


def cudagraphify_impl(model, inputs, static_input_idxs, *args, **kwargs):
    fn_cache: Dict[Tuple[int, ...], Callable[..., Any]] = {}

    # Detect int inputs: we need to index on these
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    del inputs

    def deferred_cudagraphify(inputs):
        int_key = get_ints(inputs)
        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)

        log.info("recording cudagraph tree for %s", int_key)

        # first get indices we need to check to align, then update our static inputs,
        # and finally copy
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        copy_misaligned_inputs(inputs, check_input_idxs)

        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
        fn = align_inputs_from_check_idxs(fn, inputs_to_check=check_input_idxs)
        fn_cache[int_key] = fn

        return out

    return deferred_cudagraphify


def cudagraphify(
    model,
    inputs,
    static_input_idxs=(),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    stack_traces: Optional[StackTraces] = None,
):
    manager = get_container(device_index).get_tree_manager()
    assert not (is_backward and is_inference)
    mode = (
        CompilationMode.BACKWARD
        if is_backward
        else (CompilationMode.INFERENCE if is_inference else CompilationMode.FORWARD)
    )

    return manager.add_function(
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
    )


class StorageWeakRefWrapper:
    """
    Wrapper around a storage weak ref. Will deallocate it upon expiration if invoked.
    """

    __slots__ = ["ref", "_data_ptr", "extra_ref_check"]

    storage_ref: Optional[StorageWeakRef]

    def __init__(
        self,
        inp: Union[Tensor, UntypedStorage],
        extra_ref_check: Optional[Callable[[], None]] = None,
    ):
        """
        extra_ref_check is an additional check we need to run to check if the
        weak ref has expired. in checking storage use count we assume extra_ref_check
        will hold an additional reference to the storage.
        """
        if isinstance(inp, Tensor):
            stor = inp.untyped_storage()
        else:
            assert isinstance(inp, UntypedStorage)
            stor = inp
        self.ref = StorageWeakRef(stor)
        self._data_ptr = stor.data_ptr()
        self.extra_ref_check = extra_ref_check

    @classmethod
    def from_weakref_and_data_ptr(cls, cdata, data_ptr, extra_ref_check=None):
        instance = cls.__new__(cls)
        instance._data_ptr = data_ptr
        instance.ref = StorageWeakRef.from_weakref(cdata)
        instance.extra_ref_check = extra_ref_check
        return instance

    def __call__(self) -> Optional[StorageWeakRefPointer]:
        if self.expired():
            return None

        return self.ref.cdata

    def swap_weakref(self, cdata):
        self.ref.__del__()
        self.ref.cdata = cdata

    def data_ptr(self) -> int:
        "NB: returns the data ptr even if the storage has expired"
        return self._data_ptr

    def remove_extra_reference(self):
        self.extra_ref_check = None

    def expired(self):
        if self.extra_ref_check is not None and not self.extra_ref_check():
            return False

        # if extra_ref_check is not None we expect an additional reference
        stor_count = torch._C._storage_Use_Count(self.ref.cdata)
        return (stor_count - (self.extra_ref_check is not None)) == 0

    def __repr__(self):
        if self.ref is None or self.ref.expired():
            return f"StorageWeakRefWrapper to {self.data_ptr()}; dead"
        else:
            return f"StorageWeakRefWrapper to {self.data_ptr()}; alive"


def is_live(weak_ref: Optional[StorageWeakRefWrapper]) -> bool:
    return maybe_deref(weak_ref) is not None


def maybe_deref(
    weak_ref: Optional[StorageWeakRefWrapper],
) -> Optional[Tuple[UntypedStorage, int]]:
    if weak_ref is None:
        return None
    r = weak_ref()
    if r is None:
        return None
    # NB: r.data_ptr() does not necessarily equal weak_ref.data_ptr()
    return r, weak_ref.data_ptr()


@contextlib.contextmanager
def _use_cuda_memory_pool_manager(device, mem_pool, stream):
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.device(device):
        torch._C._cuda_beginAllocateCurrentStreamToPool(device, mem_pool)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateCurrentStreamToPool(device)
            torch._C._cuda_releasePool(device, mem_pool)


def map_to_ref(t: Optional[Tensor]) -> Optional[StorageWeakRefWrapper]:
    if not isinstance(t, torch.Tensor):
        assert t is None
        return None
    return StorageWeakRefWrapper(t)


# A path index of (depth, offset) indices into a graph that is `depth`` number of nodes from the root
# at graph output offset
PathOutputIndex = Tuple[int, int]

# For each node in the path, for each output, is the output alive
PathLiveness = List[List[bool]]

StackTraces = List[Optional[str]]


class CUDAWarmupNode:
    """
    Simplified Wrapper around A CUDA Model that wraps outputs in storage refs and exposes
    apis to get the live storages in the current chain of warmup.

    A CUDAWarmupNode may have either CUDAGraphNode or CUDAWarmupNode as a parent, but may only have
    CUDAWarmupNode as children, because we cannot record or execute with tensors which do not have stable
    memory addresses.

    CUDAWarmupNode and CUDAGraphNode have a number of differences that make it easier to use separate classes.
    - Much of the CUDAGraphNode logic & initialization is based on the tensor properties of first recording. In the
    first instance of warmup, these are not finalized yet.
    - All Inputs to the RecordedFunction must be copied over to the cuda graph memory pool, this is unnecessary in warmup.
    - CUDAWarmup is only used once and so does not need to optimize as much bookkeeping. It is much simpler.

    NB: this class and CUDAGraphNode need to expose `path_live_weakrefs`, `all_outputs_are_dead`, and
    `self.outputs_weakrefs`, `stack_traces`, and `tensor_weakrefs` for compatibility.
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        parent,
        cuda_graphs_pool: Tuple[int, int],
        existing_cuda_graph: torch.cuda.Graph,
        device_index: int,
        stack_traces: Optional[StackTraces],
        stream: torch.cuda.Stream,
        already_warm: bool,
    ):
        self.wrapped_function = wrapped_function
        self.parent = parent
        self.cuda_graphs_pool = cuda_graphs_pool
        self.outputs_weakrefs: List[Optional[StorageWeakRefWrapper]] = []
        self.tensor_weakrefs: List[Optional[TensorWeakRef]] = []
        self.existing_cuda_graph = existing_cuda_graph
        self.has_run = False
        self.device_index = device_index
        self.stack_traces = stack_traces
        self.stream = stream
        self.already_warm = already_warm

    def run(self, new_inputs):
        assert not self.has_run, "Wrapped function should never be run twice"

        # See: output_is_alias_of_persistent_static_inputs below. We should only be returning freshly created
        # storages in path_live_weakrefs.
        existing_path_data_ptrs = {
            t.data_ptr() for t in self.path_live_weakrefs() if t()
        }
        non_cudagraph_inps = set()
        for i in range(len(new_inputs)):
            if (
                isinstance(new_inputs[i], torch.Tensor)
                and new_inputs[i].untyped_storage().data_ptr()
                not in existing_path_data_ptrs
            ):
                non_cudagraph_inps.add(new_inputs[i].untyped_storage().data_ptr())

        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)

        with torch.cuda.device(
            self.device_index
        ), disable_conv_cache_emptying(), clear_cublas_manager(), _use_cuda_memory_pool_manager(
            self.device_index, self.cuda_graphs_pool, self.stream
        ), get_history_recording():
            out = self.wrapped_function.model(new_inputs)

        # sync up stream used in `_use_cuda_memory_pool_manager` - TODO - wait stream instead ?
        torch.cuda.synchronize()

        assert len(new_inputs) == 0

        # sdpa returns cpu tensors when not recording cuda graph
        def add_ref(o):
            return (
                o is not None
                and isinstance(o, torch.Tensor)
                and o.is_cuda
                and o.untyped_storage().data_ptr() not in non_cudagraph_inps
                and o.untyped_storage().data_ptr() != 0
            )

        self.outputs_weakrefs.extend(
            [map_to_ref(o) if add_ref(o) else None for o in out]
        )
        self.tensor_weakrefs.extend(
            [TensorWeakRef(o) if add_ref(o) else None for o in out]
        )

        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            out_refs = self.path_live_weakrefs()
            new_storages = [
                t for t in out_refs if t.data_ptr() not in non_cudagraph_inps
            ]
            check_memory_pool(self.device_index, self.cuda_graphs_pool, new_storages)

        return out

    @property
    def _path_from_root(self):
        nodes = []
        node = self
        while node:
            nodes.append(node)
            node = node.parent

        yield from reversed(nodes)

    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        "Returns all live storages weakrefs that created by nodes in this path"
        for node in self._path_from_root:
            for output in node.outputs_weakrefs:
                if is_live(output):
                    yield output

    def all_outputs_are_dead(self):
        return not list(self.path_live_weakrefs())


# Aliases for List that say what the indices denote
InputList = List  # input indexes
OutputList = List  # output indexes
LevelList = List  # levels (distance from root of tree)


class OutputAliasInfo:
    pass


class _UnaliasedStorage(OutputAliasInfo):
    "Singleton to mark that the graph output constructs a new alias or is None"
    pass


UnaliasedStorage = _UnaliasedStorage()


class AliasesPriorGraphOutput(OutputAliasInfo):
    "Marks that the graph output aliases an output of a prior graph"
    __slots__ = ["index"]

    index: PathOutputIndex

    def __init__(self, index: PathOutputIndex):
        assert isinstance(index, tuple)
        self.index = index


class AliasesNewOutput(OutputAliasInfo):
    "Marks that the graph output aliases an index in the new, returned outputs"

    __slots__ = ["index"]

    index: int

    def __init__(self, index):
        assert isinstance(index, int)
        self.index = index


class CUDAGraphNode:
    """
    A single recording of a function into a CUDA Graph. Recordings of CUDA Graphs share a single memory pool
    and are structured into a tree, where there is a single recording that can precede it (parent) and multiple
    subsequent recordings that may follow (children). A node will have no parent if it is the first recording
    in a tree; i.e., when it is first recorded, there are no live tensors from a previous recording which
    would force a dependency.

    On first recording, all of the live tensors in the current CUDA Graph Node path will be
    reflected in the corresponding private pool. On subsequent executions, the caching allocator
    is unaffected when the graph is replayed.

    In order to support recording a subsequent cuda graph recording after execution of this graph,
    we checkpoint the state of the memory pool so that it may later be resumed.

    WrappedFunction should have already been warmed up prior to invocation.

    See [setCheckpointPoolState] for further explanation, as well as
    https://user-images.githubusercontent.com/13564/222815509-374f3400-f83d-4f7d-8fa6-4a092b3250bb.png
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        id: GraphID,
        parent: Optional[CUDAGraphNode],
        inputs: List[Tensor],
        cuda_graphs_pool: Tuple[int, int],
        device_index: int,
        stack_traces: Optional[StackTraces],
        stream: torch.cuda.Stream,
    ):
        assert isinstance(inputs, (list, tuple))

        self.wrapped_function = wrapped_function
        self.id = id
        self.device = device_index
        self.stack_traces = stack_traces
        self.stream = stream

        # if this is a root parent will be None. use weakref to prevent reference cycle
        self._parent = weakref.ref(parent) if parent is not None else None
        # reference to the shared memory pool for the entire cuda graphs tree
        self.cuda_graphs_pool = cuda_graphs_pool

        # A single wrapped function may be recorded multiple times if memory patterns or
        # invariants change from one execution to the next
        self.children: Dict[FunctionID, List[CUDAGraphNode]] = defaultdict(list)

        # StorageWeakRef maintains whether the Storage C++ object remains allocated,
        # not whether the corresponding memory has been deallocated. In order
        # to use them to track memory deallocations we must maintain a single StorageWeakRef
        # for all Storages that reference that memory (even if we are constructing Storages
        # that do not have a deallocator function). We maintain one single storage_cache
        # as we execute any tree path. When we retrieve a storage from the cache we
        # check that it is still alive, and we hash based on observed recording data ptr
        # and storage cdata.

        # we preserve a single reference to executed outputs that is then referenced
        # in children to avoid children having to chase parent pointers in the hot path
        # DO NOT reassign output_weakrefs, only call `clear()`
        # Path is a series of nodes from root to the current node
        self.outputs_weakrefs: OutputList[Optional[StorageWeakRefWrapper]] = []
        self.path_weakrefs: LevelList[OutputList[Optional[StorageWeakRefWrapper]]] = [
            node.outputs_weakrefs for node in self._path_from_root
        ]
        self.path_stacktraces: LevelList[StackTraces] = [
            node.stack_traces for node in self._path_from_root
        ]
        self.tensor_weakrefs: OutputList[Optional[TensorWeakRef]] = []

        # tensors which are outputs of previous graphs in the tree
        self.cudagraph_managed_idxs: List[int] = [
            idx
            for idx, t in enumerate(inputs)
            if isinstance(t, torch.Tensor) and self._is_cuda_graph_recorded_tensor(t)
        ]

        self.static_input_idxs: List[int] = list(
            set(wrapped_function.static_input_idxs) | set(self.cudagraph_managed_idxs)
        )

        self.static_input_data_ptrs: InputList[int] = [
            (
                inputs[i].data_ptr()
                if isinstance(inputs[i], torch.Tensor) and i in self.static_input_idxs
                else None
            )
            for i in range(len(inputs))
        ]

        # When we checkpoint, and free generations, we will be manually freeing the outputs
        # of CUDAGraphNodes. We should not be freeing parameters, not do we need to account for
        # their liveness (they are static), so we need to compute which outputs are aliases of
        # parameters. Some static inputs are saved tensors from the forward that die in the backward.
        # Their locations are static but lifetimes are not. We only include the persistent static
        # data ptrs below because the non persistent data ptrs may be outputs of this record and
        # fresh allocations.

        # precompute expanded dims to avoid computing in the hot path
        self.expanded_dims: List[List[int]] = [
            get_expanded_dims(x)
            if isinstance(x, torch.Tensor) and idx not in self.static_input_idxs
            else []
            for idx, x in enumerate(inputs)
        ]

        # For each node in path, which outputs were observed to be live
        # before invoking graph recording, and after graph recording
        self.recorded_liveness_before_graph: LevelList[OutputList[bool]] = []
        self.recorded_liveness_after_graph: LevelList[OutputList[bool]] = []

        # List of Tuples of (depth, output_index) that index into node at depth
        # number of nodes from root and output_index of outputs. Will index into
        # path_weakrefs.
        self.expected_dead_indices_before_graph: List[PathOutputIndex] = []
        self.expected_dead_indices_after_graph: List[PathOutputIndex] = []

        # all live indices after graph recording
        self.live_indices_after_graph: List[PathOutputIndex] = []

        if self.parent is not None:
            previous_liveness = self.parent.recorded_liveness_after_graph
            curr_liveness = self._get_liveness(self.path_weakrefs)

            different_indices = self._get_different_indices(
                previous_liveness, curr_liveness
            )

            self.recorded_liveness_before_graph = curr_liveness
            self.expected_dead_indices_before_graph = different_indices

        recording_inputs = self._allocate_and_copy_recording_inputs(inputs)
        # recording inputs will copy over memory, so we can free non recording inputs
        inputs.clear()
        del inputs

        # graph used for recording model invocation
        self.graph = torch.cuda.CUDAGraph()

        # we allocate non-static inputs within the same memory pool as the CUDAGraph
        # which we will record the model with. For memory efficiency, it is important
        # to reclaim the input memory when the inputs are no longer live. To accomplish this,
        # we reconstruct tensors at the correct data pointers of our inputs which are
        # non owning and do not prevent deallocation. On subsequent executions, input values
        # will be copied over to these tensors.
        self.reconstructed_inputs: InputList[Tensor] = [
            self._reconstruct_from_tensor_metadata(self._tensor_metadata(x))
            if isinstance(x, torch.Tensor)
            else x
            for x in recording_inputs
        ]

        # DO THE RECORDING!!!
        # We record the CUDA graph in the constructor of CUDAGraphNode, which
        # gives you what the CPU side compute of the function would do.  We
        # don't throw the recording outputs away: their memory is
        # correctly accounted for in the CUDAGraphs caching allocator.  This
        # means on the very FIRST run of the CUDA graph node, we can directly
        # do more recording, because we have a valid caching allocator state.
        # NB: This relies on run() being called immediately after the
        # constructor, otherwise this optimization would not be valid.

        # initialized below in _record

        self.checkpointed_caching_state: Optional[AllocatorState] = None

        # Output Storage Alias information, can be:
        # - A new, unaliased storage, or the output is None
        # - An alias of an output of a prior graph
        # - An alias of an output already created in the reconstructed outputs
        # This is None if the output in question is an int
        self.output_storage_alias: OutputList[Optional[OutputAliasInfo]] = []

        # is the output Storage unaliased in subsequent outputs, of all subsequent paths
        # if it is, we cached the output tensor and adjust storage liveness tracking to also
        # check if the output tensor does not have an additional python reference.
        # If a descendent node discovers it has an alias of a prior output, then the output
        # will no longer be cached in the ancestor.
        # The large majority of tensors are unaliased, and preserving aliased output tensors would add
        # significant additional complexity with marginal gains
        # The cached tensor outputs are added on the first execution, and cleared whenever we need
        # to do subsequent recording
        self.unaliased_in_all_paths: OutputList[bool] = []
        self.cached_tensor_outputs: OutputList[Optional[Tensor]] = []

        # if an output aliases a static, persistent input then the corresponding Tensor will
        # be set here. These are different than cached tensors, because they are tensors that
        # are aliases of parameters that are always live.
        self.static_output_tensors: OutputList[Optional[Tensor]] = []

        # Cleared after recording
        self.recording_outputs: Optional[
            OutputList[Union[torch.Tensor, int]]
        ] = self._record(wrapped_function.model, recording_inputs)
        self.outputs_metadata: OutputList[Union[Dict[str, Any], int, None]] = []

        # As with inputs, we do not want to keep the outputs permanently alive because that would prevent
        # their memory being reclaimed in subsequent cuda graph recordings. We record the tensor metadata
        # needed to reconstruct instead.
        assert self.recording_outputs is not None
        for out in self.recording_outputs:
            if isinstance(out, torch.Tensor):
                self.outputs_metadata.append(
                    self._tensor_metadata(out, ignore_storage_offset=False)
                )
            else:
                assert isinstance(out, (int, type(None))), type(out)
                self.outputs_metadata.append(out)

        self.graph.replay()

    def _copy_input(self, idx, dst, src):
        expanded_dims = self.expanded_dims[idx]
        dst = index_expanded_dims(dst, expanded_dims)
        src = index_expanded_dims(src, expanded_dims)
        # TODO - one jit kernel across multiple inputs
        dst.copy_(src)

    def run_first_inputs(self, new_inputs):
        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_before_invocation()

        # graph is already invoked in the __init__
        # inputs are copied over in _allocate_recording_inputs and subsequently cleared
        assert len(new_inputs) == 0
        outputs = self.recording_outputs
        self.recording_outputs = None
        return outputs

    def run(self, new_inputs):
        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_before_invocation()

        assert len(self.static_input_data_ptrs) == len(new_inputs)
        # NB: this ranges over non-static inputs too
        for idx, data_ptr in enumerate(self.static_input_data_ptrs):
            if idx in self.cudagraph_managed_idxs:
                continue
            if not isinstance(new_inputs[idx], torch.Tensor):
                pass
            elif data_ptr is not None:
                # static input, e.g., parameter
                assert data_ptr == new_inputs[idx].data_ptr()
            else:
                # non-static input, need to copy it into CUDA graph
                dst = self.reconstructed_inputs[idx]
                src = new_inputs[idx]
                self._copy_input(idx, dst, src)

        new_inputs.clear()
        self.run_graph()

        outputs = self.reconstruct_outputs()
        self.debug_check_invariants_after_invocation()

        return outputs

    def reconstruct_outputs(self):
        "Reconstruct output tensors according to their saved metadata and alias information"

        # Cached tensors will not yet be set on the first execution
        # They are also cleared in checkpointing, so if we checkpoint this node
        # and then execute it again we will need to repopulate cached tensors
        if not self.cached_tensor_outputs:
            self._initialize_cached_tensors()

        outputs: List[torch.Tensor] = []

        for i, (storage_info, metadata) in enumerate(
            zip(self.output_storage_alias, self.outputs_metadata)
        ):
            if not isinstance(metadata, dict):  # tensor metadata
                assert isinstance(metadata, (int, type(None)))
                outputs.append(metadata)
                continue

            cached_t = self.cached_tensor_outputs[i]
            if cached_t is not None:
                # No need to update weakrefs, already correctly initialized
                outputs.append(cached_t)
                continue

            static_t = self.static_output_tensors[i]
            if static_t is not None:
                assert self.outputs_weakrefs[i] is None
                outputs.append(static_t)
                continue

            storage = self.prepare_alias_info_for_tensor_construction(
                storage_info, metadata
            )

            if isinstance(storage, UntypedStorage) or storage is None:
                out = self._reconstruct_from_tensor_metadata(metadata, storage)
            else:
                assert isinstance(storage, int)
                out = self._reconstruct_from_tensor_metadata(
                    metadata, outputs[storage].untyped_storage()
                )

            outputs.append(out)
            w = self.outputs_weakrefs[i]
            assert w is not None
            w.swap_weakref(out.untyped_storage()._weak_ref())

        return outputs

    def prepare_alias_info_for_tensor_construction(
        self,
        out_alias_info: Optional[OutputAliasInfo],
        metadata: Union[Dict[str, Any], int, None],
    ) -> Union[UntypedStorage, None, int]:
        if (
            isinstance(metadata, (int, type(None)))
            or out_alias_info is UnaliasedStorage
        ):
            return None

        if isinstance(out_alias_info, AliasesPriorGraphOutput):
            depth, existing_output_index = out_alias_info.index
            ref = self.path_weakrefs[depth][existing_output_index]
            assert ref is not None
            return torch.UntypedStorage._new_with_weak_ptr(ref())

        assert isinstance(out_alias_info, AliasesNewOutput)
        return out_alias_info.index

    def prepare_storages_for_construction(
        self,
    ) -> List[Union[UntypedStorage, None, int]]:
        output_storages = []
        for output_storage_alias, metadata in zip(
            self.output_storage_alias, self.outputs_metadata
        ):
            output_storages.append(
                self.prepare_alias_info_for_tensor_construction(
                    output_storage_alias, metadata
                )
            )

        return output_storages

    def run_graph(self):
        self.graph.replay()

    def all_outputs_are_dead(self):
        "All outputs of the path from this node to its root are dead"
        for depth, output_index in self.live_indices_after_graph:
            if is_live(self.path_weakrefs[depth][output_index]):
                return False
        return True

    def _record(self, model, inputs):
        "Record the model"

        # see: output_is_alias_of_persistent_static_inputs above
        static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper] = {
            inputs[i].untyped_storage().data_ptr(): StorageWeakRefWrapper(inputs[i])
            for i in self.wrapped_function.static_input_idxs
            if isinstance(inputs[i], torch.Tensor)
            and not self._is_cuda_graph_recorded_tensor(inputs[i])
        }

        if config.triton.slow_path_cudagraph_asserts:
            # need to use parent live weakrefs because live_indices isnt set yet
            memory = (
                [] if self.parent is None else list(self.parent.path_live_weakrefs())
            )
            memory += [
                StorageWeakRefWrapper(elem)
                for i, elem in enumerate(inputs)
                if isinstance(elem, torch.Tensor)
                and i not in self.wrapped_function.static_input_idxs
                and elem.data_ptr() != 0
            ]
            check_memory_pool(self.device, self.cuda_graphs_pool, memory)

        with preserve_rng_state(), torch.cuda.device(
            self.device
        ), clear_cublas_manager(), torch.cuda.graph(
            self.graph, stream=self.stream, pool=self.cuda_graphs_pool
        ), get_history_recording():
            static_outputs = model(inputs)

        # running model should reclaim memory
        assert len(inputs) == 0

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

        self._add_first_outputs(static_outputs, static_input_persistent_storage_ptrs)

        return static_outputs

    def _add_first_outputs(
        self,
        outputs,
        static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper],
    ):
        "Add the outputs from the first invocation of the node and set up metadata"

        # getting liveness before we have added the outputs to path, so the length
        # of the two lists is equal
        prev_liveness = self.recorded_liveness_before_graph
        curr_liveness = self._get_liveness(self.path_weakrefs)

        delta = self._get_different_indices(prev_liveness, curr_liveness)
        self.expected_dead_indices_after_graph = delta

        assert len(self.outputs_weakrefs) == 0
        # index from data pointer to index in outputs
        output_new_storages_index: Dict[StorageDataPtr, int] = {}

        self.unaliased_in_all_paths = [False for _ in range(len(outputs))]
        self.static_output_tensors = [None for _ in range(len(outputs))]

        for i, o in enumerate(outputs):
            if o is None or not isinstance(o, torch.Tensor):
                self.output_storage_alias.append(UnaliasedStorage)
                continue

            torch._check(
                o.is_cuda,
                lambda: (
                    "Expected all cuda outputs in cuda graph recording. Non cuda output "
                    f"from {self.stack_traces[i] if self.stack_traces else '(unknown)'}"
                ),
            ),

            ref = static_input_persistent_storage_ptrs.get(
                o.untyped_storage().data_ptr(), None
            )
            # also treat empty storages as static outputs because we do not need to manage their lifetime
            # and they should not participate in checkpointing
            is_empty_storage = o.data_ptr() == 0
            if ref and ref() is not None or is_empty_storage:
                self.output_storage_alias.append(None)
                self.static_output_tensors[i] = o
                continue

            path_ref = self._is_alias_of_live_recorded_tensor(o)
            if path_ref is not None:
                self._mark_prior_graph_output_as_aliased(path_ref)
                self.output_storage_alias.append(AliasesPriorGraphOutput(path_ref))
                continue

            if o.untyped_storage().data_ptr() in output_new_storages_index:
                index = output_new_storages_index[o.untyped_storage().data_ptr()]
                self.unaliased_in_all_paths[index] = False
                self.output_storage_alias.append(AliasesNewOutput(index))
                continue

            output_new_storages_index[o.untyped_storage().data_ptr()] = i
            self.output_storage_alias.append(UnaliasedStorage)
            self.unaliased_in_all_paths[i] = True

        if self.stack_traces is None:
            self.stack_traces = [None for _ in range(len(outputs))]
        else:
            assert len(self.stack_traces) == len(
                outputs
            ), "Wrong number of stack traces passed in"

        assert not self.outputs_weakrefs
        for out, static_output_tensor in zip(outputs, self.static_output_tensors):
            if not isinstance(out, torch.Tensor) or static_output_tensor is not None:
                self.outputs_weakrefs.append(None)
                self.tensor_weakrefs.append(None)
            else:
                self.outputs_weakrefs.append(StorageWeakRefWrapper(out))
                self.tensor_weakrefs.append(TensorWeakRef(out))

        self.recorded_liveness_after_graph = self._get_liveness(self.path_weakrefs)
        self.checkpointed_caching_state = torch._C._cuda_getCheckpointState(
            self.device, self.cuda_graphs_pool
        )

        # now, get liveness with outputs added
        for depth in range(len(self.path_weakrefs)):
            for output_index in range(len(self.path_weakrefs[depth])):
                if is_live(self.path_weakrefs[depth][output_index]):
                    self.live_indices_after_graph.append((depth, output_index))

        self.debug_check_invariants_after_invocation()
        if config.triton.slow_path_cudagraph_asserts:
            check_memory_pool(
                self.device, self.cuda_graphs_pool, list(self.path_live_weakrefs())
            )

    def _mark_prior_graph_output_as_aliased(self, index: PathOutputIndex):
        "Remove a graph output from the unaliased, cached tensors in an ancestor node"
        depth, output_index = index
        node = list(self._path_from_root)[depth]
        node.unaliased_in_all_paths[output_index] = False
        x = self.path_weakrefs[depth][output_index]
        assert x is not None
        x.remove_extra_reference()

    def _initialize_cached_tensors(self):
        # we should not be clearing output_weakrefs, and they should be set in the first
        # record run
        assert len(self.outputs_weakrefs) == len(self.outputs_metadata)

        for i, (storage_info, metadata, make_cached) in enumerate(
            zip(
                self.output_storage_alias,
                self.outputs_metadata,
                self.unaliased_in_all_paths,
            )
        ):
            if not make_cached:
                self.cached_tensor_outputs.append(None)
                continue

            assert storage_info is UnaliasedStorage
            assert isinstance(metadata, dict)
            s = self.create_storage(metadata)
            out = self._reconstruct_from_tensor_metadata(metadata, storage=s)

            # XXX: let autograd know that there will be an additional reference to the tensor
            # that can be ignored when deciding whether to do gradient buffer inplacing.
            # Otherwise, inplacing could differ between tracing and subsequent execution.
            # For some models we tested this led to inputs no longer being in cudagraph pools,
            # leading to spurious re-recordings.
            # It also tells AMP cache that even though the tensor impls cannot be cached
            # in dtype conversions.

            torch._C._add_cached_tensor(out)

            self_ref = weakref.ref(self)

            # one reference in our array, and calling sys.getrefcount bumps the refcount by one
            def check_refcount(i):
                self_loc = self_ref()
                if self_loc is None:
                    return False
                return self_loc.get_output_refcount(i) == 2

            check = functools.partial(check_refcount, i=i)

            self.outputs_weakrefs[i] = StorageWeakRefWrapper(out, extra_ref_check=check)
            self.cached_tensor_outputs.append(out)

    def get_output_refcount(self, index):
        return sys.getrefcount(self.cached_tensor_outputs[index])

    @property
    def parent(self):
        "unwraps the weakref to _parent"
        return self._parent() if self._parent is not None else None

    @property
    def _path_to_root(self):
        "Returns all nodes in the path starting at self and ending at root"
        node = self
        while node:
            yield node
            node = node.parent

    @property
    def _path_from_root(self):
        "Returns all nodes in the path starting at the root and ending at self"
        nodes = reversed(list(self._path_to_root))
        yield from nodes

    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor):
        "Is this tensor an output of a node in this path"
        for output_refs in self.path_weakrefs:
            for storage_weak_ref in output_refs:
                if storage_weak_ref is None:
                    continue
                # don't need to check liveness of storage since the cuda graph managed
                # memory is never released.
                data_ptr = storage_weak_ref.data_ptr()
                if t.untyped_storage().data_ptr() == data_ptr:
                    return True

        return False

    def _is_alias_of_live_recorded_tensor(
        self, t: torch.Tensor
    ) -> Optional[PathOutputIndex]:
        for depth, output_refs in enumerate(self.path_weakrefs):
            for output_index, storage_ref in enumerate(output_refs):
                if (storage_and_ptr := maybe_deref(storage_ref)) is not None:
                    storage, ptr = storage_and_ptr
                    if ptr == t.untyped_storage().data_ptr():
                        return (depth, output_index)

        return None

    @staticmethod
    def _check_liveness(
        indices: List[PathOutputIndex],
        output_refs: List[List[Optional[StorageWeakRefWrapper]]],
    ):
        "Check that all of the indices specified are dead references"
        for depth, output_index in indices:
            w = output_refs[depth][output_index]
            assert w is not None
            if w() is not None:
                return False
        return True

    def add_child(self, function_id: FunctionID, node: CUDAGraphNode):
        "Adds node as a a child of self"
        self.children[function_id].append(node)

    @staticmethod
    def _get_different_indices(
        prev: List[List[bool]], curr: List[List[bool]]
    ) -> List[PathOutputIndex]:
        "Find indices where the two lists differ."
        dead_indices = []
        assert len(prev) <= len(curr)
        for i, (outputs1, outputs2) in enumerate(zip(prev, curr)):
            assert len(outputs1) == len(outputs2)
            for j, (output1, output2) in enumerate(zip(outputs1, outputs2)):
                if output1 != output2:
                    dead_indices.append((i, j))

        return dead_indices

    @staticmethod
    def _get_liveness(
        weakrefs: List[List[Optional[StorageWeakRefWrapper]]],
    ) -> List[List[bool]]:
        "Maps weakrefs to true if the reference is alive and false otherwise"
        if len(weakrefs) == 0:
            return []

        return [pytree.tree_map(is_live, outputs) for outputs in weakrefs]

    def debug_assert_invariants(
        self, expected_liveness: List[List[bool]], newly_dead: List[PathOutputIndex]
    ):
        if not config.triton.fast_path_cudagraph_asserts:
            return

        for i, node in enumerate(self._path_from_root):
            assert self.path_weakrefs[i] is node.outputs_weakrefs

        nodes = list(self._path_from_root)

        live_blocks = get_block_addrs(self.cuda_graphs_pool)

        live_storage_data_ptrs = set()
        live_storage_weak_ptrs = set()

        for depth, outputs_liveness in enumerate(expected_liveness):
            for output_idx, output_liveness in enumerate(outputs_liveness):
                # tensor can die early, but it can't be alive when it should be dead
                w = self.path_weakrefs[depth][output_idx]
                if (stor_weak_ptr_and_data_ptr := maybe_deref(w)) is not None:
                    assert output_liveness
                    stor_weak_ptr, stor_data_ptr = stor_weak_ptr_and_data_ptr
                    assert (stor_data_ptr in live_storage_data_ptrs) == (
                        stor_weak_ptr in live_storage_weak_ptrs
                    )
                    live_storage_data_ptrs.add(stor_data_ptr)
                    live_storage_weak_ptrs.add(stor_weak_ptr)

                    is_persistent_alias = (
                        nodes[depth].static_output_tensors[output_idx] is not None
                    )

                    if is_persistent_alias:
                        assert stor_data_ptr not in live_blocks

        for depth, output_index in newly_dead:
            assert not is_live(self.path_weakrefs[depth][output_index])

    def debug_check_invariants_before_invocation(self):
        self.debug_assert_invariants(
            self.recorded_liveness_before_graph, self.expected_dead_indices_before_graph
        )

    def debug_check_invariants_after_invocation(self):
        self.debug_assert_invariants(
            self.recorded_liveness_before_graph, self.expected_dead_indices_after_graph
        )

    def data_ptrs_dead_since_invocation(self) -> List[int]:
        """
        Since this node was invoked, return data ptrs of all tensor outputs that have died
        in the current executing tree path.
        """
        curr_liveness = self._get_liveness(self.path_weakrefs)
        _get_different_indices = self._get_different_indices(
            self.recorded_liveness_after_graph, curr_liveness
        )

        path = list(self._path_from_root)
        ptrs_to_deallocate = []
        for depth, output_index in _get_different_indices:
            ptrs_to_deallocate.append(
                path[depth].outputs_metadata[output_index]["data_ptr"]
            )

        return ptrs_to_deallocate

    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        for i, j in self.live_indices_after_graph:
            out = self.path_weakrefs[i][j]
            if out is not None and is_live(out):
                yield out

    def remove_node_cached_tensors(self):
        for t in self.cached_tensor_outputs:
            if t is not None:
                torch._C._remove_cached_tensor(t)
        self.cached_tensor_outputs.clear()

        for i, unaliased in enumerate(self.unaliased_in_all_paths):
            if unaliased:
                n = self.outputs_weakrefs[i]
                assert n is not None
                n.remove_extra_reference()

    def remove_path_cached_tensors(self):
        for node in self._path_from_root:
            node.remove_node_cached_tensors()

    def clear_path_state(self):
        "Clear the path state in this current executing node"
        # this doesnt actually do anything right now, leaving it as placeholder
        pass

    @staticmethod
    def _tensor_metadata(x, ignore_storage_offset=True):
        assert isinstance(x, torch.Tensor)
        # We ignore the storage offset for inputs, but not for outputs
        # TODO: - should we make the storage resizable ?
        return {
            "nbytes": x.untyped_storage().nbytes(),
            "data_ptr": x.untyped_storage().data_ptr(),
            "size": x.shape,
            "stride": x.stride(),
            "dtype": x.dtype,
            "device": x.device,
            "storage_offset": x.storage_offset() if not ignore_storage_offset else 0,
        }

    def _reconstruct_from_tensor_metadata(
        self, metadata: Dict[str, Any], storage=None
    ) -> Tensor:
        s = self.create_storage(metadata) if storage is None else storage
        return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, s)

    def create_storage(self, metadata):
        return torch._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )

    def _allocate_and_copy_recording_inputs(
        self, inputs
    ) -> List[Union[torch.Tensor, int]]:
        """
        Allocate inputs for non static, non cudagraph managraphed managed tensors in the memory pool
        and copy over the tensor values.
        """

        torch.cuda.synchronize()
        self.stream.wait_stream(torch.cuda.current_stream())
        recording_inputs = []

        with warnings.catch_warnings(record=True), torch.cuda.device(
            self.device
        ), _use_cuda_memory_pool_manager(
            self.device,
            mem_pool=self.cuda_graphs_pool,
            stream=self.stream,
        ):
            for i, inp in enumerate(inputs):
                if not isinstance(inp, torch.Tensor):
                    assert isinstance(inp, int)
                    recording_inputs.append(inp)
                elif i not in self.static_input_idxs:
                    # static_input does an allocation!
                    recording_inputs.append(static_input(inp))
                    # copy over and clear non recording input
                    self._copy_input(i, recording_inputs[-1], inp)
                    inputs[i] = None
                    del inp
                else:
                    recording_inputs.append(inp)

        return recording_inputs

    def check_invariants(self, inputs: List[Tensor]) -> bool:
        """
        Checks if this node can be run. The same pattern of tensor liveness and tensors
        managed in the cudagraph private pool must remain stable.
        """

        # previously managed data pointers remain stable
        for idx in self.cudagraph_managed_idxs:
            if inputs[idx].data_ptr() != self.static_input_data_ptrs[idx]:
                return False

        if not self._check_liveness(
            self.expected_dead_indices_before_graph, self.path_weakrefs
        ):
            return False

        # the cudagraph managed tensors which died upon recording must also die upon
        # this invocation. it is too late to check after we've replayed the graph,
        # because we would have already written over their memory.
        for idx in self.cudagraph_managed_idxs:
            inputs[idx] = None

        torch._check(
            self._check_liveness(
                self.expected_dead_indices_after_graph, self.path_weakrefs
            ),
            lambda: "TODO: graph recording observed an input tensor deallocate during graph "
            " recording that did not occur during replay. Please file an issue.",
        )
        return True

    def num_descendants(self) -> int:
        "Total number of descendents of this node"
        num_desc = 0
        for children in self.children.values():
            for child in children:
                num_desc += 1
                num_desc += child.num_descendants()
        return num_desc


def get_cudagraph_segments(pool_id):
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


def get_block_addrs(pool_id, live_only=True):
    blocks = []

    for segment in get_cudagraph_segments(pool_id):
        addr = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated" or not live_only:
                blocks.append(addr)

            addr += block["size"]

    return blocks


def format_tb(caching_allocator_trace):
    formatted_traceback = []

    for entry in caching_allocator_trace["frames"]:
        formatted_traceback.append(
            traceback.FrameSummary(entry["filename"], entry["line"], entry["name"])
        )

    return "".join(traceback.format_list(formatted_traceback))


def check_memory_pool(device, pool_id, live_storages_ptrs: List[StorageWeakRefWrapper]):
    assert all(
        isinstance(elem, StorageWeakRefWrapper) for elem in live_storages_ptrs
    )  # noqa: C419
    unique_storages = {stor.data_ptr() for stor in live_storages_ptrs if stor()}

    # check if there is a divergence first, then do the expensive snapshot call after
    # we know it will error
    if torch._C._cuda_checkPoolLiveAllocations(device, pool_id, unique_storages):
        return

    # at this point we are past the fast-path. we have seen rare cases where a dead tensor is dead,
    # but hasn't been gc'd yet, and gives false positive for allocated_not_in_live_storages
    gc.collect()

    segments = get_cudagraph_segments(pool_id)

    allocated_not_in_live_storages = {}

    for segment in segments:
        addr = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                if addr not in unique_storages:
                    allocated_not_in_live_storages[addr] = block
                else:
                    unique_storages.remove(addr)

            addr += block["size"]

    torch._check(
        len(unique_storages) == 0,
        lambda: f"These storage data ptrs are not allocated in pool {pool_id} but should be {unique_storages}",
    )

    if allocated_not_in_live_storages != 0:
        formatted = []
        for dp, block in allocated_not_in_live_storages.items():
            trace = (
                format_tb(block["history"][-1]) if block.get("history", None) else None
            )
            formatted.append(f"Data Pointer: {dp}, history: \n{trace}")
        formatted_s = "\n".join(formatted)
        msg = (
            f"These live storage data ptrs are in the cudagraph pool but not "
            f"accounted for as an output of cudagraph trees: \n\n{formatted_s}"
        )
        raise RuntimeError(msg)


class ExecutionState(Enum):
    """
    Represents the state of the CUDAGraph Tree. Will be None if there is no live current memory allocated
    in the cuda graph pool. Otherwise will reflect the state of the most recently executed node.
    """

    NONE = auto()
    WARMUP = auto()
    RECORDING = auto()
    EXECUTION = auto()


class CompilationMode(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    INFERENCE = auto()


class CUDAGraphTreeManager:
    """
    Groups individual recordings or executions of cuda graphs into a tree of recordings,
    and checks required invariants, and manages warmups of graphs.

    When graphs are recorded in the same tree, it enforces subsequent execution
    to follow the same order and have the same output tensor livespans. To remove
    unnecessary coupling of cuda graphs (and additional imposed invariants),
    the tree manager will end a currently recording tree whenever it is valid - when
    the memory pool no longer has any live allocations.

    We ignore outputs from a previous generation that correspond to prior model outputs.
    Currently this is hardcoded `GenerationTracker.generation` tracked in torch dynamo.
    # TODO: make generation increment configurable, warn on overwrite.

    We run graph warmups in the cudagraph memory pool and return the result on the first invocation
    of a function. For many models it is important to reclaim activations as you run the backward.
    If we were to warm up the model and keep an extra copy of the inputs around to subsequently
    use for recording, we would incur a memory penalty. Additionally, if we are part way through training
    your model and need to recompile, memory will be allocated to the cuda graph pool, so we run this
    warmup run in the cuda graph memory pool. As for recording, warm up needs the state of live tensors
    to be accurately reflected so we checkpoint the allocator state if we need to warm up following graph
    replay.
    """

    def __init__(self, device_index: int):
        # roots are functions which have no dependencies on an other node. I.e.,
        # when they are first invoked, none of their inputs are outputs are outputs
        # of another node, nor are there any live outputs of another node whose
        # liveness would create a dependency.
        self.roots: Dict[FunctionID, List[CUDAGraphNode]] = defaultdict(list)

        # mapping from function id to wrapped function
        self.ids_to_funcs: Dict[FunctionID, WrappedFunction] = {}

        self.ids_to_stack_traces: Dict[FunctionID, StackTraces] = {}

        self.warmed_up_functions: Set[FunctionID] = set()
        # if we fail to increment generation, and are stuck warming up,
        # only warn on each function once
        self.warned_functions: Set[FunctionID] = set()
        torch._C._set_cached_tensors_enabled(True)

        # NB: cuda caching allocator will remember the stream a segment is allocated to
        # and only allocate that segment to the same stream. we need to use a single stream
        # for all allocations to the memory pool, otherwise the allocations to separate streams
        # will not be reused; separate recordings would have use the same memory pool, but not
        # the same memory.

        with torch.cuda.device(device_index):
            torch.cuda.synchronize()
            self.stream = torch.cuda.Stream()
            self.stream.wait_stream(torch.cuda.current_stream())

            self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()
            # Keeps Memory Pool Alive
            self.graph = torch.cuda.CUDAGraph()

            self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()

            with warnings.catch_warnings(record=True), torch.cuda.graph(
                self.graph,
                pool=self.cuda_graphs_thread_pool,
                stream=self.stream,
            ):
                pass

        self.graph_counter = itertools.count(0)
        self.func_counter = itertools.count(0)

        # whether we the current node is in a state of warmup, recording, execution. If
        # there is no current node the state will be ExecutionState.None.
        self.path_state = ExecutionState.NONE
        self.device_index = device_index

        # the most recently invoked cudagraph wrapping of a function. Will be None
        # when there is no output from a previous recording or execution whose memory
        # we need to respect in the cuda caching allocation. If you incremented generation,
        # this will also be none, as ignore those allocations.
        self.current_node: Optional[CUDAGraphNode] = None

        # current generation of cudagraph invocations. when torch.compile is run
        # we increment the current generation. are willing to ignore live outputs
        # of a previous generation in checking liveness.
        self.current_gen: int = -1

        # number of instances we are in execution and failed to match to an
        # existing child
        self.debug_fail_counter = 0
        # number of instances we had to checkpoint the function
        self.debug_checkpointing_counter = 0

        self.id_to_mode: Dict[FunctionID, CompilationMode] = {}

        # Note: [Backward Generation Handling]
        # We generally perform a sequence of forward executions followed by backward executions.
        # If multiple torch.compile wrapped forwards are executed with their backwards pending,
        # we should not disregard the outputs from a prior torch.compile since the entire training
        # loop hasn't completed.  Occasionally, a backward pass corresponding to a forward pass may
        # not be executed, so we cannot wait for all pending forward pass backward completions, so
        # we cannot wait for all backwards to have been invoked. Instead we wait for a single backward
        # invocation. Triggering a backward pass typically doesn't lead to another torch.compile
        # invocation, making it less likely for the generation to increase between multiple
        # backward calls. The following use case is covered by this approach:
        # mod1 = torch.compile(...)
        # mod2 = torch.compile(...)
        # mod2(mod1(x)).sum().backward()

        self.running_forwards_with_pending_backwards = False

    def run(self, new_inputs: List[Tensor], function_id: FunctionID):
        assert self.graph is not None, "Running CUDAGraph after shutdown"
        out = self._run(new_inputs, function_id)

        # The forwards are only pending following invocation, not before
        mode = self.id_to_mode[function_id]
        if mode == CompilationMode.FORWARD:
            self.running_forwards_with_pending_backwards = True
        elif mode == CompilationMode.BACKWARD:
            self.running_forwards_with_pending_backwards = False

        return out

    def set_to_running_backward(self):
        self.running_forwards_with_pending_backwards = False

    def _run(self, new_inputs: List[Tensor], function_id: FunctionID):
        # we will try to end the current execution lazily, since
        # we dont want to do unnecessary checking of the existing outputs
        # on the hot path, but both recording and warmup only happen once
        # so we check up front
        if self.in_recording:
            self.try_end_curr_recording(function_id)

        if self.in_warmup:
            self.try_end_curr_warmup(function_id)

        # warming up a function and subsequentally recording may use different memory addresses
        # because both depend on the state of the caching allocator. if we warm up graph A,
        # then warm up graph B and make more allocations, the subsequent recording of A will not
        # necessarily use the same addresses as in the warm up. Thus any warm up of a node can only
        # be followed by warm up runs.
        if (
            not (
                function_id in self.warmed_up_functions
                or config.triton.skip_cudagraph_warmup
            )
        ) or self.in_warmup:
            # If we are in the middle of executing cuda graphs, then we need to checkpoint memory state.
            # Both Recording and Warmup will be reflected in the allocator and dont need changes
            if self.path_state == ExecutionState.EXECUTION:
                self.apply_checkpoint_execution_state_in_allocator()

            return self.run_eager(new_inputs, function_id)

        child_nodes = (
            self.roots if self.current_node is None else self.current_node.children
        )

        if not self.in_recording:
            for child in child_nodes[function_id]:
                # here we are checking memory consistency between recording and execution,
                # as well as things like stability of tensor locations, etc
                # and other
                if child.check_invariants(new_inputs):
                    return self.execute_node(child, new_inputs)

            # now that we know the new function can't be run as a child of the
            # current node, if it is a root, try to end the current execution.
            # as noted above, we want to do this lazily to avoid having to
            # check all existing outputs
            if self.current_node is not None and function_id in self.roots:
                self.try_end_curr_execution()

                # run again to hit the root matching case which must succeed
                if self.current_node is None:
                    return self.run(new_inputs, function_id)

            # at this point, we necessarily will do a new recording
            self.debug_fail_counter += 1

            self.try_end_curr_execution()
            if self.current_node is not None:
                self.apply_checkpoint_execution_state_in_allocator()

        # now, we are in a recording state !
        return self.record_function(new_inputs, function_id)

    def shutdown(self):
        """
        Remove all cached tensors in all nodes. Because cached tensors can hold gradients which in turn
        might reference a backward which invokes a CUDA Graph Node, we have to manually clear them on shutdown
        to avoid a reference cycle.
        """
        nodes = []
        for roots in self.roots.values():
            nodes.extend(roots)

        while nodes:
            node = nodes.pop()
            for children in node.children.values():
                nodes.extend(children)
            node.remove_node_cached_tensors()
            node.graph = None

        self.graph = None
        self.roots = None  # type: ignore[assignment]
        self.current_node = None

    def record_function(self, new_inputs, function_id) -> List[Optional[Tensor]]:
        graph_id = self.new_graph_id()
        log.debug(
            "Recording function %d of graph recording id %d",
            function_id.id,
            graph_id.id,
        )
        torch.cuda.synchronize()
        node = CUDAGraphNode(
            self.ids_to_funcs[function_id],
            graph_id,
            self.current_node,
            new_inputs,
            self.cuda_graphs_thread_pool,
            self.device_index,
            self.ids_to_stack_traces[function_id],
            self.stream,
        )
        if self.current_node is None:
            self.roots[function_id].append(node)
        else:
            self.current_node.add_child(function_id, node)
        self.current_node = node
        self.path_state = ExecutionState.RECORDING
        self.update_generation()
        torch.cuda.synchronize()
        return node.run_first_inputs(new_inputs)

    def execute_node(self, node: CUDAGraphNode, new_inputs) -> List[Optional[Tensor]]:
        self.current_node = node
        self.path_state = ExecutionState.EXECUTION
        self.update_generation()
        return node.run(new_inputs)

    def run_eager(self, new_inputs, function_id: FunctionID):
        # this is only stored on current node, because when we start a new path,
        # we will deallocate it
        already_warm = function_id in self.warmed_up_functions
        if not already_warm:
            log.debug("Running warmup of function %d", function_id.id)
        else:
            log.debug(
                "Running eager of function %d because ancestor needed to warm up",
                function_id.id,
            )
        self.warmed_up_functions.add(function_id)
        node = CUDAWarmupNode(
            self.ids_to_funcs[function_id],
            self.current_node,
            self.cuda_graphs_thread_pool,
            self.graph,
            self.device_index,
            self.ids_to_stack_traces[function_id],
            self.stream,
            already_warm,
        )
        self.current_node = node
        self.path_state = ExecutionState.WARMUP
        self.update_generation()
        return node.run(new_inputs)

    def new_graph_id(self) -> GraphID:
        return GraphID(next(self.graph_counter))

    def new_func_id(self) -> FunctionID:
        return FunctionID(next(self.func_counter))

    def add_function(
        self,
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
    ) -> Tuple[Callable[..., Any], List[Optional[Tensor]]]:
        id = self.new_func_id()
        self.ids_to_stack_traces[id] = stack_traces
        self.ids_to_funcs[id] = WrappedFunction(model, static_input_idxs, id)
        self.id_to_mode[id] = mode
        fn = functools.partial(self.run, function_id=id)

        # container needs to set clean up when fn dies
        get_container(self.device_index).add_strong_reference(fn)
        return fn, fn(inputs)

    @property
    def in_recording(self):
        return self.path_state == ExecutionState.RECORDING

    @property
    def in_warmup(self):
        return self.path_state == ExecutionState.WARMUP

    def get_roots(self) -> Iterator[CUDAGraphNode]:
        for nodes in self.roots.values():
            yield from nodes

    @property
    def current_node(self):
        return self._current_node

    @current_node.setter
    def current_node(self, value):
        self._current_node = value
        if value is None:
            self.path_state = ExecutionState.NONE

    def update_generation(self):
        self.current_gen = self.get_curr_generation()

    @staticmethod
    def get_curr_generation() -> int:
        if MarkStepBox.mark_step_counter != 0:
            return MarkStepBox.mark_step_counter

        return GenerationTracker.generation

    @staticmethod
    def user_invoked_mark_step():
        return MarkStepBox.mark_step_counter != 0

    def can_start_new_generation(self) -> bool:
        if not self.in_new_torch_compile_invocation():
            return False

        if self.user_invoked_mark_step():
            return True

        return not self.running_forwards_with_pending_backwards

    def in_new_torch_compile_invocation(self):
        return self.current_gen != self.get_curr_generation()

    def try_end_curr_recording(self, function_id: FunctionID) -> None:
        """
        Check if the current recording can be terminated, either because all outputs of the
        previously recorded node are dead or because it was executed in a different
        generation. Will set current_node to None and in_recording to False if successful.
        """
        assert self.in_recording
        assert self.current_node is not None

        # multiple invocations, allow overwriting the previous generation
        if self.can_start_new_generation():
            self.dealloc_current_path_weakrefs()
            self.clear_current_path_state_and_set_to_none()
            return

        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()
            return

        self.check_warn_on_unable_to_start_executing(function_id)

    def try_end_curr_execution(self) -> None:
        """
        Check if the current executing node can be terminated, either because all outputs of the
        previously executed node are dead or because it was executed in a different generation.
        Will set current_node to None if successful.
        """

        assert not self.in_recording
        if self.current_node is None:
            return

        if self.can_start_new_generation():
            self.clear_current_path_state_and_set_to_none()
            return

        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()

    def try_end_curr_warmup(self, function_id: FunctionID):
        if self.can_start_new_generation():
            self.dealloc_current_path_weakrefs()
            self.current_node = None
            return

        if self.current_node.all_outputs_are_dead():
            self.current_node = None
            return

        self.check_warn_on_unable_to_start_executing(function_id)

    def check_warn_on_unable_to_start_executing(self, function_id: FunctionID):
        "Warn if we in a potential loop where we are unable to hit fast path"
        if (
            function_id in self.warned_functions
            or not self.in_new_torch_compile_invocation()
        ):
            return

        existing_nodes = [
            node
            for node in self.current_node._path_from_root
            if node.wrapped_function.id == function_id
        ]

        if len(existing_nodes) <= 1:
            return

        # repeated same pattern
        parents = {
            n.parent.wrapped_function.id
            for n in itertools.chain(existing_nodes, (self.current_node,))
            if n.parent is not None
        }
        if len(parents) == len(existing_nodes):
            return

        self.warned_functions.add(function_id)
        warnings.warn(
            "Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards. "
            "Consider running with torch.no_grad() or using torch._inductor.cudagraph_mark_step_begin() "
            "before each model invocation"
        )

    def dealloc_current_path_weakrefs(self):
        # TODO: we could also allow the these weak refs to continue to be allocated,
        # but that adds some complications.
        for node in self.current_node._path_from_root:
            assert len(node.tensor_weakrefs) == len(node.stack_traces)
            for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
                ten = None if t is None else t()
                if ten is None:
                    continue

                stack_trace = (
                    stack_trace.strip()
                    if stack_trace
                    else "[Could not find stack trace]"
                )
                msg = (
                    "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. "
                    f"Stack trace: {stack_trace}. "
                    "To prevent overwriting, clone the tensor outside of torch.compile() "
                    "before running the model again."
                )
                torch._C._set_storage_access_error_msg(ten, msg)

        deleted = set()
        for storage_ref in self.current_node.path_live_weakrefs():
            if storage_ref() and storage_ref.data_ptr() not in deleted:
                deleted.add(storage_ref.data_ptr())
                torch._C._free_And_Remove_DeleterFn(storage_ref())

    def clear_current_path_state_and_set_to_none(self):
        self.current_node.clear_path_state()
        self.current_node = None

    def apply_checkpoint_execution_state_in_allocator(self):
        """
        Checkpoint the current execution state in the caching allocator so that
        additional cudagraph recordings can be made respecting existent live storages.
        """
        self.debug_checkpointing_counter += 1
        log.debug(
            "Checkpointing cuda caching allocator state. Number of checkpoints %d",
            self.debug_checkpointing_counter,
        )

        state = self.current_node.checkpointed_caching_state
        device = self.current_node.device
        assert state is not None and device is not None

        # currently we deallocate on instead of allowing stale recordings
        stale_storages: List[int] = []

        # remove cached tensors, otherwise they would prevent memory from being
        # reclaimed in subsequent recordings
        self.current_node.remove_path_cached_tensors()
        live_storages_wrappers = list(self.current_node.path_live_weakrefs())

        live_storages_weak_refs = [t() for t in live_storages_wrappers]
        ptrs_to_deallocate = self.current_node.data_ptrs_dead_since_invocation()
        torch._C._cuda_setCheckpointPoolState(
            device, state, stale_storages, live_storages_weak_refs
        )

        # NB: deduplicate aliased outputs
        for ptr in set(ptrs_to_deallocate):
            torch._C._cuda_cudaCachingAllocator_raw_delete(ptr)

        # Now the live blocks should be exactly equal to the live storages in private pool
        if config.triton.slow_path_cudagraph_asserts:
            check_memory_pool(
                self.device_index, self.cuda_graphs_thread_pool, live_storages_wrappers
            )
            for wrapper in live_storages_wrappers:
                assert wrapper()
                assert torch._C._has_Standard_Deleter(wrapper())
                assert wrapper.data_ptr() not in ptrs_to_deallocate

    def live_cudagraph_pool_storages_in_curr_execution(
        self,
    ) -> List[StorageWeakRefPointer]:
        if self.current_node is None:
            return []
        # explicitly ignoring previous recorded outputs from past path
        return [t() for t in self.current_node.path_live_weakrefs()]
