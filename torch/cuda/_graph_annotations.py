"""Annotate CUDA graph kernel nodes during capture.

During CUDA graph capture, ``mark_kernels`` records the current capture
frontier and the direct dependents already attached to that frontier.
On scope exit it walks only the newly added dependent edges to find the
nodes created within the scope. Each kernel, memcpy, or memset node found
is annotated by its ``toolsId`` so it can later be matched to profiler
trace events.

``mark_kernels`` now snapshots capture state from whatever stream is
current on scope entry, so that stream must already be participating in
the capture. ``mark_stream`` handles this by starting ``mark_kernels``
before switching to the target stream.

The annotations are baked into a Chrome profiler trace by
``prof.export_chrome_trace(path, cuda_graph_annotations=get_kernel_annotations())``.

Requires ``cuda.bindings`` package and a CUDA driver that supports
``cudaGraphNodeGetToolsId`` (CUDA >= 13.1 or appropriate cuda-compat).
When unavailable, ``mark_kernels`` silently becomes a no-op.

Annotation recording is enabled per capture via ``torch.cuda.graph``'s
``enable_annotations`` argument, which also resolves and remaps the recorded
annotations to the exec graph automatically on context exit.

Usage during capture::

    from torch.cuda._graph_annotations import mark_kernels, get_kernel_annotations

    with torch.cuda.graph(graph, enable_annotations=True):
        with mark_kernels("phase_A"):
            y = workload_a(x)
        with mark_kernels("phase_B"):
            z = workload_b(y)

    annotations = get_kernel_annotations()

When you need to drive this outside the context manager's automatic path,
``resolve_and_remap(graph)`` is shorthand for ``resolve_pending_annotations()``
followed by ``remap_to_exec_graph(graph)``; call those directly for finer
control (e.g. resolving once before remapping several graphs).
"""

from __future__ import annotations

import importlib.metadata
import pickle
import threading
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING, TypeAlias
from typing_extensions import deprecated


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

import torch
from torch.cuda._utils import (
    _check_cuda_bindings,
    _check_cuda_bindings_driver,
    _HAS_CUDA_BINDINGS,
)


try:
    from cuda.bindings import (  # pyrefly: ignore[missing-import]
        driver as _cuda_driver,
        runtime as _cuda_runtime,
    )
except ImportError:
    _cuda_driver = None  # type: ignore[assignment]
    _cuda_runtime = None  # type: ignore[assignment]

if not _HAS_CUDA_BINDINGS:
    # This module imports the bindings itself, so keep it in step with the shared gate,
    # which also reports them absent on ROCm -- where they import but cannot work.
    _cuda_driver = None  # type: ignore[assignment]
    _cuda_runtime = None  # type: ignore[assignment]


logger = getLogger(__name__)


_CaptureState: TypeAlias = tuple[Any, list[Any]]
_ExistingDirectDependents: TypeAlias = dict[int, set[int]]


# Tri-state: None = not probed, True = available, False = unavailable.
# Deferred to first use to avoid premature CUDA initialization.
_tools_id_available: bool | None = None

# Whether annotation recording is active. Scoped to a ``torch.cuda.graph``
# capture: set by the context manager on entry (from its ``enable_annotations``
# argument) and cleared on exit. When False, mark_kernels/mark_stream and the
# capture-id stamp are no-ops. Module-level (not thread-local) because capture
# can span threads (e.g. autograd).
_annotations_enabled: bool = False

# How the active capture discovers which nodes a ``mark_kernels`` scope contains:
#
# "edge_walk" -- snapshot the capture frontier on scope entry and walk the dependent
#   edges added by scope exit. Needs nothing but the CUDA runtime, but only sees nodes
#   reachable from the snapshotted frontier and rescans a nested scope's nodes once per
#   enclosing scope.
# "cupti" -- CUPTI names each node as it is created (see
#   ``torch.cuda._graph_node_callbacks``), so the scope only has to publish which
#   annotation is active; ``_active_scopes`` below is that ambient state.
#
# Set alongside _annotations_enabled by ``torch.cuda.graph``; only meaningful while
# annotations are enabled.
_annotation_backend: str = "edge_walk"

# The annotations currently published for nodes as they are created, innermost last: each
# entry already includes its enclosing scopes, with the inner one winning shared keys, so
# the innermost entry is what to annotate with. Only the "cupti" backend reads this (at
# node-creation time); the edge walk derives the same information from graph topology after
# the fact.
#
# Entries are (annotation, region): ``region`` is the TLS region a backward bracket
# installed alongside its entry, and None for a ``mark_kernels`` scope. A bracket's entry
# has no guaranteed pop -- the autograd engine skips the posthook when the node raises --
# and this list is module-global (capture can span threads), so nothing restores it on
# unwind. What does get restored is the region TLS, via the engine task's
# ThreadLocalStateGuard, so an entry whose region is no longer current has unwound and is
# dropped by current_annotation() below. Without that, a backward failure caught inside the
# capture would leave every later unmarked node wearing the failed node's annotation.
_active_scopes: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

# Id of the top-level capture graph of the active capture, read once at capture_begin (see
# maybe_stamp_capture_root) and then used by everyone who needs it: mark_kernels compares
# against it to detect a scope inside a conditional-node body, the CUPTI backend filters out
# body nodes the same way, and the graph object is stamped with it for the later remap.
_capture_root_graph_id: int | None = None


def capture_root_graph_id() -> int | None:
    """The active capture's top-level graph id, or ``None`` outside a capture that recorded
    one. Not a public API."""
    return _capture_root_graph_id


def _set_annotations_enabled(enabled: bool) -> None:
    """Set whether annotation recording is active. Used by ``torch.cuda.graph``
    to scope annotations to a capture; not a public API."""
    global _annotations_enabled, _capture_root_graph_id
    _annotations_enabled = enabled
    if not enabled:
        _capture_root_graph_id = None
        # A capture that raised mid-scope would otherwise leak its scopes into the next one.
        _active_scopes.clear()


def _set_annotation_backend(backend: str) -> None:
    """Set how ``mark_kernels`` scopes discover their nodes for this capture.

    Separate from :func:`_set_annotations_enabled` because the two are decided at different
    points: recording is enabled before ``capture_begin`` (``maybe_stamp_capture_root`` is
    gated on it), while the backend is only final once the CUPTI callback has actually been
    armed, which needs the capture live. Not a public API."""
    global _annotation_backend
    _annotation_backend = backend


def current_annotation() -> dict[str, Any] | None:
    """The annotation to attribute a node created right now to, or ``None``.

    Entries whose region has been restored out from under them are dropped first: see
    :data:`_active_scopes`, a backward bracket whose node raised leaves one behind. Not a
    public API."""
    while _active_scopes:
        annotation, region = _active_scopes[-1]
        if region is None or region is _current_region():
            return annotation
        _active_scopes.pop()
    return None


def record_node_annotation(tools_id: int, annotation: dict[str, Any]) -> None:
    """Attribute one graph node, keyed by its capture-side ``toolsId``.

    The CUPTI backend's entry point into the same store the edge walk fills, so both
    backends are remapped to exec-graph ids by ``remap_to_exec_graph`` identically. Not a
    public API."""
    _merge_annotation(tools_id, annotation)


def _graph_id(graph: Any) -> int:
    """The driver's unique id for a ``cudaGraph_t``.

    Identifies a graph across its whole lifetime, unlike the handle itself: that is a
    pointer, so a graph created after another is destroyed can reuse the value and
    compare equal to it."""
    return _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetId(graph)  # pyrefly: ignore[missing-attribute]
    )


def maybe_stamp_capture_root(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Record the top-level capture graph of the capture just begun, and stamp it on the
    graph for the later remap.

    Called by ``torch.cuda.graph`` right after ``capture_begin``, while the current stream is
    still capturing into the top-level graph. It has to go through the stream because the
    ``cudaGraph_t`` does not exist until ``capture_end`` -- ``raw_cuda_graph()`` raises before
    then -- but the template graph keeps that one id for its whole life, so this single read
    serves every consumer: the conditional-body checks in ``mark_kernels``, the CUPTI
    backend's body-node filter, and ``remap_to_exec_graph`` via the stamp below. Not a public
    API."""
    global _capture_root_graph_id
    _capture_root_graph_id = None
    if not _annotations_enabled or _is_tools_id_unavailable():
        return
    stream_handle = torch.cuda.current_stream().cuda_stream
    state = _get_capture_state(stream_handle)
    if state is None:
        return
    _capture_root_graph_id = _graph_id(state[0])
    torch_cuda_graph._capture_graph_id = _capture_root_graph_id
    # Fresh capture: annotations are keyed by this capture id until remapped.
    torch_cuda_graph._remapped_exec_id = None


def _probe_tools_id() -> bool:
    """Probe whether cudaGraphNodeGetToolsId is supported by the driver.

    Calls with a null node and accepts only the errors that prove the API is
    really there: cudaErrorInvalidValue, i.e. the driver got as far as rejecting
    the argument. Every other status means we cannot use it -- an old NVIDIA
    driver answers cudaErrorCallRequiresNewerDriver, while an environment with no
    CUDA driver behind the bindings at all (ROCm, where cuda-bindings can still
    import) answers cudaErrorInsufficientDriver. Allowlisting rather than
    denylisting keeps a new failure mode from reading as "supported".
    """
    if not hasattr(_cuda_runtime, "cudaGraphNodeGetToolsId"):
        # API is missing from cuda-bindings - likely version too old
        cuda_bindings_version = importlib.metadata.version("cuda-bindings")

        logger.warning(
            "cudaGraphNodeGetToolsId API not found in cuda-bindings. "
            "Current version: %s, required: >= 13.1.0. "
            "CUDA graph kernel annotations will be disabled. "
            "To enable annotations, upgrade cuda-bindings: "
            "pip install --upgrade cuda-bindings",
            cuda_bindings_version,
        )
        return False
    err, *_ = _cuda_runtime.cudaGraphNodeGetToolsId(
        0
    )  # pyrefly: ignore[missing-attribute]
    if (
        err
        not in (
            _cuda_runtime.cudaError_t.cudaSuccess,  # pyrefly: ignore[missing-attribute]
            _cuda_runtime.cudaError_t.cudaErrorInvalidValue,  # pyrefly: ignore[missing-attribute]
        )
    ):
        logger.info(
            "cudaGraphNodeGetToolsId is unusable (%s); it needs a CUDA driver >= 13.1 "
            "or an equivalent cuda-compat. CUDA graph kernel annotations will be "
            "disabled",
            err,
        )
        return False
    return True


def _is_tools_id_unavailable() -> bool:
    """Return True if cudaGraphNodeGetToolsId is not usable."""
    global _tools_id_available
    if not _HAS_CUDA_BINDINGS:
        return True
    if _tools_id_available is not None:
        return not _tools_id_available
    _tools_id_available = _probe_tools_id()
    return not _tools_id_available


def is_available() -> bool:
    r"""is_available() -> bool

    Return whether CUDA graph annotation recording is supported.

    Requires a CUDA device, the ``cuda-bindings`` package, and a driver
    that supports ``cudaGraphNodeGetToolsId`` (CUDA >= 13.1 or an
    equivalent cuda-compat package). When this returns ``False``,
    :func:`mark_kernels` is a silent no-op and no annotations are
    recorded.

    The first call may probe the CUDA driver; the result is cached.
    """
    return torch.cuda.is_available() and not _is_tools_id_unavailable()


def _get_capture_state(stream: Any) -> _CaptureState | None:
    """Return ``(graph, frontier)`` for an active capture, else ``None``."""
    status, _id, graph, _deps, _edge_data, _num_deps = _check_cuda_bindings(
        _cuda_runtime.cudaStreamGetCaptureInfo(  # pyrefly: ignore[missing-attribute]
            stream
        )
    )
    if (
        status
        != _cuda_runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive  # pyrefly: ignore[missing-attribute]
    ):
        return None
    return graph, list(_deps[:_num_deps])


def _get_root_nodes(graph: Any) -> list[Any]:
    """Return the current root nodes for the graph."""
    _, num_roots = _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetRootNodes(  # pyrefly: ignore[missing-attribute]
            graph
        )
    )
    if num_roots == 0:
        return []
    roots, num_roots = _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetRootNodes(  # pyrefly: ignore[missing-attribute]
            graph, pNumRootNodes=num_roots
        )
    )
    return list(roots[:num_roots])


def _get_dependent_nodes(node: Any) -> list[Any]:
    """Return the direct dependents of a graph node."""
    _, _, num_dependents = _check_cuda_bindings(
        _cuda_runtime.cudaGraphNodeGetDependentNodes(  # pyrefly: ignore[missing-attribute]
            node
        )
    )
    if num_dependents == 0:
        return []
    dependents, _edge_data, num_dependents = _check_cuda_bindings(
        _cuda_runtime.cudaGraphNodeGetDependentNodes(  # pyrefly: ignore[missing-attribute]
            node, pNumDependentNodes=num_dependents
        )
    )
    return list(dependents[:num_dependents])


def _get_node_type(node: Any) -> Any:
    """Return graph node type without tripping runtime bugs on newer node kinds.

    The runtime ``cudaGraphNodeGetType`` can return ``cudaErrorUnknown`` (999)
    for valid nodes whose driver type is ``CU_GRAPH_NODE_TYPE_BATCH_MEM_OP``.
    Query via the driver API instead.
    """
    return _check_cuda_bindings_driver(
        _cuda_driver.cuGraphNodeGetType(node)  # pyrefly: ignore[missing-attribute]
    )


def _collect_descendants(
    start_nodes: list[Any],
    *,
    existing_direct_dependents: _ExistingDirectDependents | None = None,
    include_start_nodes: bool = False,
) -> dict[int, Any]:
    """Walk dependent edges starting at ``start_nodes``.

    ``existing_direct_dependents`` maps each node in ``start_nodes`` to
    the direct dependent node keys that were already present at scope
    entry. Those edges are skipped so the traversal only follows nodes
    added after scope entry.
    """
    existing_direct_dependents = existing_direct_dependents or {}
    seen = {int(node) for node in start_nodes}
    descendants: dict[int, Any] = {}
    stack = list(start_nodes)

    if include_start_nodes:
        for node in start_nodes:
            descendants[int(node)] = node

    while stack:
        node = stack.pop()
        old_dependents = existing_direct_dependents.get(int(node), set())
        for dependent in _get_dependent_nodes(node):
            dependent_key = int(dependent)
            if dependent_key in old_dependents or dependent_key in seen:
                continue
            seen.add(dependent_key)
            descendants[dependent_key] = dependent
            stack.append(dependent)

    return descendants


# toolsId -> the node's merged annotation. Exactly one dict per node: every writer goes
# through _merge_annotation, and a node is only written more than once when scopes overlap
# on it, which is precisely what the merge resolves.
_kernel_annotations: dict[int, dict[str, Any]] = {}


def _merge_annotation(tools_id: int, annotation: Any) -> None:
    """Merge one scope's annotation into a node's entry; the first write wins per key.

    First-wins is what makes nested scopes resolve inner-first: scopes are recorded in
    completion order, so the innermost scope containing a node reaches it first and keeps
    the keys it shares with its enclosing scopes, while their extra keys still come
    through. Non-dict annotations are normalized to ``{"name": ...}``; ``mark_kernels``
    already does that for strings, but the store is written by other callers too.
    """
    incoming = annotation if isinstance(annotation, dict) else {"name": annotation}
    entry = _kernel_annotations.get(tools_id)
    if entry is None:
        _kernel_annotations[tools_id] = dict(incoming)
        return
    for key, value in incoming.items():
        entry.setdefault(key, value)


def annotation_for(tools_id: int) -> dict[str, Any] | None:
    """The merged annotation recorded for one graph node, or ``None``. The in-process
    accessor behind the profiler's graph annotation resolver, handing out the stored dict
    rather than the list-wrapped public view. Not a public API."""
    return _kernel_annotations.get(tools_id)


# Node types we annotate (kernels, memcpys, memsets, batch mem ops, event
# record/wait nodes, and host nodes), as driver node-type enums. Event and host
# nodes are stream-ordered but do not occupy the device; annotating them lets the
# profiler place their spans on the intended stream lane. Initialized lazily to
# avoid touching cuda.bindings at import time.
_ANNOTATABLE_TYPES: set[Any] | None = None

# The same set as raw driver enum values, for callers handed a plain int rather than a
# CUgraphNodeType (CUPTI reports ``GraphData.node_type`` that way). Derived from
# _get_annotatable_types so the membership is defined in exactly one place; cached because
# the CUPTI backend consults it once per node created.
_ANNOTATABLE_TYPE_VALUES: frozenset[int] | None = None


def _get_annotatable_types() -> set[Any]:
    global _ANNOTATABLE_TYPES
    if _ANNOTATABLE_TYPES is None:
        node_types = _cuda_driver.CUgraphNodeType  # pyrefly: ignore[missing-attribute]
        _ANNOTATABLE_TYPES = {
            node_types.CU_GRAPH_NODE_TYPE_KERNEL,
            node_types.CU_GRAPH_NODE_TYPE_MEMCPY,
            node_types.CU_GRAPH_NODE_TYPE_MEMSET,
            node_types.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            node_types.CU_GRAPH_NODE_TYPE_EVENT_RECORD,
            node_types.CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            node_types.CU_GRAPH_NODE_TYPE_HOST,
        }
    return _ANNOTATABLE_TYPES


def _get_annotatable_type_values() -> frozenset[int]:
    """:func:`_get_annotatable_types` as raw driver enum values."""
    global _ANNOTATABLE_TYPE_VALUES
    if _ANNOTATABLE_TYPE_VALUES is None:
        _ANNOTATABLE_TYPE_VALUES = frozenset(int(t) for t in _get_annotatable_types())
    return _ANNOTATABLE_TYPE_VALUES


# Node types whose work lives in a separate cudaGraph_t (child graphs, conditional
# bodies). The dependent-edge walk does not descend into such a node, and the nodes
# inside are numbered in the body graph's id space, which remap_to_exec_graph never
# rekeys -- so annotations there would be silently lost. See mark_kernels. Initialized
# lazily, like _ANNOTATABLE_TYPES above: _cuda_driver is None when cuda.bindings is
# absent, so reading the enum at import time would break `import torch`.
_NESTED_GRAPH_TYPES: set[Any] | None = None


def _get_nested_graph_types() -> set[Any]:
    global _NESTED_GRAPH_TYPES
    if _NESTED_GRAPH_TYPES is None:
        node_types = _cuda_driver.CUgraphNodeType  # pyrefly: ignore[missing-attribute]
        _NESTED_GRAPH_TYPES = {
            node_types.CU_GRAPH_NODE_TYPE_GRAPH,
            node_types.CU_GRAPH_NODE_TYPE_CONDITIONAL,
        }
    return _NESTED_GRAPH_TYPES


# Pending scopes: (annotation, toolsIds discovered for the scope).
#
# Notably, this is NOT the current dynamic scope; instead, we fill this in
# after we exit each mark_kernels region, and this holds the FULL extra
# CUDA graph node attribution, steadily growing as we execute.
#
# Scopes are recorded as we exit a `mark_kernels`.  Later, when we resolve
# all of the annotations, we do merges on the annotations where
# first-recorded-wins (it works: you end up preferring inner scopes over
# outer scopes, what you'd expect).  For example:
#
#     with mark_kernels({"name": "outer", "color": "red"}):
#         with mark_kernels({"name": "inner"}):
#             y = x + 1  # toolsId: 101
#         z = y * 2      # toolsId: 102
#
# where the add captures kernel node with toolsId 101 and the mul 102. After
# executing ALL of this code, the _pending_scopes would be (inner exits first,
# so it is recorded first):
#
#     ({"name": "inner"}, [101])
#     ({"name": "outer", "color": "red"}, [101, 102])
#
# toolsId 101 appears in both lists, and so the first-recorded-wins rule then
# means we merge all the dicts to {"name": "inner", "color": "red"}.
_pending_scopes: list[tuple[Any, list[int]]] = []


# TLS slot carrying the current annotation region: the annotations of every
# open backward-annotating ``mark_kernels`` scope collapsed into one dict
# (inner scopes win common keys). It lives in ThreadLocalPythonObjects rather
# than a ContextVar or threading.local because that is the one Python-writable
# TLS that ``at::ThreadLocalState`` snapshots into autograd engine worker and
# device threads, so a backward bracket can re-establish the region for nodes
# created while it executes (higher-order grad, checkpoint recomputation) even
# across engine thread hops. Region dicts are never mutated after creation, so
# holding a reference is a snapshot.
_REGION_TLS_KEY = "cuda_graph_annotation_region"


def _current_region() -> dict[str, Any] | None:
    if torch._C._is_key_in_tls(_REGION_TLS_KEY):
        return torch._C._get_obj_in_tls(_REGION_TLS_KEY)
    return None


def _enter_region(collapsed: dict[str, Any]) -> dict[str, Any] | None:
    """Install ``collapsed`` as the current region; return the previous one."""
    prev = _current_region()
    torch._C._stash_obj_in_tls(_REGION_TLS_KEY, collapsed)
    return prev


def _exit_region(prev: dict[str, Any] | None) -> None:
    if prev is None:
        torch._C._remove_obj_from_tls(_REGION_TLS_KEY)
    else:
        torch._C._stash_obj_in_tls(_REGION_TLS_KEY, prev)


# node.metadata marker recording that a node's backward hook pair is already
# installed, so redundant firings of ``_freeze_region_hook`` (one per live
# push, e.g. nested scopes) attach only once. An identity sentinel so no
# other metadata user can collide with it.
_HOOKED_KEY: Any = object()


class _KernelScope(NamedTuple):
    """Capture-frontier snapshot taken at scope entry, consumed at scope exit."""

    graph: Any
    frontier: list[Any]
    entry_root_keys: set[int] | None
    entry_direct_dependents: _ExistingDirectDependents


def _begin_kernel_scope() -> _KernelScope | None:
    """Snapshot the current stream's capture frontier; ``None`` if not capturing."""
    stream = torch.cuda.current_stream().cuda_stream
    capture_state = _get_capture_state(stream)
    if capture_state is None:
        return None
    graph, frontier = capture_state

    entry_root_keys: set[int] | None = None
    entry_direct_dependents = {
        int(node): {int(dep) for dep in _get_dependent_nodes(node)} for node in frontier
    }
    if not frontier:
        entry_root_keys = {int(node) for node in _get_root_nodes(graph)}
    return _KernelScope(graph, frontier, entry_root_keys, entry_direct_dependents)


def _end_kernel_scope(scope: _KernelScope) -> list[int]:
    """Walk nodes captured since ``scope`` was begun; return their toolsIds."""
    if scope.frontier:
        scope_nodes = _collect_descendants(
            scope.frontier,
            existing_direct_dependents=scope.entry_direct_dependents,
        )
    else:
        new_roots = [
            node
            for node in _get_root_nodes(scope.graph)
            if int(node) not in (scope.entry_root_keys or set())
        ]
        scope_nodes = _collect_descendants(new_roots, include_start_nodes=True)

    annotatable = _get_annotatable_types()
    nested = _get_nested_graph_types()
    nested_seen: set[str] = set()
    tools_ids: list[int] = []
    for node in scope_nodes.values():
        node_type = _get_node_type(node)
        if node_type in nested:
            nested_seen.add(node_type.name)
        if node_type not in annotatable:
            continue
        tools_ids.append(
            _check_cuda_bindings(
                _cuda_runtime.cudaGraphNodeGetToolsId(  # pyrefly: ignore[missing-attribute]
                    node
                )
            )
        )

    if nested_seen:
        # The annotations recorded above are still correct -- the exec graph preserves
        # top-level node ids -- they just do not cover the nested body.
        warnings.warn(
            f"mark_kernels: this scope contains {sorted(nested_seen)} nodes; the work "
            "inside them is in a separate cudaGraph_t that this walk does not descend "
            "into, so it is left unannotated",
            stacklevel=4,
        )
    return tools_ids


class _BracketState(threading.local):
    """Per-thread stack of open backward brackets for one hook pair.

    Thread-local so concurrent executions of a retained node (separate
    graph tasks on different engine threads) cannot interleave each
    other's scope snapshots; within one thread pre/post pair up LIFO.

    If the node throws, its posthook never runs (the engine's
    call_function has no try/finally around fn) and the prehook's entry
    is left on the stack. That is harmless. Say a retained node runs
    three times on one thread and the second run throws:

        run 1: prehook push [e1]; posthook pops e1     -> []
        run 2: prehook push [e2]; node throws, no pop  -> [e2]
        run 3: prehook push [e2, e3]; posthook pops e3 -> [e2]

    Pops always take the top, so the leaked e2 sits below all later
    entries and every subsequent run still pops its own. The only loss
    is e2 itself: that run's kernels go unrecorded, which is moot since
    its backward failed. (The region/creation-hook TLS the prehook set
    is not leaked -- the engine task's ThreadLocalStateGuard restores
    those on unwind.)
    """

    def __init__(self) -> None:
        # Entries are (scope, prev_region, tagged_region, published, depth) or _SKIPPED.
        self.stack: list[Any] = []


# Bracket-stack marker for "prehook did nothing": distinct from an entry with
# scope=None, which means the region and creation hook were installed but no
# capture was active (the posthook must still pop them).
_SKIPPED: Any = object()


def _freeze_region_hook(node: Any) -> None:
    """Node creation hook: freeze the current region into ``node``'s bracket.

    Reads the region TLS slot and, if a region is open, installs the node's
    single backward hook pair closing over that snapshot. Every backward-
    annotating ``mark_kernels`` scope pushes one of these (and so does an
    executing hooked node, to cover nodes created during its backward), so
    a node created under nested scopes sees several identical firings; the
    ``node.metadata`` marker makes only the first attach.

    AccumulateGrad is always excluded: a leaf's node is created once and
    cached (possibly during warmup), so which scope would own it is an
    accident of first use, and its work (the ``.grad`` accumulation)
    belongs to the leaf rather than to any forward region. If accumulation
    ever needs annotation it should get a dedicated leaf-accumulation
    region, not first-use assignment.
    """
    if isinstance(node, torch._C._functions.AccumulateGrad):
        return
    if _HOOKED_KEY in node.metadata:
        return
    frozen = _current_region()
    if frozen is None:
        return
    node.metadata[_HOOKED_KEY] = True
    _attach_backward_hooks(node, frozen)


def _attach_backward_hooks(node: Any, frozen: dict[str, Any]) -> None:
    """Register one pre/post hook pair bracketing ``node``'s backward execution.

    ``frozen`` is the region that was current when the node was created:
    the annotations of every scope then open, already collapsed inner-wins.
    The pair brackets the node's execution the same way ``mark_kernels``
    brackets forward work, and splits on the backend the same way: under the
    edge walk it snapshots the capture frontier before and collects the new
    nodes after, under CUPTI it publishes itself as the ambient annotation
    for the nodes created while it runs. It runs on the autograd engine
    thread executing the node, which during a whole-graph capture
    participates in the capture; when backward runs outside a capture the
    prehook sees no active capture and the pair is a no-op.

    Executing the node is semantically re-entering its frozen region, so
    the prehook merges ``frozen`` over the live region (a ``mark_kernels``
    enclosing the whole ``backward()`` call, if any) and installs the
    result while the node runs. That gives nodes created during its
    backward (``create_graph=True``, checkpoint recomputation) the forward
    region's ownership -- a creation hook is pushed alongside so they get
    hooked even when this backward runs eagerly and no scope's hook is
    live -- and it puts a ``mark_kernels`` opened inside the execution
    dynamically inner, so it outranks ``frozen``, which outranks the
    enclosing scope. The posthook records the merged region once; scope
    completion order then yields exactly that precedence at resolve.

    The TLS pops are exception-safe without a finally: each engine task runs
    under an ``at::ThreadLocalStateGuard`` that restores both the
    creation-hook TLS and the region slot even when the node throws. The
    ambient-annotation entry the CUPTI path publishes is module-global and so
    is not restored that way; it is dropped on next read instead (see
    :data:`_active_scopes`).
    """
    state = _BracketState()

    def creation_hook(child: Any) -> None:
        _freeze_region_hook(child)

    def prehook(_grad_outputs: Any) -> None:
        if _is_tools_id_unavailable():
            state.stack.append(_SKIPPED)
            return
        # Re-establish ownership even when this backward is not itself
        # captured: kernels of nodes created here may be captured by a
        # later (e.g. second-order) annotated capture.
        merged = {**(_current_region() or {}), **frozen}
        tagged = {**merged, "autograd_phase": "backward"}
        prev = _enter_region(merged)
        torch._C._autograd._push_node_creation_hook(creation_hook)
        # Under CUPTI the nodes this backward creates are attributed as they are created,
        # so the bracket publishes itself instead of walking edges -- and it has to, since
        # what CUPTI would otherwise record is the scope lexically open around the
        # ``backward()`` call, which this outranks. The entry is published with the region
        # installed above, which is what lets current_annotation() drop it if this node
        # raises and its posthook never runs; the truncation below covers the same for any
        # entry left above it.
        published = _annotations_enabled and _annotation_backend == "cupti"
        depth = len(_active_scopes)
        if published:
            _active_scopes.append((tagged, merged))
        scope = (
            _begin_kernel_scope() if _annotations_enabled and not published else None
        )
        state.stack.append((scope, prev, tagged, published, depth))

    def posthook(_grad_inputs: Any, _grad_outputs: Any) -> None:
        if not state.stack:
            return
        entry = state.stack.pop()
        if entry is _SKIPPED:
            return
        scope, prev, tagged, published, depth = entry
        if published:
            del _active_scopes[depth:]
        torch._C._autograd._pop_node_creation_hook()
        _exit_region(prev)
        if scope is None:
            return
        tools_ids = _end_kernel_scope(scope)
        if tools_ids:
            _pending_scopes.append((tagged, tools_ids))

    node.register_prehook(prehook)
    node.register_hook(posthook)


@contextmanager  # type: ignore[arg-type]
def _annotation_region(annotation: dict[str, Any], backward: bool):
    """Publish ``annotation`` as the current region for the body, hooking the autograd
    nodes created in it when ``backward``.

    Backend-independent, and shared by both of ``mark_kernels``' paths. The region is
    entered even when ``backward`` is False, since it must reflect the full dynamic scope
    for nodes hooked by a nested ``backward=True`` scope (or by an executing hooked node)
    to freeze this annotation too.
    """

    def creation_hook(node: Any) -> None:
        _freeze_region_hook(node)

    prev = _enter_region({**(_current_region() or {}), **annotation})
    try:
        if backward:
            with torch.autograd.graph.node_creation_hook(creation_hook):
                yield
        else:
            yield
    finally:
        _exit_region(prev)


@contextmanager  # type: ignore[arg-type]
def mark_kernels(annotation: str | dict[str, Any], *, backward: bool = True):
    r"""mark_kernels(annotation, *, backward=True)

    Context manager that annotates GPU work captured within its scope.

    Must be used inside an active :class:`torch.cuda.graph` capture with
    ``enable_annotations=True``. Every kernel, memcpy, and memset node the
    capture adds within the scope is tagged with :attr:`annotation`. Outside
    a capture, with annotations disabled, or when :func:`is_available` is
    ``False``, the context manager is a no-op.

    When scopes overlap on the same node (e.g. nested scopes), their
    annotation dicts are merged key-by-key with the inner scope winning
    common keys.

    By default backward work is annotated too: autograd nodes created by
    forward operations inside the scope get hooks (via
    :class:`torch.autograd.graph.node_creation_hook`) that bracket their
    backward execution, so when the backward pass is itself captured --
    in the same capture as the forward or in a later one -- its kernels
    are tagged with the same annotation, plus an ``"autograd_phase":
    "backward"`` key marking them as backward work (``"autograd_phase"``
    is therefore reserved: backward annotation overwrites it). When
    backward runs outside a capture the hooks are a no-op. Ownership
    extends to higher-order gradients: nodes created while a hooked node
    executes (``create_graph=True``, checkpoint recomputation) inherit its
    annotations, so a later grad-of-grad capture is attributed too.
    ``AccumulateGrad`` nodes are never annotated: a leaf's node is created
    once and cached, so scope ownership would be an accident of first use.
    Pass ``backward=False`` to annotate only the forward work, e.g. when a
    wrapper implements its own backward attribution. The keyword's
    presence also serves as the feature probe for that native backward
    support: ``"backward" in inspect.signature(mark_kernels).parameters``.

    Implementation: on entry, records the current stream's capture frontier
    and its existing direct dependents; on scope exit, walks only the
    dependent nodes added since entry (falling back to newly created graph
    roots when the scope is the first captured work).

    Args:
        annotation (str or dict): Metadata to attach to each captured node.
            A string ``s`` is recorded as ``{"name": s}``. Dict values must
            be picklable. The key ``"name"`` names the region in trace
            tooling; ``"stream"`` is reserved for stream-lane assignment.
        backward (bool): Whether to also annotate the backward kernels of
            autograd nodes created inside the scope. Default: ``True``.

    .. note::
        The nodes to annotate must be reachable from the capture frontier of
        the stream that is current on scope entry. Work on a different
        already-capturing stream must be synchronized with the current
        stream first.

    .. note::
        Child-graph and conditional nodes have bodies in a separate
        ``cudaGraph_t`` that this walk does not descend into, so their work is
        left unannotated and a warning is issued. Descending is possible
        (``cudaGraphNodeGetParams`` exposes the body graphs), but would not be
        enough on its own: a body's nodes are numbered in that graph's id space
        and are renumbered again when the exec graph inlines them, and nothing
        exposes that renumbering, so :func:`remap_to_exec_graph` could not key
        the annotations to what a profiler reports. For the same reason a scope
        *inside* a conditional body (``torch.cond`` / ``torch.while_loop``)
        records nothing at all.

    .. warning::
        This API is in prototype and may change in future releases.

    Example::

        >>> # xdoctest: +SKIP("requires cuda-bindings and driver >= 13.1")
        >>> g = torch.cuda.CUDAGraph()
        >>> x = torch.randn(8, device="cuda")
        >>> with torch.cuda.graph(g, enable_annotations=True):
        ...     with torch.cuda.graph_annotations.mark_kernels("phase_A"):
        ...         y = x + 1
    """
    if not _annotations_enabled or _is_tools_id_unavailable():
        yield
        return

    if isinstance(annotation, str):
        annotation = {"name": annotation}

    if _annotation_backend == "cupti":
        # Nodes are attributed as CUPTI reports their creation, so the scope only has to
        # publish itself as the ambient annotation -- no frontier snapshot, and no rescan
        # per enclosing scope. Merge into the enclosing scope on the way in (inner wins), so
        # leaving restores the enclosing annotation as it was. Leaving truncates rather than
        # pops one, to also clear any entry a failed backward left above this one.
        enclosing = current_annotation()
        depth = len(_active_scopes)
        _active_scopes.append(
            ({**enclosing, **annotation} if enclosing else annotation, None)
        )
        try:
            with _annotation_region(annotation, backward):
                yield
        finally:
            del _active_scopes[depth:]
        return

    scope = _begin_kernel_scope()
    if scope is None:
        yield
        return

    if (
        _capture_root_graph_id is not None
        and _graph_id(scope.graph) != _capture_root_graph_id
    ):
        # Inside a conditional node's body: torch.cond / torch.while_loop capture into a
        # separate cudaGraph_t. Its node ids are in that graph's id space and are
        # renumbered again in the exec graph, so anything recorded here would be a key
        # that matches nothing in a trace. Record nothing rather than dead keys.
        warnings.warn(
            "mark_kernels: this scope is inside a CUDA graph conditional-node body "
            "(torch.cond / torch.while_loop), which is captured into a separate "
            "cudaGraph_t whose node ids are never remapped to the exec graph; "
            "nothing is annotated for it",
            stacklevel=3,
        )
        yield
        return

    with _annotation_region(annotation, backward):
        yield

    tools_ids = _end_kernel_scope(scope)
    if tools_ids:
        _pending_scopes.append((annotation, tools_ids))


def resolve_pending_annotations() -> None:
    """Resolve pending scope toolsIds into kernel annotations."""
    if not _pending_scopes:
        return

    try:
        # _pending_scopes is in scope-completion order (innermost first) and
        # _merge_annotation is first-wins, so this applies the documented precedence.
        for annotation, tools_ids in _pending_scopes:
            for tools_id in tools_ids:
                _merge_annotation(tools_id, annotation)
    except Exception:
        logger.exception("resolve_pending_annotations failed")
    finally:
        _pending_scopes.clear()


def discard_capture_annotations(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Drop what this capture recorded, for a capture that never reached an exec graph.

    ``resolve_pending_annotations`` runs before ``capture_end`` because the rekey in
    ``instantiate()`` consumes what it writes, so entries land keyed by the capture
    graph's id. If ``capture_end`` then raises, that rekey never happens and the entries
    keep an id no exec graph will ever hold: ``remove_kernel_annotations`` matches exec
    ids, so the graph-destroy path cannot reach them and they last for the life of the
    process. Only called on that error path. Not a public API."""
    _pending_scopes.clear()
    capture_graph_id = torch_cuda_graph._capture_graph_id
    # A remap already happened (keep_graph=False instantiates inside capture_end), so the
    # entries are on a real exec id and are the graph's to purge on destroy, not ours.
    if capture_graph_id is None or torch_cuda_graph._remapped_exec_id is not None:
        return
    for key in [k for k in _kernel_annotations if k >> 32 == capture_graph_id]:
        del _kernel_annotations[key]


def remap_to_exec_graph(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Remap annotation keys from capture graph ID to exec graph ID.

    During capture, toolsId encodes the capture graph's ID in the upper
    32 bits. After instantiation, the profiler uses the exec graph's ID.
    This function rewrites the keys so annotations match the trace.

    The graph's capture id is read from the ``_capture_graph_id`` stamped on it
    by ``maybe_stamp_capture_root`` at capture_begin, so only the annotations
    belonging to this graph are rekeyed. This is order-independent and correct
    when several graphs are captured in sequence: call once per graph. Graphs
    captured with annotations disabled have no capture id and are skipped.

    The exec graph id is only defined once the graph is instantiated. With
    ``keep_graph=True`` instantiation is deferred past the ``torch.cuda.graph()``
    context, so the remap is driven from the graph's ``instantiate()``/
    ``replay()`` instead of context exit. Each ``instantiate()`` (even on an
    unmodified template) produces a fresh exec graph id, so this rekeys from
    wherever the annotations are currently keyed -- the capture id before the
    first remap, the previous exec id after a re-instantiate -- to the current
    exec id. ``_remapped_exec_id`` tracks that current key; when it already
    matches the live exec id (e.g. replay after instantiate) this is a no-op.
    """
    capture_graph_id = torch_cuda_graph._capture_graph_id
    if not _kernel_annotations or capture_graph_id is None:
        return

    exec_graph_id = _check_cuda_bindings(
        _cuda_runtime.cudaGraphExecGetId(  # pyrefly: ignore[missing-attribute]
            torch_cuda_graph.raw_cuda_graph_exec()
        )
    )

    current_key_id = (
        capture_graph_id
        if torch_cuda_graph._remapped_exec_id is None
        else torch_cuda_graph._remapped_exec_id
    )
    if current_key_id == exec_graph_id:
        return

    remapped = _rekey_annotations(_kernel_annotations, current_key_id, exec_graph_id)
    _kernel_annotations.clear()
    _kernel_annotations.update(remapped)
    torch_cuda_graph._remapped_exec_id = exec_graph_id


def _rekey_annotations(
    annotations: dict[int, dict[str, Any]],
    capture_graph_id: int,
    exec_graph_id: int,
) -> dict[int, dict[str, Any]]:
    """Rekey one graph's annotations from its capture id to its exec id.

    A toolsId packs the graph id in the upper 32 bits and the node id in the
    lower 32. Only entries whose upper bits match ``capture_graph_id`` are
    rewritten to ``exec_graph_id``; entries from other graphs (already remapped
    to their own exec ids, or pending their own remap) are kept as-is. The
    rewritten keys cannot collide with anything: they share their upper bits and
    differ in node id, and the driver mints graph and exec ids from one counter
    that it does not reuse, so no other entry is keyed on a freshly minted exec id.
    """
    remapped: dict[int, dict[str, Any]] = {}
    for tools_id, annotation in annotations.items():
        if tools_id >> 32 != capture_graph_id:
            remapped[tools_id] = annotation
            continue
        node_id = tools_id & 0xFFFFFFFF
        remapped[(exec_graph_id << 32) | node_id] = annotation
    return remapped


def resolve_and_remap(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Resolve any pending scopes and remap one graph in a single call.

    Shorthand for ``resolve_pending_annotations()`` followed by
    ``remap_to_exec_graph(graph)``; the pair normally run after a capture.
    """
    resolve_pending_annotations()
    remap_to_exec_graph(torch_cuda_graph)


class _AnnotationsView(Mapping[int, "list[Any]"]):
    """Read-only view of the annotation store, presenting each node's annotation as a
    one-element list.

    The store holds exactly one merged dict per node, but the public mapping has always
    had list values and pickles of it are read back by out-of-tree consumers, so the
    shape is kept. Wrapping on read rather than storing lists is what makes "at most one
    annotation per node" explicit.
    """

    def __getitem__(self, tools_id: int) -> list[Any]:
        return [_kernel_annotations[tools_id]]

    def __iter__(self) -> Iterator[int]:
        return iter(_kernel_annotations)

    def __len__(self) -> int:
        return len(_kernel_annotations)

    def __repr__(self) -> str:
        return repr(dict(self))


_annotations_view = _AnnotationsView()


def get_kernel_annotations() -> Mapping[int, list[Any]]:
    r"""get_kernel_annotations() -> Mapping[int, list]

    Return the live registry of recorded kernel annotations.

    Keys are opaque integers matching the ``graph node id`` field that
    CUPTI-based profilers attach to kernel events; values are one-element
    lists holding the annotation dict recorded for that node -- annotations
    from overlapping scopes are merged into that single dict. The registry
    accumulates across captures and is global to the process.

    The returned mapping is a **live view**: it is updated in place when a
    graph is instantiated (annotation keys are rekeyed to the executable
    graph's ids), so a reference obtained early stays current. Keys are
    valid for joining against a profiler trace once the corresponding
    graphs have been instantiated. The mapping is read-only; snapshot it
    with ``dict(...)`` if isolation is needed.

    .. warning::
        This API is in prototype and may change in future releases.

    Example::

        >>> # xdoctest: +SKIP("requires cuda-bindings and driver >= 13.1")
        >>> annotations = torch.cuda.graph_annotations.get_kernel_annotations()
        >>> with open("annotations.pkl", "wb") as f:
        ...     pickle.dump(dict(annotations), f)
    """
    return _annotations_view


def _reset_kernel_annotations() -> None:
    """Empty the annotation registry.

    Backward brackets already attached to live autograd nodes are NOT revoked -- they
    cannot be detached, and they go on recording into the emptied registry. The
    implementation behind the deprecated :func:`clear_kernel_annotations`, and what tests
    use to isolate themselves without tripping its deprecation warning. Not a public
    API."""
    _kernel_annotations.clear()
    _pending_scopes.clear()


def save_kernel_annotations(path: str | Path) -> None:
    """Save the current kernel annotations to a pickle file.

    The file can be passed directly to the Chrome trace annotator
    (``torch.cuda._annotate_cuda_graph_trace``).
    """
    with open(path, "wb") as f:
        pickle.dump(dict(_kernel_annotations), f)


@deprecated(
    "`torch.cuda.graph_annotations.clear_kernel_annotations` is deprecated. The registry "
    "bounds itself: annotations are rekeyed to the exec graph on instantiate and dropped "
    "when that graph is destroyed, so a global wipe is not needed and discards annotations "
    "for graphs that are still live.",
    category=FutureWarning,
)
def clear_kernel_annotations() -> None:
    r"""clear_kernel_annotations() -> None

    Clear all recorded kernel annotations.

    .. deprecated:: 2.15
        The registry is self-bounding, so nothing needs to call this. Annotations are
        rekeyed to the exec graph id on instantiation and dropped when that graph is
        destroyed; in a long-running workload -- where graphs are captured once and
        replayed for the whole run -- a global wipe instead discards annotations for
        graphs that are still live and still being joined against.

    Forgets everything recorded so far. It does not stop anything from recording:
    the backward hooks :func:`mark_kernels` attached to live autograd nodes cannot be
    detached and go on writing into the emptied registry, and a forward scope open
    across the clear registers on exit as usual. In particular this breaks backward
    projection across a forward/backward capture pair -- the forward graph's entries
    are gone and its scope no longer names the backward graph's kernels.

    .. warning::
        This API is in prototype and may change in future releases.
    """
    _reset_kernel_annotations()


def remove_kernel_annotations(exec_graph_ids: Iterable[int]) -> None:
    """Drop kernel-annotation entries whose exec graph id (tools_id >> 32) is in
    exec_graph_ids, so the map does not grow across the run. Run by the annotation
    resolver's graph-destroy handler."""
    ids = set(exec_graph_ids)
    if not ids:
        return
    for key in [k for k in _kernel_annotations if k >> 32 in ids]:
        del _kernel_annotations[key]


def register_fqn_annotation_hooks(
    model: "torch.nn.Module",
) -> list[Any]:
    """Register forward hooks that annotate CUDA graph kernels with module FQNs.

    For use with standalone CUDA graphs (without Inductor).  Each module's
    forward pass is wrapped with ``mark_kernels(fqn)`` during graph capture so
    that kernel nodes are annotated with their layer name.  Nested modules
    produce overlapping scopes; ``resolve_pending_annotations`` picks the
    innermost annotation for each kernel node.

    The FQN format matches the Inductor convention: ``L.<module_path>`` where
    the root module is ``L`` and submodules use dotted paths, e.g.
    ``L.networks.0.conv``.

    Must be called before ``torch.cuda.graph()`` capture.  Remove the returned
    handles after capture to avoid overhead during replay.

    Args:
        model: The ``nn.Module`` to annotate.

    Returns:
        List of ``RemovableHook`` handles.  Call ``h.remove()`` on each after
        capture is complete.

    Example::

        from torch.cuda._graph_annotations import (
            enable_annotations,
            register_fqn_annotation_hooks,
            remap_to_exec_graph,
            clear_kernel_annotations,
        )

        clear_kernel_annotations()
        enable_annotations()
        handles = register_fqn_annotation_hooks(model)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = model(x)
            resolve_pending_annotations()

        for h in handles:
            h.remove()

        remap_to_exec_graph(g)
    """
    handles: list[Any] = []
    # Stack per module to handle re-entrant calls (e.g. same module used twice).
    active_cms: dict[int, list[Any]] = defaultdict(list)

    for name, module in model.named_modules():
        fqn = f"L.{name}" if name else "L"

        def pre_hook(mod: Any, _input: Any, fqn: str = fqn) -> None:
            cm = mark_kernels({"module_name": fqn})
            active_cms[id(mod)].append(cm)
            cm.__enter__()

        def post_hook(mod: Any, _input: Any, _output: Any) -> None:
            stack = active_cms.get(id(mod))
            if stack:
                cm = stack.pop()
                cm.__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    return handles


# Counter-based stream ID registry. IDs start at 60 (above the highest
# observed non-graphed CUDA stream ID) so every assigned lane is visually
# distinct in Perfetto and doesn't collide with real streams.
_stream_id_counter: int = 60
_stream_id_map: dict[int, int] = {}


def _get_stream_id(stream: torch.cuda.Stream) -> int:
    """Return a small, stable stream ID for the given CUDA stream."""
    global _stream_id_counter
    key = stream.cuda_stream
    if key not in _stream_id_map:
        _stream_id_map[key] = _stream_id_counter
        _stream_id_counter += 1
    return _stream_id_map[key]


def get_stream_for_pg(pg_key: str) -> int:
    """Return a unique stream ID for the given process group key."""
    global _stream_id_counter
    if pg_key not in _stream_id_map:
        _stream_id_map[pg_key] = _stream_id_counter  # type: ignore[assignment]
        _stream_id_counter += 1
    return _stream_id_map[pg_key]  # type: ignore[return-value]


@contextmanager  # type: ignore[arg-type]
def mark_stream(stream: torch.cuda.Stream, annotation: str | dict[str, Any]):
    """Switch to stream, inject its ID into annotation, and mark kernels.

    If *stream* is already the current stream, no stream switch or stream ID
    injection happens — the kernels stay on whatever stream is active (which
    keeps the trace faithful when e.g. FSDP uses the current stream for
    copy-in instead of a separate one). When switching to a different stream,
    this snapshots the current capturing branch before the target stream
    runs marked work. If the target stream is already capturing, the marked
    work must still be synchronized with the current stream so it is
    reachable from that snapped frontier.
    """
    if not _annotations_enabled:
        with torch.cuda.stream(stream):
            yield
        return
    if stream.cuda_stream == torch.cuda.current_stream().cuda_stream:
        with mark_kernels(annotation):
            yield
    else:
        if isinstance(annotation, str):
            annotation = {"name": annotation}
        if isinstance(annotation, dict):
            # Copy rather than write through: the annotation is stored by reference, so an
            # in-place "stream" would leak back to the caller and, when one dict is reused
            # across mark_stream calls, retag every region already recorded with it to the
            # last lane written.
            annotation = {**annotation, "stream": _get_stream_id(stream)}
        with mark_kernels(annotation):
            with torch.cuda.stream(stream):
                yield
