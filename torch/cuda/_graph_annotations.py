"""Annotate CUDA graph kernel nodes during capture.

During CUDA graph capture, ``mark_kernels`` records the current capture
frontier and the direct dependents already attached to that frontier.
On scope exit it walks only the newly added dependent edges to find the
nodes created within the scope. Each kernel or memcpy node found is
annotated by its ``toolsId`` so it can later be matched to profiler
trace events.

``mark_kernels`` now snapshots capture state from whatever stream is
current on scope entry, so that stream must already be participating in
the capture. ``mark_stream`` handles this by starting ``mark_kernels``
before switching to the target stream.

The annotations can be pickled and later merged into a Chrome profiler
trace using ``torch.cuda._annotate_cuda_graph_trace``.

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
"""

import importlib.metadata
from collections import defaultdict
from contextlib import contextmanager
from logging import getLogger
from typing import Any, TypeAlias

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


def _set_annotations_enabled(enabled: bool) -> None:
    """Set whether annotation recording is active. Used by ``torch.cuda.graph``
    to scope annotations to a capture; not a public API."""
    global _annotations_enabled
    _annotations_enabled = enabled


def _probe_tools_id() -> bool:
    """Probe whether cudaGraphNodeGetToolsId is supported by the driver.

    Calls with a null node: cudaErrorInvalidValue means the API exists
    in the driver (good), cudaErrorCallRequiresNewerDriver means it
    does not (bad).
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
        == _cuda_runtime.cudaError_t.cudaErrorCallRequiresNewerDriver  # pyrefly: ignore[missing-attribute]
    ):
        logger.info(
            "cudaGraphNodeGetToolsId requires a newer driver "
            "(missing cuda-compat?); "
            "CUDA graph kernel annotations will be disabled"
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


# toolsId -> list of annotation objects.
_kernel_annotations: defaultdict[int, list[Any]] = defaultdict(list)

# Node types we annotate (kernels, memcpys, and batch mem ops), as driver
# node-type enums. Initialized lazily to avoid touching cuda.bindings at import
# time.
_ANNOTATABLE_TYPES: set[Any] | None = None


def _get_annotatable_types() -> set[Any]:
    global _ANNOTATABLE_TYPES
    if _ANNOTATABLE_TYPES is None:
        node_types = _cuda_driver.CUgraphNodeType  # pyrefly: ignore[missing-attribute]
        _ANNOTATABLE_TYPES = {
            node_types.CU_GRAPH_NODE_TYPE_KERNEL,
            node_types.CU_GRAPH_NODE_TYPE_MEMCPY,
            node_types.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
        }
    return _ANNOTATABLE_TYPES


# Pending scopes: (annotation, toolsIds discovered for the scope).
_pending_scopes: list[tuple[Any, list[int]]] = []


@contextmanager  # type: ignore[arg-type]
def mark_kernels(annotation: str | dict[str, Any]):
    """Context manager that records new scope nodes for later annotation.

    During capture, records the current stream frontier and its existing
    direct dependents on entry. On scope exit, traces only the dependent
    nodes added since entry. After capture, ``resolve_pending_annotations``
    merges overlapping scopes and stores the final toolsId annotations.
    If the scope is the first captured work, the entry frontier is empty,
    so ``mark_kernels`` falls back to the newly created graph roots.

    Must be called inside an active ``torch.cuda.graph()`` capture. The
    nodes you expect to annotate must be reachable from the stream frontier
    that is current on entry. If work runs on a different already-capturing
    branch, it must first be synchronized with the current stream so that
    branch becomes reachable from the entry frontier. If the current stream
    is not capturing, or if ``cudaGraphNodeGetToolsId`` is not available,
    the context manager is a no-op.

    Args:
        annotation: Arbitrary object appended to the annotation list for
            every kernel/memcpy node captured within this scope.
    """
    if not _annotations_enabled or _is_tools_id_unavailable():
        yield
        return

    if isinstance(annotation, str):
        annotation = {"str": annotation}

    stream = _cuda_runtime.cudaStream_t(  # pyrefly: ignore[missing-attribute]
        init_value=torch.cuda.current_stream().cuda_stream
    )
    capture_state = _get_capture_state(stream)
    if capture_state is None:
        yield
        return
    graph, frontier = capture_state

    entry_root_keys: set[int] | None = None
    entry_direct_dependents = {
        int(node): {int(dep) for dep in _get_dependent_nodes(node)} for node in frontier
    }
    if not frontier:
        entry_root_keys = {int(node) for node in _get_root_nodes(graph)}

    yield

    if frontier:
        scope_nodes = _collect_descendants(
            frontier,
            existing_direct_dependents=entry_direct_dependents,
        )
    else:
        new_roots = [
            node
            for node in _get_root_nodes(graph)
            if int(node) not in (entry_root_keys or set())
        ]
        scope_nodes = _collect_descendants(new_roots, include_start_nodes=True)

    if not scope_nodes:
        return

    annotatable = _get_annotatable_types()
    tools_ids: list[int] = []
    for node in scope_nodes.values():
        node_type = _get_node_type(node)
        if node_type not in annotatable:
            continue
        tools_ids.append(
            _check_cuda_bindings(
                _cuda_runtime.cudaGraphNodeGetToolsId(  # pyrefly: ignore[missing-attribute]
                    node
                )
            )
        )

    if tools_ids:
        _pending_scopes.append((annotation, tools_ids))


def maybe_stamp_capture_graph_id(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Record the captured graph's id on the graph object for later remap.

    Called from ``capture_end`` in the window after capture ends and before the
    graph is finalized, where the template ``cudaGraph_t`` is live for both
    ``keep_graph`` modes (and reachable via ``raw_cuda_graph``). ``remap_to_exec_graph``
    later matches this graph's annotations to its exec id via this stamp, without
    relying on call ordering. No-op if annotations are disabled or
    ``cudaGraphNodeGetToolsId`` is unavailable. Harmless when no ``mark_kernels``
    regions run: the annotation map stays empty so resolve and remap are no-ops.
    """
    if not _annotations_enabled or _is_tools_id_unavailable():
        return
    # Past the _is_tools_id_unavailable() guard cuda-bindings is present and the
    # driver supports the toolsId API (same version gate as cudaGraphGetId), so
    # any error here is unexpected: error-check and let it raise. cudaGraphGetId
    # accepts the raw cudaGraph_t handle (int) directly.
    torch_cuda_graph._capture_graph_id = _check_cuda_bindings(
        _cuda_runtime.cudaGraphGetId(  # pyrefly: ignore[missing-attribute]
            torch_cuda_graph.raw_cuda_graph()
        )
    )
    # Fresh capture: annotations are keyed by this capture id until remapped.
    torch_cuda_graph._remapped_exec_id = None


def resolve_pending_annotations() -> None:
    """Resolve pending scope toolsIds into kernel annotations."""
    if not _pending_scopes:
        return

    try:
        per_tools_id: defaultdict[int, list[Any]] = defaultdict(list)
        for annotation, tools_ids in _pending_scopes:
            for tools_id in tools_ids:
                per_tools_id[tools_id].append(annotation)

        for tools_id, annotations in per_tools_id.items():
            if len(annotations) == 1:
                _kernel_annotations[tools_id].append(annotations[0])
                continue

            merged: dict[str, Any] = {}
            for annotation in annotations:
                if isinstance(annotation, dict):
                    for key, value in annotation.items():
                        merged.setdefault(key, value)
                else:
                    merged.setdefault("name", annotation)
            _kernel_annotations[tools_id].append(merged)
    except Exception:
        logger.exception("resolve_pending_annotations failed")
    finally:
        _pending_scopes.clear()


def remap_to_exec_graph(torch_cuda_graph: torch.cuda.CUDAGraph) -> None:
    """Remap annotation keys from capture graph ID to exec graph ID.

    During capture, toolsId encodes the capture graph's ID in the upper
    32 bits. After instantiation, the profiler uses the exec graph's ID.
    This function rewrites the keys so annotations match the trace.

    The graph's capture id is read from the ``_capture_graph_id`` stamped on it
    by ``maybe_stamp_capture_graph_id`` in the capture_end window, so only the annotations
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
    annotations: dict[int, list[Any]],
    capture_graph_id: int,
    exec_graph_id: int,
) -> dict[int, list[Any]]:
    """Rekey one graph's annotations from its capture id to its exec id.

    A toolsId packs the graph id in the upper 32 bits and the node id in the
    lower 32. Only entries whose upper bits match ``capture_graph_id`` are
    rewritten to ``exec_graph_id``; entries from other graphs (already remapped
    to their own exec ids, or pending their own remap) are kept as-is. When two
    capture-side ids collide on the same rekeyed id their lists are merged.
    """
    remapped: dict[int, list[Any]] = {}
    for tools_id, ann_list in annotations.items():
        if tools_id >> 32 != capture_graph_id:
            remapped[tools_id] = ann_list
            continue
        node_id = tools_id & 0xFFFFFFFF
        new_tools_id = (exec_graph_id << 32) | node_id
        if new_tools_id in remapped:
            remapped[new_tools_id].extend(ann_list)
        else:
            remapped[new_tools_id] = list(ann_list)
    return remapped


def get_kernel_annotations() -> dict[int, list[Any]]:
    """Return the current kernel annotation map (toolsId -> annotations)."""
    return _kernel_annotations


def clear_kernel_annotations() -> None:
    """Clear all recorded kernel annotations and pending scopes."""
    _kernel_annotations.clear()
    _pending_scopes.clear()


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
            annotation = {"str": annotation}
        if isinstance(annotation, dict):
            annotation["stream"] = _get_stream_id(stream)
        with mark_kernels(annotation):
            with torch.cuda.stream(stream):
                yield
