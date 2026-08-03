"""Attribute CUDA-graph nodes as CUPTI reports their creation.

The CUPTI-backed discovery half of :mod:`torch.cuda._graph_annotations`. Where the
edge-walk backend infers a ``mark_kernels`` scope's membership from graph topology after
the fact, this registers a ``RESOURCE`` / ``GRAPHNODE_CREATED`` handler on the CUPTI
monitor's shared subscriber and records each node against whatever scope is open when
CUPTI announces it. Two measured consequences:

* A scope entered while the *current* stream is not yet capturing records nothing at all
  under the walk -- it snapshots that stream's capture state on entry, finds none, and
  no-ops -- even though the work inside it is captured. CUPTI reports each node as it is
  created, so the scope is attributed normally.
* The walk rescans a nested scope's nodes once per enclosing scope; CUPTI visits each node
  once. Measured at depth 16 over ~2550 nodes, annotation overhead is 73.6ms for the walk
  against 33.5ms here, and the gap grows with depth.

Note the walk does *not* miss work merely because another stream issued it: it follows
dependency edges across streams fine, so long as the current stream was capturing when the
scope opened.

Both backends key annotations by the node's capture-side ``toolsId`` and are remapped to
exec-graph ids by ``remap_to_exec_graph`` identically, so downstream consumers cannot tell
them apart.

Requires the CUPTI monitor to already hold a subscription -- see
``torch.profiler._cupti.monitor.has_live_subscription``. The handler runs synchronously on
the capturing thread inside the CUDA call, so it stays as short as it can and never raises
(the monitor's switchboard swallows exceptions, but a raise would still cost a traceback
per node).
"""

from __future__ import annotations

from logging import getLogger
from typing import Any


logger = getLogger(__name__)


# CUpti_CallbackDomain.CUPTI_CB_DOMAIN_RESOURCE and
# CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED. Resolved from cupti-python on first arm rather
# than hardcoded; these fallbacks are only used if the enums are unavailable.
_RESOURCE_DOMAIN = 3
_CBID_GRAPHNODE_CREATED = 13

# Set while a capture is armed: the top-level capture graph's id, snapshotted at arm time
# (right after capture_begin, while the capture stream is still capturing into the
# top-level graph). Nodes reported for any other graph belong to a child-graph or
# conditional body, whose ids live in that body graph's space and are renumbered again in
# the exec graph -- so recording them would produce keys matching nothing. They are
# dropped until the clone-based remap can express them.
_capture_root_graph_id: int | None = None

# The handler token from the monitor, kept so disarm can unregister it.
_handler: Any = None

# Driver node-type values we attribute, resolved once (see _annotatable_node_types).
_annotatable: frozenset[int] | None = None


def _annotatable_node_types() -> frozenset[int]:
    """Kernel / memcpy / memset / batch-mem-op node types, as raw driver enum values.

    Mirrors ``_graph_annotations._get_annotatable_types`` but as ints, since CUPTI reports
    ``GraphData.node_type`` as a plain value.
    """
    global _annotatable
    if _annotatable is None:
        from cuda.bindings import driver  # pyrefly: ignore[missing-import]

        node_types = driver.CUgraphNodeType
        _annotatable = frozenset(
            int(t)
            for t in (
                node_types.CU_GRAPH_NODE_TYPE_KERNEL,
                node_types.CU_GRAPH_NODE_TYPE_MEMCPY,
                node_types.CU_GRAPH_NODE_TYPE_MEMSET,
                node_types.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            )
        )
    return _annotatable


def _on_graph_node_created(_domain: int, _cbid: int, cbdata: int) -> None:
    """Record the ambient ``mark_kernels`` annotation for a freshly created graph node.

    Runs on the capturing thread inside the CUDA call. ``cbdata`` is a raw
    ``CUpti_ResourceData*``; its ``resource_descriptor`` is the ``CUpti_GraphData`` carrying
    the node handle and type.
    """
    from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

    from torch.cuda._graph_annotations import current_annotation, record_node_annotation

    annotation = current_annotation()
    if annotation is None:
        return
    graph_data = _cupti.GraphData.from_ptr(
        _cupti.ResourceData.from_ptr(cbdata).resource_descriptor
    )
    if int(graph_data.node_type) not in _annotatable_node_types():
        return
    tools_id = _tools_id(graph_data.node)
    if tools_id is None or tools_id >> 32 != _capture_root_graph_id:
        return
    record_node_annotation(tools_id, annotation)


def _tools_id(node: Any) -> int | None:
    """The node's ``toolsId`` -- the value CUPTI later reports as ``graph node id`` -- or
    ``None`` on any driver error. Must not raise into the callback."""
    from cuda.bindings import runtime  # pyrefly: ignore[missing-import]

    err, tools_id = runtime.cudaGraphNodeGetToolsId(int(node))
    if int(err) != 0:
        return None
    return int(tools_id)


def is_available() -> bool:
    """True when this backend can be used right now: the CUPTI monitor holds a
    subscription, and cupti-python is importable.

    Does not create the monitor. A capture asking for ``annotation_backend="auto"`` falls
    back to the edge walk when this is ``False``.
    """
    try:
        # Importing the monitor already requires cupti-python (it raises
        # ModuleNotFoundError without it), so this covers both conditions.
        from torch.profiler._cupti.monitor import has_live_subscription
    except ImportError:
        return False
    return has_live_subscription()


def register(*, force: bool = False) -> bool:
    """Register the node-creation handler, bringing the CUPTI subscription up.

    Separate from :func:`arm` -- and called *before* ``capture_begin`` -- so that failing to
    obtain CUPTI cannot leave a capture half-started. Returns ``False`` when the backend is
    unavailable, so the caller can fall back to the edge walk.

    ``force`` brings the CUPTI monitor up instead of requiring a live subscription. That is a
    deliberate, opt-in cost: once we hold a CUPTI subscription, kineto's one-shot init fails
    permanently, so a later ``torch.profiler`` run records no GPU activity. Only
    ``annotation_backend="cupti"`` asks for it.
    """
    global _handler, _RESOURCE_DOMAIN, _CBID_GRAPHNODE_CREATED
    if _handler is not None:
        raise RuntimeError("graph-node callbacks are already registered")
    if not force and not is_available():
        return False
    try:
        from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.monitor import CuptiMonitor
    except ImportError:
        return False

    _RESOURCE_DOMAIN = int(_cupti.CallbackDomain.RESOURCE)
    _CBID_GRAPHNODE_CREATED = int(_cupti.CallbackIdResource.GRAPHNODE_CREATED)
    try:
        _handler = CuptiMonitor().register_callback_handler(
            _RESOURCE_DOMAIN, _CBID_GRAPHNODE_CREATED, _on_graph_node_created
        )
    except Exception:
        # Subscribing can fail outright -- e.g. another CUPTI consumer already holds a
        # subscription it did not offer to share. Fall back rather than fail the capture.
        logger.debug("graph-node callback registration failed", exc_info=True)
        return False
    return True


def arm(capture_stream: Any) -> bool:
    """Enable the callback for the capture just begun on ``capture_stream``.

    Snapshots the top-level capture graph so child-graph and conditional-body nodes can be
    filtered out. Returns ``False`` when nothing is registered or the stream is not
    capturing.
    """
    global _capture_root_graph_id
    if _handler is None:
        return False
    from torch.profiler._cupti.monitor import CuptiMonitor

    root = _capture_graph_id(capture_stream)
    if root is None:
        return False
    _capture_root_graph_id = root
    CuptiMonitor().arm_callback(_RESOURCE_DOMAIN, _CBID_GRAPHNODE_CREATED)
    return True


def disarm() -> None:
    """Disable and unregister the node-creation callback. Idempotent, so it is safe in a
    ``finally`` for a capture that raised."""
    global _capture_root_graph_id, _handler
    if _handler is None:
        return
    from torch.profiler._cupti.monitor import CuptiMonitor

    monitor = CuptiMonitor()
    try:
        monitor.disarm_callback(_RESOURCE_DOMAIN, _CBID_GRAPHNODE_CREATED)
        monitor.unregister_callback_handler(_handler)
    finally:
        _handler = None
        _capture_root_graph_id = None


def _capture_graph_id(capture_stream: Any) -> int | None:
    """The id of the graph ``capture_stream`` is capturing into, or ``None`` if it is not
    capturing."""
    from cuda.bindings import runtime  # pyrefly: ignore[missing-import]

    from torch.cuda._utils import _check_cuda_bindings

    stream = runtime.cudaStream_t(init_value=capture_stream.cuda_stream)
    status, _id, graph, _deps, _edge_data, _num_deps = _check_cuda_bindings(
        runtime.cudaStreamGetCaptureInfo(stream)
    )
    if status != runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return None
    return _check_cuda_bindings(runtime.cudaGraphGetId(graph))
