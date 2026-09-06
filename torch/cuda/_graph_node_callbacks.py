"""Attribute CUDA-graph nodes as CUPTI reports their creation.

The CUPTI-backed discovery half of :mod:`torch.cuda._graph_annotations`. Where the
edge-walk backend infers a ``mark_kernels`` scope's membership from graph topology after
the fact, this registers a ``RESOURCE`` / ``GRAPHNODE_CREATED`` handler on Cuspy's
shared subscriber and records each node against whatever scope is open when
CUPTI announces it. A scope entered while the *current* stream is not yet capturing records
nothing at all under the walk; CUPTI correctly attributes kernels. The walk rescans a nested
scope's nodes once per enclosing scope; CUPTI visits each node once.

Both backends key annotations by the node's capture-side ``toolsId`` and are remapped to
exec-graph ids by ``remap_to_exec_graph`` identically, so downstream consumers cannot tell
them apart.

Needs a CUPTI subscription, but does not require one to already exist:
``annotation_backend="auto"`` picks this backend only when Cuspy is already holding
one (see ``torch.profiler._cuspy.core.has_live_subscription``), while
``annotation_backend="cupti"`` brings Cuspy up to take one. The handler runs
synchronously on the capturing thread inside the CUDA call, so it stays as short as it can
and never raises (Cuspy's switchboard swallows exceptions, but a raise would still
cost a traceback per node).
"""

from __future__ import annotations

import warnings
from logging import getLogger
from typing import Any


logger = getLogger(__name__)


# The handler token from Cuspy. Carries the (domain, cbid) it was registered for, so
# it is also what arm/disarm address the callback by; kept so disarm can unregister it.
_handler: Any = None

# Nodes dropped during the armed capture for belonging to a child-graph or conditional
# body. Counted rather than warned about on the spot: warnings.warn can be configured to
# raise, and that must not happen inside CUPTI's C dispatch. disarm() reports the total.
_dropped_body_nodes: int = 0


def _on_graph_node_created(_domain: int, _cbid: int, cbdata: int) -> None:
    """Record the ambient ``mark_kernels`` annotation for a freshly created graph node.

    Runs on the capturing thread inside the CUDA call. ``cbdata`` is a raw
    ``CUpti_ResourceData*``; its ``resource_descriptor`` is the ``CUpti_GraphData`` carrying
    the node handle and type.
    """
    from cuda.bindings import runtime  # pyrefly: ignore[missing-import]
    from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

    from torch.cuda._graph_annotations import (
        _get_annotatable_type_values,
        capture_root_graph_id,
        current_annotation,
        record_node_annotation,
    )
    from torch.cuda._utils import _check_cuda_bindings

    # Cheapest rejection first: the node type is already in hand, the annotation costs a
    # merge when scopes are nested, and the toolsId lookup is a driver call.
    graph_data = _cupti.GraphData.from_ptr(
        _cupti.ResourceData.from_ptr(cbdata).resource_descriptor
    )
    if int(graph_data.node_type) not in _get_annotatable_type_values():
        return
    annotation = current_annotation()
    if annotation is None:
        return
    # toolsId is the value CUPTI later reports as "graph node id". mark_kernels gates on
    # _is_tools_id_unavailable, so a driver without this API leaves no scope open and we
    # returned above -- an error here is genuinely unexpected, and Cuspy's switchboard
    # logs it rather than letting it reach CUPTI's C dispatch.
    tools_id = _check_cuda_bindings(runtime.cudaGraphNodeGetToolsId(graph_data.node))
    # Nodes reported for any other graph belong to a child-graph or conditional body, whose
    # ids live in that body graph's space and are renumbered again in the exec graph -- so
    # recording them would produce keys matching nothing. disarm() warns about the total.
    if tools_id >> 32 != capture_root_graph_id():
        global _dropped_body_nodes
        _dropped_body_nodes += 1
        return
    record_node_annotation(tools_id, annotation)


def is_available() -> bool:
    """True when this backend can be used right now: Cuspy holds a
    subscription, and cupti-python is importable.

    Does not create Cuspy. A capture asking for ``annotation_backend="auto"`` falls
    back to the edge walk when this is ``False``.
    """
    try:
        # Importing Cuspy already requires cupti-python (it raises
        # ModuleNotFoundError without it), so this covers both conditions.
        from torch.profiler._cuspy.core import has_live_subscription
    except ImportError:
        return False
    return has_live_subscription()


def register(*, force: bool = False) -> bool:
    """Register the node-creation handler, bringing the CUPTI subscription up.

    Separate from :func:`arm` -- and called *before* ``capture_begin`` -- so that failing to
    obtain CUPTI cannot leave a capture half-started. Returns ``False`` when the backend is
    unavailable, so the caller can fall back to the edge walk.

    ``force`` brings Cuspy up instead of requiring a live subscription. That is a
    deliberate, opt-in cost: once we hold a CUPTI subscription, kineto's one-shot init fails
    permanently, so a later ``torch.profiler`` run records no GPU activity. Only
    ``annotation_backend="cupti"`` asks for it.
    """
    global _handler
    if _handler is not None:
        raise RuntimeError("graph-node callbacks are already registered")
    if not force and not is_available():
        return False
    # force=True skips the is_available() check above, so cupti-python may still be missing
    # here; report that as "unavailable" and let the caller raise something actionable.
    try:
        from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

        from torch.profiler._cuspy.core import Cuspy
    except ImportError:
        return False

    # Importing Cuspy requires cupti-python, so its enums are available too -- there is
    # no case where a hardcoded (domain, cbid) fallback would be reachable.
    try:
        _handler = Cuspy().register_callback_handler(
            int(_cupti.CallbackDomain.RESOURCE),
            int(_cupti.CallbackIdResource.GRAPHNODE_CREATED),
            _on_graph_node_created,
        )
    except Exception:
        # Subscribing can fail outright -- e.g. another CUPTI consumer already holds a
        # subscription it did not offer to share. Fall back rather than fail the capture.
        logger.debug("graph-node callback registration failed", exc_info=True)
        return False
    return True


def arm() -> bool:
    """Enable the callback for the capture just begun on the current stream.

    Returns ``False`` when nothing is registered, or when the capture did not record a
    top-level graph id -- the handler filters body nodes against that id, so without it
    every node would be dropped and the caller should fall back to the edge walk.
    """
    global _dropped_body_nodes
    if _handler is None:
        return False
    from torch.cuda._graph_annotations import capture_root_graph_id
    from torch.profiler._cuspy.core import Cuspy

    if capture_root_graph_id() is None:
        return False
    _dropped_body_nodes = 0
    Cuspy().arm_callback(_handler.domain, _handler.cbid)
    return True


def disarm() -> None:
    """Disable and unregister the node-creation callback, and report any work that went
    unannotated. Idempotent, so it is safe in a ``finally`` for a capture that raised."""
    global _handler
    if _handler is None:
        return
    from torch.profiler._cuspy.core import Cuspy

    cuspy = Cuspy()
    try:
        cuspy.disarm_callback(_handler.domain, _handler.cbid)
        cuspy.unregister_callback_handler(_handler)
    finally:
        _handler = None
    # Warn here rather than from the handler: this runs on the normal path, where a
    # warnings filter promoting warnings to errors is harmless. The edge walk reports the
    # same situation at scope entry; reporting it on the drop instead covers both a scope
    # inside a body and a scope containing one, and says how much was actually lost.
    if _dropped_body_nodes:
        warnings.warn(
            f"mark_kernels: {_dropped_body_nodes} node(s) created inside a CUDA graph "
            "child-graph or conditional-node body (torch.cond / torch.while_loop) were "
            "not annotated -- such a body is captured into a separate cudaGraph_t whose "
            "node ids are never remapped to the exec graph, so an annotation there would "
            "match nothing in a profiler trace",
            stacklevel=2,
        )
