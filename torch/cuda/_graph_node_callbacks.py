r"""Observe CUDA-graph nodes as CUPTI reports their creation.

The CUPTI-backed discovery half of :mod:`torch.cuda._graph_annotations`. Where the
edge-walk backend infers a ``mark_kernels`` scope's membership from graph topology after
the fact, this registers a ``RESOURCE`` / ``GRAPHNODE_CREATED`` handler on the CUPTI
monitor's shared subscriber and records each node against whatever scope is open when
CUPTI announces it. A scope entered while the *current* stream is not yet capturing records
nothing at all under the walk; CUPTI correctly attributes kernels. The walk rescans a nested
scope's nodes once per enclosing scope; CUPTI visits each node once.

The same handler also records Python launch stacks when requested, so both consumers share
the node-type filter, child/body-node rejection, and capture-side ``toolsId`` lookup.

Needs a CUPTI subscription, but does not require one to already exist:
``annotation_backend="auto"`` picks this backend only when the monitor is already holding
one (see ``torch.profiler._cupti.monitor.has_live_subscription``), while
``annotation_backend="cupti"`` brings the monitor up to take one. The handler runs
synchronously on the capturing thread inside the CUDA call, so it stays as short as it can.
The monitor's switchboard logs and suppresses exceptions before they reach CUPTI's C
dispatch.
"""

from __future__ import annotations

import warnings
from logging import getLogger
from threading import Lock
from typing import Any


logger = getLogger(__name__)


# The handler token from the monitor. Carries the (domain, cbid) it was registered for, so
# it is also what arm/disarm address the callback by; kept so disarm can unregister it.
_handler: Any = None
_handler_lock = Lock()

# Nodes dropped during the armed capture for belonging to a child-graph or conditional
# body. Counted rather than warned about on the spot: warnings.warn can be configured to
# raise, and that must not happen inside CUPTI's C dispatch. disarm() reports the total.
_dropped_body_nodes: int = 0
_dropped_body_nodes_lock = Lock()


def _on_graph_node_created(domain: int, cbid: int, cbdata: int) -> None:
    """Record metadata for a freshly created graph node.

    Runs on the capturing thread inside the CUDA call. ``cbdata`` is a raw
    ``CUpti_ResourceData*``; its ``resource_descriptor`` is the ``CUpti_GraphData`` carrying
    the node handle and type.
    """
    handler = _handler
    if handler is None or domain != handler.domain or cbid != handler.cbid:
        return

    from cuda.bindings import runtime  # pyrefly: ignore[missing-import]
    from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

    from torch.cuda import _graph_py_stacks
    from torch.cuda._graph_annotations import (
        _get_annotatable_type_values,
        capture_root_graph_id,
        current_annotation,
        record_node_annotation,
    )
    from torch.cuda._utils import _check_cuda_bindings

    annotation = current_annotation()
    capture_stack = _graph_py_stacks._is_capturing()
    if annotation is None and not capture_stack:
        return

    graph_data = _cupti.GraphData.from_ptr(
        _cupti.ResourceData.from_ptr(cbdata).resource_descriptor
    )
    if int(graph_data.node_type) not in _get_annotatable_type_values():
        return
    # toolsId is the value CUPTI later reports as "graph node id". The graph context probes
    # this API before arming the callback, so an error here is genuinely unexpected and the
    # monitor's switchboard logs it rather than letting it reach CUPTI's C dispatch.
    tools_id = _check_cuda_bindings(runtime.cudaGraphNodeGetToolsId(graph_data.node))
    # Nodes reported for any other graph belong to a child-graph or conditional body, whose
    # ids live in that body graph's space and are renumbered again in the exec graph -- so
    # recording them would produce keys matching nothing. disarm() warns about the total.
    if tools_id >> 32 != capture_root_graph_id():
        global _dropped_body_nodes
        with _dropped_body_nodes_lock:
            _dropped_body_nodes += 1
        return
    if annotation is not None:
        record_node_annotation(tools_id, annotation)
    if capture_stack:
        _graph_py_stacks._record(tools_id)


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
    permanently, so a later ``torch.profiler`` run records no GPU activity. The CUPTI
    annotation backend and Python stack capture ask for it.
    """
    global _handler
    with _handler_lock:
        if _handler is not None:
            raise RuntimeError("graph-node callbacks are already registered")
        if not force and not is_available():
            return False
        # force=True skips the is_available() check above, so cupti-python may still be
        # missing here; report that as "unavailable" and let the caller raise something
        # actionable.
        try:
            from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

            from torch.profiler._cupti.monitor import CuptiMonitor
        except ImportError:
            return False

        # Importing the monitor requires cupti-python, so its enums are available too --
        # there is no case where a hardcoded (domain, cbid) fallback would be reachable.
        try:
            _handler = CuptiMonitor().register_callback_handler(
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
    with _handler_lock:
        if _handler is None:
            return False
        from torch.cuda._graph_annotations import capture_root_graph_id
        from torch.profiler._cupti.monitor import CuptiMonitor

        if capture_root_graph_id() is None:
            return False
        with _dropped_body_nodes_lock:
            _dropped_body_nodes = 0
        try:
            CuptiMonitor().arm_callback(_handler.domain, _handler.cbid)
        except Exception:
            logger.debug("graph-node callback arm failed", exc_info=True)
            return False
        return True


def disarm() -> None:
    r"""Disable and unregister the node-creation callback, and report skipped body nodes.

    Idempotent and non-throwing, so it is safe in a ``finally`` for a capture that raised.
    """
    global _dropped_body_nodes, _handler
    with _handler_lock:
        handler = _handler
        _handler = None
        if handler is not None:
            try:
                from torch.profiler._cupti.monitor import CuptiMonitor

                monitor = CuptiMonitor()
                try:
                    monitor.disarm_callback(handler.domain, handler.cbid)
                except Exception:
                    logger.exception("graph-node callback disarm failed")
                try:
                    monitor.unregister_callback_handler(handler)
                except Exception:
                    logger.exception("graph-node callback unregister failed")
            except Exception:
                logger.exception("graph-node callback monitor teardown failed")
    with _dropped_body_nodes_lock:
        dropped_body_nodes = _dropped_body_nodes
        _dropped_body_nodes = 0
    # Report after teardown rather than running Python warning machinery in CUPTI's C
    # dispatch. If a warning filter raises, log it to preserve disarm's non-throwing contract.
    if dropped_body_nodes:
        try:
            warnings.warn(
                f"CUDA graph node observation skipped {dropped_body_nodes} node(s) inside a "
                "child-graph or conditional-node body (torch.cond / torch.while_loop). "
                "Such a body is captured into a separate cudaGraph_t whose node ids are "
                "never remapped to the exec graph, so its annotations or Python launch "
                "stacks would match nothing in a profiler trace.",
                stacklevel=3,
            )
        except Exception:
            logger.exception("graph-node callback drop warning failed")
