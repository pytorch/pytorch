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

import sys
import warnings
from logging import getLogger
from typing import Any


logger = getLogger(__name__)


# The handler token from Cuspy. Carries the (domain, cbid) it was registered for, so
# it is also what arm/disarm address the callback by; kept so disarm can unregister it.
_handler: Any = None
_record_py_stacks: bool = False
_stack_cache: dict[tuple[tuple[str, int, str], ...], str] = {}
_noise_cache: dict[str, bool] = {}
_STACK_NOISE = (
    "/torch/",
    "/cuda/bindings/",
    "/cupti/",
    "/triton/runtime/",
    "/triton/backends/",
    "/importlib/",
    "<frozen",
)

# Nodes dropped during the armed capture for belonging to a child-graph or conditional
# body. Counted rather than warned about on the spot: warnings.warn can be configured to
# raise, and that must not happen inside CUPTI's C dispatch. disarm() reports the total.
_dropped_body_nodes: int = 0


def _capture_py_stack() -> str | None:
    r"""Capture user frames, innermost first, sharing strings for repeated call sites."""
    frames: list[tuple[str, int, str]] = []
    frame = sys._getframe(1)
    try:
        # Bound both the walk and its output for deeply nested launch stacks.
        for _ in range(256):
            if frame is None or len(frames) == 64:
                break
            code = frame.f_code
            filename = code.co_filename
            noise = _noise_cache.get(filename)
            if noise is None:
                path = filename.replace("\\", "/")
                noise = any(part in path for part in _STACK_NOISE)
                _noise_cache[filename] = noise
            if not noise:
                frames.append((filename, frame.f_lineno, code.co_name))
            frame = frame.f_back
    finally:
        del frame
    if not frames:
        return None
    key = tuple(frames)
    stack = _stack_cache.get(key)
    if stack is None:
        stack = "\n".join(
            f"{filename}:{line}:{name}" for filename, line, name in frames
        )
        _stack_cache[key] = stack
    return stack


def _on_graph_node_created(_domain: int, _cbid: int, cbdata: int) -> None:
    r"""Record the ambient annotation and optional launch stack for a new graph node.

    Runs on the capturing thread inside the CUDA call. ``cbdata`` is a raw
    ``CUpti_ResourceData*``; its ``resource_descriptor`` is the ``CUpti_GraphData`` carrying
    the node handle and type.
    """
    from cuda.bindings import runtime  # pyrefly: ignore[missing-import]
    from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

    from torch.cuda._graph_annotations import (
        _get_annotatable_types,
        _kernel_py_stacks,
        capture_root_graph_id,
        current_annotation,
        node_type_has_source_id,
        note_body_graph_id,
        note_sourceless_node,
        record_node_annotation,
        source_keyed,
    )
    from torch.cuda._utils import _check_cuda_bindings

    # Cheapest rejection first: the node type is already in hand, the annotation costs a
    # merge when scopes are nested, and the toolsId lookup is a driver call.
    graph_data = _cupti.GraphData.from_ptr(
        _cupti.ResourceData.from_ptr(cbdata).resource_descriptor
    )
    if graph_data.node_type not in _get_annotatable_types():
        return
    annotation = current_annotation()
    if annotation is None and not _record_py_stacks:
        return
    # arm() requires a capture-root stamp, which gates on toolsId support. An error
    # here is unexpected; Cuspy logs it without letting it reach CUPTI's C dispatch.
    tools_id = _check_cuda_bindings(runtime.cudaGraphNodeGetToolsId(graph_data.node))
    # Nodes reported for any other graph belong to a child-graph or conditional body. Their
    # ids are in that body graph's own space, which remap_to_exec_graph does not rekey, so
    # they are only worth recording when the annotations stay on the capture graph: under
    # key_by="source" CUPTI reports the body work with a sourceGraphNodeId in exactly that
    # space, and the entry resolves. Otherwise the key would match nothing -- drop it, and
    # let disarm() warn about the total.
    body_graph_id = tools_id >> 32
    if body_graph_id != capture_root_graph_id():
        if not source_keyed():
            global _dropped_body_nodes
            _dropped_body_nodes += 1
            return
        # Neither the capture nor the exec graph's id, so the destroy purge needs telling.
        note_body_graph_id(body_graph_id)
    # The node type is only in hand here, and the registry needs it to know which entries
    # a source-keyed capture must still alias into exec space (see
    # _graph_annotations.note_sourceless_node).
    if not node_type_has_source_id(graph_data.node_type):
        note_sourceless_node(tools_id)
    if annotation is not None:
        record_node_annotation(tools_id, annotation)
    if _record_py_stacks:
        stack = _capture_py_stack()
        if stack is not None:
            _kernel_py_stacks[tools_id] = stack


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


def register(*, force: bool = False, record_py_stacks: bool = False) -> bool:
    r"""Register the node-creation handler, bringing the CUPTI subscription up.

    Separate from :func:`arm` -- and called *before* ``capture_begin`` -- so that failing to
    obtain CUPTI cannot leave a capture half-started. Returns ``False`` when the backend is
    unavailable, so the caller can fall back to the edge walk.

    ``force`` brings Cuspy up instead of requiring a live subscription. That is a
    deliberate, opt-in cost: once we hold a CUPTI subscription, kineto's one-shot init fails
    permanently, so a later ``torch.profiler`` run records no GPU activity.
    ``backend="cupti"`` and ``record_py_stacks=True`` opt into this cost.
    """
    global _handler, _record_py_stacks
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
    _record_py_stacks = record_py_stacks
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
    global _handler, _record_py_stacks
    if _handler is None:
        return
    from torch.profiler._cuspy.core import Cuspy

    cuspy = Cuspy()
    try:
        cuspy.disarm_callback(_handler.domain, _handler.cbid)
        cuspy.unregister_callback_handler(_handler)
    finally:
        _handler = None
        _record_py_stacks = False
        # The registry owns the strings now; caches only need to span one capture.
        _stack_cache.clear()
        _noise_cache.clear()
    # Warn here rather than from the handler: this runs on the normal path, where a
    # warnings filter promoting warnings to errors is harmless. The edge walk reports the
    # same situation at scope entry; reporting it on the drop instead covers both a scope
    # inside a body and a scope containing one, and says how much was actually lost.
    if _dropped_body_nodes:
        warnings.warn(
            f"mark_kernels: {_dropped_body_nodes} node(s) created inside a CUDA graph "
            "child-graph or conditional-node body (torch.cond / torch.while_loop) were "
            "not annotated -- such a body is captured into a separate cudaGraph_t whose "
            "node ids are not remapped to the exec graph, so an annotation there would "
            "match nothing in a trace",
            stacklevel=2,
        )
