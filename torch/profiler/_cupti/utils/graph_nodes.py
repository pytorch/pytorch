# mypy: allow-untyped-defs
"""CUDA-graph node toolsId primitives shared by the comm watchdog.

``cudaGraphNodeGetToolsId(node)`` is the same id the CUPTI profiler reports as a
record's ``graph_node_id``. :data:`HAVE_NODE_TOOLS_ID` says whether the running
driver exposes it (``cuda.bindings`` + ``cudaGraphNodeGetToolsId``, CUDA >= 13.1 /
cuda-compat); it is False otherwise. The toolsId encodes the CAPTURE graph's id in
its high 32 bits; after instantiation the profiler uses the EXEC graph's id, so
:func:`remap_to_exec` rewrites the high bits (mirrors
``_graph_annotations.remap_to_exec_graph``).

The capture-time node walk that uses these (to find a collective's kernel nodes)
lives with its only user, ``_GraphCommAnchor`` in
``torch.profiler._cupti.comms.hook``.
"""

from __future__ import annotations

from typing import Any


try:
    from cuda.bindings import runtime as _rt  # pyrefly: ignore[missing-import]

    _HAS_CUDA_BINDINGS = True
except ImportError:
    _rt: Any = None
    _HAS_CUDA_BINDINGS = False


def _check(ret: Any) -> Any:
    err, *rest = ret if isinstance(ret, (tuple, list)) else (ret,)
    if err != _rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cuda.bindings runtime call failed: {err}")
    return rest[0] if len(rest) == 1 else tuple(rest)


def _tools_id_available() -> bool:
    if not _HAS_CUDA_BINDINGS or not hasattr(_rt, "cudaGraphNodeGetToolsId"):
        return False
    # A null node returns InvalidValue if the API exists, NewerDriver if not.
    err, *_ = _rt.cudaGraphNodeGetToolsId(0)
    return err != _rt.cudaError_t.cudaErrorCallRequiresNewerDriver


HAVE_NODE_TOOLS_ID = _tools_id_available()


def remap_to_exec(tools_ids: list[int], exec_graph_handle: int) -> list[int]:
    """Rewrite capture-graph toolsIds to the exec graph's ids (high 32 bits ->
    exec graph id). ``exec_graph_handle`` is ``CUDAGraph.raw_cuda_graph_exec()``."""
    if not HAVE_NODE_TOOLS_ID or not tools_ids:
        return list(tools_ids)
    handle = _rt.cudaGraphExec_t(init_value=exec_graph_handle)
    exec_id = _check(_rt.cudaGraphExecGetId(handle))
    return [(exec_id << 32) | (tid & 0xFFFFFFFF) for tid in tools_ids]
