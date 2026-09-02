"""Capture Python launch stacks for CUDA graph nodes."""

from __future__ import annotations

import os
import re
import threading
import warnings
from functools import lru_cache
from typing import Any

from torch.utils._traceback import CapturedTraceback, shorten_filename


_MAX_STACKS_PER_GRAPH = 100_000
_TORCH_ROOT = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
_CWD = os.path.realpath(os.getcwd())
_HOME = os.path.realpath(os.path.expanduser("~"))
_FRAME_PATTERN = re.compile(r'^\s*File "(.*)", line ([0-9]+), in (.*)$')

_active_graph: Any = None
_state_lock = threading.RLock()


def _begin_capture(graph: Any) -> None:
    """Attach a bounded stack store to the graph for the active capture."""
    global _active_graph

    with _state_lock:
        if _active_graph is not None:
            raise RuntimeError("CUDA graph Python stack capture is already active")
        clear_stacks(graph)
        graph._py_stack_dropped = 0
        _active_graph = graph


def _end_capture(graph: Any) -> None:
    """Stop recording stacks for ``graph``. Idempotent for error cleanup."""
    global _active_graph

    with _state_lock:
        if _active_graph is graph:
            _active_graph = None


def _is_capturing() -> bool:
    with _state_lock:
        return _active_graph is not None


def _record(tools_id: int) -> None:
    """Record a node stack from the shared CUPTI graph-node callback."""
    with _state_lock:
        graph = _active_graph
        if graph is None or tools_id in graph._py_stack_traces:
            return
        if len(graph._py_stack_traces) >= _MAX_STACKS_PER_GRAPH:
            graph._py_stack_dropped += 1
            return
        graph._py_stack_traces[tools_id] = CapturedTraceback.extract(skip=1)


@lru_cache(maxsize=4096)
def _format_filename(filename: str) -> str | None:
    real_filename = os.path.realpath(filename)
    try:
        if os.path.commonpath((real_filename, _TORCH_ROOT)) == _TORCH_ROOT:
            return None
    except ValueError:
        pass
    for base in (_CWD, _HOME):
        if os.path.dirname(base) == base:
            continue
        try:
            if os.path.commonpath((real_filename, base)) == base:
                return shorten_filename(real_filename, base=base)
        except ValueError:
            pass
    return os.path.basename(filename)


def _format_stack(frames: list[str]) -> str:
    stack: list[str] = []
    for frame in reversed(frames):
        lines = frame.splitlines()
        if not lines:
            continue
        match = _FRAME_PATTERN.match(lines[0])
        if match is None:
            continue
        filename, lineno, name = match.groups()
        filename = _format_filename(filename)
        if filename is not None:
            stack.append(f"{filename}:{lineno}:{name}")
    return "\n".join(stack) or "<no Python frames outside PyTorch>"


def take_stacks(graph: Any) -> dict[int, str]:
    r"""take_stacks(graph) -> dict[int, str]

    Return and clear the Python launch stacks recorded for a CUDA graph.

    Keys are exec-graph ``toolsId`` values after the graph has been instantiated.
    Formatting and path shortening happen here, outside the capture callback.

    Args:
        graph (torch.cuda.CUDAGraph): the graph whose captured stacks are returned.

    Returns:
        dict[int, str]: A mapping from graph-node ``toolsId`` to its Python stack.
    """
    with _state_lock:
        if graph._py_stack_traces and graph._remapped_exec_id is None:
            raise RuntimeError("instantiate the CUDA graph before taking Python stacks")
        traces = graph._py_stack_traces
        graph._py_stack_traces = {}
        dropped = graph._py_stack_dropped
        graph._py_stack_dropped = 0

    cache: dict[str, str] = {}
    stacks: dict[int, str] = {}
    try:
        formatted = CapturedTraceback.format_all(list(traces.values()))
        for tools_id, frames in zip(traces, formatted):
            stack = _format_stack(frames)
            stacks[tools_id] = cache.setdefault(stack, stack)
    finally:
        for trace in traces.values():
            trace.cleanup()
    if dropped:
        warnings.warn(
            f"CUDA graph Python stack capture dropped {dropped} node(s) after its "
            f"{_MAX_STACKS_PER_GRAPH}-node limit",
            stacklevel=3,
        )
    return stacks


def clear_stacks(graph: Any) -> None:
    r"""clear_stacks(graph) -> None

    Discard Python launch stacks recorded for a CUDA graph.

    Args:
        graph (torch.cuda.CUDAGraph): the graph whose captured stacks are discarded.
    """
    with _state_lock:
        for trace in graph._py_stack_traces.values():
            trace.cleanup()
        graph._py_stack_traces.clear()
        graph._py_stack_dropped = 0


def _remap_to_exec_graph(graph: Any, capture_graph_id: int, exec_graph_id: int) -> None:
    from torch.cuda._graph_annotations import _rekey_annotations

    with _state_lock:
        graph._py_stack_traces = _rekey_annotations(
            graph._py_stack_traces, capture_graph_id, exec_graph_id
        )


__all__ = ["clear_stacks", "take_stacks"]
