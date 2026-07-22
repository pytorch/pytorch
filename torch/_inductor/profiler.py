from __future__ import annotations

import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
import weakref
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from . import config


if TYPE_CHECKING:
    from torch.profiler import profile


__all__ = ["inductor_trace_handler"]

log = logging.getLogger(__name__)

_FLOW_TYPES = {"fwdbwd", "ac2g"}
_COMPILED_GRAPH_PREFIX = "## Call CompiledFxGraph "
_COMPILED_GRAPH_SUFFIX = " ##"
_SCOPED_EVENT_CATEGORIES = {
    "cuda_runtime",
    "cuLaunchKernel",
    "cpu_op",
    "cuda_driver",
    "gpu_memset",
    "python_function",
}

_compiled_graphs: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
_compiled_graphs_lock = threading.Lock()


def _register_compiled_graph(key: str, graph: Any) -> None:
    with _compiled_graphs_lock:
        _compiled_graphs[key] = graph


def _registered_graph_provenance() -> dict[str, dict[str, Any]]:
    with _compiled_graphs_lock:
        graphs = list(_compiled_graphs.items())
    provenance: dict[str, dict[str, Any]] = {}
    for key, graph in graphs:
        stack_traces = getattr(graph, "inductor_provenance_stack_traces_str", None)
        if not stack_traces:
            continue
        try:
            kernel_info = json.loads(stack_traces)
        except (TypeError, json.JSONDecodeError):
            log.warning("Ignoring invalid Inductor provenance for graph %s", key)
            continue
        if isinstance(kernel_info, dict) and kernel_info:
            provenance[key] = kernel_info
    return provenance


@dataclass(frozen=True)
class _EventItem:
    uid: int
    start: int | float
    end: int | float

    @classmethod
    def from_event(cls, uid: int, event: dict[str, Any]) -> _EventItem:
        start = event["ts"]
        return cls(uid, start, start + event.get("dur", 0))


def _find_events_covered_in(
    events: list[tuple[int, dict[str, Any]]],
    top_level_events: list[tuple[int, dict[str, Any]]],
) -> dict[int, set[int]]:
    r"""Map each region to the events in its innermost containing interval."""
    top_level_by_tid: dict[int, list[_EventItem]] = defaultdict(list)
    for uid, event in top_level_events:
        top_level_by_tid[event["tid"]].append(_EventItem.from_event(uid, event))

    starts_by_tid: dict[int, list[int | float]] = {}
    max_ends_by_tid: dict[int, list[int | float]] = {}
    for tid, event_items in top_level_by_tid.items():
        event_items.sort(key=lambda item: (item.start, -item.end))
        starts_by_tid[tid] = [item.start for item in event_items]
        max_ends: list[int | float] = []
        for item in event_items:
            max_ends.append(max(max_ends[-1], item.end) if max_ends else item.end)
        max_ends_by_tid[tid] = max_ends

    covered: dict[int, set[int]] = defaultdict(set)
    for uid, event in events:
        tid = event.get("tid")
        if tid not in top_level_by_tid:
            continue
        item = _EventItem.from_event(uid, event)
        event_items = top_level_by_tid[tid]
        idx = bisect_right(starts_by_tid[tid], item.start) - 1
        while idx >= 0:
            if max_ends_by_tid[tid][idx] < item.end:
                break
            region = event_items[idx]
            if (
                region.uid != uid
                and region.start <= item.start
                and item.end <= region.end
            ):
                covered[region.uid].add(uid)
                break
            idx -= 1
    return covered


def _flow_type(event: dict[str, Any]) -> str | None:
    name = event.get("name")
    category = event.get("cat")
    if name in _FLOW_TYPES:
        return name
    if category in _FLOW_TYPES:
        return category
    return None


def _build_flow_mapping(
    events: list[tuple[int, dict[str, Any]]], flow_type: str
) -> tuple[dict[int, int], dict[int, int]]:
    r"""Build source/destination mappings for one profiler flow namespace."""
    flow_pairs: dict[tuple[str, int], list[int | None]] = {}
    previous_real_event: int | None = None
    for uid, event in events:
        event_flow_type = _flow_type(event)
        if event_flow_type is None:
            previous_real_event = uid
            continue
        if event_flow_type != flow_type or previous_real_event is None:
            continue

        pair = flow_pairs.setdefault((event_flow_type, int(event["id"])), [None, None])
        if event.get("ph") == "s":
            pair[0] = previous_real_event
        elif event.get("ph") == "f":
            pair[1] = previous_real_event

    src_to_dst = {}
    dst_to_src = {}
    for src, dst in flow_pairs.values():
        if src is not None and dst is not None:
            src_to_dst[src] = dst
            dst_to_src[dst] = src
    return src_to_dst, dst_to_src


def _compiled_graph_key(event: dict[str, Any]) -> str | None:
    name = event.get("name", "")
    if not isinstance(name, str) or not (
        name.startswith(_COMPILED_GRAPH_PREFIX)
        and name.endswith(_COMPILED_GRAPH_SUFFIX)
    ):
        return None
    return name[len(_COMPILED_GRAPH_PREFIX) : -len(_COMPILED_GRAPH_SUFFIX)]


def _stack_from_kernel_info(kernel_info: Any) -> list[str] | None:
    if isinstance(kernel_info, dict):
        return kernel_info.get("stack_traces")
    return kernel_info


def _stack_for_kernel(graph_info: dict[str, Any], kernel_name: str) -> list[str] | None:
    kernel_info = graph_info.get(kernel_name)
    if kernel_info is None:
        prefix = kernel_name + ":"
        kernel_info = next(
            (info for name, info in graph_info.items() if name.startswith(prefix)),
            None,
        )
    return _stack_from_kernel_info(kernel_info)


def _single_stack_for_graph(graph_info: dict[str, Any]) -> list[str] | None:
    stacks: list[list[str]] = []
    for kernel_info in graph_info.values():
        stack = _stack_from_kernel_info(kernel_info)
        if stack is not None and stack not in stacks:
            stacks.append(stack)
    return stacks[0] if len(stacks) == 1 else None


def _assign_stack(
    event: dict[str, Any],
    graph_info: dict[str, Any],
    kernel_name: str,
) -> bool:
    stack = _stack_for_kernel(graph_info, kernel_name)
    if stack is None:
        return False
    event.setdefault("args", {})["stack"] = stack
    return True


def _assign_single_stack(event: dict[str, Any], graph_info: dict[str, Any]) -> bool:
    stack = _single_stack_for_graph(graph_info)
    if stack is None:
        return False
    event.setdefault("args", {})["stack"] = stack
    return True


def _has_kernel_info(graph_info: dict[str, Any], kernel_name: str) -> bool:
    return _stack_for_kernel(graph_info, kernel_name) is not None


def _extern_kernel_name(name: str) -> str:
    match = re.search(r"\b(extern\w+)\s*,", name)
    return match.group(1) if match else name


def _add_inductor_kernel_stacks(
    trace: dict[str, Any],
    graph_provenance: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    r"""Add Inductor source stacks to GPU kernel events in a Chrome trace."""
    if graph_provenance is None:
        graph_provenance = _registered_graph_provenance()
    trace_events = trace.get("traceEvents")
    if not graph_provenance or not isinstance(trace_events, list):
        return trace

    indexed_events = list(enumerate(trace_events))
    real_events: list[tuple[int, dict[str, Any]]] = []
    compiled_graph_calls: list[tuple[int, dict[str, Any]]] = []
    extern_events: list[tuple[int, dict[str, Any]]] = []
    kernel_uids_by_external_id: dict[Any, list[int]] = defaultdict(list)

    for uid, event in indexed_events:
        if _flow_type(event) is not None:
            continue
        real_events.append((uid, event))
        name = event.get("name", "")
        category = event.get("cat", "")
        if _compiled_graph_key(event) is not None:
            compiled_graph_calls.append((uid, event))
        if name and "extern_kernels" in name and category in {
            "cpu_op",
            "python_function",
        }:
            extern_events.append((uid, event))
        if category == "kernel":
            external_id = event.get("args", {}).get("External id")
            if external_id is not None:
                kernel_uids_by_external_id[external_id].append(uid)

    graph_info_by_call = {
        call_uid: graph_info
        for call_uid, call_event in compiled_graph_calls
        if (key := _compiled_graph_key(call_event)) is not None
        and (graph_info := graph_provenance.get(key))
    }
    if not graph_info_by_call:
        return trace

    calls_with_provenance = [
        event for event in compiled_graph_calls if event[0] in graph_info_by_call
    ]
    scoped_events = [
        (uid, event)
        for uid, event in real_events
        if event.get("cat") in _SCOPED_EVENT_CATEGORIES
        and "CallFrom" not in event.get("args", {})
    ]
    ops_by_call = _find_events_covered_in(scoped_events, calls_with_provenance)
    ops_by_extern = _find_events_covered_in(scoped_events, extern_events)
    src_to_dst, _ = _build_flow_mapping(indexed_events, "ac2g")

    def kernel_events_for_op(op_uid: int) -> list[dict[str, Any]]:
        if op_uid in src_to_dst:
            return [trace_events[src_to_dst[op_uid]]]
        external_id = trace_events[op_uid].get("args", {}).get("External id")
        if external_id is None:
            return []
        return [
            trace_events[kernel_uid]
            for kernel_uid in kernel_uids_by_external_id.get(external_id, [])
        ]

    for call_uid, op_uids in ops_by_call.items():
        graph_info = graph_info_by_call[call_uid]
        for op_uid in op_uids:
            for kernel_event in kernel_events_for_op(op_uid):
                kernel_name = kernel_event.get("name", "")
                assigned_stack = False
                warn_if_unrecognized = False
                if not _has_kernel_info(graph_info, kernel_name):
                    for extern_uid, related_ops in ops_by_extern.items():
                        if op_uid not in related_ops:
                            continue
                        assigned_stack = _assign_stack(
                            kernel_event,
                            graph_info,
                            _extern_kernel_name(
                                trace_events[extern_uid].get("name", "")
                            ),
                        )
                        if assigned_stack:
                            break
                    if not assigned_stack:
                        assigned_stack = _assign_single_stack(
                            kernel_event, graph_info
                        )
                elif not kernel_name.startswith("triton_"):
                    warn_if_unrecognized = True
                    assigned_stack = _assign_single_stack(kernel_event, graph_info)
                else:
                    assigned_stack = _assign_stack(
                        kernel_event, graph_info, kernel_name
                    )
                if not assigned_stack and warn_if_unrecognized:
                    log.warning(
                        "Kernel %s cannot be recognized as a custom kernel or "
                        "Triton kernel. Try profiling with with_stack=True.",
                        kernel_name,
                    )
    return trace


def _export_inductor_trace(
    prof: profile,
    path: str,
    *,
    use_python_export: bool = False,
) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    raw_fd, raw_path = tempfile.mkstemp(dir=output_dir, suffix=".pt.trace.json")
    os.close(raw_fd)
    processed_path: str | None = None

    try:
        prof.export_chrome_trace(raw_path, use_python_export=use_python_export)
    except Exception:
        os.unlink(raw_path)
        raise

    def install_raw_trace() -> None:
        nonlocal raw_path
        os.replace(raw_path, path)
        raw_path = ""

    try:
        graph_provenance = _registered_graph_provenance()
        if not graph_provenance:
            install_raw_trace()
            return

        if not config.triton.unique_kernel_names:
            log.warning(
                "Inductor profiler trace may omit Triton kernel stacks because "
                "TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0."
            )
        if config.cpp_wrapper:
            log.warning(
                "Inductor profiler trace cannot add kernel stacks when cpp_wrapper "
                "is enabled."
            )
            install_raw_trace()
            return

        with open(raw_path) as f:
            trace = json.load(f)

        num_events = len(trace.get("traceEvents", []))
        max_events = config.trace.provenance_tracking_max_events
        if max_events > 0 and num_events > max_events:
            log.warning(
                "Skipping Inductor provenance: trace has %d events "
                "(exceeds limit of %d). Set TORCH_COMPILE_DEBUG_MAX_EVENTS=0 "
                "to disable this protection or increase the limit.",
                num_events,
                max_events,
            )
            install_raw_trace()
            return

        _add_inductor_kernel_stacks(trace, graph_provenance)

        processed_fd, processed_path = tempfile.mkstemp(
            dir=output_dir, suffix=".pt.trace.json"
        )
        with os.fdopen(processed_fd, "w") as f:
            json.dump(trace, f, indent=1)
        os.replace(processed_path, path)
        processed_path = None
        os.unlink(raw_path)
        raw_path = ""
    except MemoryError:
        install_raw_trace()
        log.error(
            "MemoryError while adding Inductor provenance; preserved the raw trace "
            "at %s.",
            path,
        )
        raise
    except Exception:
        install_raw_trace()
        log.exception(
            "Failed to add Inductor provenance; preserved the raw trace at %s", path
        )
    finally:
        if raw_path:
            try:
                os.unlink(raw_path)
            except FileNotFoundError:
                pass
        if processed_path is not None:
            try:
                os.unlink(processed_path)
            except FileNotFoundError:
                pass


def inductor_trace_handler(
    dir_name: str,
    worker_name: str | None = None,
    use_python_export: bool = False,
) -> Callable[[profile], None]:
    r"""inductor_trace_handler(dir_name, worker_name=None, use_python_export=False) -> Callable

    Create an ``on_trace_ready`` callback that exports Inductor provenance.

    Provenance must be enabled while compiling and profiling with
    ``trace.provenance_tracking_to_timeline`` or ``TORCH_COMPILE_DEBUG_EXTEND=1``.

    Args:
        dir_name (str): directory into which trace files are written.
        worker_name (str, optional): prefix that identifies the worker. Default:
          the host name and process ID.
        use_python_export (bool, optional): use the Python Chrome trace exporter.
          Default: ``False``.

    Returns:
        Callable: a callback suitable for ``torch.profiler.profile(on_trace_ready=...)``.

    Examples::

        >>> # xdoctest: +SKIP
        >>> from torch._inductor.profiler import inductor_trace_handler
        >>> handler = inductor_trace_handler("./traces")
        >>> with torch.profiler.profile(on_trace_ready=handler):
        ...     compiled_model(inputs)
    """  # noqa: B950
    worker_name = worker_name or f"{socket.gethostname()}_{os.getpid()}"

    def handler(prof: profile) -> None:
        os.makedirs(dir_name, exist_ok=True)
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        _export_inductor_trace(
            prof,
            os.path.join(dir_name, file_name),
            use_python_export=use_python_export,
        )

    return handler
