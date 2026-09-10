# mypy: allow-untyped-defs
import ast
import gzip
import json
import logging
import os
import re
import tempfile
from bisect import bisect_left
from collections import defaultdict
from typing import Any

import torch._inductor.config as inductor_config
from torch.utils._ordered_set import OrderedSet


__all__ = ["inductor_trace_handler"]

log = logging.getLogger(__name__)

# https://github.com/pytorch/kineto/blob/a054a4be0db117c579a21747debf19c863631f26/libkineto/src/output_json.cpp#L559
# Kernel and CUDA runtime API events are connected by ac2g; forward and backward
# ATen ops are connected by fwdbwd.
_profile_flow_types = OrderedSet(["fwdbwd", "ac2g"])


class _EventItem:
    def __init__(self, event):
        self.e = event
        self.start, self.end = self._interval_of()

    def _interval_of(self):
        start = self.e["ts"]
        duration = self.e.get("dur", 0)
        return start, start + duration


def _find_events_covered_in(
    events: list[dict[str, Any]],
    top_level_events: list[dict[str, Any]],
):
    """Find events covered in top level events.

    Returns a dict mapping top-level event uid to covered event uids.
    """
    top_level_by_tid = defaultdict(list)
    for event in top_level_events:
        top_level_by_tid[event["tid"]].append(_EventItem(event))
    for tid in top_level_by_tid:
        top_level_by_tid[tid] = sorted(top_level_by_tid[tid], key=lambda x: x.start)
    starts_by_tid = {
        tid: [event.start for event in event_items]
        for tid, event_items in top_level_by_tid.items()
    }
    top_level_id_to_events = defaultdict(OrderedSet)

    for event in events:
        if event.get("cat") not in {
            "cuda_runtime",
            "cuLaunchKernel",
            "cpu_op",
            "cuda_driver",
            "gpu_memset",
            "python_function",
        }:
            continue
        if "CallFrom" in event.get("args", {}):
            continue
        tid = event.get("tid")
        if tid not in top_level_by_tid:
            continue
        item = _EventItem(event)
        event_items = top_level_by_tid[tid]
        idx = bisect_left(starts_by_tid[tid], item.start) - 1
        if idx < 0 or idx >= len(event_items):
            continue
        top_level_item = event_items[idx]
        if top_level_item.start <= item.start and item.end <= top_level_item.end:
            top_level_id_to_events[top_level_item.e["uid"]].add(event["uid"])
    return top_level_id_to_events


class _InductorTraceProcessor:
    def _assign_uniq_id_to_event(self, trace):
        if trace.get("uid_assigned"):
            return {event["uid"]: event for event in trace["traceEvents"]}
        uid_2_events = {}
        for uid, event in enumerate(trace["traceEvents"]):
            event["uid"] = uid
            uid_2_events[uid] = event
        trace["uid_assigned"] = True
        return uid_2_events

    def _build_flow_mapping(self, trace, flow_events):
        """Build src/dst event mappings from profiler flow events."""
        if not flow_events:
            return {}, {}

        flow_pair: dict[int, list[int | None]] = {}
        prev_event = None
        for event in trace["traceEvents"]:
            if (
                event.get("name") not in _profile_flow_types
                and event.get("cat") not in _profile_flow_types
            ):
                prev_event = event
                continue
            if event.get("ph") == "s":
                if prev_event is None:
                    continue
                pair = flow_pair.setdefault(int(event["id"]), [None, None])
                pair[0] = prev_event["uid"]
            elif event.get("ph") == "f":
                if prev_event is None:
                    continue
                pair = flow_pair.setdefault(int(event["id"]), [None, None])
                pair[1] = prev_event["uid"]
            prev_event = event

        src2dst = {}
        dst2src = {}
        for src, dst in flow_pair.values():
            if src is None or dst is None:
                continue
            src2dst[src] = dst
            dst2src[dst] = src
        return src2dst, dst2src

    def _maybe_triton_call(self, kernel_name):
        return kernel_name.startswith("triton_")

    def _has_kernel_info(self, compile_info, graph_key, kernel_name):
        graph_info = compile_info.get(graph_key, {})
        if kernel_name in graph_info:
            return True
        kernel_prefix = kernel_name + ":"
        return any(name.startswith(kernel_prefix) for name in graph_info)

    def _maybe_extern_call(self, event, compile_info, fwd_key, bw_keys, need_bw):
        kernel_name = event["name"]
        return not self._has_kernel_info(compile_info, fwd_key, kernel_name) and (
            not need_bw
            or all(
                not self._has_kernel_info(compile_info, bw_key, kernel_name)
                for bw_key in bw_keys
            )
        )

    def add_to_chrome_trace(self, origin_trace):
        """Add Inductor kernel stack information to a Chrome trace."""
        from torch._inductor.debug import get_kernel_information_jsons

        compile_linenos = get_kernel_information_jsons()
        if not compile_linenos:
            return origin_trace

        trace = origin_trace
        uid_2_events = self._assign_uniq_id_to_event(trace)

        real_events = []
        flow_events = []
        compiled_events = []
        extern_events = []
        for event in trace["traceEvents"]:
            name = event.get("name", "")
            cat = event.get("cat", "")
            if name in _profile_flow_types or cat in _profile_flow_types:
                flow_events.append(event)
                continue

            real_events.append(event)
            if cat == "cpu_op" and (
                name == "CompiledFunctionBackward" or "Torch-Compiled Region" in name
            ):
                compiled_events.append(event)
            if (
                name
                and "extern_kernels" in name
                and cat
                in (
                    "cpu_op",
                    "python_function",
                )
            ):
                extern_events.append(event)

        compiled_events = sorted(compiled_events, key=lambda x: _EventItem(x).start)
        compiled_event_items = [_EventItem(event) for event in compiled_events]
        compiled_event_starts = [item.start for item in compiled_event_items]
        ops_in_compile_region = _find_events_covered_in(real_events, compiled_events)
        ops_in_extern_region = _find_events_covered_in(real_events, extern_events)
        src2dst, dst2src = self._build_flow_mapping(trace, flow_events)
        kernel_uids_by_external_id = defaultdict(list)
        for event in real_events:
            if event.get("cat") != "kernel":
                continue
            external_id = event.get("args", {}).get("External id")
            if external_id is not None:
                kernel_uids_by_external_id[external_id].append(event["uid"])

        def _related_compile_region(compile_event):
            src_event = uid_2_events[dst2src[compile_event["uid"]]]
            src_item = _EventItem(src_event)
            idx = bisect_left(compiled_event_starts, src_item.start) - 1
            # Compiled regions are expected to be nearly nested around their
            # source events, so the nearest preceding region usually matches.
            while idx >= 0:
                region_item = compiled_event_items[idx]
                if (
                    src_item.start > region_item.start
                    and src_item.end < region_item.end
                ):
                    return region_item.e
                idx -= 1
            raise ValueError(f"Cannot find compile region for {compile_event}")

        def _parse_compile_region_key(key):
            try:
                parsed_key = ast.literal_eval(key)
            except (SyntaxError, ValueError):
                return None
            if (
                not isinstance(parsed_key, tuple)
                or len(parsed_key) != 2
                or not isinstance(parsed_key[0], str)
                or not isinstance(parsed_key[1], bool)
            ):
                return None
            return parsed_key

        def _backward_graph_keys(compile_info, compile_name):
            bw_graph_key = str((compile_name, True))
            keys = [bw_graph_key]
            if bw_graph_key in compile_info:
                return keys

            prefix = compile_name + "_"
            for key, info in compile_info.items():
                if not info:
                    continue
                parsed_key = _parse_compile_region_key(key)
                if parsed_key is None:
                    continue
                graph_name, is_backward = parsed_key
                if is_backward and graph_name.startswith(prefix):
                    keys.append(key)
            return keys

        def _stack_from_kernel_info(kernel_info):
            if isinstance(kernel_info, dict):
                return kernel_info.get("stack_traces")
            return kernel_info

        def _stack_for_kernel(compile_info, graph_key, kernel_name):
            graph_info = compile_info.get(graph_key, {})
            kernel_info = graph_info.get(kernel_name)
            if kernel_info is None:
                kernel_prefix = kernel_name + ":"
                for name, info in graph_info.items():
                    if name.startswith(kernel_prefix):
                        kernel_info = info
                        break
            return _stack_from_kernel_info(kernel_info)

        def _single_stack_for_graph(compile_info, graph_key):
            stacks = [
                stack
                for stack in (
                    _stack_from_kernel_info(kernel_info)
                    for kernel_info in compile_info.get(graph_key, {}).values()
                )
                if stack is not None
            ]
            if len(stacks) == 1:
                return stacks[0]
            return None

        def _assign_stack(event, compile_info, fwd_key, bw_keys, kernel_name):
            for bw_key in bw_keys:
                stack = _stack_for_kernel(compile_info, bw_key, kernel_name)
                if stack is not None:
                    event.setdefault("args", {})["stack"] = stack
                    return True
            stack = _stack_for_kernel(compile_info, fwd_key, kernel_name)
            if stack is not None:
                event.setdefault("args", {})["stack"] = stack
                return True
            return False

        def _assign_single_stack(event, compile_info, fwd_key, bw_keys):
            for bw_key in bw_keys:
                stack = _single_stack_for_graph(compile_info, bw_key)
                if stack is not None:
                    event.setdefault("args", {})["stack"] = stack
                    return True
            stack = _single_stack_for_graph(compile_info, fwd_key)
            if stack is not None:
                event.setdefault("args", {})["stack"] = stack
                return True
            return False

        def _kernel_events_for_op(op_id):
            if op_id in src2dst:
                return [uid_2_events[src2dst[op_id]]]
            external_id = uid_2_events[op_id].get("args", {}).get("External id")
            if external_id is None:
                return []
            return [
                uid_2_events[kernel_uid]
                for kernel_uid in kernel_uids_by_external_id.get(external_id, [])
            ]

        def _extern_kernel_name(name):
            match = re.search(r"\b(extern\w+)\s*,", name)
            return match.group(1) if match else name

        for compile_func_id, scoped_ops in ops_in_compile_region.items():
            compile_event = uid_2_events[compile_func_id]
            compile_name = compile_event["name"]
            need_bw = "Backward" in compile_name

            if need_bw:
                try:
                    compile_name = _related_compile_region(compile_event)["name"]
                except ValueError:
                    continue

            fw_graph_key = str((compile_name, False))
            bw_graph_keys = _backward_graph_keys(compile_linenos, compile_name)
            if fw_graph_key not in compile_linenos and all(
                bw_graph_key not in compile_linenos for bw_graph_key in bw_graph_keys
            ):
                continue

            for op_id in scoped_ops:
                kernel_events = _kernel_events_for_op(op_id)
                if not kernel_events:
                    continue
                for kernel_event in kernel_events:
                    assigned_stack = False
                    warn_if_unrecognized = False
                    if self._maybe_extern_call(
                        kernel_event,
                        compile_linenos,
                        fw_graph_key,
                        bw_graph_keys,
                        need_bw,
                    ):
                        for extern_call, related_ops in ops_in_extern_region.items():
                            if op_id in related_ops:
                                extern_event = uid_2_events[extern_call]
                                assigned_stack = _assign_stack(
                                    kernel_event,
                                    compile_linenos,
                                    fw_graph_key,
                                    bw_graph_keys,
                                    _extern_kernel_name(extern_event.get("name", "")),
                                )
                                if assigned_stack:
                                    break
                        if not assigned_stack:
                            assigned_stack = _assign_single_stack(
                                kernel_event,
                                compile_linenos,
                                fw_graph_key,
                                bw_graph_keys,
                            )
                    elif not self._maybe_triton_call(kernel_event.get("name", "")):
                        warn_if_unrecognized = True
                        assigned_stack = _assign_single_stack(
                            kernel_event,
                            compile_linenos,
                            fw_graph_key,
                            bw_graph_keys,
                        )
                    else:
                        assigned_stack = _assign_stack(
                            kernel_event,
                            compile_linenos,
                            fw_graph_key,
                            bw_graph_keys,
                            kernel_event.get("name", ""),
                        )
                    if not assigned_stack and warn_if_unrecognized:
                        log.warning(
                            "Kernel %s cannot be recognized as a custom kernel or triton kernel. "
                            "Please try profile with stack=True.",
                            kernel_event.get("name", ""),
                        )
        return trace


def _add_inductor_provenance(path):
    from torch._inductor.debug import get_kernel_information_jsons

    try:
        # Kineto owns the initial serialization path.  Re-read the exported
        # trace here so timeline provenance stays decoupled from Kineto's
        # JSON writer and remains a Python-only post-processing step.
        with open(path) as f:
            trace = json.load(f)

        num_events = len(trace.get("traceEvents", []))
        max_events = inductor_config.trace.provenance_tracking_max_events
        if max_events > 0 and num_events > max_events:
            log.warning(
                "Skipping provenance tracking: trace has %d events "
                "(exceeds limit of %d). Set TORCH_COMPILE_DEBUG_MAX_EVENTS=0 "
                "to disable this protection or increase the limit.",
                num_events,
                max_events,
            )
            return

        if not inductor_config.triton.unique_kernel_names:
            log.warning(
                "Profiling trace does not contain Triton kernel stack traces "
                "because TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0."
            )
        if inductor_config.cpp_wrapper:
            log.warning(
                "Profiling trace does not contain compiled kernel stack traces "
                "because cpp_wrapper is enabled."
            )
            return
        try:
            log.info("Add stack trace to compiled kernel.")
            trace = _InductorTraceProcessor().add_to_chrome_trace(trace)
            with open(path, "w") as f:
                json.dump(trace, f, indent=1)
        except MemoryError:
            log.error(
                "MemoryError during add_to_chrome_trace. "
                "Try increasing TORCH_COMPILE_DEBUG_MAX_EVENTS or disable provenance tracking."
            )
            raise
        except Exception:
            log.exception("Failed to add stack trace to compiled kernel")
    finally:
        get_kernel_information_jsons().clear()


def inductor_trace_handler(
    dir_name: str,
    worker_name: str | None = None,
    use_gzip: bool = False,
    use_python_export: bool = False,
):
    """
    Outputs tracing files with Inductor provenance to dir_name.

    Provenance tracking must also be enabled with TORCH_COMPILE_DEBUG_EXTEND=1
    or trace.provenance_tracking_to_timeline=True.
    """
    import socket
    import time

    from torch.profiler import kineto_available

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        path = os.path.join(dir_name, file_name)
        provenance_enabled = (
            kineto_available() and inductor_config.trace.provenance_tracking_to_timeline
        )

        if getattr(prof, "_use_cupti_monitor", False):
            observer = getattr(prof, "_cupti_profiler_observer", None)
            has_monitor_export = (
                observer is not None
                and observer.available
                and getattr(prof, "_monitor_window_id", None) is not None
            )
            prof.export_chrome_trace(
                path,
                use_python_export=use_python_export,
            )
            if provenance_enabled and has_monitor_export:
                from torch._inductor.debug import get_kernel_information_jsons

                get_kernel_information_jsons().clear()
            return

        if use_gzip:
            with tempfile.NamedTemporaryFile("w+b", suffix=".json") as fp:
                prof.export_chrome_trace(
                    fp.name,
                    use_python_export=use_python_export,
                )
                if provenance_enabled:
                    _add_inductor_provenance(fp.name)
                with open(fp.name, "rb") as fin, gzip.open(path, "wb") as fout:
                    fout.writelines(fin)
        else:
            prof.export_chrome_trace(
                path,
                use_python_export=use_python_export,
            )
            if provenance_enabled:
                _add_inductor_provenance(path)

    return handler_fn
