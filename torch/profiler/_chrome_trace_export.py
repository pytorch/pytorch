# mypy: allow-untyped-defs
"""Stream chrome trace JSON directly from ITraceActivity objects.

Bypasses Kineto's C++ ChromeTraceLogger, writing events one-at-a-time
through a (possibly gzip-compressed) text writer so we never materialize
the full JSON string in memory.

Optionally bakes CUDA-graph kernel annotations into the trace during that
same pass (see the ``annotations`` argument), which is why they are injected
here rather than by post-processing the written file: reading the trace back
to splice them in would pay the dominant gzip cost twice.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time as _time
from typing import Any, IO, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

import torch


_TRIMONTH_SECONDS = 7889238

_FLOW_NAMES = {1: "fwdbwd", 2: "ac2g"}

_PARAM_COMMS_CALL_NAME = "record_param_comms"

_EXCLUDED_EXTERNAL_ID_TYPES = {
    "gpu_memcpy",
    "gpu_memset",
    "kernel",
    "cuda_runtime",
    "cuda_driver",
    "privateuse1_runtime",
    "privateuse1_driver",
}

# Graphed work events the annotation pass looks up and reassigns a lane for.
# gpu_user_annotation is excluded: CUPTI replicates it onto every stream a graph
# replays on, so it is noise that follows the work rather than work itself.
_WORK_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

# A node's id is only available inside the metadataJson fragment, which is spliced
# verbatim rather than parsed. TestMetadataJsonFormat pins the spelling.
_GNODE_RE = re.compile(r'"graph node id":\s*(\d+)')

# Annotation key naming the lane its events are moved to; lanes without it keep the
# default "stream N" name. Set by whoever records the annotation (a comms wrapper
# naming a process group's lane is the motivating case).
_LANE_NAME_KEY = "Process Group Description"

# Top-level envelope flag marking a trace whose CUDA-graph annotations are already
# baked in, so a consumer can tell it from one that still needs them joined on.
_INLINE_ANNOTATED_KEY = "cudaGraphInlineAnnotated"


def _trimester_base_ns() -> int:
    """Epoch nanoseconds at the start of the current trimonth interval.

    Matches libkineto's ChromeTraceBaseTime (floor to 7889238-second
    intervals) to keep JSON timestamps small enough for double precision.
    """
    return (int(_time.time()) // _TRIMONTH_SECONDS) * _TRIMONTH_SECONDS * 1_000_000_000


def _ns_to_us(time_ns: int) -> str:
    return f"{time_ns // 1000}.{time_ns % 1000:03d}"


def _sanitize_tid(tid: int) -> int:
    if tid == -(2**63):
        return 0
    return abs(tid)


def _json_escape(s: str) -> str:
    return json.dumps(s)


def _write_metadata_event(
    f: IO[str], name: str, ts: str, pid, tid, arg_key: str, arg_value: str
):
    f.write(
        f'{{"ph":"M","name":"{name}","ts":{ts},'
        f'"pid":{pid},"tid":{tid},'
        f'"args":{{"{arg_key}":{arg_value}}}}},\n'
    )


def _cuda_versions() -> dict[str, int]:
    """CUDA driver and runtime versions, which kineto's C++ exporter writes as
    top-level trace keys from ``CuptiActivityProfiler::logGpuVersions()``.

    The Python profiler result does not expose them. The runtime version comes from
    torch, whose CUDART is the one kineto is compiled against and asks -- calling
    ``cudaRuntimeGetVersion`` through cuda.bindings would report *its* CUDART instead,
    which need not be the same one. The driver version is process-global, so any
    caller agrees on it; it needs cuda.bindings and is left out without it.
    """
    if not torch.cuda.is_available():
        return {}
    versions = {"cuda_runtime_version": torch._C._cuda_getCompiledVersion()}
    try:
        from cuda.bindings import (  # pyrefly: ignore[missing-import]
            runtime as cuda_runtime,
        )

        from torch.cuda._utils import _check_cuda_bindings

        versions["cuda_driver_version"] = _check_cuda_bindings(
            cuda_runtime.cudaDriverGetVersion()
        )
    except (ImportError, RuntimeError):
        pass
    return versions


def _device_properties() -> list[dict[str, Any]]:
    """Per-device properties in the shape kineto's ``createDevicePropertiesJson``
    writes (it serializes ``cudaDeviceProp``).

    Read through cuda.bindings when available -- the same ``cudaGetDeviceProperties``
    kineto uses, and the only source exposing ``regsPerBlock`` -- otherwise from
    torch's properties object, which carries everything but that field.
    """
    try:
        from cuda.bindings import (  # pyrefly: ignore[missing-import]
            runtime as cuda_runtime,
        )
    except ImportError:
        cuda_runtime = None  # type: ignore[assignment]

    props: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        if cuda_runtime is not None:
            err, p = cuda_runtime.cudaGetDeviceProperties(i)
            if not err.value:
                name = p.name
                props.append(
                    {
                        "id": i,
                        "name": name.decode() if isinstance(name, bytes) else name,
                        "totalGlobalMem": p.totalGlobalMem,
                        "computeMajor": p.major,
                        "computeMinor": p.minor,
                        "maxThreadsPerBlock": p.maxThreadsPerBlock,
                        "maxThreadsPerMultiprocessor": p.maxThreadsPerMultiProcessor,
                        "regsPerBlock": p.regsPerBlock,
                        "warpSize": p.warpSize,
                        "sharedMemPerBlock": p.sharedMemPerBlock,
                        "numSms": p.multiProcessorCount,
                        "regsPerMultiprocessor": p.regsPerMultiprocessor,
                        "sharedMemPerBlockOptin": p.sharedMemPerBlockOptin,
                        "sharedMemPerMultiprocessor": p.sharedMemPerMultiprocessor,
                    }
                )
                continue
        tp = torch.cuda.get_device_properties(i)
        props.append(
            {
                "id": i,
                "name": tp.name,
                "totalGlobalMem": tp.total_memory,
                "computeMajor": tp.major,
                "computeMinor": tp.minor,
                "maxThreadsPerBlock": tp.max_threads_per_block,  # pyrefly: ignore[missing-attribute]
                "maxThreadsPerMultiprocessor": tp.max_threads_per_multi_processor,  # pyrefly: ignore[missing-attribute]
                "warpSize": tp.warp_size,
                "sharedMemPerBlock": tp.shared_memory_per_block,  # pyrefly: ignore[missing-attribute]
                "numSms": tp.multi_processor_count,
                "regsPerMultiprocessor": tp.regs_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
                "sharedMemPerBlockOptin": tp.shared_memory_per_block_optin,  # pyrefly: ignore[missing-attribute]
                "sharedMemPerMultiprocessor": tp.shared_memory_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
            }
        )
    return props


def _annotation_args(annotation) -> tuple[str | None, int | None, str | None]:
    """Render one node's annotation into an args fragment, its lane, and its lane name.

    Values are stringified except ``stream``, which stays a number: it drives the
    event's ``tid``. The whole dict is escaped in one ``json.dumps`` with the braces
    stripped rather than key by key.
    """
    stream: int | None = None
    lane_name: str | None = None
    chunks: list[str] = []
    for ann in annotation if isinstance(annotation, list) else [annotation]:
        if not isinstance(ann, dict):
            chunks.append(f'"annotation": {_json_escape(str(ann))}')
            continue
        if lane_name is None and _LANE_NAME_KEY in ann:
            lane_name = str(ann[_LANE_NAME_KEY])
        norm: dict[str, Any] = {}
        for key, value in ann.items():
            if key == "stream":
                stream = int(value)
                norm["stream"] = stream
            else:
                norm[str(key)] = str(value)
        if norm:
            chunks.append(json.dumps(norm)[1:-1])
    return (", ".join(chunks) if chunks else None), stream, lane_name


def _graph_node_id(metadata_json: str) -> int:
    match = _GNODE_RE.search(metadata_json)
    return int(match.group(1)) if match else 0


def export_chrome_trace(
    kineto_results,
    path: str,
    metadata: dict[str, str] | None = None,
    annotations: Mapping[int, Any] | None = None,
    default_stream: int = 7,
):
    """Export chrome trace from ITraceActivity objects, streaming to disk.

    ``kineto_results`` is a ``_ProfilerResult`` that exposes
    ``trace_activities()`` and ``trace_start_ns()``.

    ``annotations`` maps a CUDA-graph node id to the annotation recorded for it
    (pass ``torch.cuda.graph_annotations.get_kernel_annotations()``). Graphed work
    events matching one get its fields spliced into their ``args`` and move to the
    lane it names; graphed work with no annotation moves to ``default_stream``. This
    is what a graph replay needs to be readable: CUPTI reports it on whatever
    hardware streams the graph executor picked, often hundreds of them.

    Writes ``.json`` or ``.json.gz`` depending on the file extension.
    """
    activities = kineto_results.trace_activities()
    base_ns = _trimester_base_ns()
    annotations = annotations or {}

    seen_devices: dict[int, int] = {}
    seen_resources: dict[tuple[int, int], int] = {}
    host_pid = os.getpid()
    min_ts = 2**63
    max_end_ts = 0
    has_annotations = False

    for act in activities:
        did = act.device_id()
        rid = _sanitize_tid(act.resource_id())
        ts = act.timestamp()
        dur = act.duration()
        min_ts = min(min_ts, ts)
        max_end_ts = max(max_end_ts, ts + max(dur, 0))
        if did not in seen_devices:
            seen_devices[did] = ts
        key = (did, rid)
        if key not in seen_resources:
            seen_resources[key] = ts
        # Whether the envelope flag is written, decided here with an early stop so the
        # common case costs one extra metadata_json() call on one activity. The lookups
        # themselves happen inline in the write pass below, not cached per activity.
        if not has_annotations and annotations and act.type() in _WORK_CATEGORIES:
            md = act.metadata_json()
            if md and annotations.get(_graph_node_id(md)):
                has_annotations = True

    # Lanes graphed work was moved to: (did, tid) -> (first ts, lane name or None).
    # Their thread_name metadata is written after the events, which chrome traces allow.
    reassigned: dict[tuple[int, int], tuple[int, str | None]] = {}
    # Lanes that ended up with real work, and the gpu_user_annotation slices held back
    # until it is known whether theirs did: a slice on a lane whose kernels all moved
    # away is orphaned noise.
    work_tids: set[tuple[int, int]] = set()
    deferred_annotations: list[tuple[int, int, str]] = []

    def _rel(ns: int) -> str:
        return _ns_to_us(max(ns - base_ns, 0))

    with gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w") as f:
        f.write("{\n")
        f.write('"schemaVersion": 1,\n')
        f.write(f'"deviceProperties": {json.dumps(_device_properties())},\n')
        if metadata:
            for k, v in metadata.items():
                f.write(f"{_json_escape(k)}: {v},\n")
        for version_key, version in _cuda_versions().items():
            f.write(f'"{version_key}": {version},\n')
        f.write('"displayTimeUnit": "ms",\n')
        f.write(f'"baseTimeNanoseconds": {base_ns},\n')
        if has_annotations:
            # Before traceEvents so a consumer can detect it with a header-only read.
            f.write(f'"{_INLINE_ANNOTATED_KEY}": true,\n')
        f.write('"traceEvents": [\n')

        for did, ts in sorted(seen_devices.items()):
            ts_str = _rel(ts)
            if did == host_pid or did < 0:
                label = "CPU" if did == host_pid else "Overhead"
                sort_idx = did if did >= 0 else 0x1000000
            else:
                label = f"GPU {did}"
                sort_idx = 5000000 + did

            _write_metadata_event(f, "process_name", ts_str, did, 0, "name", '"python"')
            _write_metadata_event(
                f, "process_labels", ts_str, did, 0, "labels", f'"{label}"'
            )
            _write_metadata_event(
                f,
                "process_sort_index",
                ts_str,
                did,
                0,
                "sort_index",
                str(sort_idx),
            )

        for (did, rid), ts in sorted(seen_resources.items()):
            ts_str = _rel(ts)
            if did == host_pid or did < 0:
                rname = f"thread {rid}"
            else:
                rname = f"stream {rid} "
            _write_metadata_event(
                f, "thread_name", ts_str, did, rid, "name", f'"{rname}"'
            )
            _write_metadata_event(
                f, "thread_sort_index", ts_str, did, rid, "sort_index", str(rid)
            )

        for act in activities:
            ts = act.timestamp()
            dur = act.duration()

            did = act.device_id()
            rid = _sanitize_tid(act.resource_id())
            cat = act.type()
            name = act.name()
            ts_str = _rel(ts)
            dur_str = _ns_to_us(max(dur, 0))
            out_tid = rid

            args_parts = []
            linked_corr = act.linked_correlation_id()
            if linked_corr:
                args_parts.append(f'"External id": {linked_corr}')
            elif cat not in _EXCLUDED_EXTERNAL_ID_TYPES:
                corr = act.correlation_id()
                if corr:
                    args_parts.append(f'"External id": {corr}')

            md = act.metadata_json()
            if md:
                args_parts.append(md)

            if cat == "kernel":
                linked = act.linked_activity()
                if linked is not None and linked.name() == _PARAM_COMMS_CALL_NAME:
                    linked_md = linked.metadata_json()
                    if linked_md:
                        args_parts.append(linked_md)

            if annotations and cat in _WORK_CATEGORIES and md and _graph_node_id(md):
                annotation = annotations.get(_graph_node_id(md))
                stream = lane_name = None
                if annotation:
                    ann_str, stream, lane_name = _annotation_args(annotation)
                    if ann_str:
                        args_parts.append(ann_str)
                out_tid = stream if stream is not None else default_stream
                if out_tid != rid:
                    prev = reassigned.get((did, out_tid))
                    if lane_name is None and prev is not None:
                        lane_name = prev[1]
                    reassigned[(did, out_tid)] = (ts, lane_name)
                    # args.stream has to follow the lane. The annotation supplies it
                    # when it names one; otherwise override kineto's hardware stream
                    # (last key wins). Either way the hardware stream would be lost
                    # with the tid, so keep it as original_stream.
                    if stream is None:
                        args_parts.append(f'"stream": {out_tid}')
                    args_parts.append(f'"original_stream": {rid}')

            if cat in _WORK_CATEGORIES:
                work_tids.add((did, out_tid))

            if cat == "cpu_instant_event":
                # Kineto's C++ logger writes these ("[memory]" and friends) as instant
                # events with thread scope and no duration; as complete events they
                # would render as bars running to the next event.
                event = (
                    f'{{"ph":"i","s":"t","cat":{_json_escape(cat)},'
                    f'"name":{_json_escape(name)},'
                    f'"pid":{did},"tid":{out_tid},'
                    f'"ts":{ts_str}'
                )
            else:
                event = (
                    f'{{"ph":"X","cat":{_json_escape(cat)},'
                    f'"name":{_json_escape(name)},'
                    f'"pid":{did},"tid":{out_tid},'
                    f'"ts":{ts_str},"dur":{dur_str}'
                )
            if args_parts:
                event += f',"args":{{{",".join(args_parts)}}}'
            event += "},\n"

            if annotations and cat == "gpu_user_annotation":
                deferred_annotations.append((did, out_tid, event))
                continue
            f.write(event)

            flow_id = act.flow_id()
            if flow_id > 0:
                flow_cat = _FLOW_NAMES.get(act.flow_type(), "ac2g")
                if act.flow_start():
                    f.write(
                        f'{{"ph":"s","id":{flow_id},'
                        f'"pid":{did},"tid":{out_tid},'
                        f'"ts":{ts_str},"cat":"{flow_cat}","name":"{flow_cat}"}},\n'
                    )
                else:
                    f.write(
                        f'{{"ph":"f","id":{flow_id},'
                        f'"pid":{did},"tid":{out_tid},'
                        f'"ts":{ts_str},"cat":"{flow_cat}","name":"{flow_cat}","bp":"e"}},\n'
                    )

        for a_did, a_tid, a_event in deferred_annotations:
            if (a_did, a_tid) in work_tids:
                f.write(a_event)

        for (r_did, r_tid), (r_ts, r_name) in sorted(reassigned.items()):
            if (r_did, r_tid) in seen_resources:
                continue
            r_ts_str = _rel(r_ts)
            _write_metadata_event(
                f,
                "thread_name",
                r_ts_str,
                r_did,
                r_tid,
                "name",
                _json_escape(r_name if r_name else f"stream {r_tid} "),
            )
            _write_metadata_event(
                f, "thread_sort_index", r_ts_str, r_did, r_tid, "sort_index", str(r_tid)
            )

        if activities:
            its = _rel(min_ts)
            trace_dur = _ns_to_us(max(max_end_ts - min_ts, 0))
            f.write(
                f'{{"ph":"X","cat":"Trace","name":"PyTorch Profiler (0)",'
                f'"pid":"Spans","tid":"PyTorch Profiler",'
                f'"ts":{its},"dur":{trace_dur},"args":{{"Op count": 0}}}},\n'
            )
            _write_metadata_event(
                f,
                "process_sort_index",
                its,
                '"Spans"',
                0,
                "sort_index",
                str(0x20000000),
            )
            f.write(
                f'{{"ph":"i","s":"g","name":"Iteration Start: PyTorch Profiler",'
                f'"pid":"Traces","tid":"Trace PyTorch Profiler","ts":{its}}},\n'
            )

        end_ts = _rel(max_end_ts + 1000)
        f.write(
            f'{{"ph":"i","s":"g","name":"Record Window End",'
            f'"pid":"","tid":"","ts":{end_ts}}}\n'
        )

        f.write(f'],\n"traceName": {_json_escape(path)}\n}}\n')
