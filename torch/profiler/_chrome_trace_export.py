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

# The hardware stream in that same fragment, rewritten in place when an event is moved to
# a logical lane so its args do not carry two "stream" keys.
_STREAM_RE = re.compile(r'"stream":\s*\d+')

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
    from torch.cuda._utils import _check_cuda_bindings, _cuda_bindings_runtime

    # The shared gate already did the import, and nulls this out both when the package is
    # missing and on ROCm, where it imports but every call fails.
    if _cuda_bindings_runtime is None:
        return versions
    try:
        versions["cuda_driver_version"] = _check_cuda_bindings(
            _cuda_bindings_runtime.cudaDriverGetVersion()
        )
    except RuntimeError:
        # cuda.bindings drives its own CUDART, not the one torch initialized, so this can
        # fail where torch is fine. Losing a version key beats aborting the export.
        pass
    return versions


def _device_properties() -> list[dict[str, Any]]:
    """Per-device properties in the shape kineto's ``createDevicePropertiesJson``
    writes (it serializes ``cudaDeviceProp``).

    Every field kineto emits except ``regsPerBlock``, which torch's properties object
    does not carry; it is left out rather than guessed from ``regsPerMultiprocessor``,
    which only happens to match on current parts. ``sharedMemPerBlockOptin`` is
    NVIDIA-only in that binding, so it is omitted on ROCm rather than raising.
    """
    props: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        entry = {
            "id": i,
            "name": p.name,
            "totalGlobalMem": p.total_memory,
            "computeMajor": p.major,
            "computeMinor": p.minor,
            "maxThreadsPerBlock": p.max_threads_per_block,  # pyrefly: ignore[missing-attribute]
            "maxThreadsPerMultiprocessor": p.max_threads_per_multi_processor,  # pyrefly: ignore[missing-attribute]
            "warpSize": p.warp_size,
            "sharedMemPerBlock": p.shared_memory_per_block,  # pyrefly: ignore[missing-attribute]
            "numSms": p.multi_processor_count,
            "regsPerMultiprocessor": p.regs_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
            "sharedMemPerMultiprocessor": p.shared_memory_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
        }
        # NVIDIA-only in torch's binding, hence the guard rather than a plain read.
        if hasattr(p, "shared_memory_per_block_optin"):
            entry["sharedMemPerBlockOptin"] = (
                p.shared_memory_per_block_optin
            )  # pyrefly: ignore[missing-attribute]
        props.append(entry)
    return props


def _annotation_args(annotation) -> tuple[str | None, int | None, str | None]:
    """Render one node's annotation into an args fragment, its lane, and its lane name.

    Values are stringified, and ``stream`` is returned rather than rendered: it names the
    lane, which the caller writes into the event's own ``stream`` field. The dict is
    escaped in one ``json.dumps`` with the braces stripped rather than key by key. Values
    must be dicts -- ``mark_kernels`` normalizes a bare string to ``{"name": s}`` before it
    reaches the registry, so nothing else ever gets here; anything that is not a dict is
    skipped rather than allowed to abort the export.
    """
    stream: int | None = None
    lane_name: str | None = None
    chunks: list[str] = []
    for ann in annotation if isinstance(annotation, list) else [annotation]:
        if not isinstance(ann, dict):
            continue
        if lane_name is None and _LANE_NAME_KEY in ann:
            lane_name = str(ann[_LANE_NAME_KEY])
        norm: dict[str, Any] = {}
        for key, value in ann.items():
            if key == "stream":
                stream = int(value)
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
    graph_lanes: str = "none",
    default_stream: int = 7,
):
    """Export chrome trace from ITraceActivity objects, streaming to disk.

    ``kineto_results`` is a ``_ProfilerResult`` that exposes
    ``trace_activities()`` and ``trace_start_ns()``.

    ``annotations`` maps a CUDA-graph node id to the annotation recorded for it
    (pass ``torch.cuda.graph_annotations.get_kernel_annotations()``); matching events get
    its fields spliced into their ``args``. An empty mapping means the same as ``None``:
    events are written exactly as kineto reported them.

    ``graph_lanes`` decides whether graphed events are also moved onto display lanes.
    With ``"none"`` (default) nothing moves, so the trace keeps the stream layout kineto
    reported; a recorded ``stream`` is then reported as ``args["annotated_stream"]``
    rather than acted on. With ``"all"`` each graphed event moves to the lane its
    annotation names (what ``mark_stream`` records) and the rest onto ``default_stream``
    (7 by convention) -- worth it when a replay is smeared over the hundreds of hardware
    streams the graph executor picked, and not otherwise, since it piles everything onto
    one lane when nothing named a stream. A moved event carries its lane in ``tid`` and
    ``args["stream"]``, and the stream it actually ran on as ``args["original_stream"]``.

    ``"all"`` requires ``annotations``; on its own it would only collapse every graphed
    event onto one lane, so it raises instead.

    Writes ``.json`` or ``.json.gz`` depending on the file extension.
    """
    if graph_lanes not in ("all", "none"):
        raise ValueError(f"graph_lanes must be 'all' or 'none', got {graph_lanes!r}")
    if graph_lanes == "all" and not annotations:
        raise ValueError(
            "graph_lanes='all' needs cuda_graph_annotations: without them every graphed "
            "event would land on default_stream, which loses the stream layout and names "
            "nothing in return"
        )
    activities = kineto_results.trace_activities()
    base_ns = _trimester_base_ns()
    # No annotations to inject means no graph-lane pass at all: an empty mapping reads the
    # same as None rather than collapsing every graphed event onto one lane for nothing.
    annotate_graphs = bool(annotations)
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
        if not has_annotations and annotate_graphs and act.type() in _WORK_CATEGORIES:
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
            annotation_parts: list[str] = []

            node_id = (
                _graph_node_id(md)
                if annotate_graphs and cat in _WORK_CATEGORIES and md
                else 0
            )
            if node_id:
                annotation = annotations.get(node_id)
                stream = lane_name = None
                if annotation:
                    ann_str, stream, lane_name = _annotation_args(annotation)
                    if ann_str:
                        annotation_parts.append(ann_str)
                if graph_lanes == "all":
                    out_tid = stream if stream is not None else default_stream
                if out_tid != rid:
                    prev = reassigned.get((did, out_tid))
                    if lane_name is None and prev is not None:
                        lane_name = prev[1]
                    reassigned[(did, out_tid)] = (ts, lane_name)
                    # args.stream has to name the lane the event renders on. Rewrite the
                    # one kineto put in the metadata fragment rather than appending a
                    # second: duplicate keys resolve differently from parser to parser.
                    md, rewritten = _STREAM_RE.subn(f'"stream": {out_tid}', md, count=1)
                    if not rewritten:
                        annotation_parts.append(f'"stream": {out_tid}')
                    # The hardware stream the work ran on is otherwise unrecoverable
                    # once the tid and args.stream both name the lane.
                    annotation_parts.append(f'"original_stream": {rid}')
                elif stream is not None:
                    # Not moved, so nothing else records the lane the annotation asked
                    # for. Keep it under its own key rather than dropping it -- args.stream
                    # still has to mean the stream the event is on.
                    annotation_parts.append(f'"annotated_stream": {stream}')

            if md:
                args_parts.append(md)

            if cat == "kernel":
                linked = act.linked_activity()
                if linked is not None and linked.name() == _PARAM_COMMS_CALL_NAME:
                    linked_md = linked.metadata_json()
                    if linked_md:
                        args_parts.append(linked_md)

            args_parts.extend(annotation_parts)

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

            if annotate_graphs and cat == "gpu_user_annotation":
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
