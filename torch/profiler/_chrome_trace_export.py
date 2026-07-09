# mypy: allow-untyped-defs
"""Stream chrome trace JSON directly from ITraceActivity objects.

Bypasses Kineto's C++ ChromeTraceLogger, writing events one-at-a-time
through a (possibly gzip-compressed) text writer so we never materialize
the full JSON string in memory.
"""

from __future__ import annotations

import gzip
import json
import os
import time as _time
from typing import IO

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


def export_chrome_trace(
    kineto_results,
    path: str,
    metadata: dict[str, str] | None = None,
):
    """Export chrome trace from ITraceActivity objects, streaming to disk.

    ``kineto_results`` is a ``_ProfilerResult`` that exposes
    ``trace_activities()`` and ``trace_start_ns()``.

    Writes ``.json`` or ``.json.gz`` depending on the file extension.
    """
    activities = kineto_results.trace_activities()
    base_ns = _trimester_base_ns()

    seen_devices: dict[int, int] = {}
    seen_resources: dict[tuple[int, int], int] = {}
    host_pid = os.getpid()
    min_ts = 2**63
    max_end_ts = 0

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

    def _rel(ns: int) -> str:
        return _ns_to_us(max(ns - base_ns, 0))

    with gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w") as f:
        device_props = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            device_props.append(
                {
                    "id": i,
                    "name": p.name,
                    "totalGlobalMem": p.total_memory,
                    "computeMajor": p.major,
                    "computeMinor": p.minor,
                    "maxThreadsPerBlock": p.max_threads_per_block,  # pyrefly: ignore[missing-attribute]
                    "maxThreadsPerMultiprocessor": p.multi_processor_count,
                    "regsPerMultiprocessor": p.regs_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
                    "warpSize": p.warp_size,
                }
            )

        f.write("{\n")
        f.write('"schemaVersion": 1,\n')
        f.write(f'"deviceProperties": {json.dumps(device_props)},\n')
        if metadata:
            for k, v in metadata.items():
                f.write(f"{_json_escape(k)}: {v},\n")
        f.write('"displayTimeUnit": "ms",\n')
        f.write(f'"baseTimeNanoseconds": {base_ns},\n')
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

            f.write(
                f'{{"ph":"X","cat":{_json_escape(cat)},'
                f'"name":{_json_escape(name)},'
                f'"pid":{did},"tid":{rid},'
                f'"ts":{ts_str},"dur":{dur_str}'
            )
            if args_parts:
                f.write(f',"args":{{{",".join(args_parts)}}}')
            f.write("},\n")

            flow_id = act.flow_id()
            if flow_id > 0:
                flow_cat = _FLOW_NAMES.get(act.flow_type(), "ac2g")
                if act.flow_start():
                    f.write(
                        f'{{"ph":"s","id":{flow_id},'
                        f'"pid":{did},"tid":{rid},'
                        f'"ts":{ts_str},"cat":"{flow_cat}","name":"{flow_cat}"}},\n'
                    )
                else:
                    f.write(
                        f'{{"ph":"f","id":{flow_id},'
                        f'"pid":{did},"tid":{rid},'
                        f'"ts":{ts_str},"cat":"{flow_cat}","name":"{flow_cat}","bp":"e"}},\n'
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
