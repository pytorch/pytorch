import json
import pathlib
import tempfile
import time

from torch._C._distributed_c10d import _register_handler, _Request, _Response
from torch.profiler import _ExperimentalConfig, profile


def _torch_profile(req: _Request, resp: _Response) -> None:
    experimental_config = _ExperimentalConfig(
        profile_all_threads=True,
    )
    duration = float(req.get_param("duration"))
    with profile(record_shapes=True, experimental_config=experimental_config) as prof:
        time.sleep(duration)

    with tempfile.NamedTemporaryFile(prefix="torch_debug", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        resp.set_content(pathlib.Path(f.name).read_bytes(), "application/json")
        resp.set_status(200)


_register_handler("torch_profile", _torch_profile)


_MAX_TRACE_ENTRIES = 20000


def _memory_snapshot(req: _Request, resp: _Response) -> None:
    import torch

    try:
        snapshot = torch.cuda.memory._snapshot()
        events: list = []  # pyrefly: ignore
        mb = 1024 * 1024

        num_devices = len(snapshot.get("device_traces", []))
        trace_counts = [len(t) for t in snapshot.get("device_traces", [])]
        events.append(
            {
                "ph": "i",
                "s": "g",
                "ts": 0,
                "name": "snapshot_debug",
                "pid": "GPU Memory Debug",
                "tid": 0,
                "args": {
                    "num_devices": num_devices,
                    "trace_counts_per_device": trace_counts,
                    "snapshot_keys": list(snapshot.keys()),
                    "num_segments": len(snapshot.get("segments", [])),
                },
            }
        )

        for device_idx, traces in enumerate(snapshot.get("device_traces", [])):
            recent = traces[-_MAX_TRACE_ENTRIES:]
            pid = f"GPU Memory (device {device_idx})"

            # Track open allocs to pair with frees into duration events
            open_allocs: dict[int, dict] = {}
            for entry in recent:
                action = entry.get("action", "")
                ts = entry.get("time_us", 0)
                addr = entry.get("addr", 0)
                size = entry.get("size", 0)
                stream = entry.get("stream", 0)
                size_mb = round(size / mb, 4)

                if action == "alloc":
                    open_allocs[addr] = {
                        "ts": ts,
                        "size": size,
                        "stream": stream,
                    }
                elif action in ("free_requested", "free_completed"):
                    alloc = open_allocs.pop(addr, None)
                    if alloc is not None:
                        dur = max(ts - alloc["ts"], 1)
                        events.append(
                            {
                                "ph": "X",
                                "name": f"alloc {size_mb} MB",
                                "cat": "gpu_memory",
                                "ts": alloc["ts"],
                                "dur": dur,
                                "pid": pid,
                                "tid": alloc["stream"],
                                "args": {
                                    "addr": hex(addr),
                                    "size_bytes": size,
                                    "size_MB": size_mb,
                                },
                            }
                        )
                    else:
                        events.append(
                            {
                                "ph": "i",
                                "s": "t",
                                "name": f"free {size_mb} MB",
                                "cat": "gpu_memory",
                                "ts": ts,
                                "pid": pid,
                                "tid": stream,
                                "args": {"addr": hex(addr), "size_MB": size_mb},
                            }
                        )
                elif action in ("segment_alloc", "segment_free", "oom"):
                    events.append(
                        {
                            "ph": "i",
                            "s": "g",
                            "name": action,
                            "cat": "gpu_memory",
                            "ts": ts,
                            "pid": pid,
                            "tid": stream,
                            "args": {
                                "addr": hex(addr),
                                "size_MB": size_mb,
                            },
                        }
                    )

            # Still-open allocs become duration events extending to the last ts
            last_ts = recent[-1].get("time_us", 0) if recent else 0
            for addr, alloc in open_allocs.items():
                dur = max(last_ts - alloc["ts"], 1)
                alloc_mb = round(alloc["size"] / mb, 4)
                events.append(
                    {
                        "ph": "X",
                        "name": f"alloc {alloc_mb} MB (live)",
                        "cat": "gpu_memory",
                        "ts": alloc["ts"],
                        "dur": dur,
                        "pid": pid,
                        "tid": alloc["stream"],
                        "args": {
                            "addr": hex(addr),
                            "size_MB": alloc_mb,
                            "still_allocated": True,
                        },
                    }
                )

        # Add memory_stats counters and summary (same format as before)
        stats = dict(torch.cuda.memory_stats())
        events.extend(
            [
                {
                    "ph": "C",
                    "ts": 0,
                    "name": "Allocated (MB)",
                    "pid": "GPU Memory",
                    "tid": 0,
                    "args": {"value": stats.get("allocated_bytes.all.current", 0) / mb},
                },
                {
                    "ph": "C",
                    "ts": 0,
                    "name": "Reserved (MB)",
                    "pid": "GPU Memory",
                    "tid": 0,
                    "args": {"value": stats.get("reserved_bytes.all.current", 0) / mb},
                },
                {
                    "ph": "C",
                    "ts": 0,
                    "name": "Peak Allocated (MB)",
                    "pid": "GPU Memory",
                    "tid": 0,
                    "args": {"value": stats.get("allocated_bytes.all.peak", 0) / mb},
                },
                {
                    "ph": "C",
                    "ts": 0,
                    "name": "Peak Reserved (MB)",
                    "pid": "GPU Memory",
                    "tid": 0,
                    "args": {"value": stats.get("reserved_bytes.all.peak", 0) / mb},
                },
                {
                    "ph": "C",
                    "ts": 0,
                    "name": "Active Blocks (MB)",
                    "pid": "GPU Memory",
                    "tid": 0,
                    "args": {
                        "value": sum(
                            b.get("size", 0)
                            for s in torch.cuda.memory_snapshot()
                            for b in s.get("blocks", [])
                            if b.get("state") == "active_allocated"
                        )
                        / mb
                    },
                },
                {
                    "ph": "C",
                    "ts": 0,
                    "name": "Inactive Blocks (MB)",
                    "pid": "GPU Memory",
                    "tid": 0,
                    "args": {
                        "value": sum(
                            b.get("size", 0)
                            for s in torch.cuda.memory_snapshot()
                            for b in s.get("blocks", [])
                            if b.get("state") != "active_allocated"
                        )
                        / mb
                    },
                },
            ]
        )
        events.append(
            {
                "ph": "i",
                "s": "g",
                "ts": 0,
                "name": "Memory Summary",
                "pid": "GPU Memory",
                "tid": 0,
                "args": {
                    "allocated_MB": round(
                        stats.get("allocated_bytes.all.current", 0) / mb, 2
                    ),
                    "reserved_MB": round(
                        stats.get("reserved_bytes.all.current", 0) / mb, 2
                    ),
                    "peak_allocated_MB": round(
                        stats.get("allocated_bytes.all.peak", 0) / mb, 2
                    ),
                    "peak_reserved_MB": round(
                        stats.get("reserved_bytes.all.peak", 0) / mb, 2
                    ),
                    "num_alloc_retries": stats.get("num_alloc_retries", 0),
                    "num_ooms": stats.get("num_ooms", 0),
                    "summary": torch.cuda.memory_summary(),
                },
            }
        )

        resp.set_content(json.dumps(events).encode(), "application/json")
        resp.set_status(200)
    except Exception as e:
        resp.set_content(json.dumps({"error": str(e)}).encode(), "application/json")
        resp.set_status(500)


_register_handler("memory_snapshot", _memory_snapshot)
