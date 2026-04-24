"""
Tracing for torchmux: produces two chrome://tracing JSON files.

Natural trace: shows actual serial execution with time slices per
worker. Includes snapshot/restore overhead spans so you can see what
time is real compute vs mux overhead.

Synthetic trace: reconstructs what a parallel run would look like.
All workers' compute phases are aligned at the start (overlapped),
and each collective is placed at max(worker_compute_durations).
Snapshot/restore spans are omitted (they don't exist in real
parallel execution).
"""

import json
import os


def export_natural(events_by_rank, path, nproc):
    """Export the natural (serial) trace.

    Each worker is a separate process in the chrome trace. Events are
    placed at their actual wall-clock times.

    Args:
        events_by_rank: dict[int, list[tuple]] where each tuple is
            (category, name, start_us, dur_us)
        path: output file path
        nproc: number of workers
    """
    trace_events = []

    for rank in range(nproc):
        trace_events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": rank,
                "tid": 0,
                "args": {"name": f"Worker {rank}"},
            }
        )

    if not events_by_rank:
        _write(path, trace_events)
        return

    all_starts = []
    for events in events_by_rank.values():
        for _, _, start, _ in events:
            all_starts.append(start)
    base_ts = min(all_starts) if all_starts else 0

    for rank, events in events_by_rank.items():
        for cat, name, start_us, dur_us in events:
            trace_events.append(
                {
                    "ph": "X",
                    "cat": cat,
                    "name": name,
                    "pid": rank,
                    "tid": 0,
                    "ts": int(start_us - base_ts),
                    "dur": int(dur_us),
                }
            )

    _write(path, trace_events)


def export_synthetic(events_by_rank, path, nproc):
    """Export the synthetic (parallel) trace.

    Reconstructs what a parallel run would look like by overlapping
    workers' compute phases between collectives.

    Algorithm:
      1. Split each worker's events into phases (by collective boundaries).
      2. In each phase, all workers' compute spans start at the same time.
      3. The collective is placed at max(worker_compute_durations).
      4. Snapshot/restore spans are omitted.
    """
    trace_events = []
    for rank in range(nproc):
        trace_events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": rank,
                "tid": 0,
                "args": {"name": f"Worker {rank}"},
            }
        )

    if not events_by_rank:
        _write(path, trace_events)
        return

    phases = _split_into_phases(events_by_rank, nproc)

    cursor = 0.0
    for phase in phases:
        max_compute = 0.0
        coll_name = None
        coll_dur = 0.0

        for rank in range(nproc):
            worker_events = phase.get(rank, [])
            compute_dur = sum(d for c, _, _, d in worker_events if c == "compute")
            max_compute = max(max_compute, compute_dur)
            for c, n, _, d in worker_events:
                if c == "collective":
                    coll_name = n
                    coll_dur = max(coll_dur, d)

        per_rank_compute_end = {}
        for rank in range(nproc):
            worker_events = phase.get(rank, [])
            t = cursor
            for cat, name, _, dur_us in worker_events:
                if cat != "compute":
                    continue
                trace_events.append(
                    {
                        "ph": "X",
                        "cat": "compute",
                        "name": name,
                        "pid": rank,
                        "tid": 0,
                        "ts": int(t),
                        "dur": int(dur_us),
                    }
                )
                t += dur_us
            per_rank_compute_end[rank] = t

        if coll_name:
            coll_end = cursor + max_compute + coll_dur
            for rank in range(nproc):
                coll_start = per_rank_compute_end.get(rank, cursor)
                trace_events.append(
                    {
                        "ph": "X",
                        "cat": "collective",
                        "name": coll_name,
                        "pid": rank,
                        "tid": 0,
                        "ts": int(coll_start),
                        "dur": int(coll_end - coll_start),
                    }
                )
            cursor = coll_end
        else:
            cursor += max_compute

    _write(path, trace_events)


def _split_into_phases(events_by_rank, nproc):
    """Split events into phases separated by collectives.

    A phase is a dict[rank, list[events]] containing the compute
    events and the ending collective for each worker between two
    consecutive collective boundaries.
    """
    per_rank_phases = {}
    for rank in range(nproc):
        events = events_by_rank.get(rank, [])
        phases = []
        current = []
        for ev in events:
            cat = ev[0]
            current.append(ev)
            if cat == "collective":
                phases.append(current)
                current = []
        if current:
            phases.append(current)
        per_rank_phases[rank] = phases

    max_phases = max((len(p) for p in per_rank_phases.values()), default=0)

    result = []
    for i in range(max_phases):
        phase = {}
        for rank in range(nproc):
            if i < len(per_rank_phases[rank]):
                phase[rank] = per_rank_phases[rank][i]
        result.append(phase)
    return result


def _write(path, trace_events):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    trace = {"traceEvents": trace_events, "displayTimeUnit": "ms"}
    with open(path, "w") as f:
        json.dump(trace, f)
