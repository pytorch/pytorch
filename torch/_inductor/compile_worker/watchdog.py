"""Shared-memory heartbeat so the compile-worker sidecar's watchdog can report
which phase a slow/stuck worker is in (see SubprocMain._watchdog_loop).

A single multiprocessing.Array is allocated in the sidecar (create()) before it
forks its worker pool; every forked worker inherits the same buffer and writes
only its own slot -- single writer per slot, so no lock is needed there and a
rare torn read is harmless for a coarse once-a-minute diagnostic. The sidecar's
watchdog thread reads all slots.

Slots are recycled off the pool's lifecycle rather than by polling worker
liveness: the sidecar calls reset() every time it (re)builds the worker pool,
and workers then claim slots 0..nprocs-1 with a per-generation counter. A worker
death tears down and rebuilds the whole pool (ProcessPoolExecutor's behavior), so
reset() runs before the replacement generation forks and there is never more than
one live worker per slot. (If the pool were ever configured to replace workers
individually -- e.g. max_tasks_per_child -- a replacement would simply run out of
counter and report nothing, rather than collide with a live worker's slot.)

Everything is a no-op unless the sidecar allocated the buffer and the worker
claimed a slot, so the report_phase() calls sprinkled through the compile path
are free in the main process and in non-forking pools (spawn workers do not
inherit the buffer and simply degrade to duration-only reporting).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import time
from enum import IntEnum
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


class Phase(IntEnum):
    # Persisted as ints in the shared buffer; append new phases, don't renumber.
    RUNNING = 0
    QUERYING_CACHE = 1
    COMPILING = 2


# Per-slot layout: _FIELDS longs.
_FIELDS = 7
_F_JOB_ID = 0
_F_PHASE = 1
_F_PHASE_START_NS = 2
_F_PID = 3
_F_JOB_START_NS = 4
_F_MEMORY_LIMIT = 5
_F_TIMEOUT = 6
_EMPTY = -1  # job_id sentinel for an idle slot

# Allocated in the sidecar (create) before forking; inherited by fork workers.
# _counter's lock also serializes slot claims and reset().
_heartbeat: Any | None = None
_counter: Any | None = None
_nprocs: int = 0
# Claimed post-fork by each worker (init_worker_slot).
_slot: int | None = None


def create(nprocs: int) -> None:
    """Sidecar: allocate the shared buffer before forking the worker pool."""
    global _heartbeat, _counter, _nprocs
    ctx = multiprocessing.get_context("fork")
    _nprocs = nprocs
    _counter = ctx.Value("i", 0)
    buf = ctx.Array("q", nprocs * _FIELDS, lock=False)
    for s in range(nprocs):
        buf[s * _FIELDS + _F_JOB_ID] = _EMPTY
    _heartbeat = buf


def reset() -> None:
    """Sidecar: recycle all slots for a fresh worker generation. Call before
    (re)building the worker pool so the new workers claim clean slots 0..n-1."""
    if _heartbeat is None or _counter is None:
        return
    with _counter.get_lock():
        _counter.value = 0
        for s in range(_nprocs):
            base = s * _FIELDS
            _heartbeat[base + _F_JOB_ID] = _EMPTY
            _heartbeat[base + _F_PID] = 0


def init_worker_slot() -> None:
    """Worker: claim this generation's next slot. No-op if the sidecar didn't
    allocate a buffer (main process / spawn pools)."""
    global _slot
    if _heartbeat is None or _counter is None or _nprocs <= 0:
        return
    with _counter.get_lock():
        slot = _counter.value
        _counter.value = slot + 1
    # Only the first nprocs workers of a generation get a slot; any beyond that
    # (which shouldn't happen without per-worker replacement) report nothing.
    if slot < _nprocs:
        _slot = slot
        _heartbeat[slot * _FIELDS + _F_PID] = os.getpid()


def set_current_job(job_id: int, memory_limit: int, timeout: int) -> None:
    """Worker: mark the start of a job in this worker's slot."""
    if _heartbeat is None or _slot is None:
        return
    base = _slot * _FIELDS
    _heartbeat[base + _F_PHASE] = int(Phase.RUNNING)
    _heartbeat[base + _F_PHASE_START_NS] = time.monotonic_ns()
    _heartbeat[base + _F_JOB_START_NS] = time.monotonic_ns()
    _heartbeat[base + _F_MEMORY_LIMIT] = memory_limit
    _heartbeat[base + _F_TIMEOUT] = timeout
    # Written last so a concurrent reader sees a fully-populated slot or none.
    _heartbeat[base + _F_JOB_ID] = job_id


def report_phase(phase: Phase) -> None:
    """Worker: record the phase this worker just entered."""
    if _heartbeat is None or _slot is None:
        return
    base = _slot * _FIELDS
    _heartbeat[base + _F_PHASE_START_NS] = time.monotonic_ns()
    _heartbeat[base + _F_PHASE] = int(phase)


def clear_current_job() -> None:
    """Worker: mark this slot idle when its job finishes."""
    if _heartbeat is None or _slot is None:
        return
    _heartbeat[_slot * _FIELDS + _F_JOB_ID] = _EMPTY


def enabled() -> bool:
    """Sidecar: whether phase heartbeats are active (fork pools only)."""
    return _heartbeat is not None


def read_heartbeats() -> dict[int, tuple[int, int, int, int, int, int]]:
    """Sidecar: job_id -> (phase, phase_start_ns, pid, job_start_ns, memory_limit, timeout) for currently-busy slots.
    phase_start_ns is worker monotonic_ns; job_start_ns is job start time; on Linux CLOCK_MONOTONIC is
    system-wide so the sidecar can diff it against its own monotonic_ns.
    """
    if _heartbeat is None:
        return {}
    out: dict[int, tuple[int, int, int, int, int, int]] = {}
    for s in range(_nprocs):
        base = s * _FIELDS
        job_id = _heartbeat[base + _F_JOB_ID]
        if job_id == _EMPTY:
            continue
        fields = (
            _heartbeat[base + _F_PHASE],
            _heartbeat[base + _F_PHASE_START_NS],
            _heartbeat[base + _F_PID],
            _heartbeat[base + _F_JOB_START_NS],
            _heartbeat[base + _F_MEMORY_LIMIT],
            _heartbeat[base + _F_TIMEOUT],
        )
        if _heartbeat[base + _F_JOB_ID] == job_id:
            out[job_id] = fields
    return out


_PAGE_SIZE_KB = os.sysconf("SC_PAGE_SIZE") // 1024 if hasattr(os, "sysconf") else 4


# Lazy one-shot probe for CONFIG_PROC_CHILDREN; None = not yet probed.
_has_proc_children: bool | None = None


def _probe_proc_children() -> bool:
    global _has_proc_children
    if _has_proc_children is not None:
        return _has_proc_children
    if sys.platform != "linux":
        _has_proc_children = False
        return False
    pid = os.getpid()
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
        if tasks:
            with open(f"/proc/{pid}/task/{tasks[0]}/children") as f:
                f.read()
        _has_proc_children = True
    except OSError:
        _has_proc_children = False
        log.error(
            "CONFIG_PROC_CHILDREN not available on this kernel; "
            "compile-worker memory limits only see the worker process "
            "itself, not its child compilers (nvcc, ptxas, cc1plus, ...)"
        )
    return _has_proc_children


def _children_via_proc_children(pid: int) -> list[int]:
    """Get child PIDs via /proc/<pid>/task/*/children (CONFIG_PROC_CHILDREN)."""
    result: list[int] = []
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return result
    for task in tasks:
        try:
            with open(f"/proc/{pid}/task/{task}/children") as f:
                children = f.read().split()
        except (OSError, ValueError):
            continue
        for child in children:
            try:
                result.append(int(child))
            except ValueError:
                continue
    return result


def _children_via_ppid(pid: int) -> list[int]:
    """Get child PIDs by scanning /proc/*/stat for matching PPID.

    Fallback for kernels without CONFIG_PROC_CHILDREN.  O(all processes) per
    call -- _subtree_memory_kb recurses, so a depth-N tree does N full scans.
    ~30us per /proc/*/stat read; ~90ms for 1000 processes at depth 3.
    Acceptable at the watchdog's 1-60s interval.

    A single-pass ppid->children map would avoid repeated scans, but needs
    invalidation as processes spawn/die between ticks.
    """
    result: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return result
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
            # Field 4 (1-indexed) is the PPID; find it after the comm field
            # which is enclosed in parentheses and may contain spaces.
            close_paren = stat.rfind(")")
            if close_paren < 0:
                continue
            fields = stat[close_paren + 2 :].split()
            ppid = int(fields[1])
            if ppid == pid:
                result.append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    return result


def _subtree_memory_kb(
    pid: int, visited: OrderedSet[int], read_fn: Callable[[int], int]
) -> int:
    if pid in visited:
        return 0
    visited.add(pid)
    total = read_fn(pid)
    get_children = (
        _children_via_proc_children if _probe_proc_children() else _children_via_ppid
    )
    for child_pid in get_children(pid):
        total += _subtree_memory_kb(child_pid, visited, read_fn)
    return total


def _read_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE_KB
    except (OSError, IndexError, ValueError):
        return 0


def _read_pss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if "Pss:" in line:
                    return int(line.split()[1])
    except (OSError, IndexError, ValueError):
        pass
    return 0


def rss_kb(pid: int, visited: OrderedSet[int]) -> int:
    return _subtree_memory_kb(pid, visited, _read_rss_kb)


def pss_kb(pid: int, visited: OrderedSet[int]) -> int:
    return _subtree_memory_kb(pid, visited, _read_pss_kb)


def _subtree_pss_kb_for_test() -> int:
    # Test helper: this process's subtree PSS (self + children) in kB.
    # Zero-arg so SubprocPool can pickle it into a worker; getpid() must run
    # in the worker, not the parent.
    return pss_kb(os.getpid(), OrderedSet())


def is_worker_memory_limit_exceeded(pid: int, memory_limit: int) -> bool:
    """Sidecar: whether the worker's subtree memory exceeds its limit."""
    visited: OrderedSet[int] = OrderedSet()
    rss = rss_kb(pid, visited)
    if rss < memory_limit:
        return False
    visited = OrderedSet()
    pss = pss_kb(pid, visited)
    return pss >= memory_limit
