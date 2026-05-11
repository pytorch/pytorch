"""Coordinator: spawns persistent workers, dispatches files, drains results.

Uses raw multiprocessing with a spawn context (matches test/run_test.py).
We avoid `concurrent.futures.ProcessPoolExecutor` because it marks the entire
pool as broken on a single worker SIGSEGV, which is fatal for our use case
where any of 1000+ test files might have an import-time crash.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty
from typing import Callable, Optional

from tools.zippo.worker import worker_main


# A single collected test: (pytest node id, tuple of marker names).
TestEntry = tuple[str, tuple[str, ...]]


# Per-file collection timeout. Pytest collection should normally complete in
# seconds even for heavy modules; if a single file blows past this, the worker
# is presumed hung (e.g. distributed init at module import, infinite loop in
# collection-time code) and is force-killed. 2 minutes is enough headroom for
# the heaviest files (test/test_ops.py expands ~68k parametrize entries) on
# slower CI runners while still catching genuine hangs in reasonable time.
DEFAULT_TASK_TIMEOUT_S = 120.0


class Coordinator:
    def __init__(
        self,
        num_workers: int,
        max_tasks_per_worker: int,
        progress_cb: Optional[Callable[[int, int, int, int], None]] = None,
        failure_cb: Optional[Callable[[str, str], None]] = None,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
    ) -> None:
        self.num_workers = num_workers
        self.max_tasks_per_worker = max_tasks_per_worker
        self.progress_cb = progress_cb
        self.failure_cb = failure_cb
        self.task_timeout_s = task_timeout_s

        self._ctx = mp.get_context("spawn")
        self._task_q: mp.Queue = self._ctx.Queue()
        self._result_q: mp.Queue = self._ctx.Queue()

        self._workers: dict[int, mp.Process] = {}
        self._in_flight: dict[int, str] = {}
        self._in_flight_started: dict[int, float] = {}

        self.results: dict[str, list[TestEntry]] = {}
        self.failures: list[tuple[str, str]] = []

    def _spawn_worker(self) -> None:
        proc = self._ctx.Process(
            target=worker_main,
            args=(
                self._task_q,
                self._result_q,
                self.max_tasks_per_worker,
            ),
            daemon=True,
        )
        proc.start()
        assert proc.pid is not None
        self._workers[proc.pid] = proc

    def _sweep_workers(self, work_remaining: bool) -> int:
        """Reap dead workers and force-kill hung ones.

        Returns the count of in-flight tasks that were closed out (either by
        worker death or by timeout) so the caller can advance `processed`.
        """
        closed_out = 0

        dead_pids = [pid for pid, p in self._workers.items() if not p.is_alive()]
        for pid in dead_pids:
            proc = self._workers.pop(pid)
            ec = proc.exitcode
            proc.join(timeout=1)
            path = self._in_flight.pop(pid, None)
            self._in_flight_started.pop(pid, None)
            if path is not None:
                msg = f"worker died (exitcode {ec})"
                self.failures.append((path, msg))
                if self.failure_cb is not None:
                    self.failure_cb(path, msg)
                closed_out += 1
            if work_remaining:
                self._spawn_worker()

        now = time.monotonic()
        hung_pids = [
            pid
            for pid, started in self._in_flight_started.items()
            if pid in self._workers and now - started > self.task_timeout_s
        ]
        for pid in hung_pids:
            proc = self._workers.pop(pid)
            path = self._in_flight.pop(pid, None)
            self._in_flight_started.pop(pid, None)
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            if path is not None:
                msg = f"timed out after {int(self.task_timeout_s)}s"
                self.failures.append((path, msg))
                if self.failure_cb is not None:
                    self.failure_cb(path, msg)
                closed_out += 1
            if work_remaining:
                self._spawn_worker()

        return closed_out

    def run(self, paths: list[str]) -> None:
        total = len(paths)
        if total == 0:
            return

        for _ in range(self.num_workers):
            self._spawn_worker()

        # Lazy feed: keep at most 2*N paths queued at any time. Avoids OS
        # pipe-buffer blocking for large path lists and keeps the dead-worker
        # sweep on the same loop as result handling.
        backlog = max(2 * self.num_workers, self.num_workers + 1)
        paths_iter = iter(paths)
        queued = 0
        processed = 0

        for _ in range(min(backlog, total)):
            self._task_q.put(next(paths_iter))
            queued += 1

        while processed < queued:
            try:
                msg = self._result_q.get(timeout=1.0)
            except Empty:
                processed += self._sweep_workers(work_remaining=processed < total)
                if self.progress_cb is not None:
                    self.progress_cb(
                        processed,
                        len(self.failures),
                        total,
                        len(self._in_flight),
                    )
                continue

            kind = msg[0]
            if kind == "start":
                _, pid, path = msg
                self._in_flight[pid] = path
                self._in_flight_started[pid] = time.monotonic()
            elif kind == "ok":
                _, pid, path, entries = msg
                self._in_flight.pop(pid, None)
                self._in_flight_started.pop(pid, None)
                self.results[path] = entries
                processed += 1
            elif kind == "err":
                _, pid, path, errmsg = msg
                self._in_flight.pop(pid, None)
                self._in_flight_started.pop(pid, None)
                self.failures.append((path, errmsg))
                if self.failure_cb is not None:
                    self.failure_cb(path, errmsg)
                processed += 1

            if self.progress_cb is not None and kind in ("ok", "err"):
                self.progress_cb(
                    processed, len(self.failures), total, len(self._in_flight)
                )

            if queued < total:
                try:
                    self._task_q.put(next(paths_iter))
                    queued += 1
                except StopIteration:
                    pass

        for _ in range(len(self._workers)):
            self._task_q.put(None)
        for proc in list(self._workers.values()):
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

        self._task_q.close()
        self._result_q.close()
        self._task_q.join_thread()
        self._result_q.join_thread()
