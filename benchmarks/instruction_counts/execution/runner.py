"""Run benchmarks while handling parallelism, isolation, and fault tolerance."""
import math
import multiprocessing
import subprocess
import textwrap
import threading
import time
from typing import Dict, List, Optional, Set, Tuple, Union

from worker.main import WorkerFailure, WorkerOutput

from execution.work import InProgress, PYTHON_CMD, SHELL, WorkOrder


CPU_COUNT: int = multiprocessing.cpu_count()


class WorkerFailed(Exception):
    """Raised in the main process when a worker failure is detected."""

    def __init__(self, cmd: str, wrapped_trace: Optional[str] = None) -> None:
        self.cmd: str = cmd
        self.wrapped_trace: Optional[str] = wrapped_trace
        super().__init__()


class CorePool:
    """Allocator style helper class to assign individual tasks to a core range.

    Pinning tasks to separate cores (or core ranges if `num_threads` > 1)
    serves two purposes. First, it prevents the machine from being overloaded,
    which can result in OOMs or Callgrind crashes. Second, it helps reduce
    noise in the wall times, which are collected as a secondary metric. For
    multi-threaded workloads, adjacency is important. Often pairs of cores
    share silicon (e.g. cache), while far away cores may lie on separate NUMA
    nodes. For this reason, CorePool will only allocate contiguous core ranges.
    This falls short of full architecture awareness, and instead tries to find
    a balance between rigor and engineering complexity.
    """

    def __init__(self, min_core_id: int, max_core_id: int) -> None:
        assert min_core_id >= 0
        assert max_core_id >= min_core_id
        assert max_core_id < CPU_COUNT

        self._min_core_id: int = min_core_id
        self._max_core_id: int = max_core_id
        self._num_cores = max_core_id - min_core_id + 1
        print(f"Core pool created: cores {self._min_core_id}-{self._max_core_id}")

        self._available: List[bool] = [
            True for _ in range(min_core_id, min_core_id + self._num_cores)
        ]

        self._reservations: Dict[str, Tuple[int, ...]] = {}
        self._lock = threading.Lock()

    def reserve(self, n: int) -> Optional[str]:
        """Simple first-fit policy.

        If successful, return a string for `taskset`. Otherwise, return None.
        """
        with self._lock:
            for lower_index in range(self._num_cores - n + 1):
                indices = tuple(range(lower_index, lower_index + n))
                if all(self._available[i] for i in indices):
                    for i in indices:
                        self._available[i] = False

                    lower_core = indices[0] + self._min_core_id
                    upper_core = indices[-1] + self._min_core_id
                    key = f"{lower_core}-{upper_core}" if n > 1 else f"{lower_core}"
                    self._reservations[key] = indices
                    return key
        return None

    def release(self, key: str) -> None:
        with self._lock:
            for i in self._reservations[key]:
                self._available[i] = True
            self._reservations.pop(key)


class Runner:
    def __init__(
        self,
        work_items: Tuple[WorkOrder, ...],
        core_pool: Optional[CorePool] = None,
        cadence: float = 1.0,
    ) -> None:
        self._work_items: Tuple[WorkOrder, ...] = work_items
        self._core_pool: CorePool = core_pool or CorePool(0, CPU_COUNT - 4)
        self._cadence: float = cadence

        # Working state.
        self._work_queue: List[WorkOrder] = list(work_items)
        self._active_jobs: List[InProgress] = []
        self._results: Dict[WorkOrder, WorkerOutput] = {}

        # Debug information for ETA and error messages.
        self._start_time: float = -1
        self._durations: Dict[WorkOrder, float] = {}
        self._currently_processed: Optional[WorkOrder] = None

        if len(work_items) != len(set(work_items)):
            raise ValueError("Duplicate work items.")

    def run(self) -> Dict[WorkOrder, WorkerOutput]:
        try:
            return self._run()

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt (ctrl-c) detected. Shutting down children.")
            self._force_shutdown(verbose=False)
            raise

        except subprocess.TimeoutExpired:
            print("\n\nJob timed out. Shutting down children.")
            self._force_shutdown(verbose=True)
            raise

        except WorkerFailed as e:
            print("Shutting down all outstanding jobs before re-raising.")
            self._force_shutdown(verbose=True)
            print(f"Cmd: {e.cmd}")
            if e.wrapped_trace:
                print(e.wrapped_trace)
            else:
                print("Unknown failure. (Worker did not report exception contents.)")
            raise

        except BaseException:
            print("\n\nUnknown exception. Shutting down jobs before re-raising.")
            self._force_shutdown(verbose=True)
            raise

    def _run(self) -> Dict[WorkOrder, WorkerOutput]:
        self._start_time = time.time()
        self._canary_import()
        while self._work_queue or self._active_jobs:
            t0 = time.time()
            self._update_active_jobs()
            self._enqueue_new_jobs()
            self._print_progress()
            time.sleep(max(self._cadence - (time.time() - t0), 0.0))
        print(f"\nTotal time: {time.time() - self._start_time:.0f} seconds")
        return self._results.copy()

    def _update_active_jobs(self) -> None:
        active_jobs: List[InProgress] = []
        for job in self._active_jobs:
            self._currently_processed = job.work_order
            if not job.check_finished():
                active_jobs.append(job)
                continue

            result: Union[WorkerOutput, WorkerFailure] = job.result
            if isinstance(result, WorkerOutput):
                self._results[job.work_order] = result
                assert job.cpu_list is not None
                self._core_pool.release(job.cpu_list)
                self._durations[job.work_order] = job.duration

            else:
                assert isinstance(result, WorkerFailure)
                raise WorkerFailed(cmd=job.proc.cmd, wrapped_trace=result.failure_trace)
        self._currently_processed = None
        self._active_jobs.clear()
        self._active_jobs.extend(active_jobs)

    def _enqueue_new_jobs(self) -> None:
        work_queue: List[WorkOrder] = []
        for i, work_order in enumerate(self._work_queue):
            self._currently_processed = work_order
            cpu_list = self._core_pool.reserve(work_order.timer_args.num_threads)

            if cpu_list is None:
                work_queue.append(work_order)
            else:
                self._active_jobs.append(InProgress(work_order, cpu_list))

                # Stagger creation. This helps with contention.
                time.sleep(0.5)
        self._currently_processed = None
        self._work_queue.clear()
        self._work_queue.extend(work_queue)

    def _print_progress(self) -> None:
        fraction = f"{len(self._results)} / {len(self._work_items)}"
        elapsed = f"{time.time() - self._start_time:.0f} seconds"
        if len(self._results) < 5:
            eta = "Unknown"
        else:
            remaining = len(self._work_items) - len(self._results)
            iters_remaining = math.ceil(remaining / self._core_pool._num_cores)
            mean_time = sum(self._durations.values()) / len(self._durations)
            eta_minutes = math.ceil(iters_remaining * mean_time / 60)
            eta = f"~{eta_minutes:.0f} minute{'s' if eta_minutes > 1 else ''}"
        print(f"\r{fraction} ({elapsed}), ETA: {eta}", end="")

    def _force_shutdown(self, verbose: bool = False) -> None:
        """Try to interrupt jobs, and kill if need be.
        We would prefer to softly terminate jobs so that they have a chance to
        clean up before shutting down.
        """
        for job in self._active_jobs:
            job.proc.interrupt()

        if verbose and self._currently_processed is not None:
            print(
                textwrap.dedent(
                    f"""
                Failed when processing the following Job:
                  Label:      {self._currently_processed.label}
                  AutoLabels: {self._currently_processed.autolabels}
                  Source cmd: {self._currently_processed.source_cmd}
            """
                ).strip()
                + "\n"
            )

        if self._active_jobs:
            time.sleep(0.5)

        remaining_jobs = [j for j in self._active_jobs if j.proc.poll() is None]
        if remaining_jobs:
            print(
                f"SIGINT sent to {len(self._active_jobs)} jobs, "
                f"{len(remaining_jobs)} have not yet exited.\n"
                "Entering short cleanup loop, after which stragglers will "
                "be forcibly terminated."
            )

            for _ in range(5):
                time.sleep(2.0)
                remaining_jobs = [j for j in remaining_jobs if j.proc.poll() is None]
                if remaining_jobs:
                    print(f"{len(remaining_jobs)} still remain.")
                else:
                    print("All remaining jobs have gracefully terminated.")
                    return

            print(f"{len(remaining_jobs)} jobs refused to exit. Forcibly terminating.")
            for j in remaining_jobs:
                j.proc.terminate()

    def _canary_import(self) -> None:
        """Make sure we can import torch before launching a slew of workers."""
        source_cmds: Set[str] = set()
        for w in self._work_items:
            if w.source_cmd is not None:
                source_cmds.add(f"{w.source_cmd} && ")

        for source_cmd in source_cmds or {""}:
            cmd = f'{source_cmd}{PYTHON_CMD} -c "import torch"'
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                executable=SHELL,
            )

            if proc.returncode:
                raise ImportError(
                    f"Failed to import torch in subprocess: {cmd}\n{proc.stdout}"
                )
