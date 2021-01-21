"""Orchestrates benchmark collection across many cores."""
import statistics
import subprocess
import textwrap
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from execution.cores import CorePool, CPU_COUNT, SLACK
from execution.work import PYTHON_CMD, InProgress, WorkOrder
from worker.main import MIN_RUN_TIME, WorkerFailure, WorkerOutput

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


class WorkerFailed(Exception):
    """Raised in the main process when a worker failure is detected."""
    def __init__(self, wrapped_trace: Optional[str] = None) -> None:
        self.wrapped_trace: Optional[str] = wrapped_trace
        super().__init__()


class Runner:
    _work_items: Tuple[WorkOrder, ...]
    _core_pool: CorePool
    _display_progress: bool
    _start_time: Optional[float]
    _work_queue: List[WorkOrder]
    _active_jobs: List[InProgress]
    _currently_processed: Optional[WorkOrder]
    _results: Dict[WorkOrder, WorkerOutput]
    _durations: Dict[WorkOrder, float]

    def __init__(
        self,
        work_items: Tuple[WorkOrder, ...],
        core_pool: Optional[CorePool] = None,
        display_progress: bool = True,
    ) -> None:
        self._work_items = work_items
        self._core_pool = core_pool or CorePool()
        self._display_progress = display_progress
        self._start_time = None
        self._work_queue = list(work_items)
        self._active_jobs = []
        self._currently_processed = None
        self._results = {}
        self._durations = {}

        if len(work_items) != len(set(work_items)):
            raise ValueError('Duplicate work items.')

    def run(self) -> Dict[WorkOrder, WorkerOutput]:
        try:
            return self._run()

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt (ctrl-c) detected. Shutting down children.")
            self._force_shutdown()
            raise

        except subprocess.TimeoutExpired:
            print("\n\nJob timed out. Shutting down children.")
            self._force_shutdown(verbose=True)
            raise

        except WorkerFailed as e:
            print('Shutting down all outstanding jobs before re-raising.')
            self._force_shutdown(verbose=True)
            if e.wrapped_trace:
                print(e.wrapped_trace)
            else:
                print('Unknown failure. (Worker did not report exception contents.)')
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
            time.sleep(max(1.0 - (time.time() - t0), 0.0))
        print(f"\nTotal time: {time.time() - self._start_time:.0f} seconds")
        return self._results.copy()

    def _update_active_jobs(self) -> None:
        active_jobs: List[InProgress] = []
        for job in self._active_jobs:
            self._currently_processed = job.work_order
            if not job.ready:
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
                raise WorkerFailed(wrapped_trace=result.failure_trace)
        self._currently_processed = None
        self._active_jobs.clear()
        self._active_jobs.extend(active_jobs)

    def _enqueue_new_jobs(self) -> None:
        work_queue: List[WorkOrder] = []
        for i, work_order in enumerate(self._work_queue):
            self._currently_processed = work_order

            cpu_list: Optional[str] = None
            if i < 20:
                # We want to prevent the loop from greedily scheduling all
                # single core jobs and leaving multi core jobs until the end,
                # so we only attempt to schedule up to 20 jobs.
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

    def group_by_language(self, items: Iterable[WorkOrder]) -> Dict[Language, float]:
        grouped: Dict[Language, List[WorkOrder]] = {}
        for w in items:
            grouped.setdefault(w.timer_args.language, [])
            grouped[w.timer_args.language].append(w)

        return {
            k: statistics.mean((self._durations[w] for w in v))
            for k, v in grouped.items()
        }

    @staticmethod
    def _naive_cost(w: WorkOrder) -> float:
        """Simple heuristic for guesstimating run time.

        This is used to estimate ETA before any jobs have finished.
        """
        return (
            MIN_RUN_TIME +

            # Callgrind takes about just under minute.
            50.0 +

            # C++ compilation takes about 20 seconds, and there are two
            # of them. (One for wall time and one for callgrind.)
            (2 * 20.0 if w.timer_args.language == Language.CPP else 0.0)
        )

    def _print_progress(self) -> None:
        if not self._display_progress:
            return

        approximate_estimate: bool = False
        time_estimates = self.group_by_language(self._results.keys())
        cpu_time_estimates: List[Tuple[int, float]] = []

        for w in [job.work_order for job in self._active_jobs] + list(self._work_queue):
            if w.timer_args.language in time_estimates:
                time_estimate = time_estimates[w.timer_args.language]
            else:
                approximate_estimate = True
                time_estimate = self._naive_cost(w)
            cpu_time_estimates.append((w.timer_args.num_threads, time_estimate))

        # Factor in elapsed time for active jobs.
        for i, job in enumerate(self._active_jobs):
            cpu_time_estimates[i] = (
                cpu_time_estimates[i][0],
                max(cpu_time_estimates[i][1] - job.duration, 0.0))

        # Assume 95% of ideal core utilization, which tends to be a stable heuristic.
        overall_remaining = sum(c * t for c, t in cpu_time_estimates) / (CPU_COUNT - SLACK) / 0.95

        # If the time remaining is < 10 minutes, switch to a more precise
        # bin-packing scheme which will better predict straggler effects.
        # This isn't a particularly efficient algorithm and it's not EXACTLY
        # what CorePool does, but it's good enough for an estimate. (And it's
        # not on the hot path.)
        if overall_remaining < 600:
            core_times = [0.0 for _ in range(CPU_COUNT - SLACK)]
            for num_cores, time_estimate in cpu_time_estimates:
                for i in range(num_cores):
                    core_times[i] = core_times[i + num_cores - 1] + time_estimate
                core_times.sort()
            overall_remaining = max(core_times)

        if not overall_remaining:
            eta_str = "ETA: Soon"
        else:
            eta_str = (
                f"ETA{' (approximate)' if approximate_estimate else ''}: "
                f"{overall_remaining:.0f} seconds")

        core_seconds_used = (
            sum((self._durations[w] * w.timer_args.num_threads) for w in self._results.keys()) +
            sum(job.duration for job in self._active_jobs))

        assert self._start_time is not None
        elapsed = time.time() - self._start_time
        packing_efficiency = core_seconds_used / (elapsed * (CPU_COUNT - SLACK))

        print(
            f"\r{len(self._results)} / {len(self._work_items)} "
            f"{eta_str}, Job packing efficiency: {packing_efficiency * 100:.1f}%".ljust(80),
            end="",
        )

    def _force_shutdown(self, verbose: bool = False) -> None:
        """Try to interrupt jobs, and kill if need be.

        We would prefer to softly terminate jobs so that they have a chance to
        clean up before shutting down.
        """
        for job in self._active_jobs:
            job.proc.interrupt()

        if verbose and self._currently_processed is not None:
            print(textwrap.dedent(f"""
                Failed when processing the following Job:
                  Label:      {self._currently_processed.label}
                  AutoLabels: {self._currently_processed.auto_labels}
                  Source cmd: {self._currently_processed.source_cmd}
            """).strip() + "\n")

        if self._active_jobs:
            time.sleep(0.5)

        remaining_jobs = [j for j in self._active_jobs if j.proc.poll() is None]
        if remaining_jobs:
            print(
                f'SIGINT sent to {len(self._active_jobs)} jobs, '
                f'{len(remaining_jobs)} have not yet exited.\n'
                'Entering short cleanup loop, after which stragglers will '
                'be forcibly terminated.'
            )

            for _ in range(5):
                time.sleep(1.0)
                remaining_jobs = [j for j in remaining_jobs if j.proc.poll() is None]
                if remaining_jobs:
                    print(f'{len(remaining_jobs)} still remain.')
                else:
                    print('All remaining jobs have gracefully terminated.')
                    return

            print(f'{len(remaining_jobs)} jobs refused to exit. Forcibly terminating.')
            for j in remaining_jobs:
                j.proc.terminate()

    def _canary_import(self) -> None:
        """Make sure we can import torch before launching a slew of workers."""
        source_cmds: Set[str] = set()
        for w in self._work_items:
            if w.source_cmd is not None:
                source_cmds.add(f"{w.source_cmd} && ")

        for source_cmd in (source_cmds or {""}):
            cmd = f'{source_cmd}{PYTHON_CMD} -c "import torch"'
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                executable="/bin/bash",
            )

            if proc.returncode:
                raise ImportError(
                    f'Failed to import torch in subprocess: {cmd}\n{proc.stdout}')
