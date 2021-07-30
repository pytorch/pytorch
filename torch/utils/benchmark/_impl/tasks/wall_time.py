import statistics
import textwrap
import timeit
import typing

import torch
from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl.tasks import base as task_base
from torch.utils.benchmark._impl.templates import jit as jit_template
from torch.utils.benchmark._impl.workers import base as worker_base
from torch.utils.benchmark._impl.workers import in_process_worker


TimerCallback = typing.Callable[[int, float], typing.NoReturn]


class CommonStatistics:

    sorted_x: typing.Tuple[float]
    median: float
    mean: float
    p25: float
    p75: float

    def __init__(self, x: typing.List[float]):
        self.sorted_x = tuple(sorted(x))
        _sorted_x = torch.tensor(self.sorted_x, dtype=torch.float64)
        self.median = _sorted_x.quantile(.5).item()
        self.mean = _sorted_x.mean().item()
        self.p25 = _sorted_x.quantile(.25).item()
        self.p75 = _sorted_x.quantile(.75).item()

    @property
    def iqr(self):
        return self.p75 - self.p25


class TimeitTask(task_base.TaskBase):

    _worker: worker_base.WorkerBase
    _should_cuda_sync: typing.Optional[bool]

    def __init__(
        self,
        work_spec: constants.WorkSpec,
        timer: typing.Optional[typing.Callable[[],float]] = None,
        worker: typing.Optional[worker_base.WorkerBase] = None,
    ) -> None:
        self._work_spec = work_spec
        self._worker = worker or in_process_worker.InProcessWorker({})
        self.worker.run(jit_template.generate(work_spec=self._work_spec))

        # See `measure` for more details.
        self._should_cuda_sync = None

        # `timeit.Timer` allows users to override the timer used, so we have
        # to support that functionality.
        self._custom_timer: bool = False
        if self._work_spec.language == constants.Language.CPP:
            assert timer is None

        elif timer not in (None, timeit.default_timer):
            self.worker.store("_timeit_task_timer", timer, in_memory=True)
            self._custom_timer = True

    @property
    def worker(self) -> worker_base.WorkerBase:
        return self._worker

    def timeit(self, number: int) -> float:
        result, self._should_cuda_sync = self.measure(
            n_iter=number,
            num_threads=self._work_spec.num_threads,
            custom_timer=self._custom_timer,
            should_cuda_sync=self._should_cuda_sync
        )
        return result

    @task_base.run_in_worker(scoped=True)
    def measure(
        n_iter: int,
        num_threads: int,
        custom_timer: bool,
        should_cuda_sync: typing.Optional[bool],
    ) -> typing.Tuple[float, bool]:
        from torch.utils.benchmark._impl import runtime_utils
        from torch.utils.benchmark._impl.templates import jit as jit_template

        with runtime_utils.set_torch_threads(num_threads):
            # The first time `measure` is called, we must determine if it is
            # necessary to syncronize CUDA. In some cases this can be done
            # statically: if PyTorch is not built with CUDA or a GPU is not
            # present, we know a priori that we don't need to sync.
            #
            # Alternatively, if we cannot make such a static determination we
            # run a few times using the Kineto profier, and check if any CUDA
            # events are observed.
            #
            # We pass `should_cuda_sync` back to the caller so that TimeitTask
            # can remember if it needs to sync, and subsequent calls can skip
            # the overhead of checking.
            if should_cuda_sync is None:
                should_cuda_sync = runtime_utils.ShouldCudaSynchronize.cuda_present()

            if should_cuda_sync is None:
                with runtime_utils.ShouldCudaSynchronize() as watch_cuda:
                    jit_template.get().call(n_iter=3)
                should_cuda_sync = watch_cuda.cuda_detected

            if not isinstance(should_cuda_sync, bool):
                raise ValueError(
                    "`should_cuda_sync` should have been narrowed to a bool, "
                    f"instead got `{should_cuda_sync}`. ({type(should_cuda_sync)})")

            # This is placed in the global namespace during Task init.
            kwargs = {"timer": globals()["_timeit_task_timer"]} if custom_timer else {}

            return jit_template.get().measure_wall_time(
                n_iter=n_iter,
                n_warmup_iter=max(int(n_iter // 100), 2),
                cuda_sync=should_cuda_sync,
                **kwargs,
            ), should_cuda_sync

    def _estimate_block_size(self, min_run_time: float) -> int:
        # Estimate the block size needed for measurement to be negligible
        # compared to the inner loop.
        overhead = statistics.median([self.timeit(0) for _ in range(5)])
        number = 1
        while True:
            time_taken = self.timeit(number)
            relative_overhead = overhead / time_taken
            if relative_overhead <= 1e-4 and time_taken >= min_run_time / 1000:
                break
            if time_taken > min_run_time:
                break
            # Avoid overflow in C++ pybind11 interface
            if number * 10 > 2147483647:
                break
            number *= 10
        return number

    def _threaded_measurement_loop(
        self,
        number: int,
        time_hook: typing.Callable[[], float],
        stop_hook: typing.Callable[[typing.List[float]], bool],
        min_run_time: float,
        max_run_time: typing.Optional[float] = None,
        callback: typing.Optional[TimerCallback] = None,
    ) -> typing.List[float]:
        total_time = 0.0
        can_stop = False
        times: typing.List[float] = []
        while (total_time < min_run_time) or (not can_stop):
            time_spent = time_hook()
            times.append(time_spent)
            total_time += time_spent
            if callback:
                callback(number, time_spent)
            can_stop = stop_hook(times)
            if max_run_time and total_time > max_run_time:
                break
        return times

    def adaptive_autorange(
            self,
            threshold: float = 0.1,
            *,
            min_run_time: float = 0.01,
            max_run_time: float = 10.0,
            callback: typing.Optional[TimerCallback] = None,
    ) -> typing.Tuple[int, typing.List[float]]:
        number = self._estimate_block_size(min_run_time=0.05)

        def time_hook() -> float:
            return self.timeit(number)

        def stop_hook(times: typing.List[float]) -> bool:
            if len(times) > 3:
                s = CommonStatistics(times)
                return s.iqr / s.median < threshold
            return False

        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook, min_run_time, max_run_time, callback=callback)

        return number, times

    def blocked_autorange(
        self,
        callback: typing.Optional[TimerCallback] = None,
        min_run_time: float = 0.2,
    ) -> typing.Tuple[int, typing.List[float]]:
        number = self._estimate_block_size(min_run_time)

        def time_hook() -> float:
            return self.timeit(number)

        def stop_hook(times: typing.List[float]) -> bool:
            return True

        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook,
            min_run_time=min_run_time,
            callback=callback)

        return number, times
