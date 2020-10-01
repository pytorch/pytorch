"""Timer class based on the timeit.Timer class, but torch aware."""

import timeit
from typing import Callable, List, NoReturn, Optional

import numpy as np
import torch
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface


__all__ = ["Timer", "timer"]


if torch.has_cuda and torch.cuda.is_available():
    def timer():
        torch.cuda.synchronize()
        return timeit.default_timer()
else:
    timer = timeit.default_timer


class Timer(object):
    _timer_cls = timeit.Timer

    def __init__(
        self,
        stmt="pass",
        setup="pass",
        timer=timer,
        globals: Optional[dict] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads=1,
    ):
        if not isinstance(stmt, str):
            raise ValueError("Currently only a `str` stmt is supported.")

        # We copy `globals` to prevent mutations from leaking, (for instance,
        # `eval` adds the `__builtins__` key) and include `torch` if not
        # specified as a convenience feature.
        globals = dict(globals or {})
        globals.setdefault("torch", torch)
        self._globals = globals

        self._timer = self._timer_cls(stmt=stmt, setup=setup, timer=timer, globals=globals)
        self._task_spec = common.TaskSpec(
            stmt=stmt,
            setup=setup,
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
            num_threads=num_threads,
        )

    def timeit(self, number=1000000):
        with common.set_torch_threads(self._task_spec.num_threads):
            # Warmup
            self._timer.timeit(number=max(int(number // 100), 1))

            return common.Measurement(
                number_per_run=number,
                raw_times=[self._timer.timeit(number=number)],
                task_spec=self._task_spec
            )

    def repeat(self, repeat=-1, number=-1):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback=None):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def _threaded_measurement_loop(
        self,
        number: int,
        time_hook: Callable[[], float],
        stop_hook: Callable[[List[float]], bool],
        min_run_time: float,
        max_run_time: Optional[float] = None,
        callback: Optional[Callable[[int, float], NoReturn]] = None
    ):
        total_time = 0.0
        can_stop = False
        times: List[float] = []
        with common.set_torch_threads(self._task_spec.num_threads):
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

    def _estimate_block_size(self, min_run_time: float):
        with common.set_torch_threads(self._task_spec.num_threads):
            # Estimate the block size needed for measurement to be negligible
            # compared to the inner loop. This also serves as a warmup.
            overhead = np.median([self._timer.timeit(0) for _ in range(5)])
            number = 1
            while True:
                time_taken = self._timer.timeit(number)
                relative_overhead = overhead / time_taken
                if relative_overhead <= 1e-4 and time_taken >= min_run_time / 1000:
                    break
                if time_taken > min_run_time:
                    break
                number *= 10
        return number

    def adaptive_autorange(
            self,
            threshold=0.1,
            max_run_time=10,
            callback: Optional[Callable[[int, float], NoReturn]] = None,
            min_run_time=0.01
    ):
        number = self._estimate_block_size(min_run_time=0.05)

        def time_hook() -> float:
            return self._timer.timeit(number)

        def stop_hook(times) -> bool:
            if len(times) > 3:
                return common.Measurement(
                    number_per_run=number,
                    raw_times=times,
                    task_spec=self._task_spec
                ).meets_confidence(threshold=threshold)
            return False
        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook, min_run_time, max_run_time, callback=callback)

        return common.Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec
        )

    def blocked_autorange(self, callback=None, min_run_time=0.2):
        number = self._estimate_block_size(min_run_time)

        def time_hook() -> float:
            return self._timer.timeit(number)

        def stop_hook(times) -> bool:
            return True

        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook,
            min_run_time=min_run_time,
            callback=callback)

        return common.Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec
        )

    def collect_callgrind(self, number=100, collect_baseline=True):
        if not isinstance(self._task_spec.stmt, str):
            raise ValueError("`collect_callgrind` currently only supports string `stmt`")

        # __init__ adds torch, and Timer adds __builtins__
        allowed_keys = {"torch", "__builtins__"}
        if any(k not in allowed_keys for k in self._globals.keys()):
            raise ValueError(
                "`collect_callgrind` does not currently support passing globals. "
                "Please define a `setup` str instead.")

        if self._globals.get("torch", torch) is not torch:
            raise ValueError("`collect_callgrind` does not support mocking out `torch`.")

        # Check that the statement is valid. It doesn't guarantee success, but it's much
        # simpler and quicker to raise an exception for a faulty `stmt` or `setup` in
        # the parent process rather than the valgrind subprocess.
        self._timer.timeit(1)
        return valgrind_timer_interface.wrapper_singleton().collect_callgrind(
            stmt=self._task_spec.stmt,
            setup=self._task_spec.setup,
            number=number,
            num_threads=self._task_spec.num_threads,
            collect_baseline=collect_baseline)
