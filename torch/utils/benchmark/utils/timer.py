"""Timer class based on the timeit.Timer class, but torch aware."""

import timeit
import textwrap
from typing import Any, Callable, Dict, List, NoReturn, Optional

import numpy as np
import torch
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface


__all__ = ["Timer", "timer"]


if torch.has_cuda and torch.cuda.is_available():
    def timer() -> float:
        torch.cuda.synchronize()
        return timeit.default_timer()
else:
    timer = timeit.default_timer


class Timer(object):
    _timer_cls = timeit.Timer

    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        timer: Callable[[], float] = timer,
        globals: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads: int = 1,
    ):
        if not isinstance(stmt, str):
            raise ValueError("Currently only a `str` stmt is supported.")

        # We copy `globals` to prevent mutations from leaking, (for instance,
        # `eval` adds the `__builtins__` key) and include `torch` if not
        # specified as a convenience feature.
        globals = dict(globals or {})
        globals.setdefault("torch", torch)
        self._globals = globals

        # Convenience adjustment so that multi-line code snippets defined in
        # functions do not IndentationError inside timeit.Timer. The leading
        # newline removal is for the initial newline that appears when defining
        # block strings. For instance:
        #   textwrap.dedent("""
        #     print("This is a stmt")
        #   """)
        # produces '\nprint("This is a stmt")\n'.
        #
        # Stripping this down to 'print("This is a stmt")' doesn't change
        # what gets executed, but it makes __repr__'s nicer.
        stmt = textwrap.dedent(stmt)
        stmt = (stmt[1:] if stmt[0] == "\n" else stmt).rstrip()
        setup = textwrap.dedent(setup)
        setup = (setup[1:] if setup[0] == "\n" else setup).rstrip()

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

    def timeit(self, number: int = 1000000) -> common.Measurement:
        with common.set_torch_threads(self._task_spec.num_threads):
            # Warmup
            self._timer.timeit(number=max(int(number // 100), 1))

            return common.Measurement(
                number_per_run=number,
                raw_times=[self._timer.timeit(number=number)],
                task_spec=self._task_spec
            )

    def repeat(self, repeat: int = -1, number: int = -1) -> None:
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback: Optional[Callable[[int, float], NoReturn]] = None) -> None:
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def _threaded_measurement_loop(
        self,
        number: int,
        time_hook: Callable[[], float],
        stop_hook: Callable[[List[float]], bool],
        min_run_time: float,
        max_run_time: Optional[float] = None,
        callback: Optional[Callable[[int, float], NoReturn]] = None
    ) -> List[float]:
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

    def _estimate_block_size(self, min_run_time: float) -> int:
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
            threshold: float = 0.1,
            *,
            min_run_time: float = 0.01,
            max_run_time: float = 10.0,
            callback: Optional[Callable[[int, float], NoReturn]] = None,
    ) -> common.Measurement:
        number = self._estimate_block_size(min_run_time=0.05)

        def time_hook() -> float:
            return self._timer.timeit(number)

        def stop_hook(times: List[float]) -> bool:
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

    def blocked_autorange(
        self,
        callback: Optional[Callable[[int, float], NoReturn]] = None,
        min_run_time: float = 0.2,
    ) -> common.Measurement:
        number = self._estimate_block_size(min_run_time)

        def time_hook() -> float:
            return self._timer.timeit(number)

        def stop_hook(times: List[float]) -> bool:
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

    def collect_callgrind(
        self,
        number: int = 100,
        collect_baseline: bool = True
    ) -> valgrind_timer_interface.CallgrindStats:
        if not isinstance(self._task_spec.stmt, str):
            raise ValueError("`collect_callgrind` currently only supports string `stmt`")

        # Check that the statement is valid. It doesn't guarantee success, but it's much
        # simpler and quicker to raise an exception for a faulty `stmt` or `setup` in
        # the parent process rather than the valgrind subprocess.
        self._timer.timeit(1)
        return valgrind_timer_interface.wrapper_singleton().collect_callgrind(
            task_spec=self._task_spec,
            globals=self._globals,
            number=number,
            collect_baseline=collect_baseline)
