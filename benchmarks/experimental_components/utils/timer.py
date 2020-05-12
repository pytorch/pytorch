"""Timer class based on the timeit.Timer class, but torch aware."""

import logging
import sys
import timeit
from typing import Callable, List, Optional

import numpy as np
import torch
import utils.common as common


__all__ = ["Timer"]


if torch.has_cuda:
    def timer():
        torch.cuda.synchronize()
        return timeit.default_timer()
else:
    timer = timeit.default_timer


class Timer(object):
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
        self._stmt = stmt
        self._setup = setup
        self._timer = timer
        self._globals = globals

        self._label = label
        self._sub_label = sub_label
        self._description = description
        self._env = env
        self._num_threads = num_threads
        self._timer = timeit.Timer(stmt=stmt, setup=setup, timer=timer, globals=globals)

    def _construct_measurement(self, number_per_run: int, times: List[float]):
        return common.Measurement(
            number_per_run=number_per_run,
            times=times,
            num_threads=self._num_threads,
            label=self._label,
            sub_label=self._sub_label,
            description=self._description,
            env=self._env,
            stmt=self._stmt,
        )

    def timeit(self, number=1000000):
        # Warmup
        self._timer.timeit(number=max(int(number // 100), 1))
        with common.set_torch_threads(self._num_threads):
            return self._construct_measurement(
                number_per_run=number, times=[self._timer.timeit(number=number)]
            )

    def repeat(self, repeat=-1, number=-1):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback=None):
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def blocked_autorange(self, callback=None, min_run_time=0.2):
        with common.set_torch_threads(self._num_threads):
            # Estimate the block size needed for measurement to be negligible
            # compared to the inner loop. This also serves as a warmup.
            overhead = np.median([self._timer.timeit(0) for _ in range(5)])
            number = 1
            while True:
                time_taken = self._timer.timeit(number)
                relative_overhead = overhead / time_taken
                if overhead <= 1e-5 and time_taken >= min_run_time / 1000:
                    break
                number *= 10

            total_time = 0
            times = []

            while total_time < min_run_time:
                time_taken = self._timer.timeit(number)
                total_time += time_taken
                if callback:
                    callback(number, time_taken)
                times.append(time_taken)

            return self._construct_measurement(number_per_run=number, times=times)
