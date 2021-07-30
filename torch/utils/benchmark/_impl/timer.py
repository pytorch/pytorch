import timeit
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.benchmark._impl import common
from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl.tasks import wall_time
from torch.utils.benchmark._impl.workers import in_process_worker
from torch.utils.benchmark._impl.workers import noise_police_worker
from torch.utils.benchmark._impl.workers import subprocess_worker


class Timer:
    """Helper class for measuring execution time of PyTorch statements.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    The PyTorch Timer is based on `timeit.Timer` (and in fact borrows from
    `timeit.Timer` in its implementation), but with several key differences:

    1) Runtime aware:
        Timer will perform warmups (important as some elements of PyTorch are
        lazily initialized), set threadpool size so that comparisons are
        apples-to-apples, and synchronize asynchronous CUDA functions when
        necessary.

    2) Focus on replicates:
        When measuring code, and particularly complex kernels / models,
        run-to-run variation is a significant confounding factor. It is
        expected that all measurements should include replicates to quantify
        noise and allow median computation, which is more robust than mean.
        To that effect, this class deviates from the `timeit` API by
        conceptually merging `timeit.Timer.repeat` and `timeit.Timer.autorange`.
        (Exact algorithms are discussed in method docstrings.) The `timeit`
        method is replicated for cases where an adaptive strategy is not
        desired.

    3) Optional metadata:
        When defining a Timer, one can optionally specify `label`, `sub_label`,
        `description`, and `env`. (Defined later) These fields are included in
        the representation of result object and by the `Compare` class to group
        and display results for comparison.

    4) Instruction counts
        In addition to wall times, Timer can run a statement under Callgrind
        and report instructions executed.

    Directly analogous to `timeit.Timer` constructor arguments:

        `stmt`, `setup`, `timer`, `globals`

    PyTorch Timer specific constructor arguments:

        `label`, `sub_label`, `description`, `env`, `num_threads`

    Args:
        stmt: Code snippet to be run in a loop and timed.

        setup: Optional setup code. Used to define variables used in `stmt`

        global_setup: (C++ only)
            Code which is placed at the top level of the file for things like
            `#include` statements.

        timer: (Python only)
            Callable which returns the current time.

        globals: (Python only)
            A dict which defines the global variables when `stmt` is being
            executed. This is the other method for providing variables which
            `stmt` needs.

        label:
            String which summarizes `stmt`. For instance, if `stmt` is
            "torch.nn.functional.relu(torch.add(x, 1, out=out))"
            one might set label to "ReLU(x + 1)" to improve readability.

        sub_label:
            Provide supplemental information to disambiguate measurements
            with identical stmt or label. For instance, in our example
            above sub_label might be "float" or "int", so that it is easy
            to differentiate:
            "ReLU(x + 1): (float)"

            "ReLU(x + 1): (int)"
            when printing Measurements or summarizing using `Compare`.

        description:
            String to distinguish measurements with identical label and
            sub_label. The principal use of `description` is to signal to
            `Compare` the columns of data. For instance one might set it
            based on the input size  to create a table of the form: ::

                                        | n=1 | n=4 | ...
                                        ------------- ...
                ReLU(x + 1): (float)    | ... | ... | ...
                ReLU(x + 1): (int)      | ... | ... | ...


            using `Compare`. It is also included when printing a Measurement.

        env:
            This tag indicates that otherwise identical tasks were run in
            different environments, and are therefore not equivilent, for
            instance when A/B testing a change to a kernel. `Compare` will
            treat Measurements with different `env` specification as distinct
            when merging replicate runs.

        num_threads:
            The size of the PyTorch threadpool when executing `stmt`. Single
            threaded performace is important as both a key inference workload
            and a good indicator of intrinsic algorithmic efficiency, so the
            default is set to one. This is in contrast to the default PyTorch
            threadpool size which tries to utilize all cores.
    """

    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        global_setup: str = "",
        timer: Optional[Callable[[], float]] = None,
        globals: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads: int = 1,
        language: Union[constants.Language, str] = constants.Language.PYTHON,
    ):
        self._work_spec = constants.WorkSpec(
            stmt=stmt,
            setup=setup,
            global_setup=global_setup,
            num_threads=num_threads,
            language=language,
        )

        self._metadata: constants.TaskSpec = constants.WorkMetadata(
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
        )

        if self._work_spec.language == constants.Language.CPP:
            if globals is not None:
                raise ValueError("Cannot pass `globals` for C++ snippet.")

            if timer is not None:
                raise ValueError("Cannot override `timer` for C++ snippet.")

        # We copy `globals` to prevent mutations from leaking.
        # (For instance, `eval` adds the `__builtins__` key)
        self._globals = dict(globals or {})

        # Include `torch` if not specified as a convenience feature.
        if self._work_spec.language == constants.Language.PYTHON:
            self._globals.setdefault("torch", torch)

        self._timer = timer

    def _make_timeit_task(self) -> wall_time.TimeitTask:
        return wall_time.TimeitTask(
            work_spec=self._work_spec,
            timer=self._timer,
            worker=in_process_worker.InProcessWorker(globals=self._globals)
        )

    def timeit(self, number: int = 1000000) -> float:
        return self._make_timeit_task().timeit(number=number)

    def repeat(self, repeat: int = -1, number: int = -1) -> None:
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback: Optional[wall_time.TimerCallback] = None) -> None:
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def adaptive_autorange(
            self,
            threshold: float = 0.1,
            *,
            min_run_time: float = 0.01,
            max_run_time: float = 10.0,
            callback: Optional[wall_time.TimerCallback] = None,
    ) -> common.Measurement:
        number_per_run, raw_times = self._make_timeit_task().adaptive_autorange(
            threshold=threshold,
            min_run_time=min_run_time,
            max_run_time=max_run_time,
            callback=callback,
        )
        return common.Measurement(number_per_run, raw_times)

    def blocked_autorange(
        self,
        callback: Optional[wall_time.TimerCallback] = None,
        min_run_time: float = 0.2,
    ) -> common.Measurement:
        """Measure many replicates while keeping timer overhead to a minimum.

        At a high level, blocked_autorange executes the following pseudo-code::

            `setup`

            total_time = 0
            while total_time < min_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                total_time += (timer() - start)

        Note the variable `block_size` in the inner loop. The choice of block
        size is important to measurement quality, and must balance two
        competing objectives:

            1) A small block size results in more replicates and generally
               better statistics.

            2) A large block size better amortizes the cost of `timer`
               invocation, and results in a less biased measurement. This is
               important because CUDA syncronization time is non-trivial
               (order single to low double digit microseconds) and would
               otherwise bias the measurement.

        blocked_autorange sets block_size by running a warmup period,
        increasing block size until timer overhead is less than 0.1% of
        the overall computation. This value is then used for the main
        measurement loop.

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
        number_per_run, raw_times = self._make_timeit_task().blocked_autorange(
            callback=callback,
            min_run_time=min_run_time,
        )
        return common.Measurement(number_per_run, raw_times)


class SubprocessTimer(Timer):

    # TODO: warn if globals are provided
    _timeit_worker: Optional[subprocess_worker.SubprocessWorker] = None

    def _make_timeit_worker(self) -> subprocess_worker.SubprocessWorker:
        return subprocess_worker.SubprocessWorker()

    def _make_timeit_task(self) -> wall_time.TimeitTask:
        # Process startup and teardown cost is non-trivial, so we reuse the
        # Worker for the lifetime of the timer.
        if self._timeit_worker is None:
            self._timeit_worker = self._make_timeit_worker()

        return wall_time.TimeitTask(
            work_spec=self._work_spec,
            timer=self._timer,
            worker=self._timeit_worker,
        )


class NoisePoliceTimer(SubprocessTimer):

    def _make_timeit_worker(self) -> subprocess_worker.SubprocessWorker:
        return noise_police_worker.NoisePoliceWorker()
