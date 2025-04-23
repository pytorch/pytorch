"""Timer class based on the timeit.Timer class, but torch aware."""
import enum
import timeit
import textwrap
from typing import overload, Any, Callable, NoReturn, Optional, Union

import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface


__all__ = ["Timer", "timer", "Language"]


if torch.backends.cuda.is_built() and torch.cuda.is_available():  # type: ignore[no-untyped-call]
    def timer() -> float:
        torch.cuda.synchronize()
        return timeit.default_timer()
elif torch.xpu.is_available():
    def timer() -> float:
        torch.xpu.synchronize()
        return timeit.default_timer()
elif torch._C._get_privateuse1_backend_name() != "privateuseone":
    privateuse1_device_handler = getattr(torch, torch._C._get_privateuse1_backend_name(), None) \
        if torch._C._get_privateuse1_backend_name() != "cpu" else None

    def timer() -> float:
        if privateuse1_device_handler:
            privateuse1_device_handler.synchronize()
        return timeit.default_timer()
else:
    timer = timeit.default_timer


class Language(enum.Enum):
    PYTHON = 0
    CPP = 1


class CPPTimer:
    def __init__(
        self,
        stmt: str,
        setup: str,
        global_setup: str,
        timer: Callable[[], float],
        globals: dict[str, Any],
    ) -> None:
        if timer is not timeit.default_timer:
            raise NotImplementedError(
                "PyTorch was built with CUDA and a GPU is present; however "
                "Timer does not yet support GPU measurements. If your "
                "code is CPU only, pass `timer=timeit.default_timer` to the "
                "Timer's constructor to indicate this. (Note that this will "
                "produce incorrect results if the GPU is in fact used, as "
                "Timer will not synchronize CUDA.)"
            )

        if globals:
            raise ValueError("C++ timing does not support globals.")

        self._stmt: str = textwrap.dedent(stmt)
        self._setup: str = textwrap.dedent(setup)
        self._global_setup: str = textwrap.dedent(global_setup)
        self._timeit_module: Optional[TimeitModuleType] = None

    def timeit(self, number: int) -> float:
        if self._timeit_module is None:
            self._timeit_module = cpp_jit.compile_timeit_template(
                stmt=self._stmt,
                setup=self._setup,
                global_setup=self._global_setup,
            )

        return self._timeit_module.timeit(number)


class Timer:
    """Helper class for measuring execution time of PyTorch statements.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    The PyTorch Timer is based on `timeit.Timer` (and in fact uses
    `timeit.Timer` internally), but with several key differences:

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

        timer:
            Callable which returns the current time. If PyTorch was built
            without CUDA or there is no GPU present, this defaults to
            `timeit.default_timer`; otherwise it will synchronize CUDA before
            measuring the time.

        globals:
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
            different environments, and are therefore not equivalent, for
            instance when A/B testing a change to a kernel. `Compare` will
            treat Measurements with different `env` specification as distinct
            when merging replicate runs.

        num_threads:
            The size of the PyTorch threadpool when executing `stmt`. Single
            threaded performance is important as both a key inference workload
            and a good indicator of intrinsic algorithmic efficiency, so the
            default is set to one. This is in contrast to the default PyTorch
            threadpool size which tries to utilize all cores.
    """

    _timer_cls: type[TimerClass] = timeit.Timer

    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        global_setup: str = "",
        timer: Callable[[], float] = timer,
        globals: Optional[dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads: int = 1,
        language: Union[Language, str] = Language.PYTHON,
    ):
        if not isinstance(stmt, str):
            raise ValueError("Currently only a `str` stmt is supported.")

        # We copy `globals` to prevent mutations from leaking.
        # (For instance, `eval` adds the `__builtins__` key)
        self._globals = dict(globals or {})

        timer_kwargs = {}
        if language in (Language.PYTHON, "py", "python"):
            # Include `torch` if not specified as a convenience feature.
            self._globals.setdefault("torch", torch)
            self._language: Language = Language.PYTHON
            if global_setup:
                raise ValueError(
                    f"global_setup is C++ only, got `{global_setup}`. Most "
                    "likely this code can simply be moved to `setup`."
                )

        elif language in (Language.CPP, "cpp", "c++"):
            assert self._timer_cls is timeit.Timer, "_timer_cls has already been swapped."
            self._timer_cls = CPPTimer
            setup = ("" if setup == "pass" else setup)
            self._language = Language.CPP
            timer_kwargs["global_setup"] = global_setup

        else:
            raise ValueError(f"Invalid language `{language}`.")

        # Convenience adjustment so that multi-line code snippets defined in
        # functions do not IndentationError (Python) or look odd (C++). The
        # leading newline removal is for the initial newline that appears when
        # defining block strings. For instance:
        #   textwrap.dedent("""
        #     print("This is a stmt")
        #   """)
        # produces '\nprint("This is a stmt")\n'.
        #
        # Stripping this down to 'print("This is a stmt")' doesn't change
        # what gets executed, but it makes __repr__'s nicer.
        stmt = textwrap.dedent(stmt)
        stmt = (stmt[1:] if stmt and stmt[0] == "\n" else stmt).rstrip()
        setup = textwrap.dedent(setup)
        setup = (setup[1:] if setup and setup[0] == "\n" else setup).rstrip()

        self._timer = self._timer_cls(
            stmt=stmt,
            setup=setup,
            timer=timer,
            globals=valgrind_timer_interface.CopyIfCallgrind.unwrap_all(self._globals),
            **timer_kwargs,
        )
        self._task_spec = common.TaskSpec(
            stmt=stmt,
            setup=setup,
            global_setup=global_setup,
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
            num_threads=num_threads,
        )

    def _timeit(self, number: int) -> float:
        # Even calling a timer in C++ takes ~50 ns, so no real operation should
        # take less than 1 ns. (And this prevents divide by zero errors.)
        return max(self._timer.timeit(number), 1e-9)

    def timeit(self, number: int = 1000000) -> common.Measurement:
        """Mirrors the semantics of timeit.Timer.timeit().

        Execute the main statement (`stmt`) `number` times.
        https://docs.python.org/3/library/timeit.html#timeit.Timer.timeit
        """
        with common.set_torch_threads(self._task_spec.num_threads):
            # Warmup
            self._timeit(number=max(int(number // 100), 2))

            return common.Measurement(
                number_per_run=number,
                raw_times=[self._timeit(number=number)],
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
        stop_hook: Callable[[list[float]], bool],
        min_run_time: float,
        max_run_time: Optional[float] = None,
        callback: Optional[Callable[[int, float], NoReturn]] = None
    ) -> list[float]:
        total_time = 0.0
        can_stop = False
        times: list[float] = []
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
            overhead = torch.tensor([self._timeit(0) for _ in range(5)]).median().item()
            number = 1
            while True:
                time_taken = self._timeit(number)
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

    def blocked_autorange(
        self,
        callback: Optional[Callable[[int, float], NoReturn]] = None,
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
               important because CUDA synchronization time is non-trivial
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
        number = self._estimate_block_size(min_run_time)

        def time_hook() -> float:
            return self._timeit(number)

        def stop_hook(times: list[float]) -> bool:
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

    def adaptive_autorange(
            self,
            threshold: float = 0.1,
            *,
            min_run_time: float = 0.01,
            max_run_time: float = 10.0,
            callback: Optional[Callable[[int, float], NoReturn]] = None,
    ) -> common.Measurement:
        """Similar to `blocked_autorange` but also checks for variablility in measurements
        and repeats until iqr/median is smaller than `threshold` or `max_run_time` is reached.


        At a high level, adaptive_autorange executes the following pseudo-code::

            `setup`

            times = []
            while times.sum < max_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                times.append(timer() - start)

                enough_data = len(times)>3 and times.sum > min_run_time
                small_iqr=times.iqr/times.mean<threshold

                if enough_data and small_iqr:
                    break

        Args:
            threshold: value of iqr/median threshold for stopping

            min_run_time: total runtime needed before checking `threshold`

            max_run_time: total runtime  for all measurements regardless of `threshold`

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
        number = self._estimate_block_size(min_run_time=0.05)

        def time_hook() -> float:
            return self._timeit(number)

        def stop_hook(times: list[float]) -> bool:
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

    @overload
    def collect_callgrind(
        self,
        number: int,
        *,
        repeats: None,
        collect_baseline: bool,
        retain_out_file: bool,
    ) -> valgrind_timer_interface.CallgrindStats:
        ...

    @overload
    def collect_callgrind(
        self,
        number: int,
        *,
        repeats: int,
        collect_baseline: bool,
        retain_out_file: bool,
    ) -> tuple[valgrind_timer_interface.CallgrindStats, ...]:
        ...

    def collect_callgrind(
        self,
        number: int = 100,
        *,
        repeats: Optional[int] = None,
        collect_baseline: bool = True,
        retain_out_file: bool = False,
    ) -> Any:
        """Collect instruction counts using Callgrind.

        Unlike wall times, instruction counts are deterministic
        (modulo non-determinism in the program itself and small amounts of
        jitter from the Python interpreter.) This makes them ideal for detailed
        performance analysis. This method runs `stmt` in a separate process
        so that Valgrind can instrument the program. Performance is severely
        degraded due to the instrumentation, however this is ameliorated by
        the fact that a small number of iterations is generally sufficient to
        obtain good measurements.

        In order to to use this method `valgrind`, `callgrind_control`, and
        `callgrind_annotate` must be installed.

        Because there is a process boundary between the caller (this process)
        and the `stmt` execution, `globals` cannot contain arbitrary in-memory
        data structures. (Unlike timing methods) Instead, globals are
        restricted to builtins, `nn.Modules`'s, and TorchScripted functions/modules
        to reduce the surprise factor from serialization and subsequent
        deserialization. The `GlobalsBridge` class provides more detail on this
        subject. Take particular care with nn.Modules: they rely on pickle and
        you may need to add an import to `setup` for them to transfer properly.

        By default, a profile for an empty statement will be collected and
        cached to indicate how many instructions are from the Python loop which
        drives `stmt`.

        Returns:
            A `CallgrindStats` object which provides instruction counts and
            some basic facilities for analyzing and manipulating results.
        """
        if not isinstance(self._task_spec.stmt, str):
            raise ValueError("`collect_callgrind` currently only supports string `stmt`")

        if repeats is not None and repeats < 1:
            raise ValueError("If specified, `repeats` must be >= 1")

        # Check that the statement is valid. It doesn't guarantee success, but it's much
        # simpler and quicker to raise an exception for a faulty `stmt` or `setup` in
        # the parent process rather than the valgrind subprocess.
        self._timeit(1)
        is_python = (self._language == Language.PYTHON)
        assert is_python or not self._globals
        result = valgrind_timer_interface.wrapper_singleton().collect_callgrind(
            task_spec=self._task_spec,
            globals=self._globals,
            number=number,
            repeats=repeats or 1,
            collect_baseline=collect_baseline and is_python,
            is_python=is_python,
            retain_out_file=retain_out_file,
        )

        return (result[0] if repeats is None else result)
