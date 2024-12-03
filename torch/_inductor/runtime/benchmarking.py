import inspect
import time
from functools import cached_property, wraps
from itertools import chain
from statistics import median
from typing import Any, Callable, Dict, List, Tuple
from typing_extensions import Concatenate, ParamSpec, Self, TypeVar

import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.config import benchmarking as benchmarking_config, is_fbcode


logger = torch._logging.getArtifactLogger(__name__, "benchmarking")


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def maybe_time(
    fn: Callable[Concatenate[Any, P], T]
) -> Callable[Concatenate[Any, P], T]:
    """Wrapper that logs the duration of `fn`, in milliseconds, along with a representation
    of the function's args and kwargs, if logging is enabled. It is expected that `fn` is
    a method of `Benchmarker` or one of its subclasses; typing limitations prevent us from
    declaring this directly. If logging is disabled, this becomes a no-op.
    """

    # no-op if benchmarking-specific logging is disabled
    if not torch._logging._internal.log_state.is_artifact_enabled("benchmarking"):
        return fn

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        start_t = time.perf_counter()
        result = fn(*args, **kwargs)
        logger.debug(
            "Call `benchmarking.%s.%s(*args=%r, **kwargs=%r)` took %f milliseconds.",
            self.__class__.__name__,
            fn.__name__,
            args,
            kwargs,
            (time.perf_counter() - start_t) * MILLISECONDS_PER_SECOND,
        )
        return result

    return wrapper


def count(fn: Callable[Concatenate[Any, P], T]) -> Callable[Concatenate[Any, P], T]:
    """Wrapper that increments relevant dynamo counters on `fn` call. It is expected that
    `fn` is a method of `Benchmarker` or one of its subclass; typing limitations prevent
    us from declaring this directly. The counter incrementation follows the formula,

    `counters["inductor"]["benchmarking.Foo.bar] += 1`

    where `Foo` is the class whose' instance called the function, and `bar` is the function name.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        counters["inductor"][
            "benchmarking." + self.__class__.__name__ + "." + fn.__name__
        ] += 1
        return fn(self, *args, **kwargs)

    return wrapper


class Benchmarker:
    def __init__(self: Self) -> None:
        pass

    @maybe_time
    @count
    def benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any, ...],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> float:
        """Benchmark `fn(*fn_args, *fn_kwargs)` and return the runtime, in milliseconds (the
        actual runtime calculation is dictated by the benchmarking implementation, but may be
        one of [mean, median, minimum, etc.]). Functions as a convenience wrapper around
        device-specific implementations, like `benchmark_cpu` and `benchmark_gpu`. Raises
        `ValueError(...)` if we can't safely infer the device type of `fn`; for example,
        if multiple device types are found in `fn_args` and `fn_kwargs`, or if no device
        types are found.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - **kwargs: The benchmarking implementation's kwargs.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
        with dynamo_timed("Benchmarker.benchmark", log_pt2_compile_event=True):
            inferred_device = None
            for arg_or_kwarg in chain(fn_args, fn_kwargs.values()):
                if not isinstance(arg_or_kwarg, torch.Tensor):
                    continue
                if inferred_device is None:
                    inferred_device = arg_or_kwarg.device
                elif arg_or_kwarg.device != inferred_device:
                    raise ValueError(
                        "Can't safely infer the device type of `fn` with multiple device types in `fn_args` and `fn_kwargs`!"
                    )
            if inferred_device is None:
                raise ValueError(
                    "Can't safely infer the device type of `fn` with no device types in `fn_args` or `fn_kwargs`! You should be calling `.benchmark_cpu` or `.benchmark_gpu` directly."  # noqa: B950
                )
            _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
            if inferred_device == torch.device("cpu"):
                return self.benchmark_cpu(_callable, **kwargs)
            # TODO(nmacchioni): For non-CPU functions we default to using the GPU-specific benchmarking
            # implementation which was written specifically with CUDA devices in mind, we may want to
            # explore alternate implementations for other device types.
            return self.benchmark_gpu(_callable, **kwargs)

    @maybe_time
    @count
    def benchmark_cpu(
        self: Self, _callable: Callable[[], Any], warmup: int = 20, rep: int = 100
    ) -> float:
        """Benchmark the CPU callable, `_callable`, and return the median runtime,
        in milliseconds.

        Arguments:
        - _callable: The CPU callable to benchmark.

        Keyword Arguments:
        - warmup: Optionally, the duration, in milliseconds, to run `_callable`
        before benchmarking starts.
        - rep: Optionally, the duration, in milliseconds, to run `_callable`
        during benchmarking.

        Returns:
        - The median runtime of `_callable`, in milliseconds.
        """

        def run_for(ms: int) -> List[float]:
            timings = []
            run_start_t = time.perf_counter()
            while True:
                start_t = time.perf_counter()
                _callable()
                end_t = time.perf_counter()
                timings.append((end_t - start_t) * MILLISECONDS_PER_SECOND)
                if ((end_t - run_start_t) * MILLISECONDS_PER_SECOND) > ms:
                    break
            return timings

        run_for(warmup)
        return median(run_for(rep))

    @maybe_time
    @count
    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


class TritonBenchmarker(Benchmarker):
    @cached_property
    @maybe_time
    @count
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @maybe_time
    @count
    def benchmark_gpu(self: Self, _callable: Callable[[], Any], **kwargs: Any) -> float:
        """Benchmark the GPU callable, `_callable`, and return the runtime, in milliseconds.

        Arguments:
        - _callable: The GPU callable to benchmark.

        Keyword Arguments:
        - quantiles: Optionally, a tuple of floats denoting the requested quantiles.
        - return_mode: Optionally, the requested return mode. Currently, Triton's
        `do_bench` supports min, max, mean, and median return modes.
        - **kwargs: Additional kwargs passed to Triton's `do_bench`.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        """

        # this method may be used as a fallback if certain benchmarking features are disabled,
        # in which case those requests may contain kwargs that are not specific to Triton's
        # `do_bench_gpu`. as such, we may need to prune out kwargs that do not exists in
        # `do_bench_gpu`'s signature.
        do_bench_params = inspect.signature(self.triton_do_bench).parameters
        for kwarg in list(kwargs.keys()):
            if kwarg not in do_bench_params:
                del kwargs[kwarg]

        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        elif "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


def is_feature_enabled(feature_name: str) -> bool:
    """Method to decide whether or not a feature is enabled. For more context, see the
    benchmarking configuration section in `torch._inductor.config.benchmarking`.
    """
    feature_config = getattr(benchmarking_config, feature_name)
    if feature_config.env_val is not None:
        if feature_config.env_val == "1":
            return True
        elif feature_config.env_val == "0":
            return False
    if not is_fbcode():
        return feature_config.oss_default
    if feature_config.local_version is not None:
        return (
            feature_config.local_version
            >= torch._utils_internal.justknobs_getval_int(
                f"pytorch/benchmarking:{feature_name.upper()}_VERSION"
            )
        )
    return False


def maybe_fallback(
    fn: Callable[Concatenate[Any, P], T]
) -> Callable[Concatenate[Any, P], T]:
    """Wrapper that controls feature fallbacks. It is expected that `fn` is a
    method of one of `Benchmarker`'s subclasses with a corresponding `feature_name`
    attribute; typing limitations prevent us from declaring this directly. When
    `fn` is called, in the form `fn(self, ...)`, we will check that the feature
    `self.feature_name` is enabled; if the feature `feature_name` is not enabled,
    we will fallback to the parent class' implementation of `fn`.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        if not is_feature_enabled(self.feature_name):
            fallback_fn = getattr(super(self.__class__, self), fn.__name__)
            counters["inductor"]["benchmarking." + self.feature_name + ".disabled"] += 1
            logger.debug(
                "Feature `%s` is disabled, `benchmarking.%s.%s` will fallback to `benchmarking.%s.%s`.",
                self.feature_name,
                self.__class__.__name__,
                fn.__name__,
                self.__class__.__base__.__name__,
                fallback_fn.__name__,
            )
            return fallback_fn(*args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


class InductorBenchmarker(TritonBenchmarker):
    feature_name = "inductor_benchmarker"

    @cached_property
    def L2_cache_size(self: Self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.L2_cache_size

    def get_event_pairs(
        self: Self, iters: int
    ) -> List[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Get `iters` pairs of CUDA events."""
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_event_pairs_min_timing(
        self: Self, event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]
    ) -> float:
        """Get the minimum timing, in milliseconds, for a group of CUDA event pairs."""
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    @maybe_fallback
    @maybe_time
    @count
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration: int = 25,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using a custom benchmarking implementation.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - estimation_iters: Optionally, the number of iterations to run `_callable`
        during runtime estimation.
        - memory_warmup_iters: Optionally, the number of iterations to flush the L2
        cache before starting benchmarking.
        - benchmark_iters: Optionally, the number of iterations to run `_callable`
        during the benchmarking.
        - max_benchmark_duration: Optionally, the maximum duration of the benchmarking,
        in milliseconds. An estimated duration is calculated based on the values
        of `memory_warmup_iters` and `benchmark_iters`, along with the estimated
        runtime of `_callable` and various other factors, and we then shrink
        `benchmark_iters` to fit in the alloted maximum duration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - The minimum runtime of `_callable`, in milliseconds.
        """
        # we don't want any outside errors propagating into benchmarking
        torch.cuda.synchronize()

        # warmup `_callable` (and catches any failures in the process)
        _callable()
        torch.cuda.synchronize()

        # see https://github.com/triton-lang/triton/pull/840 for why `dtype=torch.int`
        buffer = torch.empty(self.L2_cache_size // 4, dtype=torch.int, device="cuda")
        buffer.zero_()

        # estimate the runtime of `_callable`
        event_pairs = self.get_event_pairs(estimation_iters)
        for start_event, end_event in event_pairs:
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()
        estimated_timing = self.get_event_pairs_min_timing(event_pairs)

        # adjust `benchmark_iters` to fit in the maximum benchmarking duration
        benchmark_iters = max(
            min(benchmark_iters, int(max_benchmark_duration // estimated_timing)), 1
        )

        # do the memory warmup
        for _ in range(memory_warmup_iters):
            buffer.zero_()

        # benchmark `_callable`
        event_pairs = self.get_event_pairs(benchmark_iters)
        for start_event, end_event in event_pairs:
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()
        benchmarked_timing = self.get_event_pairs_min_timing(event_pairs)

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor performance benchmarks
        del buffer

        # return the minimum of `estimated_timing` and `benchmarked_timing`,
        # we just want the minimum timing overall so we might as well check both
        return min(estimated_timing, benchmarked_timing)


benchmarker = InductorBenchmarker()
