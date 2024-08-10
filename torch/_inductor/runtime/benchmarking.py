import inspect
import time
from functools import cached_property, wraps
from statistics import median
from typing import Any, Callable, Dict, List, Tuple
from typing_extensions import Concatenate, ParamSpec, Self, TypeVar

import torch
from torch._dynamo.utils import counters
from torch._inductor.config import benchmarking as benchmarking_config, is_fbcode
from torch._inductor.utils import is_cpu_device


log = torch._logging.getArtifactLogger(__name__, "benchmarking")


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def maybe_time(fn: Callable[P, T]) -> Callable[P, T]:
    """Wrapper that logs function durations, in milliseconds, along with the
    function's args and kwargs if logging is enabled, otherwise a no-op.
    """
    if not torch._logging._internal.log_state.is_artifact_enabled("benchmarking"):
        return fn

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_s = time.perf_counter()
        result = fn(*args, **kwargs)
        log.debug(
            "fn:%r args:[%r, %r] took %f milliseconds.",
            fn.__name__,
            args,
            kwargs,
            (time.perf_counter() - start_s) * MILLISECONDS_PER_SECOND,
        )
        return result

    return wrapper


def count(fn: Callable[Concatenate[Any, P], T]) -> Callable[Concatenate[Any, P], T]:
    """Wrapper that increments dynamo counters on function call for subclasses of `Benchmarker`;
    counter scheme is `counters["inductor"]["benchmarking.Foo.bar"]` where "Foo" is the subclass
    and "bar" is the function.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        counters["inductor"][
            "benchmarking." + type(self).__name__ + "." + fn.__name__
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
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> float:
        """Construct benchmarkable callable and dispatch benchmark request to the appropriate
        benchmarking function depending on the device type of `fn_args` and `fn_kwargs`.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - **kwargs: The benchmarker's keyword arguments.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
        if is_cpu_device(list(fn_args) + list(fn_kwargs.values())):
            return self.benchmark_cpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)
        return self.benchmark_gpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)

    @maybe_time
    @count
    def benchmark_cpu(
        self: Self, _callable: Callable[[], Any], warmup: int = 20, rep: int = 100
    ) -> float:
        """Benchmark a CPU callable.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - warmup: Duration to run the callable before benchmarking, in milliseconds.
        - rep: Duration to run the benchmarking, in milliseconds.

        Returns:
        - The median runtime of `_callable`, in milliseconds.
        """

        def run_for(ms: int) -> List[float]:
            timings = []
            run_start_s = time.perf_counter()
            while True:
                start_s = time.perf_counter()
                _callable()
                end_s = time.perf_counter()
                timings.append((end_s - start_s) * MILLISECONDS_PER_SECOND)
                if ((end_s - run_start_s) * MILLISECONDS_PER_SECOND) > ms:
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
        """Lazily import Triton's do_bench."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @maybe_time
    @count
    def benchmark_gpu(self: Self, _callable: Callable[[], Any], **kwargs: Any) -> float:
        """Benchmark a GPU callable using Triton's do_bench.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - quantiles: A tuple of floats denoting the requested quantiles.
        - return_mode: The requested return mode, one of "min", "max", "mean", or "median".
        - **kwargs: Additional kwargs passed to triton.testing.do_bench.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        """

        # this may be used as a fallback if other features are disabled, in that case we
        # need to prune any additional kwargs that are not part of do_bench's signature
        do_bench_sig = inspect.signature(self.triton_do_bench)
        if "**kwargs" not in str(do_bench_sig):
            for kwarg in list(kwargs.keys()):
                if kwarg not in do_bench_sig.parameters:
                    del kwargs[kwarg]

        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        elif "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


def is_feature_enabled(feature_name: str) -> bool:
    """Generic method to decide if we should enable a feature. A feature can be enabled
    in various ways, with priority in descending order:

    1. Exporting `TORCHINDUCTOR_BENCHMARKING_{feature_name.upper()}=X`. If `X=1`, the
    feature will be enabled. If `X=0`, the feature will be disabled.
    2a. [OSS Only] Setting `torch._inductor.config.benchmarking.{feature_name}.oss_default`.
    2b. [Internal Only] Feature is gated by JK enablement. The local feature version, hardcoded
    as `torch._inductor.fb.benchmarking.{feature_name.upper() + "_VERSION"}`, is compared against
    the JK feature version, `"pytorch/benchmarking:{feature_name.upper()}_VERSION"`. If the
    local feature version is greater than or equal to the JK feature version, the feature is
    considered enabled. Otherwise, the feature is disabled.
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
    """Wrapper that falls back to the parent class' equivalent method if the caller
    object's `self.should_fallback` evaluates to `True`.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        if not is_feature_enabled(self.feature_name):
            fallback_fn = getattr(super(type(self), self), fn.__name__)
            log.debug(
                "benchmarking.%s.%s falls back to benchmarking.%s.%s.",
                type(self).__name__,
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
        """Get the L2 cache size of the current device."""
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
        - _callable: The callable to benchmark:

        Keyword Arguments:
        - estimation_iters: The number of iterations to run `_callable` during
        runtime estimation.
        - memory_warmup_iters: The number of iterations to flush the L2 cache
        before benchmarking.
        - benchmark_iters: The number of iterations to run `_callable` during
        benchmarking.
        - max_benchmark_duration: The maximum duration of the benchmarking,
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
            min(benchmark_iters, max_benchmark_duration // estimated_timing), 1
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

        # return the minimum of estimated_timing and benchmarked_timing, since
        # we just want the minimum timing overall we might check both
        return min(estimated_timing, benchmarked_timing)


benchmarker = InductorBenchmarker()
