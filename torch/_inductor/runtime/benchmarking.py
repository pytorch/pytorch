import time
from functools import cached_property, wraps
from statistics import median
from typing import Any, Callable, Dict, List, Tuple
from typing_extensions import ParamSpec, Self, TypeVar

import torch
from torch._inductor.utils import is_cpu_device


log = torch._logging.getArtifactLogger(__name__, "benchmarking")


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def maybe_time(fn: Callable[P, T]) -> Callable[P, T]:
    if not torch._logging._internal.log_state.is_artifact_enabled("benchmarking"):
        return fn

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_s = time.perf_counter()
        result = fn(*args, **kwargs)
        log.debug(
            "fn:%r args:[%r, %r] took %f seconds.",
            fn.__name__,
            args,
            kwargs,
            time.perf_counter() - start_s,
        )
        return result

    return wrapper


class Benchmarker:
    def __init__(self: Self) -> None:
        pass

    @maybe_time
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

    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


class TritonBenchmarker(Benchmarker):
    @cached_property
    @maybe_time
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's do_bench."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @maybe_time
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
        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        elif "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


benchmarker = TritonBenchmarker()
