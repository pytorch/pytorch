import time
from functools import cached_property, wraps
from itertools import chain
from statistics import median
from typing import Any, Callable, Dict, List, Tuple
from typing_extensions import Concatenate, ParamSpec, Self, TypeVar

import torch
from torch._dynamo.utils import counters


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
        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        elif "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


benchmarker = TritonBenchmarker()
