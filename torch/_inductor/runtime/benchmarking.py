import functools
import inspect
import time
from collections.abc import Callable
from functools import cached_property, wraps
from itertools import chain
from statistics import median
from typing import Any, Concatenate, Optional, Union
from typing_extensions import ParamSpec, Self, TypeVar

import torch
import torch._inductor.config as inductor_config
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch.utils._debug_mode import DebugMode


logger = torch._logging.getArtifactLogger(__name__, "benchmarking")
use_experimental_benchmarker = (
    inductor_config.use_experimental_benchmarker and torch.cuda.is_available()
)


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def may_distort_benchmarking_result(fn: Callable[..., Any]) -> Callable[..., Any]:
    from torch._inductor import config

    if config.test_configs.distort_benchmarking_result == "":
        return fn

    def distort(
        ms: list[float] | tuple[float, ...] | float,
    ) -> list[float] | tuple[float, ...] | float:
        if isinstance(ms, (list, tuple)):
            return type(ms)(distort(val) for val in ms)  # type: ignore[misc]

        distort_method = config.test_configs.distort_benchmarking_result
        assert isinstance(ms, float)
        if distort_method == "inverse":
            return 1.0 / ms if ms else 0.0
        elif distort_method == "random":
            import random

            return random.random()
        else:
            raise RuntimeError(f"Unrecognized distort method {distort_method}")

    @functools.wraps(fn)
    def wrapper(
        *args: list[Any], **kwargs: dict[str, Any]
    ) -> list[float] | tuple[float, ...] | float:
        ms = fn(*args, **kwargs)

        return distort(ms)

    return wrapper


def may_ban_benchmarking() -> None:
    if torch._inductor.config.deterministic:
        raise RuntimeError("""In the deterministic mode of Inductor, we will avoid those
        benchmarkings that would cause non deterministic results. Only benchmarkings in the vetted
        scenarios are allowed. Example include autotuning for triton configs of pointwise kernels.

        When you see this exception, you can do one of the following two things:
        1. if the benchmarking you are doing does not introduce any non-determinism, you can just
        add is_vetted_benchmarking=True to you benchmark_gpu call. That would solve the issue.

        2. if the benchmarking you are doing indeed introduces non-determinism, you'll need to disable
        such feature in deterministic mode or find an alternative implementation that is deterministic.
        """)


def time_and_count(
    fn: Callable[Concatenate[Any, P], T],
) -> Callable[Concatenate[Any, P], T]:
    """
    Wraps `fn` to increment the appropriate dynamo counters. It is expected that `fn`
    is a method of `Benchmarker` or one of its subclasses; typing limitations prevent
    us from declaring this directly.

    NOTE: If you're tempted to add a dynamo_timed call here, this function can be
    called enough that the dynamo_timed overhead is not negligible.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        fn_qual_name = f"{self.__class__.__name__}.{fn.__name__}"
        counters["inductor"][f"benchmarking.{fn_qual_name}"] += 1
        return fn(self, *args, **kwargs)

    return wrapper


class Benchmarker:
    """
    A device-agnostic benchmarking utility for measuring the runtime of
    inductor generated callables.
    """

    def __init__(self: Self) -> None:
        pass

    def infer_device(self, *fn_args: Any, **fn_kwargs: Any) -> torch.device:
        inferred_device: Optional[torch.device] = None
        for arg_or_kwarg in chain(fn_args, fn_kwargs.values()):
            # Some callables take nested structures as arguments so use the
            # flattened form to find any tensors
            for arg_or_kwarg_leaf in pytree.tree_leaves(arg_or_kwarg):
                if not isinstance(arg_or_kwarg_leaf, torch.Tensor):
                    continue
                if inferred_device is None:
                    inferred_device = arg_or_kwarg_leaf.device
                elif arg_or_kwarg_leaf.device != inferred_device:
                    raise ValueError(
                        "Can't safely infer the device type of `fn` with multiple device types in `fn_args` and `fn_kwargs`!"
                    )

        if inferred_device is None:
            raise ValueError(
                "Can't safely infer the device type of `fn` with no device types"
                " in `fn_args` or `fn_kwargs`. Use a direct benchmarking method instead e.g. "
                "`Benchmarker.benchmark_cpu` or `Benchmarker.benchmark_gpu`."
            )

        return inferred_device

    @time_and_count
    def benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: Optional[tuple[Any, ...]] = None,
        fn_kwargs: Optional[dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark `fn(*fn_args, *fn_kwargs)` and return the runtime, in milliseconds (the
        actual runtime calculation is dictated by the benchmarking implementation, but may be
        one of [mean, median, minimum, etc.]). Functions as a convenience wrapper around
        device-specific implementations, like `benchmark_cpu` and `benchmark_gpu`. Raises
        `ValueError(...)` if we can't safely infer the device type of `fn`; for example,
        if multiple device types are found in `fn_args` and `fn_kwargs`, or if no device
        types are found. To bypass device inference, provide the device to the `device`
        parameter.

        WARNING: if `fn` mutates `fn_args` or `fn_kwargs`, benchmarking may fail unexpectedly.
        For example, if `fn` clears a mutable object, subsequent invocations of `fn` during
        benchmarking will fail. In such cases, `fn` should handle cloning its arguments internally.
        If device inference is required, `Benchmarker.infer_device` can be used prior to calling
        this method without any arguments for `fn_args` and `fn_kwargs`.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - device: Which device to use for benchmarking. If not provided the device will be attempted
        to be inferred from `fn_args` and `fn_kwargs`.
        - **kwargs: The benchmarking implementation's kwargs.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
        inferred_device: Optional[torch.device] = None
        if device is not None:
            inferred_device = (
                torch.device(device) if isinstance(device, str) else device
            )
        else:
            if fn_args is None and fn_kwargs is None:
                raise ValueError(
                    "`fn_args` and `fn_kwargs` cannot both be None if `device` is not provided."
                )

            fn_args = fn_args or tuple()
            fn_kwargs = fn_kwargs or {}
            inferred_device = self.infer_device(*fn_args, **fn_kwargs)

        assert isinstance(inferred_device, torch.device)

        fn_args = fn_args or tuple()
        fn_kwargs = fn_kwargs or {}

        # No need to wrap if the callable takes no arguments
        if len(fn_args) == 0 and len(fn_kwargs) == 0:
            _callable = fn
        else:
            _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731

        warmup = kwargs.pop("warmup", inductor_config.inductor_default_autotune_warmup)
        rep = kwargs.pop("rep", inductor_config.inductor_default_autotune_rep)

        # Surfacing all kernels during autotuning is super noisy; filtering these out.
        with DebugMode._benchmarking_inductor():
            if inferred_device == torch.device("cpu"):
                return self.benchmark_cpu(_callable, warmup=warmup, rep=rep, **kwargs)
            # TODO(nmacchioni): For non-CPU functions we default to using the GPU-specific benchmarking
            # implementation which was written specifically with CUDA devices in mind, we may want to
            # explore alternate implementations for other device types.
            return self.benchmark_gpu(_callable, warmup=warmup, rep=rep, **kwargs)

    @time_and_count
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

        def run_for(ms: int) -> list[float]:
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

    @time_and_count
    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError

    @time_and_count
    def benchmark_gpu_with_cuda_graph(
        self: Self,
        _callable: Callable[[], Any],
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using CUDA graph capture and replay.

        This captures the callable into a CUDA graph and benchmarks the graph replay,
        which eliminates kernel launch overhead for fair comparison between different
        implementations.
        """
        # Warmup
        _callable()
        torch.cuda.synchronize()

        # Capture into CUDA graph
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph, capture_error_mode="thread_local"):
            _callable()
        torch.cuda.synchronize()

        return self.benchmark_gpu(cuda_graph.replay, **kwargs)


class TritonBenchmarker(Benchmarker):
    @cached_property
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @may_distort_benchmarking_result
    @time_and_count
    # pyrefly: ignore [bad-override]
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        is_vetted_benchmarking: bool = False,
        **kwargs: Any,
    ) -> float:
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
        if not is_vetted_benchmarking:
            may_ban_benchmarking()

        do_bench_params = inspect.signature(self.triton_do_bench).parameters
        for kwarg in list(kwargs.keys()):
            if kwarg not in do_bench_params:
                del kwargs[kwarg]
        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        elif "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


class InductorBenchmarker(TritonBenchmarker):  # noqa: docstring_linter
    @cached_property
    def L2_cache_size(self: Self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.L2_cache_size

    def get_event_pairs(
        self: Self, iters: int
    ) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Get `iters` pairs of CUDA events."""
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_event_pairs_min_timing(
        self: Self, event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ) -> float:
        """Get the minimum timing, in milliseconds, for a group of CUDA event pairs."""
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    @may_distort_benchmarking_result
    @time_and_count
    def benchmark_gpu(  # type: ignore[override]
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration: int = 25,
        return_mode: str = "min",
        grad_to_none: list[torch.Tensor] | None = None,
        is_vetted_benchmarking: bool = False,
        **kwargs: Any,
    ) -> float | list[float]:
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
        `benchmark_iters` to fit in the allotted maximum duration.
        - return_mode: Return mode for benchmark results. Options are "min" (default),
        "all" (returns all measurements).
        - grad_to_none: Optionally, a list of tensors whose gradients should be cleared
        before each benchmark iteration.
        - is_vetted_benchmarking: in deterministic mode, we only allow
        benchmarking in vetted cases.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - If return_mode="min": The minimum runtime of `_callable`, in milliseconds.
        - If return_mode="all": List of all runtime measurements, in milliseconds.
        """

        if not is_vetted_benchmarking:
            may_ban_benchmarking()

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
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()
        estimated_timing = self.get_event_pairs_min_timing(event_pairs)

        # adjust `benchmark_iters` to fit in the maximum benchmarking duration
        if estimated_timing > 0:
            benchmark_iters = max(
                min(benchmark_iters, int(max_benchmark_duration // estimated_timing)), 1
            )

        # do the memory warmup
        for _ in range(memory_warmup_iters):
            buffer.zero_()

        # benchmark `_callable`
        event_pairs = self.get_event_pairs(benchmark_iters)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor performance benchmarks
        del buffer

        # Return based on the requested mode
        if return_mode == "all":
            # Get all timings from event pairs
            all_timings = [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
            return all_timings
        elif return_mode == "min":
            benchmarked_timing = self.get_event_pairs_min_timing(event_pairs)
            # return the minimum of `estimated_timing` and `benchmarked_timing`,
            # we just want the minimum timing overall so we might as well check both
            return min(estimated_timing, benchmarked_timing)
        else:
            raise ValueError(
                f"Unsupported return_mode: {return_mode}. Use 'min' or 'all'."
            )


benchmarker = (
    InductorBenchmarker() if use_experimental_benchmarker else TritonBenchmarker()
)
