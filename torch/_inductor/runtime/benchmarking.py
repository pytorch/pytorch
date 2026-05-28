import contextlib
import functools
import inspect
import time
from collections.abc import Callable, Iterator
from functools import cached_property, wraps
from itertools import chain
from statistics import median
from typing import Any, Concatenate
from typing_extensions import ParamSpec, Self, TypeVar

import torch
import torch._inductor.config as inductor_config
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import counters
from torch.utils._debug_mode import DebugMode
from torch.utils._ordered_set import OrderedSet


logger = torch._logging.getArtifactLogger(__name__, "benchmarking")
GPU_BENCHMARK_DEVICE_TYPES = ("cuda", "xpu", "mtia")
_CALLABLE_PROFILE_EVENT_NAME = "_CALLABLE"


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def _get_default_gpu_device_type() -> str:
    avail_gpus = [
        device_type
        for device_type in GPU_BENCHMARK_DEVICE_TYPES
        if getattr(torch, device_type).is_available()
    ]
    assert len(avail_gpus) <= 1
    return "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()


def _normalize_gpu_device_type(device_type: str | torch.device | None) -> str:
    if device_type is None:
        return _get_default_gpu_device_type()
    if isinstance(device_type, torch.device):
        return device_type.type
    return torch.device(device_type).type


_GpuBenchmarkLockContext = Callable[[], contextlib.AbstractContextManager[None]]
_gpu_benchmark_lock_context: _GpuBenchmarkLockContext | None = None


def set_gpu_benchmark_lock_context(
    context_factory: _GpuBenchmarkLockContext | None,
) -> _GpuBenchmarkLockContext | None:
    """Override the process-local GPU benchmark lock context.

    This lets benchmark harnesses provide the context used by Inductor GPU
    benchmark calls. Some benchmark helpers delegate to other benchmark
    methods, so harness contexts should support nested entry from the same
    thread. Returning the previous context lets callers restore it in tests.
    """
    global _gpu_benchmark_lock_context
    previous = _gpu_benchmark_lock_context
    _gpu_benchmark_lock_context = context_factory
    return previous


@contextlib.contextmanager
def maybe_gpu_benchmark_lock() -> Iterator[None]:
    """Optionally enter the registered GPU benchmark lock context."""
    context_factory = _gpu_benchmark_lock_context
    if context_factory is None:
        yield
        return
    with context_factory():
        yield


def gpu_benchmark_lock(fn: Callable[P, T]) -> Callable[P, T]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with maybe_gpu_benchmark_lock():
            return fn(*args, **kwargs)

    return wrapper


# Device-type → benchmarking function registry.
# Keys must match torch.device.type (e.g., "cpu", "cuda", "mps", "xpu", ...).
# Values are callables with signature:
#   fn(self: Benchmarker, _callable: Callable[..., Any], *, warmup: int, rep: int, **kwargs) -> Any
_BENCHMARK_DISPATCH: dict[str, Callable[..., Any]] = {}


def register_benchmarker(
    device_type: str,
    fn: Callable[..., Any],
    *,
    override: bool = False,
) -> None:
    """
    Register a device-type specific benchmarker.

    Args:
        device_type: torch.device.type string (e.g., "cuda", "cpu", "mps", "xpu").
        fn: callable(self, _callable, *, warmup, rep, **kwargs) -> Any
        override: allow overriding an existing registration.
    """
    if not isinstance(device_type, str) or not device_type:
        raise ValueError(
            "device_type must be a non-empty string matching torch.device.type"
        )
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not override and device_type in _BENCHMARK_DISPATCH:
        raise ValueError(
            f"Benchmarker for device_type '{device_type}' already registered"
        )
    _BENCHMARK_DISPATCH[device_type] = fn


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
        benchmarkings that would cause non-deterministic results. Only benchmarkings in the vetted
        scenarios are allowed. Examples include autotuning for triton configs of pointwise kernels.

        When you see this exception, you can do one of the following two things:
        1. if the benchmarking you are doing does not introduce any non-determinism, you can just
        add is_vetted_benchmarking=True to your benchmark_gpu call. That would solve the issue.

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

    def infer_device(self, *fn_args: Any, **fn_kwargs: Any) -> torch.device:
        inferred_device: torch.device | None = None
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
        fn_args: tuple[Any, ...] | None = None,
        fn_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
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
        inferred_device: torch.device | None = None
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
            # Keep a true zero-arg callable type to satisfy type checkers.
            def _callable() -> Any:
                return fn()
        else:
            _args = fn_args
            _kwargs = fn_kwargs

            def _callable() -> Any:
                return fn(*_args, **_kwargs)

        warmup = kwargs.pop("warmup", inductor_config.inductor_default_autotune_warmup)
        rep = kwargs.pop("rep", inductor_config.inductor_default_autotune_rep)

        # Surfacing all kernels during autotuning is super noisy; filtering these out.
        with DebugMode._benchmarking_inductor():
            # First, try a registered device-specific benchmarker
            benchmark_fn: Callable[..., Any] | None = _BENCHMARK_DISPATCH.get(
                inferred_device.type
            )
            if benchmark_fn is not None:
                return benchmark_fn(self, _callable, warmup=warmup, rep=rep, **kwargs)

            # Backward-compatible default:
            # - CPU  -> CPU benchmark path
            # - else -> GPU benchmark path (legacy behavior retained for non-CPU)
            if inferred_device == torch.device("cpu"):
                return self.benchmark_cpu(_callable, warmup=warmup, rep=rep, **kwargs)
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
    @gpu_benchmark_lock
    def benchmark_gpu_with_cuda_graph(
        self: Self,
        _callable: Callable[[], Any],
        grad_to_none: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using CUDA graph capture and replay.

        This captures the callable into a CUDA graph and benchmarks the graph replay,
        which eliminates kernel launch overhead for fair comparison between different
        implementations.
        """
        # Warmup
        torch.cuda.synchronize()
        _callable()
        torch.cuda.synchronize()

        # Side-stream warmup then capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            _callable()
        stream.synchronize()

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            cuda_graph, stream=stream, capture_error_mode="thread_local"
        ):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            _callable()

        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # grad clearing is captured in the graph, don't pass it through.
        return self.benchmark_gpu(cuda_graph.replay, **kwargs)


# Make built-in defaults explicit via the registry
def _default_cpu_bench(self, f, *, warmup, rep, **kw):
    return self.benchmark_cpu(f, warmup=warmup, rep=rep, **kw)


def _default_cuda_bench(self, f, *, warmup, rep, **kw):
    kw.setdefault("device_type", "cuda")
    return self.benchmark_gpu(f, warmup=warmup, rep=rep, **kw)


def _default_xpu_bench(self, f, *, warmup, rep, **kw):
    kw.setdefault("device_type", "xpu")
    return self.benchmark_gpu(f, warmup=warmup, rep=rep, **kw)


register_benchmarker("cpu", _default_cpu_bench, override=True)
register_benchmarker("cuda", _default_cuda_bench, override=True)
register_benchmarker("xpu", _default_xpu_bench, override=True)


def _get_callable_device_kernel_time_us(
    kineto_events: Any,
    profiler_events: Any,
    rep: int,
    profiler_device_type: Any,
) -> float:
    from torch.autograd import DeviceType

    benchmark_event_ids: OrderedSet[int] = OrderedSet()

    def collect_cpu_event_ids(event: Any) -> None:
        if event.device_type != DeviceType.CPU:
            return

        benchmark_event_ids.add(event.id)
        for child in event.cpu_children:
            collect_cpu_event_ids(child)

    benchmark_events = [
        event
        for event in profiler_events
        if event.name == _CALLABLE_PROFILE_EVENT_NAME
        and event.device_type == DeviceType.CPU
    ]
    if len(benchmark_events) != rep:
        raise RuntimeError(
            f"Expected {rep} {_CALLABLE_PROFILE_EVENT_NAME} profiling events. "
            f"Found {len(benchmark_events)} events."
        )

    for event in benchmark_events:
        collect_cpu_event_ids(event)

    # An op may re-dispatch internally (e.g. aten::sum with no dim calls
    # aten::sum with dim), producing a kineto CPU event whose corr ID the
    # device kernel links to but which is absent from the profiler event tree.
    # Include kineto corr IDs of all CPU events within the _CALLABLE time
    # window to cover these hidden dispatches.
    callable_kineto_windows: list[tuple[int, int]] = []
    for ev in kineto_events:
        if (
            ev.name() == _CALLABLE_PROFILE_EVENT_NAME
            and ev.device_type() == DeviceType.CPU
        ):
            callable_kineto_windows.append((ev.start_ns(), ev.end_ns()))

    if callable_kineto_windows:
        for ev in kineto_events:
            if ev.device_type() != DeviceType.CPU:
                continue
            ev_start = ev.start_ns()
            ev_end = ev.end_ns()
            for win_start, win_end in callable_kineto_windows:
                if ev_start >= win_start and ev_end <= win_end:
                    benchmark_event_ids.add(ev.correlation_id())
                    break

    device_time_us = 0.0
    for event in kineto_events:
        linked_correlation_id = event.linked_correlation_id()
        correlation_id = event.correlation_id()
        activity_type = event.activity_type()
        if (
            event.device_type() == profiler_device_type
            and activity_type != "gpu_user_annotation"
            and (
                linked_correlation_id in benchmark_event_ids
                or (
                    linked_correlation_id == 0 and correlation_id in benchmark_event_ids
                )
            )
            and event.name() != "Context Sync"
        ):
            device_time_us += (event.end_ns() - event.start_ns()) / 1000.0

    return device_time_us


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
    @gpu_benchmark_lock
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
        try:
            if "quantiles" in kwargs:
                return self.triton_do_bench(_callable, **kwargs)[0]
            elif "return_mode" in kwargs:
                return self.triton_do_bench(_callable, **kwargs)
            return self.triton_do_bench(_callable, **kwargs, return_mode="median")
        except Exception as e:
            # ErrorInvalidConfiguration
            # Return inf to skip this config during autotuning
            error_str = str(e).lower()
            if "invalid configuration" in error_str:
                logger.warning(
                    "Skipping benchmark due to invalid configuration error: %s",
                    error_str,
                )
                return float("inf")
            raise


class InductorBenchmarker(TritonBenchmarker):  # noqa: docstring_linter
    def __init__(self: Self) -> None:
        super().__init__()
        self._in_cudagraph_benchmark = False

    @cached_property
    def L2_cache_size(self: Self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
        return self.get_device_cache_size()

    def get_device_cache_size(
        self: Self, device_type: str | torch.device | None = None
    ) -> int:
        """Get the L2/global cache size, in bytes, of the current device."""
        if "L2_cache_size" in self.__dict__:
            return self.__dict__["L2_cache_size"]

        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)
        device = device_interface.current_device()
        props = device_interface.get_device_properties(device)
        for attr in ("L2_cache_size", "last_level_cache_size"):
            cache_size = getattr(props, attr, None)
            if cache_size:
                return cache_size
        return 256 * 1024 * 1024

    def get_event_pairs(
        self: Self, iters: int, device_type: str | torch.device | None = None
    ) -> list[tuple[Any, Any]]:
        """Get `iters` pairs of device events."""
        device_interface = get_interface_for_device(
            _normalize_gpu_device_type(device_type)
        )
        return [
            (
                device_interface.Event(enable_timing=True),
                device_interface.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_event_pairs_min_timing(
        self: Self, event_pairs: list[tuple[Any, Any]]
    ) -> float:
        """Get the minimum timing, in milliseconds, for a group of event pairs."""
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    @time_and_count
    def benchmark_gpu_with_cuda_graph(
        self: Self,
        _callable: Callable[[], Any],
        grad_to_none: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> float:
        # Prevent benchmark_gpu from re-entering this method
        # when autotune_cudagraph_benchmarking is enabled.
        self._in_cudagraph_benchmark = True
        try:
            result = super().benchmark_gpu_with_cuda_graph(
                _callable, grad_to_none=grad_to_none, **kwargs
            )
        finally:
            self._in_cudagraph_benchmark = False
        return result

    @may_distort_benchmarking_result
    @time_and_count
    @gpu_benchmark_lock
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
        device_type: str | torch.device | None = None,
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

        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)
        if (
            device_type == "cuda"
            and inductor_config.autotune_cudagraph_benchmarking
            and not self._in_cudagraph_benchmark
        ):
            try:
                return self.benchmark_gpu_with_cuda_graph(
                    _callable,
                    estimation_iters=estimation_iters,
                    memory_warmup_iters=memory_warmup_iters,
                    benchmark_iters=benchmark_iters,
                    max_benchmark_duration=max_benchmark_duration,
                    return_mode=return_mode,
                    grad_to_none=grad_to_none,
                    device_type=device_type,
                )
            except RuntimeError:
                logger.warning(
                    "CUDA graph capture failed during benchmarking, "
                    "falling back to eager benchmarking",
                    exc_info=True,
                )

        # we don't want any outside errors propagating into benchmarking
        device_interface.synchronize()

        # warmup `_callable` (and catches any failures in the process)
        _callable()
        device_interface.synchronize()

        # see https://github.com/triton-lang/triton/pull/840 for why `dtype=torch.int`
        buffer = torch.empty(
            self.get_device_cache_size(device_type) // 4,
            dtype=torch.int,
            device=device_type,
        )
        buffer.zero_()

        # estimate the runtime of `_callable`
        event_pairs = self.get_event_pairs(estimation_iters, device_type=device_type)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        device_interface.synchronize()
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
        event_pairs = self.get_event_pairs(benchmark_iters, device_type=device_type)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        device_interface.synchronize()

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


class TorchProfilerBenchmarker(InductorBenchmarker):  # noqa: docstring_linter
    """Benchmarker that uses torch.profiler for GPU kernel benchmarking."""

    @time_and_count
    @gpu_benchmark_lock
    def benchmark_gpu(  # type: ignore[override]
        self: Self,
        _callable: Callable[[], Any],
        warmup: int = 25,
        rep: int = 100,
        estimation_iters: int = 5,
        memory_warmup_iters: int = 10,
        max_benchmark_duration: int = 25,
        return_mode: str = "mean",
        grad_to_none: list[torch.Tensor] | None = None,
        device_type: str | torch.device | None = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using torch.profiler.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - warmup: Ignored (kept for API compat). Warmup is handled by the
        estimation phase which runs estimation_iters with cache flushing.
        - rep: Optionally, the maximum number of iterations to run during benchmarking.
        - estimation_iters: Optionally, the number of iterations used to estimate
        the runtime of `_callable` for dynamic rep adjustment. These iterations
        also serve as warmup for the callable.
        - memory_warmup_iters: Optionally, the number of buffer.zero_() iterations
        to run after estimation to bring the cache into a steady state before
        the profiled benchmark phase.
        - max_benchmark_duration: Optionally, the maximum duration of the profiled
        benchmark phase, in milliseconds. The rep count is reduced if the
        estimated total would exceed this budget.
        - return_mode: Return mode for benchmark results. Options are "min", "mean" (default),
        or "max".
        - grad_to_none: Optionally, a list of tensors whose gradients should be cleared
        before each benchmark iteration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - The runtime of `_callable` in milliseconds, computed according to return_mode.
        """
        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)

        # we don't want any outside errors propagating into benchmarking
        device_interface.synchronize()

        # warmup `_callable` (and catches any failures in the process)
        _callable()
        device_interface.synchronize()

        # Keep Triton's 256 MB cache flush on ROCm. On other backends, reuse
        # the shared L2-sized flush from InductorBenchmarker.
        # see https://github.com/triton-lang/triton/pull/840 for why `dtype=torch.int`
        if torch.version.hip:
            buffer_size_bytes = 256 * 1024 * 1024
        else:
            buffer_size_bytes = self.get_device_cache_size(device_type)
        buffer = torch.empty(
            buffer_size_bytes // 4, dtype=torch.int, device=device_type
        )
        buffer.zero_()

        # Estimation phase with separate event pairs — also serves as warmup.
        # Using per-iteration event pairs lets us take the min, matching
        # InductorBenchmarker's approach for a more robust estimate.
        event_pairs = self.get_event_pairs(estimation_iters, device_type=device_type)
        for start_event, end_event in event_pairs:
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        device_interface.synchronize()
        estimated_ms = self.get_event_pairs_min_timing(event_pairs)
        if estimated_ms > 0:
            rep = max(min(rep, int(max_benchmark_duration / estimated_ms)), 1)

        # Light memory warmup: flush the cache into a steady state before
        # the profiled run.  No callable — estimation already warmed it.
        for _ in range(memory_warmup_iters):
            buffer.zero_()

        # benchmark with profiler
        # Use both CPU and device activities, otherwise record_function
        # will not record the region.
        device_type_upper = device_type.upper()
        profile_activity = getattr(torch.profiler.ProfilerActivity, device_type_upper)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                profile_activity,
            ],
            record_shapes=False,
        ) as prof:
            for _ in range(rep):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                buffer.zero_()

                with torch.profiler.record_function("_CALLABLE"):
                    _callable()

        device_interface.synchronize()

        # Extract _CALLABLE GPU time directly from raw kineto events.
        # This avoids prof.key_averages() which triggers expensive lazy
        # processing: _parse_kineto_results (wrapping every raw event in
        # a Python FunctionEvent), _build_tree, and grouping/aggregation.
        from torch.autograd import DeviceType as _DeviceType

        profiler_device_type = getattr(_DeviceType, device_type_upper)
        callable_gpu_time_us = 0.0
        for kineto_event in prof.profiler.kineto_results.events():
            if (
                kineto_event.name() == _CALLABLE_PROFILE_EVENT_NAME
                and kineto_event.device_type() == profiler_device_type
            ):
                callable_gpu_time_us += (
                    kineto_event.end_ns() - kineto_event.start_ns()
                ) / 1000.0

        if callable_gpu_time_us <= 0:
            callable_gpu_time_us = _get_callable_device_kernel_time_us(
                prof.profiler.kineto_results.events(),
                prof.events(),
                rep,
                profiler_device_type,
            )

        if callable_gpu_time_us <= 0:
            raise AssertionError(
                f"TorchProfilerBenchmarker: '_CALLABLE' {device_type_upper} event not found in "
                "raw kineto results, and no correlated device events were found."
            )

        # TODO: Revisit incorporating launch overhead effects.
        total_time_us = callable_gpu_time_us
        avg_time_ms = (total_time_us / rep) / 1000.0

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor performance benchmarks
        del buffer

        # Return based on the requested mode
        # Note: For profiler-based benchmarking, we return the mean time per iteration
        # min/max modes are kept for API compatibility but return the mean
        if return_mode in ("min", "mean", "max"):
            return avg_time_ms
        else:
            raise ValueError(
                f"Unsupported return_mode: {return_mode}. Use 'min', 'mean', or 'max'."
            )


def _make_default_benchmarker() -> Benchmarker:
    if inductor_config.use_torch_profiler_benchmarker:
        return TorchProfilerBenchmarker()
    if inductor_config.use_experimental_benchmarker:
        return InductorBenchmarker()
    return TritonBenchmarker()


benchmarker = _make_default_benchmarker()
