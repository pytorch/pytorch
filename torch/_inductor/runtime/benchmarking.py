from __future__ import annotations

from functools import cached_property, lru_cache, partial, wraps
from importlib import import_module
from random import randint
from statistics import median
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

import torch
from torch._dynamo.utils import counters
from torch._inductor.config import benchmarking as benchmarking_config, is_fbcode


log = torch._logging.getArtifactLogger(__name__, "benchmarking")


def time_and_log(fn: Callable[..., Any]) -> Callable[..., Any]:
    if not torch._logging._internal.log_state.is_artifact_enabled("benchmarking"):
        return fn

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        result = fn(*args, **kwargs)
        log.debug(
            "{function_name} took {elapsed_time_s} seconds.",
            extra=dict(
                function_name=fn.__name__,
                elapsed_time_s=perf_counter() - start_time,
            ),
        )
        return result

    return wrapper


def should_generic(
    config_name: str,
    default_oss: bool,
    internal_if_enabled: bool,
    internal_if_disabled: bool,
    jk_name: str,
) -> bool:
    config_val = getattr(benchmarking_config, config_name)

    @lru_cache(None)
    def is_jk_enabled(name: str) -> bool:
        try:
            value = getattr(import_module("torch._inductor.fb.benchmarking"), name)
        except ModuleNotFoundError:
            return False
        else:
            return value >= torch._utils_internal.justknobs_getval_int(
                f"pytorch/benchmarking:{name}"
            )

    if config_val is not None:
        return config_val
    if not is_fbcode():
        return default_oss
    if is_jk_enabled(jk_name):
        return internal_if_enabled
    return internal_if_disabled


should_fallback_to_original_benchmarking = partial(
    should_generic,
    "fallback_to_original_benchmarking",
    False,
    False,
    True,
    "fallback_to_original_benchmarking_version",
)
should_enable_lazy_benchmarking = partial(
    should_generic,
    "enable_lazy_benchmarking",
    True,
    False,
    True,
    "enable_lazy_benchmarking_version",
)
should_enable_early_ranking = partial(
    should_generic,
    "enable_early_ranking",
    True,
    False,
    True,
    "enable_early_ranking_verison",
)
should_enable_early_pruning = partial(
    should_generic,
    "enable_early_pruning",
    True,
    False,
    True,
    "enable_early_pruning_verison",
)


def maybe_fallback_to_original_benchmarking(
    original_fn_name: str,
) -> Callable[..., Any]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(self: Benchmarker, *args: Any, **kwargs: Any) -> Any:
            if should_fallback_to_original_benchmarking():
                log.debug(
                    "Falling back to original benchmarking function {original_fn_name} from {fn_name}.",
                    extra=dict(original_fn_name=original_fn_name, fn_name=fn.__name__),
                )
                counters["inductor"][
                    "benchmarking_fallback_to_original_benchmarking"
                ] += 1
                return getattr(self, original_fn_name)(*args, **kwargs)
            return fn(self, *args, **kwargs)

        return wrapper

    return decorator


def maybe_fallback_to_non_lazy_benchmarking(
    non_lazy_fn_name: str,
) -> Callable[..., Any]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(self: Benchmarker, *args: Any, **kwargs: Any) -> Any:
            if not should_enable_lazy_benchmarking():
                log.debug(
                    "Falling back to non-lazy benchmarking function {non_lazy_fn_name} from {fn_name}.",
                    extra=dict(non_lazy_fn_name=non_lazy_fn_name, fn_name=fn.__name__),
                )
                counters["inductor"][
                    "benchmarking_fallback_to_non_lazy_benchmarking"
                ] += 1
                return getattr(self, non_lazy_fn_name)(*args, **kwargs)
            return fn(self, *args, **kwargs)

        return wrapper

    return decorator


class LazyBenchmark:
    def __init__(self: Self, benchmark: Callable[[], float]) -> None:
        self.benchmark = benchmark

    @cached_property
    def timing_ms(self: Self) -> float:
        counters["inductor"]["benchmarking_finalize_lazy_benchmark"] += 1
        timing_ms = self.benchmark()
        # I don't think this helps with saving memory at all,
        # but at least it gives good signal if we ever try
        # to call self.benchmark again
        del self.benchmark
        return timing_ms

    def __float__(self: Self) -> float:
        return float(self.timing_ms)

    def __format__(self: Self, format_spec: str) -> str:
        return format(self.timing_ms, format_spec)

    def __str__(self: Self) -> str:
        return str(self.timing_ms)

    def __lt__(self: Self, other: Any) -> bool:
        return self.timing_ms < other

    def __le__(self: Self, other: Any) -> bool:
        return self.timing_ms <= other

    def __gt__(self: Self, other: Any) -> bool:
        return self.timing_ms > other

    def __ge__(self: Self, other: Any) -> bool:
        return self.timing_ms >= other

    def __add__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms + other
        return LazyBenchmark(lambda: self.timing_ms + other)

    def __radd__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other + self.timing_ms
        return LazyBenchmark(lambda: other + self.timing_ms)

    def __sub__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms - other
        return LazyBenchmark(lambda: self.timing_ms - other)

    def __rsub__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other - self.timing_ms
        return LazyBenchmark(lambda: other - self.timing_ms)

    def __mul__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms * other
        return LazyBenchmark(lambda: self.timing_ms * other)

    def __rmul__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other * self.timing_ms
        return LazyBenchmark(lambda: other * self.timing_ms)

    def __truediv__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms / other
        return LazyBenchmark(lambda: self.timing_ms / other)

    def __rtruediv__(self: Self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other / self.timing_ms
        return LazyBenchmark(lambda: other / self.timing_ms)


def is_cpu_device(inputs: List[Any]) -> bool:
    return all(
        _input.device == torch.device("cpu")
        for _input in inputs
        if isinstance(_input, torch.Tensor)
    )


class Benchmarker:
    def __init__(self: Self) -> None:
        self.memory_cache: Dict[str, float] = {}
        self.kwargs_hash_to_futures_gpu: Dict[
            str, List[Tuple[Callable[[], Any], str]]
        ] = {}

    @cached_property
    def L2_cache_size(self: Self) -> int:
        counters["inductor"]["benchmarking_L2_cache_size"] += 1
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.L2_cache_size

    @cached_property
    def gpu_queue_limit(self: Self) -> int:
        counters["inductor"]["benchmarking_gpu_queue_limit"] += 1
        # ensures the queue is empty
        torch.cuda.synchronize()
        # capping the search space for the queue limit to 2500 is good enough,
        # current queue limit on my machine (A100) is stable at ~1000, and
        # in the event that we do artificially cap the queue limit to 2500
        # we really shouldn't see any significant slowdowns
        torch.cuda._sleep(
            int(
                (self.cpu_launch_overhead_ms_per_event_record * 2500)
                / self.gpu_time_ms_per_gpu_clock_cycle
            )
        )
        for idx in range(2500):
            start_time_s = perf_counter()
            torch.cuda.Event(enable_timing=True).record()
            elapsed_time_ms = (perf_counter() - start_time_s) * 1000
            # recording an event is near instantaneous, unless we have hit
            # the queue limit, so 1ms seems like a good enough upper bound
            if elapsed_time_ms > 1:
                break
        torch.cuda.synchronize()
        return idx

    @cached_property
    def cpu_launch_overhead_ms_per_event_record(self: Self) -> float:
        counters["inductor"][
            "benchmarking_cpu_launch_overhead_ms_per_event_record"
        ] += 1
        # ensures the queue is empty
        torch.cuda.synchronize()
        start_time_s = perf_counter()
        for _ in range(100):
            torch.cuda.Event(enable_timing=True).record()
        torch.cuda.synchronize()
        return ((perf_counter() - start_time_s) * 1000) / 100

    @cached_property
    def cpu_launch_overhead_ms_per_gpu_cache_clear(self: Self) -> float:
        return self.cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear[0]

    @cached_property
    def gpu_time_ms_per_gpu_cache_clear(self: Self) -> float:
        return self.cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear[1]

    @cached_property
    def cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear(
        self: Self,
    ) -> Tuple[float, float]:
        counters["inductor"][
            "benchmarking_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear"
        ] += 1
        buffer = torch.empty(self.L2_cache_size // 4, dtype=torch.int, device="cuda")
        buffer.zero_()
        # synchronize after zeroing the buffer to reduce uncertainty
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_time_s = perf_counter()
        # 100 buffer zeroes is long enough to reduce uncertainty
        for _ in range(100):
            buffer.zero_()
        cpu_launch_overhead_ms_per_gpu_cache_clear = (
            (perf_counter() - start_time_s) * 1000
        ) / 100
        end_event.record()
        torch.cuda.synchronize()
        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor benchmarks
        del buffer
        return (
            cpu_launch_overhead_ms_per_gpu_cache_clear,
            start_event.elapsed_time(end_event) / 100,
        )

    @cached_property
    def gpu_time_ms_per_gpu_clock_cycle(self: Self) -> float:
        counters["inductor"]["benchmarking_gpu_time_ms_per_gpu_clock_cycle"] += 1
        # ensures the queue is empty
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # sleeping for 1000000 clock cycles is long enough
        # to average out most of the uncertainty
        torch.cuda._sleep(1000000)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / 1000000

    @maybe_fallback_to_original_benchmarking("original_do_bench")
    def benchmark(
        self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> float:
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        # should we be checking if all args and kwargs are on the same device?
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)

    def original_do_bench(
        self: Self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> float:
        args = list(fn_args)
        args.extend(fn_kwargs.values())
        if is_cpu_device(args):
            return self.original_do_bench_cpu(
                lambda: fn(*fn_args, **fn_kwargs), **kwargs
            )
        else:
            return self.original_do_bench_gpu(
                lambda: fn(*fn_args, **fn_kwargs), **kwargs
            )

    @time_and_log
    @maybe_fallback_to_original_benchmarking("original_do_bench_cpu")
    def benchmark_cpu(
        self: Self,
        _callable: Callable[[], Any],
        warmup_iters: int = 5,
        benchmark_iters: int = 20,
        **kwargs: Any,
    ) -> float:
        # duplicate of original implementation from runtime_utils
        timings_ms = []
        for _ in range(warmup_iters):
            _callable()
        for _ in range(benchmark_iters):
            start_time_s = perf_counter()
            _callable()
            timings_ms.append((perf_counter() - start_time_s) * 1000)
        return median(timings_ms)

    def original_do_bench_cpu(
        self: Self,
        fn: Callable[[], Any],
        warmup: int = 5,
        times: int = 20,
        **kwargs: Any,
    ) -> float:
        assert times > 0
        for _ in range(warmup):
            fn()
        durations = []
        for _ in range(times):
            t0 = perf_counter()
            fn()
            t1 = perf_counter()
            durations.append((t1 - t0) * 1000)
        # return the median time
        sorted_durations = sorted(durations)
        if times % 2 == 0:
            return (sorted_durations[times // 2 - 1] + sorted_durations[times // 2]) / 2
        else:
            return sorted_durations[times // 2]

    @time_and_log
    @maybe_fallback_to_original_benchmarking("original_do_bench_many_cpu")
    def benchmark_many_cpu(
        self: Self,
        callables: List[Callable[[], Any]],
        warmup_iters: int = 5,
        benchmark_iters: int = 20,
        **kwargs: Any,
    ) -> List[float]:
        # implement this to maintain consistency between cpu/gpu benchmarking functionality
        return [
            self.benchmark_cpu(_callable, warmup_iters, benchmark_iters)
            for _callable in callables
        ]

    def original_do_bench_many_cpu(
        self: Self,
        fns: List[Callable[[], Any]],
        warmup: int = 5,
        times: int = 20,
        **kwargs: Any,
    ) -> List[float]:
        return [
            self.original_do_bench_cpu(fn, warmup=warmup, times=times, **kwargs)
            for fn in fns
        ]

    @time_and_log
    @maybe_fallback_to_original_benchmarking("original_do_bench_gpu")
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration_ms: int = 25,
        **kwargs: Any,
    ) -> float:
        # we don't want any outside errors propagating into benchmarking
        torch.cuda.synchronize()

        # initialize here, avoids double buffer allocation
        self.cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear

        _callable()
        torch.cuda.synchronize()

        buffer = torch.empty(self.L2_cache_size // 4, dtype=torch.int, device="cuda")
        buffer.zero_()

        event_pairs = self.get_event_pairs(estimation_iters)
        start_time_s = perf_counter()
        for start_event, end_event in event_pairs:
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        # before we synchronize we want to measure the cpu-side cost of our buffer
        # zero, launching the callable, and the associated event records
        cpu_launch_overhead_ms_per_iter = (
            (perf_counter() - start_time_s) * 1000
        ) / estimation_iters
        torch.cuda.synchronize()
        estimated_timing_ms = self.get_min_timing_ms(event_pairs)

        # we don't want to benchmark for longer than max_benchmark_duration milliseconds,
        # so we reduce benchmark_iters to fit within this constraint if necessary
        benchmark_iters = self.reduce_benchmark_iters_to_max_benchmark_duration_ms(
            memory_warmup_iters,
            benchmark_iters,
            cpu_launch_overhead_ms_per_iter,
            estimated_timing_ms,
            1,
            max_benchmark_duration_ms,
        )
        # the self.gpu_queue_limit'th event queued to the GPU will execute synchronously
        # on the CPU; this is bad since we try to overlap the CPU launch period with a
        # GPU sleep to avoid the CPU launch overhead in our GPU event timings. what ends
        # up happening in this case is that the self.gpu_queue_limit'th event will wait
        # until the queue shrinks (i.e. after the GPU sleep ends) before queueing itself.
        # this has two issues, first we will waste a lot of time needlesly as the CPU
        # waits for the GPU to finish its sleep, and second we will no longer have a
        # proper overlap between the CPU launch overhead and the GPU sleep which can
        # result in incorrect timing values for particularly small kernels as the CPU
        # will eventually catch up to the GPU and we will end up in lock-step execution.
        # we can prevent this by benchmarking in blocks. we calculate the number of blocks
        # required by assuming that each benchmark iteration spawns approximately 5 events:
        # 1 for the buffer zeroing, 1 for the callable, 2 for the event records, and 1
        # extra allocated for any sub-kernels the callable may launch of which the most
        # common case is allocating an output Tensor. we then evenly split the total
        # number of benchmark iterations across all blocks.
        benchmark_blocks = ((benchmark_iters * 5) // (self.gpu_queue_limit - 1)) + 1
        benchmark_iters_per_block = benchmark_iters // benchmark_blocks
        memory_warmup_iters_per_block = memory_warmup_iters // benchmark_blocks
        required_gpu_sleep_cycles_per_block = self.get_required_gpu_sleep_cycles(
            memory_warmup_iters_per_block,
            benchmark_iters_per_block,
            cpu_launch_overhead_ms_per_iter,
        )

        # ensures the queue is empty
        torch.cuda.synchronize()
        event_pairs = self.get_event_pairs(benchmark_iters)
        for block_idx in range(benchmark_blocks):
            block_start = block_idx * benchmark_iters_per_block
            if block_idx == (benchmark_blocks - 1):
                block_end = benchmark_iters
            else:
                block_end = (block_idx + 1) * benchmark_iters_per_block
            torch.cuda._sleep(required_gpu_sleep_cycles_per_block)
            for _ in range(memory_warmup_iters_per_block):
                buffer.zero_()
            for start_event, end_event in event_pairs[block_start:block_end]:
                buffer.zero_()
                start_event.record()
                _callable()
                end_event.record()
            torch.cuda.synchronize()
        timing_ms = self.get_min_timing_ms(event_pairs)

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor benchmarks
        del buffer

        # we may as well use the estimation loop signal as well
        return min(estimated_timing_ms, timing_ms)

    def original_do_bench_gpu(
        self: Self,
        fn: Callable[[], Any],
        warmup: int = 25,
        rep: int = 100,
        grad_to_none: Optional[torch.Tensor] = None,
        quantiles: Optional[Tuple[float, ...]] = None,
        fast_flush: bool = True,
        return_mode: str = "median",
        **kwargs: Any,
    ) -> float:
        """
        Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
        the 20-th and 80-th performance percentile.

        :param fn: Function to benchmark
        :type fn: Callable
        :param warmup: Warmup time (in ms)
        :type warmup: int
        :param rep: Repetition time (in ms)
        :type rep: int
        :param grad_to_none: Reset the gradient of the provided tensor to None
        :type grad_to_none: torch.tensor, optional
        :param quantiles: Performance percentile to return in addition to the median.
        :type quantiles: tuple[float]
        :param fast_flush: Use faster kernel to flush L2 between measurements
        :type fast_flush: bool
        """
        assert return_mode in ["min", "max", "mean", "median"]
        import torch

        fn()
        torch.cuda.synchronize()

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2
        # doesn't contain any input data before the run
        if fast_flush:
            cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
        else:
            cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

        # Estimate the runtime of the function
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        # Warm-up
        for _ in range(n_warmup):
            fn()
        # Benchmark
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            start_event[i].record()
            fn()
            end_event[i].record()
        # Record clocks
        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )
        if quantiles is not None:
            ret = torch.quantile(
                times, torch.tensor(quantiles, dtype=torch.float)
            ).tolist()
            return ret[0]
        return getattr(torch, return_mode)(times).item()

    @time_and_log
    @maybe_fallback_to_original_benchmarking("original_do_bench_many_gpu")
    def benchmark_many_gpu(
        self: Self,
        callables: List[Callable[[], Any]],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration_ms: int = 25,
        ranking_key: Optional[str] = None,
        pruning_key: Optional[str] = None,
        pruning_factor: float = 1.1,
        pruning_limit: int = 5,
        **kwargs: Any,
    ) -> List[float]:
        # we don't want any outside errors propagating into benchmarking
        torch.cuda.synchronize()

        # initialize here, avoids double buffer allocation
        self.cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear

        # this will fail the entire group if any of the callables fail,
        # in the case that users directly call this method they should
        # handle this case on their end. in the case that we get here
        # from a grouping of lazy benchmarks, we should never have failures
        # since they will have already been checked previously
        for _callable in callables:
            _callable()
        torch.cuda.synchronize()

        buffer = torch.empty(self.L2_cache_size // 4, dtype=torch.int, device="cuda")
        buffer.zero_()

        interleaved_event_pairs = self.get_interleaved_event_pairs(
            len(callables), estimation_iters
        )
        torch.cuda._sleep(int(1 / self.gpu_time_ms_per_gpu_clock_cycle))
        start_time_s = perf_counter()
        for event_pairs in interleaved_event_pairs:
            for _callable, (start_event, end_event) in zip(callables, event_pairs):
                buffer.zero_()
                start_event.record()
                _callable()
                end_event.record()
        cpu_launch_overhead_ms_per_iter = (
            (perf_counter() - start_time_s) * 1000
        ) / estimation_iters
        torch.cuda.synchronize()
        estimated_timings_ms = self.get_interleaved_min_timings_ms(
            interleaved_event_pairs
        )

        if ranking_key is not None:
            if should_enable_early_ranking():
                log.debug(
                    "Returning early ranking for ranking key {ranking_key}.",
                    extra=dict(ranking_key=ranking_key),
                )
                counters["inductor"]["benchmarking_early_ranking"] += 1
                # explicitly delete the buffer, sometimes helps memory
                # footprint metrics in OSS Inductor benchmarks
                del buffer
                return estimated_timings_ms
            else:
                log.debug("Early ranking is disabled, continuing benchmarking.")

        callable_to_timing_ms = {}
        for _callable, estimated_timing_ms in zip(callables, estimated_timings_ms):
            callable_to_timing_ms[_callable] = estimated_timing_ms

        if pruning_key is not None:
            if should_enable_early_pruning():
                cpu_launch_overhead_ms_per_iter_per_callable = (
                    cpu_launch_overhead_ms_per_iter / len(callables)
                )
                target_timing_ms = min(estimated_timings_ms) * pruning_factor
                callables_to_benchmark: List[Callable[[], Any]] = []
                for _callable, timing_ms in sorted(
                    callable_to_timing_ms.items(), key=lambda x: x[1]
                ):
                    if len(callables_to_benchmark) == pruning_limit:
                        break
                    if timing_ms <= target_timing_ms:
                        callables_to_benchmark.append(_callable)
                counters["inductor"]["benchmarking_early_pruning"] += len(
                    callables
                ) - len(callables_to_benchmark)
                log.debug(
                    "Early pruning pruned {num_pruned_callables} from {num_callables} total callables,"  # noqa: G003
                    + " continuing benchmarking with remaining {num_unpruned_callables} callables.",
                    extra=dict(
                        num_pruned_callables=len(callables)
                        - len(callables_to_benchmark),
                        num_callables=len(callables),
                        num_unpruned_callables=len(callables_to_benchmark),
                    ),
                )
                cpu_launch_overhead_ms_per_iter = (
                    cpu_launch_overhead_ms_per_iter_per_callable
                    * len(callables_to_benchmark)
                )
                estimated_timings_ms = [
                    callable_to_timing_ms[_callable]
                    for _callable in callables_to_benchmark
                ]
            else:
                log.debug("Early pruning is disabled, continuing benchmarking.")
                callables_to_benchmark = callables
        else:
            callables_to_benchmark = callables

        # see benchmark_gpu for details
        benchmark_iters = self.reduce_benchmark_iters_to_max_benchmark_duration_ms(
            memory_warmup_iters,
            benchmark_iters,
            cpu_launch_overhead_ms_per_iter,
            max(estimated_timings_ms),
            len(callables_to_benchmark),
            max_benchmark_duration_ms,
        )
        benchmark_blocks = (
            (benchmark_iters * 5 * len(callables_to_benchmark))
            // (self.gpu_queue_limit - 1)
        ) + 1
        benchmark_iters_per_block = benchmark_iters // benchmark_blocks
        memory_warmup_iters_per_block = memory_warmup_iters // benchmark_blocks
        required_gpu_sleep_cycles_per_block = self.get_required_gpu_sleep_cycles(
            memory_warmup_iters_per_block,
            benchmark_iters_per_block,
            cpu_launch_overhead_ms_per_iter,
        )

        # ensures the queue is empty
        torch.cuda.synchronize()
        interleaved_event_pairs = self.get_interleaved_event_pairs(
            len(callables_to_benchmark), benchmark_iters
        )
        for block_idx in range(benchmark_blocks):
            block_start = block_idx * benchmark_iters_per_block
            if block_idx == (benchmark_blocks - 1):
                block_end = benchmark_iters
            else:
                block_end = (block_idx + 1) * benchmark_iters_per_block
            torch.cuda._sleep(required_gpu_sleep_cycles_per_block)
            for _ in range(memory_warmup_iters_per_block):
                buffer.zero_()
            for event_pairs in interleaved_event_pairs[block_start:block_end]:
                for _callable, (start_event, end_event) in zip(callables_to_benchmark, event_pairs):
                    buffer.zero_()
                    start_event.record()
                    _callable()
                    end_event.record()
            torch.cuda.synchronize()
        timings_ms = self.get_interleaved_min_timings_ms(interleaved_event_pairs)

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor benchmarks
        del buffer

        # we may as well use the estimation loop signal as well
        for _callable, timing_ms in zip(callables_to_benchmark, timings_ms):
            callable_to_timing_ms[_callable] = min(
                callable_to_timing_ms[_callable], timing_ms
            )
        return [callable_to_timing_ms[_callable] for _callable in callables]

    def original_do_bench_many_gpu(
        self: Self,
        fns: List[Callable[[], Any]],
        warmup: int = 25,
        rep: int = 100,
        grad_to_none: Optional[torch.Tensor] = None,
        quantiles: Optional[Tuple[float, ...]] = (0.5, 0.2, 0.8),
        fast_flush: bool = True,
        return_mode: str = "mean",
        **kwargs: Any,
    ) -> List[float]:
        return [
            self.original_do_bench_gpu(
                fn,
                warmup=warmup,
                rep=rep,
                grad_to_none=grad_to_none,
                quantiles=quantiles,
                fast_flush=fast_flush,
                return_mode=return_mode,
                **kwargs,
            )
            for fn in fns
        ]

    @maybe_fallback_to_original_benchmarking("original_do_bench")
    @maybe_fallback_to_non_lazy_benchmarking("benchmark")
    def lazy_benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> Union[LazyBenchmark, float]:
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.lazy_benchmark_cpu(_callable, **kwargs)
        else:
            return self.lazy_benchmark_gpu(_callable, **kwargs)

    @time_and_log
    @maybe_fallback_to_original_benchmarking("original_do_bench_cpu")
    @maybe_fallback_to_non_lazy_benchmarking("benchmark_cpu")
    def lazy_benchmark_cpu(
        self: Self,
        _callable: Callable[[], Any],
        **kwargs: Any,
    ) -> Union[LazyBenchmark, float]:
        # should we just immediately benchmark on CPU?
        return LazyBenchmark(lambda: self.benchmark_cpu(_callable, **kwargs))

    @time_and_log
    @maybe_fallback_to_original_benchmarking("original_do_bench_gpu")
    @maybe_fallback_to_non_lazy_benchmarking("benchmark_gpu")
    def lazy_benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        **kwargs: Any,
    ) -> Union[LazyBenchmark, float]:
        # we should try the callable before queueing it for benchmarking, in
        # case it throws an exception. we could catch and handle any exception
        # later on, but some codepaths expect and handle certain exceptions
        _callable()
        torch.cuda.synchronize()

        # we want to group benchmarks based on the kwargs hash, this handles
        # grouping benchmarks by ranking keys and pruning keys, and also ensures
        # that we only benchmark callables that should run under the same conditions
        # with respect to warmup, benchmarking, etc.
        kwargs_hash = str(hash(tuple(sorted(kwargs.items()))))
        # we've seen that just hash(_callable) and the kwargs_hash are not enough to
        # differentiate callables; if _callable is something like a lambda, which then
        # goes out of scope and gets garbage collected, its memory address may be later
        # reused for a different _callable. if this is the case, the latter _callable would
        # incorrectly exist in the memory cache, which could lead to memory leaks if we
        # have a lazy benchmark grouping of one, because we would never remove _callable
        # from the lazy benchmark queue and as such any memory referenced by _callable
        # would remain allocated
        key = str(hash(_callable) + randint(-(2**100), 2**100)) + kwargs_hash
        self.kwargs_hash_to_futures_gpu.setdefault(kwargs_hash, []).append(
            (_callable, key)
        )

        def benchmark() -> float:
            # all but the first benchmark in a grouping of lazy benchmarks
            # should be cached in memory, so we should return that cached timing
            if key in self.memory_cache:
                return self.memory_cache[key]

            futures_gpu = self.kwargs_hash_to_futures_gpu.pop(kwargs_hash)
            callables, keys = zip(*futures_gpu)
            callables, keys = list(callables), list(keys)

            try:
                timings_ms = self.benchmark_many_gpu(callables, **kwargs)
            except Exception as e:  # noqa: TRY302
                raise e
            else:
                self.memory_cache.update(zip(keys, timings_ms))
                return self.memory_cache[key]
            finally:
                # we have seen cases where not explicitly deleting the GPU futures
                # can prevent the memory allocated for the callables from being
                # properly and timely cleaned up, which can have fatal interactions
                # in cudagraphs mode
                del futures_gpu

        return LazyBenchmark(benchmark)

    def reduce_benchmark_iters_to_max_benchmark_duration_ms(
        self: Self,
        memory_warmup_iters: int,
        benchmark_iters: int,
        cpu_launch_overhead_ms_per_iter: float,
        gpu_time_ms_per_iter: float,
        num_callables: int,
        max_benchmark_duration_ms: int,
    ) -> int:
        # use the full max_benchmark_duration_ms per callable
        total_allotted_time_ms = num_callables * max_benchmark_duration_ms
        # only count the CPU launch overhead for the memory warmup, since the
        # GPU time should be overlapped with the end of the CPU launch overhead
        # for the benchmark iterations
        memory_warmup_duration_ms = (
            memory_warmup_iters * self.cpu_launch_overhead_ms_per_gpu_cache_clear
        )
        allotted_time_ms_for_benchmark_iters = (
            total_allotted_time_ms - memory_warmup_duration_ms
        )
        # count both the CPU launch overhead and the GPU time per iteration,
        # since the CPU launch overhead should be overlapped by the GPU sleep
        # and the GPU time should occur during the synchronization
        time_ms_per_benchmark_iter = (
            cpu_launch_overhead_ms_per_iter + gpu_time_ms_per_iter
        )
        reduced_benchmark_iters = int(
            allotted_time_ms_for_benchmark_iters / time_ms_per_benchmark_iter
        )
        # we want the minimum of benchmark_iters (user-specified) and
        # reduced_benchmark_iters (derived from user-specified max_benchmark_duration_ms),
        # with an absolute minimum of 1
        reduced_benchmark_iters = max(min(benchmark_iters, reduced_benchmark_iters), 1)
        return reduced_benchmark_iters

    def get_required_gpu_sleep_cycles(
        self: Self,
        memory_warmup_iters: int,
        benchmark_iters: int,
        cpu_launch_overhead_ms_per_iter: float,
    ) -> int:
        cpu_launch_overhead_ms_for_memory_warmup = (
            memory_warmup_iters * self.cpu_launch_overhead_ms_per_gpu_cache_clear
        )
        cpu_launch_overhead_ms_for_benchmarking = (
            benchmark_iters * cpu_launch_overhead_ms_per_iter
        )
        total_cpu_launch_overhead_ms = (
            cpu_launch_overhead_ms_for_memory_warmup
            + cpu_launch_overhead_ms_for_benchmarking
        )
        # we want the GPU sleep to overlap the CPU launch overhead of
        # the memory warmup only, and not the GPU time. we can and should
        # have the actual memory warmup begin on the GPU before we finish
        # queueing the benchmark iterations because we are not timing the
        # memory warmups and can therefore save on some total duration
        required_gpu_sleep_duration_ms = total_cpu_launch_overhead_ms - (
            self.gpu_time_ms_per_gpu_cache_clear * memory_warmup_iters
        )
        required_gpu_sleep_cycles = max(
            int(required_gpu_sleep_duration_ms / self.gpu_time_ms_per_gpu_clock_cycle),
            0,
        )
        return required_gpu_sleep_cycles

    def get_event_pairs(
        self: Self, num_pairs: int
    ) -> List[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_pairs)
        ]

    def get_interleaved_event_pairs(
        self: Self, num_callables: int, num_pairs: int
    ) -> List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]:
        return [self.get_event_pairs(num_callables) for _ in range(num_pairs)]

    def get_min_timing_ms(
        self: Self, event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]
    ) -> float:
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    def get_interleaved_min_timings_ms(
        self: Self,
        interleaved_event_pairs: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]],
    ) -> List[float]:
        return [
            self.get_min_timing_ms(list(event_pairs))
            for event_pairs in zip(*interleaved_event_pairs)
        ]


benchmarker = Benchmarker()
