import functools
import random
import time
from functools import cached_property
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch._dynamo.utils import counters
from torch._inductor.config import benchmarking as benchmarking_config
from torch._inductor.utils import is_cpu_device


log = torch._logging.getArtifactLogger(__name__, "benchmarking")


def time_and_log(fn: Callable[..., Any]) -> Callable[..., Any]:
    if not torch._logging._internal.log_state.is_artifact_enabled("benchmarking"):
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        log.debug(f"{fn.__name__} took {time.perf_counter() - start_time} seconds.")
        return result
    return wrapper


class LazyBenchmark:
    def __init__(self, benchmark: Callable[[], float]) -> None:
        self.benchmark = benchmark

    @cached_property
    def timing_ms(self) -> float:
        counters["inductor"]["benchmarking_finalize_lazy_benchmark"] += 1
        timing_ms = self.benchmark()
        # I don't think this helps with saving memory at all,
        # but at least it gives good signal if we ever try
        # to call self.benchmark again
        del self.benchmark
        return timing_ms

    def __float__(self) -> float:
        return float(self.timing_ms)
    
    def __format__(self, format_spec: str) -> str:
        return format(self.timing_ms, format_spec)
    
    def __str__(self) -> str:
        return str(self.timing_ms)
    
    def __lt__(self, other: Any) -> bool:
        return self.timing_ms < other
    
    def __le__(self, other: Any) -> bool:
        return self.timing_ms <= other
    
    def __gt__(self, other: Any) -> bool:
        return self.timing_ms > other
    
    def __ge__(self, other: Any) -> bool:
        return self.timing_ms >= other
    
    def __add__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms + other
        return LazyBenchmark(lambda: self.timing_ms + other)
    
    def __radd__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other + self.timing_ms
        return LazyBenchmark(lambda: other + self.timing_ms)
    
    def __sub__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms - other
        return LazyBenchmark(lambda: self.timing_ms - other)
    
    def __rsub__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other - self.timing_ms
        return LazyBenchmark(lambda: other - self.timing_ms)
    
    def __mul__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms * other
        return LazyBenchmark(lambda: self.timing_ms * other)
    
    def __rmul__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other * self.timing_ms
        return LazyBenchmark(lambda: other * self.timing_ms)
    
    def __truediv__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return self.timing_ms / other
        return LazyBenchmark(lambda: self.timing_ms / other)
    
    def __rtruediv__(self, other: Any) -> Any:
        if not hasattr(self, "benchmark"):
            return other / self.timing_ms
        return LazyBenchmark(lambda: other / self.timing_ms)


class Benchmarker:
    def __init__(self) -> None:
        self.memory_cache: Dict[str, float] = {}
        self.kwargs_hash_to_futures_gpu: Dict[
            str, Tuple[Callable[[], Any], str]
        ] = {}

    @cached_property
    def L2_cache_size(self) -> int:
        counters["inductor"]["benchmarking_L2_cache_size"] += 1
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.L2_cache_size

    @cached_property
    def gpu_queue_limit(self) -> int:
        counters["inductor"]["benchmarking_gpu_queue_limit"] += 1
        # ensures the queue is empty
        torch.cuda.synchronize()
        # capping the search space for the queue limit to 2500 is good enough,
        # current queue limit on my machine (A100) is stable at ~1000, and
        # in the event that we do artificially cap the queue limit to 2500
        # we really shouldn't see any significant slowdowns
        torch.cuda._sleep(
            int(
                (self.get_cpu_launch_overhead_ms_per_event_record() * 2500)
                / self.gpu_time_ms_per_gpu_clock_cycle
            )
        )
        for idx in range(2500):
            start_time_s = time.perf_counter()
            torch.cuda.Event(enable_timing=True).record()
            elapsed_time_ms = (time.perf_counter() - start_time_s) * 1000
            # recording an event is near instantaneous, unless we have hit
            # the queue limit, so 1ms seems like a good enough upper bound
            if elapsed_time_ms > 1:
                break
        torch.cuda.synchronize()
        return idx

    @functools.lru_cache(None)
    def get_cpu_launch_overhead_ms_per_event_record(self) -> float:
        counters["inductor"]["benchmarking_get_cpu_launch_overhead_ms_per_event_record"] += 1
        # ensures the queue is empty
        torch.cuda.synchronize()
        start_time_s = time.perf_counter()
        for _ in range(100):
            torch.cuda.Event(enable_timing=True).record()
        torch.cuda.synchronize()
        return ((time.perf_counter() - start_time_s) * 1000) / 100

    @cached_property
    def cpu_launch_overhead_ms_per_gpu_cache_clear(self) -> float:
        counters["inductor"]["benchmarking_cpu_launch_overhead_ms_per_gpu_cache_clear"] += 1
        return self.get_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear()[0]

    @cached_property
    def gpu_time_ms_per_gpu_cache_clear(self) -> float:
        counters["inductor"]["benchmarking_gpu_time_ms_per_gpu_cache_clear"] += 1
        return self.get_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear()[1]

    @functools.lru_cache(None)
    def get_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear(
        self,
    ) -> Tuple[float, float]:
        counters["inductor"]["benchmarking_get_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear"] += 1
        buffer = torch.empty(
            int(self.L2_cache_size // 4), dtype=torch.int, device="cuda"
        )
        buffer.zero_()
        # synchronize after zeroing the buffer to reduce uncertainty
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start_time_s = time.perf_counter()
        # 100 buffer zeroes is long enough to reduce uncertainty
        for _ in range(100):
            buffer.zero_()
        cpu_launch_overhead_ms_per_gpu_cache_clear = (
            (time.perf_counter() - start_time_s) * 1000
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
    def gpu_time_ms_per_gpu_clock_cycle(self) -> float:
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

    def benchmark(
        self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any
    ) -> float:
        counters["inductor"]["benchmarking_benchmark"] += 1
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        # should we be checking if all args and kwargs are on the same device?
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)

    @time_and_log
    def benchmark_cpu(
        self,
        _callable: Callable[[], Any],
        warmup_iters: int = 5,
        benchmark_iters: int = 20,
    ) -> float:
        counters["inductor"]["benchmarking_benchmark_cpu"] += 1
        # duplicate of original implementation from runtime_utils
        timings_ms = []
        for _ in range(warmup_iters):
            _callable()
        for _ in range(benchmark_iters):
            start_time_s = time.perf_counter()
            _callable()
            timings_ms.append((time.perf_counter() - start_time_s) * 1000)
        return median(timings_ms)

    @time_and_log
    def benchmark_many_cpu(
        self,
        callables: List[Callable[[], Any]],
        warmup_iters: int = 5,
        benchmark_iters: int = 20,
    ) -> List[float]:
        counters["inductor"]["benchmarking_benchmark_many_cpu"] += 1
        # implement this to maintain consistency between cpu/gpu benchmarking functionality
        return [
            self.benchmark_cpu(_callable, warmup_iters, benchmark_iters)
            for _callable in callables
        ]

    @time_and_log
    def benchmark_gpu(
        self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration_ms: int = 25,
    ) -> float:
        counters["inductor"]["benchmarking_benchmark_gpu"] += 1
        # we don't want any outside errors propagating into benchmarking
        torch.cuda.synchronize()

        # initialize here, avoids double buffer allocation
        self.get_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear()

        _callable()
        torch.cuda.synchronize()

        buffer = torch.empty(
            int(self.L2_cache_size // 4), dtype=torch.int, device="cuda"
        )
        buffer.zero_()

        event_pairs = self.get_event_pairs(estimation_iters)
        start_time_s = time.perf_counter()
        for start_event, end_event in event_pairs:
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        # before we synchronize we want to measure the cpu-side cost of our buffer
        # zero, launching the callable, and the associated event records
        cpu_launch_overhead_ms_per_iter = (
            (time.perf_counter() - start_time_s) * 1000
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

    @time_and_log
    def benchmark_many_gpu(
        self,
        callables: List[Callable[[], Any]],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration_ms: int = 25,
        ranking_key: Optional[str] = None,
        pruning_key: Optional[str] = None,
        pruning_factor: float = 1.25,
    ) -> List[float]:
        counters["inductor"]["benchmarking_benchmark_many_gpu"] += 1
        # we don't want any outside errors propagating into benchmarking
        torch.cuda.synchronize()

        # initialize here, avoids double buffer allocation
        self.get_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear()

        # this will fail the entire group if any of the callables fail,
        # in the case that users directly call this method they should
        # handle this case on their end. in the case that we get here
        # from a grouping of lazy benchmarks, we should never have failures
        # since they will have already been checked previously
        for _callable in callables:
            _callable()
        torch.cuda.synchronize()

        buffer = torch.empty(
            int(self.L2_cache_size // 4), dtype=torch.int, device="cuda"
        )
        buffer.zero_()

        interleaved_event_pairs = self.get_interleaved_event_pairs(
            len(callables), estimation_iters
        )
        start_time_s = time.perf_counter()
        for event_pairs in interleaved_event_pairs:
            for _callable, (start_event, end_event) in zip(callables, event_pairs):
                buffer.zero_()
                start_event.record()
                _callable()
                end_event.record()
        cpu_launch_overhead_ms_per_iter = (
            (time.perf_counter() - start_time_s) * 1000
        ) / estimation_iters
        torch.cuda.synchronize()
        estimated_timings_ms = self.get_interleaved_min_timings_ms(
            interleaved_event_pairs
        )

        if ranking_key is not None:
            if benchmarking_config.enable_early_ranking:
                counters["inductor"]["benchmarking_early_ranking"] += 1
                log.debug(f"Returning early ranking for ranking key {ranking_key}.")
                # explicitly delete the buffer, sometimes helps memory
                # footprint metrics in OSS Inductor benchmarks
                del buffer
                return estimated_timings_ms
            else:
                log.debug("Early ranking is disabled. Continuing full benchmarking cycle.")

        callable_to_timing_ms = {}
        for _callable, estimated_timing_ms in zip(callables, estimated_timings_ms):
            callable_to_timing_ms[_callable] = estimated_timing_ms


        if pruning_key is not None:
            if benchmarking_config.enable_early_pruning:
                counters["inductor"]["benchmarking_early_pruning"] += 1
                cpu_launch_overhead_ms_per_iter_per_callable = (
                    cpu_launch_overhead_ms_per_iter / len(callables)
                )
                target_timing_ms = min(estimated_timings_ms) * pruning_factor
                callables_to_benchmark = [
                    _callable
                    for _callable in callables
                    if callable_to_timing_ms[_callable] < target_timing_ms
                ]
                log.debug(f"Pruned to {len(callables_to_benchmark)} callables for pruning key {pruning_key}.")
                cpu_launch_overhead_ms_per_iter = (
                    cpu_launch_overhead_ms_per_iter_per_callable
                    * len(callables_to_benchmark)
                )
                estimated_timings_ms = [
                    callable_to_timing_ms[_callable] for _callable in callables_to_benchmark
                ]
            else:
                log.debug("Early pruning is disabled. Continuing full benchmarking cycle.")
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
                for _callable, (start_event, end_event) in zip(
                    callables_to_benchmark, event_pairs
                ):
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

    def lazy_benchmark(
        self,
        fn: Callable[..., Any],
        fn_args: Tuple[Any],
        fn_kwargs: Dict[str, Any],
        **kwargs: Any
    ) -> LazyBenchmark:
        counters["inductor"]["benchmarking_lazy_benchmark"] += 1
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        if not benchmarking_config.enable_lazy_benchmarking:
            log.debug("Lazy benchmarking is disabled. Immediately proceeding to CPU benchmarking.")
            return self.benchmark_cpu(_callable, **kwargs)
        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.lazy_benchmark_cpu(_callable, **kwargs)
        else:
            return self.lazy_benchmark_gpu(_callable, **kwargs)

    @time_and_log
    def lazy_benchmark_cpu(
        self,
        _callable: Callable[[], Any],
        ranking_key: Optional[str] = None,
        pruning_key: Optional[str] = None,
        **kwargs: Any
    ) -> Union[LazyBenchmark, float]:
        counters["inductor"]["benchmarking_lazy_benchmark_cpu"] += 1
        if not benchmarking_config.enable_lazy_benchmarking:
            log.debug("Lazy benchmarking is disabled. Immediately proceeding to CPU benchmarking.")
            return self.benchmark_cpu(_callable, **kwargs)
        # should we just immediately benchmark on CPU?
        return LazyBenchmark(lambda: self.benchmark_cpu(_callable, **kwargs))

    @time_and_log
    def lazy_benchmark_gpu(
        self,
        _callable: Callable[[], Any],
        ranking_key: Optional[str] = None,
        pruning_key: Optional[str] = None,
        **kwargs: Any
    ) -> LazyBenchmark:
        counters["inductor"]["benchmarking_lazy_benchmark_gpu"] += 1

        if not benchmarking_config.enable_lazy_benchmarking:
            log.debug("Lazy benchmarking is disabled. Immediately proceeding to GPU benchmarking.")
            return self.benchmark_gpu(_callable, **kwargs)

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
        key = str(hash(_callable) + random.randint(-(2**100), 2**100)) + kwargs_hash
        self.kwargs_hash_to_futures_gpu[
            kwargs_hash
        ] = self.kwargs_hash_to_futures_gpu.get(kwargs_hash, []) + [(_callable, key)]

        futures_gpu = self.kwargs_hash_to_futures_gpu.pop(kwargs_hash)
        del futures_gpu

        return self.benchmark_gpu(_callable, **kwargs)

        def benchmark() -> float:
            # all but the first benchmark in a grouping of lazy benchmarks
            # should be cached in memory, so we should return that cached timing
            if key in self.memory_cache:
                return self.memory_cache[key]

            futures_gpu = self.kwargs_hash_to_futures_gpu.pop(kwargs_hash)
            callables, keys = zip(*futures_gpu)
            callables, keys = list(callables), list(keys)
            timings_ms = self.benchmark_many_gpu(
                callables, ranking_key=ranking_key, pruning_key=pruning_key, **kwargs
            )
            self.memory_cache.update(zip(keys, timings_ms))
            # we may or may not have to delete the futures explicitly to
            # cleanup the memory, just do it now for safety
            del futures_gpu
            return self.memory_cache[key]

        return LazyBenchmark(benchmark)

    def reduce_benchmark_iters_to_max_benchmark_duration_ms(
        self,
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
        self,
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
        required_gpu_sleep_cycles = int(
            required_gpu_sleep_duration_ms / self.gpu_time_ms_per_gpu_clock_cycle
        )
        return required_gpu_sleep_cycles

    def get_event_pairs(
        self, num_pairs: int
    ) -> List[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_pairs)
        ]

    def get_interleaved_event_pairs(
        self, num_callables: int, num_pairs: int
    ) -> List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]:
        return [self.get_event_pairs(num_callables) for _ in range(num_pairs)]

    def get_min_timing_ms(
        self, event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]
    ) -> float:
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    def get_interleaved_min_timings_ms(
        self,
        interleaved_event_pairs: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]],
    ) -> List[float]:
        return [
            self.get_min_timing_ms(event_pairs)  # type: ignore
            for event_pairs in zip(*interleaved_event_pairs)
        ]


benchmarker = Benchmarker()
