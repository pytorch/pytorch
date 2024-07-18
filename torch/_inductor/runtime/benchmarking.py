import functools
import time
from collections import defaultdict
from functools import cached_property
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
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        log.debug(f"{fn.__name__} took {time.perf_counter() - start_time} seconds.")
        return result
    return wrapper


class LazyBenchmark:
    def __init__(self, benchmark: Callable[[], float]) -> None:
        self.benchmark = benchmark

    @cached_property
    def timing(self) -> float:
        counters["inductor"]["benchmarking_finalize_lazy_benchmark"] += 1
        timing = self.benchmark()
        del self.benchmark
        return timing
    
    __float__ = lambda self: self.timing
    __format__ = lambda self, format_spec: format(self.timing, format_spec)
    __str__ = lambda self: str(self.timing)
    
    __lt__ = lambda self, other: other > self.timing
    __le__ = lambda self, other: other >= self.timing

    __gt__ = lambda self, other: other < self.timing
    __ge__ = lambda self, other: other <= self.timing
    
    __add__ = lambda self, other: LazyBenchmark(lambda: self.timing + other)
    __radd__ = lambda self, other: LazyBenchmark(lambda: other + self.timing)

    __sub__ = lambda self, other: LazyBenchmark(lambda: self.timing - other)
    __rsub__ = lambda self, other: LazyBenchmark(lambda: other - self.timing)

    __mul__ = lambda self, other: LazyBenchmark(lambda: self.timing * other)
    __rmul__ = lambda self, other: LazyBenchmark(lambda: other * self.timing)

    __truediv__ = lambda self, other: LazyBenchmark(lambda: self.timing / other)
    __rtruediv__ = lambda self, other: LazyBenchmark(lambda: other / self.timing)


class Benchmarker:
    def __init__(self) -> None:
        self.memory_cache: Dict[str, Optional[float]] = defaultdict(lambda: None)
        self.kwargs_hash_to_futures_gpu: Dict[str, Tuple[LazyBenchmark, Callable[..., Any]]] = {}
  
    @cached_property
    def L2_cache_size(self) -> int:
        counters["inductor"]["benchmarking_L2_cache_size"] += 1
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.L2_cache_size

    @cached_property
    def gpu_time_per_gpu_clock_cycle(self) -> float:
        counters["inductor"]["benchmarking_gpu_time_per_gpu_clock_cycle"] += 1
        start_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1000000)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / 1000000

    @functools.lru_cache(None)
    def get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear(self) -> Tuple[float, float]:
        counters["inductor"]["benchmarking_get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear"] += 1
        buffer = torch.empty(int(self.L2_cache_size // 4), dtype=torch.int, device="cuda")
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        start_time = time.perf_counter()
        for _ in range(100):
            buffer.zero_()
        cpu_launch_overhead = ((time.perf_counter() - start_time) / 1000) / 100
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        del buffer
        return cpu_launch_overhead, start_event.elapsed_time(end_event)
    
    @cached_property
    def cpu_launch_overhead_per_gpu_cache_clear(self) -> float:
        counters["inductor"]["benchmarking_cpu_launch_overhead_per_gpu_cache_clear"] += 1
        return self.get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear()[0]
    
    @cached_property
    def gpu_time_per_gpu_cache_clear(self) -> float:
        counters["inductor"]["benchmarking_gpu_time_per_gpu_cache_clear"] += 1
        return self.get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear()[1]

    def benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> float:
        counters["inductor"]["benchmarking_benchmark"] += 1
        _callable = lambda: fn(*fn_args, **fn_kwargs)
        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)
    
    @time_and_log
    def benchmark_cpu(self, _callable: Callable[[], Any], warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
        counters["inductor"]["benchmarking_benchmark_cpu"] += 1

        def benchmark(_callable, iters):
            timings = []
            for _ in range(iters):
                start_time = time.perf_counter()
                _callable()
                timings.append((time.perf_counter() - start_time) * 1000)
            return timings
        
        def get_median_timing(timings):
            timings = sorted(timings)
            if ((len(timings) % 2) == 0):
                lower_timing = timings[(len(timings) // 2) - 1]
                upper_timing = timings[len(timings) // 2]
                median_timing = (lower_timing + upper_timing) / 2
            else:
                median_timing = timings[len(timings) // 2]
            return median_timing

        for _ in range(warmup_iters):
            _callable()
        timings = benchmark(_callable, benchmark_iters)

        timing = get_median_timing(timings)
        return timing
    
    @time_and_log
    def benchmark_many_cpu(self, callables: List[Callable[[], Any]], warmup_iters: int = 5, benchmark_iters: int = 20) -> List[float]:
        counters["inductor"]["benchmarking_benchmark_many_cpu"] += 1
        return [self.benchmark_cpu(_callable, warmup_iters, benchmark_iters) for _callable in callables]
    
    @time_and_log
    def benchmark_gpu(self, _callable: Callable[[], Any], estimation_iters: int = 5, memory_warmup_iters: int = 100, benchmark_iters: int = 100, max_benchmark_duration: int = 25) -> float:        
        counters["inductor"]["benchmarking_benchmark_gpu"] += 1
        
        self.get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear()
        
        def benchmark(_callable, buffer, iters):
            event_pairs = self.get_event_pairs(iters)
            start_time = time.perf_counter()
            for start_event, end_event in event_pairs:
                buffer.zero_()
                start_event.record()
                _callable()
                end_event.record()
            cpu_launch_overhead_per_iter = ((time.perf_counter() - start_time) * 1000) / iters
            torch.cuda.synchronize()
            timing = self.get_min_timing(event_pairs)
            return timing, cpu_launch_overhead_per_iter
                
        try:
            _callable()
            torch.cuda.synchronize()
        except Exception as e:
            counters["inductor"]["benchmarking_callable_initialization_failed"] += 1
            log.debug(f"Callable {hash(_callable)} failed during initialization with exception {e}.")
            return float("inf")

        buffer = torch.empty(int(self.L2_cache_size // 4), dtype=torch.int, device="cuda")
        buffer.zero_()

        estimated_timing, cpu_launch_overhead_per_iter = benchmark(_callable, buffer, estimation_iters)
        benchmark_iters = self.get_reduced_benchmark_iters(memory_warmup_iters, benchmark_iters, cpu_launch_overhead_per_iter, estimated_timing, 1, max_benchmark_duration)

        required_gpu_sleep_cycles = self.get_required_gpu_sleep_cycles(memory_warmup_iters, benchmark_iters, cpu_launch_overhead_per_iter)
        torch.cuda._sleep(required_gpu_sleep_cycles)

        for _ in range(memory_warmup_iters):
            buffer.zero_()
        timing, _ = benchmark(_callable, buffer, benchmark_iters)
        
        del buffer

        return timing
    
    @time_and_log
    def benchmark_many_gpu(self, callables: List[Callable[[], Any]], estimation_iters: int = 5, memory_warmup_iters: int = 100, benchmark_iters: int = 100, max_benchmark_duration: int = 25, ranking_key: Optional[str] = None, pruning_key: Optional[str] = None) -> List[float]:        
        counters["inductor"]["benchmarking_benchmark_many_gpu"] += 1
        
        self.get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear()
        
        def benchmark(callables, buffer, iters):
            interleaved_event_pairs = self.get_interleaved_event_pairs(len(callables), iters)
            start_time = time.perf_counter()
            for event_pairs in interleaved_event_pairs:
                for _callable, (start_event, end_event) in zip(callables, event_pairs):
                    buffer.zero_()
                    start_event.record()
                    _callable()
                    end_event.record()
            cpu_launch_overhead_per_iter = ((time.perf_counter() - start_time) * 1000) / iters
            torch.cuda.synchronize()
            timings = self.get_interleaved_min_timings(interleaved_event_pairs)
            return timings, cpu_launch_overhead_per_iter

        callable_to_timing = {}
        callables_to_benchmark = []

        for _callable in callables:
            try:
                _callable()
                torch.cuda.synchronize()
            except Exception as e:
                counters["inductor"]["benchmarking_callable_initialization_failed"] += 1
                log.debug(f"Callable {hash(_callable)} failed during initialization with exception {e}.")
                callable_to_timing[_callable] = float("inf")
            else:
                callables_to_benchmark.append(_callable)
        
        if len(callables_to_benchmark) == 0:
            timings = [callable_to_timing[_callable] for _callable in callables]
            return timings
    
        buffer = torch.empty(int(self.L2_cache_size // 4), dtype=torch.int, device="cuda")
        buffer.zero_()

        estimated_timings, cpu_launch_overhead_per_iter = benchmark(callables_to_benchmark, buffer, estimation_iters)

        for _callable, estimated_timing in zip(callables_to_benchmark, estimated_timings):
            callable_to_timing[_callable] = estimated_timing

        if ranking_key is not None:
            if benchmarking_config.enable_early_ranking:
                counters["inductor"]["benchmarking_early_ranking"] += 1
                log.debug(f"Returning early ranking for ranking key {ranking_key}.")
                timings = [callable_to_timing[_callable] for _callable in callables]
                del buffer
                return timings
            else:
                log.debug("Early ranking is disabled. Continuing full benchmarking cycle.")

        if pruning_key is not None:
            if benchmarking_config.enable_early_pruning:
                counters["inductor"]["benchmarking_early_pruning"] += 1
                cpu_launch_overhead_per_iter_per_callable = cpu_launch_overhead_per_iter / len(callables_to_benchmark)
                target_timing = min(estimated_timings) * 1.25
                callables_to_benchmark = [_callable for _callable in callables_to_benchmark if callable_to_timing[_callable] < target_timing]
                log.debug(f"Pruned to {len(callables_to_benchmark)} callables for pruning key {pruning_key}.")
                cpu_launch_overhead_per_iter = cpu_launch_overhead_per_iter_per_callable * len(callables_to_benchmark)
                estimated_timings = [callable_to_timing[_callable] for _callable in callables_to_benchmark]
            else:
                log.debug("Early pruning is disabled. Continuing full benchmarking cycle.")

        benchmark_iters = self.get_reduced_benchmark_iters(memory_warmup_iters, benchmark_iters, cpu_launch_overhead_per_iter, sum(estimated_timings), len(callables_to_benchmark), max_benchmark_duration)

        required_gpu_sleep_cycles = self.get_required_gpu_sleep_cycles(memory_warmup_iters, benchmark_iters, cpu_launch_overhead_per_iter)
        torch.cuda._sleep(required_gpu_sleep_cycles)

        for _ in range(memory_warmup_iters):
            buffer.zero_()
        timings, _ = benchmark(callables_to_benchmark, buffer, benchmark_iters)
        
        del buffer

        for _callable, timing in zip(callables_to_benchmark, timings):
            callable_to_timing[_callable] = min(callable_to_timing[_callable], timing)
        timings = [callable_to_timing[_callable] for _callable in callables]
        return timings

    def lazy_benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> LazyBenchmark:
        counters["inductor"]["benchmarking_lazy_benchmark"] += 1

        _callable = lambda: fn(*fn_args, **fn_kwargs)
        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.lazy_benchmark_cpu(_callable, **kwargs)
        else:
            return self.lazy_benchmark_gpu(_callable, **kwargs)
    
    def lazy_benchmark_cpu(self, _callable: Callable[[], Any], ranking_key: Optional[str] = None, pruning_key: Optional[str] = None, **kwargs: Dict[str, Any]) -> Union[LazyBenchmark, float]:
        counters["inductor"]["benchmarking_lazy_benchmark_cpu"] += 1
        if not benchmarking_config.enable_lazy_benchmarking:
            log.debug("Lazy benchmarking is disabled. Immediately proceeding to CPU benchmarking.")
            return self.benchmark_cpu(_callable, **kwargs)
        return LazyBenchmark(lambda: self.benchmark_cpu(_callable, **kwargs))

    def lazy_benchmark_gpu(self, _callable: Callable[[], Any], ranking_key: Optional[str] = None, pruning_key: Optional[str] = None, **kwargs: Dict[str, Any]) -> LazyBenchmark:
        counters["inductor"]["benchmarking_lazy_benchmark_gpu"] += 1

        if not benchmarking_config.enable_lazy_benchmarking:
            log.debug("Lazy benchmarking is disabled. Immediately proceeding to GPU benchmarking.")
            return self.benchmark_gpu(_callable, **kwargs)

        kwargs_hash = hash(tuple(sorted(kwargs.items())))
        key = hash(_callable) + kwargs_hash
        self.kwargs_hash_to_futures_gpu[kwargs_hash] = self.kwargs_hash_to_futures_gpu.get(kwargs_hash, []) + [(_callable, key)]

        def initialize() -> float:
            if key in self.memory_cache:
                return self.memory_cache[key]

            futures_gpu = self.kwargs_hash_to_futures_gpu.pop(kwargs_hash)
            callables, keys = zip(*futures_gpu)
            callables, keys = list(callables), list(keys)
            timings = self.benchmark_many_gpu(callables, ranking_key=ranking_key, pruning_key=pruning_key, **kwargs)
            self.memory_cache.update(zip(keys, timings))
            del callables
            return self.memory_cache[key]
        
        return LazyBenchmark(initialize)

    def get_reduced_benchmark_iters(self, memory_warmup_iters: int, benchmark_iters: int, cpu_launch_overhead_per_iter: float, gpu_time_per_iter: float, num_callables: int, max_benchmark_duration: int) -> int:
        # client-prescribed maximum allotted benchmarking duration
        total_allotted_time = num_callables * max_benchmark_duration

        # adjusting benchmarking duration to account for total memory warmup duration (cpu launch overhead + gpu time)
        memory_warmup_duration = memory_warmup_iters * (self.get_cpu_launch_overhead_per_gpu_cache_clear() + self.get_gpu_time_per_gpu_cache_clear())
        allotted_time_for_benchmark_iters = total_allotted_time - memory_warmup_duration

        # calculate reduced benchmark iters based on remaining allotted time
        time_per_benchmark_iter = cpu_launch_overhead_per_iter + gpu_time_per_iter
        reduced_benchmark_iters = int(allotted_time_for_benchmark_iters / time_per_benchmark_iter)

        reduced_benchmark_iters = max(min(benchmark_iters, reduced_benchmark_iters), 1)
        return reduced_benchmark_iters
    
    def get_required_gpu_sleep_cycles(self, memory_warmup_iters: int, benchmark_iters: int, cpu_launch_overhead_per_iter: float) -> int:
        # calculate the total cpu launch overhead including memory warmup and benchmarking stages
        cpu_launch_overhead_for_memory_warmup = memory_warmup_iters * self.get_cpu_launch_overhead_per_gpu_cache_clear()
        cpu_launch_overhead_for_benchmarking = benchmark_iters * cpu_launch_overhead_per_iter
        total_cpu_launch_overhead = cpu_launch_overhead_for_memory_warmup + cpu_launch_overhead_for_benchmarking

        # calculate required gpu sleep cycles to approximately match the total cpu launch overhead
        required_gpu_sleep_cycles = int(total_cpu_launch_overhead / self.gpu_time_per_gpu_clock_cycle)
        return required_gpu_sleep_cycles

    def get_event_pairs(self, num_pairs: int) -> List[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_pairs)
        ]
    
    def get_interleaved_event_pairs(self, num_callables: int, num_pairs: int) -> List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]:
        return [self.get_event_pairs(num_callables) for _ in range(num_pairs)]

    def get_min_timing(self, event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
        return min([start_event.elapsed_time(end_event) for start_event, end_event in event_pairs])
    
    def get_interleaved_min_timings(self, interleaved_event_pairs: List[List[Tuple[torch.cuda.Event, torch.cuda.Event]]]) -> float:
        return [self.get_min_timing(event_pairs) for event_pairs in zip(*interleaved_event_pairs)]

    
benchmarker = Benchmarker()
