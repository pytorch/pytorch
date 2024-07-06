import functools
import time
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch._inductor.utils import is_cpu_device


class Benchmarker:
    def __init__(self) -> None:
        pass

    def benchmark(self, fn: Callable[..., Any], fn_args: List[Any], fn_kwargs: Dict[str, Any], **kwargs: Dict[str, Any]) -> float:
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)
    
    def benchmark_cpu(self, _callable: Callable[[], Any], warmup_iters: int = 5, benchmark_iters: int = 20) -> float:
        timings = []

        for _ in range(warmup_iters):
            _callable()
        for _ in range(benchmark_iters):
            start_time = time.perf_counter()
            _callable()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)
        
        sorted_timings = sorted(timings)
        if benchmark_iters % 2  == 0:
            lower_timing = sorted_timings[(benchmark_iters // 2) - 1]
            upper_timing = sorted_timings[benchmark_iters // 2]
            benchmark = (lower_timing + upper_timing) / 2
        else:
            benchmark = sorted_timings[benchmark_iters // 2]

        return benchmark
    
    def benchmark_gpu(self, _callable: Callable[[], Any], estimation_iters: int = 5, memory_warmup_iters: int = 1000, benchmark_iters: int = 100, max_benchmark_duration: int = 25) -> float:        
        def benchmark(buffer, _callable, iters, measure_launch_overhead=False):
            event_pairs = self.get_event_pairs(iters)
            if measure_launch_overhead:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            for start_event, end_event in event_pairs:
                buffer.zero_()
                start_event.record()
                _callable()
                end_event.record()
            if measure_launch_overhead:
                end_time = time.perf_counter()
            torch.cuda.synchronize()
            if measure_launch_overhead:
                return self.get_min_timing(event_pairs), end_time - start_time
            return self.get_min_timing(event_pairs)
        
        def get_required_sleep_cycles(launch_overhead, memory_warmup_iters, benchmark_iters) -> int:
            total_overhead = (launch_overhead * benchmark_iters) + (self.get_launch_overhead_per_buffer_clear() * memory_warmup_iters)
            required_sleep_cycles = (total_overhead / self.get_time_per_million_sleep_cycles()) * 1000000
            required_sleep_cycles *= 1.5
            return required_sleep_cycles

        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")

        estimated_timing, launch_overhead = benchmark(buffer, _callable, estimation_iters, measure_launch_overhead=True)
        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / estimated_timing), 1))

        required_sleep_cycles = get_required_sleep_cycles(launch_overhead, memory_warmup_iters, benchmark_iters)
        torch.cuda._sleep(required_sleep_cycles)

        for _ in range(memory_warmup_iters):
            buffer.zero_()
        timing = benchmark(buffer, _callable, benchmark_iters)
        
        del buffer

        return time

    @functools.lru_cache(None)
    def get_cache_size(self) -> int:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.l2CacheSize
    
    @functools.lru_cache(None)
    def get_time_per_million_sleep_cycles(self) -> float:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        torch.cuda._sleep(1000000)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time
    
    @functools.lru_cache(None)
    def get_launch_overhead_per_buffer_clear(self) -> float:
        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(100):
            buffer.zero_()
        end_time = time.perf_counter()
        torch.cuda.synchronize()
        del buffer
        return (end_time - start_time) / 100

    def get_event_pairs(self, iters: int) -> List[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_min_timing(self, event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
        return min([start_event.elapsed_time(end_event) for start_event, end_event in event_pairs])

    
benchmarker = Benchmarker()