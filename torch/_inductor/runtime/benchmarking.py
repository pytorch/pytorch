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
        def time(buffer, _callable, iters):
            event_pairs = self.get_event_pairs(iters)
            for start_event, end_event in event_pairs:
                    buffer.zero_()
                    start_event.record()
                    _callable()
                    end_event.record()
            torch.cuda.synchronize()
            return self.get_min_timing(event_pairs)
       
        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")

        estimated_timing = time(buffer, _callable, estimation_iters)
        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / estimated_timing), 1))

        for _ in range(memory_warmup_iters):
            buffer.zero_()
        benchmark = time(buffer, _callable, benchmark_iters)
        
        del buffer

        return benchmark

    @functools.lru_cache(None)
    def get_cache_size(self) -> int:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.l2CacheSize

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