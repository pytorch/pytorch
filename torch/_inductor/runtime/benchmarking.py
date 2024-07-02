import functools
import time
from pathlib import Path

import torch
from torch._inductor import config as inductor_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import is_cpu_device


class Benchmarker:
    def __init__(self):
        self.futures_gpu = {}
        self.kwargs_hash_to_kwargs = {}
    
    def benchmark(self, fn, fn_args, fn_kwargs, **kwargs):
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.benchmark_gpu(_callable, **kwargs)
    
    def benchmark_cpu(self, _callable, key=None, warmup_iters=5, benchmark_iters=20):
        cached_benchmark = self.get_cached_benchmark(key)
        if cached_benchmark != None:
            return cached_benchmark

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
        
        if key != None:
            self.put_cached_benchmark(key, benchmark)
        
        return benchmark
    
    def benchmark_many_cpu(self, callables, keys=[], warmup_iters=5, benchmark_iters=20):
        if keys == []:
            benchmarks = [self.benchmark_cpu(_callable, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters) for _callable in callables]
        else:
            assert len(callables) == len(keys)
            benchmarks = []
            for _callable, key in zip(callables, keys):
                benchmarks.append(self.benchmark_cpu(_callable, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters, key=key))
        return benchmarks
    
    def benchmark_gpu(self, _callable, key=None, estimation_iters=5, groups=5, memory_warmup_iters=1000, benchmark_iters=100, max_benchmark_duration=25):
        print("benchmarking single callable")
        
        cached_benchmark = self.get_cached_benchmark(key)
        if cached_benchmark != None:
            return cached_benchmark
        
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
        benchmark_iters_per_group = max(benchmark_iters // groups, 1)

        per_group_timing = []
        for _ in range(groups):
            for _ in range(memory_warmup_iters):
                buffer.zero_()
            per_group_timing.append(time(buffer, _callable, benchmark_iters_per_group))
        benchmark = min(per_group_timing)
        
        del buffer

        if key != None:
            self.put_cached_benchmark(key, benchmark)

        return benchmark
    
    def benchmark_many_gpu(self, callables, keys=[], estimation_iters=5, groups=5, memory_warmup_iters=1000, benchmark_iters=100, max_benchmark_duration=25):
        print(f"benchmarking {len(callables)} callables")
        
        if keys != []:
            assert len(callables) == len(keys)
            cached_benchmarks = [self.get_cached_benchmark(key) for key in keys]
    
            all_keys_cached = True
            for cached_benchmark in cached_benchmarks:
                if cached_benchmark == None:
                    all_keys_cached = False
                    break
            
            if all_keys_cached:
                return cached_benchmarks

        def time_interleaved(buffer, callables, iters):
            interleaved_event_pairs = self.get_interleaved_event_pairs(callables, iters)
            for event_pairs in interleaved_event_pairs:
                for _callable, (start_event, end_event) in zip(callables, event_pairs):
                    buffer.zero_()
                    start_event.record()
                    _callable()
                    end_event.record()
            torch.cuda.synchronize()
            return self.get_interleaved_min_timing(interleaved_event_pairs)

        if len(callables) == 0:
            return []
        elif len(callables) == 1:
            key = keys[0] if keys != [] else None
            return [self.benchmark_gpu(callables[0], key=key, estimation_iters=estimation_iters, groups=groups, memory_warmup_iters=memory_warmup_iters, benchmark_iters=benchmark_iters)]
       
        buffer = torch.empty(int(self.get_cache_size() // 4), dtype=torch.int, device="cuda")

        estimated_timings = time_interleaved(buffer, callables, estimation_iters)
        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / max(estimated_timings)), 1))
        benchmark_iters_per_group = max(benchmark_iters // groups, 1)

        per_group_timings = []
        for _ in range(groups):
            for _ in range(memory_warmup_iters):
                buffer.zero_()
            per_group_timings.append(time_interleaved(buffer, callables, benchmark_iters_per_group))
        benchmarks = [min(callable_timings) for callable_timings in zip(*per_group_timings)]
        
        del buffer

        if keys != []:
            assert len(callables) == len(keys)
            for key, benchmark in zip(keys, benchmarks):
                self.put_cached_benchmark(key, benchmark)

        return benchmarks
    
    def lazy_benchmark(self, fn, fn_args, fn_kwargs, **kwargs):
        _callable = lambda: fn(*fn_args, **fn_kwargs)

        fn_args_and_kwargs = list(fn_args) + list(fn_kwargs.values())
        if is_cpu_device(fn_args_and_kwargs):
            return self.benchmark_cpu(_callable, **kwargs)
        else:
            return self.lazy_benchmark_gpu(_callable, **kwargs)
    
    def lazy_benchmark_gpu(self, _callable, key=None, **kwargs):
        kwargs_hash = hash(tuple(sorted(kwargs.items())))

        if key == None:
            key = hash(_callable) + kwargs_hash
        
        self.futures_gpu[kwargs_hash] = self.futures_gpu.get(kwargs_hash, [])
        if key not in list(zip(*self.futures_gpu[kwargs_hash])): 
            self.futures_gpu[kwargs_hash].append((_callable, key))

        def finalize():
            benchmark = benchmarker.get_cached_benchmark(key, bypass_cache_configs=True)
            if benchmark != None:
                return benchmark
            
            futures_gpu = benchmarker.futures_gpu.pop(kwargs_hash)
            callables, keys = zip(*futures_gpu)
            callables, keys = list(callables), list(keys)

            benchmarker.benchmark_many_gpu(callables, keys=keys, **kwargs)
            benchmark = benchmarker.get_cached_benchmark(key, bypass_cache_configs=True)

            return benchmark

        return LazyBenchmark(finalize)

    def get_local_cache_dir(self):
        cache_path = Path(cache_dir()) / "benchmarks"
        cache_path.mkdir(parents=True, exist_ok=True)
        if cache_path == "/tmp/torchinductor_nmacchioni":
            breakpoint()
        return cache_path
    
    def get_cached_benchmark(self, key, bypass_cache_configs=False):
        if (not inductor_config.benchmark_local_cache or inductor_config.force_disable_caches) and not bypass_cache_configs:
            return None

        if key == None:
            return None

        cache_path = self.get_local_cache_dir() / str(key)
        if not cache_path.is_file():
            return None
        
        with open(cache_path, "r") as fp:
            cached_benchmark = float(fp.read().strip())
        
        return cached_benchmark
    
    def put_cached_benchmark(self, key, benchmark):
        if key != None:        
            cache_path = self.get_local_cache_dir() / str(key)
            with open(cache_path, "w") as fp:
                fp.write(str(benchmark))

    @functools.lru_cache(None)
    def get_cache_size(self):
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.l2CacheSize

    def get_event_pairs(self, iters):
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_interleaved_event_pairs(self, fns, iters):
        return [self.get_event_pairs(len(fns)) for _ in range(iters)]

    def get_min_timing(self, event_pairs):
        return min([start_event.elapsed_time(end_event) for start_event, end_event in event_pairs])

    def get_interleaved_min_timing(self, interleaved_event_pairs):
        return [self.get_min_timing(event_pairs) for event_pairs in zip(*interleaved_event_pairs)]

class LazyBenchmark:
    def __init__(self, finalize):
        self.finalize = finalize
        self.value = None
    
    def __finalize__(self):
        if self.value == None:
            self.value = self.finalize()
        
    def __float__(self):
        self.__finalize__()
        return self.value

    def __lt__(self, other):
        self.__finalize__()
        return other > self.value
    
    def __le__(self, other):
        self.__finalize__()
        return other >= self.value
    
    def __gt__(self, other):
        self.__finalize__()
        return other < self.value
    
    def __ge__(self, other):
        self.__finalize__()
        return other <= self.value

    def __add__(self, other):
        return LazyBenchmark(lambda: self.finalize() + other)
    
    def __radd__(self, other):
        return LazyBenchmark(lambda: other + self.finalize())

    def __sub__(self, other):
        return LazyBenchmark(lambda: self.finalize() - other)
    
    def __rsub__(self, other):
        return LazyBenchmark(lambda: other - self.finalize())
            
    
benchmarker = Benchmarker()