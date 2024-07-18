# Owner(s): ["module: inductor"]

import functools
import logging
import time
import unittest

import torch

from torch._dynamo.utils import counters
from torch._inductor.runtime.benchmarking import Benchmarker, LazyBenchmark
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU, GPU_TYPE


log = logging.getLogger(__name__)


def patches(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)
    return wrapped


class TestBenchmarking(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
    
    def get_cpu_fn_args_kwargs(self):
        return torch.sum, [torch.randn(1000)], {}
    
    def get_gpu_fn_args_kwargs(self):
        return torch.sum, [torch.randn(1000, device=GPU_TYPE)], {}

    def get_cpu_callable(self):
        fn, args, kwargs = self.get_cpu_fn_args_kwargs()
        return lambda: fn(*args, **kwargs)
    
    def get_gpu_callable(self):
        fn, args, kwargs = self.get_gpu_fn_args_kwargs()
        return lambda: fn(*args, **kwargs)
    
    def sanity_check_cpu_benchmark(self, _callable):
        start_time = time.perf_counter()
        for _ in range(10):
            _callable()
        return ((time.perf_counter() - start_time) * 1000) / 10
    
    def sanity_check_gpu_benchmark(self, _callable):
        start_event = torch.cuda.Event(enable_timing=True)
        for _ in range(10):
            _callable()
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / 10
    
    @unittest.skipIf(not HAS_CPU, "Skipping CPU tests.")
    @patches
    def test_benchmarker_basics_cpu(self):
        benchmarker = Benchmarker()

        # test benchmarker.benchmark
        fn, args, kwargs = self.get_cpu_fn_args_kwargs()
        timing = benchmarker.benchmark(fn, args, kwargs)
        sanity_check_timing = self.sanity_check_cpu_benchmark(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_cpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_cpu
        _callable = self.get_cpu_callable()
        timing = benchmarker.benchmark_cpu(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_cpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_many_cpu with single callable
        timings = benchmarker.benchmark_many_cpu([_callable for _ in range(1)])
        self.assertEqual(all([timing <= sanity_check_timing for timing in timings], True))
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_many_cpu"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_cpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_many_cpu with many callables
        timings = benchmarker.benchmark_many_cpu([_callable for _ in range(10)])
        self.assertEqual(all([timing <= sanity_check_timing for timing in timings], True))
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_many_cpu"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_cpu"], 10)
        counters.clear()
    
    @unittest.skipIf(not HAS_GPU, "Skipping GPU tests.")
    @patches
    def test_benchmarker_basics_gpu(self):
        benchmarker = Benchmarker()

        # test benchmarker.benchmark
        fn, args, kwargs = self.get_gpu_fn_args_kwargs()
        timing = benchmarker.benchmark(fn, args, kwargs)
        sanity_check_timing = self.sanity_check_gpu_benchmark(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_gpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_gpu
        _callable = self.get_gpu_callable()
        timing = benchmarker.benchmark_gpu(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_gpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_many_gpu with single callable
        timings = benchmarker.benchmark_many_gpu([_callable for _ in range(1)])
        self.assertEqual(all([timing <= sanity_check_timing for timing in timings], True))
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_many_gpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_many_gpu with many callables
        timings = benchmarker.benchmark_many_gpu([_callable for _ in range(10)])
        self.assertEqual(all([timing <= sanity_check_timing for timing in timings], True))
        self.assertEqual(counters["inductor"]["benchmarking_benchmark_many_gpu"], 1)
        counters.clear()

        # test benchmarker.benchmark_gpu with failing callable
        bad_callable = lambda: 1 / 0
        timing = benchmarker.benchmark_gpu(bad_callable)
        self.assertEqual(timing, float("inf"))
        self.assertEqual(counters["inductor"]["benchmarking_callable_initialization_failed"], 1)
        counters.clear()

        # test benchmarker.benchmark_many_gpu with all failing callables
        bad_callable = lambda: 1 / 0
        timings = benchmarker.benchmark_many_gpu([bad_callable for _ in range(10)])
        self.assertEqual(all([timing == float("inf") for timing in timings]), True)
        self.assertEqual(counters["inductor"]["benchmarking_callable_initialization_failed"], 10)
        counters.clear()

        # test benchmarker.benchmark_many_gpu with some failing callables
        good_callable = self.get_gpu_callable()
        bad_callable = lambda: 1 / 0
        good_callable_timing, bad_callable_timing = benchmarker.benchmark_many_gpu([_callable, bad_callable])
        self.assertEqual(good_callable_timing <= sanity_check_timing, True)
        self.assertEqual(bad_callable_timing, float("inf"))
        self.assertEqual(counters["inductor"]["benchmarking_callable_initialization_failed"], 1)
        counters.clear()
    
    @unittest.skipIf(not HAS_GPU, "Skipping GPU tests.")
    @patches
    def test_benchmarker_early_ranking(self):
        benchmarker = Benchmarker()
        _callable = self.get_gpu_callable()
        benchmarker.benchmark_gpu([_callable for _ in range(10)], ranking_key="testing")
        self.assertEqual(counters["inductor"]["benchmarking_early_ranking"], 1)
        counters.clear()
    
    @unittest.skipIf(not HAS_GPU, "Skipping GPU tests.")
    @patches
    def test_benchmarker_early_pruning(self):
        benchmarker = Benchmarker()
        _callable = self.get_gpu_callable()
        benchmarker.benchmark_gpu([_callable for _ in range(10)], pruning_key="testing")
        self.assertEqual(counters["inductor"]["benchmarking_early_pruning"], 1)
        counters.clear()

    @unittest.skipIf(not HAS_CPU, "Skipping CPU tests.")
    @patches
    def test_benchmarker_properties_cpu(self):
        benchmarker = Benchmarker()
        benchmarker.benchmark_cpu(self.get_cpu_callable())
        self.assertEqual(counters["inductor"]["benchmarking_L2_cache_size"], 0)
        self.assertEqual(counters["inductor"]["benchmarking_gpu_time_per_gpu_clock_cycle"], 0)
        self.assertEqual(counters["inductor"]["benchmarking_get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_cpu_launch_overhead_per_gpu_cache_clear"], 0)
        self.assertEqual(counters["inductor"]["benchmarking_gpu_time_per_gpu_cache_clear"], 0)
        counters.clear()
    
    @unittest.skipIf(not HAS_GPU, "Skipping GPU tests.")
    @patches
    def test_benchmarker_properties_gpu(self):
        benchmarker = Benchmarker()
        benchmarker.benchmark_gpu(self.get_gpu_callable())
        benchmarker.benchmark_gpu(self.get_gpu_callable())
        self.assertEqual(counters["inductor"]["benchmarking_L2_cache_size"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_gpu_time_per_gpu_clock_cycle"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_get_cpu_launch_overhead_and_gpu_time_per_gpu_cache_clear"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_cpu_launch_overhead_per_gpu_cache_clear"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_gpu_time_per_gpu_cache_clear"], 1)
        counters.clear()

    @patches
    def test_lazy_benchmark_magic_methods(self):
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark), float(0.0))
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(format(lazy_benchmark, f"{0}"), format(0.0, f"{0}"))
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(str(lazy_benchmark), str(0.0))
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(0.0 > -1.0, True)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(0.0 >= -1.0, True)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(0.0 < 1.0, True)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(0.0 <= 1.0, True)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0) + 1.0
        self.assertEqual(lazy_benchmark.value, 1.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = 1.0 + LazyBenchmark(lambda: 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, 1.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0) - 1.0
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, -1.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = 1.0 - LazyBenchmark(lambda: 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, 1.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0) * 1.0
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = 1.0 * LazyBenchmark(lambda: 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = LazyBenchmark(lambda: 0.0) / 1.0
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

        lazy_benchmark = 0.0 / LazyBenchmark(lambda: 1.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0)
        self.assertEqual(lazy_benchmark.value, 0.0)
        self.assertEqual(counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 2)
        counters.clear()

if __name__ == "__main__":
    run_tests("cuda")