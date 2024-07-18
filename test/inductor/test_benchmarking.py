# Owner(s): ["module: inductor"]

import logging
import time
import unittest

import torch

from torch._dynamo.utils import counters
from torch._inductor.runtime.benchmarking import Benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU, GPU_TYPE


log = logging.getLogger(__name__)


def patches(fn):
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)
    return wrapped


class TestBench(TestCase):
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

        fn, args, kwargs = self.get_cpu_fn_args_kwargs()
        timing = benchmarker.benchmark(fn, args, kwargs)
        sanity_check_timing = self.sanity_check_cpu_benchmark(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark"], 1)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_cpu"], 1)
        counters.clear()

        _callable = self.get_cpu_callable()
        timing = benchmarker.benchmark_cpu(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_cpu"], 1)
        counters.clear()

        timings = benchmarker.benchmark_many_cpu([_callable for _ in range(10)])
        self.assertEqual(all([timing <= sanity_check_timing for timing in timings], True))
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_many_cpu"], 1)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_cpu"], 10)
        counters.clear()
    
    @unittest.skipIf(not HAS_GPU, "Skipping GPU tests.")
    @patches
    def test_benchmarker_basics_cpu(self):
        benchmarker = Benchmarker()

        fn, args, kwargs = self.get_gpu_fn_args_kwargs()
        timing = benchmarker.benchmark(fn, args, kwargs)
        sanity_check_timing = self.sanity_check_gpu_benchmark(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark"], 1)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_gpu"], 1)
        counters.clear()

        _callable = self.get_gpu_callable()
        timing = benchmarker.benchmark_gpu(_callable)
        self.assertEqual(timing <= sanity_check_timing, True)
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_gpu"], 1)
        counters.clear()

        timings = benchmarker.benchmark_many_gpu([_callable for _ in range(10)])
        self.assertEqual(all([timing <= sanity_check_timing for timing in timings], True))
        self.assertEqual(counters["inductor"]["benchmarking"]["benchmark_many_gpu"], 1)
        counters.clear()

    @unittest.skipIf(not HAS_CPU, "Skipping CPU tests.")
    @patches
    def test_benchmarker_properties_cpu(self):
        benchmarker = Benchmarker()

        benchmarker.benchmark_cpu(self.get_cpu_callable())
        self.assertEqual(counters["inductor"]["benchmarking"]["L2_cache_size"], 0)
        self.assertEqual(counters["inductor"]["benchmarking"]["gpu_time_per_gpu_clock_cycle"], 0)
        self.assertEqual(counters["inductor"]["benchmarking"]["cpu_launch_overhead_per_gpu_cache_clear"], 0)
        self.assertEqual(counters["inductor"]["benchmarking"]["gpu_time_per_gpu_cache_clear"], 0)
        counters.clear()
    
    @unittest.skipIf(not HAS_GPU, "Skipping GPU tests.")
    @patches
    def test_benchmarker_properties_gpu(self):
        benchmarker = Benchmarker()

        benchmarker.benchmark_gpu(self.get_gpu_callable())
        benchmarker.benchmark_gpu(self.get_gpu_callable())
        self.assertEqual(counters["inductor"]["benchmarking"]["L2_cache_size"], 1)
        self.assertEqual(counters["inductor"]["benchmarking"]["gpu_time_per_gpu_clock_cycle"], 1)
        self.assertEqual(counters["inductor"]["benchmarking"]["cpu_launch_overhead_per_gpu_cache_clear"], 1)
        self.assertEqual(counters["inductor"]["benchmarking"]["gpu_time_per_gpu_cache_clear"], 1)
        counters.clear()


if __name__ == "__main__":
    run_tests("cuda")