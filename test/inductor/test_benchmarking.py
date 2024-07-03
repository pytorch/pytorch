# Owner(s): ["module: inductor"]

import logging

import torch
from torch._inductor.runtime.benchmarking import benchmarker, LazyBenchmark

from torch._inductor.test_case import run_tests, TestCase


log = logging.getLogger(__name__)


class TestBench(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.x_1 = torch.rand(1024, 10).cuda().half()
        cls.w_1 = torch.rand(512, 10).cuda().half()
        cls.function_1 = torch.nn.functional.Linear
        cls.args_1 = [cls.x_1, cls.w_1]
        cls.kwargs_1 = {}
        cls.benchmarkable_1 = lambda: cls.function_1(cls.x_1, cls.w_1)

        cls.x_2 = torch.rand(1024, 10).cuda().half()
        cls.w_2 = torch.rand(512, 10).cuda().half()
        cls.function_2 = torch.nn.functional.Linear
        cls.args_2 = [cls.x_2, cls.w_2]
        cls.kwargs_2 = {}
        cls.benchmarkable_2 = lambda: cls.function_2(cls.x_2, cls.w_2)
    
    def test_benchmark(self):
        benchmark = benchmarker.benchmark(self.function_1, self.args_1, self.kwargs_1)
        self.assertEqual(type(benchmark), float)
        self.assertGreater(benchmark, 0.0)
    
    def test_benchmark_gpu(self):
        benchmark = benchmarker.benchmark_gpu(self.benchmarkable_1)
        self.assertEqual(type(benchmark), float)
        self.assertGreater(benchmark, 0.0)
    
    def test_benchmark_many_gpu(self):
        benchmarks = benchmarker.benchmark_many_gpu([self.function_1, self.function_2])
        self.assertEqual(type(benchmarks[0]), float)
        self.assertGreater(benchmarks[0], 0.0)
        self.assertEqual(type(benchmarks[1]), float)
        self.assertGreater(benchmarks[1], 0.0)
    
    def test_lazy_benchmark_gpu(self):
        lazy_benchmark = benchmarker.lazy_benchmark_gpu(self.function_1)
        self.assertEqual(type(lazy_benchmark), LazyBenchmark)

        new_lazy_benchmark = lazy_benchmark
        self.assertEqual(type(new_lazy_benchmark), LazyBenchmark)

        new_lazy_benchmark = new_lazy_benchmark + 1
        self.assertEqual(type(new_lazy_benchmark), LazyBenchmark)
    
        new_lazy_benchmark = 1 + new_lazy_benchmark
        self.assertEqual(type(new_lazy_benchmark), LazyBenchmark)

        new_lazy_benchmark = new_lazy_benchmark - 1
        self.assertEqual(type(new_lazy_benchmark), LazyBenchmark)

        new_lazy_benchmark = 1 - new_lazy_benchmark
        self.assertEqual(type(new_lazy_benchmark), LazyBenchmark)

        should_evaluate_to_bool = new_lazy_benchmark > 0.0
        self.assertEqual(type(should_evaluate_to_bool), bool)

        should_evaluate_to_bool = new_lazy_benchmark >= 0.0
        self.assertEqual(type(should_evaluate_to_bool), bool)

        should_evaluate_to_bool = new_lazy_benchmark < 0.0
        self.assertEqual(type(should_evaluate_to_bool), bool)

        should_evaluate_to_bool = new_lazy_benchmark <= 0.0
        self.assertEqual(type(should_evaluate_to_bool), bool)

        should_evaluate_to_float = float(new_lazy_benchmark)
        self.assertEqual(type(should_evaluate_to_float), float)


if __name__ == "__main__":
    run_tests("cuda")
