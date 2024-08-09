# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor.runtime.benchmarking import Benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


@instantiate_parametrized_tests
class TestBenchmarker(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        counters.clear()

    def counter_value(self, benchmarker_cls, fn_name):
        return counters["inductor"][
            f"benchmarking.{benchmarker_cls.__name__}.{fn_name}"
        ]

    def make_sum(self, device, size=100):
        fn, args, kwargs = torch.sum, (torch.randn(size, device=device),), {}
        _callable = lambda: fn(*args, **kwargs)  # noqa: E731
        return (fn, args, kwargs), _callable

    @property
    def benchmarker(self):
        return Benchmarker()

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker
        and params["device"] == GPU_TYPE,
    )
    @parametrize("benchmarker_cls", (Benchmarker))
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_benchmark(self, benchmarker_cls, device):
        benchmarker = benchmarker_cls()
        (fn, args, kwargs), _ = self.make_sum(device)
        _ = benchmarker.benchmark(fn, args, kwargs)
        self.assertEqual(self.counter_value(benchmarker_cls, "benchmark"), 1)
        self.assertEqual(
            self.counter_value(
                benchmarker_cls, "benchmark_cpu" if device == "cpu" else "benchmark_gpu"
            ),
            1,
        )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", (Benchmarker))
    def test_benchmark_cpu(self, benchmarker_cls, device="cpu"):
        benchmarker = benchmarker_cls()
        _, _callable = self.make_sum(device)
        _ = benchmarker.benchmark_cpu(_callable)
        self.assertEqual(self.counter_value(benchmarker_cls, "benchmark_cpu"), 1)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker,
    )
    @parametrize("benchmarker_cls", (Benchmarker))
    def test_benchmark_gpu(self, benchmarker_cls, device=GPU_TYPE):
        benchmarker = benchmarker_cls()
        _, _callable = self.make_sum(device)
        _ = benchmarker.benchmark_gpu(_callable)
        self.assertEqual(self.counter_value(benchmarker_cls, "benchmark_gpu"), 1)


if __name__ == "__main__":
    run_tests()
