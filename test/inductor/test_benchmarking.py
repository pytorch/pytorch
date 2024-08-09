# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import (
    Benchmarker,
    InductorBenchmarker,
    TritonBenchmarker,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
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

    @property
    def benchmarker(self):
        return Benchmarker()

    def counter_value(self, fn_name):
        return counters["inductor"][
            f"benchmarking.{type(self.benchmarker).__name__}.{fn_name}"
        ]

    def make_sum(self, device, size=100):
        fn, args, kwargs = torch.sum, (torch.randn(size, device=device),), {}
        _callable = lambda: fn(*args, **kwargs)  # noqa: E731
        return (fn, args, kwargs), _callable

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_benchmark(self, device):
        benchmarker = self.benchmarker
        (fn, args, kwargs), _ = self.make_sum(device)
        if device == "cpu":
            _ = benchmarker.benchmark(fn, *args, **kwargs)
            self.assertEqual(self.counter_value("benchmark"), 1)
            self.assertEqual(self.counter_value("benchmark_cpu"), 1)
        else:
            self.assertExpectedRaises(
                NotImplementedError, lambda: benchmarker.benchmark(fn, *args, **kwargs)
            )
            self.assertEqual(self.counter_value("benchmark"), 1)
            self.assertEqual(self.counter_value("benchmark_gpu"), 1)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    def test_benchmark_cpu(self, device="cpu"):
        benchmarker = self.benchmarker
        _, _callable = self.make_sum(device)
        _ = benchmarker.benchmark_cpu(_callable)
        self.assertEqual(self.counter_value("benchmark_cpu"), 1)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_gpu(self, device=GPU_TYPE):
        benchmarker = self.benchmarker
        _, _callable = self.make_sum(device)
        self.assertExpectedRaises(
            NotImplementedError, lambda: benchmarker.benchmark_gpu(_callable)
        )
        self.assertEqual(self.counter_value("benchmark_gpu"), 1)


@instantiate_parametrized_tests
class TestTritonBenchmarker(TestBenchmarker):
    @property
    def benchmarker(self):
        return TritonBenchmarker()

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_benchmark(self, device):
        benchmarker = self.benchmarker
        (fn, args, kwargs), _ = self.make_sum(device)
        _ = benchmarker.benchmark(fn, *args, **kwargs)
        self.assertEqual(self.counter_value("benchmark"), 1)
        if device == "cpu":
            self.assertEqual(self.counter_value("benchmark_cpu"), 1)
        else:
            self.assertEqual(self.counter_value("benchmark_gpu"), 1)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_gpu(self, device=GPU_TYPE):
        benchmarker = self.benchmarker
        _, _callable = self.make_sum(device)
        _ = benchmarker.benchmark_gpu(_callable)
        self.assertEqual(self.counter_value("triton_do_bench"), 1)
        self.assertEqual(self.counter_value("benchmark_gpu"), 1)


@instantiate_parametrized_tests
class TestInductorBenchmarker(TestTritonBenchmarker):
    @property
    def benchmarker(self):
        return InductorBenchmarker()

    @property
    def feature_name(self):
        return "inductor_benchmarker"

    @config.path({"is_fbcode": lambda: False})
    @parametrize(
        "config_name,config_val,expected_should_fallback",
        [
            ("env_var", "1", False),
            ("env_var", "0", True),
            ("env_var", "", None),
            ("oss_default", True, False),
            ("oss_default", False, True),
        ],
    )
    def test_should_fallback(self, config_name, config_val, expected_should_fallback):
        @config.patch({f"benchmarking.{self.feature_name}.{config_name}": config_val})
        def inner():
            return self.benchmarker.should_fallback()

        if expected_should_fallback is not None:
            self.assertEqual(expected_should_fallback, inner())
        else:
            self.assertEqual(
                expected_should_fallback,
                getattr(config.benchmarking, self.feature_name).oss_default,
            )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @parametrize("should_fallback", (True, False))
    def test_benchmark_gpu(self, should_fallback, device=GPU_TYPE):
        benchmarker = self.benchmarker
        _, _callable = self.make_sum(device)
        if should_fallback:
            benchmarker.should_fallback = lambda: True
            _ = benchmarker.benchmark_gpu(_callable)
            self.assertEqual(super().counter_value("triton_do_bench"), 1)
            self.assertEqual(super().counter_value("benchmark_gpu"), 1)
        else:
            benchmarker.should_fallback = lambda: False
            self.assertEqual(self.counter_value("benchmark_gpu"), 1)


if __name__ == "__main__":
    run_tests()
