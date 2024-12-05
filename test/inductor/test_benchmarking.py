# Owner(s): ["module: inductor"]

import itertools
import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import (
    Benchmarker,
    InductorBenchmarker,
    is_feature_enabled,
    TritonBenchmarker,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


ALL_BENCHMARKER_CLASSES = (
    Benchmarker,
    TritonBenchmarker,
    InductorBenchmarker,
)


@instantiate_parametrized_tests
class TestBenchmarker(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        counters.clear()

    @staticmethod
    def get_counter_value(benchmarker_cls, fn_name):
        return counters["inductor"][
            f"benchmarking.{benchmarker_cls.__name__}.{fn_name}"
        ]

    @staticmethod
    def make_params(device, size=100):
        fn, fn_args, fn_kwargs = torch.sum, (torch.randn(size, device=device),), {}
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        return (fn, fn_args, fn_kwargs), _callable

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker
        and params["device"] == GPU_TYPE,
    )
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_benchmark_smoke(self, benchmarker_cls, device):
        benchmarker = benchmarker_cls()
        (fn, fn_args, fn_kwargs), _ = self.make_params(device)
        timing = benchmarker.benchmark(fn, fn_args, fn_kwargs)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark"), 1)
        self.assertEqual(
            self.get_counter_value(
                benchmarker_cls, "benchmark_cpu" if device == "cpu" else "benchmark_gpu"
            ),
            1,
        )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_cpu_smoke(self, benchmarker_cls, device="cpu"):
        benchmarker = benchmarker_cls()
        _, _callable = self.make_params(device)
        timing = benchmarker.benchmark_cpu(_callable)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_cpu"), 1)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["benchmarker_cls"] is Benchmarker,
    )
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_gpu_smoke(self, benchmarker_cls, device=GPU_TYPE):
        benchmarker = benchmarker_cls()
        _, _callable = self.make_params(device)
        timing = benchmarker.benchmark_gpu(_callable)
        self.assertGreater(timing, 0)
        self.assertEqual(self.get_counter_value(benchmarker_cls, "benchmark_gpu"), 1)
        if benchmarker_cls is TritonBenchmarker:
            self.assertEqual(
                self.get_counter_value(benchmarker_cls, "triton_do_bench"), 1
            )

    @unittest.skipIf(not HAS_CPU and not HAS_GPU, "requires CPU or GPU")
    @unittest.expectedFailure
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_safely_infers_device_no_devices(
        self, benchmarker_cls, device="cpu" if HAS_CPU else GPU_TYPE
    ):
        benchmarker = benchmarker_cls()
        (fn, _, _), _ = self.make_params(device)
        benchmarker.benchmark(fn, (), {})

    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @unittest.expectedFailure
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmark_safely_infers_device_many_devices(self, benchmarker_cls):
        benchmarker = benchmarker_cls()
        (fn, cpu_args, cpu_kwargs), _ = self.make_params("cpu")
        (_, gpu_args, gpu_kwargs), _ = self.make_params(GPU_TYPE)
        many_devices_args = cpu_args + gpu_args
        many_devices_kwargs = cpu_kwargs
        many_devices_kwargs.update(gpu_kwargs)
        benchmarker.benchmark(fn, many_devices_args, many_devices_kwargs)

    @unittest.skipIf(config.is_fbcode(), "test does not run in fbcode")
    @parametrize("feature_name", ("inductor_benchmarker",))
    @parametrize(
        "config_name,config_val,expected",
        [
            ("env_val", "1", True),
            ("env_val", "0", False),
            ("env_val", None, None),
            ("oss_default", True, True),
            ("oss_default", False, False),
        ],
    )
    def test_is_feature_enabled(
        self,
        feature_name,
        config_name,
        config_val,
        expected,
    ):
        @config.patch({f"benchmarking.{feature_name}.{config_name}": config_val})
        def inner():
            return is_feature_enabled(feature_name)

        if expected is not None:
            self.assertEqual(inner(), expected)
        else:
            self.assertEqual(
                inner(),
                getattr(config.benchmarking, feature_name).oss_default,
            )
    
    @unittest.skipIf(not HAS_CPU or not HAS_GPU, "requires CPU and GPU")
    @parametrize("fn_name", ("benchmark", "benchmark_cpu", "benchmark_gpu",))
    @parametrize("enabled", (True, False,))
    @parametrize("device", (GPU_TYPE, "cpu",))
    def test_inductor_benchmarker_fallback(self, fn_name, enabled, device):
        @config.patch({
            f"benchmarking.{InductorBenchmarker.feature_name}.env_val": 1 if enabled else 0
        })
        def inner():
            benchmarker = InductorBenchmarker()
            _, _callable = self.make_params(device)
            _ = getattr(benchmarker, fn_name)(_callable)

        inner()
        if not enabled and fn_name == "benchmark_gpu":
            # "benchmark_gpu" is the only `InductorBenchmarker`-specific feature that
            # should get disabled
            self.assertEqual(self.get_counter_value(TritonBenchmarker, fn_name), 1)
        else:
            # all other benchmark functions should still pass, since they are inherited
            self.assertEqual(self.get_counter_value(InductorBenchmarker, fn_name), 1)


if __name__ == "__main__":
    run_tests()
