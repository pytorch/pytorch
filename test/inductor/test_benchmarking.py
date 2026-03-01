# Owner(s): ["module: inductor"]

import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor.config import (
    inductor_default_autotune_rep,
    inductor_default_autotune_warmup,
)
from torch._inductor.runtime.benchmarking import Benchmarker, TritonBenchmarker
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
        (fn, cpu_args, cpu_kwargs), _ = self.make_sum("cpu")
        (_, gpu_args, gpu_kwargs), _ = self.make_sum(GPU_TYPE)
        many_devices_args = cpu_args + gpu_args
        many_devices_kwargs = cpu_kwargs
        many_devices_kwargs.update(gpu_kwargs)
        benchmarker.benchmark(fn, many_devices_args, many_devices_kwargs)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_warmup_and_rep_defaults(self):
        """Test that benchmark_gpu receives default warmup and rep values when not specified."""
        captured_kwargs = {}

        def capture_benchmark_gpu(self, _callable, **kwargs):
            captured_kwargs.update(kwargs)
            return 1.0  # Return a dummy timing

        benchmarker = TritonBenchmarker()
        (fn, fn_args, fn_kwargs), _ = self.make_params(GPU_TYPE)

        with patch.object(TritonBenchmarker, "benchmark_gpu", capture_benchmark_gpu):
            benchmarker.benchmark(fn, fn_args, fn_kwargs)

        self.assertEqual(captured_kwargs["warmup"], inductor_default_autotune_warmup)
        self.assertEqual(captured_kwargs["rep"], inductor_default_autotune_rep)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    def test_benchmark_warmup_and_rep_custom_values(self):
        """Test that benchmark_gpu receives custom warmup and rep values when specified."""
        captured_kwargs = {}

        def capture_benchmark_gpu(self, _callable, **kwargs):
            captured_kwargs.update(kwargs)
            return 1.0  # Return a dummy timing

        benchmarker = TritonBenchmarker()
        (fn, fn_args, fn_kwargs), _ = self.make_params(GPU_TYPE)

        custom_warmup = 50
        custom_rep = 200

        with patch.object(TritonBenchmarker, "benchmark_gpu", capture_benchmark_gpu):
            benchmarker.benchmark(
                fn, fn_args, fn_kwargs, warmup=custom_warmup, rep=custom_rep
            )

        self.assertEqual(captured_kwargs["warmup"], custom_warmup)
        self.assertEqual(captured_kwargs["rep"], custom_rep)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmarker_cpu_override_dispatch(self, benchmarker_cls, device="cpu"):
        # Registers a custom handler for 'cpu' and verifies dispatch uses it instead of the default path.
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = benchmarker_cls()

        # Snapshot registry and restore at the end to avoid cross-test pollution.
        orig = dict(_bench._BENCHMARK_DISPATCH)
        try:
            seen = {"cpu_override": 0}

            def custom_cpu(self, fn, *, warmup, rep, **kw):
                seen["cpu_override"] += 1
                return "cpu-override"

            # Override the built-in 'cpu' registration
            _bench.register_benchmarker("cpu", custom_cpu, override=True)

            # Ensure default CPU/GPU methods are NOT called if registry override works.
            with (
                patch.object(
                    benchmarker_cls,
                    "benchmark_cpu",
                    side_effect=AssertionError(
                        "benchmark_cpu should not be called when a custom 'cpu' handler is registered"
                    ),
                    create=True,
                ),
                patch.object(
                    benchmarker_cls,
                    "benchmark_gpu",
                    side_effect=AssertionError(
                        "benchmark_gpu should not be called for 'cpu' device"
                    ),
                    create=True,
                ),
            ):
                (fn, fn_args, fn_kwargs), _ = self.make_params(device)
                out = benchmarker.benchmark(fn, fn_args, fn_kwargs)
                self.assertEqual(out, "cpu-override")
                self.assertEqual(seen["cpu_override"], 1)
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @parametrize("benchmarker_cls", ALL_BENCHMARKER_CLASSES)
    def test_benchmarker_cpu_override_runs_callable(
        self, benchmarker_cls, device="cpu"
    ):
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = benchmarker_cls()
        orig = dict(_bench._BENCHMARK_DISPATCH)
        try:
            # Override CPU but still route to benchmark_cpu internally
            def custom_cpu(self, f, *, warmup, rep, **kw):
                # Just delegate to the original path; we want to ensure `f()` calls the user's fn.
                return self.benchmark_cpu(f, warmup=warmup, rep=rep, **kw)

            _bench.register_benchmarker("cpu", custom_cpu, override=True)
            # Define a simple op and ensure it actually runs without TypeError
            (fn, fn_args, fn_kwargs), _ = self.make_params(device)
            out = benchmarker.benchmark(fn, fn_args, fn_kwargs, warmup=1, rep=1)
            self.assertGreater(out, 0)
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)


@unittest.skipIf(not HAS_GPU, "requires GPU")
class TestDoBenchUsingProfilingMulti(TestCase):
    """Tests for do_bench_using_profiling_multi function."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)

    def test_single_callable_consistency(self):
        """Single callable should produce results comparable to do_bench_using_profiling."""
        from torch._inductor.utils import (
            do_bench_using_profiling,
            do_bench_using_profiling_multi,
        )

        def kernel():
            x = torch.randn(1000, 1000, device=GPU_TYPE)
            return x @ x

        time_single = do_bench_using_profiling(kernel, is_vetted_benchmarking=True)
        [time_multi] = do_bench_using_profiling_multi(
            [kernel], is_vetted_benchmarking=True
        )

        # Within 5% due to profiler variance (benchmarking is inherently noisy)
        self.assertGreater(time_single, 0)
        self.assertGreater(time_multi, 0)
        ratio = time_multi / time_single
        self.assertGreater(ratio, 0.95, f"Multi timing {time_multi} too low vs single {time_single}")
        self.assertLess(ratio, 1.05, f"Multi timing {time_multi} too high vs single {time_single}")

    def test_multiple_callables_ordering(self):
        """Multiple callables should preserve relative timing order."""
        from torch._inductor.utils import do_bench_using_profiling_multi

        def small_kernel():
            x = torch.randn(500, 500, device=GPU_TYPE)
            return x @ x

        def large_kernel():
            x = torch.randn(2000, 2000, device=GPU_TYPE)
            return x @ x

        times = do_bench_using_profiling_multi(
            [small_kernel, large_kernel], is_vetted_benchmarking=True
        )

        self.assertEqual(len(times), 2)
        # Large kernel should be slower
        self.assertGreater(
            times[1],
            times[0],
            f"Large kernel ({times[1]}) should be slower than small kernel ({times[0]})",
        )

    def test_multiple_callables_different_sizes(self):
        """Test with three callables of different sizes."""
        from torch._inductor.utils import do_bench_using_profiling_multi

        def tiny_kernel():
            x = torch.randn(100, 100, device=GPU_TYPE)
            return x @ x

        def medium_kernel():
            x = torch.randn(1000, 1000, device=GPU_TYPE)
            return x @ x

        def large_kernel():
            x = torch.randn(2000, 2000, device=GPU_TYPE)
            return x @ x

        times = do_bench_using_profiling_multi(
            [tiny_kernel, medium_kernel, large_kernel], is_vetted_benchmarking=True
        )

        self.assertEqual(len(times), 3)
        # All times should be positive
        for t in times:
            self.assertGreater(t, 0)
        # Times should be in increasing order
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        from torch._inductor.utils import do_bench_using_profiling_multi

        with self.assertRaises(ValueError):
            do_bench_using_profiling_multi([], is_vetted_benchmarking=True)

    def test_result_count_matches_input(self):
        """Number of results should match number of input callables."""
        from torch._inductor.utils import do_bench_using_profiling_multi

        def kernel():
            x = torch.randn(500, 500, device=GPU_TYPE)
            return x @ x

        for n in [1, 2, 3, 5]:
            times = do_bench_using_profiling_multi(
                [kernel] * n, is_vetted_benchmarking=True
            )
            self.assertEqual(
                len(times), n, f"Expected {n} results, got {len(times)}"
            )


if __name__ == "__main__":
    run_tests()
