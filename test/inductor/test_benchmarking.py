# Owner(s): ["module: inductor"]

import functools
import itertools
import time

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import Benchmarker, LazyBenchmark
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_gpu,
)


class TestLazyBenchmark(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def is_not_finalized(self):
        self.assertEqual(
            counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0
        )

    def is_singly_finalized(self):
        self.assertEqual(
            counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1
        )

    def test_lazyness(self):
        # test LazyBenchmark should not finalize on creation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    def test_timing_ms(self):
        # test LazyBenchmark.timing_ms implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark.timing_ms, 0.0)
        self.is_singly_finalized()

    def test_timing_ms_cached(self):
        # test LazyBenchmark.timing_ms is cached
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark.timing_ms, 0.0)
        self.assertEqual(lazy_benchmark.timing_ms, 0.0)
        self.is_singly_finalized()

    def test_float(self):
        # test LazyBenchmark.__float__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark), 0.0)

    def test_format(self):
        # test LazyBenchmark.__format__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(f"{lazy_benchmark}", "0.0")

    def test_str(self):
        # test LazyBenchmark.__str__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(str(lazy_benchmark), "0.0")

    def test_lt(self):
        # test LazyBenchmark.__lt__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark < 1.0, True)

    def test_le(self):
        # test LazyBenchmark.__le__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark < 1.0, True)

    def test_gt(self):
        # test LazyBenchmark.__gt__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark > -1.0, True)

    def test_ge(self):
        # test LazyBenchmark.__ge__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark >= -1.0, True)

    def test_add(self):
        # test LazyBenchmark.__add__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark + 1), 1.0)

    def test_add_lazyness(self):
        # test LazyBenchmark.__add__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) + 1
        self.is_not_finalized()

    def test_radd(self):
        # test LazyBenchmark.__radd__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(1 + lazy_benchmark), 1.0)

    def test_radd_lazyness(self):
        # test LazyBenchmark.__radd__ does not finalize
        lazy_benchmark = 1 + LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    def test_sub(self):
        # test LazyBenchmark.__sub__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark - 1), -1.0)

    def test_sub_lazyness(self):
        # test LazyBenchmark.__sub__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) - 1
        self.is_not_finalized()

    def test_rsub(self):
        # test LazyBenchmark.__rsub__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(1 - lazy_benchmark), 1.0)

    def test_rsub_lazyness(self):
        # test LazyBenchmark.__rsub__ does not finalize
        lazy_benchmark = 1 - LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    def test_mul(self):
        # test LazyBenchmark.__mul__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark * 1), 0.0)

    def test_mul_lazyness(self):
        # test LazyBenchmark.__mul__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) * 1
        self.is_not_finalized()

    def test_rmul(self):
        # test LazyBenchmark.__rmul__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(1 * lazy_benchmark), 0.0)

    def test_rmul_lazyness(self):
        # test LazyBenchmark.__rmul__ does not finalize
        lazy_benchmark = 1 * LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    def test_truediv(self):
        # test LazyBenchmark.__truediv__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark / 1), 0.0)

    def test_truediv_lazyness(self):
        # test LazyBenchmark.__truediv__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) / 1
        self.is_not_finalized()

    def test_rtruediv(self):
        # test LazyBenchmark.__rtruediv__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 1.0)
        self.assertEqual(float(0 / lazy_benchmark), 0.0)

    def test_rtruediv_lazyness(self):
        # test LazyBenchmark.__rtruediv__ does not finalize
        lazy_benchmark = 0 / LazyBenchmark(lambda: 1.0)
        self.is_not_finalized()


class TestBenchmarking(TestCase):
    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def get_various_kernels(self, device):
        def make_sum(size):
            fn, args, kwargs = torch.sum, [torch.randn(size, device=device)], {}
            _callable = lambda: fn(*args, **kwargs)
            return (fn, args, kwargs, _callable)

        sums = [make_sum(size) for size in [10, 100, 1000, 10000, 100000]]

        def make_mm(size):
            fn, args, kwargs = (
                torch.mm,
                [
                    torch.randn(size, size, device=device),
                    torch.randn(size, size, device=device),
                ],
                {},
            )
            _callable = lambda: fn(*args, **kwargs)
            return (fn, args, kwargs, _callable)

        mms = [make_mm(size) for size in [32, 64, 128, 256, 512]]

        return [*sums, *mms]

    def diff(self, baseline, experimental):
        return abs(experimental - baseline) / baseline

    def counter_fallback_to_original_benchmarking(self):
        return counters["inductor"]["benchmarking_fallback_to_original_benchmarking"]

    def counter_fallback_to_non_lazy_benchmarking(self):
        return counters["inductor"]["benchmarking_fallback_to_non_lazy_benchmarking"]

    def counter_benchmarking_L2_cache_size(self):
        return counters["inductor"]["benchmarking_L2_cache_size"]

    def counter_benchmarking_gpu_queue_limit(self):
        return counters["inductor"]["benchmarking_gpu_queue_limit"]

    def counter_benchmarking_cpu_launch_overhead_ms_per_event_record(self):
        return counters["inductor"][
            "benchmarking_cpu_launch_overhead_ms_per_event_record"
        ]

    def counter_benchmarking_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear(
        self,
    ):
        return counters["inductor"][
            "benchmarking_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear"
        ]

    def counter_benchmarking_gpu_time_ms_per_gpu_clock_cycle(self):
        return counters["inductor"]["benchmarking_gpu_time_ms_per_gpu_clock_cycle"]

    def counter_benchmarking_early_ranking(self):
        return counters["inductor"]["benchmarking_early_ranking"]

    def counter_benchmarking_earling_pruning(self):
        return counters["inductor"]["benchmarking_earling_pruning"]


class TestBenchmarkingCPU(TestBenchmarking):
    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def get_various_kernels(self):
        return super(TestBenchmarking, self).get_various_kernels(device="cpu")

    def sanity_check(self, _callable, timing_ms):
        start_time_s = time.perf_counter()
        for _ in range(10):
            _callable()
        baseline_timing_ms = ((time.perf_counter() - start_time_s) * 1000) / 10
        self.assertEqual(self.diff(baseline_timing_ms, timing_ms) < 0.25, True)

    def gpu_properties_are_not_initialized(self):
        gpu_properties = [
            self.counter_benchmarking_L2_cache_size(),
            self.counter_benchmarking_gpu_queue_limit(),
            self.counter_benchmarking_cpu_launch_overhead_ms_per_event_record(),
            self.counter_benchmarking_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear(),
            self.counter_benchmarking_gpu_time_ms_per_gpu_clock_cycle(),
        ]
        for gpu_property in gpu_properties:
            self.assertEqual(gpu_property, 0)

    def patches(fn):
        @functools.wraps(fn)
        def wrapped(self):
            counters.clear()
            torch.manual_seed(12345)
            return fn(self, Benchmarker(), self.get_various_kernels())

        return wrapped

    @patches
    def test_benchmark_smoke(self, benchmarker, kernels):
        for fn, args, kwargs, _callable in kernels:
            timing_ms = benchmarker.benchmark(fn, args, kwargs)
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @patches
    def test_benchmark_cpu_smoke(self, benchmarker, kernels):
        for _, _, _, _callable in kernels:
            timing_ms = benchmarker.benchmark_cpu(_callable)
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @patches
    def test_benchmark_many_cpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = benchmarker.benchmark_many_cpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @patches
    def test_lazy_benchmark_cpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = benchmarker.lazy_benchmark_cpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()


class TestBenchmarkingGPU(TestBenchmarking):
    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def get_various_kernels(self):
        return super(TestBenchmarking, self).get_various_kernels(device=GPU_TYPE)
    
    def sanity_check(self, _callable, timing_ms):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(10):
            _callable()
        end_event.record()
        torch.cuda.synchronize()
        roofline_timing_ms = start_event.elapsed_time(end_event) / 10
        self.assertEqual(timing_ms <= roofline_timing_ms, True)

    def patches(fn):
        @requires_gpu()
        @functools.wraps(fn)
        def wrapped(self):
            counters.clear()
            torch.manual_seed(12345)
            return fn(self, Benchmarker(), self.get_various_kernels())

        return wrapped

    @patches
    def test_benchmark_smoke(self, benchmarker, kernels):
        for fn, args, kwargs, _callable in kernels:
            timing_ms = benchmarker.benchmark(fn, args, kwargs)
            self.sanity_check(_callable, timing_ms)

    @patches
    def test_benchmark_gpu_smoke(self, benchmarker, kernels):
        for _, _, _, _callable in kernels:
            timing_ms = benchmarker.benchmark_gpu(_callable)
            self.sanity_check(_callable, timing_ms)

    @patches
    def test_benchmark_many_gpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = benchmarker.benchmark_many_gpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)

    @patches
    def test_lazy_benchmark_cpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = benchmarker.lazy_benchmark_gpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)


if __name__ == "__main__":
    run_tests()
