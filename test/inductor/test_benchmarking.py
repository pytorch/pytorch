# Owner(s): ["module: inductor"]

import functools
import time

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import Benchmarker, LazyBenchmark
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


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


def cpu_patches(fn):
    @config.patch({"benchmarking.fallback_to_original_benchmarking": False})
    @config.patch({"benchmarking.enable_lazy_benchmarking": True})
    @functools.wraps(fn)
    def wrapped(self):
        counters.clear()
        torch.manual_seed(12345)
        return fn(self, Benchmarker(), self.get_various_kernels())

    return wrapped


class TestBenchmarking(TestCase):
    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def get_various_kernels_by_device(self, device):
        def make_sum(size):
            fn, args, kwargs = torch.sum, [torch.randn(size, device=device)], {}
            _callable = lambda: fn(*args, **kwargs)  # noqa: E731
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
            _callable = lambda: fn(*args, **kwargs)  # noqa: E731
            return (fn, args, kwargs, _callable)

        mms = [make_mm(size) for size in [32, 64, 128, 256, 512]]

        return [*sums, *mms]

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

    def counter_benchmarking_early_pruning(self):
        return counters["inductor"]["benchmarking_early_pruning"]


class TestBenchmarkingCPU(TestBenchmarking):
    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def get_various_kernels(self):
        return self.get_various_kernels_by_device(device="cpu")

    def sanity_check(self, _callable, timing_ms):
        start_time_s = time.perf_counter()
        for _ in range(10):
            _callable()
        baseline_timing_ms = ((time.perf_counter() - start_time_s) * 1000) / 10
        self.assertEqual(baseline_timing_ms, timing_ms, atol=1, rtol=0.25)

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

    @cpu_patches
    def test_benchmark_smoke(self, benchmarker, kernels):
        for fn, args, kwargs, _callable in kernels:
            timing_ms = benchmarker.benchmark(fn, args, kwargs)
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @cpu_patches
    def test_benchmark_cpu_smoke(self, benchmarker, kernels):
        for _, _, _, _callable in kernels:
            timing_ms = benchmarker.benchmark_cpu(_callable)
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @cpu_patches
    def test_benchmark_many_cpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = benchmarker.benchmark_many_cpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @cpu_patches
    def test_lazy_benchmark_smoke(self, benchmarker, kernels):
        fn_args_kwargs_list = [(fn, args, kwargs) for fn, args, kwargs, _ in kernels]
        timings_ms = [
            benchmarker.lazy_benchmark(*fn_args_kwargs)
            for fn_args_kwargs in fn_args_kwargs_list
        ]
        timings_ms = [float(timing_ms) for timing_ms in timings_ms]
        callables = [_callable for _, _, _, _callable in kernels]
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @cpu_patches
    def test_lazy_benchmark_cpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = [
            benchmarker.lazy_benchmark_cpu(_callable) for _callable in callables
        ]
        timings_ms = [float(timing_ms) for timing_ms in timings_ms]
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()


def gpu_patches(fn):
    @requires_gpu()
    @config.patch({"benchmarking.fallback_to_original_benchmarking": False})
    @config.patch({"benchmarking.enable_lazy_benchmarking": True})
    @config.patch({"benchmarking.enable_early_ranking": True})
    @config.patch({"benchmarking.enable_early_pruning": True})
    @functools.wraps(fn)
    def wrapped(self):
        counters.clear()
        torch.manual_seed(12345)
        return fn(self, Benchmarker(), self.get_various_kernels())

    return wrapped


class TestBenchmarkingGPU(TestBenchmarking):
    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def get_various_kernels(self):
        return self.get_various_kernels_by_device(device=GPU_TYPE)

    def sanity_check(self, _callable, timing_ms):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(10):
            _callable()
        end_event.record()
        torch.cuda.synchronize()
        baseline_timing_ms = start_event.elapsed_time(end_event) / 10
        if timing_ms > baseline_timing_ms:
            self.assertEqual(baseline_timing_ms, timing_ms, atol=0.1, rtol=0.25)

    @gpu_patches
    def test_benchmark_smoke(self, benchmarker, kernels):
        for fn, args, kwargs, _callable in kernels:
            timing_ms = benchmarker.benchmark(fn, args, kwargs)
            self.sanity_check(_callable, timing_ms)

    @gpu_patches
    def test_benchmark_gpu_smoke(self, benchmarker, kernels):
        for _, _, _, _callable in kernels:
            timing_ms = benchmarker.benchmark_gpu(_callable)
            self.sanity_check(_callable, timing_ms)

    @gpu_patches
    def test_benchmark_many_gpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = benchmarker.benchmark_many_gpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)

    @gpu_patches
    def test_lazy_benchmark_smoke(self, benchmarker, kernels):
        fn_args_kwargs_list = [(fn, args, kwargs) for fn, args, kwargs, _ in kernels]
        timings_ms = [
            benchmarker.lazy_benchmark(*fn_args_kwargs)
            for fn_args_kwargs in fn_args_kwargs_list
        ]
        timings_ms = [float(timing_ms) for timing_ms in timings_ms]
        callables = [_callable for _, _, _, _callable in kernels]
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.gpu_properties_are_not_initialized()

    @gpu_patches
    def test_lazy_benchmark_gpu_smoke(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = [
            benchmarker.lazy_benchmark_gpu(_callable) for _callable in callables
        ]
        timings_ms = [float(timing_ms) for timing_ms in timings_ms]
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)

    @gpu_patches
    def test_benchmark_many_gpu_single_callable(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels][:1]
        timing_ms = benchmarker.benchmark_many_gpu(callables)[0]
        self.sanity_check(callables[0], timing_ms)

    @gpu_patches
    def test_benchmark_many_gpu_early_ranking(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = benchmarker.benchmark_many_gpu(callables, ranking_key="test")
        self.assertEqual(self.counter_benchmarking_early_ranking(), 1)

    @gpu_patches
    @config.patch({"benchmarking.enable_early_ranking": False})
    def test_benchmark_many_gpu_early_ranking_disabled(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = benchmarker.benchmark_many_gpu(callables, ranking_key="test")
        self.assertEqual(self.counter_benchmarking_early_ranking(), 0)

    @gpu_patches
    def test_benchmark_many_gpu_early_pruning(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = benchmarker.benchmark_many_gpu(callables, pruning_key="test")
        self.assertEqual(self.counter_benchmarking_early_pruning() >= 1, True)

    @gpu_patches
    @config.patch({"benchmarking.enable_early_pruning": False})
    def test_benchmark_many_gpu_early_pruning_disabled(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = benchmarker.benchmark_many_gpu(callables, pruning_key="test")
        self.assertEqual(self.counter_benchmarking_early_pruning(), 0)

    @gpu_patches
    @config.patch({"benchmarking.enable_lazy_benchmarking": False})
    def test_lazy_benchmark_gpu_disabled(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        timings_ms = [
            benchmarker.lazy_benchmark_gpu(_callable) for _callable in callables
        ]
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check(_callable, timing_ms)
        self.assertEqual(
            self.counter_fallback_to_non_lazy_benchmarking(), len(callables)
        )

    @gpu_patches
    @config.patch({"benchmarking.fallback_to_original_benchmarking": True})
    def test_benchmark_fallback(self, benchmarker, kernels):
        for fn, args, kwargs, _ in kernels:
            _ = benchmarker.benchmark(fn, args, kwargs)
        self.assertEqual(self.counter_fallback_to_original_benchmarking(), len(kernels))

    @gpu_patches
    @config.patch({"benchmarking.fallback_to_original_benchmarking": True})
    def test_benchmark_gpu_fallback(self, benchmarker, kernels):
        for _, _, _, _callable in kernels:
            _ = benchmarker.benchmark_gpu(_callable)
        self.assertEqual(self.counter_fallback_to_original_benchmarking(), len(kernels))

    @gpu_patches
    @config.patch({"benchmarking.fallback_to_original_benchmarking": True})
    def test_benchmark_many_gpu_fallback(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = benchmarker.benchmark_many_gpu(callables)
        self.assertEqual(self.counter_fallback_to_original_benchmarking(), 1)

    @gpu_patches
    @config.patch({"benchmarking.fallback_to_original_benchmarking": True})
    def test_lazy_benchmark_gpu_fallback(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = [benchmarker.lazy_benchmark_gpu(_callable) for _callable in callables]
        self.assertEqual(self.counter_fallback_to_original_benchmarking(), len(kernels))

    @gpu_patches
    def test_lazy_benchmark_gpu_same_callable_no_memory_leak(
        self, benchmarker, kernels
    ):
        callables = [kernels[0][3] for _ in range(len(kernels))]
        timings_ms = [
            benchmarker.lazy_benchmark_gpu(_callable) for _callable in callables
        ]
        _ = float(timings_ms[0])
        self.assertEqual(benchmarker.kwargs_hash_to_futures_gpu, {})

    @gpu_patches
    def test_lazy_benchmark_gpu_group_by_kwargs(self, benchmarker, kernels):
        callables = [_callable for _, _, _, _callable in kernels]
        _ = [
            benchmarker.lazy_benchmark_gpu(_callable, estimation_iters=10 * idx)
            for idx, _callable in enumerate(callables)
        ]
        self.assertEqual(len(benchmarker.kwargs_hash_to_futures_gpu), len(kernels))

    @gpu_patches
    def test_L2_cache_size(self, benchmarker, *args):
        self.assertEqual(benchmarker.L2_cache_size > 0, True)

    @gpu_patches
    def test_gpu_queue_limit(self, benchmarker, *args):
        gpu_queue_limit = benchmarker.gpu_queue_limit
        torch.cuda.synchronize()
        torch.cuda._sleep(
            int(
                (benchmarker.cpu_launch_overhead_ms_per_event_record * gpu_queue_limit)
                / benchmarker.gpu_time_ms_per_gpu_clock_cycle
            )
        )
        for _ in range(gpu_queue_limit - 1):
            start_time_s = time.perf_counter()
            torch.cuda.Event(enable_timing=True).record()
            elapsed_time_ms = (time.perf_counter() - start_time_s) * 100
            self.assertEqual(elapsed_time_ms < 1)
        start_time_s = time.perf_counter()
        torch.cuda.Event(enable_timing=True).record()
        elapsed_time_ms = (time.perf_counter() - start_time_s) * 1000
        torch.cuda.synchronize()
        self.assertEqual(elapsed_time_ms > 1)

    @gpu_patches
    def test_cpu_launch_overhead_ms_per_event_record(self, benchmarker, *args):
        torch.cuda.synchronize()
        start_time_s = time.perf_counter()
        for _ in range(1000):
            torch.cuda.Event(enable_timing=True).record()
        elapsed_time_ms = (time.perf_counter() - start_time_s) * 1000
        torch.cuda.synchronize()
        self.assertEqual(
            benchmarker.cpu_launch_overhead_ms_per_event_record,
            elapsed_time_ms / 1000,
            atol=0.1,
            rtol=0.25,
        )

    @gpu_patches
    def test_cpu_launch_overhead_ms_per_gpu_cache_clear(self, benchmarker, *args):
        buffer = torch.empty(
            int(benchmarker.L2_cache_size // 4), dtype=torch.int, device="cuda"
        )
        torch.cuda.synchronize()
        start_time_s = time.perf_counter()
        for _ in range(100):
            buffer.zero_()
        elapsed_time_ms = (time.perf_counter() - start_time_s) * 1000
        torch.cuda.synchronize()
        self.assertEqual(
            benchmarker.cpu_launch_overhead_ms_per_gpu_cache_clear,
            elapsed_time_ms / 100,
            atol=0.1,
            rtol=0.25,
        )

    @gpu_patches
    def test_gpu_time_ms_per_gpu_cache_clear(self, benchmarker, *args):
        buffer = torch.empty(
            int(benchmarker.L2_cache_size // 4), dtype=torch.int, device="cuda"
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        torch.cuda._sleep(
            int(
                (benchmarker.cpu_launch_overhead_ms_per_gpu_cache_clear * 100)
                / benchmarker.gpu_time_ms_per_gpu_clock_cycle
            )
        )
        start_event.record()
        for _ in range(100):
            buffer.zero_()
        end_event.record()
        torch.cuda.synchronize()
        self.assertEqual(
            benchmarker.gpu_time_ms_per_gpu_cache_clear,
            start_event.elapsed_time(end_event) / 100,
            atol=0.1,
            rtol=0.25,
        )

    @gpu_patches
    def test_gpu_time_ms_per_gpu_clock_cycle(self, benchmarker, *args):
        gpu_clock_cycles_to_sleep = int(
            100 / benchmarker.gpu_time_ms_per_gpu_clock_cycle
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        torch.cuda._sleep(gpu_clock_cycles_to_sleep)
        end_event.record()
        torch.cuda.synchronize()
        self.assertEqual(
            benchmarker.gpu_time_ms_per_gpu_clock_cycle,
            start_event.elapsed_time(end_event) / gpu_clock_cycles_to_sleep,
            atol=0.1,
            rtol=0.25,
        )


if __name__ == "__main__":
    run_tests()
