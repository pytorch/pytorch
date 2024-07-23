# Owner(s): ["module: inductor"]

import functools
import itertools
import logging
import time

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import Benchmarker, LazyBenchmark
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA,
    requires_gpu,
)


log = logging.getLogger(__name__)


def patches(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


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

    @patches
    def test_lazyness(self):
        # test LazyBenchmark should not finalize on creation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    @patches
    def test_timing_ms(self):
        # test LazyBenchmark.timing_ms implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark.timing_ms, 0.0)
        self.is_singly_finalized()

    @patches
    def test_timing_ms_cached(self):
        # test LazyBenchmark.timing_ms is cached
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark.timing_ms, 0.0)
        self.assertEqual(lazy_benchmark.timing_ms, 0.0)
        self.is_singly_finalized()

    @patches
    def test_float(self):
        # test LazyBenchmark.__float__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark), 0.0)

    @patches
    def test_format(self):
        # test LazyBenchmark.__format__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(f"{lazy_benchmark}", "0.0")

    @patches
    def test_str(self):
        # test LazyBenchmark.__str__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(str(lazy_benchmark), "0.0")

    @patches
    def test_lt(self):
        # test LazyBenchmark.__lt__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark < 1.0, True)

    @patches
    def test_le(self):
        # test LazyBenchmark.__le__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark < 1.0, True)

    @patches
    def test_gt(self):
        # test LazyBenchmark.__gt__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark > -1.0, True)

    @patches
    def test_ge(self):
        # test LazyBenchmark.__ge__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(lazy_benchmark >= -1.0, True)

    @patches
    def test_add(self):
        # test LazyBenchmark.__add__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark + 1), 1.0)

    @patches
    def test_add_lazyness(self):
        # test LazyBenchmark.__add__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) + 1
        self.is_not_finalized()

    @patches
    def test_radd(self):
        # test LazyBenchmark.__radd__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(1 + lazy_benchmark), 1.0)

    @patches
    def test_radd_lazyness(self):
        # test LazyBenchmark.__radd__ does not finalize
        lazy_benchmark = 1 + LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    @patches
    def test_sub(self):
        # test LazyBenchmark.__sub__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark - 1), -1.0)

    @patches
    def test_sub_lazyness(self):
        # test LazyBenchmark.__sub__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) - 1
        self.is_not_finalized()

    @patches
    def test_rsub(self):
        # test LazyBenchmark.__rsub__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(1 - lazy_benchmark), 1.0)

    @patches
    def test_rsub_lazyness(self):
        # test LazyBenchmark.__rsub__ does not finalize
        lazy_benchmark = 1 - LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    @patches
    def test_mul(self):
        # test LazyBenchmark.__mul__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark * 1), 0.0)

    @patches
    def test_mul_lazyness(self):
        # test LazyBenchmark.__mul__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) * 1
        self.is_not_finalized()

    @patches
    def test_rmul(self):
        # test LazyBenchmark.__rmul__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(1 * lazy_benchmark), 0.0)

    @patches
    def test_rmul_lazyness(self):
        # test LazyBenchmark.__rmul__ does not finalize
        lazy_benchmark = 1 * LazyBenchmark(lambda: 0.0)
        self.is_not_finalized()

    @patches
    def test_truediv(self):
        # test LazyBenchmark.__truediv__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 0.0)
        self.assertEqual(float(lazy_benchmark / 1), 0.0)

    @patches
    def test_truediv_lazyness(self):
        # test LazyBenchmark.__truediv__ does not finalize
        lazy_benchmark = LazyBenchmark(lambda: 0.0) / 1
        self.is_not_finalized()

    @patches
    def test_rtruediv(self):
        # test LazyBenchmark.__rtruediv__ implementation
        lazy_benchmark = LazyBenchmark(lambda: 1.0)
        self.assertEqual(float(0 / lazy_benchmark), 0.0)

    @patches
    def test_rtruediv_lazyness(self):
        # test LazyBenchmark.__rtruediv__ does not finalize
        lazy_benchmark = 0 / LazyBenchmark(lambda: 1.0)
        self.is_not_finalized()


class TestBenchmarking(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def get_fn_args_kwargs_callable(self, device):
        fn, args, kwargs = torch.sum, [torch.randn(1000, device=device)], {}
        return fn, args, kwargs, lambda: fn(*args, **kwargs)

    def sanity_check_cpu_benchmark(self, _callable, timing_ms):
        start_time_s = time.perf_counter()
        for _ in range(10):
            _callable()
        baseline_timing_ms = ((time.perf_counter() - start_time_s) * 1000) / 10
        self.assertEqual(self.diff(baseline_timing_ms, timing_ms) < 0.25, True)

    def sanity_check_gpu_benchmark(self, _callable, timing_ms):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(10):
            _callable()
        end_event.record()
        torch.cuda.synchronize()
        roofline_timing_ms = start_event.elapsed_time(end_event) / 10
        self.assertEqual(timing_ms <= roofline_timing_ms, True)

    def diff(baseline, experimental):
        return abs(experimental - baseline) / baseline

    def gpu_properties_are_not_initialized(self):
        self.assertEqual(counters["inductor"]["benchmarking_L2_cache_size"], 0)
        self.assertEqual(counters["inductor"]["benchmarking_gpu_queue_limit"], 0)
        self.assertEqual(
            counters["inductor"][
                "benchmarking_cpu_launch_overhead_ms_per_event_record"
            ],
            0,
        )
        self.assertEqual(
            counters["inductor"][
                "benchmarking_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear"
            ],
            0,
        )
        self.assertEqual(
            counters["inductor"]["benchmarking_gpu_time_ms_per_gpu_clock_cycle"], 0
        )

    def gpu_properties_are_singly_initialized(self):
        self.assertEqual(counters["inductor"]["benchmarking_L2_cache_size"], 1)
        self.assertEqual(counters["inductor"]["benchmarking_gpu_queue_limit"], 1)
        self.assertEqual(
            counters["inductor"][
                "benchmarking_cpu_launch_overhead_ms_per_event_record"
            ],
            1,
        )
        self.assertEqual(
            counters["inductor"][
                "benchmarking_cpu_launch_overhead_ms_and_gpu_time_ms_per_gpu_cache_clear"
            ],
            1,
        )
        self.assertEqual(
            counters["inductor"]["benchmarking_gpu_time_ms_per_gpu_clock_cycle"], 1
        )

    @requires_gpu()
    @patches
    def test_L2_cache_size(self):
        benchmarker = Benchmarker()
        self.assertEqual(benchmarker.L2_cache_size > 0)

    @requires_gpu()
    @patches
    def test_gpu_queue_limit(self):
        benchmarker = Benchmarker()
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

    @requires_gpu()
    @patches
    def test_cpu_launch_overhead_ms_per_event_record(self):
        benchmarker = Benchmarker()
        torch.cuda.synchronize()
        start_time_s = time.perf_counter()
        for _ in range(1000):
            torch.cuda.Event(enable_timing=True).record()
        elapsed_time_ms = (time.perf_counter() - start_time_s) * 1000
        torch.cuda.synchronize()
        self.assertEqual(
            self.diff(
                benchmarker.cpu_launch_overhead_ms_per_event_record,
                elapsed_time_ms / 1000,
            )
            < 0.25,
            True,
        )

    @requires_gpu()
    @patches
    def test_cpu_launch_overhead_ms_per_gpu_cache_clear(self):
        benchmarker = Benchmarker()
        buffer = torch.empty(
            int(self.L2_cache_size // 4), dtype=torch.int, device="cuda"
        )
        torch.cuda.synchronize()
        start_time_s = time.perf_counter()
        for _ in range(100):
            buffer.zero_()
        elapsed_time_ms = (time.perf_counter() - start_time_s) * 1000
        torch.cuda.synchronize()
        self.assertEqual(
            self.diff(
                benchmarker.cpu_launch_overhead_ms_per_gpu_cache_clear,
                elapsed_time_ms / 100,
            )
            < 0.25,
            True,
        )

    @requires_gpu()
    @patches
    def test_gpu_time_ms_per_gpu_cache_clear(self):
        benchmarker = Benchmarker()
        buffer = torch.empty(
            int(self.L2_cache_size // 4), dtype=torch.int, device="cuda"
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
            self.diff(
                benchmarker.gpu_time_ms_per_gpu_cache_clear,
                start_event.elapsed_time(end_event) / 100,
            )
            < 0.25,
            True,
        )

    @requires_gpu()
    @patches
    def test_gpu_time_ms_per_gpu_clock_cycle(self):
        benchmarker = Benchmarker()
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
            self.diff(
                benchmarker.gpu_time_ms_per_gpu_clock_cycle,
                start_event.elapsed_time(end_event) / gpu_clock_cycles_to_sleep,
            )
            < 0.25,
            True,
        )

    @requires_gpu()
    @patches
    def test_benchmark_validity(self):
        benchmarker = Benchmarker()

        # test CPU path
        fn, args, kwargs, _callable = self.get_fn_args_kwargs_callable(device="cpu")
        timing_ms = benchmarker.benchmark(fn, args, kwargs)
        self.sanity_check_cpu_benchmark(_callable, timing_ms)

        # test GPU path
        fn, args, kwargs, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        timing_ms = benchmarker.benchmark(fn, args, kwargs)
        self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @patches
    def test_benchmark_cpu_validity(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cpu")
        timing_ms = benchmarker.benchmark_cpu(_callable)
        self.sanity_check_cpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    def test_benchmark_gpu_validity(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        timing_ms = benchmarker.benchmark_gpu(_callable)
        self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @patches
    def test_benchmark_many_cpu_validity(self):
        benchmarker = Benchmarker()
        callables = [
            self.get_fn_args_kwargs_callable(device="cpu")[3] for _ in range(10)
        ]
        timings_ms = benchmarker.benchmark_many_cpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check_cpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    def test_benchmark_many_gpu_validity(self):
        benchmarker = Benchmarker()
        callables = [
            self.get_fn_args_kwargs_callable(device="cuda")[3] for _ in range(10)
        ]
        timings_ms = benchmarker.benchmark_many_gpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    def test_lazy_benchmark_gpu_validity(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        timing_ms = float(benchmarker.lazy_benchmark_gpu(_callable))
        self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    def test_benchmark_many_gpu_single_callable(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        timing_ms = benchmarker.benchmark_many_gpu([_callable])[0]
        self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    def test_benchmark_many_gpu_callables_not_mangled(self):
        benchmarker = Benchmarker()
        fast_fn, fast_args, fast_kwargs = (
            torch.sum,
            [torch.randn(1000, device="cuda")],
            {},
        )
        fast_callable = lambda: fast_fn(*fast_args, **fast_kwargs)
        slow_fn, slow_args, slow_kwargs = (
            torch.mm,
            [
                torch.randn(4096, 4096, device="cuda"),
                torch.randn(4096, 4096, device="cuda"),
            ],
            {},
        )
        slow_callable = lambda: slow_fn(*slow_args, **slow_kwargs)
        callables = [
            fast_callable,
            fast_callable,
            slow_callable,
            slow_callable,
            fast_callable,
            slow_callable,
        ]
        timings_ms = benchmarker.benchmark_many_gpu(callables)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_early_ranking": True})
    def test_benchmark_many_gpu_early_ranking(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        callables = [_callable for _ in range(10)]
        timings_ms = benchmarker.benchmark_many_gpu(callables, ranking_key="test")
        self.assertEqual(counters["inductor"]["benchmarking_early_ranking"], 1)
        self.assertEqual(
            max(
                [
                    self.diff(*paired_timings_ms)
                    for paired_timings_ms in itertools.product(timings_ms)
                ]
            )
            < 0.25
        )

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_early_ranking": False})
    def test_benchmark_many_gpu_early_ranking_disabled(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        callables = [_callable for _ in range(10)]
        _ = benchmarker.benchmark_many_gpu(callables, ranking_key="test")
        self.assertEqual(counters["inductor"]["benchmarking_early_ranking"], 0)

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_early_pruning": True})
    def test_benchmark_many_gpu_early_pruning(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        callables = [_callable for _ in range(10)]
        timings_ms = benchmarker.benchmark_many_gpu(callables, pruning_key="test")
        self.assertEqual(counters["inductor"]["benchmarking_early_pruning"], 1)
        for _callable, timing_ms in zip(callables, timings_ms):
            self.sanity_check_gpu_benchmark(_callable, timing_ms)

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_early_pruning": False})
    def test_benchmark_many_gpu_early_pruning_disabled(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        callables = [_callable for _ in range(10)]
        _ = benchmarker.benchmark_many_gpu(callables, pruning_key="test")
        self.assertEqual(counters["inductor"]["benchmarking_early_pruning"], 0)

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_lazy_benchmarking": True})
    def test_lazy_benchmark_gpu(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        lazy_benchmark = benchmarker.lazy_benchmark_gpu(_callable)
        self.assertEqual(counters["inductor"]["benchmarking_lazy_benchmark"], 1)
        self.assertEqual(
            counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 0
        )
        _ = float(lazy_benchmark)
        self.assertEqual(
            counters["inductor"]["benchmarking_finalize_lazy_benchmark"], 1
        )

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_lazy_benchmarking": False})
    def test_lazy_benchmark_gpu_disabled(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        _ = benchmarker.lazy_benchmark_gpu(_callable)
        self.assertEqual(counters["inductor"]["benchmarking_lazy_benchmark"], 0)

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_lazy_benchmarking": True})
    def test_lazy_benchmark_gpu_same_callable_no_memory_leak(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        lazy_benchmark = benchmarker.lazy_benchmark_gpu(_callable)
        _ = benchmarker.lazy_benchmark_gpu(_callable)
        _ = float(lazy_benchmark)
        self.assertEqual(benchmarker.kwargs_hash_to_futures_gpu, {})

    @requires_gpu()
    @patches
    @config.patch({"benchmarking.enable_lazy_benchmarking": True})
    def test_lazy_benchmark_gpu_group_by_kwargs(self):
        benchmarker = Benchmarker()
        _, _, _, _callable = self.get_fn_args_kwargs_callable(device="cuda")
        lazy_benchmark = benchmarker.lazy_benchmark_gpu(_callable)
        _ = benchmarker.lazy_benchmark_gpu(_callable, estimation_iters=100)
        _ = float(lazy_benchmark)
        self.assertEqual(len(benchmarker.memory_cache), 1)


if __name__ == "__main__":
    run_tests()
