# Owner(s): ["module: inductor"]

import contextlib
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._inductor.config import (
    inductor_default_autotune_rep,
    inductor_default_autotune_warmup,
)
from torch._inductor.runtime.benchmarking import (
    Benchmarker,
    TorchProfilerBenchmarker,
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

    def test_default_profiler_benchmarker_selection_supports_xpu_only(self):
        from torch._inductor.runtime import benchmarking as _bench

        with (
            patch.object(
                _bench.inductor_config, "use_torch_profiler_benchmarker", True
            ),
            patch.object(_bench.inductor_config, "use_experimental_benchmarker", False),
            patch.object(
                _bench.torch.cuda,
                "is_available",
                side_effect=AssertionError("should not query CUDA availability"),
            ),
            patch.object(
                _bench.torch.xpu,
                "is_available",
                side_effect=AssertionError("should not query XPU availability"),
            ),
            patch.object(
                _bench.torch.mtia,
                "is_available",
                side_effect=AssertionError("should not query MTIA availability"),
            ),
        ):
            self.assertIsInstance(
                _bench._make_default_benchmarker(), TorchProfilerBenchmarker
            )

    def test_profiler_benchmarker_fallback_uses_correlated_device_events(self):
        from torch._inductor.runtime import benchmarking as _bench
        from torch.autograd import DeviceType

        class FakeProfilerEvent:
            def __init__(self, name, device_type, event_id, cpu_children=()):
                self.name = name
                self.device_type = device_type
                self.id = event_id
                self.cpu_children = list(cpu_children)

        class FakeKinetoEvent:
            def __init__(
                self,
                name,
                device_type,
                linked_correlation_id,
                correlation_id,
                activity_type,
                start_ns,
                end_ns,
            ):
                self._name = name
                self._device_type = device_type
                self._linked_correlation_id = linked_correlation_id
                self._correlation_id = correlation_id
                self._activity_type = activity_type
                self._start_ns = start_ns
                self._end_ns = end_ns

            def name(self):
                return self._name

            def device_type(self):
                return self._device_type

            def linked_correlation_id(self):
                return self._linked_correlation_id

            def correlation_id(self):
                return self._correlation_id

            def activity_type(self):
                return self._activity_type

            def start_ns(self):
                return self._start_ns

            def end_ns(self):
                return self._end_ns

        child_event = FakeProfilerEvent("aten::sum", DeviceType.CPU, 11)
        profiler_events = [
            FakeProfilerEvent(
                _bench._CALLABLE_PROFILE_EVENT_NAME,
                DeviceType.CPU,
                7,
                (child_event,),
            )
        ]
        kineto_events = [
            FakeKinetoEvent("unrelated", DeviceType.XPU, 0, 99, "kernel", 0, 1000),
            FakeKinetoEvent(
                "xpu_kernel", DeviceType.XPU, 11, 101, "kernel", 1000, 6000
            ),
            FakeKinetoEvent(
                _bench._CALLABLE_PROFILE_EVENT_NAME,
                DeviceType.XPU,
                7,
                7,
                "gpu_user_annotation",
                1000,
                9000,
            ),
        ]

        self.assertEqual(
            _bench._get_callable_device_kernel_time_us(
                kineto_events, profiler_events, 1, DeviceType.XPU
            ),
            5.0,
        )

    def test_gpu_benchmark_lock_without_registered_context_is_noop(self):
        from torch._inductor.runtime import benchmarking as _bench

        with patch(
            "torch.cuda.current_device",
            side_effect=AssertionError("should not query device"),
        ):
            with _bench.maybe_gpu_benchmark_lock():
                pass

    def test_gpu_benchmark_lock_uses_registered_context(self):
        from torch._inductor.runtime import benchmarking as _bench

        calls = []

        @contextlib.contextmanager
        def custom_context():
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with patch(
                "torch.cuda.current_device",
                side_effect=AssertionError("custom context should not query device"),
            ):
                with _bench.maybe_gpu_benchmark_lock():
                    calls.append("body")
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(calls, ["enter", "body", "exit"])

    def test_set_gpu_benchmark_lock_context_returns_previous_context(self):
        from torch._inductor.runtime import benchmarking as _bench

        @contextlib.contextmanager
        def first_context():
            yield

        @contextlib.contextmanager
        def second_context():
            yield

        previous = _bench.set_gpu_benchmark_lock_context(first_context)
        try:
            self.assertIs(
                _bench.set_gpu_benchmark_lock_context(second_context),
                first_context,
            )
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

    def test_do_bench_using_profiling_uses_gpu_benchmark_lock(self):
        from torch._inductor import utils as inductor_utils
        from torch._inductor.runtime import benchmarking as _bench

        calls = []

        @contextlib.contextmanager
        def custom_context():
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        def fake_do_bench(fn, warmup, rep, is_vetted_benchmarking):
            calls.append((warmup, rep, is_vetted_benchmarking))
            fn()
            return 3.0

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with patch.object(
                inductor_utils,
                "_do_bench_using_profiling",
                side_effect=fake_do_bench,
            ):
                result = inductor_utils.do_bench_using_profiling(
                    lambda: calls.append("fn"),
                    warmup=1,
                    rep=2,
                    is_vetted_benchmarking=True,
                )
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(result, 3.0)
        self.assertEqual(calls, ["enter", (1, 2, True), "fn", "exit"])

    def test_benchmark_gpu_with_cuda_graph_uses_gpu_benchmark_lock(self):
        from torch._inductor.runtime import benchmarking as _bench

        class FakeCUDAGraph:
            def replay(self):
                calls.append("replay")

        class FakeBenchmarker(Benchmarker):
            @_bench.gpu_benchmark_lock
            def benchmark_gpu(self, _callable, **kwargs):
                calls.append("benchmark_gpu")
                _callable()
                return 9.0

        benchmarker = FakeBenchmarker()
        calls = []

        @contextlib.contextmanager
        def custom_context():
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.CUDAGraph", FakeCUDAGraph),
                patch("torch.cuda.graph", return_value=contextlib.nullcontext()),
            ):
                result = benchmarker.benchmark_gpu_with_cuda_graph(
                    lambda: calls.append("call")
                )
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(result, 9.0)
        self.assertEqual(
            calls,
            [
                "enter",
                "call",
                "call",
                "enter",
                "benchmark_gpu",
                "replay",
                "exit",
                "exit",
            ],
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @parametrize(
        "hip_value, expected_buffer_size_bytes",
        ((None, 1024), ("mock-hip", 256 * 1024 * 1024)),
    )
    def test_torch_profiler_benchmarker_reuses_inductor_helpers(
        self, hip_value, expected_buffer_size_bytes, device=GPU_TYPE
    ):
        benchmarker = TorchProfilerBenchmarker()
        benchmarker.__dict__["L2_cache_size"] = 1024
        _, _callable = self.make_params(device, size=16)

        captured_buffer_lengths = []
        captured_buffer_devices = []
        original_empty = torch.empty

        def empty_spy(*args, **kwargs):
            captured_buffer_lengths.append(args[0])
            captured_buffer_devices.append(kwargs["device"])
            return original_empty(*args, **kwargs)

        with patch.object(
            benchmarker,
            "get_event_pairs",
            wraps=benchmarker.get_event_pairs,
        ) as mock_get_event_pairs:
            with patch.object(torch.version, "hip", hip_value):
                with patch(
                    "torch._inductor.runtime.benchmarking.torch.empty",
                    side_effect=empty_spy,
                ):
                    timing = benchmarker.benchmark_gpu(
                        _callable,
                        rep=1,
                        estimation_iters=1,
                        memory_warmup_iters=0,
                    )

        self.assertGreater(timing, 0)
        mock_get_event_pairs.assert_called_once_with(1, device_type=device)
        self.assertGreater(len(captured_buffer_lengths), 0)
        self.assertEqual(captured_buffer_lengths[0], expected_buffer_size_bytes // 4)
        self.assertEqual(captured_buffer_devices[0], device)


if __name__ == "__main__":
    run_tests()
