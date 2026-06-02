# Owner(s): ["module: inductor"]

import contextlib
import os
import tempfile
import threading
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

    def test_gpu_benchmark_lock_uses_visible_cuda_device(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        from torch._inductor.runtime import benchmarking as _bench

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "4,7",
            }
            with (
                patch.dict(os.environ, env),
                patch("torch.cuda.current_device", return_value=1),
            ):
                with _bench.maybe_gpu_benchmark_lock():
                    with _bench.maybe_gpu_benchmark_lock():
                        pass

            lock_path = os.path.join(lock_dir, "gpu_7.lock")
            self.assertTrue(os.path.exists(lock_path))
            with open(lock_path) as f:
                metadata = f.read()
            self.assertIn("gpu=7\n", metadata)
            self.assertIn("mode=exclusive\n", metadata)
            self.assertIn("label=inductor_benchmark\n", metadata)

    def test_gpu_benchmark_lock_tracks_nested_device_changes(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        from torch._inductor.runtime import benchmarking as _bench

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "4,7",
            }
            with patch.dict(os.environ, env):
                with patch("torch.cuda.current_device", return_value=0):
                    with _bench.maybe_gpu_benchmark_lock():
                        with patch("torch.cuda.current_device", return_value=1):
                            with _bench.maybe_gpu_benchmark_lock():
                                pass

            self.assertTrue(os.path.exists(os.path.join(lock_dir, "gpu_4.lock")))
            self.assertTrue(os.path.exists(os.path.join(lock_dir, "gpu_7.lock")))

    def test_gpu_benchmark_lock_serializes_same_device_threads(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        from torch._inductor.runtime import benchmarking as _bench

        thread_state = threading.local()
        attempting = [threading.Event(), threading.Event()]
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()
        errors = []

        def current_device():
            return thread_state.device

        def worker(index):
            thread_state.device = 0
            attempting[index].set()
            try:
                with _bench.maybe_gpu_benchmark_lock():
                    entered[index].set()
                    release.wait(timeout=5)
            except Exception as e:
                errors.append(e)

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "4",
            }
            with (
                patch.dict(os.environ, env),
                patch("torch.cuda.current_device", side_effect=current_device),
            ):
                threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
                try:
                    threads[0].start()
                    self.assertTrue(entered[0].wait(timeout=5))
                    threads[1].start()
                    self.assertTrue(attempting[1].wait(timeout=5))
                    self.assertFalse(entered[1].wait(timeout=0.1))
                finally:
                    release.set()
                    for thread in threads:
                        thread.join(timeout=5)

        if errors:
            raise errors[0]
        self.assertTrue(entered[1].is_set())

    def test_gpu_benchmark_lock_allows_different_device_threads(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        from torch._inductor.runtime import benchmarking as _bench

        thread_state = threading.local()
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()
        errors = []

        def current_device():
            return thread_state.device

        def worker(index):
            thread_state.device = index
            try:
                with _bench.maybe_gpu_benchmark_lock():
                    entered[index].set()
                    release.wait(timeout=5)
            except Exception as e:
                errors.append(e)

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "4,7",
            }
            with (
                patch.dict(os.environ, env),
                patch("torch.cuda.current_device", side_effect=current_device),
            ):
                threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
                try:
                    threads[0].start()
                    self.assertTrue(entered[0].wait(timeout=5))
                    threads[1].start()
                    self.assertTrue(entered[1].wait(timeout=5))
                finally:
                    release.set()
                    for thread in threads:
                        thread.join(timeout=5)

        if errors:
            raise errors[0]

    def test_gpu_benchmark_lock_prefers_hip_visible_devices_on_rocm(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        from torch._inductor.runtime import benchmarking as _bench

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "0",
                "HIP_VISIBLE_DEVICES": "2",
            }
            with (
                patch.dict(os.environ, env),
                patch.object(torch.version, "hip", "mock-hip"),
                patch("torch.cuda.current_device", return_value=0),
            ):
                with _bench.maybe_gpu_benchmark_lock():
                    pass

            self.assertTrue(os.path.exists(os.path.join(lock_dir, "gpu_2.lock")))
            self.assertFalse(os.path.exists(os.path.join(lock_dir, "gpu_0.lock")))

    def test_gpu_benchmark_lock_disabled_does_not_query_device(self):
        from torch._inductor.runtime import benchmarking as _bench

        with patch.dict(os.environ, {}, clear=True):
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
            with patch.dict(os.environ, {"INDUCTOR_GPU_BENCH_LOCK": "1"}):
                with patch(
                    "torch.cuda.current_device",
                    side_effect=AssertionError(
                        "custom context should not query device"
                    ),
                ):
                    with _bench.maybe_gpu_benchmark_lock():
                        calls.append("body")
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(calls, ["enter", "body", "exit"])

    def test_gpu_benchmark_lock_registered_context_is_reentrant(self):
        from torch._inductor.runtime import benchmarking as _bench

        calls = []
        locked = False

        @contextlib.contextmanager
        def custom_context():
            nonlocal locked
            self.assertFalse(locked)
            locked = True
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")
                locked = False

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with patch.dict(os.environ, {"INDUCTOR_GPU_BENCH_LOCK": "1"}):
                with _bench.maybe_gpu_benchmark_lock():
                    with _bench.maybe_gpu_benchmark_lock():
                        calls.append("body")
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(calls, ["enter", "body", "exit"])

    def test_gpu_benchmark_lock_registered_context_can_upgrade_outer_context(self):
        from torch._inductor.runtime import benchmarking as _bench

        mode = None
        calls = []

        @contextlib.contextmanager
        def outer_context():
            nonlocal mode
            self.assertIsNone(mode)
            mode = "shared"
            calls.append("shared_enter")
            try:
                yield
            finally:
                calls.append(f"{mode}_exit")
                mode = None

        @contextlib.contextmanager
        def custom_context():
            nonlocal mode
            self.assertEqual(mode, "shared")
            mode = "exclusive"
            calls.append("exclusive_enter")
            try:
                yield
            finally:
                calls.append("exclusive_exit")
                mode = "shared"

        previous = _bench.set_gpu_benchmark_lock_context(custom_context)
        try:
            with patch.dict(os.environ, {"INDUCTOR_GPU_BENCH_LOCK": "1"}):
                with outer_context():
                    with _bench.maybe_gpu_benchmark_lock():
                        calls.append(mode)
                    calls.append(mode)
        finally:
            _bench.set_gpu_benchmark_lock_context(previous)

        self.assertEqual(
            calls,
            [
                "shared_enter",
                "exclusive_enter",
                "exclusive",
                "exclusive_exit",
                "shared",
                "shared_exit",
            ],
        )

    def test_do_bench_using_profiling_uses_gpu_benchmark_lock(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        from torch._inductor import utils as inductor_utils

        calls = []

        def fake_do_bench(fn, warmup, rep, is_vetted_benchmarking):
            calls.append((warmup, rep, is_vetted_benchmarking))
            fn()
            return 3.0

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "3",
            }
            with (
                patch.dict(os.environ, env),
                patch("torch.cuda.current_device", return_value=0),
                patch.object(
                    inductor_utils,
                    "_do_bench_using_profiling",
                    side_effect=fake_do_bench,
                ),
            ):
                result = inductor_utils.do_bench_using_profiling(
                    lambda: calls.append("fn"),
                    warmup=1,
                    rep=2,
                    is_vetted_benchmarking=True,
                )

            self.assertEqual(result, 3.0)
            self.assertEqual(calls, [(1, 2, True), "fn"])
            with open(os.path.join(lock_dir, "gpu_3.lock")) as f:
                metadata = f.read()
            self.assertIn("gpu=3\n", metadata)
            self.assertIn("mode=exclusive\n", metadata)

    def test_benchmark_uses_inferred_cuda_device_context(self):
        from torch._inductor.runtime import benchmarking as _bench

        benchmarker = TritonBenchmarker()
        orig = dict(_bench._BENCHMARK_DISPATCH)
        entered_devices = []

        @contextlib.contextmanager
        def fake_cuda_device(device):
            entered_devices.append(device)
            yield

        try:
            _bench.register_benchmarker(
                "cuda",
                lambda self, f, *, warmup, rep, **kw: 7.0,
                override=True,
            )
            with patch("torch.cuda.device", side_effect=fake_cuda_device):
                result = benchmarker.benchmark(lambda: None, device="cuda:1")
        finally:
            _bench._BENCHMARK_DISPATCH.clear()
            _bench._BENCHMARK_DISPATCH.update(orig)

        self.assertEqual(result, 7.0)
        self.assertEqual(entered_devices, [torch.device("cuda:1")])

    def test_benchmark_gpu_with_cuda_graph_uses_gpu_benchmark_lock(self):
        try:
            import fcntl  # noqa: F401
        except ImportError:
            self.skipTest("requires fcntl")

        class FakeCUDAGraph:
            def replay(self):
                pass

        benchmarker = Benchmarker()
        calls = []

        with tempfile.TemporaryDirectory() as lock_dir:
            env = {
                "INDUCTOR_GPU_BENCH_LOCK": "1",
                "INDUCTOR_GPU_BENCH_LOCK_DIR": lock_dir,
                "CUDA_VISIBLE_DEVICES": "5",
            }
            with (
                patch.dict(os.environ, env),
                patch("torch.cuda.current_device", return_value=0),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.CUDAGraph", FakeCUDAGraph),
                patch("torch.cuda.graph", return_value=contextlib.nullcontext()),
                patch.object(Benchmarker, "benchmark_gpu", return_value=9.0),
            ):
                result = benchmarker.benchmark_gpu_with_cuda_graph(
                    lambda: calls.append("call")
                )

            self.assertEqual(result, 9.0)
            self.assertEqual(calls, ["call", "call"])
            with open(os.path.join(lock_dir, "gpu_5.lock")) as f:
                metadata = f.read()
            self.assertIn("gpu=5\n", metadata)
            self.assertIn("mode=exclusive\n", metadata)

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
