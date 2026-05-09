# Owner(s): ["module: inductor"]
import operator
import os
import tempfile
from threading import Event
from unittest.mock import mock_open, patch

import torch._inductor.config as config
from torch._inductor.async_compile import (
    _get_available_memory_fraction,
    _get_compile_threads_for_memory,
    AsyncCompile,
)
from torch._inductor.compile_worker.subproc_pool import (
    raise_testexc,
    SubprocException,
    SubprocPool,
)
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import HAS_CPU


class TestCompileWorker(TestCase):
    def make_pool(self, size):
        return SubprocPool(size)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_basic_jobs(self):
        pool = self.make_pool(2)
        try:
            a = pool.submit(operator.add, 100, 1)
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_exception(self):
        pool = self.make_pool(2)
        try:
            a = pool.submit(raise_testexc)
            with self.assertRaisesRegex(
                SubprocException,
                "torch._inductor.compile_worker.subproc_pool.TestException",
            ):
                a.result()
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_crash(self):
        pool = self.make_pool(2)
        try:
            with self.assertRaises(Exception):
                a = pool.submit(os._exit, 1)
                a.result()

            # Pool should still be usable after a crash
            b = pool.submit(operator.add, 100, 1)
            c = pool.submit(operator.sub, 100, 1)
            self.assertEqual(b.result(), 101)
            self.assertEqual(c.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_quiesce(self):
        pool = self.make_pool(2)
        try:
            a = pool.submit(operator.add, 100, 1)
            pool.quiesce()
            pool.wakeup()
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_quiesce_repeatedly(self):
        pool = SubprocPool(2)
        try:
            a = pool.submit(operator.add, 100, 1)
            pool.quiesce()
            pool.wakeup()
            b = pool.submit(operator.sub, 100, 1)
            pool.quiesce()
            pool.quiesce()
            pool.wakeup()
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(a.result(), 101)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_logging(self):
        os.environ["MAST_HPC_JOB_NAME"] = "test_job"
        os.environ["ROLE_RANK"] = "0"
        with tempfile.NamedTemporaryFile(delete=True) as temp_log:
            os.environ["TORCHINDUCTOR_WORKER_LOGPATH"] = temp_log.name
            pool = self.make_pool(2)
            try:
                pool.submit(operator.add, 100, 1)
                self.assertEqual(os.path.exists(temp_log.name), True)
            finally:
                pool.shutdown()


@config.patch("quiesce_async_compile_time", 0.1)
class TestCompileWorkerWithTimer(TestCompileWorker):
    def make_pool(self, size):
        return SubprocPool(size, quiesce=True)


class TestTimer(TestCase):
    def test_basics(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(0.1, doit)
        t.sleep_time = 0.1
        t.record_call()
        self.assertTrue(done.wait(4))
        t.quit()

    def test_repeated_calls(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(0.1, doit)
        t.sleep_time = 0.1
        for _ in range(10):
            t.record_call()
            self.assertTrue(done.wait(4))
            done.clear()
        t.quit()

    def test_never_fires(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(999, doit)
        t.sleep_time = 0.1
        t.record_call()
        self.assertFalse(done.wait(4))
        t.quit()

    def test_spammy_calls(self):
        done = Event()

        def doit():
            done.set()

        t = Timer(1, doit)
        t.sleep_time = 0.1
        for _ in range(400):
            t.record_call()
        self.assertTrue(done.wait(4))
        t.quit()


class TestDynamicWakeup(TestCase):
    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_wakeup_with_fewer_workers(self):
        pool = SubprocPool(4)
        try:
            a = pool.submit(operator.add, 100, 1)
            self.assertEqual(a.result(), 101)
            pool.quiesce()
            pool.wakeup(nprocs=2)
            b = pool.submit(operator.sub, 100, 1)
            self.assertEqual(b.result(), 99)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_wakeup_scale_up(self):
        pool = SubprocPool(4)
        try:
            pool.wakeup(nprocs=2)
            a = pool.submit(operator.add, 10, 5)
            self.assertEqual(a.result(), 15)
            pool.wakeup(nprocs=4)
            b = pool.submit(operator.mul, 10, 5)
            self.assertEqual(b.result(), 50)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_eager_quiesce_cycle(self):
        pool = SubprocPool(4)
        try:
            pool.wakeup(nprocs=4)
            a = pool.submit(operator.add, 1, 2)
            self.assertEqual(a.result(), 3)

            pool.quiesce()
            pool.wakeup(nprocs=2)

            b = pool.submit(operator.add, 3, 4)
            self.assertEqual(b.result(), 7)

            pool.quiesce()
            pool.wakeup(nprocs=4)

            c = pool.submit(operator.add, 5, 6)
            self.assertEqual(c.result(), 11)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_submit_after_reduced_wakeup(self):
        # After quiesce + wakeup(small), submitting jobs should not scale the
        # sidecar pool back up to the original nprocs. Before the fix in
        # _start_pool, _submit_inner used a stale self.nprocs default which
        # defeated memory-aware scaling.
        pool = SubprocPool(8)
        try:
            pool.wakeup(nprocs=8)
            a = pool.submit(operator.add, 1, 2)
            self.assertEqual(a.result(), 3)

            pool.quiesce()
            pool.wakeup(nprocs=2)

            for i in range(5):
                f = pool.submit(operator.add, i, 10)
                self.assertEqual(f.result(), i + 10)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_multi_cycle_quiesce_wakeup(self):
        # Simulates multiple compilation cycles: full wakeup -> work ->
        # quiesce -> reduced wakeup -> full wakeup -> work -> quiesce.
        pool = SubprocPool(4)
        try:
            pool.wakeup(nprocs=4)
            self.assertEqual(pool.submit(operator.add, 1, 1).result(), 2)

            pool.quiesce()
            pool.wakeup(nprocs=1)

            pool.quiesce()
            pool.wakeup(nprocs=4)
            self.assertEqual(pool.submit(operator.mul, 3, 7).result(), 21)

            pool.quiesce()
            pool.wakeup(nprocs=2)
            self.assertEqual(pool.submit(operator.sub, 10, 4).result(), 6)
        finally:
            pool.shutdown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_crash_recovery_uses_reduced_pool(self):
        # After wakeup with a reduced nprocs, a BrokenProcessPool crash
        # recovery should recreate the pool at the reduced size, not the
        # original constructor size.
        pool = SubprocPool(8)
        try:
            pool.wakeup(nprocs=2)
            a = pool.submit(operator.add, 1, 2)
            self.assertEqual(a.result(), 3)

            with self.assertRaises(Exception):
                pool.submit(os._exit, 1).result()

            b = pool.submit(operator.add, 10, 20)
            self.assertEqual(b.result(), 30)
        finally:
            pool.shutdown()


class TestMemoryAwareThreads(TestCase):
    MEMINFO_TEMPLATE = (
        "MemTotal:       {total} kB\n"
        "MemFree:        {free} kB\n"
        "MemAvailable:   {available} kB\n"
    )

    def _mock_meminfo(self, total_kb, available_kb):
        content = self.MEMINFO_TEMPLATE.format(
            total=total_kb, available=available_kb, free=available_kb
        )
        return patch(
            "torch._inductor.async_compile.open",
            mock_open(read_data=content),
        )

    def test_available_memory_fraction(self):
        with self._mock_meminfo(100000, 50000):
            frac = _get_available_memory_fraction()
            self.assertAlmostEqual(frac, 0.5, places=2)

    def test_available_memory_fraction_unavailable(self):
        with patch(
            "torch._inductor.async_compile.open",
            side_effect=OSError("no /proc/meminfo"),
        ):
            self.assertIsNone(_get_available_memory_fraction())

    @config.patch("compile_worker_memory_threshold", 0.8)
    @config.patch("compile_threads_min", 2)
    def test_threads_no_scaling_when_memory_ok(self):
        with self._mock_meminfo(100000, 90000):
            self.assertEqual(_get_compile_threads_for_memory(32), 32)

    @config.patch("compile_worker_memory_threshold", 0.8)
    @config.patch("compile_threads_min", 2)
    def test_threads_scaled_down_when_memory_low(self):
        # available=50% => scale = (0.5 - 0.4) / (0.8 - 0.4) = 0.25
        # result = int(32 * 0.25) = 8
        with self._mock_meminfo(100000, 50000):
            self.assertEqual(_get_compile_threads_for_memory(32), 8)

    @config.patch("compile_worker_memory_threshold", 0.8)
    @config.patch("compile_threads_min", 2)
    def test_threads_min_when_memory_very_low(self):
        with self._mock_meminfo(100000, 10000):
            self.assertEqual(_get_compile_threads_for_memory(32), 2)

    @config.patch("compile_worker_memory_threshold", 0.0)
    def test_threshold_zero_disables_scaling(self):
        with self._mock_meminfo(100000, 1000):
            self.assertEqual(_get_compile_threads_for_memory(32), 32)

    @config.patch("compile_worker_memory_threshold", 0.8)
    @config.patch("compile_threads_min", 0)
    def test_threads_min_zero_clamps_to_one(self):
        with self._mock_meminfo(100000, 10000):
            self.assertEqual(_get_compile_threads_for_memory(32), 1)


class TestEagerQuiesce(TestCase):
    def setUp(self):
        super().setUp()
        from torch._inductor.async_compile import shutdown_compile_workers

        shutdown_compile_workers()

    def tearDown(self):
        from torch._inductor.async_compile import shutdown_compile_workers

        shutdown_compile_workers()
        super().tearDown()

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    @config.patch("quiesce_async_compile_eager", True)
    @config.patch("compile_threads_min", 0)
    @config.patch("compile_threads", 4)
    @config.patch("worker_start_method", "subprocess")
    def test_eager_quiesce_full_shutdown(self):
        pool = AsyncCompile.process_pool()
        self.assertIsInstance(pool, SubprocPool)
        pool.wakeup(nprocs=4)
        a = pool.submit(operator.add, 1, 2)
        self.assertEqual(a.result(), 3)

        AsyncCompile._eager_quiesce()
        self.assertTrue(AsyncCompile._pool_needs_wakeup)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    @config.patch("quiesce_async_compile_eager", True)
    @config.patch("compile_threads_min", 2)
    @config.patch("compile_threads", 4)
    @config.patch("worker_start_method", "subprocess")
    def test_eager_quiesce_warm_start(self):
        pool = AsyncCompile.process_pool()
        self.assertIsInstance(pool, SubprocPool)
        pool.wakeup(nprocs=4)
        a = pool.submit(operator.add, 1, 2)
        self.assertEqual(a.result(), 3)

        AsyncCompile._eager_quiesce()
        self.assertTrue(AsyncCompile._pool_needs_wakeup)

        b = pool.submit(operator.add, 3, 4)
        self.assertEqual(b.result(), 7)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    @config.patch("quiesce_async_compile_eager", True)
    @config.patch("compile_threads_min", 1)
    @config.patch("compile_threads", 4)
    @config.patch("worker_start_method", "subprocess")
    def test_pool_needs_wakeup_triggers_wakeup(self):
        pool = AsyncCompile.process_pool()
        self.assertIsInstance(pool, SubprocPool)
        AsyncCompile._ready_future = pool.submit(AsyncCompile._get_ready)
        AsyncCompile._ready_future.result()

        AsyncCompile._eager_quiesce()
        self.assertTrue(AsyncCompile._pool_needs_wakeup)

        result = AsyncCompile.use_process_pool()
        self.assertTrue(result)
        self.assertFalse(AsyncCompile._pool_needs_wakeup)

    @config.patch("quiesce_async_compile_eager", True)
    @config.patch("compile_threads", 4)
    def test_eager_quiesce_skips_uncreated_pool(self):
        AsyncCompile.process_pool.cache_clear()
        self.assertEqual(AsyncCompile.process_pool.cache_info().currsize, 0)
        AsyncCompile._eager_quiesce()
        self.assertEqual(AsyncCompile.process_pool.cache_info().currsize, 0)


class TestSetTritonLibdevicePath(TestCase):
    @config.patch({"compile_threads": 1, "eager_numerics.use_pytorch_libdevice": True})
    def test_libdevice_path_no_subprocess(self):
        """Test libdevice path is set with compile_threads=1 (no subprocess)."""
        self._test_libdevice_path_with_compilation()

    @config.patch("eager_numerics.use_pytorch_libdevice", True)
    def test_libdevice_path_default_threads(self):
        """Test libdevice path is set with default compile_threads (subprocess)."""
        self._test_libdevice_path_with_compilation()

    @config.patch(
        {
            "eager_numerics.use_pytorch_libdevice": True,
            "eager_numerics.division_rounding": True,
            "emulate_precision_casts": True,
            "compile_threads": 1,
        }
    )
    def test_pow_bitwise_precision(self):
        """Test that compiled pow matches eager bitwise with system libdevice."""
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if CUDA_HOME is None:
            self.skipTest("CUDA_HOME not set")
        expected = os.path.join(CUDA_HOME, "nvvm", "libdevice", "libdevice.10.bc")
        if not os.path.isfile(expected):
            self.skipTest(f"libdevice not found at {expected}")

        torch._dynamo.reset()
        torch.manual_seed(42)
        base = torch.randn(1000, device="cuda", dtype=torch.float32).abs() + 1e-6
        exp = torch.randn(1000, device="cuda", dtype=torch.float32)

        eager_result = torch.pow(base, exp)
        compiled_result = torch.compile(torch.pow)(base, exp)
        self.assertEqual(eager_result, compiled_result, atol=0, rtol=0)

    def _test_libdevice_path_with_compilation(self):
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        if CUDA_HOME is None:
            self.skipTest("CUDA_HOME not set")

        expected = os.path.join(CUDA_HOME, "nvvm", "libdevice", "libdevice.10.bc")
        if not os.path.isfile(expected):
            self.skipTest(f"libdevice not found at {expected}")

        # Compile a simple function that uses pow (which uses libdevice)
        @torch.compile
        def fn(x):
            return torch.pow(x, 2.0)

        x = torch.randn(10, device="cuda", dtype=torch.float32)
        fn(x)

        # Verify libdevice path was set
        from triton import knobs

        self.assertEqual(knobs.nvidia.libdevice_path, expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU:
        run_tests()
