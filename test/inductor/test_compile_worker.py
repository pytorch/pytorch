# Owner(s): ["module: inductor"]
import operator
import os
import sys
import tempfile
from threading import Event

import torch._inductor.config as config
from torch._inductor.compile_worker.subproc_pool import (
    CompactSubprocPickler,
    get_default_subproc_kind,
    raise_testexc,
    SubprocException,
    SubprocKind,
    SubprocPool,
    SubprocPickler,
)
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import HAS_CPU


class TestCompileWorker(TestCase):
    def make_pool(self, size):
        return SubprocPool(size)

    def test_get_default_subproc_kind_returns_fork_on_linux(self):
        kind = get_default_subproc_kind()
        if sys.platform == "linux":
            self.assertEqual(kind, SubprocKind.FORK)
        else:
            self.assertEqual(kind, SubprocKind.SPAWN)

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_subproc_pool_uses_platform_default_kind(self):
        pool = SubprocPool(1)
        try:
            self.assertEqual(pool.kind, get_default_subproc_kind())
        finally:
            pool.shutdown()

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


class TestCompactSubprocPickler(TestCase):
    def setUp(self):
        self.pickler = CompactSubprocPickler()
        self.standard_pickler = SubprocPickler()

    def _roundtrip(self, obj):
        data = self.pickler.dumps(obj)
        return self.pickler.loads(data)

    def test_none(self):
        self.assertIsNone(self._roundtrip(None))
        self.assertLess(
            len(self.pickler.dumps(None)), len(self.standard_pickler.dumps(None))
        )

    def test_bool(self):
        self.assertTrue(self._roundtrip(True))
        self.assertFalse(self._roundtrip(False))
        self.assertEqual(len(self.pickler.dumps(True)), 1)
        self.assertEqual(len(self.pickler.dumps(False)), 1)

    def test_small_int(self):
        for val in [-128, -1, 0, 1, 42, 127]:
            self.assertEqual(self._roundtrip(val), val)
            self.assertEqual(len(self.pickler.dumps(val)), 2)

    def test_large_int(self):
        for val in [-129, 128, 1000, -1000, 2**30, -(2**30)]:
            self.assertEqual(self._roundtrip(val), val)

    def test_float(self):
        for val in [0.0, 1.5, -3.14159, 1e100, float("inf")]:
            self.assertEqual(self._roundtrip(val), val)
        result = self._roundtrip(float("nan"))
        self.assertTrue(result != result)

    def test_small_bytes(self):
        for val in [b"", b"hello", b"\x00\x01\x02", b"x" * 255]:
            self.assertEqual(self._roundtrip(val), val)

    def test_large_bytes(self):
        self.assertEqual(self._roundtrip(b"x" * 1000), b"x" * 1000)

    def test_small_str(self):
        for val in ["", "hello", "unicode: \u00e9\u00e8", "x" * 255]:
            self.assertEqual(self._roundtrip(val), val)

    def test_large_str(self):
        self.assertEqual(self._roundtrip("x" * 1000), "x" * 1000)

    def test_empty_containers(self):
        self.assertEqual(self._roundtrip(()), ())
        self.assertEqual(self._roundtrip([]), [])
        self.assertEqual(self._roundtrip({}), {})
        self.assertEqual(len(self.pickler.dumps(())), 1)
        self.assertEqual(len(self.pickler.dumps([])), 1)
        self.assertEqual(len(self.pickler.dumps({})), 1)

    def test_complex_objects_fallback_to_pickle(self):
        obj = {"key": [1, 2, 3], "nested": {"a": "b"}}
        self.assertEqual(self._roundtrip(obj), obj)
        self.assertEqual(self._roundtrip([1, 2, 3]), [1, 2, 3])
        self.assertEqual(self._roundtrip((1, 2)), (1, 2))

    def test_compact_is_smaller_for_simple_types(self):
        test_values = [
            None,
            True,
            False,
            0,
            42,
            -1,
            "",
            "hi",
            b"",
            b"hi",
            (),
            [],
            {},
        ]
        for val in test_values:
            compact_size = len(self.pickler.dumps(val))
            pickle_size = len(self.standard_pickler.dumps(val))
            self.assertLessEqual(
                compact_size,
                pickle_size,
                f"Compact encoding for {val!r} should not be larger than pickle",
            )

    def test_empty_data_raises(self):
        with self.assertRaises(ValueError):
            self.pickler.loads(b"")

    @skipIfWindows(msg="pass_fds not supported on Windows.")
    def test_integration_with_subproc_pool(self):
        pool = SubprocPool(1, pickler=CompactSubprocPickler())
        try:
            result = pool.submit(operator.add, 10, 20)
            self.assertEqual(result.result(), 30)
        finally:
            pool.shutdown()


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU:
        run_tests()
