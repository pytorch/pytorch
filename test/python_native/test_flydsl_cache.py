# Owner(s): ["module: dsl-native-ops"]
import threading

from torch._native.flydsl.cache import flydsl_jit_cache
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlyDSLCache(TestCase):
    def test_compile_args_are_excluded_from_cache_key(self):
        calls = []

        @flydsl_jit_cache
        def compile_fn(*args, compile_args, **kwargs):
            calls.append((args, compile_args, kwargs))
            return object()

        first = compile_fn("key", beta=2, alpha=1, compile_args="first")
        second = compile_fn("key", alpha=1, beta=2, compile_args="second")

        self.assertIs(first, second)
        self.assertEqual(calls, [(("key",), "first", {"alpha": 1, "beta": 2})])
        self.assertEqual(compile_fn.cache_info().hits, 1)
        self.assertEqual(compile_fn.cache_info().misses, 1)
        self.assertEqual(compile_fn.cache_info().currsize, 1)

    def test_positional_and_keyword_arguments_do_not_collide(self):
        calls = []

        @flydsl_jit_cache
        def compile_fn(value):
            calls.append(value)
            return object()

        positional = compile_fn(("value", 1))
        keyword = compile_fn(value=1)

        self.assertIsNot(positional, keyword)
        self.assertEqual(calls, [("value", 1), 1])
        self.assertEqual(compile_fn.cache_info().misses, 2)

    def test_none_result_and_compile_args_none_are_cached(self):
        calls = 0

        @flydsl_jit_cache
        def compile_fn(key, *, compile_args):
            nonlocal calls
            calls += 1
            self.assertIsNone(compile_args)
            return None

        self.assertIsNone(compile_fn("key", compile_args=None))
        self.assertIsNone(compile_fn("key", compile_args=None))
        self.assertEqual(calls, 1)
        self.assertEqual(compile_fn.cache_info().hits, 1)
        self.assertEqual(compile_fn.cache_info().misses, 1)

    def test_cache_clear_resets_entries_and_counters(self):
        calls = 0

        @flydsl_jit_cache
        def compile_fn(key):
            nonlocal calls
            calls += 1
            return key

        self.assertEqual(compile_fn("key"), "key")
        compile_fn.cache_clear()
        self.assertEqual(compile_fn.cache_info().hits, 0)
        self.assertEqual(compile_fn.cache_info().misses, 0)
        self.assertEqual(compile_fn.cache_info().currsize, 0)
        self.assertEqual(compile_fn("key"), "key")
        self.assertEqual(calls, 2)

    def test_same_specialization_compiles_once(self):
        calls = 0
        started = threading.Event()
        release = threading.Event()
        results = []

        @flydsl_jit_cache
        def compile_fn(key):
            nonlocal calls
            calls += 1
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return object()

        first = threading.Thread(target=lambda: results.append(compile_fn("key")))
        second = threading.Thread(target=lambda: results.append(compile_fn("key")))
        first.start()
        self.assertTrue(started.wait(timeout=5))
        second.start()
        try:
            # `calls == 1` alone would also hold if the second thread simply had
            # not been scheduled yet, so require it to still be blocked -- that
            # is the property the per-key lock exists for.
            second.join(timeout=0.5)
            self.assertTrue(second.is_alive())
            self.assertEqual(calls, 1)
        finally:
            release.set()
            first.join()
            second.join()

        self.assertEqual(calls, 1)
        self.assertIs(results[0], results[1])

    def test_different_specializations_compile_concurrently(self):
        started = {key: threading.Event() for key in ("first", "second")}
        release = threading.Event()
        results = {}

        @flydsl_jit_cache
        def compile_fn(key):
            started[key].set()
            self.assertTrue(release.wait(timeout=5))
            return key

        first = threading.Thread(
            target=lambda: results.setdefault("first", compile_fn("first"))
        )
        second = threading.Thread(
            target=lambda: results.setdefault("second", compile_fn("second"))
        )
        first.start()
        self.assertTrue(started["first"].wait(timeout=5))
        second.start()
        try:
            self.assertTrue(started["second"].wait(timeout=5))
        finally:
            release.set()
            first.join()
            second.join()

        self.assertEqual(results, {"first": "first", "second": "second"})

    def test_cache_clear_during_compile_resets_counters(self):
        started = threading.Event()
        release = threading.Event()
        result = []

        @flydsl_jit_cache
        def compile_fn(key):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return "compiled"

        worker = threading.Thread(target=lambda: result.append(compile_fn("key")))
        worker.start()
        self.assertTrue(started.wait(timeout=5))
        compile_fn.cache_clear()
        release.set()
        worker.join()

        # The in-flight caller gets its result and stores it, so the clear does
        # not guarantee an empty cache -- only that the counters restart and
        # that nothing is left waiting behind a stale key lock.
        self.assertEqual(result, ["compiled"])
        info = compile_fn.cache_info()
        self.assertEqual((info.hits, info.misses), (0, 0))
        self.assertEqual(info.currsize, 1)

    def test_cache_clear_drops_key_locks(self):
        @flydsl_jit_cache
        def compile_fn(key):
            return key

        compile_fn("key")
        self.assertEqual(len(compile_fn._key_locks), 1)
        compile_fn.cache_clear()
        self.assertEqual(len(compile_fn._key_locks), 0)


if __name__ == "__main__":
    run_tests()
