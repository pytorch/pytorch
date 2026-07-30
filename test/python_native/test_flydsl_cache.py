# Owner(s): ["module: dsl-native-ops"]

from torch._native.flydsl.cache import flydsl_jit_cache, jit_cache
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlyDSLCache(TestCase):
    def test_jit_cache_alias(self):
        self.assertIs(jit_cache, flydsl_jit_cache)

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


if __name__ == "__main__":
    run_tests()
