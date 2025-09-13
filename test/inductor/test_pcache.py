# Owner(s): ["module: inductor"]
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from os import environ
from random import randint
from typing_extensions import Self

from torch._inductor import pcache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


key_gen: Generator[pcache.Key, None, None] = (
    f"dummy_key_{randint(0, 100000)}" for _ in iter(int, 1)
)
value_gen: Generator[pcache.Value, None, None] = (
    f"dummy_value_{randint(0, 100000)}".encode() for _ in iter(int, 1)
)


Caches: list[type[pcache.Cache]] = [pcache.InMemoryCache]
AsyncCaches: list[type[pcache.AsyncCache]] = [pcache.InductorOnDiskCache]


@instantiate_parametrized_tests
class CacheTest(TestCase):
    @parametrize("Cache", Caches)
    def test_get_insert_get(self: Self, Cache: type[pcache.Cache]) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        cache: pcache.Cache = Cache()

        # make sure our key is fresh
        while cache.get(key) is not None:
            key = next(key_gen)

        # first get should return None, no hit
        self.assertIsNone(cache.get(key))
        # insert should return True, having set key -> value
        self.assertTrue(cache.insert(key, value))
        # second get should return value, hit
        self.assertEqual(cache.get(key), value)

    @parametrize("Cache", Caches)
    def test_insert_insert(self: Self, Cache: type[pcache.Cache]) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        cache: pcache.Cache = Cache()

        if cache.get(key) is None:
            # if key isn't already cached, cache it
            self.assertTrue(cache.insert(key, value))

        # second insert should not update the value
        self.assertFalse(cache.insert(key, value))

    def test_in_memory_cache_from_env_var(
        self: Self, Cache: type[pcache.InMemoryCache] = pcache.InMemoryCache
    ) -> None:
        key_1: pcache.Key = next(key_gen)
        value_1: pcache.Value = next(value_gen)

        key_2: pcache.Key = next(key_gen)
        while key_2 == key_1:
            key_2 = next(key_gen)
        value_2: pcache.Value = next(value_gen)

        key_3: pcache.Key = next(key_gen)
        while key_3 in (key_1, key_2):
            key_3 = next(key_gen)

        env_var = "INMEMORYCACHE_TEST"
        env_val = f"{key_1},{value_1!r};{key_2},{value_2!r}"
        environ[env_var] = env_val

        cache = Cache.from_env_var(env_var)

        # key_1 -> value_1 is in env_val, so we should hit
        self.assertEqual(cache.get(key_1), value_1)
        # key_2 -> value_2 is in env_val, so we should hit
        self.assertEqual(cache.get(key_2), value_2)
        # key_3 -> value_3 is not in env_val, so we should miss
        self.assertIsNone(cache.get(key_3))

    def test_in_memory_cache_from_env_var_bad_kv_pair(
        self: Self, Cache: type[pcache.InMemoryCache] = pcache.InMemoryCache
    ) -> None:
        key_1: pcache.Key = next(key_gen)
        value_1: pcache.Value = next(value_gen)

        env_var = "INMEMORYCACHE_TEST"
        # missing "," delimiter
        env_val = f"{key_1}{value_1!r};"
        kv_pair = env_val[:-1]
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError, f"Malformed kv_pair {kv_pair} in env_var {env_var}!"
        ):
            _ = Cache.from_env_var(env_var)

    def test_in_memory_cache_from_env_var_bad_value(
        self: Self, Cache: type[pcache.InMemoryCache] = pcache.InMemoryCache
    ) -> None:
        key_1: pcache.Key = next(key_gen)
        # exclude b' prefix and ' suffix
        value_1: str = "bad_value"

        env_var = "INMEMORYCACHE_TEST"
        env_val = f"{key_1},{value_1};"
        kv_pair = env_val[:-1]
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError, f"Malformed value {value_1} in kv_pair {kv_pair}!"
        ):
            _ = Cache.from_env_var(env_var)

    def test_in_memory_cache_from_env_var_one_key_many_values(
        self: Self, Cache: type[pcache.InMemoryCache] = pcache.InMemoryCache
    ) -> None:
        key_1: pcache.Key = next(key_gen)
        value_1: pcache.Value = next(value_gen)
        value_2: pcache.Value = next(value_gen)

        env_var = "INMEMORYCACHE_TEST"
        env_val = f"{key_1},{value_1!r};{key_1},{value_2!r}"
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError,
            f"Duplicated values for key {key_1}, got {value_1!r} and {value_2!r}!",
        ):
            _ = Cache.from_env_var(env_var)


@instantiate_parametrized_tests
class AsyncCacheTest(TestCase):
    @parametrize("AsyncCache", AsyncCaches)
    @parametrize("Executor", [ThreadPoolExecutor, None])
    def test_get_insert_get(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] | None = None,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor() if Executor is not None else None

        if executor is None:
            # make sure our key is fresh
            while async_cache.get(key) is not None:
                key = next(key_gen)

            # first get should miss
            self.assertIsNone(async_cache.get(key))
            # insert should set key -> value mapping
            self.assertTrue(async_cache.insert(key, value))
            # second get should hit
            self.assertEqual(async_cache.get(key), value)
        else:
            # make sure our key is fresh
            while async_cache.get_async(key, executor).result() is not None:
                key = next(key_gen)

            # first get should miss
            self.assertIsNone(async_cache.get_async(key, executor).result())
            # insert should set key -> value mapping
            self.assertTrue(async_cache.insert_async(key, value, executor).result())
            # second get should hit
            self.assertEqual(async_cache.get_async(key, executor).result(), value)
            executor.shutdown()

    @parametrize("AsyncCache", AsyncCaches)
    @parametrize("Executor", [ThreadPoolExecutor, None])
    def test_insert_insert(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] | None = None,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor() if Executor is not None else None

        if executor is None:
            if async_cache.get(key) is None:
                # set key -> value mapping if unset
                self.assertTrue(async_cache.insert(key, value))
            # second insert should not override the prior insert
            self.assertFalse(async_cache.insert(key, value))
        else:
            if async_cache.get_async(key, executor).result() is None:
                # set key -> value mapping if unset
                self.assertTrue(async_cache.insert_async(key, value, executor).result())
            # second insert should not override the prior insert
            self.assertFalse(async_cache.insert_async(key, value, executor).result())
            executor.shutdown()

    @parametrize("AsyncCache", AsyncCaches)
    def test_concurrent_insert_insert(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] = ThreadPoolExecutor,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor()

        # make sure our key is fresh
        while async_cache.get_async(key, executor).result() is not None:
            key = next(key_gen)

        insert_1: Future[bool] = async_cache.insert_async(key, value, executor)
        insert_2: Future[bool] = async_cache.insert_async(key, value, executor)

        # only one insert should succeed
        self.assertTrue(insert_1.result() ^ insert_2.result())
        executor.shutdown()

    @parametrize("AsyncCache", AsyncCaches)
    def test_concurrent_get_insert(
        self: Self,
        AsyncCache: type[pcache.AsyncCache],
        Executor: type[ThreadPoolExecutor] = ThreadPoolExecutor,
    ) -> None:
        key: pcache.Key = next(key_gen)
        value: pcache.Value = next(value_gen)

        async_cache: pcache.AsyncCache = AsyncCache()
        executor: ThreadPoolExecutor = Executor()

        # make sure our key is fresh
        while async_cache.get_async(key, executor).result() is not None:
            key = next(key_gen)

        # try get first
        get_1: Future[pcache.Value | None] = async_cache.get_async(key, executor)
        insert_1: Future[bool] = async_cache.insert_async(key, value, executor)

        if get_1.result() is not None:
            # if the get succeeded it should return the value stored by the insert
            self.assertEqual(get_1.result(), value)

        # either way the insert should succeed as the key is fresh
        self.assertTrue(insert_1.result())

        # make sure our key is fresh
        while async_cache.get_async(key, executor).result() is not None:
            key = next(key_gen)

        # try insert first
        insert_2: Future[bool] = async_cache.insert_async(key, value, executor)
        get_2: Future[pcache.Value | None] = async_cache.get_async(key, executor)

        if get_2.result() is not None:
            # if the get succeeded it should return the value stored by the insert
            self.assertEqual(get_2.result(), value)

        # either way the insert should succeed as the key is fresh
        self.assertTrue(insert_2.result())

        executor.shutdown()


if __name__ == "__main__":
    run_tests()
