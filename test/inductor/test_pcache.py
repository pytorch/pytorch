# Owner(s): ["module: inductor"]
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from inspect import isclass
from os import environ
from random import randint
from typing_extensions import Self

from torch._inductor import pcache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


# abstract cache classes don't go through the testing
# process, as they have unimplemented components and are
# not meant to be utilized directly by the end user
ABSTRACT_CACHES: list[type[pcache.Cache]] = [
    pcache.Cache,
    pcache.AsyncCache,
]

STR_BYTES_CACHES: list[type[pcache.Cache]] = []
STR_BYTES_ASYNC_CACHES: list[type[pcache.AsyncCache]] = []

UNSUPPORTED_CACHES: list[type[pcache.Cache]] = []


for obj_name in dir(pcache):
    obj = getattr(pcache, obj_name)
    if not isclass(obj) or not issubclass(obj, pcache.Cache) or obj in ABSTRACT_CACHES:
        continue
    # we only have Key=str, Value=bytes tests setup
    for _orig_base in obj.__orig_bases__:
        if issubclass(_orig_base.__origin__, pcache.Cache):
            key_type, value_type = _orig_base.__args__
            if (key_type != str) or (value_type != bytes):
                UNSUPPORTED_CACHES.append(obj)
                continue
    # check association from strongest to weakest
    if issubclass(obj, pcache.AsyncCache):
        STR_BYTES_ASYNC_CACHES.append(obj)
    if issubclass(obj, pcache.Cache):
        STR_BYTES_CACHES.append(obj)


class TestMixin:
    def str_key(self: Self) -> str:
        return f"key-{randint(0, 2**32)}"

    def str_key_not_in(self: Self, cache: pcache.Cache[str, pcache.Value]) -> str:
        while cache.get(str_key := self.str_key()) is not None:
            continue

        return str_key

    def str_keys_not_in(
        self: Self, cache: pcache.Cache[str, pcache.Value], num: int
    ) -> list[str]:
        str_keys: list[str] = []

        while len(str_keys) < num:
            str_key = self.str_key_not_in(cache)
            if str_key not in str_keys:
                str_keys.append(str_key)

        return str_keys

    def bytes_value(self: Self) -> bytes:
        return f"value-{randint(0, 2**32)}".encode()


@instantiate_parametrized_tests
class CacheTest(TestMixin, TestCase):
    @parametrize("Cache", STR_BYTES_CACHES)
    def test_str_bytes_get_hit(self: Self, Cache: type[pcache.Cache]) -> None:
        cache: pcache.Cache = Cache()

        key = self.str_key_not_in(cache)
        value = self.bytes_value()

        self.assertIsNone(cache.get(key))
        self.assertTrue(cache.insert(key, value))
        self.assertEqual(cache.get(key), value)

    @parametrize("Cache", STR_BYTES_CACHES)
    def test_str_bytes_get_miss(self: Self, Cache: type[pcache.Cache]) -> None:
        cache: pcache.Cache = Cache()

        key = self.str_key_not_in(cache)

        self.assertIsNone(cache.get(key))
        self.assertIsNone(cache.get(key))

    @parametrize("Cache", STR_BYTES_CACHES)
    def test_str_bytes_insert_no_overwrite(
        self: Self, Cache: type[pcache.Cache]
    ) -> None:
        cache: pcache.Cache = Cache()

        key = self.str_key_not_in(cache)
        value_1, value_2 = self.bytes_value(), self.bytes_value()

        self.assertIsNone(cache.get(key))
        self.assertTrue(cache.insert(key, value_1))
        self.assertFalse(cache.insert(key, value_2))
        self.assertEqual(cache.get(key), value_1)

    @parametrize("Cache", STR_BYTES_CACHES)
    def test_str_bytes_get_insert_thread_safe(
        self: Self, Cache: type[pcache.Cache]
    ) -> None:
        cache: pcache.Cache = Cache()
        executor: ThreadPoolExecutor = ThreadPoolExecutor()

        num_iters = 1000

        keys = self.str_keys_not_in(cache, num_iters)
        values = [self.bytes_value() for _ in range(num_iters)]

        get_futures = executor.map(cache.get, keys)
        insert_futures = executor.map(cache.insert, keys, values)

        for value, get_result, insert_result in zip(
            values, get_futures, insert_futures
        ):
            if get_result is not None:
                self.assertIsEqual(get_result, value)
                self.assertTrue(insert_result)

        executor.shutdown()

    @parametrize("Cache", STR_BYTES_CACHES)
    def test_str_bytes_insert_no_overwrite_thread_safe(
        self: Self, Cache: type[pcache.Cache]
    ) -> None:
        cache: pcache.Cache = Cache()
        executor: ThreadPoolExecutor = ThreadPoolExecutor()

        num_iters = 1000

        key = self.str_key_not_in(cache)
        keys = [key for _ in range(num_iters)]
        values = [self.bytes_value() for _ in range(num_iters)]

        insert_futures = executor.map(cache.insert, keys, values)

        hit_count = 0
        for value, insert_result in zip(values, insert_futures):
            self.assertLessEqual(hit_count, 1)
            if insert_result:
                hit_count += 1
                self.assertEqual(cache.get(key), value)

        executor.shutdown()


@instantiate_parametrized_tests
class AsyncCacheTest(TestMixin, TestCase):
    @parametrize("Cache", STR_BYTES_ASYNC_CACHES)
    def test_str_bytes_get_hit_async(
        self: Self, Cache: type[pcache.AsyncCache]
    ) -> None:
        cache: pcache.AsyncCache = Cache()
        executor: ThreadPoolExecutor = ThreadPoolExecutor()

        key = self.str_key_not_in(cache)
        value = self.bytes_value()

        self.assertIsNone(cache.get_async(key, executor).result())
        self.assertTrue(cache.insert_async(key, value, executor).result())
        self.assertEqual(cache.get_async(key, executor).result(), value)

        executor.shutdown()

    @parametrize("Cache", STR_BYTES_ASYNC_CACHES)
    def test_str_bytes_get_miss_async(
        self: Self, Cache: type[pcache.AsyncCache]
    ) -> None:
        cache: pcache.AsyncCache = Cache()
        executor: ThreadPoolExecutor = ThreadPoolExecutor()

        key = self.str_key_not_in(cache)

        self.assertIsNone(cache.get_async(key, executor).result())
        self.assertIsNone(cache.get_async(key, executor).result())

        executor.shutdown()

    @parametrize("Cache", STR_BYTES_ASYNC_CACHES)
    def test_str_bytes_insert_async_no_overwrite(
        self: Self, Cache: type[pcache.AsyncCache]
    ) -> None:
        cache: pcache.AsyncCache = Cache()
        executor: ThreadPoolExecutor = ThreadPoolExecutor()

        num_iters = 10

        key = self.str_key_not_in(cache)
        values = [self.bytes_value() for _ in range(num_iters)]

        self.assertIsNone(cache.get_async(key, executor).result())

        futures = []
        for value in values:
            futures.append(cache.insert_async(key, value, executor))

        for future, value in zip(futures, values):
            if future.result():
                self.assertTrue(cache.get(key), value)

        executor.shutdown()


@instantiate_parametrized_tests
class OtherTest(TestMixin, TestCase):
    def test_str_bytes_in_memory_cache_from_env_var(self: Self) -> None:
        num_iters = 100

        keys = [self.str_key() for _ in range(num_iters)]
        values = [self.bytes_value() for _ in range(num_iters)]

        env_var = "INMEMORYCACHE_TEST"
        env_val = ";".join([f"{key},{value!r}" for key, value in zip(keys, values)])
        environ[env_var] = env_val

        cache = pcache.InMemoryCache.from_env_var(env_var)

        for key, value in zip(keys, values):
            self.assertEqual(cache.get(key), value)

        for key in keys[num_iters:]:
            self.assertIsNone(cache.get(key))

    def test_str_bytes_in_memory_cache_from_env_var_bad_kv_pair(self: Self) -> None:
        key = self.str_key()
        value = self.bytes_value()

        env_var = "INMEMORYCACHE_TEST"
        # no comma separator
        env_val = f"{key}{value!r};"
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError,
            f"Malformed kv_pair {env_val[:-1]!r} in env_var {env_var!r}, missing comma separator!",
        ):
            _ = pcache.InMemoryCache.from_env_var(env_var)

    def test_str_bytes_in_memory_cache_from_env_var_bad_value_not_bytes(
        self: Self,
    ) -> None:
        key = self.str_key()
        # value is str, not bytes
        value = self.str_key()

        env_var = "INMEMORYCACHE_TEST"
        env_val = f"{key},{value};"
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError,
            f"Malformed value {value!r} in kv_pair {env_val[:-1]!r}, expected b'...' format!",
        ):
            _ = pcache.InMemoryCache.from_env_var(env_var)

    def test_str_bytes_in_memory_cache_from_env_var_bad_value_not_encoded(
        self: Self,
    ) -> None:
        key = self.str_key()
        # value is not encoded properly
        value = f"b'{chr(256)}'"

        env_var = "INMEMORYCACHE_TEST"
        env_val = f"{key},{value};"
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError, f"Malformed value {value!r} in kv_pair {env_val[:-1]!r}!"
        ):
            _ = pcache.InMemoryCache.from_env_var(env_var)

    def test_str_bytes_in_memory_cache_from_env_var_one_key_many_values(
        self: Self,
    ) -> None:
        num_iters = 2

        key = self.str_key()
        keys = [key for _ in range(num_iters)]
        values = [self.bytes_value() for _ in range(num_iters)]

        env_var = "INMEMORYCACHE_TEST"
        env_val = ";".join([f"{key},{value!r}" for key, value in zip(keys, values)])
        environ[env_var] = env_val

        with self.assertRaisesRegex(
            ValueError,
            f"Duplicated values for key {key!r}, got {values[0]!r} and {values[1]!r}!",
        ):
            _ = pcache.InMemoryCache.from_env_var(env_var)

    def test_no_unsupported_caches(self: Self) -> None:
        self.assertEqual(UNSUPPORTED_CACHES, [])


if __name__ == "__main__":
    run_tests()
