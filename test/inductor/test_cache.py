# Owner(s): ["module: inductor"]
from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from inspect import isclass
from os import environ
from pathlib import Path
from random import randint
from tempfile import gettempdir
from typing import Any, TYPE_CHECKING
from typing_extensions import Self
from unittest.mock import patch

from torch._inductor import cache as icache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class TestMixin:
    @staticmethod
    def abstract_cache_types() -> set[type[icache.Cache]]:
        return {icache.Cache, icache.AsyncCache}

    @staticmethod
    def cache_types() -> Sequence[type[icache.Cache]]:
        cache_types: list[type[icache.Cache]] = []

        for obj_name in dir(icache):
            obj = getattr(icache, obj_name)

            if not isclass(obj) or not issubclass(obj, icache.Cache):
                continue

            if obj in TestMixin.abstract_cache_types():
                continue

            cache_types.append(obj)
        return cache_types

    @staticmethod
    def async_cache_types() -> Sequence[type[icache.AsyncCache]]:
        return [
            cache_type
            for cache_type in TestMixin.cache_types()
            if issubclass(cache_type, icache.AsyncCache)
        ]

    @staticmethod
    def on_disk_cache_types() -> Sequence[type[icache.OnDiskCache]]:
        return [
            cache_type
            for cache_type in TestMixin.cache_types()
            if issubclass(cache_type, icache.OnDiskCache)
        ]

    @staticmethod
    def key_types() -> Sequence[type[icache.Key]]:
        return [*icache.Key.__constraints__]

    @staticmethod
    def value_types() -> Sequence[type[icache.Value]]:
        return [*icache.Value.__constraints__]

    @staticmethod
    def cache_type_supports_key_and_value_types(
        cache_type: type[icache.Cache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> bool:
        assert len(cache_type.__orig_bases__) == 1
        generic_base = cache_type.__orig_bases__[0]
        _key_type, _value_type = generic_base.__args__
        if ((_key_type != icache.Key) and (_key_type != key_type)) or (
            (_value_type != icache.Value) and (_value_type != value_type)
        ):
            return False
        return True

    def key_not_in(
        self: Self,
        cache: icache.Cache[icache.Key, icache.Value],
        key_fn: Callable[[], icache.Key],
    ) -> icache.Key:
        while cache.get(key := key_fn()) is not None:
            continue
        return key

    def keys_not_in(
        self: Self,
        cache: icache.Cache[icache.Key, icache.Value],
        key_fn: Callable[[], icache.Key],
        num: int,
    ) -> list[icache.Key]:
        keys = []
        while len(keys) < num:
            if (key := self.key_not_in(cache, key_fn)) not in keys:
                keys.append(key)
        return keys

    def key(self: Self, key_type: type[icache.Key]) -> icache.Key:
        if key_type == str:
            return f"s{randint(0, 2**32)}"
        elif key_type == int:
            return randint(0, 2**32)
        elif key_type == tuple[Any, ...]:
            return (self.key(str), self.key(int))
        else:
            raise NotImplementedError

    def values_unalike(
        self: Self, value_fn: Callable[[], icache.Value], num: int
    ) -> list[icache.Value]:
        values = []
        while len(values) < num:
            if (value := value_fn()) not in values:
                values.append(value)
        return values

    def value(self: Self, value_type: type[icache.Value]) -> icache.Value:
        if value_type == str:
            return f"s{randint(0, 2**32)}"
        elif value_type == int:
            return randint(0, 2**32)
        elif value_type == tuple[Any, ...]:
            return (self.value(str), self.value(int))
        elif value_type == bytes:
            return self.value(str).encode()
        elif value_type == dict[Any, Any]:
            return {
                "zero": self.value(str),
                1: self.value(int),
                (2): self.value(tuple[Any, ...]),
                b"three": self.value(bytes),
            }
        elif value_type == list[Any]:
            return [self.value(str), self.value(int), self.value(dict[Any, Any])]
        else:
            raise NotImplementedError

    def maybe_randomize_base_dir(self: Self, cache: icache.Cache) -> None:
        # multi on disk caches might exist at any time, and the tests
        # assume they are isolated so we should randomize their base dir
        if isinstance(cache, icache.OnDiskCache):
            cache.base_dir = cache.base_dir / f"{hash(cache)}"


@instantiate_parametrized_tests
class CacheTest(TestMixin, TestCase):
    @parametrize("cache_type", TestMixin.cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_get(
        self: Self,
        cache_type: type[icache.Cache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Checks that a cache returns None for missing keys, and after insertion,
        # returns the correct value for each key.
        if not self.cache_type_supports_key_and_value_types(
            cache_type, key_type, value_type
        ):
            return

        cache: icache.Cache = cache_type()
        self.maybe_randomize_base_dir(cache)
        key_1, key_2 = self.keys_not_in(cache, lambda: self.key(key_type), 2)
        value_1, value_2 = self.values_unalike(lambda: self.value(value_type), 2)

        self.assertIsNone(cache.get(key_1))
        self.assertIsNone(cache.get(key_2))

        self.assertTrue(cache.insert(key_1, value_1))
        self.assertTrue(cache.insert(key_2, value_2))

        self.assertEqual(cache.get(key_1), value_1)
        self.assertEqual(cache.get(key_2), value_2)

    @parametrize("cache_type", TestMixin.cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_insert(
        self: Self,
        cache_type: type[icache.Cache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Verifies that inserting a new key succeeds, inserting the same key again fails,
        # and the value for the key remains the first inserted value.
        if not self.cache_type_supports_key_and_value_types(
            cache_type, key_type, value_type
        ):
            return

        cache: icache.Cache = cache_type()
        self.maybe_randomize_base_dir(cache)
        key = self.key_not_in(cache, lambda: self.key(key_type))
        value_1, value_2 = self.values_unalike(lambda: self.value(value_type), 2)

        self.assertIsNone(cache.get(key))

        self.assertTrue(cache.insert(key, value_1))
        self.assertFalse(cache.insert(key, value_2))

        self.assertEqual(cache.get(key), value_1)

    @parametrize("cache_type", TestMixin.cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_get_concurrent(
        self: Self,
        cache_type: type[icache.Cache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Ensures that concurrent reads (get) from the cache return the correct values
        # for all inserted keys, even under parallel access.
        if not self.cache_type_supports_key_and_value_types(
            cache_type, key_type, value_type
        ):
            return

        executor, iters = ThreadPoolExecutor(), 100

        cache: icache.Cache = cache_type()
        self.maybe_randomize_base_dir(cache)
        keys = self.keys_not_in(cache, lambda: self.key(key_type), iters)
        values = self.values_unalike(lambda: self.value(value_type), iters)

        for key, value in zip(keys, values):
            self.assertIsNone(cache.get(key))
            self.assertTrue(cache.insert(key, value))

        gets = executor.map(cache.get, keys)
        for value, get in zip(values, gets):
            self.assertEqual(get, value)

        executor.shutdown()

    @parametrize("cache_type", TestMixin.cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_insert_concurrent(
        self: Self,
        cache_type: type[icache.Cache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Ensures that concurrent inserts work as expected: only the first insert for each key
        # succeeds, and the cache contains the correct value for each key after all inserts.
        if not self.cache_type_supports_key_and_value_types(
            cache_type, key_type, value_type
        ):
            return

        executor, iters = ThreadPoolExecutor(), 50

        cache: icache.Cache = cache_type()
        self.maybe_randomize_base_dir(cache)
        keys = self.keys_not_in(cache, lambda: self.key(key_type), iters) * 2
        values = self.values_unalike(lambda: self.value(value_type), iters * 2)

        for key in keys:
            self.assertIsNone(cache.get(key))

        inserts = executor.map(cache.insert, keys, values)
        inserted = {}
        for key, value, insert in zip(keys, values, inserts):
            if insert:
                self.assertEqual(cache.get(key), value)
                self.assertTrue(key not in inserted)
                inserted[key] = value

        self.assertTrue(set(keys) == set(inserted.keys()))
        for key, value in inserted.items():
            self.assertEqual(cache.get(key), value)

        executor.shutdown()

    @parametrize("cache_type", TestMixin.cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    @parametrize("get_first", [True, False])
    def test_combo_concurrent(
        self: Self,
        cache_type: type[icache.Cache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
        get_first: bool,
    ) -> None:
        # Tests a mix of concurrent get and insert operations, with the order of operations
        # varied by the get_first parameter, to ensure correctness under interleaved access.
        if not self.cache_type_supports_key_and_value_types(
            cache_type, key_type, value_type
        ):
            return

        executor, iters = ThreadPoolExecutor(), 50

        cache: icache.Cache = cache_type()
        self.maybe_randomize_base_dir(cache)
        keys = self.keys_not_in(cache, lambda: self.key(key_type), iters) * 2
        values = self.values_unalike(lambda: self.value(value_type), iters * 2)

        for key in keys:
            self.assertIsNone(cache.get(key))

        get_futures, insert_futures = [], []
        for key, value in zip(keys, values):
            if get_first:
                get_futures.append(executor.submit(cache.get, key))
                insert_futures.append(executor.submit(cache.insert, key, value))
            else:
                insert_futures.append(executor.submit(cache.insert, key, value))
                get_futures.append(executor.submit(cache.get, key))

        inserted = {}
        for key, value, get_future, insert_future in zip(
            keys, values, get_futures, insert_futures
        ):
            if (get := get_future.result()) is not None:
                if insert_future.result():
                    self.assertEqual(get, value)
                    self.assertTrue(key not in inserted)
                    inserted[key] = value
            else:
                if insert_future.result():
                    self.assertTrue(key not in inserted)
                    inserted[key] = value

        self.assertTrue(set(keys) == set(inserted.keys()))
        for key, value in inserted.items():
            self.assertEqual(cache.get(key), value)

        executor.shutdown()


@instantiate_parametrized_tests
class AsyncCacheTest(TestMixin, TestCase):
    @parametrize("async_cache_type", TestMixin.async_cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_get_async(
        self: Self,
        async_cache_type: type[icache.AsyncCache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Verifies that asynchronous get and insert operations work as expected:
        # get_async returns None for missing keys, insert_async inserts values,
        # and get_async returns the correct value after insertion.
        if not self.cache_type_supports_key_and_value_types(
            async_cache_type, key_type, value_type
        ):
            return

        async_cache: icache.AsyncCache = async_cache_type()
        self.maybe_randomize_base_dir(async_cache)
        key_1, key_2 = self.keys_not_in(async_cache, lambda: self.key(key_type), 2)
        value_1, value_2 = self.values_unalike(lambda: self.value(value_type), 2)

        executor = ThreadPoolExecutor()

        get_1 = async_cache.get_async(key_1, executor)
        get_2 = async_cache.get_async(key_2, executor)
        self.assertIsNone(get_1.result())
        self.assertIsNone(get_2.result())

        insert_1 = async_cache.insert_async(key_1, value_1, executor)
        insert_2 = async_cache.insert_async(key_2, value_2, executor)
        self.assertTrue(insert_1.result())
        self.assertTrue(insert_2.result())

        get_1 = async_cache.get_async(key_1, executor)
        get_2 = async_cache.get_async(key_2, executor)
        self.assertEqual(get_1.result(), value_1)
        self.assertEqual(get_2.result(), value_2)

        executor.shutdown()

    @parametrize("async_cache_type", TestMixin.async_cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_insert_async(
        self: Self,
        async_cache_type: type[icache.AsyncCache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Ensures that only one of two concurrent insert_async calls for the same key succeeds,
        # and the cache contains the value from the successful insert.
        if not self.cache_type_supports_key_and_value_types(
            async_cache_type, key_type, value_type
        ):
            return

        async_cache: icache.AsyncCache = async_cache_type()
        self.maybe_randomize_base_dir(async_cache)
        key = self.key_not_in(async_cache, lambda: self.key(key_type))
        value_1, value_2 = self.values_unalike(lambda: self.value(value_type), 2)

        executor = ThreadPoolExecutor()

        get = async_cache.get_async(key, executor)
        self.assertIsNone(get.result())

        insert_1 = async_cache.insert_async(key, value_1, executor)
        insert_2 = async_cache.insert_async(key, value_2, executor)
        self.assertTrue(insert_1.result() ^ insert_2.result())

        get = async_cache.get_async(key, executor)
        if insert_1.result():
            self.assertEqual(get.result(), value_1)
        else:
            self.assertEqual(get.result(), value_2)

        executor.shutdown()

    @parametrize("async_cache_type", TestMixin.async_cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_get_async_concurrent(
        self: Self,
        async_cache_type: type[icache.AsyncCache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Ensures that concurrent asynchronous get operations return the correct values
        # for all inserted keys.
        if not self.cache_type_supports_key_and_value_types(
            async_cache_type, key_type, value_type
        ):
            return

        executor, iters = ThreadPoolExecutor(), 100

        async_cache: icache.AsyncCache = async_cache_type()
        self.maybe_randomize_base_dir(async_cache)
        keys = self.keys_not_in(async_cache, lambda: self.key(key_type), iters)
        values = self.values_unalike(lambda: self.value(value_type), iters)

        for key, value in zip(keys, values):
            self.assertIsNone(async_cache.get(key))
            self.assertTrue(async_cache.insert(key, value))

        gets = executor.map(lambda key: async_cache.get_async(key, executor), keys)
        for value, get in zip(values, gets):
            self.assertEqual(get.result(), value)

        executor.shutdown()

    @parametrize("async_cache_type", TestMixin.async_cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_insert_async_concurrent(
        self: Self,
        async_cache_type: type[icache.AsyncCache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
    ) -> None:
        # Ensures that concurrent asynchronous insert operations only allow the first insert
        # for each key to succeed, and the cache contains the correct value for each key.
        if not self.cache_type_supports_key_and_value_types(
            async_cache_type, key_type, value_type
        ):
            return

        executor, iters = ThreadPoolExecutor(), 50

        async_cache: icache.AsyncCache = async_cache_type()
        self.maybe_randomize_base_dir(async_cache)
        keys = self.keys_not_in(async_cache, lambda: self.key(key_type), iters) * 2
        values = self.values_unalike(lambda: self.value(value_type), iters * 2)

        for key in keys:
            self.assertIsNone(async_cache.get(key))

        inserts = executor.map(
            lambda key, value: async_cache.insert_async(key, value, executor),
            keys,
            values,
        )
        inserted = {}
        for key, value, insert in zip(keys, values, inserts):
            if insert.result():
                self.assertEqual(async_cache.get(key), value)
                self.assertTrue(key not in inserted)
                inserted[key] = value

        self.assertTrue(set(keys) == set(inserted.keys()))
        for key, value in inserted.items():
            self.assertTrue(async_cache.get(key), value)

        executor.shutdown()

    @parametrize("async_cache_type", TestMixin.async_cache_types())
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    @parametrize("get_first", [True, False])
    def test_combo_async_concurrent(
        self: Self,
        async_cache_type: type[icache.AsyncCache],
        key_type: type[icache.Key],
        value_type: type[icache.Value],
        get_first: bool,
    ) -> None:
        # Tests a mix of concurrent asynchronous get and insert operations, with the order
        # of operations varied by the get_first parameter, to ensure correctness under
        # interleaved async access.
        if not self.cache_type_supports_key_and_value_types(
            async_cache_type, key_type, value_type
        ):
            return

        executor, iters = ThreadPoolExecutor(), 50

        async_cache: icache.AsyncCache = async_cache_type()
        self.maybe_randomize_base_dir(async_cache)
        keys = self.keys_not_in(async_cache, lambda: self.key(key_type), iters) * 2
        values = self.values_unalike(lambda: self.value(value_type), iters * 2)

        for key in keys:
            self.assertIsNone(async_cache.get(key))

        get_futures, insert_futures = [], []
        for key, value in zip(keys, values):
            if get_first:
                get_futures.append(async_cache.get_async(key, executor))
                insert_futures.append(async_cache.insert_async(key, value, executor))
            else:
                insert_futures.append(async_cache.insert_async(key, value, executor))
                get_futures.append(async_cache.get_async(key, executor))

        inserted = {}
        for key, value, get_future, insert_future in zip(
            keys, values, get_futures, insert_futures
        ):
            if (get := get_future.result()) is not None:
                if insert_future.result():
                    self.assertEqual(get, value)
                    self.assertTrue(key not in inserted)
                    inserted[key] = value
            else:
                if insert_future.result():
                    self.assertTrue(key not in inserted)
                    inserted[key] = value

        self.assertTrue(set(keys) == set(inserted.keys()))
        for key, value in inserted.items():
            self.assertEqual(async_cache.get(key), value)

        executor.shutdown()


@instantiate_parametrized_tests
class OtherTest(TestMixin, TestCase):
    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    @parametrize("with_whitespace", [True, False])
    @parametrize("with_semicolon_suffix", [True, False])
    def test_in_memory_cache_from_env_var(
        self: Self,
        key_type: type[icache.Key],
        value_type: type[icache.Value],
        with_whitespace: bool,
        with_semicolon_suffix: bool,
    ) -> None:
        # Verifies that InMemoryCache.from_env_var correctly parses environment variables
        # with various whitespace and semicolon suffixes, and loads all key-value pairs.
        keys = self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3)
        values = self.values_unalike(lambda: self.value(value_type), 3)

        ws = "" if not with_whitespace else " "

        env_var = "IN_MEMORY_CACHE_FROM_ENV_VAR_TEST"
        env_val = ";".join(
            [
                f"{ws}{pickle.dumps(key)!r}{ws},{ws}{pickle.dumps(value)!r}{ws}"
                for key, value in zip(keys, values)
            ]
        ) + (";" if with_semicolon_suffix else "")
        environ[env_var] = env_val

        cache = icache.InMemoryCache.from_env_var(env_var)
        for key, value in zip(keys, values):
            self.assertEqual(cache.get(key), value)

    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_in_memory_cache_from_env_var_missing_comma_separator(
        self: Self, key_type: type[icache.Key], value_type: type[icache.Value]
    ) -> None:
        # Ensures that InMemoryCache.from_env_var raises CacheError if the environment
        # variable is missing the required comma separator between key and value.
        keys = self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3)
        values = self.values_unalike(lambda: self.value(value_type), 3)

        env_var = "IN_MEMORY_CACHE_FROM_ENV_VAR_MISSING_COMMA_SEPARATOR_TEST"
        env_val = ";".join(
            [
                f"{pickle.dumps(key)!r}{pickle.dumps(value)!r}"
                for key, value in zip(keys, values)
            ]
        )
        environ[env_var] = env_val

        with self.assertRaises(icache.CacheError):
            _ = icache.InMemoryCache.from_env_var(env_var)

    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_in_memory_cache_from_env_var_bad_encoding(
        self: Self, key_type: type[icache.Key], value_type: type[icache.Value]
    ) -> None:
        # Ensures that InMemoryCache.from_env_var raises CacheError if the key or value
        # encoding in the environment variable is invalid (not a valid Python literal).
        keys = self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3)
        values = self.values_unalike(lambda: self.value(value_type), 3)

        env_var = "IN_MEMORY_CACHE_FROM_ENV_VAR_BAD_ENCODING_TEST"
        env_val = ";".join(
            [
                f"{pickle.dumps(key)!r}/,{pickle.dumps(value)!r}/"
                for key, value in zip(keys, values)
            ]
        )
        environ[env_var] = env_val

        with self.assertRaises(icache.CacheError):
            _ = icache.InMemoryCache.from_env_var(env_var)

    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_in_memory_cache_from_env_var_not_un_pickle_able(
        self: Self, key_type: type[icache.Key], value_type: type[icache.Value]
    ) -> None:
        # Ensures that InMemoryCache.from_env_var raises CacheError if the key or value
        # cannot be unpickled (e.g., due to data corruption).
        keys = self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3)
        values = self.values_unalike(lambda: self.value(value_type), 3)

        env_var = "IN_MEMORY_CACHE_FROM_ENV_VAR_NOT_UN_PICKLE_ABLE_TEST"
        env_val = ";".join(
            [
                f"{pickle.dumps(key)[::-1]!r},{pickle.dumps(value)[::-1]!r}"
                for key, value in zip(keys, values)
            ]
        )
        environ[env_var] = env_val

        with self.assertRaises(icache.CacheError):
            _ = icache.InMemoryCache.from_env_var(env_var)

    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_in_memory_cache_from_env_var_duplicated_entries(
        self: Self, key_type: type[icache.Key], value_type: type[icache.Value]
    ) -> None:
        # Verifies that duplicate key-value pairs are allowed if the value is consistent,
        # but raises CacheError if the same key appears with different values.

        keys = (
            self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3) * 2
        )
        values = self.values_unalike(lambda: self.value(value_type), 3) * 2

        env_var = "IN_MEMORY_CACHE_FROM_ENV_VAR_DUPLICATED_ENTRIES_TEST"
        env_val = ";".join(
            [
                f"{pickle.dumps(key)!r},{pickle.dumps(value)!r}"
                for key, value in zip(keys, values)
            ]
        )
        environ[env_var] = env_val

        # duplicate key => value entries are okay, as long as value is consistent
        cache = icache.InMemoryCache.from_env_var(env_var)
        for key, value in zip(keys, values):
            self.assertEqual(cache.get(key), value)

        keys = (
            self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3) * 2
        )
        values = self.values_unalike(lambda: self.value(value_type), 6)

        env_var = "IN_MEMORY_CACHE_FROM_ENV_VAR_DUPLICATED_ENTRIES_TEST"
        env_val = ";".join(
            [
                f"{pickle.dumps(key)!r},{pickle.dumps(value)!r}"
                for key, value in zip(keys, values)
            ]
        )
        environ[env_var] = env_val

        # duplicate key => value entries with inconsistent values are not okay
        with self.assertRaises(icache.CacheError):
            _ = icache.InMemoryCache.from_env_var(env_var)

    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_in_memory_cache_from_file_path(
        self: Self, key_type: type[icache.Key], value_type: type[icache.Value]
    ) -> None:
        # Checks that InMemoryCache.from_file_path correctly loads a cache from a file
        # containing a pickled dictionary of key-value pairs.
        keys = self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3)
        values = self.values_unalike(lambda: self.value(value_type), 3)

        cache = icache.InMemoryCache()

        for key, value in zip(keys, values):
            self.assertTrue(cache.insert(key, value))

        fpath = Path(gettempdir()) / "IN_MEMORY_CACHE_FROM_FILE_PATH_TEST"
        with open(fpath, "wb") as fp:
            pickle.dump(cache._cache, fp)

        from_file_path_cache = icache.InMemoryCache.from_file_path(fpath)

        for key, value in zip(keys, values):
            self.assertEqual(from_file_path_cache.get(key), value)

    @parametrize("key_type", TestMixin.key_types())
    @parametrize("value_type", TestMixin.value_types())
    def test_in_memory_cache_from_file_path_not_un_pickle_able(
        self: Self, key_type: type[icache.Key], value_type: type[icache.Value]
    ) -> None:
        # Ensures that InMemoryCache.from_file_path raises CacheError if the file contents
        # cannot be unpickled (e.g., due to corruption).
        keys = self.keys_not_in(icache.InMemoryCache(), lambda: self.key(key_type), 3)
        values = self.values_unalike(lambda: self.value(value_type), 3)

        cache = icache.InMemoryCache()

        for key, value in zip(keys, values):
            self.assertTrue(cache.insert(key, value))

        fpath = (
            Path(gettempdir())
            / "IN_MEMORY_CACHE_FROM_FILE_PATH_NOT_UN_PICKLE_ABLE_TEST"
        )
        with open(fpath, "wb") as fp:
            pickled_cache = pickle.dumps(cache._cache)
            pickled_cache = pickled_cache[::-1]
            fp.write(pickled_cache)

        with self.assertRaises(icache.CacheError):
            _ = icache.InMemoryCache.from_file_path(fpath)

    def test_in_memory_cache_from_file_path_not_dict(self: Self) -> None:
        # This test verifies that InMemoryCache.from_file_path raises a CacheError
        # when the file does not contain a pickled dictionary. It writes a pickled
        # list to a temporary file, then attempts to load it as a cache. The cache
        # expects a dictionary structure; loading a non-dictionary should raise an error.
        fpath = Path(gettempdir()) / "IN_MEMORY_CACHE_FROM_FILE_PATH_NOT_DICT_TEST"
        with open(fpath, "wb") as fp:
            pickled_cache = pickle.dumps([1, 2, 3])
            fp.write(pickled_cache)

        with self.assertRaises(icache.CacheError):
            _ = icache.InMemoryCache.from_file_path(fpath)

    @parametrize("on_disk_cache_type", TestMixin.on_disk_cache_types())
    def test_on_disk_cache_fpath_from_key_un_pickle_able(
        self: Self, on_disk_cache_type: type[icache.OnDiskCache]
    ) -> None:
        # This test checks that _fpath_from_key raises a CacheError when given a
        # key that cannot be pickled. It passes a lambda function (which is not
        # pickle-able) as the key. The cache uses pickling to serialize keys for
        # file storage. If a key cannot be pickled, the cache should fail gracefully
        # and raise a clear error.
        cache: icache.OnDiskCache = on_disk_cache_type()
        un_pickle_able_key = lambda: None  # noqa: E731

        with self.assertRaises(icache.CacheError):
            _ = cache._fpath_from_key(un_pickle_able_key)

    @parametrize("on_disk_cache_type", TestMixin.on_disk_cache_types())
    def test_on_disk_cache_version_bump(
        self: Self, on_disk_cache_type: type[icache.OnDiskCache]
    ) -> None:
        # This test ensures that cache entries are invalidated when the cache version
        # changes, and that new entries can be inserted and retrieved after a version bump.
        # It inserts a key-value pair, then simulates a version bump by patching the cache
        # version. After the version change, it verifies that the old entry is no longer
        # retrievable (invalidated), and that a new entry can be inserted and retrieved.
        # Versioning is used to invalidate stale cache entries when the cache format or
        # logic changes.
        cache: icache.OnDiskCache = on_disk_cache_type()
        key = self.key_not_in(cache, lambda: self.key(str))
        value = self.value(str)

        self.assertIsNone(cache.get(key))
        self.assertTrue(cache.insert(key, value))
        self.assertEqual(cache.get(key), value)

        old_version = icache.OnDiskCache.version
        bump_version = old_version + 1
        with patch.object(icache.OnDiskCache, "version", bump_version):
            self.assertIsNone(cache.get(key))
            self.assertTrue(cache.insert(key, value))
            self.assertEqual(cache.get(key), value)

        self.assertIsNone(cache.get(key))
        self.assertTrue(cache.insert(key, value))
        self.assertEqual(cache.get(key), value)


if __name__ == "__main__":
    run_tests()
