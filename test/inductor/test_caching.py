# Owner(s): ["module: inductor"]
# pyre-strict
from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from contextlib import contextmanager
from functools import wraps
from itertools import combinations
from random import Random
from shutil import rmtree
from threading import Event, Lock
from typing import Any, TYPE_CHECKING, Union
from typing_extensions import TypeVar
from unittest.mock import patch

from filelock import FileLock

from torch._inductor.runtime.caching import (
    config,
    context,
    exceptions,
    implementations as impls,
    interfaces,
    locks,
    Memoizer,
    PersistentMemoizer,
    utils,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path


set_caching_module_enabled = lambda enabled: patch.object(  # noqa: E731
    config, "IS_CACHING_MODULE_ENABLED", lambda: enabled
)
set_deterministic_caching_enabled = lambda enabled: patch.object(  # noqa: E731
    config, "IS_DETERMINISTIC_CACHING_ENABLED", lambda: enabled
)
set_strictly_pre_populated_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "STRICTLY_PRE_POPULATED_DETERMINISM", enabled
)
set_strictly_cached_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "STRICTLY_CACHED_DETERMINISM", enabled
)
set_local_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "LOCAL_DETERMINISM", enabled
)
set_global_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "GLOBAL_DETERMINISM", enabled
)


def patch_on_disk_cache_base_dir(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        default_base_dir = impls._OnDiskCacheImpl()._base_dir
        with patch.object(
            impls._OnDiskCacheImpl,
            "_base_dir",
            default_base_dir / f"{self.sub_dir()}/rng-{self.random_string[4:]}",
        ):
            return fn(self, *args, **kwargs)

    return wrapper


def patch_remote_cache_with_on_disk_cache(fn):
    impls._OnDiskCacheImpl.has_strong_consistency = True
    return patch.object(impls, "_RemoteCacheImpl", impls._OnDiskCacheImpl)(fn)


class TestMixin:
    impl_typenames: list[str] = [
        "_InMemoryCacheImpl",
        "_OnDiskCacheImpl",
    ]
    cls_id: int = Random().randint(0, 2**32)

    def impl_from_typename(self, impl_typename: str) -> impls._CacheImpl:
        return getattr(impls, impl_typename)()

    @property
    def random_string(self) -> str:
        return f"s-{Random().randint(0, 2**32)}"

    @property
    def random_bytes(self) -> bytes:
        return f"s-{Random().randint(0, 2**32)}".encode()


@instantiate_parametrized_tests
class ConfigTest(TestCase):
    FOO_THIS_VERSION: int = 0
    FOO_JK_NAME: str = "foo_jk_name"
    FOO_OSS_DEFAULT: bool = False
    FOO_ENV_VAR_OVERRIDE: str = "foo_env_var_override"
    FOO_ENV_VAR_OVERRIDE_LOCK_FPATH: str = f"/tmp/testing/{FOO_ENV_VAR_OVERRIDE}.lock"
    FOO_ENV_VAR_OVERRIDE_LOCK: FileLock = FileLock(FOO_ENV_VAR_OVERRIDE_LOCK_FPATH)

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(cls.FOO_ENV_VAR_OVERRIDE_LOCK_FPATH, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.FOO_ENV_VAR_OVERRIDE_LOCK_FPATH, ignore_errors=True)

    def assert_versioned_config(self, expected_enabled: bool) -> None:
        config._versioned_config.cache_clear()
        actual_enabled: bool = config._versioned_config(
            self.FOO_JK_NAME,
            self.FOO_THIS_VERSION,
            self.FOO_OSS_DEFAULT,
            env_var_override=self.FOO_ENV_VAR_OVERRIDE,
        )
        self.assertEqual(actual_enabled, expected_enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_env_var_override(
        self,
        enabled: bool,
    ) -> None:
        """Test that environment variable overrides take precedence over other configuration sources.

        Verifies that when an environment variable override is set to "1" or "0",
        the _versioned_config function returns the corresponding boolean value
        regardless of other configuration settings.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(
                os.environ,
                {
                    self.FOO_ENV_VAR_OVERRIDE: "1" if enabled else "0",
                },
            ),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", not enabled),
        ):
            self.assert_versioned_config(enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_version_check(
        self,
        enabled: bool,
    ) -> None:
        """Test that _versioned_config responds correctly to version changes in Facebook environments.

        Verifies that when running in fbcode environments (is_fbcode=True), the configuration
        is enabled when the JustKnobs version matches the expected version, and disabled when
        the version differs. This ensures proper rollout control through version management.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=True,
            ),
            patch(
                "torch._utils_internal.justknobs_getval_int",
                return_value=self.FOO_THIS_VERSION + (-1 if enabled else 1),
            ),
        ):
            self.assert_versioned_config(enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_oss_default(
        self,
        enabled: bool,
    ) -> None:
        """Test that _versioned_config uses OSS default values in non-Facebook environments.

        Verifies that when running in non-fbcode environments (is_fbcode=False) with no
        environment variable overrides, the configuration falls back to the OSS default
        value. This ensures proper behavior for open-source PyTorch distributions.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", enabled),
        ):
            self.assert_versioned_config(enabled)

    def test_versioned_config_jk_failure(self) -> None:
        """Test that _versioned_config uses OSS default values in non-Facebook environments.

        Verifies that when running in non-fbcode environments (is_fbcode=False) with no
        environment variable overrides, the configuration falls back to the OSS default
        value. This ensures proper behavior for open-source PyTorch distributions.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=True,
            ),
            patch(
                "torch._utils_internal.justknobs_getval_int",
                return_value=0,
            ),
        ):
            self.assert_versioned_config(False)


@instantiate_parametrized_tests
class ContextTest(TestCase):
    def isolation_schema_from_forms_of_context_selected(
        self,
        runtime_forms_of_context_selected: Sequence[str],
        compile_forms_of_context_selected: Sequence[str],
    ) -> context.IsolationSchema:
        return context.IsolationSchema(
            runtime_context={
                form_of_context: form_of_context
                in set(runtime_forms_of_context_selected)
                for form_of_context in context._RuntimeContext.forms_of_context()
            },
            compile_context={
                form_of_context: form_of_context
                in set(compile_forms_of_context_selected)
                for form_of_context in context._CompileContext.forms_of_context()
            },
        )

    @parametrize(
        "runtime_forms_of_context_selected",
        [(), *list(combinations(context._RuntimeContext.forms_of_context(), 2))],
    )
    @parametrize(
        "compile_forms_of_context_selected",
        [(), *list(combinations(context._CompileContext.forms_of_context(), 2))],
    )
    def test_selected_isolation_context(
        self,
        runtime_forms_of_context_selected: Sequence[str],
        compile_forms_of_context_selected: Sequence[str],
    ) -> None:
        """
        Tests that isolation context generation works correctly for specific combinations
        of runtime and compile context forms.

        Verifies that the _isolation_context function properly creates isolation contexts
        based on the selected forms of runtime and compile context, ensuring that only
        the specified context forms are included in the resulting isolation context.
        """
        ischema: context.IsolationSchema = (
            self.isolation_schema_from_forms_of_context_selected(
                runtime_forms_of_context_selected, compile_forms_of_context_selected
            )
        )

        self.assertEqual(
            context._isolation_context(ischema),
            {
                "runtime_context": {
                    form_of_context: getattr(context._RuntimeContext, form_of_context)()
                    for form_of_context in runtime_forms_of_context_selected
                }
                or None,
                "compile_context": {
                    form_of_context: getattr(context._CompileContext, form_of_context)()
                    for form_of_context in compile_forms_of_context_selected
                }
                or None,
            },
        )

    @parametrize("all_runtime_context", [True, False])
    @parametrize("all_compile_context", [True, False])
    def test_all_or_none_isolation_context(
        self, all_runtime_context: bool, all_compile_context: bool
    ) -> None:
        """
        Tests isolation context generation when using all or no context forms.

        Verifies that the isolation context correctly includes all forms of context
        when set to True, or excludes all forms when set to False, for both
        runtime and compile contexts.
        """
        ischema: context.IsolationSchema = context.IsolationSchema(
            runtime_context=all_runtime_context, compile_context=all_compile_context
        )
        self.assertEqual(
            context._isolation_context(ischema),
            {
                "runtime_context": {
                    form_of_context: getattr(context._RuntimeContext, form_of_context)()
                    for form_of_context in context._RuntimeContext.forms_of_context()
                }
                if all_runtime_context
                else None,
                "compile_context": {
                    form_of_context: getattr(context._CompileContext, form_of_context)()
                    for form_of_context in context._CompileContext.forms_of_context()
                }
                if all_compile_context
                else None,
            },
        )

    def test_isolation_key_is_distinct(self) -> None:
        """
        Tests that different combinations of runtime and compile context forms
        generate unique isolation keys.

        Verifies that each possible combination of context forms produces a distinct
        isolation key, ensuring no collisions occur between different contexts.
        """
        ikeys: set[str] = set()
        for num_runtime_forms_of_context_selected in range(
            len(context._RuntimeContext.forms_of_context())
        ):
            for num_compile_forms_of_context_selected in range(
                len(context._CompileContext.forms_of_context())
            ):
                for runtime_forms_of_context_selected in combinations(
                    context._RuntimeContext.forms_of_context(),
                    num_runtime_forms_of_context_selected,
                ):
                    for compile_forms_of_context_selected in combinations(
                        context._CompileContext.forms_of_context(),
                        num_compile_forms_of_context_selected,
                    ):
                        ischema: context.IsolationSchema = (
                            self.isolation_schema_from_forms_of_context_selected(
                                runtime_forms_of_context_selected,
                                compile_forms_of_context_selected,
                            )
                        )
                        ikey: str = context._isolation_key(ischema)
                        self.assertFalse(ikey in ikeys)
                        ikeys.add(ikey)

    def test_isolation_key_is_repeatable(self) -> None:
        """
        Tests that calling the isolation key function multiple times with the same
        parameters produces the same result.

        Verifies that the isolation key generation is deterministic and consistent
        across multiple invocations with identical inputs.
        """
        self.assertEqual(context._isolation_key(), context._isolation_key())

    def test_select_runtime_context_matches_forms_of_context(self) -> None:
        """
        Tests that the selected runtime context matches the forms of context.

        Verifies that the selected runtime context includes only the forms of context
        specified in the isolation schema, ensuring that the isolation context is
        properly selected and configured.
        """
        self.assertEqual(
            set(context.SelectedRuntimeContext.__required_keys__),
            set(context._RuntimeContext.forms_of_context()),
        )

    def test_select_compile_context_matches_forms_of_context(self) -> None:
        """
        Tests that the selected compile context matches the forms of context.

        Verifies that the selected compile context includes only the forms of context
        specified in the isolation schema, ensuring that the isolation context is
        properly selected and configured.
        """
        self.assertEqual(
            set(context.SelectedCompileContext.__required_keys__),
            set(context._CompileContext.forms_of_context()),
        )


@instantiate_parametrized_tests
class ExceptionsTest(TestCase):
    exception_typenames: list[str] = [
        "CacheError",
        "SystemError",
        "LockTimeoutError",
        "FileLockTimeoutError",
        "UserError",
        "KeyEncodingError",
        "ValueEncodingError",
        "ValueDecodingError",
    ]

    @parametrize("exception_typename", exception_typenames)
    def test_exception_is_CacheError(self, exception_typename: str) -> None:
        """Test that all custom cache exceptions inherit from the base CacheError class.

        Verifies that every exception type defined in the caching exceptions module
        is properly derived from CacheError, ensuring consistent exception hierarchy
        and enabling unified exception handling throughout the caching system.
        """
        self.assertTrue(
            issubclass(getattr(exceptions, exception_typename), exceptions.CacheError)
        )

    def test_exception_other(self) -> None:
        """
        Test the inheritance relationships among custom cache exception classes.

        Verifies that the exception classes in the caching exceptions module have the correct
        subclass relationships, ensuring the exception hierarchy is as intended. This includes
        checks for both direct and indirect inheritance between base and derived exception types.
        """
        self.assertTrue(issubclass(exceptions.SystemError, exceptions.CacheError))
        self.assertTrue(issubclass(exceptions.LockTimeoutError, exceptions.SystemError))
        self.assertTrue(
            issubclass(exceptions.FileLockTimeoutError, exceptions.SystemError)
        )
        self.assertTrue(issubclass(exceptions.UserError, exceptions.CacheError))
        self.assertTrue(issubclass(exceptions.KeyEncodingError, exceptions.UserError))
        self.assertTrue(issubclass(exceptions.ValueEncodingError, exceptions.UserError))
        self.assertTrue(issubclass(exceptions.ValueDecodingError, exceptions.UserError))


@instantiate_parametrized_tests
class ImplementationsTest(TestMixin, TestCase):
    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-impls-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir, ignore_errors=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir, ignore_errors=True
        )

    def assert_key_in(self, key: str, impl: impls._CacheImpl) -> None:
        self.assertTrue(impl.get(key) is not None)

    def assert_key_not_in(self, key: str, impl: impls._CacheImpl) -> None:
        self.assertTrue(impl.get(key) is None)

    def assert_key_value_inserted_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertTrue(impl.insert(key, value))

    def assert_key_value_not_inserted_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertFalse(impl.insert(key, value))

    def assert_key_has_value_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertTrue(((get := impl.get(key)) is not None) and (get.value == value))

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_get(self, impl_typename: str) -> None:
        """Test cache get operation returns cache miss for non-existent keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle get operations for keys that do not exist in the cache. This test
        ensures that the cache properly returns a cache miss (hit=False) when
        attempting to retrieve a non-existent key.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            self.assert_key_not_in(self.random_string, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_insert(self, impl_typename: str) -> None:
        """Test cache insert operation successfully stores and retrieves key-value pairs.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle insert operations for new key-value pairs. This test ensures that:
        1. Keys initially don't exist in the cache (cache miss)
        2. Insert operations succeed for new keys
        3. The stored value can be retrieved correctly after insertion

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            key: str = self.random_string
            self.assert_key_not_in(key, impl)
            value: bytes = self.random_bytes
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_insert_will_not_overwrite(self, impl_typename: str) -> None:
        """Test cache insert operation does not overwrite existing keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle insert operations for keys that already exist in the cache. This test
        ensures that:
        1. Keys initially don't exist in the cache (cache miss)
        2. First insert operation succeeds for new keys
        3. Subsequent insert operations with the same key fail (inserted=False)
        4. The original value is preserved and not overwritten

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            key: str = self.random_string
            self.assert_key_not_in(key, impl)
            value: bytes = self.random_bytes
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_value_not_inserted_in(key, self.random_bytes, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_encoding(self, impl_typename: str) -> None:
        """Test that in-memory cache can store any value type.

        Verifies that in-memory cache implementations can store arbitrary values
        including non-serializable ones (such as lambda functions) since they don't
        require serialization. On-disk caches now expect bytes, so they skip this test.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._InMemoryCacheImpl):
                key: str = self.random_string
                value = lambda: None  # noqa: E731
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_decoding(self, impl_typename: str) -> None:
        """Test that on-disk cache implementations return raw bytes from storage.

        Verifies that on-disk cache implementations return raw bytes from disk
        without attempting to unpickle them, as values are now expected to be
        stored as bytes. This test writes raw bytes to a cache file and confirms
        they are returned as-is.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                fpath: Path = impl._fpath_from_key(key)
                test_bytes: bytes = b"foo"
                with open(fpath, "xb") as fp:
                    impl._write_version_header(fp)
                    fp.write(test_bytes)
                self.assert_key_has_value_in(key, test_bytes, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_version_mismatch(self, impl_typename: str) -> None:
        """Test that on-disk cache implementations properly handle version mismatches.

        Verifies that on-disk cache implementations correctly handle cached data when
        the cache version changes. This test ensures that:
        1. Data can be stored and retrieved with the current version
        2. When version changes, previously cached data becomes inaccessible (cache miss)
        3. New data can be stored with the new version
        4. After version change, old cached data remains inaccessible

        This version checking mechanism prevents corruption and compatibility issues
        when cache formats change between software versions. Only applies to on-disk
        implementations since in-memory caches don't persist across version changes.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                value: bytes = self.random_bytes
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)
                with patch.object(
                    impls._OnDiskCacheImpl, "_version", impl._version + 1
                ):
                    self.assert_key_not_in(key, impl)
                    self.assert_key_value_inserted_in(key, value, impl)
                    self.assert_key_has_value_in(key, value, impl)
                self.assert_key_not_in(key, impl)
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)


@instantiate_parametrized_tests
class LocksTest(TestMixin, TestCase):
    T = TypeVar("T")

    @contextmanager
    def executor(self) -> Generator[ThreadPoolExecutor, None, None]:
        executor: ThreadPoolExecutor = ThreadPoolExecutor()
        try:
            yield executor
        finally:
            executor.shutdown()

    def is_lock(self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        return hasattr(lock_or_flock, "locked")

    def is_flock(self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        return hasattr(lock_or_flock, "is_locked")

    def lock_or_flock_locked(self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        if self.is_lock(lock_or_flock):
            return lock_or_flock.locked()
        elif self.is_flock(lock_or_flock):
            return lock_or_flock.is_locked
        else:
            raise NotImplementedError

    def test_BLOCKING(self) -> None:
        self.assertEqual(locks._BLOCKING, -1.0)

    def test_NON_BLOCKING(self) -> None:
        self.assertEqual(locks._NON_BLOCKING, 0.0)

    def test_BLOCKING_WITH_TIMEOUT(self) -> None:
        self.assertGreater(locks._BLOCKING_WITH_TIMEOUT, 0.0)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1.0)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1.0)
    @parametrize("lock_typename", ["Lock", "FileLock"])
    @parametrize("lock_timeout", ["BLOCKING", "NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"])
    @parametrize("acquisition_mode", ["safe", "unsafe"])
    @parametrize("release", ["unlocked", "never", "before_timeout", "after_timeout"])
    def test_acquire_with_timeout(
        self,
        lock_typename: str,
        lock_timeout: str,
        acquisition_mode: str,
        release: str,
    ) -> None:
        """Test lock acquisition behavior with various timeout configurations and release scenarios.

        This comprehensive test verifies the lock acquisition functionality for both threading.Lock
        and FileLock objects across different timeout modes, acquisition patterns, and release timings.
        The test validates proper exception handling, timeout behavior, and correct lock state management.

        Test parameters:
        - lock_typename: Tests both "Lock" (threading.Lock) and "FileLock" (filelock.FileLock) types
        - lock_timeout: Tests "BLOCKING", "NON_BLOCKING", and "BLOCKING_WITH_TIMEOUT" modes
        - acquisition_mode: Tests both "safe" (context manager) and "unsafe" (manual) acquisition
        - release: Tests "unlocked", "never", "before_timeout", and "after_timeout" scenarios

        The test ensures that:
        - Safe acquisition properly manages lock lifecycle through context managers
        - Unsafe acquisition requires manual release and behaves correctly
        - Timeout exceptions are raised appropriately for different timeout configurations
        - Lock states are correctly maintained throughout acquisition and release cycles
        - Different lock types (Lock vs FileLock) behave consistently with their respective APIs
        """

        def inner(lock_or_flock: Union[Lock, FileLock], timeout: int) -> None:
            if self.is_lock(lock_or_flock):
                lock: Lock = lock_or_flock
                if acquisition_mode == "safe":
                    with locks._acquire_lock_with_timeout(lock, timeout=timeout):
                        self.assertTrue(self.lock_or_flock_locked(lock))
                elif acquisition_mode == "unsafe":
                    locks._unsafe_acquire_lock_with_timeout(lock, timeout=timeout)
                    self.assertTrue(self.lock_or_flock_locked(lock))
                    lock.release()
                else:
                    raise NotImplementedError
            elif self.is_flock(lock_or_flock):
                flock: FileLock = lock_or_flock
                if acquisition_mode == "safe":
                    with locks._acquire_flock_with_timeout(flock, timeout=timeout):
                        self.assertTrue(self.lock_or_flock_locked(flock))
                elif acquisition_mode == "unsafe":
                    locks._unsafe_acquire_flock_with_timeout(flock, timeout=timeout)
                    self.assertTrue(self.lock_or_flock_locked(flock))
                    flock.release()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.assertFalse(self.lock_or_flock_locked(lock_or_flock))

        if lock_typename not in ["Lock", "FileLock"]:
            raise AssertionError(f"Unexpected lock_typename: {lock_typename}")
        flock_fpath: Path = (
            impls._OnDiskCacheImpl()._cache_dir
            / f"testing-locks-instance-{self.random_string}.lock"
        )
        lock_or_flock: Union[Lock, FileLock] = (
            Lock() if lock_typename == "Lock" else FileLock(str(flock_fpath))
        )
        lock_exception_type: type = (
            exceptions.LockTimeoutError
            if lock_typename == "Lock"
            else exceptions.FileLockTimeoutError
        )

        if release == "unlocked":
            self.assertFalse(self.lock_or_flock_locked(lock_or_flock))
        elif release in ["never", "before_timeout", "after_timeout"]:
            self.assertTrue(lock_or_flock.acquire(timeout=locks._NON_BLOCKING))
            self.assertTrue(self.lock_or_flock_locked(lock_or_flock))
        else:
            raise NotImplementedError

        with self.executor() as executor:
            if lock_timeout not in [
                "BLOCKING",
                "NON_BLOCKING",
                "BLOCKING_WITH_TIMEOUT",
            ]:
                raise AssertionError(f"Unexpected lock_timeout: {lock_timeout}")
            lock_or_flock_future: Future[None] = executor.submit(
                inner,
                lock_or_flock,
                timeout={
                    "BLOCKING": locks._BLOCKING,
                    "NON_BLOCKING": locks._NON_BLOCKING,
                    "BLOCKING_WITH_TIMEOUT": locks._BLOCKING_WITH_TIMEOUT,
                }[lock_timeout],
            )

            if release == "unlocked":
                self.assertIsNone(lock_or_flock_future.result())
            elif release == "never":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT * 2))
                if lock_timeout == "BLOCKING":
                    with self.assertRaises(TimeoutError):
                        lock_or_flock_future.result(
                            timeout=locks._BLOCKING_WITH_TIMEOUT
                        )
                elif lock_timeout in ["NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError
                lock_or_flock.release()
            elif release == "before_timeout":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT / 2))
                lock_or_flock.release()
                if lock_timeout in ["BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    self.assertIsNone(lock_or_flock_future.result())
                elif lock_timeout == "NON_BLOCKING":
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError
            elif release == "after_timeout":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT * 2))
                lock_or_flock.release()
                if lock_timeout == "BLOCKING":
                    self.assertIsNone(lock_or_flock_future.result())
                elif lock_timeout in ["NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError

        flock_fpath.unlink(missing_ok=True)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1)
    @parametrize(
        "impl_typename_combos",
        list(combinations(TestMixin.impl_typenames, 1))
        + list(combinations(TestMixin.impl_typenames, 2)),
    )
    def test_acquire_many_impl_locks_with_timeout(
        self,
        impl_typename_combos: tuple[str, ...],
    ) -> None:
        impls: list[impls._CacheImpl] = []
        for impl_typename in impl_typename_combos:
            impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
            impls.append(impl)

        with locks._acquire_many_impl_locks_with_timeout(*impls):
            for impl in impls:
                if hasattr(impl, "_lock"):
                    self.assertTrue(impl._lock.locked())
                elif hasattr(impl, "_flock"):
                    self.assertTrue(impl._flock.is_locked)

        for impl in impls:
            if hasattr(impl, "_lock"):
                self.assertFalse(impl._lock.locked())
            elif hasattr(impl, "_flock"):
                self.assertFalse(impl._flock.is_locked)


@instantiate_parametrized_tests
class UtilsTest(TestMixin, TestCase):
    def test_lru_cache(self) -> None:
        """Test that the LRU cache decorator works correctly with various argument types.

        Verifies that the _lru_cache decorator properly caches function results
        and handles different types of arguments including integers, floats, strings,
        and keyword arguments. Tests that cached calls return identical results
        to non-cached calls with proper argument preservation.
        """

        @utils._lru_cache
        def foo(*args, **kwargs):
            return args, kwargs

        self.assertEqual(
            foo(0),
            (
                (0,),
                {},
            ),
        )
        self.assertEqual(
            foo(0.0),
            (
                (0.0,),
                {},
            ),
        )
        self.assertEqual(
            foo("foo"),
            (
                ("foo",),
                {},
            ),
        )
        self.assertEqual(
            foo("foo", bar="bar"),
            (
                ("foo",),
                {"bar": "bar"},
            ),
        )


@instantiate_parametrized_tests
class InterfacesTest(TestMixin, TestCase):
    """Test class for Memoizer and PersistentMemoizer interfaces."""

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-interfaces-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    # ============= Memoizer Tests =============

    @set_caching_module_enabled(True)
    def test_memoizer_record_caches_result(self) -> None:
        """Test that Memoizer.record() caches function results.

        Verifies that when a function is decorated with record(), its result
        is cached and can be retrieved later.
        """
        # Setup: create a memoizer and a function that tracks call count
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice with the same argument
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called twice (record always executes)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(True)
    def test_memoizer_replay_retrieves_cached_result(self) -> None:
        """Test that Memoizer.replay() retrieves cached results without executing the function.

        Verifies that when a function is decorated with replay(), it retrieves
        results from cache without executing the original function.
        """
        # Setup: create a memoizer, record a result, then try to replay it
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: record a result first
        compute(5)
        self.assertEqual(call_count, 1)

        # Create a replay function for the same computation
        @memoizer.replay()
        def compute_replay(x: int) -> int:
            # This should never be called
            raise AssertionError("Function should not be executed during replay")

        # Assert: replay retrieves the cached result without calling the function
        result = compute_replay(5)
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)  # No additional calls

    @set_caching_module_enabled(True)
    def test_memoizer_replay_raises_on_cache_miss(self) -> None:
        """Test that Memoizer.replay() raises KeyError on cache miss.

        Verifies that when replay() is called with arguments that have no cached
        result, it raises a KeyError.
        """
        # Setup: create a memoizer with replay decorator
        memoizer = Memoizer()

        @memoizer.replay()
        def compute(x: int) -> int:
            return x * 2

        # Execute & Assert: replay raises KeyError for uncached arguments
        with self.assertRaises(KeyError):
            compute(5)

    @set_caching_module_enabled(True)
    def test_memoizer_memoize_caches_and_retrieves(self) -> None:
        """Test that Memoizer.memoize() caches on first call and retrieves on subsequent calls.

        Verifies that memoize() combines record and replay functionality:
        - First call executes the function and caches the result
        - Subsequent calls retrieve from cache without executing
        """
        # Setup: create a memoizer and a function that tracks call count
        memoizer = Memoizer()
        call_count = 0

        @memoizer.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice with the same argument
        result1 = compute(5)
        self.assertEqual(call_count, 1)  # Function called on first invocation

        result2 = compute(5)
        self.assertEqual(call_count, 1)  # Function not called on second invocation

        # Assert: both calls return the same result
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(False)
    def test_memoizer_record_disabled_returns_original_function(self) -> None:
        """Test that Memoizer.record() returns original function when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, record()
        returns the original function without any caching behavior.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(False)
    def test_memoizer_replay_disabled_always_raises(self) -> None:
        """Test that Memoizer.replay() always raises KeyError when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, replay()
        always raises KeyError regardless of what's in the cache.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()

        @memoizer.replay()
        def compute(x: int) -> int:
            return x * 2

        # Execute & Assert: replay always raises KeyError when disabled
        with self.assertRaises(KeyError) as cm:
            compute(5)
        self.assertIn("Caching is disabled", str(cm.exception))

    @set_caching_module_enabled(False)
    def test_memoizer_memoize_disabled_returns_original_function(self) -> None:
        """Test that Memoizer.memoize() returns original function when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, memoize()
        returns the original function without any caching behavior.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()
        call_count = 0

        @memoizer.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    # ============= PersistentMemoizer Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_record_caches_to_both(self) -> None:
        """Test that PersistentMemoizer.record() caches to both memory and disk.

        Verifies that when a function is decorated with record(), its result
        is cached in both the in-memory cache and the on-disk cache.
        """
        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function
        result = compute(5)

        # Assert: result is correct and cached in memory
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)

        # Verify memory cache has the result as CacheEntry
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        # Cache now stores CacheEntry with encoded_params and encoded_result
        cache_entry = memory_hit.value
        self.assertEqual(cache_entry.encoded_result, 10)
        self.assertEqual(cache_entry.encoded_params, {"args": (5,), "kwargs": {}})

        # Verify disk cache has the result (pickled)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_replay_checks_memory_then_disk(self) -> None:
        """Test that PersistentMemoizer.replay() checks memory first, then disk.

        Verifies that replay() uses a two-level cache strategy:
        1. Check memory cache first (fast)
        2. Fall back to disk cache on memory miss
        3. Populate memory cache from disk on disk hit
        """
        # Setup: create a persistent memoizer and store only to disk
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())

        # Store a value directly to disk cache only (as CacheEntry)
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        import pickle

        # Cache now stores CacheEntry with encoded_params and encoded_result
        cache_entry = interfaces.CacheEntry(
            encoded_params={"args": (5,), "kwargs": {}},
            encoded_result=10,
        )
        pickled_value = pickle.dumps(cache_entry)
        persistent._disk_cache.insert(cache_key, pickled_value)

        # Verify it's not in memory cache yet
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)

        # Create a replay function
        @persistent.replay()
        def compute(x: int) -> int:
            raise AssertionError("Function should not be executed during replay")

        # Execute: replay retrieves from disk and populates memory
        result = compute(5)

        # Assert: result is correct
        self.assertEqual(result, 10)

        # Verify memory cache was populated from disk
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        # Memory cache should now contain the CacheEntry
        cache_entry = memory_hit.value
        self.assertEqual(cache_entry.encoded_result, 10)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_memoize_two_level_caching(self) -> None:
        """Test that PersistentMemoizer.memoize() uses two-level caching.

        Verifies that memoize() combines two-level caching behavior:
        - First call executes and caches to both memory and disk
        - Second call (same process) retrieves from memory
        - After clearing memory, retrieves from disk
        """
        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: first call - cache miss, executes function
        result1 = compute(5)
        self.assertEqual(call_count, 1)
        self.assertEqual(result1, 10)

        # Second call - memory cache hit
        result2 = compute(5)
        self.assertEqual(call_count, 1)  # No additional execution
        self.assertEqual(result2, 10)

        # Clear memory cache to simulate a new process
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        persistent._memoizer._cache = impls._InMemoryCacheImpl()

        # Third call - memory miss, disk hit, populates memory
        result3 = compute(5)
        self.assertEqual(call_count, 1)  # Still no additional execution
        self.assertEqual(result3, 10)

        # Verify memory cache was repopulated
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_record_disabled(self) -> None:
        """Test that PersistentMemoizer.record() returns original function when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, record()
        returns the original function without any caching to memory or disk.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

        # Verify nothing was cached
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_replay_disabled(self) -> None:
        """Test that PersistentMemoizer.replay() always raises when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, replay()
        always raises KeyError.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())

        @persistent.replay()
        def compute(x: int) -> int:
            return x * 2

        # Execute & Assert: replay always raises KeyError when disabled
        with self.assertRaises(KeyError) as cm:
            compute(5)
        self.assertIn("Caching is disabled", str(cm.exception))

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_memoize_disabled(self) -> None:
        """Test that PersistentMemoizer.memoize() returns original function when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, memoize()
        returns the original function without any caching behavior.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.memoize()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    # ============= Cache Loading Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_loads_cache_from_dump_file(self) -> None:
        """Test that Memoizer loads cache entries from dump file on initialization.

        Verifies that when CACHE_DUMP_FILE_PATH is configured and the file exists,
        a new Memoizer instance pre-populates its in-memory cache with the dump contents.
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with cache entries
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 2,
                "collections": {
                    "null": {
                        "key1": {"params": {"x": 1}, "result": 10},
                        "key2": {"params": {"x": 2}, "result": 20},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            # Setup: Configure CACHE_DUMP_FILE_PATH
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a new Memoizer (should load from dump)
                memoizer = interfaces.Memoizer()

                # Assert: Cache was populated from dump file
                hit1 = memoizer._cache.get("key1")
                self.assertIsNotNone(hit1)
                cache_entry1 = hit1.value
                self.assertEqual(cache_entry1.encoded_result, 10)

                hit2 = memoizer._cache.get("key2")
                self.assertIsNotNone(hit2)
                cache_entry2 = hit2.value
                self.assertEqual(cache_entry2.encoded_result, 20)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_skips_loading_when_no_dump_file_configured(self) -> None:
        """Test that Memoizer skips loading when CACHE_DUMP_FILE_PATH is not set.

        Verifies that when no dump file path is configured, the Memoizer
        initializes with an empty cache without errors.
        """
        # Setup: Configure CACHE_DUMP_FILE_PATH to return None
        with patch.object(config, "CACHE_DUMP_FILE_PATH", return_value=None):
            # Execute: Create a new Memoizer
            memoizer = interfaces.Memoizer()

            # Assert: Cache is empty (not loaded from any file)
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_handles_missing_dump_file_gracefully(self) -> None:
        """Test that Memoizer handles missing dump file gracefully.

        Verifies that when CACHE_DUMP_FILE_PATH points to a non-existent file,
        the Memoizer initializes with an empty cache without crashing.
        """
        # Setup: Configure path to non-existent file
        non_existent_path = "/tmp/this_file_does_not_exist_12345.json"

        with patch.object(
            config, "CACHE_DUMP_FILE_PATH", return_value=non_existent_path
        ):
            # Execute: Create a new Memoizer
            memoizer = interfaces.Memoizer()

            # Assert: Cache is empty (no crash)
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_handles_corrupt_dump_file_gracefully(self) -> None:
        """Test that Memoizer handles corrupt dump file gracefully.

        Verifies that when the dump file contains invalid JSON, the Memoizer
        initializes with an empty cache without crashing.
        """
        import os
        import tempfile

        # Setup: Create a corrupt dump file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            tmp_file.write("{ this is not valid json")

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a new Memoizer
                memoizer = interfaces.Memoizer()

                # Assert: Cache is empty (no crash, handled gracefully)
                self.assertEqual(len(memoizer._cache._memory), 0)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_loads_from_sub_key(self) -> None:
        """Test that PersistentMemoizer loads cache from sub_dir nested structure.

        Verifies that when sub_dir is set, the PersistentMemoizer loads entries
        from the nested collections[sub_dir] structure.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        # Setup: Create a dump file with nested structure
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 2,
                "collections": {
                    "test_subdir": {
                        "nested_key1": {"params": {"x": 1}, "result": 100},
                        "nested_key2": {"params": {"x": 2}, "result": 200},
                    },
                    "other_subdir": {
                        "other_key": {"params": {"x": 3}, "result": 300},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create PersistentMemoizer with sub_dir="test_subdir"
                pm = interfaces.PersistentMemoizer(sub_dir=Path("test_subdir"))

                # Assert: Cache loaded entries from test_subdir only
                hit1 = pm._memoizer._cache.get("nested_key1")
                self.assertIsNotNone(hit1)
                cache_entry1 = hit1.value
                self.assertEqual(cache_entry1.encoded_result, 100)

                hit2 = pm._memoizer._cache.get("nested_key2")
                self.assertIsNotNone(hit2)
                cache_entry2 = hit2.value
                self.assertEqual(cache_entry2.encoded_result, 200)

                # Assert: Did not load entries from other_subdir
                hit_other = pm._memoizer._cache.get("other_key")
                self.assertIsNone(hit_other)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_loads_from_root_when_sub_dir_empty(self) -> None:
        """Test that PersistentMemoizer loads from root when sub_dir is empty.

        Verifies that when sub_dir is empty string, entries are loaded from
        collections["null"] (the root collection).
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with mixed root and nested entries
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            # Note: "null" is the JSON representation of Python None
            # When sub_dir is empty string (""), the Memoizer's sub_key is None
            dump_data = {
                "cache_size": 3,
                "collections": {
                    "null": {
                        "root_key1": {"params": {"x": 1}, "result": 10},
                        "root_key2": {"params": {"x": 2}, "result": 20},
                    },
                    "some_subdir": {
                        "nested_key": {"params": {"x": 3}, "result": 30},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create PersistentMemoizer with empty sub_dir
                pm = interfaces.PersistentMemoizer(sub_dir="")

                # Assert: Loaded root-level entries
                hit1 = pm._memoizer._cache.get("root_key1")
                self.assertIsNotNone(hit1)
                cache_entry1 = hit1.value
                self.assertEqual(cache_entry1.encoded_result, 10)

                hit2 = pm._memoizer._cache.get("root_key2")
                self.assertIsNotNone(hit2)
                cache_entry2 = hit2.value
                self.assertEqual(cache_entry2.encoded_result, 20)

                # Assert: Did not load nested entries
                hit_nested = pm._memoizer._cache.get("nested_key")
                self.assertIsNone(hit_nested)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_replay_uses_preloaded_cache(self) -> None:
        """Test that memoizer replay successfully retrieves from preloaded cache.

        Verifies end-to-end workflow: load cache from dump file, then use
        replay to retrieve cached results without executing the function.
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with a cached result
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            # Simulate a cache entry for compute(5) -> 10
            cache_key = interfaces._BaseMemoizer._make_key(None, 5)
            # Note: "null" is the JSON representation of Python None (no sub_key)
            dump_data = {
                "cache_size": 1,
                "collections": {
                    "null": {
                        cache_key: {
                            "params": {"args": (5,), "kwargs": {}},
                            "result": 10,
                        },
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a memoizer (loads cache from dump)
                memoizer = interfaces.Memoizer()

                # Create a replay function
                @memoizer.replay()
                def compute(x: int) -> int:
                    raise AssertionError(
                        "Function should not be executed during replay"
                    )

                # Execute: Call replay (should use preloaded cache)
                result = compute(5)

                # Assert: Got cached result without executing function
                self.assertEqual(result, 10)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    # ============= Memoizer._dump_to_disk Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_creates_json_file(self) -> None:
        """Test that _dump_to_disk creates a JSON file with cached entries.

        Verifies that when _dump_to_disk is called, it creates a JSON file
        containing the cached entries in a human-readable format.
        """
        # Setup: create a memoizer and cache some values
        memoizer = interfaces.Memoizer()

        @memoizer.record()
        def compute(x: int) -> int:
            return x * 2

        compute(5)
        compute(10)

        # Execute: dump the cache to disk
        memoizer._dump_to_disk()

        # Assert: JSON file was created with correct structure
        self.assertTrue(memoizer._shared_cache_filepath.exists())

        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("collections", data)
        self.assertIn("cache_size", data)
        self.assertEqual(data["cache_size"], 2)

        # Note: "null" is the JSON representation of Python None (no sub_key)
        self.assertIn("null", data["collections"])

        # Verify entries have correct format
        for entry in data["collections"]["null"].values():
            self.assertIn("params", entry)
            self.assertIn("result", entry)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_with_sub_key(self) -> None:
        """Test that _dump_to_disk uses sub_key for nested structure.

        Verifies that when a Memoizer is initialized with a sub_key,
        the cache entries are stored under collections[sub_key].
        """
        # Setup: create a memoizer with sub_key and cache a value
        sub_key = "test_sub_key"
        memoizer = interfaces.Memoizer(sub_key=sub_key)

        @memoizer.record()
        def compute(x: int) -> int:
            return x * 2

        compute(5)

        # Execute: dump the cache to disk
        memoizer._dump_to_disk()

        # Assert: entries are stored under the sub_key
        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("collections", data)
        self.assertIn(sub_key, data["collections"])

        # The sub_key should contain the cache entries
        sub_entries = data["collections"][sub_key]
        self.assertEqual(len(sub_entries), 1)

        # Verify entry format
        for entry in sub_entries.values():
            self.assertIn("params", entry)
            self.assertIn("result", entry)
            self.assertEqual(entry["result"], 10)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_merges_with_existing(self) -> None:
        """Test that _dump_to_disk merges with existing cache data.

        Verifies that when multiple Memoizer instances dump to the same file,
        their entries are merged additively.
        """
        # Setup: create first memoizer and cache a value
        memoizer1 = interfaces.Memoizer()

        @memoizer1.record()
        def compute1(x: int) -> int:
            return x * 2

        compute1(5)
        memoizer1._dump_to_disk()

        # Create second memoizer and cache a different value
        memoizer2 = interfaces.Memoizer()

        @memoizer2.record()
        def compute2(x: int) -> int:
            return x * 3

        compute2(10)

        # Execute: dump second memoizer to disk
        memoizer2._dump_to_disk()

        # Assert: both entries are in the file
        with open(memoizer1._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertEqual(data["cache_size"], 2)
        # Note: "null" is the JSON representation of Python None (no sub_key)
        self.assertEqual(len(data["collections"]["null"]), 2)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_skips_empty_cache(self) -> None:
        """Test that _dump_to_disk does nothing when cache is empty.

        Verifies that when _dump_to_disk is called on an empty cache,
        no file is created.
        """
        # Setup: create a memoizer with no cached values
        memoizer = interfaces.Memoizer()

        # Ensure the file doesn't exist beforehand
        if memoizer._shared_cache_filepath.exists():
            memoizer._shared_cache_filepath.unlink()

        # Execute: dump the empty cache
        memoizer._dump_to_disk()

        # Assert: no file was created
        self.assertFalse(memoizer._shared_cache_filepath.exists())

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_handles_corrupt_file(self) -> None:
        """Test that _dump_to_disk handles corrupt JSON files gracefully.

        Verifies that when the existing cache file contains invalid JSON,
        _dump_to_disk starts fresh and overwrites the corrupt file.
        """
        # Setup: create a memoizer and cache a value
        memoizer = interfaces.Memoizer()

        @memoizer.record()
        def compute(x: int) -> int:
            return x * 2

        compute(5)

        # Create a corrupt JSON file
        memoizer._shared_cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(memoizer._shared_cache_filepath, "w") as f:
            f.write("{ invalid json content")

        # Execute: dump the cache (should handle corrupt file)
        memoizer._dump_to_disk()

        # Assert: file now contains valid JSON with our entry
        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("collections", data)
        self.assertEqual(data["cache_size"], 1)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_stores_encoded_params_and_result(self) -> None:
        """Test that _dump_to_disk stores both encoded params and result.

        Verifies that the dumped JSON contains both the encoded parameters
        and the encoded result for each cache entry, making it useful for debugging.
        """
        # Setup: create a memoizer with custom encoder and cache a value
        memoizer = interfaces.Memoizer()

        @memoizer.record()
        def compute(x: int, y: int) -> int:
            return x + y

        compute(5, 10)

        # Execute: dump the cache to disk
        memoizer._dump_to_disk()

        # Assert: entry contains both params and result
        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        # Get the single entry (under "null" since no sub_key was set)
        entries = data["collections"]["null"]
        self.assertEqual(len(entries), 1)

        entry = next(iter(entries.values()))
        self.assertEqual(entry["result"], 15)
        self.assertEqual(entry["params"]["args"], [5, 10])
        self.assertEqual(entry["params"]["kwargs"], {})

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_multiple_sub_keys(self) -> None:
        """Test that multiple Memoizers with different sub_keys coexist.

        Verifies that Memoizers with different sub_keys store their entries
        under separate namespaces in the same JSON file.
        """
        # Setup: create two memoizers with different sub_keys
        memoizer1 = interfaces.Memoizer(sub_key="feature_a")
        memoizer2 = interfaces.Memoizer(sub_key="feature_b")

        @memoizer1.record()
        def compute_a(x: int) -> int:
            return x * 2

        @memoizer2.record()
        def compute_b(x: int) -> int:
            return x * 3

        compute_a(5)
        compute_b(10)

        # Execute: dump both caches
        memoizer1._dump_to_disk()
        memoizer2._dump_to_disk()

        # Assert: both sub_keys exist with their respective entries
        with open(memoizer1._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("feature_a", data["collections"])
        self.assertIn("feature_b", data["collections"])

        # Verify each sub_key has one entry with correct result
        feature_a_entries = data["collections"]["feature_a"]
        feature_b_entries = data["collections"]["feature_b"]

        self.assertEqual(len(feature_a_entries), 1)
        self.assertEqual(len(feature_b_entries), 1)

        self.assertEqual(next(iter(feature_a_entries.values()))["result"], 10)
        self.assertEqual(next(iter(feature_b_entries.values()))["result"], 30)


@instantiate_parametrized_tests
class ForceDisableCachesTest(TestMixin, TestCase):
    """Test class for force_disable_caches integration with the caching module."""

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-force-disable-caches-{cls.cls_id}"

    def test_force_disable_caches_disables_caching_module(self) -> None:
        """Test that force_disable_caches=True disables the caching module.

        Verifies that when torch._inductor.config.force_disable_caches is True,
        IS_CACHING_MODULE_ENABLED() returns False even if the base config
        would otherwise enable caching.
        """
        # Setup: patch the base config to return True (enabled)
        # and force_disable_caches to return True
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=True),
            patch.object(config, "_is_force_disable_caches", return_value=True),
        ):
            # Execute & Assert: IS_CACHING_MODULE_ENABLED should return False
            self.assertFalse(config.IS_CACHING_MODULE_ENABLED())

    def test_caching_module_enabled_when_force_disable_is_false(self) -> None:
        """Test that caching works when force_disable_caches=False.

        Verifies that when force_disable_caches is False and the base config
        enables caching, IS_CACHING_MODULE_ENABLED() returns True.
        """
        # Setup: patch both configs to enabled states
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=True),
            patch.object(config, "_is_force_disable_caches", return_value=False),
        ):
            # Execute & Assert: IS_CACHING_MODULE_ENABLED should return True
            self.assertTrue(config.IS_CACHING_MODULE_ENABLED())

    def test_caching_module_disabled_when_base_config_is_false(self) -> None:
        """Test that caching is disabled when base config is False.

        Verifies that when the base versioned config returns False,
        IS_CACHING_MODULE_ENABLED() returns False regardless of force_disable_caches.
        """
        # Setup: patch base config to False, force_disable to False
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=False),
            patch.object(config, "_is_force_disable_caches", return_value=False),
        ):
            # Execute & Assert: IS_CACHING_MODULE_ENABLED should return False
            self.assertFalse(config.IS_CACHING_MODULE_ENABLED())

    @patch_on_disk_cache_base_dir
    def test_memoizer_disabled_when_force_disable_caches_true(self) -> None:
        """Test that Memoizer operations become no-ops when force_disable_caches=True.

        Verifies that when force_disable_caches is enabled, memoized functions
        always execute (no caching) and replay always raises KeyError.
        """
        # Setup: enable force_disable_caches
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=True),
            patch.object(config, "_is_force_disable_caches", return_value=True),
        ):
            memoizer = Memoizer()
            call_count = 0

            @memoizer.memoize()
            def compute(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 2

            # Execute: call twice - should execute both times (no caching)
            result1 = compute(5)
            result2 = compute(5)

            # Assert: function was called both times
            self.assertEqual(call_count, 2)
            self.assertEqual(result1, 10)
            self.assertEqual(result2, 10)


@instantiate_parametrized_tests
class FreshCacheIntegrationTest(TestMixin, TestCase):
    """Test class for fresh_cache integration with the caching module."""

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-fresh-cache-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @set_caching_module_enabled(True)
    def test_memoizer_cache_clear_clears_in_memory_cache(self) -> None:
        """Test that Memoizer.cache_clear() clears the in-memory cache.

        Verifies that calling cache_clear() on a Memoizer instance resets
        its in-memory cache to empty.
        """
        # Setup: create a memoizer and cache some values
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        compute(10)

        # Verify cache has entries
        self.assertEqual(len(memoizer._cache._memory), 2)
        self.assertEqual(call_count, 2)

        # Execute: clear the cache
        memoizer.cache_clear()

        # Assert: cache is now empty
        self.assertEqual(len(memoizer._cache._memory), 0)

        # Verify that calling the memoized function again would be a cache miss
        # by checking that replay raises KeyError
        @memoizer.replay()
        def compute_replay(x: int) -> int:
            raise AssertionError("Should not be called")

        with self.assertRaises(KeyError):
            compute_replay(5)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_cache_clear_clears_disk_cache(self) -> None:
        """Test that PersistentMemoizer.cache_clear() clears the on-disk cache.

        Verifies that calling cache_clear() on a PersistentMemoizer removes
        the on-disk cache directory.
        """
        from pathlib import Path

        # Setup: create a persistent memoizer and cache some values
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        self.assertEqual(call_count, 1)

        # Verify disk cache directory exists and has content
        disk_cache_dir = persistent._disk_cache._cache_dir
        self.assertTrue(disk_cache_dir.exists())

        # Execute: clear the persistent memoizer cache
        persistent.cache_clear()

        # Assert: disk cache directory is removed
        self.assertFalse(disk_cache_dir.exists())

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_registered_with_clear_on_fresh_cache(self) -> None:
        """Test that Memoizer is registered with clear_on_fresh_cache.

        Verifies that when a Memoizer is created, it is automatically
        registered to be cleared when fresh_cache() is invoked.
        """
        from torch._inductor.utils import _registered_caches

        # Setup: create a memoizer
        memoizer = Memoizer()

        # Assert: memoizer is in the registered caches list
        self.assertIn(memoizer, _registered_caches)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_registered_with_clear_on_fresh_cache(self) -> None:
        """Test that PersistentMemoizer is registered with clear_on_fresh_cache.

        Verifies that when a PersistentMemoizer is created, it is automatically
        registered to be cleared when fresh_cache() is invoked.
        """
        from pathlib import Path

        from torch._inductor.utils import _registered_caches

        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))

        # Assert: persistent memoizer is in the registered caches list
        self.assertIn(persistent, _registered_caches)
        # Also verify the underlying memoizer is registered
        self.assertIn(persistent._memoizer, _registered_caches)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_fresh_cache_clears_memoizer(self) -> None:
        """Test that fresh_cache() context manager clears Memoizer caches.

        Verifies that when entering fresh_cache() context, all registered
        Memoizer instances have their caches cleared.
        """
        from torch._inductor.utils import fresh_cache

        # Setup: create a memoizer and cache some values
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        self.assertEqual(len(memoizer._cache._memory), 1)

        # Execute: enter fresh_cache context (this calls clear_caches())
        with fresh_cache():
            # Assert: memoizer cache was cleared
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_fresh_cache_clears_persistent_memoizer_disk_cache(self) -> None:
        """Test that fresh_cache() clears PersistentMemoizer disk cache.

        Verifies that when entering fresh_cache() context, PersistentMemoizer
        instances have their on-disk cache directories removed.
        """
        from pathlib import Path

        from torch._inductor.utils import fresh_cache

        # Setup: create a persistent memoizer and cache some values
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)

        # Verify disk cache exists
        disk_cache_dir = persistent._disk_cache._cache_dir
        self.assertTrue(disk_cache_dir.exists())

        # Execute: enter fresh_cache context
        with fresh_cache():
            # Assert: disk cache directory was removed
            self.assertFalse(disk_cache_dir.exists())
            # And memory cache was cleared (via the underlying Memoizer)
            self.assertEqual(len(persistent._memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_cache_clear_is_idempotent(self) -> None:
        """Test that calling cache_clear() multiple times is safe.

        Verifies that clearing an already-empty cache does not raise errors.
        """
        from pathlib import Path

        # Setup: create memoizers
        memoizer = Memoizer()
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))

        # Execute: call cache_clear multiple times on empty caches
        memoizer.cache_clear()
        memoizer.cache_clear()

        persistent.cache_clear()
        persistent.cache_clear()

        # Assert: no errors raised, caches are still empty
        self.assertEqual(len(memoizer._cache._memory), 0)
        self.assertFalse(persistent._disk_cache._cache_dir.exists())


@instantiate_parametrized_tests
class ShouldPadMemoizerTest(TestMixin, TestCase):
    """Test class for _should_pad memoizer integration.

    These tests verify that the PersistentMemoizer applied to _should_pad
    correctly memoizes and replays results based on tensor metadata and
    operation parameters.
    """

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-should-pad-memoizer-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    def _create_mock_match(self) -> Any:
        """Create a mock Match object for testing.

        Returns a mock that simulates the Match object from pattern_matcher,
        providing the minimal interface needed for the should_pad_params_encoder.
        """
        from unittest.mock import MagicMock

        mock_match = MagicMock()

        # Create mock FX nodes for mat1 and mat2 kwargs
        mock_mat1_node = MagicMock()
        mock_mat1_node.op = "placeholder"

        mock_mat2_node = MagicMock()
        mock_mat2_node.op = "placeholder"

        mock_match.kwargs = {
            "mat1": mock_mat1_node,
            "mat2": mock_mat2_node,
        }

        return mock_match

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_caches_result(self) -> None:
        """Test that the should_pad_memoizer caches function results.

        Verifies that when a function decorated with should_pad_memoizer.memoize
        is called, the result is cached and can be retrieved on subsequent calls.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        # Create test inputs
        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: call the function twice with the same parameters
        result1 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)

        result2 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)  # Should use cached result

        # Assert: both calls return the same result
        self.assertTrue(result1)
        self.assertTrue(result2)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_different_shapes_different_cache_entries(
        self,
    ) -> None:
        """Test that different tensor shapes result in different cache entries.

        Verifies that the encoder correctly distinguishes between tensors with
        different shapes, ensuring they don't share cached results.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            # Return different values based on shape to verify no cross-caching
            return mat1.shape[0] > 10

        mock_match = self._create_mock_match()

        # Execute: call with different shapes
        mat1_small = torch.randn(8, 16, dtype=torch.float32)
        mat2_small = torch.randn(16, 32, dtype=torch.float32)

        mat1_large = torch.randn(12, 16, dtype=torch.float32)
        mat2_large = torch.randn(16, 32, dtype=torch.float32)

        op = torch.ops.aten.mm

        result_small = mock_should_pad(mock_match, mat1_small, mat2_small, op)
        self.assertEqual(call_count, 1)

        result_large = mock_should_pad(mock_match, mat1_large, mat2_large, op)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different (based on shape)
        self.assertFalse(result_small)  # 8 <= 10
        self.assertTrue(result_large)  # 12 > 10

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_different_dtypes_different_cache_entries(
        self,
    ) -> None:
        """Test that different tensor dtypes result in different cache entries.

        Verifies that the encoder correctly distinguishes between tensors with
        different dtypes, ensuring they don't share cached results.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            return mat1.dtype == torch.float32

        mock_match = self._create_mock_match()
        op = torch.ops.aten.mm

        # Execute: call with different dtypes but same shapes
        mat1_fp32 = torch.randn(8, 16, dtype=torch.float32)
        mat2_fp32 = torch.randn(16, 32, dtype=torch.float32)

        mat1_fp16 = torch.randn(8, 16, dtype=torch.float16)
        mat2_fp16 = torch.randn(16, 32, dtype=torch.float16)

        result_fp32 = mock_should_pad(mock_match, mat1_fp32, mat2_fp32, op)
        self.assertEqual(call_count, 1)

        result_fp16 = mock_should_pad(mock_match, mat1_fp16, mat2_fp16, op)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different (based on dtype)
        self.assertTrue(result_fp32)
        self.assertFalse(result_fp16)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_different_ops_different_cache_entries(
        self,
    ) -> None:
        """Test that different operations result in different cache entries.

        Verifies that the encoder correctly distinguishes between different
        operation types (mm vs bmm), ensuring they don't share cached results.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0
        call_ops: list[Any] = []

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            call_ops.append(op)
            return op is torch.ops.aten.mm

        mock_match = self._create_mock_match()

        # Create 2D tensors for mm
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)

        # Execute: call with different operations
        result_mm = mock_should_pad(mock_match, mat1, mat2, torch.ops.aten.mm)
        self.assertEqual(call_count, 1)

        result_addmm = mock_should_pad(mock_match, mat1, mat2, torch.ops.aten.addmm)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different (based on op)
        self.assertTrue(result_mm)
        self.assertFalse(result_addmm)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_replays_from_disk_cache(self) -> None:
        """Test that the memoizer replays results from disk cache after memory clear.

        Verifies that PersistentMemoizer correctly stores results to disk and
        can replay them after the in-memory cache is cleared.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: cache a result
        result1 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)
        self.assertTrue(result1)

        # Clear the in-memory cache to simulate a new process
        test_memoizer._memoizer._cache = impls._InMemoryCacheImpl()

        # Execute: call again - should replay from disk
        result2 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)  # Function should NOT be called again
        self.assertTrue(result2)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_should_pad_memoizer_disabled_does_not_cache(self) -> None:
        """Test that the memoizer does not cache when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, the function
        is called every time without caching.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: call the function twice
        result1 = mock_should_pad(mock_match, mat1, mat2, op)
        result2 = mock_should_pad(mock_match, mat1, mat2, op)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertTrue(result1)
        self.assertTrue(result2)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_with_input_tensor(self) -> None:
        """Test that the memoizer correctly handles the optional input tensor.

        Verifies that different input tensors (for addmm) result in different
        cache entries, and that None vs non-None input are distinguished.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            return input is not None

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        input_tensor = torch.randn(32, dtype=torch.float32)
        op = torch.ops.aten.addmm

        # Execute: call with and without input tensor
        result_with_input = mock_should_pad(
            mock_match, mat1, mat2, op, input=input_tensor
        )
        self.assertEqual(call_count, 1)

        result_without_input = mock_should_pad(mock_match, mat1, mat2, op, input=None)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different
        self.assertTrue(result_with_input)
        self.assertFalse(result_without_input)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_params_encoder_produces_consistent_keys(self) -> None:
        """Test that the encoder produces consistent keys for the same inputs.

        Verifies that calling the encoder with the same tensor metadata produces
        the same cache key, ensuring reliable cache hits.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: encode the same parameters multiple times
        encoded1 = encoders.should_pad_params_encoder(mock_match, mat1, mat2, op)
        encoded2 = encoders.should_pad_params_encoder(mock_match, mat1, mat2, op)

        # Assert: encodings are identical
        self.assertEqual(encoded1, encoded2)

        # Also verify the structure of the encoded output
        self.assertIn("mat1", encoded1)
        self.assertIn("mat2", encoded1)
        self.assertIn("op", encoded1)
        self.assertEqual(encoded1["mat1"]["shape"], tuple(mat1.shape))
        self.assertEqual(encoded1["mat2"]["shape"], tuple(mat2.shape))

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_same_shape_different_data_uses_cache(self) -> None:
        """Test that tensors with the same metadata but different data share cache.

        Verifies that the memoizer caches based on tensor metadata (shape, stride,
        dtype) and not on the actual tensor data values.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        mock_match = self._create_mock_match()
        op = torch.ops.aten.mm

        # Execute: call with different tensors that have the same metadata
        mat1_a = torch.randn(8, 16, dtype=torch.float32)
        mat2_a = torch.randn(16, 32, dtype=torch.float32)

        mat1_b = torch.randn(8, 16, dtype=torch.float32)  # Different data, same shape
        mat2_b = torch.randn(16, 32, dtype=torch.float32)  # Different data, same shape

        result1 = mock_should_pad(mock_match, mat1_a, mat2_a, op)
        self.assertEqual(call_count, 1)

        result2 = mock_should_pad(mock_match, mat1_b, mat2_b, op)
        self.assertEqual(call_count, 1)  # Should use cached result

        # Assert: both return the same cached result
        self.assertTrue(result1)
        self.assertTrue(result2)


@instantiate_parametrized_tests
class DeferredRecordingTest(TestMixin, TestCase):
    """Test class for DeferredRecording functionality."""

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-deferred-recording-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    # ============= DeferredRecording Basic Tests =============

    @set_caching_module_enabled(True)
    def test_deferred_recording_finalize_caches_result(self) -> None:
        """Test that DeferredRecording.finalize() caches the result.

        Verifies that when an encoder returns a DeferredRecording and
        finalize() is called later, the result is cached properly.
        """
        # Setup: create a memoizer and a custom encoder that returns DeferredRecording
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        call_count = 0

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function (should NOT cache immediately)
        result = compute(5)
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)

        # Verify: cache should be empty since we returned DeferredRecording
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNone(cache_hit)

        # Execute: finalize the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)  # Cache encoded value 100

        # Assert: cache should now have the result
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNotNone(cache_hit)
        cache_entry = cache_hit.value
        self.assertEqual(cache_entry.encoded_result, 100)

    @set_caching_module_enabled(True)
    def test_deferred_recording_replay_after_complete(self) -> None:
        """Test that replay works after deferred recording completes.

        Verifies end-to-end flow: record with deferred -> complete -> replay.
        """
        # Setup: create a memoizer with deferred recording
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute_record(x: int) -> int:
            return x * 2

        @memoizer.replay()
        def compute_replay(x: int) -> int:
            raise AssertionError("Should not be called during replay")

        # Execute: record with deferred
        compute_record(5)

        # Verify: replay should fail before complete
        with self.assertRaises(KeyError):
            compute_replay(5)

        # Complete the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(10)

        # Assert: replay should now work
        result = compute_replay(5)
        self.assertEqual(result, 10)

    @set_caching_module_enabled(True)
    def test_deferred_recording_with_future_pattern(self) -> None:
        """Test DeferredRecording with a Future-like pattern.

        Simulates the real use case where a Future callback completes
        the deferred recording.
        """
        # Setup: create a memoizer and simulate a Future
        memoizer = Memoizer()
        cache_populated = Event()

        def future_encoder_factory(fn) -> object:
            def future_encoder(*args: object, **kwargs: object) -> object:
                def encode(future_result: Future[int]) -> interfaces.DeferredRecording:
                    deferred = interfaces.DeferredRecording()

                    def on_complete(completed_future: Future[int]) -> None:
                        actual_result = completed_future.result()
                        deferred.finalize(actual_result)

                    future_result.add_done_callback(on_complete)
                    return deferred

                return encode

            return future_encoder

        with ThreadPoolExecutor() as executor:

            @memoizer.record(custom_result_encoder=future_encoder_factory)
            def compute_async(x: int) -> Future[int]:
                return executor.submit(lambda: x * 2)

            # Execute: call the function and get a future
            future = compute_async(5)

            # Register our own callback on the DeferredRecording to know when
            # the cache is populated. This must be done AFTER compute_async()
            # returns, so our callback is registered after the memoizer's callback.
            # This ensures our callback runs after the cache is populated.
            cache_key = interfaces._BaseMemoizer._make_key(None, 5)
            pending = memoizer._pending_deferred.get(cache_key)
            if pending is not None:
                # Register callback - will be called after memoizer's callback
                # (either when complete() runs, or immediately if already completed)
                pending.register_callback(lambda _: cache_populated.set())
            else:
                # Deferred already completed and removed from tracking.
                # This means complete() and callbacks already ran, cache is populated.
                cache_populated.set()

            # Wait for the future to complete
            result = future.result(timeout=5)
            self.assertEqual(result, 10)

            # Wait for the cache to be populated (with timeout)
            self.assertTrue(
                cache_populated.wait(timeout=5),
                "Cache was not populated within timeout",
            )

        # Assert: cache should have the result after future completed
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNotNone(cache_hit)
        cache_entry = cache_hit.value
        self.assertEqual(cache_entry.encoded_result, 10)

    @set_caching_module_enabled(True)
    def test_deferred_recording_preserves_original_return(self) -> None:
        """Test that deferred recording returns the original result.

        Verifies that even when using deferred recording, the function
        returns the original (non-encoded) result immediately.
        """
        # Setup
        memoizer = Memoizer()

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    return interfaces.DeferredRecording()

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Execute: call the function
        result = compute(5)

        # Assert: should return the original result (not the DeferredRecording)
        self.assertEqual(result, 10)
        self.assertIsInstance(result, int)

    @set_caching_module_enabled(True)
    def test_deferred_recording_finalize_without_callback(self) -> None:
        """Test that finalize() is safe when no callback is set.

        Verifies that calling finalize() on a DeferredRecording without
        a registered callback does not raise an error.
        """
        # Setup: create a standalone DeferredRecording
        deferred = interfaces.DeferredRecording()

        # Execute & Assert: finalize should not raise
        deferred.finalize(42)  # Should do nothing but not raise

    # ============= PersistentMemoizer DeferredRecording Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_deferred_recording_caches_to_both(self) -> None:
        """Test that PersistentMemoizer deferred recording caches to both memory and disk.

        Verifies that when complete() is called, the result is stored
        in both the in-memory cache and the on-disk cache.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @persistent.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Execute: call the function (should NOT cache immediately)
        result = compute(5)
        self.assertEqual(result, 10)

        # Verify: both caches should be empty
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNone(disk_hit)

        # Complete the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)

        # Assert: memory cache should have the result
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        cache_entry = memory_hit.value
        self.assertEqual(cache_entry.encoded_result, 100)

        # Assert: disk cache should also have the result
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_deferred_replays_from_disk(self) -> None:
        """Test that deferred recording can be replayed from disk cache.

        Verifies that after completing a deferred recording, the result
        can be replayed even after clearing the memory cache.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @persistent.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Record with deferred
        compute(5)
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)

        # Clear memory cache to simulate new process
        persistent._memoizer._cache = impls._InMemoryCacheImpl()

        # Create replay function
        @persistent.replay()
        def compute_replay(x: int) -> int:
            raise AssertionError("Should not be called during replay")

        # Execute: replay should work from disk
        result = compute_replay(5)

        # Assert
        self.assertEqual(result, 100)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_memoize_with_deferred(self) -> None:
        """Test that memoize() works correctly with deferred recording.

        Verifies that when using memoize() with a deferred encoder,
        subsequent calls work correctly after the deferred completes.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        deferred_objects: list[interfaces.DeferredRecording] = []
        call_count = 0

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    deferred = interfaces.DeferredRecording()
                    deferred_objects.append(deferred)
                    return deferred

                return encode

            return deferred_encoder

        @persistent.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: should execute function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call before complete: should execute again (cache miss)
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 2)

        # Complete the first deferred recording
        deferred_objects[0].finalize(10)

        # Third call after complete: should use cache
        result3 = compute(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 2)  # No additional call

    # ============= Edge Cases =============

    @set_caching_module_enabled(True)
    def test_deferred_recording_with_none_result(self) -> None:
        """Test that deferred recording works with None as the result.

        Verifies that complete(None) caches None correctly.
        """
        # Setup
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> None:
            return None

        # Execute
        compute(5)
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(None)

        # Assert: cache should have None as the result
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNotNone(cache_hit)
        cache_entry = cache_hit.value
        self.assertIsNone(cache_entry.encoded_result)

    @set_caching_module_enabled(True)
    def test_deferred_recording_multiple_calls_different_keys(self) -> None:
        """Test that multiple deferred recordings with different keys work.

        Verifies that deferred recordings for different parameters are
        independent and can be completed in any order.
        """
        # Setup
        memoizer = Memoizer()
        deferred_objects: dict[int, interfaces.DeferredRecording] = {}

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                x = args[0]

                def encode(result: object) -> interfaces.DeferredRecording:
                    deferred = interfaces.DeferredRecording()
                    deferred_objects[x] = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Record multiple calls
        compute(5)
        compute(10)
        compute(15)

        # Complete in reverse order
        deferred_objects[15].finalize(30)
        deferred_objects[5].finalize(10)
        deferred_objects[10].finalize(20)

        # Assert: all caches should be correct
        for x, expected in [(5, 10), (10, 20), (15, 30)]:
            cache_key = interfaces._BaseMemoizer._make_key(None, x)
            cache_hit = memoizer._cache.get(cache_key)
            self.assertIsNotNone(cache_hit)
            self.assertEqual(cache_hit.value.encoded_result, expected)

    # ============= Result Encoder/Decoder Factory Tests =============

    @set_caching_module_enabled(True)
    def test_encoder_factory_receives_underlying_function(self) -> None:
        """Test that the encoder factory receives the underlying function.

        Verifies that when custom_result_encoder is provided, the factory
        receives the unwrapped function as the first parameter, allowing
        the encoder to call it if needed.
        """
        # Setup: create a memoizer and track what fn is passed to the factory
        memoizer = Memoizer()
        received_fn: list[object] = []

        def encoder_factory(fn: object) -> object:
            received_fn.append(fn)

            def params_to_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> object:
                    return result

                return encode

            return params_to_encoder

        @memoizer.record(custom_result_encoder=encoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Execute: call the memoized function
        result = compute(5)

        # Assert: encoder factory received a function (the unwrapped compute)
        self.assertEqual(len(received_fn), 1)
        self.assertTrue(callable(received_fn[0]))
        self.assertEqual(result, 10)

        # Verify that calling the received fn works correctly
        underlying_fn = received_fn[0]
        self.assertEqual(underlying_fn(7), 14)

    @set_caching_module_enabled(True)
    def test_encoder_factory_fn_bypasses_memoization(self) -> None:
        """Test that calling fn from the encoder bypasses memoization.

        Verifies that the fn passed to the encoder factory is the unwrapped
        function, so calling it does not trigger caching side effects.
        """
        # Setup: create a memoizer and count direct vs memoized calls
        memoizer = Memoizer()
        direct_call_count = 0
        received_fn: list[object] = []

        def encoder_factory(fn: object) -> object:
            received_fn.append(fn)

            def params_to_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> object:
                    return result

                return encode

            return params_to_encoder

        @memoizer.record(custom_result_encoder=encoder_factory)
        def compute(x: int) -> int:
            nonlocal direct_call_count
            direct_call_count += 1
            return x * 2

        # Execute: call the memoized function once
        compute(5)
        self.assertEqual(direct_call_count, 1)

        # Now call the underlying fn directly multiple times
        underlying_fn = received_fn[0]
        underlying_fn(5)
        underlying_fn(5)
        underlying_fn(5)

        # Assert: direct calls happened but no new cache entries
        self.assertEqual(direct_call_count, 4)  # 1 memoized + 3 direct
        # Cache should only have the one entry from the memoized call
        cache_entries = len(memoizer._cache._memory)
        self.assertEqual(cache_entries, 1)

    @set_caching_module_enabled(True)
    def test_decoder_factory_receives_underlying_function(self) -> None:
        """Test that the decoder factory receives the underlying function.

        Verifies that when custom_result_decoder is provided, the factory
        receives the unwrapped function as the first parameter, allowing
        the decoder to call it as a fallback if needed.
        """
        # Setup: create a memoizer and track what fn is passed to the factory
        memoizer = Memoizer()
        received_fn: list[object] = []

        # First, populate the cache with a value
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_entry = interfaces.CacheEntry(
            encoded_params={"args": (5,), "kwargs": {}},
            encoded_result={"encoded_value": 10},
        )
        memoizer._cache.insert(cache_key, cache_entry)

        def decoder_factory(fn: object) -> object:
            received_fn.append(fn)

            def params_to_decoder(*args: object, **kwargs: object) -> object:
                def decode(encoded_result: object) -> int:
                    return encoded_result["encoded_value"]

                return decode

            return params_to_decoder

        @memoizer.replay(custom_result_decoder=decoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Execute: replay the cached value
        result = compute(5)

        # Assert: decoder factory received a function (the unwrapped compute)
        self.assertEqual(len(received_fn), 1)
        self.assertTrue(callable(received_fn[0]))
        self.assertEqual(result, 10)

        # Verify that calling the received fn works correctly
        underlying_fn = received_fn[0]
        self.assertEqual(underlying_fn(7), 14)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_decoder_receives_fn(self) -> None:
        """Test that PersistentMemoizer decoder factory receives fn.

        Verifies that the two-level cache correctly passes the fn parameter
        to the decoder factory when replaying from disk cache.
        """
        import pickle

        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        received_fn: list[object] = []

        # Populate the disk cache with a value
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_entry = interfaces.CacheEntry(
            encoded_params={"args": (5,), "kwargs": {}},
            encoded_result={"encoded_value": 10},
        )
        pickled_entry = pickle.dumps(cache_entry)
        persistent._disk_cache.insert(cache_key, pickled_entry)

        def decoder_factory(fn: object) -> object:
            received_fn.append(fn)

            def params_to_decoder(*args: object, **kwargs: object) -> object:
                def decode(encoded_result: object) -> int:
                    return encoded_result["encoded_value"]

                return decode

            return params_to_decoder

        @persistent.replay(custom_result_decoder=decoder_factory)
        def compute(x: int) -> int:
            return x * 2

        # Execute: replay from disk cache
        result = compute(5)

        # Assert: decoder factory received the underlying function
        self.assertEqual(len(received_fn), 1)
        self.assertTrue(callable(received_fn[0]))
        self.assertEqual(result, 10)

        # Verify fn works correctly
        underlying_fn = received_fn[0]
        self.assertEqual(underlying_fn(7), 14)

    @set_caching_module_enabled(True)
    def test_deferred_recording_double_finalize_raises(self) -> None:
        """Test that calling finalize() twice raises RuntimeError.

        Verifies that the race condition prevention logic correctly
        rejects duplicate finalize() calls.
        """
        # Setup: create a DeferredRecording
        deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()

        # Execute: first finalize succeeds
        deferred.finalize(42)

        # Assert: second finalize raises RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            deferred.finalize(100)

        self.assertIn("finalize() called multiple times", str(ctx.exception))

    @set_caching_module_enabled(True)
    def test_deferred_recording_callback_after_finalize(self) -> None:
        """Test that callbacks registered after finalize() are invoked immediately.

        Verifies the race condition handling where finalize() completes before
        the memoizer registers its callback - the callback should still be invoked.
        """
        # Setup: create a DeferredRecording and finalize it first
        deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()
        deferred.finalize(42)

        # Execute: register a callback after finalize
        callback_invoked = False
        callback_value: int | None = None

        def callback(result: int) -> None:
            nonlocal callback_invoked, callback_value
            callback_invoked = True
            callback_value = result

        deferred.register_callback(callback)

        # Assert: callback was invoked immediately with the finalized result
        self.assertTrue(callback_invoked)
        self.assertEqual(callback_value, 42)

    @set_caching_module_enabled(True)
    def test_deferred_recording_multiple_callbacks_order(self) -> None:
        """Test that multiple callbacks are invoked in registration order.

        Verifies that when multiple callbacks are registered before finalize(),
        they are all invoked in the order they were registered.
        """
        # Setup: create a DeferredRecording and register multiple callbacks
        deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()
        invocation_order: list[int] = []

        def make_callback(idx: int) -> object:
            def callback(result: int) -> None:
                invocation_order.append(idx)

            return callback

        # Register callbacks in order 0, 1, 2
        deferred.register_callback(make_callback(0))
        deferred.register_callback(make_callback(1))
        deferred.register_callback(make_callback(2))

        # Execute: finalize the deferred recording
        deferred.finalize(100)

        # Assert: callbacks were invoked in registration order
        self.assertEqual(invocation_order, [0, 1, 2])

    @set_caching_module_enabled(True)
    def test_deferred_recording_thread_safety(self) -> None:
        """Test that concurrent finalize() and register_callback() are thread-safe.

        Verifies that when finalize() and register_callback() are called
        concurrently from different threads, no race conditions occur.
        """
        # Setup: create multiple deferred recordings to test concurrency
        num_iterations = 100
        errors: list[Exception] = []

        def run_concurrent_test() -> None:
            deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()
            callback_results: list[int] = []
            barrier = Event()

            def callback(result: int) -> None:
                callback_results.append(result)

            def finalize_thread() -> None:
                barrier.wait()
                try:
                    deferred.finalize(42)
                except RuntimeError:
                    pass

            def register_thread() -> None:
                barrier.wait()
                deferred.register_callback(callback)

            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(finalize_thread)
                f2 = executor.submit(register_thread)
                barrier.set()
                wait([f1, f2], timeout=5)

            if len(callback_results) != 1 or callback_results[0] != 42:
                errors.append(AssertionError(f"Expected [42], got {callback_results}"))

        # Execute: run many iterations to catch race conditions
        for _ in range(num_iterations):
            run_concurrent_test()

        # Assert: no errors occurred
        self.assertEqual(errors, [])

    # ============= Interim Result Tests =============

    @set_caching_module_enabled(True)
    def test_interim_result_prevents_function_reexecution(self) -> None:
        """Test that make_interim_result prevents function re-execution.

        Verifies that when a DeferredRecording has make_interim_result set,
        subsequent calls return the interim result without re-executing the function.
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        original_result: list[int] = []
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: int) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    original_result.append(result)
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: should execute function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call: should return interim result without re-execution
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # No additional call!

        # Third call: same behavior
        result3 = compute(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 1)  # Still no additional call

        # Finalize to mimic actual behavior
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(10)

    @set_caching_module_enabled(True)
    def test_interim_result_returns_same_object(self) -> None:
        """Test that make_interim_result returns the same object.

        Verifies that all callers get the exact same object (e.g., same Future),
        not copies.
        """
        # Setup
        memoizer = Memoizer()
        shared_object = {"value": "shared"}
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: object) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: shared_object
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> object:
            return shared_object

        # Multiple calls should return the same object
        result1 = compute(5)
        result2 = compute(5)
        result3 = compute(5)

        self.assertIs(result1, result2)
        self.assertIs(result2, result3)
        self.assertIs(result1, shared_object)

        # Finalize to mimic actual behavior
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(shared_object)

    @set_caching_module_enabled(True)
    def test_interim_result_with_future_pattern(self) -> None:
        """Test make_interim_result with a Future-like pattern.

        Verifies that multiple callers get the same Future object while
        the deferred recording is pending.
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        deferred_obj: interfaces.DeferredRecording | None = None

        def future_encoder_factory(fn) -> object:
            def future_encoder(*args: object, **kwargs: object) -> object:
                def encode(future_result: Future[int]) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: future_result
                    )

                    def on_complete(completed_future: Future[int]) -> None:
                        actual_result = completed_future.result()
                        deferred.finalize(actual_result)

                    future_result.add_done_callback(on_complete)
                    deferred_obj = deferred
                    return deferred

                return encode

            return future_encoder

        with ThreadPoolExecutor() as executor:
            barrier = Event()

            @memoizer.memoize(custom_result_encoder=future_encoder_factory)
            def compute_async(x: int) -> Future[int]:
                nonlocal call_count
                call_count += 1

                def work() -> int:
                    barrier.wait()  # Wait for signal
                    return x * 2

                return executor.submit(work)

            try:
                # First call: executes function, returns Future
                future1 = compute_async(5)
                self.assertEqual(call_count, 1)

                # Second call: should return same Future without re-execution
                future2 = compute_async(5)
                self.assertEqual(call_count, 1)  # No additional call
                self.assertIs(future1, future2)  # Same Future object
            finally:
                # Release the barrier - must be in finally to prevent test hang
                # if any assertion fails before this point
                barrier.set()

            # Wait for Future to complete
            result = future1.result(timeout=5)
            self.assertEqual(result, 10)

            # Verify deferred recording completed
            self.assertIsNotNone(deferred_obj)
            self.assertIsNone(deferred_obj._callbacks)

    @set_caching_module_enabled(True)
    def test_interim_result_transitions_to_cached_result(self) -> None:
        """Test that after complete(), cached result is used instead of interim.

        Verifies the transition from interim result to final cached result.
        """
        # Setup
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None
        call_count = 0

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: int) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: executes function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call before complete: uses interim result
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)

        # Finalize the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)

        # After finalize, memoize returns the cached result
        result3 = compute(5)
        self.assertEqual(result3, 100)
        self.assertEqual(call_count, 1)  # Still no additional call

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_interim_result(self) -> None:
        """Test that PersistentMemoizer handles make_interim_result correctly.

        Verifies that make_interim_result works with PersistentMemoizer and
        doesn't register duplicate disk callbacks.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: int) -> interfaces.DeferredRecording:
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @persistent.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: executes function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call: uses interim result, no re-execution
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)

        # Third call: still uses interim result
        result3 = compute(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 1)

        # Complete the deferred recording
        self.assertIsNotNone(deferred_obj)
        # Check that only 2 callbacks are registered (one for memory, one for disk)
        self.assertEqual(len(deferred_obj._callbacks), 2)
        deferred_obj.finalize(100)

        # Verify disk persistence
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @set_caching_module_enabled(True)
    def test_interim_result_different_params_independent(self) -> None:
        """Test that interim results for different params are independent.

        Verifies that each unique parameter set has its own interim result.
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        # Keep strong references to deferred recordings to prevent GC
        deferred_objs: list[interfaces.DeferredRecording] = []

        def deferred_encoder_factory(fn) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: int) -> interfaces.DeferredRecording:
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_objs.append(deferred)
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Call with different params
        result_5 = compute(5)
        self.assertEqual(result_5, 10)
        self.assertEqual(call_count, 1)

        result_10 = compute(10)
        self.assertEqual(result_10, 20)
        self.assertEqual(call_count, 2)

        # Subsequent calls with same params use interim result
        result_5_again = compute(5)
        self.assertEqual(result_5_again, 10)
        self.assertEqual(call_count, 2)  # No additional call

        result_10_again = compute(10)
        self.assertEqual(result_10_again, 20)
        self.assertEqual(call_count, 2)  # No additional call

        # Finalize all deferred recordings
        for deferred in deferred_objs:
            deferred.finalize(0)

    @set_caching_module_enabled(True)
    def test_get_interim_result_when_not_set(self) -> None:
        """Test that get_interim_result returns None when not configured.

        Verifies that when a DeferredRecording is created without make_interim_result,
        get_interim_result returns None indicating no interim result.
        """
        # Setup: create a DeferredRecording without make_interim_result
        deferred: interfaces.DeferredRecording[int, int] = (
            interfaces.DeferredRecording()
        )

        # Execute: get the interim result
        interim = deferred.get_interim_result()

        # Assert: should indicate no interim result available
        self.assertIsNone(interim)

    @set_caching_module_enabled(True)
    def test_get_interim_result_when_callable_returns_none(self) -> None:
        """Test get_interim_result when make_interim_result returns None.

        Verifies the distinction between:
        - None: no make_interim_result configured
        - InterimResult(None): make_interim_result configured but returns None

        This is an important edge case for type safety.
        """
        # Setup: create a DeferredRecording where make_interim_result returns None
        deferred: interfaces.DeferredRecording[None, int] = (
            interfaces.DeferredRecording(make_interim_result=lambda: None)
        )

        # Execute: get the interim result
        interim = deferred.get_interim_result()

        # Assert: should indicate interim result is available (even though value is None)
        self.assertIsNotNone(interim)
        self.assertIsNone(interim.value)

    @set_caching_module_enabled(True)
    def test_get_interim_result_callable_invoked_each_time(self) -> None:
        """Test that make_interim_result is invoked on each get_interim_result call.

        Verifies the documented behavior that the callable is invoked each time,
        which allows for returning new objects if desired.
        """
        # Setup: create a DeferredRecording with a callable that tracks invocations
        invocation_count = 0

        def make_result() -> int:
            nonlocal invocation_count
            invocation_count += 1
            return invocation_count

        deferred: interfaces.DeferredRecording[int, int] = interfaces.DeferredRecording(
            make_interim_result=make_result
        )

        # Execute: call get_interim_result multiple times
        interim1 = deferred.get_interim_result()
        interim2 = deferred.get_interim_result()
        interim3 = deferred.get_interim_result()

        # Assert: callable was invoked each time with incrementing results
        self.assertIsNotNone(interim1)
        self.assertIsNotNone(interim2)
        self.assertIsNotNone(interim3)
        self.assertEqual(interim1.value, 1)
        self.assertEqual(interim2.value, 2)
        self.assertEqual(interim3.value, 3)
        self.assertEqual(invocation_count, 3)

    @set_caching_module_enabled(True)
    def test_get_interim_result_exception_propagates(self) -> None:
        """Test that exceptions from make_interim_result propagate correctly.

        Verifies that if make_interim_result raises an exception, it propagates
        to the caller and does not corrupt the DeferredRecording state.
        """
        # Setup: create a DeferredRecording with a failing callable
        call_count = 0

        def failing_callable() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return 42

        deferred: interfaces.DeferredRecording[int, int] = interfaces.DeferredRecording(
            make_interim_result=failing_callable
        )

        # Execute & Assert: first call raises exception
        with self.assertRaises(ValueError) as ctx:
            deferred.get_interim_result()
        self.assertIn("First call fails", str(ctx.exception))

        # Execute: second call should work (state not corrupted)
        interim = deferred.get_interim_result()

        # Assert: should work normally after exception
        self.assertIsNotNone(interim)
        self.assertEqual(interim.value, 42)

    @set_caching_module_enabled(True)
    def test_get_interim_result_thread_safety_with_finalize(self) -> None:
        """Test that get_interim_result and finalize are thread-safe together.

        Verifies that concurrent calls to get_interim_result and finalize
        don't cause race conditions. The interim result should still be
        accessible until the DeferredRecording is finalized.
        """
        num_iterations = 50
        errors: list[Exception] = []

        def run_concurrent_test() -> None:
            deferred: interfaces.DeferredRecording[int, int] = (
                interfaces.DeferredRecording(make_interim_result=lambda: 42)
            )
            barrier = Event()
            interim_results: list[interfaces.InterimResult[int] | None] = []

            def get_interim_thread() -> None:
                barrier.wait()
                for _ in range(10):
                    try:
                        interim_results.append(deferred.get_interim_result())
                    except Exception as e:
                        errors.append(e)

            def finalize_thread() -> None:
                barrier.wait()
                try:
                    deferred.finalize(100)
                except RuntimeError:
                    pass

            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(get_interim_thread)
                f2 = executor.submit(finalize_thread)
                barrier.set()
                wait([f1, f2], timeout=5)

            # All interim results should be valid (either 42 or from make_interim_result)
            for interim in interim_results:
                if interim is not None and interim.value != 42:
                    errors.append(AssertionError(f"Expected 42, got {interim.value}"))

        # Execute: run many iterations to catch race conditions
        for _ in range(num_iterations):
            run_concurrent_test()

        # Assert: no errors occurred
        self.assertEqual(errors, [])

    @set_caching_module_enabled(True)
    def test_interim_result_no_make_interim_causes_reexecution(self) -> None:
        """Test that without make_interim_result, function is re-executed.

        Verifies that when a DeferredRecording does NOT have make_interim_result,
        subsequent calls while pending cause the function to be re-executed
        (the original behavior before interim results were added).
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        # Keep strong references to deferred recordings to prevent GC
        deferred_objs: list[interfaces.DeferredRecording] = []

        def deferred_encoder_factory_no_interim(fn: object) -> object:
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                def encode(result: int) -> interfaces.DeferredRecording:
                    # Explicitly not setting make_interim_result
                    deferred = interfaces.DeferredRecording()
                    deferred_objs.append(deferred)
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory_no_interim)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: should execute function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call: should re-execute (no interim result available)
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 2)

        # Finalize all deferred recordings
        for deferred in deferred_objs:
            deferred.finalize(10)


if __name__ == "__main__":
    run_tests()
