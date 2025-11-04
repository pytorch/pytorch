# Owner(s): ["module: inductor"]
# pyre-strict
from __future__ import annotations

import atexit
import os
import pickle
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from contextlib import contextmanager
from functools import wraps
from itertools import combinations
from random import Random
from shutil import rmtree
from threading import Lock
from time import sleep, time
from typing import Any, TYPE_CHECKING, Union
from typing_extensions import TypeVar
from unittest.mock import patch

from filelock import FileLock

import torch
from torch._inductor.runtime.caching import (
    config,
    context,
    exceptions,
    implementations as impls,
    interfaces as intfs,
    locks,
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


def patch_deterministic_cache_intf_no_dump_on_exit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        default_init = intfs._DeterministicCacheIntf.__init__

        def patched_init(
            intf: intfs._DeterministicCacheIntf, *args: Any, **kwargs: dict[str, Any]
        ) -> None:
            default_init(intf, *args, **kwargs)
            atexit.unregister(intf._dump_imc_to_disk)

        with patch.object(intfs._DeterministicCacheIntf, "__init__", patched_init):
            return fn(*args, **kwargs)

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
        "KeyPicklingError",
        "ValueEncodingError",
        "ValuePicklingError",
        "ValueDecodingError",
        "ValueUnPicklingError",
        "CustomParamsEncoderRequiredError",
        "CustomResultEncoderRequiredError",
        "CustomResultDecoderRequiredError",
        "DeterministicCachingRequiresStrongConsistencyError",
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
        self.assertTrue(
            issubclass(exceptions.KeyPicklingError, exceptions.KeyEncodingError)
        )
        self.assertTrue(issubclass(exceptions.ValueEncodingError, exceptions.UserError))
        self.assertTrue(
            issubclass(exceptions.ValuePicklingError, exceptions.ValueEncodingError)
        )
        self.assertTrue(issubclass(exceptions.ValueDecodingError, exceptions.UserError))
        self.assertTrue(
            issubclass(exceptions.ValueUnPicklingError, exceptions.ValueDecodingError)
        )
        self.assertTrue(
            issubclass(
                exceptions.CustomParamsEncoderRequiredError, exceptions.UserError
            )
        )
        self.assertTrue(
            issubclass(
                exceptions.CustomResultEncoderRequiredError, exceptions.UserError
            )
        )
        self.assertTrue(
            issubclass(
                exceptions.CustomResultDecoderRequiredError, exceptions.UserError
            )
        )
        self.assertTrue(
            issubclass(
                exceptions.DeterministicCachingRequiresStrongConsistencyError,
                exceptions.UserError,
            )
        )


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

    def assert_key_in(self, key: Any, impl: impls._CacheImpl) -> None:
        self.assertTrue(impl.get(key) is not None)

    def assert_key_not_in(self, key: Any, impl: impls._CacheImpl) -> None:
        self.assertTrue(impl.get(key) is None)

    def assert_key_value_inserted_in(
        self, key: Any, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertTrue(impl.insert(key, value))

    def assert_key_value_not_inserted_in(
        self, key: Any, value: Any, impl: impls._CacheImpl
    ) -> None:
        self.assertFalse(impl.insert(key, value))

    def assert_key_has_value_in(
        self, key: Any, value: Any, impl: impls._CacheImpl
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
            value: str = self.random_string
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
            value: str = self.random_string
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_value_not_inserted_in(key, self.random_string, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_key_encoding(self, impl_typename: str) -> None:
        """Test that cache implementations properly handle non-serializable keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        raise KeyPicklingError when attempting to insert keys that cannot be
        pickled (such as lambda functions). This ensures proper error handling
        for invalid key types that would break the caching system.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            with self.assertRaises(exceptions.KeyPicklingError):
                impl.insert(lambda: None, None)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_encoding(self, impl_typename: str) -> None:
        """Test that on-disk cache implementations properly handle non-serializable values.

        Verifies that on-disk cache implementations correctly raise ValuePicklingError
        when attempting to insert values that cannot be pickled (such as lambda functions).
        This test only applies to on-disk implementations since in-memory caches don't
        require serialization. Ensures proper error handling for invalid value types.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                with self.assertRaises(exceptions.ValuePicklingError):
                    impl.insert(None, lambda: None)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_decoding(self, impl_typename: str) -> None:
        """Test that on-disk cache implementations properly handle corrupted cached values.

        Verifies that on-disk cache implementations correctly raise ValueUnPicklingError
        when attempting to retrieve values from cache files that contain corrupted or
        invalid pickled data. This test ensures proper error handling when cached data
        becomes corrupted on disk. Only applies to on-disk implementations since
        in-memory caches don't involve serialization/deserialization.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                fpath: Path = impl._fpath_from_key(key)
                with open(fpath, "xb") as fp:
                    impl._write_version_header(fp)
                    fp.write(b"foo")
                with self.assertRaises(exceptions.ValueUnPicklingError):
                    impl.get(key)

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
                value: str = self.random_string
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
class InterfacesTest(TestMixin, TestCase):
    intf_typenames: list[str] = [
        "_FastCacheIntf",
        "_DeterministicCacheIntf",
    ]

    @classmethod
    def sub_dir(cls) -> str:
        return f"testing-intfs-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        rmtree(impls._OnDiskCacheImpl()._base_dir / cls.sub_dir(), ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(impls._OnDiskCacheImpl()._base_dir / cls.sub_dir(), ignore_errors=True)

    def intf_from_typename(self, intf_typename: str) -> intfs._CacheIntf:
        return getattr(intfs, intf_typename)()

    def assert_call_is_cached(self, fn, params, intf, *args, **kwargs) -> None:
        self.assertTrue(intf.get(fn, params, *args, **kwargs) is not None)

    def assert_call_is_not_cached(self, fn, params, intf, *args, **kwargs) -> None:
        self.assertTrue(intf.get(fn, params, *args, **kwargs) is None)

    def assert_call_was_cached(self, fn, params, result, intf, *args, **kwargs) -> None:
        self.assertTrue(intf.insert(fn, params, result, *args, **kwargs))

    def assert_call_was_not_cached(
        self, fn, params, result, intf, *args, **kwargs
    ) -> None:
        self.assertFalse(intf.insert(fn, params, result, *args, **kwargs))

    def assert_cached_call_is(self, fn, params, result, intf, *args, **kwargs) -> None:
        self.assertTrue(
            ((get := intf.get(fn, params, *args, **kwargs)) is not None)
            and (get.value == result)
        )

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_defaults(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)
        sleep_t: int = 5

        @intf.record()
        def foo(*args, **kwargs) -> None:
            sleep(sleep_t)
            return (args, kwargs)

        args, kwargs = (
            (
                1,
                2,
                3,
            ),
            {"bar": "bar"},
        )
        params = (args, kwargs)
        self.assertEqual(foo(*args, **kwargs), params)
        start_t: float = time()
        self.assertEqual(foo(*args, **kwargs), params)
        self.assertTrue((time() - start_t) < sleep_t)
        self.assert_cached_call_is(
            foo, params, params, intf, ischema=context._DEFAULT_ISOLATION_SCHEMA
        )

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_custom_params_encoder(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)
        sleep_t: int = 5

        @intf.record(custom_params_encoder=lambda params: "bar")
        def foo(_lambda) -> None:
            sleep(sleep_t)
            return hash(_lambda)

        _lambda = lambda: None  # noqa: E731
        self.assertEqual(foo(_lambda), hash(_lambda))
        start_t: float = time()
        self.assertEqual(foo(_lambda), hash(_lambda))
        self.assertTrue((time() - start_t) < sleep_t)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_custom_result_encoder_and_decoder(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)
        sleep_t: int = 5

        @intf.record(
            custom_result_encoder=lambda value: "bar",
            custom_result_decoder=lambda encoded_value: "bar",
        )
        def foo() -> None:
            sleep(sleep_t)
            return "foo"

        self.assertEqual(foo(), "foo")
        start_t: float = time()
        self.assertEqual(foo(), "bar")
        self.assertTrue((time() - start_t) < sleep_t)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_custom_ischema(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)
        sleep_t: int = 5

        @intf.record(
            ischema=context.IsolationSchema(
                runtime_context=context.SelectedRuntimeContext(
                    inductor_configs=True,
                    torch_determinism_configs=False,
                    cuda_matmul_precision_configs=False,
                ),
                compile_context=False,
            ),
        )
        def foo(*args, **kwargs) -> None:
            sleep(sleep_t)
            return (args, kwargs)

        with patch.object(torch._inductor.config, "max_autotune", True):
            self.assertEqual(
                foo("foo"),
                (
                    ("foo",),
                    {},
                ),
            )
            start_t: float = time()
            self.assertEqual(
                foo("foo"),
                (
                    ("foo",),
                    {},
                ),
            )
            self.assertTrue((time() - start_t) < sleep_t)

        with patch.object(torch._inductor.config, "max_autotune", False):
            start_t: float = time()
            self.assertEqual(
                foo("foo"),
                (
                    ("foo",),
                    {},
                ),
            )
            self.assertTrue((time() - start_t) >= sleep_t)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_params_encoder_required(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)

        @intf.record()
        def foo(*args, **kwargs) -> None:
            return (args, kwargs)

        with self.assertRaises(exceptions.CustomParamsEncoderRequiredError):
            foo(lambda: None)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_result_encoder_required(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)

        @intf.record()
        def foo(*args, **kwargs) -> None:
            return lambda: None

        with self.assertRaises(exceptions.CustomResultEncoderRequiredError):
            foo(0)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @patch_deterministic_cache_intf_no_dump_on_exit
    @parametrize("intf_typename", intf_typenames)
    def test_result_encoder_and_decoder_required(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)

        with self.assertRaises(exceptions.CustomResultEncoderRequiredError):

            @intf.record(custom_result_decoder=lambda: None)
            def foo() -> None:
                return None

            # otherwise flake8 complains about foo unused
            _ = foo()

        with self.assertRaises(exceptions.CustomResultDecoderRequiredError):

            @intf.record(custom_result_encoder=lambda: None)
            def foo() -> None:
                return None

            # otherwise flake8 complains about foo unused
            _ = foo()

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_deterministic_cache_intf_no_dump_on_exit
    @patch_remote_cache_with_on_disk_cache
    def test_strictly_pre_populated_determinism(self) -> None:
        intf: intfs._DeterministicCacheIntf = self.intf_from_typename(
            "_DeterministicCacheIntf"
        )

        @intf.record()
        def foo(*args: Any, **kwargs: dict[str, Any]) -> None:
            return None

        args, kwargs = (
            (
                1,
                2,
                3,
            ),
            {"bar": "bar"},
        )
        params, result = (args, kwargs), None

        self.assertEqual(foo(*args, **kwargs), result)
        self.assertTrue(
            ((get := intf.get(foo, params)) is not None) and (get.value == result)
        )
        self.assertTrue((fpath := intf._dump_imc_to_disk()) is not None)

        with (
            set_strictly_pre_populated_determinism(True),
            patch.dict(
                os.environ,
                {
                    "TORCHINDUCTOR_PRE_POPULATE_DETERMINISTIC_CACHE": str(fpath),
                },
            ),
        ):
            intf: intfs._DeterministicCacheIntf = self.intf_from_typename(
                "_DeterministicCacheIntf"
            )

            @intf.record()
            def foo(*args: Any, **kwargs: dict[str, Any]) -> None:
                return None

            self.assertEqual(foo(*args, **kwargs), result)

            with self.assertRaises(
                exceptions.StrictDeterministicCachingKeyNotFoundError
            ):
                foo()

            with self.assertRaises(
                exceptions.StrictDeterministicCachingKeyNotFoundError
            ):
                intf.get(foo, ((), {}))

            with self.assertRaises(exceptions.StrictDeterministicCachingInsertionError):
                intf.insert(foo, ((), {}), None)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(True)
    @set_strictly_pre_populated_determinism(False)
    @set_strictly_cached_determinism(False)
    @patch_on_disk_cache_base_dir
    @patch_deterministic_cache_intf_no_dump_on_exit
    @patch_remote_cache_with_on_disk_cache
    def test_strictly_cached_determinism(self) -> None:
        intf: intfs._DeterministicCacheIntf = self.intf_from_typename(
            "_DeterministicCacheIntf"
        )

        @intf.record()
        def foo(*args: Any, **kwargs: dict[str, Any]) -> None:
            return None

        args, kwargs = (
            (
                1,
                2,
                3,
            ),
            {"bar": "bar"},
        )
        params, result = (args, kwargs), None

        self.assertEqual(foo(*args, **kwargs), result)
        self.assertTrue(
            ((get := intf.get(foo, params)) is not None) and (get.value == result)
        )

        with set_strictly_cached_determinism(True):
            self.assertEqual(foo(*args, **kwargs), result)

            with self.assertRaises(
                exceptions.StrictDeterministicCachingKeyNotFoundError
            ):
                foo()

            with self.assertRaises(
                exceptions.StrictDeterministicCachingKeyNotFoundError
            ):
                intf.get(foo, ((), {}))

            with self.assertRaises(exceptions.StrictDeterministicCachingInsertionError):
                intf.insert(foo, ((), {}), None)

    @set_caching_module_enabled(False)
    @set_deterministic_caching_enabled(True)
    @patch_on_disk_cache_base_dir
    @patch_deterministic_cache_intf_no_dump_on_exit
    @patch_remote_cache_with_on_disk_cache
    @parametrize("intf_typename", intf_typenames)
    def test_caching_module_disabled(self, intf_typename: str) -> None:
        intf: intfs._CacheIntf = self.intf_from_typename(intf_typename)
        sleep_t: int = 5

        @intf.record()
        def foo(*args, **kwargs) -> None:
            sleep(sleep_t)
            return (args, kwargs)

        args, kwargs = (
            (
                1,
                2,
                3,
            ),
            {"bar": "bar"},
        )
        params = (args, kwargs)

        self.assertIsNone(intf.get(foo, params))
        self.assertFalse(intf.insert(foo, params, params))

        foo(*args, **kwargs)
        start_t: float = time()
        foo(*args, **kwargs)
        self.assertTrue((time() - start_t) >= sleep_t)

    @set_caching_module_enabled(True)
    @set_deterministic_caching_enabled(False)
    @patch_on_disk_cache_base_dir
    @patch_deterministic_cache_intf_no_dump_on_exit
    @patch_remote_cache_with_on_disk_cache
    def test_deterministic_caching_disabled(self) -> None:
        intf: intfs._DeterministicCacheIntf = self.intf_from_typename(
            "_DeterministicCacheIntf"
        )

        @intf.record()
        def foo(*args, **kwargs) -> None:
            return (args, kwargs)

        args, kwargs = (
            (
                1,
                2,
                3,
            ),
            {"bar": "bar"},
        )
        params = (args, kwargs)

        with self.assertRaises(exceptions.DeterministicCachingDisabledError):
            intf.get(foo, params)

        with self.assertRaises(exceptions.DeterministicCachingDisabledError):
            intf.insert(foo, params, params)

        with self.assertRaises(exceptions.DeterministicCachingDisabledError):
            foo(*args, **kwargs)


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

        assert lock_typename in ["Lock", "FileLock"]
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
            assert lock_timeout in ["BLOCKING", "NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]
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

    @parametrize("pickle_able", [True, False])
    def test_try_pickle_key(self, pickle_able: bool) -> None:
        """Test that cache key pickling works correctly and raises appropriate exceptions.

        Verifies that the _try_pickle_key function successfully pickles serializable
        cache keys and raises KeyPicklingError for non-serializable keys like lambda
        functions. Tests both the successful pickling path and error handling.
        """
        if pickle_able:
            key: str = self.random_string
            self.assertEqual(pickle.loads(utils._try_pickle_key(key)), key)
        else:
            with self.assertRaises(exceptions.KeyPicklingError):
                _ = utils._try_pickle_key(lambda: None)

    @parametrize("pickle_able", [True, False])
    def test_try_pickle_value(self, pickle_able: bool) -> None:
        """Test that cache value pickling works correctly and raises appropriate exceptions.

        Verifies that the _try_pickle_value function successfully pickles serializable
        cache values and raises ValuePicklingError for non-serializable values like
        lambda functions. Tests both successful pickling and proper error handling.
        """
        if pickle_able:
            value: str = self.random_string
            self.assertEqual(pickle.loads(utils._try_pickle_value(value)), value)
        else:
            with self.assertRaises(exceptions.ValuePicklingError):
                _ = utils._try_pickle_value(lambda: None)

    @parametrize("unpickle_able", [True, False])
    def test_try_unpickle_value(self, unpickle_able: bool) -> None:
        """Test that cache value unpickling works correctly and raises appropriate exceptions.

        Verifies that the _try_unpickle_value function successfully unpickles valid
        pickled data and raises ValueUnPicklingError for invalid data like None.
        Tests both successful unpickling and proper error handling for corrupted data.
        """
        if unpickle_able:
            value: str = self.random_string
            self.assertEqual(utils._try_unpickle_value(pickle.dumps(value)), value)
        else:
            with self.assertRaises(exceptions.ValueUnPicklingError):
                _ = utils._try_unpickle_value(b"foo")


if __name__ == "__main__":
    run_tests()
