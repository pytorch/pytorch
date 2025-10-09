# Owner(s): ["module: inductor"]
# pyre-strict

import os
import pickle
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from random import Random
from threading import Lock
from typing import Generator, Sequence, Union
from typing_extensions import Self, TypeVar
from unittest.mock import patch

from filelock import FileLock

from torch._inductor.runtime.caching import config, context, exceptions, locks, utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class TestMixin:
    @property
    def random_string(self) -> str:
        return f"s-{Random().randint(0, 2**32)}"


@instantiate_parametrized_tests
class ConfigTest(TestCase):
    FOO_THIS_VERSION: int = 0
    FOO_JK_NAME: str = "foo_jk_name"
    FOO_OSS_DEFAULT: bool = False
    FOO_ENV_VAR_OVERRIDE: str = "foo_env_var_override"
    FOO_ENV_VAR_OVERRIDE_LOCK: FileLock = FileLock(
        f"/tmp/testing/{FOO_ENV_VAR_OVERRIDE}.lock"
    )

    def assert_versioned_config(self, expected_enabled: bool) -> None:
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
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", enabled),
        ):
            self.assert_versioned_config(enabled)


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


@instantiate_parametrized_tests
class LocksTest(TestMixin, TestCase):
    T = TypeVar("T")

    @contextmanager
    def executor(self: Self) -> Generator[ThreadPoolExecutor, None, None]:
        executor: ThreadPoolExecutor = ThreadPoolExecutor()
        try:
            yield executor
        finally:
            executor.shutdown()

    def is_lock(self: Self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        return hasattr(lock_or_flock, "locked")

    def is_flock(self: Self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        return hasattr(lock_or_flock, "is_locked")

    def lock_or_flock_locked(self: Self, lock_or_flock: Union[Lock, FileLock]) -> bool:
        if self.is_lock(lock_or_flock):
            return lock_or_flock.locked()
        elif self.is_flock(lock_or_flock):
            return lock_or_flock.is_locked
        else:
            raise NotImplementedError

    def test_BLOCKING(self: Self) -> None:
        self.assertEqual(locks._BLOCKING, -1.0)

    def test_NON_BLOCKING(self: Self) -> None:
        self.assertEqual(locks._NON_BLOCKING, 0.0)

    def test_BLOCKING_WITH_TIMEOUT(self: Self) -> None:
        self.assertGreater(locks._BLOCKING_WITH_TIMEOUT, 0.0)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1.0)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1.0)
    @parametrize("lock_typename", ["Lock", "FileLock"])
    @parametrize("lock_timeout", ["BLOCKING", "NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"])
    @parametrize("acquisition_mode", ["safe", "unsafe"])
    @parametrize("release", ["unlocked", "never", "before_timeout", "after_timeout"])
    def test_acquire_with_timeout(
        self: Self,
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
        flock_fpath: Path = Path("/tmp/LocksTest") / self.random_string
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
