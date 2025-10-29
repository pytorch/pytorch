# pyre-strict

"""Exception classes for PyTorch Inductor runtime caching.

This module defines a hierarchy of exceptions used throughout the caching system.
All custom exceptions inherit from CacheError, with UserError serving as a base
for user-facing errors that also inherit from TypeError for compatibility.
"""

from threading import Lock
from typing import Any

from filelock import FileLock


class CacheError(Exception):
    """Base class for all caching-related errors.

    This is the root exception class for all custom exceptions raised by the caching
    module, providing a common interface for error handling and logging.
    """


class SystemError(CacheError, RuntimeError):
    """Base class for system-level caching errors.

    This class represents errors that occur during cache operations, such as
    storage or retrieval failures. It inherits from RuntimeError to indicate
    that the error is not caused by user input.
    """


class LockTimeoutError(SystemError):
    """Error raised when a lock operation times out.

    This exception is raised when a lock operation exceeds the specified timeout
    limit, indicating that the lock could not be acquired within the allotted time.
    """

    def __init__(self, lock: Lock, timeout: float) -> None:
        """Initialize the lock timeout error with detailed lock information.

        Args:
            lock: The lock object that timed out.
            timeout: The timeout limit that was exceeded.
        """
        super().__init__(f"Failed to acquire lock {lock} within {timeout} seconds.")


class FileLockTimeoutError(SystemError):
    """Error raised when a file lock operation times out.

    This exception is raised when a file lock operation exceeds the specified timeout
    limit, indicating that the lock could not be acquired within the allotted time.
    """

    def __init__(self, flock: FileLock, timeout: float) -> None:
        """Initialize the file lock timeout error with detailed lock information.

        Args:
            flock: The file lock object that timed out.
            timeout: The timeout limit that was exceeded.
        """
        super().__init__(
            f"Failed to acquire file lock {flock} within {timeout} seconds."
        )


class UserError(CacheError, TypeError):
    """Base class for user-facing cache errors that also inherit from TypeError.

    This class combines CacheError with TypeError to provide compatibility
    with existing exception handling patterns while maintaining the cache
    error hierarchy. All user-facing cache errors should inherit from this class.
    """


class KeyEncodingError(UserError):
    """Base class for errors that occur during cache key encoding operations.

    Raised when cache keys cannot be properly encoded for storage or transmission.
    This includes serialization, hashing, or other encoding-related failures.
    """


class KeyPicklingError(KeyEncodingError):
    """Error raised when a cache key cannot be pickled for serialization.

    This typically occurs when trying to cache objects with keys that contain
    non-serializable components, lambda functions, or other unpickleable types.
    """

    def __init__(self, key: Any) -> None:
        """Initialize the key pickling error with detailed key information.

        Args:
            key: The cache key that failed to be pickled.
        """
        super().__init__(
            f"Failed to pickle cache key with type {type(key)} and value {key!r}."
        )


class ValueEncodingError(UserError):
    """Base class for errors that occur during cache value encoding operations.

    Raised when cache values cannot be properly encoded for storage or transmission.
    This includes serialization, compression, or other encoding-related failures.
    """


class ValuePicklingError(ValueEncodingError):
    """Error raised when a cache value cannot be pickled for serialization.

    This occurs when trying to cache objects that contain non-serializable
    components, file handles, network connections, or other unpickleable types.
    """

    def __init__(self, value: Any) -> None:
        """Initialize the value pickling error with detailed value information.

        Args:
            value: The cache value that failed to be pickled.
        """
        super().__init__(
            f"Failed to pickle cache value with type {type(value)} and value {value!r}."
        )


class ValueDecodingError(UserError):
    """Base class for errors that occur during cache value decoding operations.

    Raised when cached values cannot be properly decoded during retrieval.
    This includes deserialization, decompression, or other decoding-related failures.
    """


class ValueUnPicklingError(ValueDecodingError):
    """Error raised when cached value data cannot be unpickled during retrieval.

    This typically indicates corruption, version incompatibility, or missing
    dependencies required to reconstruct the cached object.
    """

    def __init__(self, pickled_value: bytes) -> None:
        """Initialize the value unpickling error with the problematic data.

        Args:
            pickled_value: The bytes that failed to be unpickled.
        """
        super().__init__(
            f"Failed to unpickle cache value from pickled value {pickled_value!r}."
        )


class CustomParamsEncoderRequiredError(UserError):
    """Error raised when a custom parameter encoder is required but not provided.

    This exception occurs when attempting to cache a function with parameters
    that cannot be automatically serialized. A custom encoder must be provided
    to convert the parameters into a serializable format.
    """


class CustomResultEncoderRequiredError(UserError):
    """Error raised when a custom result encoder is required but not provided.

    This exception occurs when attempting to cache a function result that
    cannot be automatically serialized. A custom encoder must be provided
    to convert the result into a serializable format.
    """


class CustomResultDecoderRequiredError(UserError):
    """Error raised when a custom result decoder is required but not provided.

    This exception occurs when a custom result encoder is provided without
    a corresponding decoder. Both encoder and decoder must be provided together
    to ensure proper serialization and deserialization of cached results.
    """


class DeterministicCachingDisabledError(UserError):
    """Error raised when attempting to use deterministic caching while it's disabled.

    This exception is raised when code tries to access the deterministic caching
    interface (dcache) but deterministic caching has been disabled through
    configuration. Use the icache interface for automatic fallback behavior.
    """


class DeterministicCachingRequiresStrongConsistencyError(UserError):
    """Error raised when deterministic caching is enabled without strong consistency.

    Deterministic caching with global synchronization requires a remote cache
    backend with strong consistency guarantees. This exception is raised when
    the configured remote cache does not provide these guarantees.
    """


class StrictDeterministicCachingKeyNotFoundError(UserError):
    """Error raised when a cache key is not found in strict deterministic mode.

    In strictly pre-populated or strictly cached deterministic modes, all cache
    keys must either be pre-populated or already cached. This exception is raised
    when attempting to access a key that doesn't exist in these strict modes.
    """


class DeterministicCachingInvalidConfigurationError(UserError):
    """Error raised when deterministic caching configuration is invalid.

    This exception occurs when deterministic caching is enabled but none of
    the required configuration modes (STRICTLY_PRE_POPULATED_DETERMINISM,
    GLOBAL_DETERMINISM, or LOCAL_DETERMINISM) are enabled.
    """


class StrictDeterministicCachingInsertionError(UserError):
    """Error raised when attempting to insert into cache in strict deterministic mode.

    In strictly pre-populated or strictly cached deterministic modes, insertions
    are not allowed as the cache should only contain pre-populated or previously
    cached values. This exception is raised when attempting to insert new entries.
    """


class DeterministicCachingIMCDumpConflictError(SystemError):
    """Error raised when in-memory cache dumps conflict with existing dumps.

    This exception occurs when attempting to dump the in-memory cache to disk
    but the dump file already exists with conflicting entries. This indicates
    a potential race condition or inconsistency in cache state.
    """
