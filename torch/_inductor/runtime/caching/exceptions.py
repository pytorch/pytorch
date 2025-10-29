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
    pass


class CustomResultEncoderRequiredError(UserError):
    pass


class CustomResultDecoderRequiredError(UserError):
    pass


class DeterministicCachingDisabledError(UserError):
    pass


class DeterministicCachingRequiresStrongConsistencyError(UserError):
    pass


class StrictDeterministicCachingKeyNotFoundError(UserError):
    pass


class DeterministicCachingInvalidConfigurationError(UserError):
    pass


class StrictDeterministicCachingInsertionError(UserError):
    pass


class DeterministicCachingIMCDumpConflictError(SystemError):
    pass
