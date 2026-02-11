"""Exception classes for PyTorch Inductor runtime caching.

This module defines a hierarchy of exceptions used throughout the caching system.
All custom exceptions inherit from CacheError, with UserError serving as a base
for user-facing errors that also inherit from TypeError for compatibility.
"""

from threading import Lock

from filelock import BaseFileLock


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

    def __init__(self, flock: BaseFileLock, timeout: float) -> None:
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


class ValueEncodingError(UserError):
    """Base class for errors that occur during cache value encoding operations.

    Raised when cache values cannot be properly encoded for storage or transmission.
    This includes serialization, compression, or other encoding-related failures.
    """


class ValueDecodingError(UserError):
    """Base class for errors that occur during cache value decoding operations.

    Raised when cached values cannot be properly decoded during retrieval.
    This includes deserialization, decompression, or other decoding-related failures.
    """
