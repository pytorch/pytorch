"""Utility functions for caching operations in PyTorch Inductor runtime.

This module provides helper functions for pickling/unpickling operations
with error handling, LRU caching decorators, and type-safe serialization
utilities used throughout the caching system.
"""

import pickle
from collections.abc import Callable
from functools import lru_cache, partial, wraps
from typing import Any
from typing_extensions import ParamSpec, TypeVar

from . import exceptions


# Type specification for function parameters
P = ParamSpec("P")
# Type variable for function return values
R = TypeVar("R")


def _lru_cache(fn: Callable[P, R]) -> Callable[P, R]:
    """LRU cache decorator with TypeError fallback.

    Provides LRU caching with a fallback mechanism that calls the original
    function if caching fails due to unhashable arguments. Uses a cache
    size of 64 with typed comparison.

    Args:
        fn: The function to be cached.

    Returns:
        A wrapper function that attempts caching with fallback to original function.
    """
    cached_fn = lru_cache(maxsize=64, typed=True)(fn)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[type-var]
        try:
            return cached_fn(*args, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            return fn(*args, **kwargs)

    return wrapper


@_lru_cache
def _try_pickle(to_pickle: Any, raise_if_failed: type = exceptions.CacheError) -> bytes:
    """Attempt to pickle an object with error handling.

    Tries to serialize an object using pickle.dumps with appropriate error
    handling and custom exception raising.

    Args:
        to_pickle: The object to be pickled.
        raise_if_failed: Exception class to raise if pickling fails.

    Returns:
        The pickled bytes representation of the object.

    Raises:
        The exception class specified in raise_if_failed if pickling fails.
    """
    try:
        pickled: bytes = pickle.dumps(to_pickle)
    except (pickle.PicklingError, AttributeError) as err:
        raise raise_if_failed(to_pickle) from err
    return pickled


# Specialized pickle function for cache keys with KeyPicklingError handling.
_try_pickle_key: Callable[[Any], bytes] = partial(
    _try_pickle, raise_if_failed=exceptions.KeyPicklingError
)
# Specialized pickle function for cache values with ValuePicklingError handling.
_try_pickle_value: Callable[[Any], bytes] = partial(
    _try_pickle, raise_if_failed=exceptions.ValuePicklingError
)


@_lru_cache
def _try_unpickle(pickled: bytes, raise_if_failed: type = exceptions.CacheError) -> Any:
    """Attempt to unpickle bytes with error handling.

    Tries to deserialize bytes using pickle.loads with appropriate error
    handling and custom exception raising.

    Args:
        pickled: The bytes to be unpickled.
        raise_if_failed: Exception class to raise if unpickling fails.

    Returns:
        The unpickled object.

    Raises:
        The exception class specified in raise_if_failed if unpickling fails.
    """
    try:
        unpickled: Any = pickle.loads(pickled)
    except pickle.UnpicklingError as err:
        raise raise_if_failed(pickled) from err
    return unpickled


# Specialized unpickle function for cache keys with KeyUnPicklingError handling.
_try_unpickle_value: Callable[[Any], bytes] = partial(
    _try_unpickle, raise_if_failed=exceptions.ValueUnPicklingError
)
