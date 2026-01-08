"""Utility functions

This module provides helper functions for LRU caching decorators used
throughout the caching system.
"""

from collections.abc import Callable
from functools import lru_cache, wraps
from typing_extensions import ParamSpec, TypeVar


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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return cached_fn(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)

    return wrapper
