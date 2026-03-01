import functools
from collections.abc import Callable
from typing import Concatenate, TypeVar
from typing_extensions import ParamSpec


_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")

# Sentinel used to indicate that cache lookup failed.
_cache_sentinel = object()


def cache_method(
    f: Callable[Concatenate[_C, _P], _T],
) -> Callable[Concatenate[_C, _P], _T]:
    """
    Like `@functools.cache` but for methods.

    `@functools.cache` (and similarly `@functools.lru_cache`) shouldn't be used
    on methods because it caches `self`, keeping it alive
    forever. `@cache_method` ignores `self` so won't keep `self` alive (assuming
    no cycles with `self` in the parameters).

    Footgun warning: This decorator completely ignores self's properties so only
    use it when you know that self is frozen or won't change in a meaningful
    way (such as the wrapped function being pure).
    """
    cache_name = "_cache_method_" + f.__name__

    @functools.wraps(f)
    def wrap(self: _C, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if kwargs:
            raise AssertionError("cache_method does not accept keyword arguments")
        if not (cache := getattr(self, cache_name, None)):
            cache = {}
            setattr(self, cache_name, cache)
        cached_value = cache.get(args, _cache_sentinel)
        if cached_value is not _cache_sentinel:
            return cached_value
        value = f(self, *args, **kwargs)
        cache[args] = value
        return value

    return wrap
