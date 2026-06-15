from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Concatenate, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future


_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")

# Sentinel used to indicate that cache lookup failed.
_cache_sentinel = object()

_prefetch_executor = ThreadPoolExecutor(max_workers=2)


def prefetchable_cache(func: Callable[[], _T]) -> Callable[[], _T]:
    """
    Like functools.cache but with a prefetch() method that starts computing
    the value in a background thread, and a set() method for prepopulating.
    """
    _cache: _T | object = _cache_sentinel
    _lock = threading.Lock()
    _future: Future[_T] | None = None

    def wrapper() -> _T:
        nonlocal _cache, _future
        with _lock:
            if _cache is not _cache_sentinel:
                return _cache  # type: ignore[return-value]
            if _future is not None:
                _cache = _future.result()
                _future = None
                return _cache  # type: ignore[return-value]
            _cache = func()
            return _cache  # type: ignore[return-value]

    def set_val(val: _T) -> None:
        nonlocal _cache
        with _lock:
            if _cache is not _cache_sentinel:
                raise RuntimeError("prefetchable_cache value already set")
            _cache = val

    def clear() -> None:
        nonlocal _cache, _future
        with _lock:
            _cache = _cache_sentinel
            _future = None

    def prefetch() -> None:
        nonlocal _future
        with _lock:
            if _cache is not _cache_sentinel or _future is not None:
                return
            _future = _prefetch_executor.submit(func)

    wrapper.set = set_val  # type: ignore[attr-defined]
    wrapper.clear = clear  # type: ignore[attr-defined]
    wrapper.prefetch = prefetch  # type: ignore[attr-defined]
    return wrapper


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
