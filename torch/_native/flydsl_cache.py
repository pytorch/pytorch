"""Small in-process specialization cache for FlyDSL native operators.

FlyDSL already caches compiler artifacts on disk. This cache keeps the
``flyc.compile`` result for one in-process specialization, so repeated eager
operator calls do not rebuild the Python launcher.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import functools
from collections import namedtuple
from threading import Lock


CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])


class _JitCacheWrapper:
    """Cache a compile helper by its explicit specialization arguments.

    ``compile_args`` contains sample tensors and a stream that FlyDSL needs on
    a cache miss. They are intentionally excluded from the key: stable values
    such as hidden size, dtype, architecture, device and epsilon are passed as
    normal arguments and form the key instead.
    """

    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._cache = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def __call__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        compile_args = kwargs.pop("compile_args", None)
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._hits += 1
            return cached

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._hits += 1
                return cached

            self._misses += 1
            if compile_args is None:
                compiled = self._fn(*args, **kwargs)
            else:
                compiled = self._fn(*args, compile_args=compile_args, **kwargs)
            self._cache[cache_key] = compiled
            return compiled

    def cache_clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def cache_info(self) -> CacheInfo:
        with self._lock:
            return CacheInfo(self._hits, self._misses, len(self._cache))


def jit_cache(fn):
    """Decorate a FlyDSL compile helper with the cache described above."""

    return _JitCacheWrapper(fn)
