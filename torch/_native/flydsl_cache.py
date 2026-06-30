"""In-process specialization cache for FlyDSL native-op compile wrappers.

FlyDSL already owns the heavy compiler artifact cache, including its persistent
on-disk entries. PyTorch still benefits from a tiny native-op level cache: it
keeps the ``flyc.compile(...)`` result for a specialization such as
``(hidden_size, dtype, arch, backend)`` so repeated operator calls can skip
rebuilding the launcher and re-entering FlyDSL's compile path.

This mirrors the call shape of Quack/CuteDSL's ``@jit_cache`` while deliberately
not copying its persistent ``.o`` cache behavior.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import functools
from collections import namedtuple
from threading import Lock


CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])


class _JitCacheWrapper:
    """Cache a compile function whose explicit arguments are specialization keys.

    ``compile_args`` is a reserved keyword for cache-miss-only sample inputs such
    as tensors or streams that ``flyc.compile`` needs to infer ABI metadata.
    Those values are intentionally excluded from the cache key.
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
    """Decorate a FlyDSL compile helper using its explicit args as the key.

    The decorated function should take stable specialization parameters as its
    normal arguments. Runtime sample objects can be passed by callers through the
    reserved ``compile_args=...`` keyword; they are forwarded only on cache miss
    and do not participate in keying.
    """

    return _JitCacheWrapper(fn)
