"""In-process specialization cache for FlyDSL native-op compile wrappers.

FlyDSL already owns the heavy compiler artifact cache, including its persistent
on-disk entries. PyTorch still benefits from a tiny native-op level cache: it
keeps the ``flyc.compile(...)`` result for a specialization such as
``(hidden_size, dtype, arch, backend)`` so repeated operator calls can skip
rebuilding the launcher and re-entering FlyDSL's compile path.

This mirrors the call shape of Quack/CuteDSL's ``@jit_cache`` while deliberately
not copying its persistent ``.o`` cache behavior.

Entries are never evicted: nothing here bounds the cache except the number of
distinct specializations the caller asks for, and each one holds a compiled
module alive for the life of the process. That is the right trade for a model
with a fixed hidden size, which is the case these kernels are dispatched for;
a caller that sweeps a specialization parameter over an open range should
``cache_clear()`` rather than expect an eviction policy.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import functools
from collections import namedtuple
from threading import Lock, RLock


CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])
_MISSING = object()


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
        self._key_locks = {}
        self._hits = 0
        self._misses = 0

    def __call__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        compile_args = kwargs.pop("compile_args", _MISSING)
        cache_key = args, tuple(sorted(kwargs.items()))

        # The hit path runs on every operator call, not just every compile, so
        # it stays off the lock. dict.get is atomic; the hit counter is a plain
        # increment and can therefore lose an update under contention, which is
        # acceptable for a diagnostic. Misses are counted under the lock.
        cached = self._cache.get(cache_key, _MISSING)
        if cached is not _MISSING:
            self._hits += 1
            return cached

        with self._lock:
            key_lock = self._key_locks.setdefault(cache_key, RLock())

        # One compile per specialization; different specializations still
        # compile concurrently.
        with key_lock:
            cached = self._cache.get(cache_key, _MISSING)
            if cached is not _MISSING:
                self._hits += 1
                return cached
            with self._lock:
                self._misses += 1

            if compile_args is _MISSING:
                compiled = self._fn(*args, **kwargs)
            else:
                compiled = self._fn(*args, compile_args=compile_args, **kwargs)

            with self._lock:
                self._cache[cache_key] = compiled
            return compiled

    def cache_clear(self) -> None:
        with self._lock:
            self._cache.clear()
            # The key locks go too, so a compile still running under the old
            # lock no longer excludes the next caller for that key: the two can
            # then compile the same specialization concurrently and both store.
            # Only a clear can produce that, and it beats leaving callers to
            # queue behind a compile whose result the clear was meant to drop.
            self._key_locks.clear()
            self._hits = 0
            self._misses = 0

    def cache_info(self) -> CacheInfo:
        """Return cache statistics.

        ``hits`` is best-effort under concurrent calls because hot cache hits
        do not take the cache lock.
        """
        with self._lock:
            return CacheInfo(self._hits, self._misses, len(self._cache))


def flydsl_jit_cache(fn):
    """Decorate a FlyDSL compile helper using its explicit args as the key.

    The decorated function should take stable specialization parameters as its
    normal arguments. Runtime sample objects can be passed by callers through the
    reserved ``compile_args=...`` keyword; they are forwarded only on cache miss
    and do not participate in keying. Declare it keyword-only: the name is
    stripped from the cache key unconditionally, so a specialization parameter
    that happened to be called ``compile_args`` would collapse every value of
    itself onto one entry and hand back a kernel compiled for another.

    Deliberately not named ``jit_cache``: the instrumentation coverage scan in
    test_instrumentation.py attributes compile sites by decorator name, and that
    name already belongs to CuTeDSL's vendored cache.
    """

    return _JitCacheWrapper(fn)
