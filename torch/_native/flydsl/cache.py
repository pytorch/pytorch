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
module plus its per-specialization lock alive for the life of the process,
both released only by ``cache_clear()``. That is the right trade for a model
with a fixed hidden size, which is the case these kernels are dispatched for;
a caller that sweeps a specialization parameter over an open range should
``cache_clear()`` rather than expect an eviction policy.
"""

from __future__ import annotations

import functools
from collections import namedtuple
from collections.abc import Callable
from threading import Lock, RLock
from typing import Any


CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])
_MISSING = object()
_CacheKey = tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]


class _JitCacheWrapper:
    """Cache a compile function whose explicit arguments are specialization keys.

    ``compile_args`` is a reserved keyword for cache-miss-only sample inputs such
    as tensors or streams that ``flyc.compile`` needs to infer ABI metadata.
    Those values are intentionally excluded from the cache key.

    The per-key lock is reentrant, so a compile function that calls back into
    its own cache with the same key recurses until the stack runs out: nothing
    is stored until the compile returns, so the reentrant call misses, retakes
    the lock, and compiles again. That is a bug in the compile function either
    way, and a RecursionError names the cycle in its traceback where a plain
    lock would just hang.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._cache: dict[_CacheKey, Any] = {}
        self._lock = Lock()
        self._key_locks: dict[_CacheKey, RLock] = {}
        self._hits = 0
        self._misses = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        compile_args = kwargs.pop("compile_args", _MISSING)
        cache_key: _CacheKey = args, tuple(sorted(kwargs.items()))

        # The hit path runs on every operator call, not just every compile, so
        # it stays off the lock. dict.get is atomic; the hit counter is a plain
        # increment and can therefore lose an update under contention, which is
        # acceptable for a diagnostic. Misses are counted under the lock.
        cached = self._cache.get(cache_key, _MISSING)
        if cached is not _MISSING:
            self._hits += 1
            return cached

        with self._lock:
            key_lock = self._key_locks.get(cache_key)
            if key_lock is None:
                key_lock = RLock()
                self._key_locks[cache_key] = key_lock

        # One compile per specialization; different specializations still
        # compile concurrently. A thread that waits here is charged for the
        # compiling thread's miss delta, so its instrumented event reads as
        # outcome="compiled" with a wall time that is mostly lock wait.
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
                # make sure we do not populate older compilation after the `cache_clear()`
                #
                # What this deliberately allows: a clear frees the key, so a
                # caller arriving after it takes a fresh lock and compiles
                # alongside the in-flight one. Both get a result; only the last
                # to finish is stored. Callers must therefore not treat the
                # returned object as identity-stable across a cache_clear()
                if self._key_locks.get(cache_key) is key_lock:
                    self._cache[cache_key] = compiled
            return compiled

    @property
    def cache(self) -> dict[_CacheKey, Any]:
        # Named to match jit_cache's ``wrapper.cache``: instrument_flydsl_compile
        # forwards all three of cache/cache_clear/cache_info onto the
        # instrumented function, and skips silently what the wrapper lacks.
        return self._cache

    def cache_clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._key_locks.clear()
            self._hits = 0
            self._misses = 0

    def cache_info(self) -> CacheInfo:
        """Return cache statistics.

        ``hits`` is best-effort under concurrent calls because hot cache hits
        do not take the cache lock: an increment can be lost, and one racing a
        ``cache_clear()`` can survive it, leaving hits above zero right after a
        clear. The lock taken here is what makes ``misses`` and ``currsize``
        exact.
        """
        with self._lock:
            return CacheInfo(self._hits, self._misses, len(self._cache))


def flydsl_jit_cache(fn: Callable[..., Any]) -> _JitCacheWrapper:
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
