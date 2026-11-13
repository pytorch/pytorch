"""In-process specialization cache for FlyDSL native-op compile wrappers.

FlyDSL already owns the heavy compiler artifact cache, including its persistent
on-disk entries. PyTorch still benefits from a tiny native-op level cache: it
keeps the ``flyc.compile(...)`` result for a specialization such as
``(hidden_size, dtype, arch, backend)`` so repeated operator calls can skip
rebuilding the launcher and re-entering FlyDSL's compile path.

This mirrors the call shape of Quack/CuteDSL's ``@jit_cache`` -- a closure over
the cache, exposing the same ``cache`` / ``cache_clear`` / ``cache_info``
attributes -- while deliberately not copying its persistent ``.o`` cache
behavior.

Entries are never evicted: nothing here bounds the cache except the number of
distinct specializations the caller asks for, and each one holds a compiled
module plus its per-specialization lock alive for the life of the process,
both released only by ``cache_clear()``. That is the right trade for a model
with a fixed hidden size, which is the case these kernels are dispatched for;
a caller that sweeps a specialization parameter over an open range should
``cache_clear()`` rather than expect an eviction policy. Note the corollary for
a decorated method: ``self`` joins the key, so the instance stays alive too.
"""

from __future__ import annotations

import functools
from collections import namedtuple
from threading import Lock
from typing import Any, Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize"])


class CachedCompile(Protocol):
    """Compile wrapper returned by ``flydsl_jit_cache`` / ``instrumented_flydsl_cache``."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def cache_clear(self) -> None: ...

    def cache_info(self) -> CacheInfo: ...


_MISSING = object()
_CacheKey = tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]


def flydsl_jit_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a FlyDSL compile helper using its explicit args as the key.

    The decorated function should take stable specialization parameters as its
    normal arguments. Runtime sample objects can be passed by callers through the
    reserved ``compile_args=...`` keyword; they are forwarded only on cache miss
    and do not participate in keying. Declare it keyword-only: the name is
    stripped from the cache key unconditionally, so a specialization parameter
    that happened to be called ``compile_args`` would collapse every value of
    itself onto one entry and hand back a kernel compiled for another.

    Recursive calls back into a cached function are not expected.

    Deliberately not named ``jit_cache``: the instrumentation coverage scan in
    test_instrumentation.py attributes compile sites by decorator name, and that
    name already belongs to CuTeDSL's vendored cache.
    """

    cache: dict[_CacheKey, Any] = {}
    key_locks: dict[_CacheKey, Lock] = {}
    lock = Lock()
    hits = 0
    misses = 0

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal hits, misses
        compile_args = kwargs.pop("compile_args", _MISSING)
        cache_key: _CacheKey = args, tuple(sorted(kwargs.items()))

        # The hit path runs on every operator call, not just every compile, so
        # it stays off the lock. dict.get is atomic; the hit counter is a plain
        # increment and can therefore lose an update under contention, which is
        # acceptable for a diagnostic. Misses are counted under the lock.
        cached = cache.get(cache_key, _MISSING)
        if cached is not _MISSING:
            hits += 1
            return cached

        with lock:
            key_lock = key_locks.get(cache_key)
            if key_lock is None:
                key_lock = Lock()
                key_locks[cache_key] = key_lock

        # One compile per specialization; different specializations still
        # compile concurrently. A thread that waits here is charged for the
        # compiling thread's miss delta, so its instrumented event reads as
        # outcome="compiled" with a wall time that is mostly lock wait.
        with key_lock:
            cached = cache.get(cache_key, _MISSING)
            if cached is not _MISSING:
                hits += 1
                return cached
            with lock:
                misses += 1

            if compile_args is _MISSING:
                compiled = fn(*args, **kwargs)
            else:
                compiled = fn(*args, compile_args=compile_args, **kwargs)

            with lock:
                # make sure we do not populate older compilation after the `cache_clear()`
                #
                # What this deliberately allows: a clear frees the key, so a
                # caller arriving after it takes a fresh lock and compiles
                # alongside the in-flight one. Both get a result; only the last
                # to finish is stored. Callers must therefore not treat the
                # returned object as identity-stable across a cache_clear()
                if key_locks.get(cache_key) is key_lock:
                    cache[cache_key] = compiled
            return compiled

    def cache_clear() -> None:
        nonlocal hits, misses
        with lock:
            cache.clear()
            key_locks.clear()
            hits = 0
            misses = 0

    def cache_info() -> CacheInfo:
        """Return cache statistics.

        ``hits`` is best-effort under concurrent calls because hot cache hits
        do not take the cache lock: an increment can be lost, and one racing a
        ``cache_clear()`` can survive it, leaving hits above zero right after a
        clear. The lock taken here is what makes ``misses`` and ``currsize``
        exact.
        """
        with lock:
            return CacheInfo(hits, misses, len(cache))

    cached: Any = wrapper
    cached.cache = cache
    cached.cache_clear = cache_clear
    cached.cache_info = cache_info
    return cached
