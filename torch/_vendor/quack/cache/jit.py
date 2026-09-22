# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Persistent ``.o`` cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files (``.o``) via ``export_to_c``. On
subsequent runs the ``.o`` is loaded via tvm_ffi (~1 ms) instead of
re-generating IR + re-JIT'ing (~500 ms per kernel).

Runtime config (``CACHE_ENABLED``, ``CACHE_DIR``, ``EXTRA_SOURCE_DIRS``)
lives in :mod:`quack.cache` (the package init).

When an async compile pool is active (see :mod:`quack.cache.async_compile`),
a cold miss is shipped to a CPU worker and :class:`CompilePending` is raised
instead of compiling in-process; the caller (pytest defer loop, autotune
bench loop) retries once the ``.o`` lands.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import os
import pickle
import sys
import tempfile
import time
import warnings
from collections import namedtuple
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

# `quack.cache` (the package itself) holds the mutable runtime flags as a
# single source of truth; reads happen via attribute access on `_state` so we
# always see the live value, not a snapshot taken at module import.
import torch._vendor.quack.cache as _state  # noqa: E402  (intentional partial-import; see __init__.py)


EXPORT_FUNC_NAME = "func"
LOCK_TIMEOUT = 60
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def get_cache_path() -> Path:
    cache_root = _state.get_cache_dir()
    if cache_root is not None:
        cache_dir = Path(cache_root)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "torch_vendor_quack_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _hash_source_dir(h, root: Path) -> None:
    """Hash all Python sources under *root* into *h*."""
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash quack + extra source dirs plus runtime ABI stamps into a fingerprint."""
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    # Hash the entire `quack` package, not just `quack/cache/`. Resolving via
    # the top-level package import keeps the fingerprint stable regardless of
    # where inside the package this file lives.
    import torch._vendor.quack as _quack

    _hash_source_dir(h, Path(_quack.__file__).resolve().parent)
    for extra_dir in _state.EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())
    return h.hexdigest()


def _key_to_hash(key: tuple) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory file lock using fcntl.flock with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# JIT cache decorator
# ---------------------------------------------------------------------------


def jit_cache(fn):
    """Decorator that caches compiled CuTe DSL kernels in-memory and on disk.

    The decorated function should return a compiled kernel (i.e. call cute.compile).
    The disk cache key is (fn.__qualname__, *args, **sorted_kwargs).

    Concurrency model
    -----------------
    The disk side uses a per-key ``{sha}.lock`` file (advisory ``flock``):

    * **Fast path (warm cache).** If the ``.o`` file already exists, we take a
      shared lock just long enough to ``load_module`` it. Many readers can
      proceed concurrently.
    * **Slow path (cold cache).** The actual ``fn(*args, **kwargs)`` compile
      runs *under* the exclusive lock. This serializes redundant compilations
      of the same key across xdist workers / processes: if N processes race
      on a cold key, only one calls ``cute.compile``; the rest wait for the
      lock, see the ``.o`` appear, and load it. (Previously the compile ran
      *between* the shared-lock check and the exclusive-lock export, so all
      N processes wasted CPU compiling the same key in parallel — wall time
      was unchanged but compile-CPU pressure scaled with concurrency, which
      starved other compiles when many keys were cold at once.)

    The lock is per-key, so distinct keys never contend with each other.
    """
    cache = {}
    hits = 0
    misses = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args

        # Snapshot once per call so a concurrent flip of ``_state.CACHE_ENABLED``
        # mid-call can't desync the disk-path branch.
        enabled = _state.CACHE_ENABLED

        # 1. In-memory hit. Same process already compiled or loaded this key.
        if cache_key in cache:
            hits += 1
            return cache[cache_key]

        # 2. Cache disabled: pure in-process compile, no disk side effects.
        if not enabled:
            misses += 1
            compiled_fn = fn(*args, **kwargs)
            cache[cache_key] = compiled_fn
            return compiled_fn

        sha = _key_to_hash((fn.__qualname__,) + cache_key)
        cache_path = get_cache_path() / _compute_source_fingerprint()
        cache_path.mkdir(parents=True, exist_ok=True)
        o_path = cache_path / f"{sha}.o"
        lock_path = cache_path / f"{sha}.lock"

        def _load_cached() -> object:
            """Load the .o into a callable; caller guarantees existence."""
            m = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
            return m[EXPORT_FUNC_NAME]

        def _quarantine_corrupt(exc: Exception) -> None:
            """A cached .o that fails to load (truncated write from a killed
            worker, missing __tvm_ffi_func, ...) is a cache miss, not an error:
            delete it so this and future processes recompile instead of failing
            forever (the CI cache persists across runs)."""
            print(
                f"quack cache: corrupt cached object for key {sha} "
                f"({type(exc).__name__}: {exc}); deleting and recompiling"
            )
            try:
                o_path.unlink()
            except OSError:
                pass

        # 3. Fast path: optimistic existence check, then shared-lock load.
        #    The unlocked ``.exists()`` is a no-cost short-circuit for warm
        #    caches; the shared lock guards against reading a partial file
        #    while a concurrent writer holds the exclusive lock.
        if o_path.exists():
            try:
                with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                    if o_path.exists():
                        try:
                            loaded = _load_cached()
                        except Exception as e:
                            # Corrupt entry: recover under the exclusive lock in
                            # the slow path (shared lock can't safely delete).
                            _quarantine_corrupt(e)
                        else:
                            cache[cache_key] = loaded
                            hits += 1
                            return loaded
            except RuntimeError:
                pass  # lock timeout; fall through to slow path

        # 3b. Async-compile pool: on a cold miss with a pool
        #     active, ship the key to a CPU subprocess and raise
        #     CompilePending instead of compiling in-process. The test runner
        #     defers the test and retries once the worker has exported the
        #     .o. Pool failures fall through to the in-process compile below
        #     so the real exception surfaces with a local traceback.
        from torch._vendor.quack.cache import async_compile as _async

        pool = _async.get_active_pool()
        if pool is not None:
            state, err = pool.poll(sha)
            if state == "new":
                # If another process (e.g. a different xdist worker's pool)
                # holds the exclusive per-key flock, it is compiling this key
                # right now: defer on it instead of submitting a duplicate.
                if _async._flock_held_exclusively(str(lock_path)):
                    pool.mark_external(sha, str(o_path), str(lock_path))
                    raise _async.CompilePending(sha, fn.__qualname__)
                if pool.submit(sha, fn, args, kwargs, o_path):
                    raise _async.CompilePending(sha, fn.__qualname__)
                # unpicklable key / <locals> qualname: compile in-process
            elif state == "pending":
                raise _async.CompilePending(sha, fn.__qualname__)
            elif state == "done":
                try:
                    with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                        if o_path.exists():
                            try:
                                loaded = _load_cached()
                            except Exception as e:
                                _quarantine_corrupt(e)
                            else:
                                cache[cache_key] = loaded
                                hits += 1
                                return loaded
                except RuntimeError:
                    pass  # lock timeout; fall through to slow path
            else:  # "failed"
                # warnings.warn, not print: this fires inside a test whose stdout
                # pytest captures and discards on pass (the in-process fallback
                # usually succeeds), so a print never reaches the user — the
                # warning lands in pytest's warnings summary instead.
                warnings.warn(
                    f"quack cache: async compile failed for {fn.__qualname__} "
                    f"[{sha[:12]}]: {err}; recompiling in-process for a real traceback",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # 4. Slow path: take EXCLUSIVE lock and compile under it. The recheck
        #    inside the lock catches the race where another process compiled
        #    while we were waiting; in that case we just load and return
        #    without duplicating the compile.
        try:
            lock = FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT)
            # Acquire outside the compile's try scope: only the acquisition
            # raises the timeout RuntimeError. A blanket `try: with lock: fn()`
            # also caught RuntimeErrors from the compile itself, mislabeling
            # real compile failures as lock timeouts and re-running the failed
            # compile a second time.
            lock.__enter__()
        except RuntimeError as e:
            # Lock acquisition timed out (heavy contention or stuck holder).
            # Fall back to in-process compile, no disk write. Better to do
            # the work twice than to fail the test.
            warnings.warn(
                f"quack cache: lock timeout for key {sha}: {e}; "
                f"falling back to in-process compile without disk cache",
                RuntimeWarning,
                stacklevel=2,
            )
            misses += 1
            compiled_fn = fn(*args, **kwargs)
            cache[cache_key] = compiled_fn
            return compiled_fn
        try:
            if o_path.exists():
                try:
                    loaded = _load_cached()
                except Exception as e:
                    _quarantine_corrupt(e)  # holds the exclusive lock: safe
                else:
                    cache[cache_key] = loaded
                    hits += 1
                    return loaded

            misses += 1
            compiled_fn = fn(*args, **kwargs)
            # Export to a private temp file, then atomically rename into
            # place: a process killed mid-export (xdist worker OOM-kill,
            # timeout) must never leave a truncated .o at the final path —
            # the advisory flock dies with the process, and a persistent
            # cache (CI keeps one in $HOME) would then fail every future
            # run on this key with "Symbols not found: __tvm_ffi_func".
            tmp_path = o_path.with_suffix(f".o.tmp.{os.getpid()}")
            try:
                compiled_fn.export_to_c(
                    object_file_path=str(tmp_path),
                    function_name=EXPORT_FUNC_NAME,
                )
                os.replace(tmp_path, o_path)
            except Exception as e:
                warnings.warn(
                    f"quack cache: export failed for key {sha}: {e} "
                    f"(this key will recompile every run)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            cache[cache_key] = compiled_fn
            return compiled_fn
        finally:
            lock.__exit__(None, None, None)

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_info():
        return CacheInfo(hits=hits, misses=misses, maxsize=None, currsize=len(cache))

    wrapper.cache = cache
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper
