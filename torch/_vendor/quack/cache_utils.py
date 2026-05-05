# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Persistent .o cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files (.o) via export_to_c.
On subsequent runs the .o is loaded via tvm_ffi (~1ms) instead of
re-generating IR + re-JIT'ing (~100ms per kernel).

Controls:
  QUACK_CACHE_ENABLED=0       — disable persistent .o cache (default: enabled)
  QUACK_CACHE_DIR=path        — override default cache directory
"""

import fcntl
import functools
import hashlib
import os
import pickle
import sys
import tempfile
import time
from collections import namedtuple
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: str | None = os.getenv("QUACK_CACHE_DIR", None)
COMPILE_ONLY: bool = False

# Downstream projects can append directories here to include their sources
# in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: list[Path] = []

EXPORT_FUNC_NAME = "func"
LOCK_TIMEOUT = 60
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def _noop_kernel(*args, **kwargs):
    pass


def get_cache_path() -> Path:
    if CACHE_DIR is not None:
        cache_dir = Path(CACHE_DIR)
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
    _hash_source_dir(h, Path(__file__).resolve().parent)
    for extra_dir in EXTRA_SOURCE_DIRS:
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
    """
    cache = {}
    hits = 0
    misses = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args

        # 1. In-memory hit
        if cache_key in cache:
            hits += 1
            return _noop_kernel if COMPILE_ONLY else cache[cache_key]

        # 2. Disk hit
        disk_key = (fn.__qualname__,) + cache_key
        if CACHE_ENABLED:
            sha = _key_to_hash(disk_key)
            cache_path = get_cache_path() / _compute_source_fingerprint()
            cache_path.mkdir(parents=True, exist_ok=True)
            o_path = cache_path / f"{sha}.o"
            lock_path = cache_path / f"{sha}.lock"
            try:
                with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                    if o_path.exists():
                        m = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
                        loaded = m[EXPORT_FUNC_NAME]
                        cache[cache_key] = loaded
                        hits += 1
                        return _noop_kernel if COMPILE_ONLY else loaded
            except RuntimeError:
                pass

        # 3. Compile
        misses += 1
        compiled_fn = fn(*args, **kwargs)

        # 4. Store
        cache[cache_key] = compiled_fn
        if CACHE_ENABLED:
            try:
                with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
                    if not o_path.exists():
                        o_path.parent.mkdir(parents=True, exist_ok=True)
                        compiled_fn.export_to_c(
                            object_file_path=str(o_path),
                            function_name=EXPORT_FUNC_NAME,
                        )
            except Exception as e:
                print(f"torch._vendor.quack cache: export failed for key {sha}: {e}")

        return _noop_kernel if COMPILE_ONLY else compiled_fn

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
