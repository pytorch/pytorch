# Manage Ahead-of-Time (AOT) compiled kernels
import fcntl
import hashlib
import os
import pickle
import sys
import tempfile
import time
from functools import lru_cache
from getpass import getuser
from pathlib import Path
from typing import Hashable, TypeAlias

import ctypes

import cutlass
import cutlass.cute as cute
import tvm_ffi
from cutlass.cutlass_dsl import JitCompiledFunction
from .fa_logging import fa_log

# Pre-load cute DSL runtime libraries with RTLD_GLOBAL so that their symbols
# (e.g. _cudaLibraryLoadData) are visible to .so modules loaded later via dlopen.
# Upstream cute.runtime.load_module loads these without RTLD_GLOBAL, which causes
# "undefined symbol" errors when loading cached kernels from disk.
for _lib_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
    if Path(_lib_path).exists():
        ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = JitCompiledFunction | tvm_ffi.Function

# Enable cache via `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`
CUTE_DSL_CACHE_ENABLED: bool = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0") == "1"


# Customize cache dir via `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`, default is
# `/tmp/${USER}/flash_attention_cute_dsl_cache``
CUTE_DSL_CACHE_DIR: str | None = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", None)


def get_cache_path() -> Path:
    if CUTE_DSL_CACHE_DIR is not None:
        cache_dir = Path(CUTE_DSL_CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "flash_attention_cute_dsl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """
    Hash all CuTe Python sources plus runtime ABI stamps into a short fingerprint.

    The fingerprint changes whenever:
    - Any .py file under flash_attn/cute is added, removed, renamed, or modified.
    - The Python minor version changes (e.g. 3.13 -> 3.14).
    - The cutlass or tvm_ffi package version changes.

    Computed once per process and cached.
    """
    cute_root = Path(__file__).resolve().parent
    h = hashlib.sha256()

    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())

    for src in sorted(cute_root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(cute_root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)

    return h.hexdigest()


class FileLock:
    """Context manager for advisory file locks using fcntl.flock.

    Supports exclusive (write) and shared (read) locks.
    Always blocks with polling until the lock is acquired or timeout is reached.

    Usage:
        with FileLock(lock_path, exclusive=True, timeout=15, label="abc"):
            # do work under lock
    """

    def __init__(
        self,
        lock_path: Path,
        exclusive: bool,
        timeout: float = 15,
        label: str = "",
    ):
        """
        Args:
            lock_path: Path to the lock file on disk.
            exclusive: True for exclusive (write) lock, False for shared (read) lock.
            timeout: Max seconds to wait for lock acquisition before raising RuntimeError.
            label: Optional human-readable label for error messages.
        """
        self.lock_path: Path = lock_path
        self.exclusive: bool = exclusive
        self.timeout: float = timeout
        self.label: str = label
        self._fd: int = -1

    @property
    def _lock_label(self) -> str:
        kind = "exclusive" if self.exclusive else "shared"
        return f"{kind} {self.label}" if self.label else kind

    def __enter__(self) -> "FileLock":
        open_flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH

        self._fd = os.open(str(self.lock_path), open_flags)

        deadline = time.monotonic() + self.timeout
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.1)
        if not acquired:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(
                f"Timed out after {self.timeout}s waiting for "
                f"{self._lock_label} lock: {self.lock_path}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class JITCache:
    """
    In-memory cache for compiled functions.
    """

    def __init__(self):
        self.cache: dict[CompileKeyType, CallableFunction] = {}

    def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        self.cache[key] = fn

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        return self.cache[key]

    def __contains__(self, key: CompileKeyType) -> bool:
        return key in self.cache

    def clear(self) -> None:
        """
        Clear in-memory cache of compiled functions
        """
        self.cache.clear()


class JITPersistentCache(JITCache):
    """
    In-memory cache for compiled functions, which is also backed by persistent storage.
    Use cutedsl ahead-of-time (AOT) compilation, only supporting enable_tvm_ffi=True
    """

    EXPORT_FUNCTION_PREFIX = "func"
    LOCK_TIMEOUT_SECONDS = 15

    def __init__(self, cache_path: Path):
        super().__init__()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = cache_path

    def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        JITCache.__setitem__(self, key, fn)
        self._try_export_to_storage(key, fn)

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        # Use __contains__ to try populating in-memory cache with persistent storage
        self.__contains__(key)
        return JITCache.__getitem__(self, key)

    def __contains__(self, key: CompileKeyType) -> bool:
        # Checks in-memory cache first, then tries loading from storage.
        # When returning True, guarantees the in-memory cache is populated.
        if JITCache.__contains__(self, key):
            return True
        return self._try_load_from_storage(key)

    def _try_load_from_storage(self, key: CompileKeyType) -> bool:
        """
        Try to load a function from persistent storage into in-memory cache.
        Returns True if loaded successfully, False if not found on disk.
        Holds a shared lock during loading to prevent concurrent writes.
        """
        sha256_hex = self._key_to_hash(key)
        obj_path = self.cache_path / f"{sha256_hex}.o"
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=False,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            if obj_path.exists():
                fa_log(1, f"Loading compiled function from disk: {obj_path}")
                m = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
                fn = getattr(m, self.EXPORT_FUNCTION_PREFIX)
                JITCache.__setitem__(self, key, fn)
                return True
            else:
                fa_log(1, f"Cache miss on disk for key hash {sha256_hex}")
        return False

    def _try_export_to_storage(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        """Export a compiled function to persistent storage under exclusive lock."""
        sha256_hex = self._key_to_hash(key)
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=True,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            obj_path = self.cache_path / f"{sha256_hex}.o"
            if obj_path.exists():
                # Another process already exported.
                fa_log(1, f"Skipping export, already on disk: {obj_path}")
                return
            fa_log(1, f"Exporting compiled function to disk: {obj_path}")
            fn.export_to_c(
                object_file_path=str(obj_path),
                function_name=self.EXPORT_FUNCTION_PREFIX,
            )
            fa_log(1, f"Successfully exported compiled function to disk: {obj_path}")

    def _key_to_hash(self, key: CompileKeyType) -> str:
        return hashlib.sha256(pickle.dumps(key)).hexdigest()

    def _lock_path(self, sha256_hex: str) -> Path:
        return self.cache_path / f"{sha256_hex}.lock"

    def clear(self) -> None:
        """
        Not only clear the in-memory cache. Also purge persistent compilation cache.
        """
        fa_log(1, f"Clearing persistent cache at {self.cache_path}")
        super().clear()
        for child in self.cache_path.iterdir():
            child.unlink()


def get_jit_cache(name: str | None = None) -> JITCache:
    """
    JIT cache factory.
    `name` is an optional identifier to create subdirectories to manage cache.

    When persistent caching is enabled, artifacts are namespaced under a
    source fingerprint directory so that code or dependency changes
    automatically invalidate stale entries.
    """
    if CUTE_DSL_CACHE_ENABLED:
        path = get_cache_path() / _compute_source_fingerprint()
        if name:
            path = path / name
        fa_log(1, f"Creating persistent JIT cache at {path}")
        return JITPersistentCache(path)
    else:
        fa_log(1, "Persistent cache disabled, using in-memory JIT cache")
        return JITCache()
