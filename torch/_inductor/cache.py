from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from ast import literal_eval
from contextlib import contextmanager
from functools import cached_property
from hashlib import sha256
from os import getenv
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from threading import Lock
from typing import Any, Generic, Iterator, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, override, Self

from torch.utils._filelock import FileLock


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor


# TypeVars can't be recursive, so generic types that fall within
# Key or Value can't be bound properly; for example, Key should
# only take tuples of other Key types: tuple[Key, ...]. this is
# a known shortcoming of torch's typing
Key = TypeVar("Key", str, int, tuple[Any, ...])
Value = TypeVar("Value", str, int, tuple[Any, ...], bytes, dict[Any, Any], list[Any])

# controls the global fresh cache context index; entering a global
# fresh cache context triggers an increment of GLOBAL_FC_IDX; exiting
# a global fresh cache context triggers a decrement of GLOBAL_FC_IDX
GLOBAL_FC_IDX: int = 0
GLOBAL_FC_IDX_LOCK: Lock = Lock()
# it is particularly important for global fresh cache context that we
# register all cache instances, such that when we exit a global fresh
# cache context we can iterate through those instances and update their
# fc idx => cache mapping (i.e. remove the now unused fresh cache)
GLOBAL_CACHE_REGISTRY: list["Cache"[Any, Any]] = []
GLOBAL_CACHE_REGISTRY_LOCK: Lock = Lock()

@contextmanager
def with_fresh_cache() -> Iterator[None]:
    """
    Context manager for a global fresh cache context.
    Increments the global fresh cache index upon entry and decrements it upon exit.
    Ensures that all cache instances are updated and unused fresh cache instances
    are removed from the global registry when the context is exited.
    """
    global GLOBAL_FC_IDX
    with GLOBAL_FC_IDX_LOCK:
        GLOBAL_FC_IDX += 1
    try:
        yield
    finally:
        with GLOBAL_FC_IDX_LOCK, GLOBAL_CACHE_REGISTRY_LOCK:
            # we need to 1) remove unreferenced fresh cache
            # instances from all cache instances and 2) update
            # the global cache registry to remove the now deleted
            # fresh cache instances
            _fc_to_del: list["Cache"[Any, Any]] = []
            for cache in GLOBAL_CACHE_REGISTRY:
                if cache._fc_idx not in cache._fc_idx_to_cache:
                    continue
                _fc = cache._fc_idx_to_cache[cache._fc_idx]
                del cache._fc_idx_to_cache[cache._fc_idx]
                _fc_to_del.append(_fc)
            for _fc in _fc_to_del:
                try:
                    GLOBAL_CACHE_REGISTRY.remove(_fc)
                except ValueError:
                    pass
                del _fc
            GLOBAL_FC_IDX -= 1


class CacheError(ValueError):
    """
    Exception raised for errors encountered during cache operations.
    """


class Cache(ABC, Generic[Key, Value]):
    """
    Abstract base class for cache implementations.
    Provides the interface for cache operations, including support for
    fresh cache contexts and thread safety.
    """

    def __new__(cls) -> Self:
        """
        Create a new cache instance and register it in the global cache registry.
        Returns:
            Self: The newly created cache instance.
        """
        global GLOBAL_CACHE_REGISTRY
        self = super().__new__(cls)
        with GLOBAL_CACHE_REGISTRY_LOCK:
            GLOBAL_CACHE_REGISTRY.append(self)
        return self
    
    def __del__(self: Self) -> None:
        """
        Remove the cache instance from the global cache registry upon deletion.
        """
        global GLOBAL_CACHE_REGISTRY
        with GLOBAL_CACHE_REGISTRY_LOCK:
            try:
                GLOBAL_CACHE_REGISTRY.remove(self)
            except ValueError:
                pass

    def __init__(self: Self) -> None:
        """
        Initialize the cache instance with thread lock and fresh cache context tracking.
        """
        self._lock = Lock()
        # is similar to GLOBAL_FC_IDX, but tracks the fresh
        # cache context index for per-cache fresh cache contexts
        self._local_fc_idx = 0
        # overall fresh cache index is calculated as a tuple
        # of the local and global fresh cache indices; using a
        # tuple allows arbitrary depth nesting of both local
        # and global fresh cache contexts. if we simply used
        # local + global as the fresh cache index we could have
        # surprising bugs, for example local=1, global=0 and
        # local=0, global=1 would resolve to the same cache instance
        # which would be decidedly incorrect
        self._fc_idx_to_cache = {
            self._fc_idx: self
        }
    
    def _construct_fc(self: Self) -> Self:
        """
        Construct a new fresh cache instance.
        Subclasses may override this method to provide custom behavior
        for fresh cache instantiation.
        Returns:
            Self: A new fresh cache instance.
        """
        return type(self)()
    
    @property
    def _fc_idx(self: Self) -> tuple[int, int]:
        """
        Get the current fresh cache index as a tuple of local and global indices.
        Returns:
            tuple[int, int]: The (local, global) fresh cache index.
        """
        return (self._local_fc_idx, GLOBAL_FC_IDX)
    
    def _get_fc(self: Self) -> Self:
        """
        Retrieve the correct fresh cache instance for the current context.
        Returns:
            Self: The cache instance for the current fresh cache context.
        """
        _cache = self._fc_idx_to_cache.get(self._fc_idx, None)
        if _cache is None:
            _cache = self._construct_fc()
            self._fc_idx_to_cache[self._fc_idx] = _cache
        return _cache
    
    def _on_fresh_cache_exit(self: Self) -> None:
        """
        Hook called when exiting a fresh cache context.
        Subclasses may override to perform cleanup.
        """

    @abstractmethod
    def _get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache (to be implemented by subclasses).
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present, else None.
        """

    @abstractmethod
    def _insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache (to be implemented by subclasses).
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        """

    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache. Entry point for typical users,
        and goes through various checks related to the fresh cache context
        for example.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present, else None.
        """
        with self._lock:
            _cache = self._get_fc()
            return _cache._get(key)

    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache. Entry point for typical users,
        and goes through various checks related to the fresh cache context
        for example.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        """
        with self._lock:
            _cache = self._get_fc()
            return _cache._insert(key, value)

    @contextmanager
    def with_fresh_cache(self: Self) -> Iterator[None]:
        """
        Context manager for a per-cache fresh cache context.
        Increments the local fresh cache index upon entry and decrements it upon exit.
        Ensures that unused fresh cache instances are removed from the registry.
        """
        with self._lock:
            self._local_fc_idx += 1
        try:
            yield
        finally:
            with self._lock:
                self._on_fresh_cache_exit()
                if self._fc_idx not in self._fc_idx_to_cache:
                    # and don't forget to decrement the local
                    # fresh cache context idx on early exit
                    self._local_fc_idx -= 1
                    return
                global GLOBAL_CACHE_REGISTRY
                # we should delete now out-of-context fresh
                # cache instances, remove their entry in our
                # fresh cache instance mapping, and their
                # reference in the global cache registry
                _fc = self._fc_idx_to_cache[self._fc_idx]
                del self._fc_idx_to_cache[self._fc_idx]
                try:
                    with GLOBAL_CACHE_REGISTRY_LOCK:
                        GLOBAL_CACHE_REGISTRY.remove(_fc)
                except ValueError:
                    pass
                del _fc
                self._local_fc_idx -= 1


class InMemoryCache(Cache[Key, Value]):
    """
    In-memory cache implementation using a dictionary and thread lock.
    """

    def __init__(self: Self) -> None:
        """
        Initialize an empty in-memory cache.
        """
        super().__init__()
        self._cache: dict[Key, Value] = {}

    @override
    def _get(self: Self, key: Key) -> Value | None:
        if (value := self._cache.get(key)) is not None:
            return value
        return None

    @override
    def _insert(self: Self, key: Key, value: Value) -> bool:
        if key in self._cache:
            # no overwrites for insert!
            return False
        self._cache[key] = value
        return True

    @classmethod
    def from_env_var(cls, env_var: str) -> Self:
        """
        Create an in-memory cache from an environment variable.
        Args:
            env_var (str): Name of the environment variable containing cache data.
        Returns:
            InMemoryCache: An instance populated from the environment variable.
        Raises:
            CacheError: If the environment variable is malformed or contains invalid data.
        """
        cache = cls()

        if (env_val := getenv(env_var)) is None:
            # env_var doesn't exist = empty cache
            return cache

        for kv_pair in env_val.split(";"):
            # ignore whitespace prefix/suffix
            kv_pair = kv_pair.strip()

            if not kv_pair:
                # kv_pair could be '' if env_val is '' or has ; suffix
                continue

            try:
                # keys and values should be comma separated
                key_bytes_repr, value_bytes_repr = kv_pair.split(",", 1)
            except ValueError as err:
                raise CacheError(
                    f"Malformed kv_pair {kv_pair!r} from env_var {env_var!r}, likely missing comma separator."
                ) from err

            # ignore whitespace prefix/suffix, again
            key_bytes_repr, value_bytes_repr = (
                key_bytes_repr.strip(),
                value_bytes_repr.strip(),
            )

            try:
                # check that key_bytes_str is an actual, legitimate encoding
                key_bytes = literal_eval(key_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err
            try:
                # check that value_bytes_str is an actual, legitimate encoding
                value_bytes = literal_eval(value_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err

            try:
                key = pickle.loads(key_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err
            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err

            # true duplicates, i.e. multiple occurrences of the same key => value
            # mapping are ok and treated as a no-op; key duplicates with differing
            # values, i.e. key => value_1 and key => value_2 where value_1 != value_2,
            # are not okay since we don't allow overwriting cached values (it's bad regardless)
            if (not cache.insert(key, value)) and (cache.get(key) != value):
                raise CacheError(
                    f"Multiple values for key {key!r} found, got {cache.get(key)!r} and {value!r}."
                )

        return cache

    @classmethod
    def from_file_path(cls, fpath: Path) -> Self:
        """
        Create an in-memory cache from a file path.
        Args:
            fpath (Path): Path to the file containing pickled cache data.
        Returns:
            InMemoryCache: An instance populated from the file.
        Raises:
            CacheError: If the file is not a valid pickled dictionary.
        """
        cache = cls()

        if not fpath.is_file():
            # fpath doesn't exit = empty cache
            return cache

        try:
            with open(fpath, "rb") as fp:
                cache._cache = pickle.load(fp)
        except pickle.UnpicklingError as err:
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents are un-pickle-able."
            ) from err

        if not isinstance(cache._cache, dict):
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents not pickled dict[Key, Value]."
            )

        return cache


class AsyncCache(Cache[Key, Value]):
    """
    Asynchronous cache implementation using ThreadPoolExecutor.
    """

    def _get_async(
        self: Self, key: Key, executor: ThreadPoolExecutor
    ) -> Future[Value | None]:
        """
        Retrieve a value from the cache asynchronously.
        Args:
            key (Key): The key to look up.
            executor (ThreadPoolExecutor): Executor for async execution.
        Returns:
            Future[Value | None]: Future for the cached value or None.
        """
        # careful here, we definitely want to call self._get;
        # if we instead were to call self.get, when the thread
        # executed the method we'd check the fc again which could
        # cause weird issues. for example, if you were to enter a
        # fresh context and call self.get_async before immediately
        # exiting the context it could theoretically be possible
        # that the context closure happens before the thread has
        # finished processing the request which could act as the
        # self.get_async call having happened outside of the context
        return executor.submit(self._get, key)

    def _insert_async(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor
    ) -> Future[bool]:
        """
        Insert a value into the cache asynchronously.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
            executor (ThreadPoolExecutor): Executor for async execution.
        Returns:
            Future[bool]: Future for the result of insertion.
        """
        # see self._get_async for why we use self._insert
        return executor.submit(self._insert, key, value)

    def get_async(
        self: Self, key: Key, executor: ThreadPoolExecutor
    ) -> Future[Value | None]:
        with self._lock:
            _cache = self._get_fc()
            return _cache._get_async(key, executor)

    def insert_async(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor
    ) -> Future[bool]:
        with self._lock:
            _cache = self._get_fc()
            return _cache._insert_async(key, value, executor)


class OnDiskCache(AsyncCache[Key, Value]):
    """
    On-disk cache implementation using files and file locks.
    Stores cache data in files on disk, with atomic operations and versioning.
    Supports custom cache directory names.
    Attributes:
        version (int): The version used for cache versioning.
        name (str): The name of the cache directory.
    """

    version: int = 0

    def __init__(self: Self, name: str | None = None) -> None:
        """
        Initialize an on-disk cache instance.
        Args:
            name (str | None, optional): The name of the cache directory. If None,
                defaults to "on_disk_cache".
        """
        super().__init__()
        self.name = name or "on_disk_cache"

    def _construct_fc(self: Self) -> Self:
        """
        Construct a new fresh cache instance for on-disk cache.
        Returns:
            Self: A new fresh cache instance with a subdirectory for the context.
        """
        _cache = type(self)()
        _cache._base_dir = self._base_dir / str(self._fc_idx)
        return _cache

    def _on_fresh_cache_exit(self: Self) -> None:
        """
        Delete the contents of the fresh cache on fresh cache
        context exit.
        """
        _cache = self._get_fc()
        rmtree(_cache._base_dir)
    
    @cached_property
    def _base_dir(self: Self) -> Path:
        """
        Get the base directory for the cache.
        Returns:
            Path: The base directory path for storing cache files.
        """
        return Path(gettempdir()) / "cache" / self.name

    def _fpath_from_key(self: Self, key: Key) -> Path:
        """
        Get the file path for a given key.
        Args:
            key (Key): The key to convert to a file path.
        Returns:
            Path: The file path for the key.
        Raises:
            CacheError: If the key is not pickle-able.
        """
        try:
            return self._base_dir / sha256(pickle.dumps(key)).hexdigest()[:32]
        except (AttributeError, pickle.PicklingError) as err:
            raise CacheError(
                f"Failed to get fpath for key {key!r}, key is not pickle-able."
            ) from err
        assert_never(key)

    def _flock_from_fpath(self: Self, fpath: Path) -> FileLock:
        """
        Get a file lock for a given file path.
        Args:
            fpath (Path): The file path.
        Returns:
            FileLock: The file lock for the path.
        """
        # fpath.name is a hex digest, meaning there are 16^4 potential values
        # for fpath.name[:4]; this is more than enough unique locks to not
        # cause additional overhead from shared locks and it also saves our
        # cache dir from becoming 50 percent locks
        return FileLock(str(fpath.parent / "locks" / fpath.name[:4]) + ".lock")

    @property
    def _version_prefix(self: Self) -> bytes:
        """
        Get the version prefix for the cache.
        Returns:
            bytes: The version prefix as bytes, derived from the cache version string.
        """
        return sha256(str(OnDiskCache.version).encode()).digest()[:4]

    @override
    def _get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present and version matches, else None.
        Raises:
            CacheError: If the value is corrupted or cannot be unpickled.
        Side Effects:
            Removes stale cache files if the version prefix does not match.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)

        with flock:
            if not fpath.is_file():
                return None

            value_bytes = None
            prefix_length = len(self._version_prefix)
            with open(fpath, "rb") as fp:
                if fp.read(prefix_length) == self._version_prefix:
                    value_bytes = fp.read()

            if value_bytes is None:
                # _version_prefix did not match, so we can't read the stale
                # cached value; we should also remove the stale cached value,
                # so that key can be re-cached by the newer version
                fpath.unlink()
                return None

            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Failed to get key {key!r}, value is potentially corrupted (value is not un-pickle-able)."
                ) from err

            return value

    @override
    def _insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        Raises:
            CacheError: If the value is not pickle-able.
        Side Effects:
            Creates the cache directory if it does not exist.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" mode is exclusive creation, meaning the file will be created
            # iff the file does not already exist (atomic w/o overwrite); use
            # flock for added atomicity guarantee and to prevent partial writes
            with flock as _, open(fpath, "xb") as fp:
                fp.write(self._version_prefix)
                pickle.dump(value, fp)
        except pickle.PicklingError as err:
            raise CacheError(
                f"Failed to insert key {key!r} with value {value!r}, value is not pickle-able."
            ) from err
        except FileExistsError:
            return False
        return True


class InductorOnDiskCache(OnDiskCache[Key, Value]):
    """
    Inductor-specific on-disk cache implementation.
    Uses a custom base directory for Inductor cache files.
    """

    def __init__(self: Self) -> None:
        """
        Initialize an inductor on-disk cache instance.
        Sets the cache directory name to "inductor_on_disk_cache".
        """
        super().__init__("inductor_on_disk_cache")

    @cached_property
    def _base_dir(self: Self) -> Path:
        """
        Get the base directory for the Inductor cache.
        Returns:
            Path: The base directory path for Inductor cache files.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache", self.name)
