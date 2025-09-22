from __future__ import annotations

from functools import cached_property
from os import getenv
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from typing import Generic, TYPE_CHECKING, TypeVar
from typing_extensions import override, Self

from torch.utils._filelock import FileLock


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor


Key = TypeVar("Key")
Value = TypeVar("Value")


class Cache(Generic[Key, Value]):
    """
    Abstract base class for cache implementations.

    Provides the interface for basic synchronous get and insert methods for storing and retrieving data.
    Subclasses must implement both methods.

    Note:
        - Not guaranteed to be thread-safe.
        - For asynchronous and thread-safe cache, see `AsyncCache`.

    Methods:
        get(key): Retrieve a value by key.
        insert(key, value): Insert a value if the key does not already exist.
    """

    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve the value associated with the given key from the cache.

        Args:
            key (Key): The key used to query the cache.

        Returns:
            Value | None: The value associated with the key, or None if not found.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Store the given value in the cache with the associated key if the key does
        not already exist in the cache, otherwise do nothing.

        Args:
            key (Key): The key to associate with the value.
            value (Value): The value to be stored in the cache.

        Returns:
            bool: True if the value was stored successfully, False if the key already exists.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError


class InMemoryCache(Cache[str, bytes]):
    """
    In-memory cache implementation.

    Stores cache data in a dictionary for fast lookups and insertions.
    Not thread-safe.
    """

    def __init__(self: Self) -> None:
        """
        Initialize the in-memory cache.
        """
        self._cache: dict[str, bytes] = {}
        self._lock: Lock = Lock()

    @override
    def get(self: Self, key: str) -> bytes | None:
        """
        Retrieve the value associated with the given key from the cache.

        Args:
            key (str): The key used to query the cache.

        Returns:
            bytes | None: The value associated with the key, or None if not found.
        """
        with self._lock:
            return self._cache.get(key)

    @override
    def insert(self: Self, key: str, value: bytes) -> bool:
        """
        Store the given value in the cache with the associated key if the key does
        not already exist in the cache, otherwise do nothing.

        Args:
            key (str): The key to associate with the value.
            value (bytes): The value to be stored in the cache.

        Returns:
            bool: True if the value was stored successfully, False if the key already exists.
        """
        with self._lock:
            if key in self._cache:
                return False
            self._cache[key] = value
            return True

    @classmethod
    def from_env_var(cls, env_var: str) -> Self:
        """
        Create a new in-memory cache instance from an environment variable.

        The environment variable should contain key-value pairs separated by ';',
        with each pair formatted as 'key,value'. The value should be a string
        representation of bytes (e.g., b'...').

        Args:
            env_var (str): The environment variable containing cache data.

        Returns:
            InMemoryCache: A new in-memory cache instance populated with data from the environment variable.

        Raises:
            ValueError: If a key is associated with two distinct values, or if the environment variable
                is malformed (e.g., missing comma, value not a bytes string).
        """
        cache: Self = cls()
        env_val: str | None = getenv(env_var, None)

        if env_val is not None:
            for kv_pair in env_val.split(";"):
                if not kv_pair:
                    # can happen if env_val is an empty string, or ends with ;
                    continue
                try:
                    key, raw_value = kv_pair.split(",", 1)
                except ValueError as err:
                    raise ValueError(
                        f"Malformed kv_pair {kv_pair!r} in env_var {env_var!r}, missing comma separator!"
                    ) from err
                # check that raw_value is a str repr of bytes
                if (not raw_value.startswith("b'")) or (not raw_value.endswith("'")):
                    raise ValueError(
                        f"Malformed value {raw_value!r} in kv_pair {kv_pair!r}, expected b'...' format!"
                    )
                # remove b' prefix and ' suffix
                str_value = raw_value[2:-1]
                try:
                    # make sure the value is legitimately encoded
                    value = bytes([ord(char) for char in str_value])
                except ValueError as err:
                    raise ValueError(
                        f"Malformed value {raw_value!r} in kv_pair {kv_pair!r}!"
                    ) from err
                # duplicates are ok, so long as the key does not point to two distinct values
                if (not cache.insert(key, value)) and (cache.get(key) != value):
                    raise ValueError(
                        f"Duplicated values for key {key!r}, got {cache.get(key)!r} and {value!r}!"
                    )

        return cache


class AsyncCache(Cache[Key, Value]):
    """
    Abstract base class for asynchronous, thread-safe cache implementations.

    Provides synchronous get/insert methods and additional asynchronous (_async) methods
    for concurrent access using a ThreadPoolExecutor. All methods are thread-safe.

    Note:
        - Use this class or its subclasses when thread safety or async access is required.
        - The _async methods return concurrent.futures.Future objects.

    Methods:
        get(key): Retrieve a value by key.
        get_async(key, executor): Asynchronously retrieve a value by key.
        insert(key, value): Insert a value.
        insert_async(key, value, executor): Asynchronously insert a value.
    """

    def get_async(
        self: Self, key: Key, executor: ThreadPoolExecutor
    ) -> Future[Value | None]:
        """
        Retrieve the value associated with the given key from the cache asynchronously.

        Args:
            key (Key): The key used to query the cache.
            executor (ThreadPoolExecutor): The executor to use for asynchronous execution.

        Returns:
            Future[Value | None]: A Future representing the result of the asynchronous operation.
        """
        return executor.submit(self.get, key)

    def insert_async(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor
    ) -> Future[bool]:
        """
        Store the given value in the cache with the associated key if the key does
        not already exist in the cache, otherwise do nothing, asynchronously.

        Args:
            key (Key): The key to associate with the value.
            value (Value): The value to be stored in the cache.
            executor (ThreadPoolExecutor): The executor to use for asynchronous execution.

        Returns:
            Future[bool]: A Future representing the result of the asynchronous operation.
        """
        return executor.submit(self.insert, key, value)


class OnDiskCache(AsyncCache[str, bytes]):
    """
    Abstract base class for on-disk cache implementations.

    Provides synchronous and asynchronous get/insert methods for storing and retrieving data on disk.
    All methods are thread-safe.

    Methods:
        get(key): Retrieve a value by key from disk.
        get_async(key, executor): Asynchronously retrieve a value by key from disk.
        insert(key, value): Insert a value on disk.
        insert_async(key, value, executor): Asynchronously insert a value on disk.
    """

    @property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the on-disk cache.

        Returns:
            Path: The base directory for the on-disk cache.
        """
        return Path(gettempdir())

    def _fpath_from_key(self: Self, key: str) -> Path:
        """
        Get the file path associated with the given key.

        Args:
            key (str): The key used to query the cache.

        Returns:
            Path: The file path associated with the key.
        """
        return self.base_dir / key

    def _flock_from_fpath(self: Self, fpath: Path) -> FileLock:
        """
        Get the file lock associated with the given file path.

        Args:
            fpath (Path): The file path to lock.

        Returns:
            FileLock: The file lock associated with the file path.
        """
        return FileLock(str(fpath) + ".lock")

    @override
    def get(self: Self, key: str) -> bytes | None:
        """
        Retrieve the value associated with the given key from the cache on disk.

        Args:
            key (str): The key used to query the cache.

        Returns:
            bytes | None: The value associated with the key, or None if not found.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)
        with flock:
            return fpath.read_bytes() if fpath.is_file() else None

    @override
    def insert(self: Self, key: str, value: bytes) -> bool:
        """
        Store the given value in the cache with the associated key on disk.

        Args:
            key (str): The key to associate with the value.
            value (bytes): The value to be stored in the cache.

        Returns:
            bool: True if the value was stored successfully, False if the key already exists.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" mode is exclusive creation, meaning the file will be created
            # iff the file does not already exist (atomic w/o overwrite)
            with flock as _, open(fpath, "xb") as fp:
                fp.write(value)
        except FileExistsError:
            return False
        return True


class InductorOnDiskCache(OnDiskCache):
    """
    On-disk cache implementation for Inductor.

    Uses the default cache directory provided by Inductor.
    """

    @cached_property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the on-disk cache.

        Returns:
            Path: The base directory for the on-disk cache.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "pcache")
