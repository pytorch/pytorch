"""Public interfaces for PyTorch Inductor runtime caching.

This module provides high-level caching interfaces for memoization and
result caching functionality.
"""

import atexit
import functools
import json
import logging
import pickle
import shutil
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import cast, Generic, Protocol, TypedDict
from typing_extensions import ParamSpec, TypeVar

from filelock import FileLock

import torch
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import clear_on_fresh_cache

from . import config, implementations, locks


logger = torch._logging.getArtifactLogger(__name__, "caching")

# Type variable for function parameters
_P = ParamSpec("_P")
# Type variable for function return type
_R = TypeVar("_R")
# Type variable for encoded result type
_EncodedR = TypeVar("_EncodedR")


class CacheDumpEntry(TypedDict):
    """A single cache entry in the dump format.

    Attributes:
        params: The encoded function parameters.
        result: The encoded function result.
    """

    params: object
    result: object


class CacheDump(TypedDict):
    """The structure of the memoizer cache dump file.

    Cache entries are organized by sub_key in the collections field.
    Memoizers without a sub_key use None as their key in collections.

    Multiple Memoizer instances with different sub_keys can coexist in the same file.

    Attributes:
        collections: Dictionary mapping sub_keys (or None for root entries)
            to their cache entries.
        cache_size: The total number of cache entries across all collections.
    """

    collections: dict[str | None, dict[str, CacheDumpEntry]]
    cache_size: int


@dataclass
class CacheEntry:
    """A cache entry containing encoded parameters and result.

    This dataclass stores the encoded form of function parameters and
    the (possibly encoded) result for human-readable cache dumps.

    Attributes:
        encoded_params: The encoded function parameters used for debugging/inspection.
        encoded_result: The (possibly encoded) result of the function call.
    """

    encoded_params: object
    encoded_result: object


@dataclass
class InterimResult(Generic[_R]):
    """Wrapper for an available interim result from a DeferredRecording.

    When get_interim_result() returns an InterimResult, it means the
    make_interim_result callable was provided and has been invoked.
    The wrapped value is the result of that invocation.

    This allows distinguishing between:
    - No interim result configured: get_interim_result() returns None
    - Interim result available: get_interim_result() returns InterimResult(value)
      where value may itself be None if the callable returns None.

    Attributes:
        value: The value returned by calling make_interim_result().
               May legitimately be None if the callable returns None.
    """

    value: _R


@dataclass
class DeferredRecording(Generic[_R, _EncodedR]):
    """Signals that recording should happen at a later time.

    When returned from a custom_result_encoder, the memoizer will:
    1. Skip immediate caching
    2. Register a completion callback on this object

    The encoder is responsible for calling `finalize(encoded_result)`
    when the actual result is ready to be cached. This is useful when
    the function returns a value whose final encoded form is not yet
    available (e.g., an object containing a pending computation),
    allowing the expensive work to remain hidden while still enabling caching.

    This class handles the race condition where the computation might complete
    before the memoizer has registered its callback. If finalize() is called
    first, the result is stored and the callbacks are invoked when registered.

    Multiple callbacks can be registered - they will all be invoked when
    finalize() is called (or immediately if already completed). Callbacks are
    guaranteed to be invoked in registration order.

    Optionally, a `make_interim_result` callable can be provided to handle
    subsequent calls while the deferred recording is pending. When a memoized
    function is called again with the same parameters while a deferred recording
    is still pending, the memoizer can use this callable to construct an
    appropriate return value instead of re-executing the function. The callable
    does not necessarily return the same object - it provides a way to derive
    a suitable response based on the pending deferred recording's context.

    This class is thread-safe: concurrent calls to finalize() and
    register_callback() are properly synchronized.

    WARNING: Callbacks are executed while holding the internal lock to ensure
    ordering guarantees. Callbacks MUST NOT call back into this DeferredRecording
    instance (e.g., calling finalize() or register_callback() from within a
    callback) as this will cause a deadlock.

    Example usage in an encoder:
        def my_encoder(*args, **kwargs):
            def encode(result: _R) -> DeferredRecording[_R, _EncodedR]:
                deferred: DeferredRecording[_R, _EncodedR] = DeferredRecording(
                        # Construct an appropriate return for subsequent calls while pending
                        make_interim_result=lambda: result
                    )

                def on_complete():
                    encoded_result: _EncodedR = _encode_result(result)
                    deferred.finalize(encoded_result)

                result.add_completion_callback(on_complete)
                return deferred

            return encode

    Attributes:
        _callbacks: List of callbacks to invoke when finalize() is called.
                   Set to None after finalize() is called to indicate completion.
        _encoded_result: The encoded result passed to finalize().
        _lock: Lock for thread-safe access to mutable state.
        make_interim_result: Optional callable that constructs a return value for subsequent
                           calls while the deferred recording is pending. If set, the memoizer
                           calls this instead of re-executing the function. The callable may
                           return the same object or construct a new appropriate response.
    """

    _callbacks: list[Callable[[_EncodedR], None]] | None = field(
        default_factory=list, repr=False
    )
    _encoded_result: _EncodedR | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    make_interim_result: Callable[[], _R] | None = field(default=None, repr=False)

    def finalize(self, encoded_result: _EncodedR) -> None:
        """Finalize the deferred recording with the encoded result.

        This method should be called by the encoder (typically via a completion
        callback) when the actual result is ready to be cached.

        All registered callbacks are invoked with the encoded result in
        registration order. If no callbacks are registered yet, the result is
        stored and callbacks will be invoked when registered.

        This method is thread-safe.

        WARNING: Callbacks are executed while holding the internal lock.
        Do not call finalize() or register_callback() from within a callback,
        as this will cause a deadlock.

        Args:
            encoded_result: The encoded result to cache.
        """
        with self._lock:
            if self._callbacks is None:
                raise RuntimeError(
                    "finalize() called multiple times on DeferredRecording"
                )

            # Store the result and get callbacks to invoke
            self._encoded_result = encoded_result
            callbacks_to_invoke = self._callbacks
            self._callbacks = None

            # Execute callbacks inside the lock to ensure ordering:
            # - All pre-finalize callbacks execute in registration order
            # - Any callback registered during/after finalize will see
            #   _callbacks=None and execute immediately (also under lock)
            for callback in callbacks_to_invoke:
                callback(encoded_result)

    def register_callback(self, callback: Callable[[_EncodedR], None]) -> None:
        """Register a completion callback.

        This method is called by memoizers to register callbacks that
        will insert cache entries. Multiple callbacks can be registered.

        If finalize() has already been called (e.g., the computation was fast),
        the callback is invoked immediately with the stored result.

        This method is thread-safe. Callbacks are guaranteed to be invoked
        in registration order.

        WARNING: Callbacks are executed while holding the internal lock.
        Do not call finalize() or register_callback() from within a callback,
        as this will cause a deadlock.

        Args:
            callback: The function to call with the encoded result.
        """
        with self._lock:
            if self._callbacks is None:
                # Already finalized - invoke immediately while holding lock
                # to maintain ordering with respect to other callbacks
                # we need to cast here, we can't assert because the encoded
                # result may genuinely be None
                callback(cast(_EncodedR, self._encoded_result))
            else:
                self._callbacks.append(callback)

    def get_interim_result(self) -> InterimResult[_R] | None:
        """Try to get an interim result for use while deferred recording is pending.

        When a memoized function is called again with the same parameters while
        a deferred recording is still pending, the memoizer uses this method to
        check if an interim result can be returned instead of re-executing the
        function.

        This method provides thread-safe access to the make_interim_result
        callable. It atomically checks if the callable was provided and invokes
        it if so.

        This method is thread-safe.

        Returns:
            InterimResult wrapping the value if make_interim_result was provided,
            or None if no interim result is available.

        Note:
            The callable is invoked each time this method is called, so if
            make_interim_result returns a new object each time, callers will
            receive different objects. To return the same object, the callable
            should capture and return a reference to a single object.
        """
        with self._lock:
            if self.make_interim_result is not None:
                return InterimResult(self.make_interim_result())
            return None


class ResultEncoderFactory(Protocol[_P, _R, _EncodedR]):
    """Protocol for custom result encoder factories.

    A result encoder factory is a callable with the following signature:
        factory(fn) -> params_to_encoder(*args, **kwargs) -> encoder(result) -> encoded

    The three levels are:
    1. factory(fn): Takes the underlying unwrapped function, returns a params_to_encoder
    2. params_to_encoder(*args, **kwargs): Takes the memoized function's arguments,
       returns an encoder function
    3. encoder(result): Converts the result R -> _EncodedR (or DeferredRecording)

    The `fn` parameter allows the encoder to call the underlying function without
    triggering memoization, which is useful when the encoder needs to re-execute
    the function (e.g., to get a fresh result for DeferredRecording).

    Example:
        def my_encoder_factory(fn: Callable) -> Callable:
            def params_to_encoder(*args, **kwargs) -> Callable:
                def encode(result: R) -> EncodedR:
                    return {"encoded": result}
                return encode
            return params_to_encoder

        @memoizer.record(custom_result_encoder=my_encoder_factory)
        def compute(x: int) -> int:
            return x * 2
    """

    def __call__(
        self,
        fn: Callable[_P, _R],
    ) -> Callable[_P, Callable[[_R], _EncodedR | DeferredRecording[_R, _EncodedR]]]: ...


class ResultDecoderFactory(Protocol[_P, _R, _EncodedR]):
    """Protocol for custom result decoder factories.

    A result decoder factory is a callable with the following signature:
        factory(fn) -> params_to_decoder(*args, **kwargs) -> decoder(encoded) -> result

    The three levels are:
    1. factory(fn): Takes the underlying unwrapped function, returns a params_to_decoder
    2. params_to_decoder(*args, **kwargs): Takes the memoized function's arguments,
       returns a decoder function
    3. decoder(encoded): Converts the encoded result _EncodedR -> R

    The `fn` parameter allows the decoder to call the underlying function without
    triggering memoization, which is useful for fallback paths when the cached
    result cannot be fully decoded.

    Example:
        def my_decoder_factory(fn: Callable) -> Callable:
            def params_to_decoder(*args, **kwargs) -> Callable:
                def decode(encoded_result: EncodedR) -> R:
                    return encoded_result["value"]
                return decode
            return params_to_decoder

        @memoizer.replay(custom_result_decoder=my_decoder_factory)
        def compute(x: int) -> int:
            return x * 2
    """

    def __call__(
        self,
        fn: Callable[_P, _R],
    ) -> Callable[_P, Callable[[_EncodedR], _R]]: ...


class _BaseMemoizer:
    """Base class for memoization interfaces.

    This class provides the common memoize method that orchestrates
    record and replay functionality.
    """

    @staticmethod
    def _make_key(
        custom_params_encoder: Callable[..., object] | None,
        *args: object,
        **kwargs: object,
    ) -> str:
        """Generate a cache key from function parameters.

        Args:
            custom_params_encoder: Optional encoder to apply to function parameters.
                                  If None, params are pickled directly.
            *args: Positional arguments to encode.
            **kwargs: Keyword arguments to encode.

        Returns:
            A 32-character hex string suitable for use as a cache key.
        """
        if custom_params_encoder is None:
            # Pickle the parameters directly
            pickled_params: bytes = pickle.dumps((args, kwargs))
        else:
            # Encode the parameters using the custom encoder
            encoded_params = custom_params_encoder(*args, **kwargs)
            # Pickle the encoded output
            pickled_params = pickle.dumps(encoded_params)

        # Hash the pickled bytes with SHA256
        hash_obj = sha256(pickled_params)

        # Get hex digest and truncate to 32 characters
        return hash_obj.hexdigest()[:32]

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result. Must be implemented by subclasses.

        This can be used standalone to cache results without necessarily
        replaying them, for example for logging or analytics purposes.

        See memoize() for a description of the arguments.
        """
        raise NotImplementedError

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result. Must be implemented by subclasses.

        This can be used standalone to use cached values that are otherwise
        too hard to automatically record.

        See memoize() for a description of the arguments.
        """
        raise NotImplementedError

    def memoize(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Memoize a function with record and replay functionality.

        This is a decorator that attempts to replay cached results first.
        If a cache miss occurs, it records the result by executing the wrapped function.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.
            custom_result_decoder: Optional decoder factory for cached results.
                                  Takes function parameters and returns a decoder
                                  function that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.memoize(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization with replay and record.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Create decorated versions using record and replay
            replay_fn = self.replay(
                custom_params_encoder,
                custom_result_decoder,
            )(fn)
            record_fn = self.record(
                custom_params_encoder,
                custom_result_encoder,
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Attempt to replay from cache, or record on cache miss.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result from cache (if hit) or from executing the function (if miss).
                """
                # Try to replay first
                try:
                    return replay_fn(*args, **kwargs)
                except KeyError:
                    # Cache miss - record the result
                    return record_fn(*args, **kwargs)

            return inner

        return wrapper


class Memoizer(_BaseMemoizer):
    """In-memory memoization interface for caching function results.

    This class provides methods for recording, retrieving, and managing
    cached function results in memory with custom encoding/decoding logic.

    Cache entries are stored as CacheEntry objects containing both the encoded
    parameters and the encoded result. This makes debugging easier since entries
    can be inspected with full context about what inputs produced each result.

    On memoizer destruction, the cache is automatically dumped to a shared JSON file
    in the cache directory for debugging and inspection purposes. Multiple Memoizer
    instances contribute to the same file additively.

    Note: Use this over functools.cache when you need to support parameters
    that functools.cache cannot handle, or when you need custom encoding/decoding
    of results.
    """

    def __init__(self, sub_key: str | None = None) -> None:
        """Initialize the Memoizer instance with an in-memory cache.

        Args:
            sub_key: Optional key for organizing cache entries in the JSON dump.
                    If provided, cache entries are stored under cache_entries[sub_key].
                    If None, cache entries are merged directly into root cache_entries.
        """
        self._cache: implementations._InMemoryCacheImpl[CacheEntry] = (
            implementations._InMemoryCacheImpl()
        )
        # Optional sub_key for nested cache structure
        self._sub_key: str | None = sub_key
        # Track pending deferred recordings by cache key.
        # Uses WeakValueDictionary to prevent memory leaks: if the DeferredRecording
        # is no longer referenced elsewhere (e.g., the async computation was cancelled
        # or failed and garbage collected), the entry is automatically removed.
        self._pending_deferred: weakref.WeakValueDictionary[str, DeferredRecording] = (
            weakref.WeakValueDictionary()
        )
        # Register atexit handler to dump cache on program exit
        if config.IS_DUMP_MEMOIZER_CACHE_ENABLED():
            atexit.register(self._dump_to_disk)
        # Pre-populate cache from dump file if configured (with sub_key now set)
        self._maybe_prepopulate_from_dump()
        # Register with clear_on_fresh_cache for fresh_cache() integration
        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        """Clear the in-memory cache.

        This method resets the in-memory cache to empty. It is called by
        the fresh_cache() context manager via clear_on_fresh_cache registration.
        """
        self._cache._memory.clear()

    def _finalize_deferred_recording(
        self,
        cache_key: str,
        encoded_params: object,
        encoded_result: _EncodedR,
    ) -> None:
        """Finalize a deferred recording by inserting the cache entry.

        This method is called when a DeferredRecording's finalize() method is invoked.
        It creates the cache entry and inserts it into the in-memory cache.

        Args:
            cache_key: The cache key for the entry.
            encoded_params: The encoded function parameters.
            encoded_result: The final encoded result to cache.
        """
        # Remove from pending tracking
        self._pending_deferred.pop(cache_key, None)
        # Insert into cache
        cache_entry = CacheEntry(
            encoded_params=encoded_params,
            encoded_result=encoded_result,
        )
        self._cache.insert(cache_key, cache_entry)

    @functools.cached_property
    def _shared_cache_filepath(self) -> Path:
        """Get the shared cache filepath for memoizer cache dumps.

        Returns:
            The path to the shared memoizer cache JSON file.
        """
        return Path(cache_dir()) / "memoizer_cache.json"

    @functools.cached_property
    def _shared_cache_lockfile(self) -> Path:
        """Get the lock file path for the shared memoizer cache.

        Returns:
            The path to the lock file for the shared cache.
        """
        return Path(cache_dir()) / "memoizer_cache.lock"

    def _read_dump_from_disk(self, filepath: Path | None = None) -> CacheDump | None:
        """Read a cache dump from disk.

        Attempts to read and parse a cache JSON file.

        Args:
            filepath: Path to the dump file to read. If None, uses the
                     shared cache filepath (self._shared_cache_filepath).

        Returns:
            The cache dump if the file exists and is valid JSON, None otherwise.
        """
        target_path = filepath if filepath is not None else self._shared_cache_filepath
        try:
            with open(target_path) as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None

    def _write_dump_to_disk(self, dump: CacheDump) -> None:
        """Write the cache dump to disk.

        Writes the provided dump to the shared cache JSON file and logs the result.

        Args:
            dump: The cache dump to write.
        """
        try:
            with open(self._shared_cache_filepath, "w") as f:
                json.dump(dump, f, indent=2)

            # Log the filepath
            if self._sub_key:
                logger.log(
                    logging.INFO,
                    "Memoizer cache (sub_key=%s) dumped to: %s",
                    self._sub_key,
                    self._shared_cache_filepath,
                )
            else:
                logger.log(
                    logging.INFO,
                    "Memoizer cache dumped to: %s",
                    self._shared_cache_filepath,
                )
        except Exception as e:
            # If dumping fails, just log it and don't crash the program
            logger.log(
                logging.WARNING,
                "Warning: Failed to dump memoizer cache: %s",
                e,
            )

    def _prepare_dump(self, existing_dump: CacheDump | None) -> CacheDump:
        """Prepare a cache dump from the current Memoizer state.

        Takes the existing dump (if any) and merges it with the current
        in-memory cache entries.

        Args:
            existing_dump: The existing dump to merge with, or None if starting fresh.

        Returns:
            The prepared cache dump ready to be written to disk.
        """
        # Start with existing data or empty structure
        if existing_dump is not None:
            dump = existing_dump
        else:
            dump: CacheDump = {"collections": {}, "cache_size": 0}

        # Ensure collections field exists
        if "collections" not in dump:
            dump["collections"] = {}

        # JSON serializes None keys as "null" string, so we need to handle both cases
        # When reading from JSON, the key will be "null" (string), not None
        lookup_key = "null" if self._sub_key is None else self._sub_key

        # Get existing entries for this sub_key to merge with
        existing_entries = dump["collections"].get(lookup_key, {})

        # Format cache entries as {"params": ..., "result": ...}
        formatted_cache: dict[str, CacheDumpEntry] = dict(existing_entries)
        for key, value in self._cache._memory.items():
            entry = value
            formatted_cache[key] = CacheDumpEntry(
                params=entry.encoded_params,
                result=entry.encoded_result,
            )

        # Store under sub_key in collections (use lookup_key to maintain JSON compatibility)
        dump["collections"][lookup_key] = formatted_cache

        # Calculate total cache size
        total_size = sum(len(collection) for collection in dump["collections"].values())
        dump["cache_size"] = total_size

        return dump

    def _dump_to_disk(self) -> None:
        """Dump the in-memory cache to a shared JSON file.

        This method is automatically called on program exit via atexit.
        It reads any existing cache data, merges it with this instance's cache,
        and writes the combined result back. Multiple Memoizer instances
        contribute to the same file additively.

        If self._sub_key is set and non-empty, cache entries are stored under
        cache_entries[sub_key]. Otherwise, they're merged directly into cache_entries.

        Cache entries are formatted as {"params": <encoded_params>, "result": <encoded_result>}
        for better human readability.

        The filepath where the cache was dumped is logged.
        """
        # Skip if cache is empty
        if not self._cache._memory:
            return

        # Ensure parent directory exists
        self._shared_cache_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Acquire file lock to ensure thread/process safety
        flock = FileLock(str(self._shared_cache_lockfile))
        with locks._acquire_flock_with_timeout(flock):
            existing_dump = self._read_dump_from_disk()
            dump = self._prepare_dump(existing_dump)
            self._write_dump_to_disk(dump)

    def _maybe_prepopulate_from_dump(self) -> None:
        """Pre-populate cache entries from a dump file if configured.

        Checks the CACHE_DUMP_FILE_PATH config option for a path to a JSON dump file
        produced by IS_DUMP_MEMOIZER_CACHE_ENABLED. If a valid path is provided and
        the file exists, loads cache entries from it into the in-memory cache.

        For Memoizer instances without a sub_key, loads entries from the root cache_entries.
        For Memoizer instances with a sub_key, loads entries from cache_entries[sub_key].

        This method is called during __init__ to pre-populate the cache.
        """
        dump_file_path = config.CACHE_DUMP_FILE_PATH()

        # Skip if no dump file configured
        if not dump_file_path:
            return

        # Read the dump file using the helper method
        dump = self._read_dump_from_disk(Path(dump_file_path))
        if dump is None:
            return

        # Extract entries to load from the dump
        entries_to_load = self._extract_entries_from_dump(dump)
        if not entries_to_load:
            return

        # Populate the cache
        self._populate_cache_from_entries(entries_to_load)

        # Log the result
        if self._sub_key:
            logger.log(
                logging.INFO,
                "Loaded %d cache entries from %s (sub_key=%s)",
                len(entries_to_load),
                dump_file_path,
                self._sub_key,
            )
        else:
            logger.log(
                logging.INFO,
                "Loaded %d cache entries from %s",
                len(entries_to_load),
                dump_file_path,
            )

    def _extract_entries_from_dump(self, dump: CacheDump) -> dict[str, CacheDumpEntry]:
        """Extract cache entries from a dump based on sub_key.

        Args:
            dump: The cache dump to extract entries from.

        Returns:
            Dictionary of cache entries to load.
        """
        collections = dump.get("collections", {})
        # JSON serializes None keys as "null" string, so we need to handle both cases
        # When reading from JSON, the key will be "null" (string), not None
        lookup_key = "null" if self._sub_key is None else self._sub_key
        return collections.get(lookup_key, {})

    def _populate_cache_from_entries(self, entries: dict[str, CacheDumpEntry]) -> None:
        """Populate the in-memory cache from dump entries.

        Args:
            entries: Dictionary of cache entries to load.
        """
        for key, entry in entries.items():
            cache_entry = CacheEntry(
                encoded_params=entry["params"],
                encoded_result=entry["result"],
            )
            self._cache.insert(key, cache_entry)

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  First receives the underlying function, then takes
                                  function parameters and returns an encoder function
                                  that converts R -> _EncodedR.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.record(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the function to compute the result
                result = fn(*args, **kwargs)

                # Generate cache key from parameters
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Encode params for human-readable dump
                if custom_params_encoder is not None:
                    encoded_params = custom_params_encoder(*args, **kwargs)
                else:
                    encoded_params = {
                        "args": args,
                        "kwargs": kwargs,
                    }

                # Encode the result if encoder is provided
                if custom_result_encoder is not None:
                    # Get the encoder function by calling the factory factory:
                    # 1. First call with fn to get params_to_encoder factory
                    # 2. Second call with params to get the encoder function
                    params_to_encoder = custom_result_encoder(fn)
                    encoder_fn = params_to_encoder(*args, **kwargs)
                    encoded_result = encoder_fn(result)
                else:
                    encoded_result = result

                # Check for deferred recording
                if isinstance(encoded_result, DeferredRecording):
                    # Track the pending deferred recording
                    self._pending_deferred[cache_key] = encoded_result
                    # Register the callback - handles race condition if encoded_result already completed
                    encoded_result.register_callback(
                        functools.partial(
                            self._finalize_deferred_recording,
                            cache_key,
                            encoded_params,
                        )
                    )
                    return result  # Return without caching

                # Store CacheEntry in cache
                cache_entry = CacheEntry(
                    encoded_params=encoded_params,
                    encoded_result=encoded_result,
                )
                self._cache.insert(cache_key, cache_entry)

                # Return the original result (not the encoded version)
                return result

            return inner

        return wrapper

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result without executing the function.

        This is a decorator that retrieves cached results instead of executing
        the function. Raises an exception if no cached result exists.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_decoder: Optional decoder factory for cached results.
                                  First receives the underlying function, then takes
                                  function parameters and returns a decoder function
                                  that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @memoizer.replay(
                custom_params_encoder=my_param_encoder,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to retrieve from cache.

            Args:
                fn: The function to wrap (not actually called).

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, always raise KeyError (cache miss)
            if not config.IS_CACHING_MODULE_ENABLED():

                @functools.wraps(fn)
                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Retrieve the cached result without calling the function.

                Args:
                    *args: Positional arguments to generate the cache key.
                    **kwargs: Keyword arguments to generate the cache key.

                Returns:
                    The cached result (decoded if decoder is provided).

                Raises:
                    KeyError: If no cached result exists for the given parameters.
                """
                # Generate cache key from parameters
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Check if result is cached
                cached_hit = self._cache.get(cache_key)
                if cached_hit is not None:
                    # Extract the cached value
                    cache_entry = cached_hit.value

                    # Decode and return the cached result
                    if custom_result_decoder is not None:
                        # Get the decoder function by calling the factory factory:
                        # 1. First call with fn to get params_to_decoder factory
                        # 2. Second call with params to get the decoder function
                        params_to_decoder = custom_result_decoder(fn)
                        decoder_fn = params_to_decoder(*args, **kwargs)
                        return decoder_fn(cast(_EncodedR, cache_entry.encoded_result))
                    return cast(_R, cache_entry.encoded_result)

                # Check for pending deferred recording with interim result
                pending = self._pending_deferred.get(cache_key)
                if pending is not None:
                    interim = pending.get_interim_result()
                    if interim is not None:
                        return interim.value

                raise KeyError(f"No cached result found for key: {cache_key}")

            return inner

        return wrapper


class PersistentMemoizer(_BaseMemoizer):
    """Persistent memoization interface for caching function results to disk.

    This class provides methods for recording, retrieving, and managing
    cached function results using a two-level cache strategy:
    1. In-memory cache (fast, checked first) - via Memoizer instance
    2. On-disk cache (persistent, checked on memory miss)

    Results are persisted across process restarts.

    On program exit, the in-memory cache entries are automatically dumped to
    the shared JSON file. If sub_dir is non-empty, entries are stored under
    a nested structure based on the sub_dir. If sub_dir is empty, entries are
    merged directly into the root cache_entries.

    Note: Use this over functools.cache when you need to support parameters
    that functools.cache cannot handle, custom result encoding and/or decoding,
    or when you need disk caching to persist results across program boundaries.
    """

    def __init__(self, sub_dir: PathLike[str] | None = None) -> None:
        """Initialize the PersistentMemoizer with two-level caching.

        Args:
            sub_dir: Optional subdirectory within the cache directory for
                    organizing cached results. Defaults to empty string if not specified.
                    If non-empty, cache entries will be stored under cache_entries[sub_dir].
                    If empty, cache entries are merged into root cache_entries.
        """
        # Use a Memoizer instance for in-memory caching
        self._memoizer: Memoizer = Memoizer(sub_key=str(sub_dir) if sub_dir else None)
        # Store on-disk cache as a separate attribute
        self._disk_cache: implementations._OnDiskCacheImpl = (
            implementations._OnDiskCacheImpl(sub_dir=sub_dir)
        )
        # Register with clear_on_fresh_cache for fresh_cache() integration
        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        """Clear the on-disk cache.

        This method removes the on-disk cache directory. The in-memory cache
        is cleared automatically by the underlying Memoizer instance, which
        registers itself with clear_on_fresh_cache.

        This is called by the fresh_cache() context manager via clear_on_fresh_cache
        registration.
        """
        # Clear on-disk cache by removing the cache directory
        # Note: in-memory cache is cleared by the Memoizer's own cache_clear
        if self._disk_cache._cache_dir.exists():
            shutil.rmtree(self._disk_cache._cache_dir, ignore_errors=True)

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: ResultEncoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding to both caches.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic. Results are stored in both
        the in-memory cache and the on-disk cache.

        This method delegates to the underlying Memoizer for memory caching,
        then adds disk persistence. For deferred recordings, it registers
        an additional callback to persist to disk when the recording completes.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  First receives the underlying function, then takes
                                  function parameters and returns an encoder function
                                  that converts R -> _EncodedR.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @persistent_memoizer.record(
                custom_params_encoder=my_param_encoder,
                custom_result_encoder=my_result_encoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to enable memoization.

            Args:
                fn: The function to wrap.

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, return the original function unchanged
            if not config.IS_CACHING_MODULE_ENABLED():
                return fn

            # Get the memory-cached version from the memoizer
            memory_record_fn = self._memoizer.record(
                custom_params_encoder, custom_result_encoder
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Call the original function and cache the result in both caches.

                Args:
                    *args: Positional arguments to pass to the function.
                    **kwargs: Keyword arguments to pass to the function.

                Returns:
                    The result of calling the original function.
                """
                # Call the memory-cached version (which calls fn and caches in memory)
                result = memory_record_fn(*args, **kwargs)

                # Also store in disk cache
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)

                # Check if this was a deferred recording
                pending = self._memoizer._pending_deferred.get(cache_key)
                if pending is not None:
                    # Deferred recording - register our own callback for disk persistence
                    # By the time our callback runs, Memoizer's callback will have
                    # already inserted the cache entry, so we just read and persist it
                    pending.register_callback(
                        functools.partial(self._persist_to_disk, cache_key)
                    )
                    return result

                # Persist the cache entry to disk
                self._persist_to_disk(cache_key)

                return result

            return inner

        return wrapper

    def _persist_to_disk(
        self,
        cache_key: str,
        _callback_result: object = None,
    ) -> None:
        """Persist a cache entry to disk.

        This method handles disk persistence for both immediate and deferred recordings:
        - For immediate recordings: called directly after memory caching
        - For deferred recordings: registered as a callback that runs after the
          Memoizer's callback has inserted the entry into memory

        Args:
            cache_key: The cache key for the entry.
            _callback_result: Unused. When called as a callback from DeferredRecording,
                             this receives the encoded result, but we ignore it and
                             read the full CacheEntry from the Memoizer's cache instead.
        """
        # Always read from memory cache - Memoizer's callback has already inserted it
        cached_hit = self._memoizer._cache.get(cache_key)
        assert cached_hit, "Cache entry must exist in memory cache"
        cache_entry = cached_hit.value

        # Store the full CacheEntry in disk cache for easier debugging
        pickled_entry: bytes = pickle.dumps(cache_entry)
        self._disk_cache.insert(cache_key, pickled_entry)

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: ResultDecoderFactory[_P, _R, _EncodedR] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result without executing the function.

        This is a decorator that retrieves cached results using a two-level
        cache strategy. It checks the in-memory cache first (fast), then
        falls back to the on-disk cache. If found on disk, the result is
        cached in memory for future access.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_decoder: Optional decoder factory for cached results.
                                  First receives the underlying function, then takes
                                  function parameters and returns a decoder function
                                  that converts _EncodedR -> R.

        Returns:
            A decorator function that can be applied to functions.

        Example:
            @persistent_memoizer.replay(
                custom_params_encoder=my_param_encoder,
                custom_result_decoder=my_result_decoder_factory,
            )
            def expensive_function(x, y):
                return x + y
        """

        def wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            """Wrap the function to retrieve from cache.

            Args:
                fn: The function to wrap (not actually called).

            Returns:
                A wrapped version of the function.
            """
            # If caching is disabled, always raise KeyError (cache miss)
            if not config.IS_CACHING_MODULE_ENABLED():

                @functools.wraps(fn)
                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

            # Get the memory replay function
            memory_replay_fn = self._memoizer.replay(
                custom_params_encoder, custom_result_decoder
            )(fn)

            @functools.wraps(fn)
            def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                """Retrieve the cached result without calling the function.

                Checks memory cache first, then disk cache. Populates memory
                cache from disk on a disk hit.

                Args:
                    *args: Positional arguments to generate the cache key.
                    **kwargs: Keyword arguments to generate the cache key.

                Returns:
                    The cached result (decoded if decoder is provided).

                Raises:
                    KeyError: If no cached result exists for the given parameters.
                """
                # Try memory cache first via memoizer
                try:
                    return memory_replay_fn(*args, **kwargs)
                except KeyError:
                    pass  # Memory miss, check disk

                # Memory miss - check disk cache
                cache_key = self._make_key(custom_params_encoder, *args, **kwargs)
                disk_hit = self._disk_cache.get(cache_key)
                if disk_hit is not None:
                    # Disk cache hit - unpickle the CacheEntry
                    pickled_value = disk_hit.value
                    cache_entry = cast(CacheEntry, pickle.loads(pickled_value))

                    # Populate memory cache for future access
                    self._memoizer._cache.insert(cache_key, cache_entry)

                    # Decode and return
                    if custom_result_decoder is not None:
                        # Get the decoder function by calling the factory factory:
                        # 1. First call with fn to get params_to_decoder factory
                        # 2. Second call with params to get the decoder function
                        params_to_decoder = custom_result_decoder(fn)
                        decoder_fn = params_to_decoder(*args, **kwargs)
                        return decoder_fn(cast(_EncodedR, cache_entry.encoded_result))
                    return cast(_R, cache_entry.encoded_result)

                # Complete miss
                raise KeyError(f"No cached result found for key: {cache_key}")

            return inner

        return wrapper
