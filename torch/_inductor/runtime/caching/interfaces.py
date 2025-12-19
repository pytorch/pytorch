# pyre-strict

"""Public interfaces for PyTorch Inductor runtime caching.

This module provides high-level caching interfaces for memoization and
result caching functionality.
"""

import functools
import pickle
from collections.abc import Callable
from hashlib import sha256
from os import PathLike
from typing import cast
from typing_extensions import ParamSpec, TypeVar

from . import config, implementations


# Type variable for function parameters
_P = ParamSpec("_P")
# Type variable for function return type
_R = TypeVar("_R")
# Type variable for encoded result type
_EncodedR = TypeVar("_EncodedR")


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


class _BaseMemoizer:
    """Base class for memoization interfaces.

    This class provides the common memoize method that orchestrates
    record and replay functionality.
    """

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
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
        custom_result_decoder: Callable[_P, Callable[[_EncodedR], _R]] | None = None,
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
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
        custom_result_decoder: Callable[_P, Callable[[_EncodedR], _R]] | None = None,
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

    Note: Use this over functools.cache when you need to support parameters
    that functools.cache cannot handle, or when you need custom encoding/decoding
    of results.
    """

    def __init__(self) -> None:
        """Initialize the Memoizer instance with an in-memory cache."""
        self._cache: implementations._InMemoryCacheImpl = (
            implementations._InMemoryCacheImpl()
        )

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.

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
                cache_key = _make_key(custom_params_encoder, *args, **kwargs)

                # Encode the result if encoder is provided
                if custom_result_encoder is not None:
                    # Get the encoder function by calling the factory with params
                    encoder_fn = custom_result_encoder(*args, **kwargs)
                    encoded_result = encoder_fn(result)
                else:
                    encoded_result = result

                # Store in cache
                self._cache.insert(cache_key, encoded_result)

                # Return the original result (not the encoded version)
                return result

            return inner

        return wrapper

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: Callable[_P, Callable[[_EncodedR], _R]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Replay a cached function result without executing the function.

        This is a decorator that retrieves cached results instead of executing
        the function. Raises an exception if no cached result exists.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_decoder: Optional decoder factory for cached results.
                                  Takes function parameters and returns a decoder
                                  function that converts _EncodedR -> R.

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

                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

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
                cache_key = _make_key(custom_params_encoder, *args, **kwargs)

                # Check if result is cached
                cached_hit = self._cache.get(cache_key)
                if cached_hit is None:
                    raise KeyError(f"No cached result found for key: {cache_key}")

                # Decode and return the cached result
                cached_value = cached_hit.value
                if custom_result_decoder is not None:
                    # Get the decoder function by calling the factory with params
                    decoder_fn = custom_result_decoder(*args, **kwargs)
                    return decoder_fn(cast(_EncodedR, cached_value))
                return cast(_R, cached_value)

            return inner

        return wrapper


class PersistentMemoizer(_BaseMemoizer):
    """Persistent memoization interface for caching function results to disk.

    This class provides methods for recording, retrieving, and managing
    cached function results using a two-level cache strategy:
    1. In-memory cache (fast, checked first) - via Memoizer instance
    2. On-disk cache (persistent, checked on memory miss)

    Results are persisted across process restarts.

    Note: Use this over functools.cache when you need to support parameters
    that functools.cache cannot handle, custom result encoding and/or decoding,
    or when you need disk caching to persist results across program boundaries.
    """

    def __init__(self, sub_dir: PathLike[str] | None = None) -> None:
        """Initialize the PersistentMemoizer with two-level caching.

        Args:
            sub_dir: Optional subdirectory within the cache directory for
                    organizing cached results. Defaults to empty string if not specified.
        """
        # Use a Memoizer instance for in-memory caching
        self._memoizer: Memoizer = Memoizer()
        # Store on-disk cache as a separate attribute
        self._disk_cache: implementations._OnDiskCacheImpl = (
            implementations._OnDiskCacheImpl(sub_dir)
        )

    def record(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_encoder: Callable[_P, Callable[[_R], _EncodedR]] | None = None,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Record a function call result with custom encoding to both caches.

        This is a decorator that wraps a function to enable memoization
        with custom encoding/decoding logic. Results are stored in both
        the in-memory cache and the on-disk cache.

        Args:
            custom_params_encoder: Optional encoder for function parameters.
                                  If None, parameters are pickled directly.
            custom_result_encoder: Optional encoder factory for function results.
                                  Takes function parameters and returns an encoder
                                  function that converts R -> _EncodedR.

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
                cache_key = _make_key(custom_params_encoder, *args, **kwargs)

                # Get the encoded result from memory cache
                # We know it must be there since memory_record_fn just cached it
                cached_hit = self._memoizer._cache.get(cache_key)
                encoded_result = cached_hit.value  # type: ignore[union-attr]

                # Store in disk cache (requires bytes, so pickle)
                pickled_result: bytes = pickle.dumps(encoded_result)
                self._disk_cache.insert(cache_key, pickled_result)

                return result

            return inner

        return wrapper

    def replay(
        self,
        custom_params_encoder: Callable[_P, object] | None = None,
        custom_result_decoder: Callable[_P, Callable[[_EncodedR], _R]] | None = None,
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
                                  Takes function parameters and returns a decoder
                                  function that converts _EncodedR -> R.

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

                def always_miss(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                    raise KeyError("Caching is disabled")

                return always_miss

            # Get the memory replay function
            memory_replay_fn = self._memoizer.replay(
                custom_params_encoder, custom_result_decoder
            )(fn)

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
                cache_key = _make_key(custom_params_encoder, *args, **kwargs)
                disk_hit = self._disk_cache.get(cache_key)
                if disk_hit is not None:
                    # Disk cache hit - unpickle the bytes
                    pickled_value = disk_hit.value
                    cached_value = pickle.loads(pickled_value)

                    # Populate memory cache for future access
                    self._memoizer._cache.insert(cache_key, cached_value)

                    # Decode and return
                    if custom_result_decoder is not None:
                        decoder_fn = custom_result_decoder(*args, **kwargs)
                        return decoder_fn(cast(_EncodedR, cached_value))
                    return cast(_R, cached_value)

                # Complete miss
                raise KeyError(f"No cached result found for key: {cache_key}")

            return inner

        return wrapper
