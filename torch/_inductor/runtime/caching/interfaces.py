"""High-level cache interfaces for function memoization and result caching.

This module provides abstract base classes and concrete implementations for different
caching strategies. The interfaces support function decoration for automatic caching
with configurable isolation contexts, custom encoders/decoders, and different
consistency guarantees.

Key components:
- _CacheIntf: Abstract base class defining the cache interface contract
- _FastCacheIntf: Fast caching using in-memory + on-disk storage for speed
- _DeterministicCacheIntf: Deterministic caching using remote storage for consistency

The interfaces support:
- Automatic function result memoization through decorators
- Configurable isolation contexts for cache key generation
- Custom parameter and result encoders/decoders
- Different consistency models (eventual vs strong consistency)
- Thread-safe operations with appropriate locking strategies
"""
from abc import ABC, abstractmethod
from functools import partial, wraps
from threading import Lock
from typing import Any, Callable, Generator, Optional, override, Self

from . import context, exceptions, locks
from . import implementations as impls
from .utils import P, R


class _CacheIntf(ABC):
    """Abstract base class defining the cache interface contract.
    
    This class establishes the fundamental interface for cache implementations,
    providing key generation, locking mechanisms, and abstract methods that
    must be implemented by concrete cache classes.
    
    The interface supports:
    - Function parameter and result encoding/decoding
    - Isolation contexts for cache key namespacing
    - Thread-safe operations through locking
    - Function decoration for automatic caching
    """
    
    def __init__(self: Self) -> None:
        """Initialize the cache interface with a lock for thread safety."""
        self._lock: Lock = Lock()
    
    @staticmethod
    def _make_key(
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
    ) -> Any:
        """Generate a cache key from function name, parameters, and isolation context.
        
        Creates a composite key that includes the function name, its parameters
        (optionally encoded), and the isolation context. This ensures cache
        entries are properly namespaced and isolated.
        
        Args:
            fn: The function being cached
            params: The function parameters (args, kwargs tuple)
            ischema: Isolation schema for context-aware caching
            custom_params_encoder: Optional encoder for parameter serialization
            
        Returns:
            A composite cache key consisting of function key and isolation key
            
        Example:
            key = _CacheIntf._make_key(
                my_function,
                (args, kwargs),
                isolation_schema,
                custom_encoder
            )
        """
        callee: str = fn.__name__
        fkey: Any = (callee, params) if not custom_params_encoder else (callee, custom_params_encoder(params))
        ikey: Any = context._isolation_key(ischema if ischema is not None else context._DEFAULT_ISOLATION_SCHEMA)
        return (fkey, ikey)
    
    @abstractmethod
    def _make_record_wrapper(
        intf: Self,
        fn: Callable[P, R],
        ischema: context.IsolationSchema = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[P, R]:
        """Create a wrapper function that adds caching to the target function.
        
        This abstract method must be implemented by concrete cache classes to
        provide function decoration with automatic caching. The wrapper should
        check the cache before calling the original function and store results
        after successful execution.
        
        Args:
            intf: The cache interface instance
            fn: The function to wrap with caching
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for function parameters
            custom_result_encoder: Optional encoder for function results
            custom_result_decoder: Optional decoder for cached results
            
        Returns:
            A wrapped version of the function with caching behavior
        """
        pass
    
    @property
    def lock(self: Self) -> Callable[[int], Generator[None, None, None]]:
        """Provide a context manager for thread-safe cache operations.
        
        Returns a function that creates a context manager for acquiring the
        cache lock with an optional timeout. This ensures thread-safe access
        to cache operations.
        
        Returns:
            A function that takes an optional timeout and returns a context manager
            
        Example:
            with cache.lock(timeout=30):
                result = cache.get(key)
        """
        return lambda timeout=locks._DEFAULT_TIMEOUT: locks._acquire_lock_with_timeout(self._lock, timeout) 

    @abstractmethod
    def get(
        self: Self,
        fn: Callable[P, R],
        params: P,
        ischema: context.IsolationSchema,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> impls.Get:
        """Retrieve a cached result for the given function and parameters.
        
        Attempts to find a previously cached result for the specified function
        call with the given parameters and isolation context. If found, the
        result is optionally decoded and returned.
        
        Args:
            fn: The function whose result is being retrieved
            params: The function parameters (args, kwargs tuple)
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for parameter serialization
            custom_result_decoder: Optional decoder for cached results
            
        Returns:
            Get result indicating cache hit/miss and the cached value if found
        """
        pass

    @abstractmethod
    def insert(
        self: Self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: context.IsolationSchema,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> impls.Insert:
        """Store a function result in the cache.
        
        Attempts to store the result of a function call in the cache using
        the generated key from the function, parameters, and isolation context.
        The result is optionally encoded before storage.
        
        Args:
            fn: The function whose result is being cached
            params: The function parameters (args, kwargs tuple)
            result: The function result to cache
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for parameter serialization
            custom_result_encoder: Optional encoder for the result
            
        Returns:
            Insert result indicating whether the insertion succeeded
        """
        pass
    
    def record(
        self: Self,
        ischema: context.IsolationSchema = context._DEFAULT_ISOLATION_SCHEMA,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[Callable[P, R], Callable[P, R]]:
        """Create a decorator for automatic function result caching.
        
        Returns a decorator that can be applied to functions to automatically
        cache their results. The decorator uses the configured isolation schema
        and optional encoders/decoders.
        
        Args:
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for function parameters
            custom_result_encoder: Optional encoder for function results
            custom_result_decoder: Optional decoder for cached results
            
        Returns:
            A decorator function that adds caching to the target function
            
        Example:
            @cache.record(isolation_schema, param_encoder, result_encoder, result_decoder)
            def expensive_computation(x, y):
                return x * y + complex_operation()
        """
        if custom_result_encoder and not custom_result_decoder:
            raise exceptions.CustomResultDecoderRequiredError(
                "Custom result encoder provided without custom result decoder."
            )
        elif not custom_result_encoder and custom_result_decoder:
            raise exceptions.CustomResultEncoderRequiredError(
                "Custom result decoder provided without custom result encoder."
            )
        return partial(
            self._make_record_wrapper,
            ischema=ischema,
            custom_params_encoder=custom_params_encoder,
            custom_result_encoder=custom_result_encoder,
            custom_result_decoder=custom_result_decoder,
        )


class _FastCacheIntf(_CacheIntf):
    """Fast cache interface optimizing for speed using in-memory + on-disk storage.
    
    This implementation prioritizes speed over consistency by using a two-tier
    caching strategy: fast in-memory cache as the primary layer and persistent
    on-disk cache as the secondary layer. The implementation ensures both layers
    remain synchronized.
    
    Features:
    - Fast reads from in-memory cache (primary tier)
    - Persistent storage via on-disk cache (secondary tier)
    - Per-function cache isolation using separate subdirectories
    - Eventual consistency between memory and disk layers
    - Optimized for single-machine scenarios where speed is critical
    
    Cache behavior:
    - get(): Checks in-memory first, falls back to on-disk
    - insert(): Writes to disk first, then mirrors to memory for consistency
    """
    
    def __init__(self: Self) -> None:
        """Initialize fast cache with in-memory and per-function on-disk caches."""
        super().__init__()
        self._imc: impls._InMemoryCacheImpl = impls._InMemoryCacheImpl()
        self._callee_to_odc: dict[str, impls._OnDiskCacheImpl] = {}
    
    def _get_odc_from_callee(self: Self, callee: str) -> impls._OnDiskCacheImpl:
        """Get or create an on-disk cache for the specified function.
        
        Creates function-specific on-disk caches using subdirectories for isolation.
        Each function gets its own cache directory to prevent key conflicts and
        improve cache organization.
        
        Args:
            callee: The function name to get/create cache for
            
        Returns:
            On-disk cache implementation for the specified function
            
        Example:
            odc = fast_cache._get_odc_from_callee("expensive_computation")
            # Creates cache in subdirectory "expensive_computation"
        """
        odc: Optional[impls._OnDiskCacheImpl] = self._callee_to_odc.get(callee)
        if not odc:
            odc = impls._OnDiskCacheImpl(sub_dir=callee)
            self._callee_to_odc[callee] = odc
        return odc
    
    @override
    def _make_record_wrapper(
        intf: Self,
        fn: Callable[P, R],
        ischema: context.IsolationSchema = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[P, R]:
        """Create a function wrapper that implements fast caching with read-through behavior.
        
        The wrapper prioritizes speed by checking in-memory cache first, then on-disk cache.
        For cache misses, it executes the function, stores the result in both cache layers,
        and returns the computed result.
        
        Args:
            intf: The cache interface instance
            fn: The function to wrap with caching
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for function parameters
            custom_result_encoder: Optional encoder for function results
            custom_result_decoder: Optional decoder for cached results
            
        Returns:
            A wrapped version of the function with fast caching behavior
        """
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with intf.lock():
                get: impls.Get = intf.get(
                    fn,
                    (args, kwargs,),
                    ischema,
                    custom_params_encoder=custom_params_encoder,
                    custom_result_decoder=custom_result_decoder,
                )

                if get.hit:
                    return get.value
                else:
                    result: R = fn(*args, **kwargs)
                    intf.insert(
                        fn,
                        (args, kwargs,),
                        result,
                        ischema,
                        custom_params_encoder,
                        custom_result_encoder,
                    )
                    return result
        return wrapper
    
    @override
    def get(
        self: Self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> impls.Get:
        """Retrieve cached result using fast two-tier lookup strategy.
        
        Implements a fast cache lookup by checking the in-memory cache first
        (fastest), then falling back to the on-disk cache. Both cache layers
        are kept synchronized to ensure consistency.
        
        Args:
            fn: The function whose result is being retrieved
            params: The function parameters (args, kwargs tuple)
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for parameter serialization
            custom_result_decoder: Optional decoder for cached results
            
        Returns:
            Get result with cache hit/miss status and value if found
            
        Raises:
            CustomParamsEncoderRequiredError: If parameters can't be encoded without custom encoder
            
        Cache lookup strategy:
        1. Check in-memory cache first (fastest)
        2. If miss, check function-specific on-disk cache
        3. Return cache miss if not found in either layer
        """
        key: Any = self._make_key(fn, params, ischema, custom_params_encoder)
        odc: impls._OnDiskCacheImpl = self._get_odc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, odc):
            try:
                # we'll check the memoization first, since that is much faster
                # than checking the on-disk cache (and the two should be consistent
                # regardless)
                imc_get: impls.Get = self._imc.get(key)
                if imc_get.hit:
                    if custom_result_decoder:
                        return impls.Get(hit=True, value=custom_result_decoder(imc_get.value))
                    else:
                        return imc_get
                else:
                    odc_get: impls.Get = odc.get(key)
                    if custom_result_decoder:
                        return impls.Get(hit=True, value=custom_result_decoder(odc_get.value))
                    return odc_get
                return impls.Get(hit=False)
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err

    @override
    def insert(
        self: Self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> impls.Insert:
        """Store result in cache using write-through strategy for consistency.
        
        Implements write-through caching by writing to persistent on-disk storage first,
        then mirroring to in-memory cache. This ensures both cache layers remain
        synchronized and the persistent layer serves as the authoritative source.
        
        Args:
            fn: The function whose result is being cached
            params: The function parameters (args, kwargs tuple)
            result: The function result to cache
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for parameter serialization
            custom_result_encoder: Optional encoder for the result
            
        Returns:
            Insert result indicating whether the insertion succeeded
            
        Raises:
            CustomParamsEncoderRequiredError: If parameters can't be encoded without custom encoder
            CustomResultEncoderRequiredError: If result can't be encoded without custom encoder
            
        Write-through strategy:
        1. Encode result using custom encoder if provided
        2. Write to on-disk cache first (authoritative source)
        3. If on-disk write succeeds, mirror to in-memory cache
        4. Both caches remain synchronized for fast future reads
        """
        key: Any = self._make_key(fn, params, ischema, custom_params_encoder)
        odc: impls._OnDiskCacheImpl = self._get_odc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, odc):
            try:
                encoded_result: Any = result if not custom_result_encoder else custom_result_encoder(result)
                # reverse order of get, as we don't want to memoize values
                # if we haven't actually inserted them into the on-disk cache
                # so that the memoization and the on-disk cache remain consistent
                odc_insert: impls.Insert = odc.insert(key, encoded_result)
                if odc_insert.inserted:
                    assert self._imc.insert(key, encoded_result)
                    return impls.Insert(inserted=True)
                return impls.Insert(inserted=False)
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err
            except exceptions.ValueEncodingError as err:
                raise exceptions.CustomResultEncoderRequiredError(
                    f"Custom result encoder required for function {fn} with parameters {params} and result {result}."
                ) from err


class _DeterministicCacheIntf(_CacheIntf):
    """Deterministic cache interface providing strong consistency across distributed systems.
    
    This implementation prioritizes consistency and determinism over speed by using
    a remote cache (ZippyDB) as the authoritative source with local in-memory caching
    for performance. The interface ensures all distributed workers see the same cached
    values through strong consistency guarantees.
    
    Features:
    - Strong consistency through remote cache backend
    - Local in-memory caching for performance optimization  
    - Deterministic behavior across distributed compilation workers
    - Conflict resolution through "first writer wins" semantics
    - Automatic fallback to remote cache on local cache misses
    
    Cache behavior:
    - get(): Checks local memory first, then remote cache with strong consistency
    - insert(): Uses remote cache's atomic operations for conflict detection
    - Deterministic resolution: If insert fails, re-read from remote cache
    
    This interface is specifically designed for distributed compilation scenarios
    where multiple workers must produce identical results.
    """
    
    def __init__(self: Self) -> None:
        """Initialize deterministic cache with local memory and remote storage.
        
        Creates both in-memory cache for speed and remote cache for consistency.
        Validates that remote cache provides strong consistency guarantees required
        for deterministic behavior.
        
        Raises:
            DeterministicCachingRequiresStrongConsistency: If remote cache doesn't provide strong consistency
        """
        super().__init__()
        self._imc: impls._InMemoryCacheImpl = impls._InMemoryCacheImpl()
        self._rc: impls._RemoteCacheImpl = impls._RemoteCacheImpl()
        if not self._rc.has_strong_consistency:
            raise exceptions.DeterministicCachingRequiresStrongConsistency()
    
    @override
    def _make_record_wrapper(
        intf: Self,
        fn: Callable[P, R],
        ischema: context.IsolationSchema = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[P, R]:
        """Create a function wrapper with deterministic conflict resolution for distributed caching.
        
        The wrapper implements "first writer wins" semantics with deterministic conflict resolution.
        If multiple workers attempt to cache the same key simultaneously, the first successful
        write wins, and subsequent attempts use the cached result to ensure deterministic behavior.
        
        Conflict resolution strategy:
        1. Check cache for existing result
        2. If miss, execute function and attempt to cache result
        3. If cache write fails (conflict), re-read from cache and use that result
        4. This ensures all workers return the same value for identical inputs
        
        Args:
            intf: The cache interface instance
            fn: The function to wrap with deterministic caching
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for function parameters
            custom_result_encoder: Optional encoder for function results
            custom_result_decoder: Optional decoder for cached results
            
        Returns:
            A wrapped version of the function with deterministic caching behavior
        """
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with intf.lock():
                get: impls.Get = intf.get(
                    fn,
                    (args, kwargs,),
                    ischema,
                    custom_params_encoder=custom_params_encoder,
                    custom_result_decoder=custom_result_decoder,
                )

                if get.hit:
                    return get.value
                else:
                    result: R = fn(*args, **kwargs)
                    if not intf.insert(
                        fn,
                        (args, kwargs,),
                        result,
                        ischema,
                        custom_params_encoder,
                        custom_result_encoder,
                    ).inserted:
                        # if we couldn't insert that means that some other callee has populated
                        # the key entry in the remote cache within the time between our first get
                        # and the insert attempt; in that case, to be deterministic, we should
                        # call get again and return that value as the assumption is that other
                        # compile workers will also use that value
                        get: impls.Get = intf.get(
                            fn,
                            (args, kwargs,),
                            ischema,
                            custom_params_encoder=custom_params_encoder,
                            custom_result_decoder=custom_result_decoder,
                        )
                        assert get.hit, "remote cache should get(key) if insert(key, _) failed"
                        return get.value
                    return result
        return wrapper

    @override
    def get(
        self: Self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> impls.Get:
        """Retrieve cached result using deterministic two-tier lookup with strong consistency.
          
        Implements cache lookup prioritizing local in-memory cache for speed while using
        remote cache as the authoritative source for strong consistency. The remote cache
        provides deterministic behavior across distributed workers.
          
        Args:
            fn: The function whose result is being retrieved
            params: The function parameters (args, kwargs tuple)
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for parameter serialization
            custom_result_decoder: Optional decoder for cached results
              
        Returns:
            Get result with cache hit/miss status and value if found
              
        Raises:
            CustomParamsEncoderRequiredError: If parameters can't be encoded without custom encoder
              
        Cache lookup strategy:
        1. Check local in-memory cache first (fastest)
        2. If miss, check remote cache with strong consistency guarantees
        3. Return cache miss if not found in either layer
          
        The remote cache provides strong consistency ensuring all distributed workers
        see the same cached values for deterministic compilation behavior.
        """
        key: Any = self._make_key(fn, params, ischema, custom_params_encoder)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, self._rc):
            try:
                # we'll check the memoization first, since that is much faster
                # than checking the remote cache and the two should be consistent
                imc_get: impls.Get = self._imc.get(key)
                if imc_get.hit:
                    if custom_result_decoder:
                        return impls.Get(hit=True, value=custom_result_decoder(imc_get.value))
                    else:
                        return imc_get
                else:
                    rc_get: impls.Get = self._rc.get(key)
                    if custom_result_decoder:
                        return impls.Get(hit=True, value=custom_result_decoder(rc_get.value))
                    return rc_get
                return impls.Get(hit=False)
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err
    
    @override
    def insert(
        self: Self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> impls.Insert:
        """Store result in cache using atomic remote operations for deterministic conflict resolution.
        
        Implements atomic write operations through the remote cache backend to provide
        "first writer wins" semantics for distributed caching. If multiple workers attempt
        to cache the same key simultaneously, only the first write succeeds, ensuring
        deterministic behavior across all distributed compilation workers.
        
        Args:
            fn: The function whose result is being cached
            params: The function parameters (args, kwargs tuple)
            result: The function result to cache
            ischema: Isolation schema for cache key generation
            custom_params_encoder: Optional encoder for parameter serialization
            custom_result_encoder: Optional encoder for the result
            
        Returns:
            Insert result indicating whether the insertion succeeded
            
        Raises:
            CustomParamsEncoderRequiredError: If parameters can't be encoded without custom encoder
            CustomResultEncoderRequiredError: If result can't be encoded without custom encoder
            
        Atomic write strategy:
        1. Encode result using custom encoder if provided
        2. Attempt atomic write to remote cache first (authoritative source)
        3. If remote write succeeds, mirror to local in-memory cache
        4. If remote write fails (key already exists), return insertion failure
        5. Both caches remain synchronized for consistent future reads
        
        This ensures deterministic behavior: if two workers compute and attempt to cache
        the same key simultaneously, both will end up using the same cached value through
        the conflict resolution mechanism in the wrapper function.
        """
        key: Any = self._make_key(fn, params, ischema, custom_params_encoder)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, self._rc):
            try:
                encoded_result: Any = result if not custom_result_encoder else custom_result_encoder(result)
                # reverse order of get, as we don't want to memoize values
                # if we haven't actually inserted them into the remote cache
                # so that the memoization and the remote cache remain consistent
                rc_insert: impls.Insert = self._rc.insert(key, encoded_result)
                if rc_insert.inserted:
                    assert self._imc.insert(key, encoded_result)
                    return impls.Insert(inserted=True)
                return impls.Insert(inserted=False)
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err
            except exceptions.ValueEncodingError as err:
                raise exceptions.CustomResultEncoderRequiredError(
                    f"Custom result encoder required for function {fn} with parameters {params} and result {result}."
                ) from err
