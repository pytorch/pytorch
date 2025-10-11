from __future__ import annotations

import atexit
import json
import os
from abc import ABC, abstractmethod
from ast import literal_eval
from functools import partial, wraps
from logging import DEBUG, getLogger, Logger
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, Callable, Optional, override

from filelock import FileLock

from . import config, context, exceptions, implementations as impls, locks
from .utils import P, R


logger: Logger = getLogger(__name__)


def _intf_callback(
    origin: str,
    action: str,
    dur: float,
    fn: Callable[P, R],
    params: P,
    *args: Any,
) -> None:
    if origin == "record":
        result: R = args[0]
        if action == "replay":
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r cached, " \
                "returned result %r in %f seconds.",
                fn.__name__,
                params,
                result,
                dur,
            )
        elif action == "record_inserted":
            fn_dur: float = args[1]
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r not cached, " \
                "calculated and cached result %r in %f seconds " \
                "of which %f seconds was spent on the function call.",
                fn.__name__,
                params,
                result,
                dur,
                fn_dur,
            )
        elif action == "record_not_inserted":
            fn_dur: float = args[1]
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r not cached, " \
                "calculated result %r but was not able to " \
                "insert it into the cache as a matching " \
                "entry already exists; returned calculated result in %f seconds " \
                "of which %f seconds was spent on the function call.",
                fn.__name__,
                params,
                result,
                dur,
                fn_dur,
            )
        elif action == "record_not_inserted_replay":
            fn_dur: float = args[1]
            cached_result: R = args[2]
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r not cached, " \
                "calculated result %r but was not able to " \
                "insert it into the synchronization cache as a matching " \
                "entry already exists; returned cached result %r in %f seconds " \
                "of which %f seconds was spent on the function call.",
                fn.__name__,
                params,
                result,
                cached_result,
                dur,
                fn_dur,
            )
    elif origin == "get":
        if action == "hit":
            result: R = args[0]
            logger.log(
                DEBUG,
                "[GET] for fn %s with params %r cached, " \
                "returned result %r in %f seconds.",
                fn.__name__,
                params,
                result,
                dur,
            )
        elif action == "miss":
            logger.log(
                DEBUG,
                "[GET] for fn %s with params %r not cached, " \
                "returned nothing in %f seconds.",
                fn.__name__,
                params,
                dur,
            )
        else:
            raise NotImplementedError
    elif origin == "insert":
        result: R = args[0]
        if action == "inserted":
            logger.log(
                DEBUG,
                "[INSERT] for fn %s with params %r and " \
                "result %r inserted in %f seconds.",
                fn.__name__,
                params,
                result,
                dur,
            )
        elif action == "not_inserted":
            logger.log(
                DEBUG,
                "[INSERT] for fn %s with params %r and " \
                "result %r not inserted in %f seconds as there is " \
                "already has a matching entry.",
                fn.__name__,
                params,
                result,
                dur,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


class _CacheIntf(ABC):    
    def __init__(self) -> None:
        self._lock: Lock = Lock()
    
    def _make_key(
        self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
    ) -> Any:
        callee: str = fn.__name__
        fkey: Any = (callee, params) if not custom_params_encoder else (callee, custom_params_encoder(params))
        ikey: Any = context._isolation_key(ischema if ischema is not None else context._DEFAULT_ISOLATION_SCHEMA)
        return (fkey, ikey)

    def _make_dummy_record_wrapper(self, fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def dummy_wrapper(*args: Any, **kwargs: Any) -> R:
            return fn(*args, **kwargs)
        return dummy_wrapper
    
    @abstractmethod
    def _make_record_wrapper(
        intf: _CacheIntf,
        fn: Callable[P, R],
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[P, R]:
        pass
    
    @abstractmethod
    def _get(
        self,
        fn: Callable[P, R],
        params: P,
        ischema: context.IsolationSchema,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Optional[impls.Hit]:
        pass
    
    @abstractmethod
    def _insert(
        self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: context.IsolationSchema,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> bool:
        pass
    
    @property
    def lock(self) -> Callable[[Optional[float]], locks._LockContextManager]:
        """Get a context manager for acquiring the file lock.

        Uses file locking to ensure thread safety across processes.

        Args:
            timeout: Optional timeout in seconds (float) for acquiring the file lock.

        Returns:
            A callable that returns a context manager for the file lock.
        """

        def _lock_with_timeout(
            timeout: Optional[float] = None,
        ) -> locks._LockContextManager:
            return locks._acquire_lock_with_timeout(self._lock, timeout)

        return _lock_with_timeout

    def get(
        self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Optional[impls.Hit]:
        if not config.IS_CACHING_MODULE_ENABLED():
            return None

        start_t: float = time()
        with self.lock():
            result: Optional[impls.Hit] = self._get(fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder, custom_result_decoder=custom_result_decoder)
        dur: float = time() - start_t

        _intf_callback(
            "get",
            "hit" if result else "miss",
            dur,
            fn,
            params,
            *((result.value,) if result else ()),
        )

        return result

    def insert(
        self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> bool:
        if not config.IS_CACHING_MODULE_ENABLED():
            return False
        
        start_t: float = time()
        with self.lock():
            inserted: bool = self._insert(fn, params, result, ischema=ischema, custom_params_encoder=custom_params_encoder, custom_result_encoder=custom_result_encoder)
        dur: float = time() - start_t

        _intf_callback(
            "insert",
            "inserted" if inserted else "not_inserted",
            dur,
            fn,
            params,
            result,
        )

        return inserted

    def record(
        self,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[Callable[P, R], Callable[P, R]]:
        if custom_result_encoder and not custom_result_decoder:
            raise exceptions.CustomResultDecoderRequiredError(
                "Custom result encoder provided without custom result decoder."
            )
        elif not custom_result_encoder and custom_result_decoder:
            raise exceptions.CustomResultEncoderRequiredError(
                "Custom result decoder provided without custom result encoder."
            )
        elif not config.IS_CACHING_MODULE_ENABLED():
            return self._make_dummy_record_wrapper
        else:
            return partial(
                self._make_record_wrapper,
                ischema=ischema,
                custom_params_encoder=custom_params_encoder,
                custom_result_encoder=custom_result_encoder,
                custom_result_decoder=custom_result_decoder,
            )


class _FastCacheIntf(_CacheIntf):
    def __init__(self) -> None:
        super().__init__()
        self._imc: impls._InMemoryCacheImpl = impls._InMemoryCacheImpl()
        self._callee_to_odc: dict[str, impls._OnDiskCacheImpl] = {}
    
    def _get_odc_from_callee(self, callee: str) -> impls._OnDiskCacheImpl:
        if not (odc := self._callee_to_odc.get(callee)):
            odc = impls._OnDiskCacheImpl(sub_dir=callee)
            self._callee_to_odc[callee] = odc
        return odc

    @override
    def _make_record_wrapper(
        intf: _CacheIntf,
        fn: Callable[P, R],
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_t: float = time()
            params: P = (args, kwargs,)
            with intf.lock():
                get: Optional[impls.Hit] = intf._get(
                    fn,
                    params,
                    ischema=ischema,
                    custom_params_encoder=custom_params_encoder,
                    custom_result_decoder=custom_result_decoder,
                )

                if get:
                    dur: float = time() - start_t
                    _intf_callback(
                        "record",
                        "replay",
                        dur,
                        fn,
                        params,
                        get.value,
                    )
                    return get.value
                else:
                    fn_start_t: float = time()
                    result: R = fn(*args, **kwargs)
                    fn_dur: float = time() - fn_start_t
                    inserted: bool = intf._insert(
                        fn,
                        params,
                        result,
                        ischema=ischema,
                        custom_params_encoder=custom_params_encoder,
                        custom_result_encoder=custom_result_encoder,
                    )
                    dur: float = time() - start_t
                    _intf_callback(
                        "record",
                        "record_inserted" if inserted else "record_not_inserted",
                        dur,
                        fn,
                        params,
                        result,
                        fn_dur,
                    )
                    return result
        return wrapper
    
    @override
    def _get(
        self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> impls.Get:
        key: Any = self._make_key(fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder)
        odc: impls._OnDiskCacheImpl = self._get_odc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, odc):
            try:
                # we'll check the memoization first, since that is much faster
                # than checking the on-disk cache (and the two should be consistent
                # regardless)
                imc_get: Optional[impls.Hit] = self._imc.get(key)
                if imc_get:
                    if custom_result_decoder:
                        return impls.Hit(value=custom_result_decoder(imc_get.value))
                    else:
                        return imc_get
                else:
                    odc_get: Optional[impls.Hit] = odc.get(key)
                    if odc_get:
                        if custom_result_decoder:
                            return impls.Hit(value=custom_result_decoder(odc_get.value))
                        return odc_get
                return None
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err

    @override
    def _insert(
        self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> impls.Insert:
        key: Any = self._make_key(fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder)
        odc: impls._OnDiskCacheImpl = self._get_odc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, odc):
            try:
                encoded_result: Any = result if not custom_result_encoder else custom_result_encoder(result)
                # reverse order of get, as we don't want to memoize values
                # if we haven't actually inserted them into the on-disk cache
                # so that the memoization and the on-disk cache remain consistent
                if odc.insert(key, encoded_result):
                    assert self._imc.insert(key, encoded_result)
                    return True
                return False
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err
            except exceptions.ValueEncodingError as err:
                raise exceptions.CustomResultEncoderRequiredError(
                    f"Custom result encoder required for function {fn} with parameters {params} and result {result}."
                ) from err


class _DeterministicCacheIntf(_CacheIntf):
    def __init__(self) -> None:
        super().__init__()
        self._imc: impls._InMemoryCacheImpl = impls._InMemoryCacheImpl()

        if (fpath := os.environ.get("TORCHINDUCTOR_PRE_POPULATE_DETERMINISTIC_CACHE")):
            flock: FileLock = FileLock(str(fpath) + ".lock")
            with locks._acquire_flock_with_timeout(flock):
                with open(fpath, "r") as fp:
                    dump_for_pre_population: dict[str, str] = json.load(fp)
                for key_r, value_r in dump_for_pre_population.items():
                    key: bytes = literal_eval(key_r)
                    value: bytes = literal_eval(value_r)
                    assert key not in self._imc._memory
                    self._imc._memory[key] = value

        if config.STRICTLY_PRE_POPULATED_DETERMINISM:
            self._get_sc_from_callee: Callable[[str], None] = lambda callee: None
        elif config.GLOBAL_DETERMINISM:
            self._rc: impls._RemoteCacheImpl = impls._RemoteCacheImpl()
            if not self._rc.has_strong_consistency:
                raise exceptions.DeterministicCachingRequiresStrongConsistencyError
            self._get_sc_from_callee: Callable[[str], impls._RemoteCacheImpl] = lambda callee: self._rc
        elif config.LOCAL_DETERMINISM:
            self._callee_to_odc: dict[str, impls._OnDiskCacheImpl] = {}
            self._get_sc_from_callee: Callable[[str], impls._OnDiskCacheImpl] = lambda callee: self._get_odc_from_callee(callee)
        else:
            raise exceptions.DeterministicCachingInvalidConfigurationError(
                "Deterministic caching must specify at least one of STRICTLY_PRE_POPULATED_DETERMINISM, " \
                "GLOBAL_DETERMINISM, or LOCAL_DETERMINISM."
            )
        
        atexit.register(self._dump_imc_to_disk)
    
    def __del__(self) -> None:
        atexit.unregister(self._dump_imc_to_disk)
        del self
        
    def _get_odc_from_callee(self, callee: str) -> impls._OnDiskCacheImpl:
        if not (odc := self._callee_to_odc.get(callee)):
            odc = impls._OnDiskCacheImpl(sub_dir=callee)
            self._callee_to_odc[callee] = odc
        return odc
    
    def _dump_imc_to_disk(self) -> Optional[Path]:
        with self.lock():
            to_dump: dict[str, str] = {
                repr(key): repr(value)
                for key, value in self._imc._memory.items()
            }
            if not to_dump:
                return None

            odc: impls._OnDiskCacheImpl = impls._OnDiskCacheImpl(sub_dir="dcache_dump")
            fpath: Path = odc._cache_dir / "imc.save"
            with odc.lock():
                r_fp, w_fp = None, None
                try:
                    w_fp = open(fpath, "x")
                except FileExistsError:
                    with open(fpath, "r") as r_fp:
                        existing_dump = json.load(r_fp)

                    for key, value in existing_dump.items():
                        if key not in to_dump:
                            to_dump[key] = value
                        else:
                            raise exceptions.DeterministicCachingIMCDumpConflict

                    w_fp = open(fpath, "w")
                finally:
                    assert w_fp is not None
                    try:
                        json.dump(to_dump, w_fp, indent=4)
                    finally:
                        w_fp.close()
        
        return fpath

    @override
    def _make_record_wrapper(
        intf: _CacheIntf,
        fn: Callable[P, R],
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_t: float = time()
            params: P = (args, kwargs,)
            with intf.lock():
                get: Optional[impls.Hit] = intf._get(
                    fn,
                    params,
                    ischema=ischema,
                    custom_params_encoder=custom_params_encoder,
                    custom_result_decoder=custom_result_decoder,
                )

                if get:
                    dur: float = time() - start_t
                    _intf_callback(
                        "record",
                        "replay",
                        dur,
                        fn,
                        params,
                        get.value,
                    )
                    return get.value
                else:
                    fn_start_t: float = time()
                    result: R = fn(*args, **kwargs)
                    fn_dur: float = time() - fn_start_t
                    if not intf._insert(
                        fn,
                        params,
                        result,
                        ischema,
                        custom_params_encoder,
                        custom_result_encoder,
                    ):
                        # if we couldn't insert that means that some other callee has populated
                        # the key entry in the remote cache within the time between our first get
                        # and the insert attempt; in that case, to be deterministic, we should
                        # call get again and return that value as the assumption is that other
                        # compile workers will also use that value
                        get: Optional[impls.Hit] = intf._get(
                            fn,
                            params,
                            ischema,
                            custom_params_encoder=custom_params_encoder,
                            custom_result_decoder=custom_result_decoder,
                        )
                        assert get is not None, "remote cache should get(key) if insert(key, _) failed"
                        dur: float = time() - start_t
                        _intf_callback(
                            "record",
                            "record_not_inserted_replay",
                            dur,
                            fn,
                            params,
                            fn_dur,
                            get.value,
                        )
                        return get.value
                    dur: float = time() - start_t
                    _intf_callback(
                        "record",
                        "record_inserted",
                        dur,
                        fn,
                        params,
                        result,
                        fn_dur,
                    )
                    return result
        return wrapper

    @override
    def _get(
        self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Optional[impls.Hit]:
        key: Any = self._make_key(fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder)
        sc: Optional[impls._OnDiskCacheImpl | impls._RemoteCacheImpl] = self._get_sc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(*([self._imc, sc] if sc else [self._imc])):
            try:
                # we'll check the memoization first, since that is much faster
                # than checking the remote cache and the two should be consistent
                imc_get: Optional[impls.Hit] = self._imc.get(key)
                if imc_get:
                    if custom_result_decoder:
                        return impls.Hit(value=custom_result_decoder(imc_get.value))
                    else:
                        return imc_get
                elif not sc:
                    raise exceptions.StrictDeterministicCachingKeyNotFoundError
                else:
                    sc_get: Optional[impls.Hit] = sc.get(key)
                    if sc_get:
                        if custom_result_decoder:
                            return impls.Hit(value=custom_result_decoder(sc_get.value))
                        return sc_get
                    elif config.STRICTLY_CACHED_DETERMINISM:
                        raise exceptions.StrictDeterministicCachingKeyNotFoundError
                return None
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err
    
    @override
    def _insert(
        self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> bool:
        if config.STRICTLY_PRE_POPULATED_DETERMINISM or config.STRICTLY_CACHED_DETERMINISM:
            raise exceptions.StrictDeterministicCachingInsertionError

        key: Any = self._make_key(fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder)
        sc: Optional[impls._OnDiskCacheImpl | impls._RemoteCacheImpl] = self._get_sc_from_callee(fn.__name__)
        assert sc, "sc should be either an on-disk cache or a remote cache if we're inserting"
        with locks._acquire_many_impl_locks_with_timeout(self._imc, sc):
            try:
                encoded_result: Any = result if not custom_result_encoder else custom_result_encoder(result)
                # reverse order of get, as we don't want to memoize values
                # if we haven't actually inserted them into the remote cache
                # so that the memoization and the remote cache remain consistent
                if sc.insert(key, encoded_result):
                    if not self._imc.insert(key, encoded_result):
                        # imc might have the mapping already, if pre-populated
                        assert self._imc.get(key) == encoded_result
                    return True
                return False
            except exceptions.KeyEncodingError as err:
                raise exceptions.CustomParamsEncoderRequiredError(fn, params) from err
            except exceptions.ValueEncodingError as err:
                raise exceptions.CustomResultEncoderRequiredError(
                    f"Custom result encoder required for function {fn} with parameters {params} and result {result}."
                ) from err

    @override
    def get(
        self,
        fn: Callable[P, R],
        params: P,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_decoder: Optional[Callable[Any, R]] = None,
    ) -> Optional[impls.Hit]:
        if not config.IS_DETERMINISTIC_CACHING_ENABLED:
            raise exceptions.DeterministicCachingIsDisabledError
        return super().get(fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder, custom_result_decoder=custom_result_decoder)
    
    @override
    def insert(
        self,
        fn: Callable[P, R],
        params: P,
        result: R,
        ischema: Optional[context.IsolationSchema] = None,
        custom_params_encoder: Optional[Callable[P, Any]] = None,
        custom_result_encoder: Optional[Callable[R, Any]] = None,
    ) -> bool:
        if not config.IS_DETERMINISTIC_CACHING_ENABLED:
            raise exceptions.DeterministicCachingIsDisabledError
        return super().insert(fn, params, result, ischema=ischema, custom_params_encoder=custom_params_encoder, custom_result_encoder=custom_result_encoder)
