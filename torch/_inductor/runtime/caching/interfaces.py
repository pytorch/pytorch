from __future__ import annotations

import atexit
import json
import os
from abc import ABC, abstractmethod
from ast import literal_eval
from enum import Enum
from functools import partial, wraps
from logging import DEBUG, getLogger, Logger
from os import PathLike
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, TYPE_CHECKING, TypeAlias
from typing_extensions import override

from filelock import FileLock

from . import config, context, exceptions, implementations as impls, locks


if TYPE_CHECKING:
    from collections.abc import Callable

    from .utils import P, R


# ideally we could annotate this as tuple[P.args, P.kwargs] but
# functionally that doesn't work as P is defined in a specific
# scope and P.args/P.kwargs are only valid in that scope
Params: TypeAlias = tuple[Any, Any]

logger: Logger = getLogger(__name__)


class _IntfCallbackOrigin(Enum):
    RECORD = "record"
    GET = "get"
    INSERT = "insert"


class _IntfCallbackAction(Enum):
    REPLAY = "replay"
    RECORD_INSERTED = "record_inserted"
    RECORD_NOT_INSERTED = "record_not_inserted"
    RECORD_NOT_INSERTED_REPLAY = "record_not_inserted_replay"
    HIT = "hit"
    MISS = "miss"
    INSERTED = "inserted"
    NOT_INSERTED = "not_inserted"


def _intf_callback(
    origin: _IntfCallbackOrigin,
    action: _IntfCallbackAction,
    dur: float,
    fn: Callable[P, R],
    params: Params,
    *args: Any,
) -> None:
    if origin == _IntfCallbackOrigin.RECORD:
        result: R = args[0]
        if action == _IntfCallbackAction.REPLAY:
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r cached, "
                "returned result %r in %f seconds.",
                fn.__name__,
                params,
                result,
                dur,
            )
        elif action == _IntfCallbackAction.RECORD_INSERTED:
            fn_dur: float = args[1]
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r not cached, "
                "calculated and cached result %r in %f seconds "
                "of which %f seconds was spent on the function call.",
                fn.__name__,
                params,
                result,
                dur,
                fn_dur,
            )
        elif action == _IntfCallbackAction.RECORD_NOT_INSERTED:
            fn_dur = args[1]
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r not cached, "
                "calculated result %r but was not able to "
                "insert it into the cache as a matching "
                "entry already exists; returned calculated result in %f seconds "
                "of which %f seconds was spent on the function call.",
                fn.__name__,
                params,
                result,
                dur,
                fn_dur,
            )
        elif action == _IntfCallbackAction.RECORD_NOT_INSERTED_REPLAY:
            fn_dur = args[1]
            cached_result: R = args[2]
            logger.log(
                DEBUG,
                "[RECORD] for fn %s with params %r not cached, "
                "calculated result %r but was not able to "
                "insert it into the synchronization cache as a matching "
                "entry already exists; returned cached result %r in %f seconds "
                "of which %f seconds was spent on the function call.",
                fn.__name__,
                params,
                result,
                cached_result,
                dur,
                fn_dur,
            )
        else:
            raise NotImplementedError
    elif origin == _IntfCallbackOrigin.GET:
        if action == _IntfCallbackAction.HIT:
            result = args[0]
            logger.log(
                DEBUG,
                "[GET] for fn %s with params %r cached, "
                "returned result %r in %f seconds.",
                fn.__name__,
                params,
                result,
                dur,
            )
        elif action == _IntfCallbackAction.MISS:
            logger.log(
                DEBUG,
                "[GET] for fn %s with params %r not cached, "
                "returned nothing in %f seconds.",
                fn.__name__,
                params,
                dur,
            )
        else:
            raise NotImplementedError
    elif origin == _IntfCallbackOrigin.INSERT:
        result = args[0]
        if action == _IntfCallbackAction.INSERTED:
            logger.log(
                DEBUG,
                "[INSERT] for fn %s with params %r and "
                "result %r inserted in %f seconds.",
                fn.__name__,
                params,
                result,
                dur,
            )
        elif action == _IntfCallbackAction.NOT_INSERTED:
            logger.log(
                DEBUG,
                "[INSERT] for fn %s with params %r and "
                "result %r not inserted in %f seconds as there is "
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
        params: Params,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
    ) -> Any:
        callee: str = fn.__name__
        fkey: Any = (
            (callee, params)
            if not custom_params_encoder
            # pyrefly: ignore [invalid-param-spec]
            else (callee, custom_params_encoder(*params[0], **params[1]))
        )
        ikey: Any = context._isolation_key(
            ischema if ischema is not None else context._DEFAULT_ISOLATION_SCHEMA
        )
        return (fkey, ikey)

    def _make_dummy_record_wrapper(self, fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def dummy_wrapper(*args: Any, **kwargs: Any) -> R:
            # pyrefly: ignore [invalid-param-spec]
            return fn(*args, **kwargs)

        # pyrefly: ignore [bad-return]
        return dummy_wrapper

    @abstractmethod
    def _make_record_wrapper(
        self,
        fn: Callable[P, R],
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> Callable[P, R]:
        pass

    @abstractmethod
    def _get(
        self,
        fn: Callable[P, R],
        params: Params,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> impls.Hit | None:
        pass

    @abstractmethod
    def _insert(
        self,
        fn: Callable[P, R],
        params: Params,
        result: R,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
    ) -> bool:
        pass

    @property
    def lock(self) -> locks._LockProtocol:
        """Get a context manager for acquiring the file lock.

        Uses file locking to ensure thread safety across processes.

        Args:
            timeout: Optional timeout in seconds (float) for acquiring the file lock.

        Returns:
            A callable that returns a context manager for the file lock.
        """

        def _lock_with_timeout(
            timeout: float | None = None,
        ) -> locks._LockContextManager:
            return locks._acquire_lock_with_timeout(self._lock, timeout)

        return _lock_with_timeout

    def get(
        self,
        fn: Callable[P, R],
        params: Params,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> impls.Hit | None:
        if not config.IS_CACHING_MODULE_ENABLED():
            return None

        start_t: float = time()
        with self.lock():  # type: ignore[call-arg]
            result: impls.Hit | None = self._get(
                fn,
                params,
                ischema=ischema,
                custom_params_encoder=custom_params_encoder,
                custom_result_decoder=custom_result_decoder,
            )
        dur: float = time() - start_t

        _intf_callback(
            _IntfCallbackOrigin.GET,
            _IntfCallbackAction.HIT if result else _IntfCallbackAction.MISS,
            dur,
            fn,
            params,
            *((result.value,) if result else ()),
        )

        return result

    def insert(
        self,
        fn: Callable[P, R],
        params: Params,
        result: R,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
    ) -> bool:
        if not config.IS_CACHING_MODULE_ENABLED():
            return False

        start_t: float = time()
        with self.lock():  # type: ignore[call-arg]
            inserted: bool = self._insert(
                fn,
                params,
                result,
                ischema=ischema,
                custom_params_encoder=custom_params_encoder,
                custom_result_encoder=custom_result_encoder,
            )
        dur: float = time() - start_t

        _intf_callback(
            _IntfCallbackOrigin.INSERT,
            _IntfCallbackAction.INSERTED
            if inserted
            else _IntfCallbackAction.NOT_INSERTED,
            dur,
            fn,
            params,
            result,
        )

        return inserted

    def record(
        self,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
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
            callee_sub_dir: PathLike[str] = Path(callee)
            odc = impls._OnDiskCacheImpl(sub_dir=callee_sub_dir)
            self._callee_to_odc[callee] = odc
        # pyrefly: ignore [unbound-name]
        return odc

    @override
    def _make_record_wrapper(
        self,
        fn: Callable[P, R],
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_t: float = time()
            params = (
                args,
                kwargs,
            )
            with self.lock():
                get: impls.Hit | None = self._get(
                    fn,
                    params,
                    ischema=ischema,
                    custom_params_encoder=custom_params_encoder,
                    custom_result_decoder=custom_result_decoder,
                )

                if get:
                    dur: float = time() - start_t
                    _intf_callback(
                        _IntfCallbackOrigin.RECORD,
                        _IntfCallbackAction.REPLAY,
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
                    inserted: bool = self._insert(
                        fn,
                        params,
                        result,
                        ischema=ischema,
                        custom_params_encoder=custom_params_encoder,
                        custom_result_encoder=custom_result_encoder,
                    )
                    dur = time() - start_t
                    _intf_callback(
                        _IntfCallbackOrigin.RECORD,
                        _IntfCallbackAction.RECORD_INSERTED
                        if inserted
                        else _IntfCallbackAction.RECORD_NOT_INSERTED,
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
        params: Params,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> impls.Hit | None:
        key: Any = self._make_key(
            fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder
        )
        odc: impls._OnDiskCacheImpl = self._get_odc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, odc):
            try:
                # we'll check the memoization first, since that is much faster
                # than checking the on-disk cache (and the two should be consistent
                # regardless)
                imc_get: impls.Hit | None = self._imc.get(key)
                if imc_get:
                    if custom_result_decoder:
                        return impls.Hit(value=custom_result_decoder(imc_get.value))
                    else:
                        return imc_get
                else:
                    odc_get: impls.Hit | None = odc.get(key)
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
        params: Params,
        result: R,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
    ) -> bool:
        key: Any = self._make_key(
            fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder
        )
        odc: impls._OnDiskCacheImpl = self._get_odc_from_callee(fn.__name__)
        with locks._acquire_many_impl_locks_with_timeout(self._imc, odc):
            try:
                encoded_result: Any = (
                    result
                    if not custom_result_encoder
                    else custom_result_encoder(result)
                )
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

        if fpath := os.environ.get("TORCHINDUCTOR_PRE_POPULATE_DETERMINISTIC_CACHE"):
            # pyrefly: ignore [bad-assignment]
            flock: FileLock = FileLock(str(fpath) + ".lock")
            with locks._acquire_flock_with_timeout(flock):
                with open(fpath) as fp:
                    dump_for_pre_population: dict[str, str] = json.load(fp)
                for key_r, value_r in dump_for_pre_population.items():
                    key: bytes = literal_eval(key_r)
                    value: bytes = literal_eval(value_r)
                    self._imc._memory[key] = value

        if config.STRICTLY_PRE_POPULATED_DETERMINISM:
            # we'll never need a synchronization cache if we're in strictly pre-populated mode,
            # as we'll only ever be checking the memoized pre-population
            self._get_sc_from_callee: Callable[
                [str], None | impls._OnDiskCacheImpl | impls._RemoteCacheImpl
            ] = lambda callee: None
        elif config.GLOBAL_DETERMINISM:
            # if we want global determinism we need to use a remote cache with strong
            # consistency as the synchronization cache
            self._rc: impls._RemoteCacheImpl = impls._RemoteCacheImpl()
            if not self._rc.has_strong_consistency:
                raise exceptions.DeterministicCachingRequiresStrongConsistencyError
            self._get_sc_from_callee = lambda callee: self._rc
        elif config.LOCAL_DETERMINISM:
            # local determinism can use the on-disk cache as the synchronization cache,
            # for cleanliness of the on-disk cache we subdir based on the callee
            self._callee_to_odc: dict[str, impls._OnDiskCacheImpl] = {}
            self._get_sc_from_callee = self._get_odc_from_callee
        else:
            raise exceptions.DeterministicCachingInvalidConfigurationError(
                "Deterministic caching must specify at least one of STRICTLY_PRE_POPULATED_DETERMINISM, "
                "GLOBAL_DETERMINISM, or LOCAL_DETERMINISM."
            )

        atexit.register(self._dump_imc_to_disk)

    def __del__(self) -> None:
        atexit.unregister(self._dump_imc_to_disk)
        del self

    def _get_odc_from_callee(self, callee: str) -> impls._OnDiskCacheImpl:
        if not (odc := self._callee_to_odc.get(callee)):
            callee_sub_dir: PathLike[str] = Path(callee)
            odc = impls._OnDiskCacheImpl(sub_dir=callee_sub_dir)
            self._callee_to_odc[callee] = odc
        # pyrefly: ignore [unbound-name]
        return odc

    def _dump_imc_to_disk(self) -> Path | None:
        with self.lock():  # type: ignore[call-arg]
            to_dump: dict[str, str] = {
                repr(key): repr(value) for key, value in self._imc._memory.items()
            }
            if not to_dump:
                return None

            odc: impls._OnDiskCacheImpl = impls._OnDiskCacheImpl(
                sub_dir=Path("dcache_dump")
            )
            fpath: Path = odc._cache_dir / "imc.save"
            with odc.lock():
                r_fp, w_fp = None, None
                try:
                    w_fp = open(fpath, "x")
                except FileExistsError:
                    with open(fpath) as r_fp:
                        existing_dump = json.load(r_fp)

                    for key, value in existing_dump.items():
                        if key not in to_dump:
                            to_dump[key] = value
                        else:
                            raise exceptions.DeterministicCachingIMCDumpConflictError from None

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
        self,
        fn: Callable[P, R],
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not config.IS_DETERMINISTIC_CACHING_ENABLED():
                raise exceptions.DeterministicCachingDisabledError
            start_t: float = time()
            params = (
                args,
                kwargs,
            )
            with self.lock():
                get: impls.Hit | None = self._get(
                    fn,
                    params,
                    ischema=ischema,
                    custom_params_encoder=custom_params_encoder,
                    custom_result_decoder=custom_result_decoder,
                )

                if get:
                    dur: float = time() - start_t
                    _intf_callback(
                        _IntfCallbackOrigin.RECORD,
                        _IntfCallbackAction.REPLAY,
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
                    if not self._insert(
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
                        get = self._get(
                            fn,
                            params,
                            ischema,
                            custom_params_encoder=custom_params_encoder,
                            custom_result_decoder=custom_result_decoder,
                        )
                        assert get is not None, (
                            "remote cache should get(key) if insert(key, _) failed"
                        )
                        dur = time() - start_t
                        _intf_callback(
                            _IntfCallbackOrigin.RECORD,
                            _IntfCallbackAction.RECORD_NOT_INSERTED_REPLAY,
                            dur,
                            fn,
                            params,
                            fn_dur,
                            get.value,
                        )
                        return get.value
                    dur = time() - start_t
                    _intf_callback(
                        _IntfCallbackOrigin.RECORD,
                        _IntfCallbackAction.RECORD_INSERTED,
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
        params: Params,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> impls.Hit | None:
        key: Any = self._make_key(
            fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder
        )
        sc: impls._OnDiskCacheImpl | impls._RemoteCacheImpl | None = (
            self._get_sc_from_callee(fn.__name__)
        )
        with locks._acquire_many_impl_locks_with_timeout(
            *([self._imc, sc] if sc else [self._imc])
        ):
            try:
                # we'll check the memoization first, since that is much faster
                # than checking the remote cache and the two should be consistent
                imc_get: impls.Hit | None = self._imc.get(key)
                if imc_get:
                    if custom_result_decoder:
                        return impls.Hit(value=custom_result_decoder(imc_get.value))
                    else:
                        return imc_get
                elif not sc:
                    raise exceptions.StrictDeterministicCachingKeyNotFoundError
                else:
                    sc_get: impls.Hit | None = sc.get(key)
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
        params: Params,
        result: R,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
    ) -> bool:
        if (
            config.STRICTLY_PRE_POPULATED_DETERMINISM
            or config.STRICTLY_CACHED_DETERMINISM
        ):
            raise exceptions.StrictDeterministicCachingInsertionError

        key: Any = self._make_key(
            fn, params, ischema=ischema, custom_params_encoder=custom_params_encoder
        )
        sc: impls._OnDiskCacheImpl | impls._RemoteCacheImpl | None = (
            self._get_sc_from_callee(fn.__name__)
        )
        assert sc, (
            "sc should be either an on-disk cache or a remote cache if we're inserting"
        )
        with locks._acquire_many_impl_locks_with_timeout(self._imc, sc):
            try:
                encoded_result: Any = (
                    result
                    if not custom_result_encoder
                    else custom_result_encoder(result)
                )
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
        params: Params,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_decoder: Callable[[Any], R] | None = None,
    ) -> impls.Hit | None:
        if not config.IS_DETERMINISTIC_CACHING_ENABLED():
            raise exceptions.DeterministicCachingDisabledError
        return super().get(
            fn,
            params,
            ischema=ischema,
            custom_params_encoder=custom_params_encoder,
            custom_result_decoder=custom_result_decoder,
        )

    @override
    def insert(
        self,
        fn: Callable[P, R],
        params: Params,
        result: R,
        ischema: context.IsolationSchema | None = None,
        custom_params_encoder: Callable[P, Any] | None = None,
        custom_result_encoder: Callable[[R], Any] | None = None,
    ) -> bool:
        if not config.IS_DETERMINISTIC_CACHING_ENABLED():
            raise exceptions.DeterministicCachingDisabledError
        return super().insert(
            fn,
            params,
            result,
            ischema=ischema,
            custom_params_encoder=custom_params_encoder,
            custom_result_encoder=custom_result_encoder,
        )
