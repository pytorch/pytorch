from functools import cache
from threading import Lock
from typing import Optional

from . import config, interfaces as intfs, locks
from .context import IsolationSchema, SelectedCompileContext, SelectedRuntimeContext
from .exceptions import (
    CacheError,
    FileLockTimeoutError,
    KeyEncodingError,
    KeyPicklingError,
    LockTimeoutError,
    SystemError,
    UserError,
    ValueDecodingError,
    ValueEncodingError,
    ValuePicklingError,
    ValueUnPicklingError,
    CustomParamsEncoderRequiredError,
    CustomResultEncoderRequiredError,
    CustomResultDecoderRequiredError,
    DeterministicCachingDisabledError,
    DeterministicCachingRequiresStrongConsistencyError,
    StrictDeterministicCachingKeyNotFoundError,
    DeterministicCachingInvalidConfigurationError
)


fcache: intfs._CacheIntf = intfs._FastCacheIntf()
dcache: intfs._CacheIntf = intfs._DeterministicCacheIntf()

_ICACHE: Optional[intfs._CacheIntf] = None
_ICACHE_LOCK: Lock = Lock()


@cache
def get_icache() -> intfs._CacheIntf:
    global _ICACHE
    with locks._acquire_lock_with_timeout(_ICACHE_LOCK):
        if _ICACHE is None:
            _ICACHE = dcache if config.IS_DETERMINISTIC_CACHING_ENABLED() else fcache
    return _ICACHE


__all__ = [
    "SelectedCompileContext",
    "SelectedRuntimeContext",
    "IsolationSchema",
    "CacheError",
    "SystemError",
    "UserError",
    "LockTimeoutError",
    "FileLockTimeoutError",
    "KeyEncodingError",
    "KeyPicklingError",
    "ValueEncodingError",
    "ValuePicklingError",
    "ValueDecodingError",
    "ValueUnPicklingError",
    "CustomParamsEncoderRequiredError",
    "CustomResultEncoderRequiredError",
    "CustomResultDecoderRequiredError",
    "DeterministicCachingDisabledError",
    "DeterministicCachingRequiresStrongConsistencyError",
    "StrictDeterministicCachingKeyNotFoundError",
    "DeterministicCachingInvalidConfigurationError",
    "fcache",
    "dcache",
    "get_icache",
]
