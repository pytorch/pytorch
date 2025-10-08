from . import config, interfaces as intfs
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
    DeterministicCachingRequiresStrongConsistency,
)


fcache: intfs._CacheIntf = intfs._FastCacheIntf()
dcache: intfs._CacheIntf = intfs._DeterministicCacheIntf()
icache: intfs._CacheIntf = dcache if config.DETERMINISTIC_CACHING else fcache

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
    "DeterministicCachingRequiresStrongConsistency",
    "fcache",
    "dcache",
    "icache",
]
