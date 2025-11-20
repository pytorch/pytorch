from threading import Lock

from . import config, interfaces as intfs, locks
from .context import IsolationSchema, SelectedCompileContext, SelectedRuntimeContext
from .exceptions import (
    CacheError,
    CustomParamsEncoderRequiredError,
    CustomResultDecoderRequiredError,
    CustomResultEncoderRequiredError,
    DeterministicCachingDisabledError,
    DeterministicCachingIMCDumpConflictError,
    DeterministicCachingInvalidConfigurationError,
    DeterministicCachingRequiresStrongConsistencyError,
    FileLockTimeoutError,
    KeyEncodingError,
    KeyPicklingError,
    LockTimeoutError,
    StrictDeterministicCachingKeyNotFoundError,
    SystemError,
    UserError,
    ValueDecodingError,
    ValueEncodingError,
    ValuePicklingError,
    ValueUnPicklingError,
)


# fast cache; does not bother supporting deterministic caching, and is essentially
# a memoized on-disk cache. use when deterministic caching is not required
fcache: intfs._CacheIntf = intfs._FastCacheIntf()
# deterministic cache; slower than fcache but provides deterministic guarantees.
# use when deterministic caching is absolutely required, as this will raise
# an exception if use is attempted when deterministic caching is disabled
dcache: intfs._CacheIntf = intfs._DeterministicCacheIntf()
# inductor cache; defaults to the deterministic cache if deterministic caching
# is enabled, otherwise uses the fast cache. use when you would like deterministic
# caching but are okay with non-deterministic caching if deterministic caching is disabled
icache: intfs._CacheIntf = (
    dcache if config.IS_DETERMINISTIC_CACHING_ENABLED() else fcache
)

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
    "DeterministicCachingIMCDumpConflictError",
    "fcache",
    "dcache",
    "icache",
]
