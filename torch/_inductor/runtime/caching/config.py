import os
from collections.abc import Callable
from functools import cache, partial

import torch
from torch._environment import is_fbcode


@cache
def _env_var_config(env_var: str, default: bool) -> bool:
    if (env_val := os.environ.get(env_var)) is not None:
        return env_val == "1"
    return default


@cache
def _versioned_config(
    jk_name: str,
    this_version: int,
    oss_default: bool,
    env_var_override: str | None = None,
) -> bool:
    """
    A versioned configuration utility that determines boolean settings based on:
    1. Environment variable override (highest priority)
    2. JustKnobs version comparison in fbcode environments
    3. OSS default fallback

    This function enables gradual rollouts of features in fbcode by comparing
    a local version against a JustKnobs-controlled remote version, while
    allowing environment variable overrides for testing and OSS defaults
    for non-fbcode environments.

    Args:
        jk_name: JustKnobs key name (e.g., "pytorch/inductor:feature_version")
        this_version: Local version number to compare against JustKnobs version
        oss_default: Default value to use in non-fbcode environments
        env_var_override: Optional environment variable name that, when set,
                         overrides all other logic

    Returns:
        bool: Configuration value determined by the priority order above
    """
    if (
        env_var_override
        and (env_var_value := os.environ.get(env_var_override)) is not None
    ):
        return env_var_value == "1"
    elif is_fbcode():
        # this method returns 0 on failure, which we should check for specifically.
        # in the case of JK failure, the safe bet is to simply disable the config
        jk_version: int = torch._utils_internal.justknobs_getval_int(jk_name)
        return (this_version >= jk_version) and (jk_version != 0)
    return oss_default


# toggles the entire caching module, but only when calling through the
# public facing interfaces. get/insert operations become no-ops in the sense
# that get will always miss and insert will never insert; record becomes a
# no-op in the sense that the function will always be called and the cache
# will never be accessed
_CACHING_MODULE_VERSION: int = 0
_CACHING_MODULE_VERSION_JK: str = "pytorch/inductor:caching_module_version"
_CACHING_MODULE_OSS_DEFAULT: bool = False
_CACHING_MODULE_ENV_VAR_OVERRIDE: str = "TORCHINDUCTOR_ENABLE_CACHING_MODULE"
IS_CACHING_MODULE_ENABLED: Callable[[], bool] = partial(
    _versioned_config,
    _CACHING_MODULE_VERSION_JK,
    _CACHING_MODULE_VERSION,
    _CACHING_MODULE_OSS_DEFAULT,
    _CACHING_MODULE_ENV_VAR_OVERRIDE,
)


# toggles the deterministic caching interface. silently disabling deterministic
# caching (i.e. by mimicking the functionality of IS_CACHING_MODULE_ENABLED) can
# be problematic if the user is directly calling the deterministic caching interface
# (for example, if they were to interface with dcache instead of icache). instead, if
# the user tries to use the deterministic caching interface while it is disabled we
# will simply throw DeterministicCachingDisabledError
_DETERMINISTIC_CACHING_VERSION: int = 0
_DETERMINISTIC_CACHING_VERSION_JK: str = (
    "pytorch/inductor:deterministic_caching_version"
)
_DETERMINISTIC_CACHING_OSS_DEFAULT: bool = False
_DETERMINISTIC_CACHING_ENV_VAR_OVERRIDE: str = (
    "TORCHINDUCTOR_ENABLE_DETERMINISTIC_CACHING"
)
IS_DETERMINISTIC_CACHING_ENABLED: Callable[[], bool] = partial(
    _versioned_config,
    _DETERMINISTIC_CACHING_VERSION_JK,
    _DETERMINISTIC_CACHING_VERSION,
    _DETERMINISTIC_CACHING_OSS_DEFAULT,
    _DETERMINISTIC_CACHING_ENV_VAR_OVERRIDE,
)

# enabling strictly pre-populated determinism forces the deterministic caching
# interface to pull from and only from a pre-populated in-memory cache. this
# in-memory cache gets pre-populated from a file path that is specified by
# environment variable "TORCHINDUCTOR_PRE_POPULATE_DETERMINISTIC_CACHE".
# coincidentally, the deterministic caching interface will dump its in-memory
# cache to disk on program exit (check the logs for the exact file path) which
# can be used as a drop-in solution for pre-population on subsequent runs. if
# strictly pre-populated determinism is enabled and a key is encountered which
# is not covered by the pre-populated in-memory cache an exception,
# StrictDeterministicCachingKeyNotFoundError, will be raised
STRICTLY_PRE_POPULATED_DETERMINISM: bool = _env_var_config(
    "TORCHINDUCTOR_STRICTLY_PRE_POPULATED_DETERMINISM",
    default=False,
)
# similar to strictly pre-populated determinism, except that any key can either
# be in the pre-populated in-memory cache or the on-disk/remote cache (depending
# on whether or not local/global determinism is enabled).
STRICTLY_CACHED_DETERMINISM: bool = _env_var_config(
    "TORCHINDUCTOR_STRICTLY_CACHED_DETERMINISM",
    default=False,
)
# local determinism ensures that caching is deterministic on a single machine,
# hence an on-disk cache is used for synchronization of results
LOCAL_DETERMINISM: bool = _env_var_config(
    "TORCHINDUCTOR_LOCAL_DETERMINISM", default=(not is_fbcode())
)
# global determinism ensures that caching is deterministic across any/all machines,
# hence a remote cache (with strong consistency!) is used for synchronization of results
GLOBAL_DETERMINISM: bool = _env_var_config(
    "TORCHINDUCTOR_GLOBAL_DETERMINISM", default=is_fbcode()
)
