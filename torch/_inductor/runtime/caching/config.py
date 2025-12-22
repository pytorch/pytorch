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


# Controls whether the Memoizer dumps its cache to a JSON file on destruction.
# This is useful for debugging and inspection purposes.
_DUMP_MEMOIZER_CACHE_ENV_VAR: str = "TORCHINDUCTOR_DUMP_MEMOIZER_CACHE"
_DUMP_MEMOIZER_CACHE_DEFAULT: bool = False
IS_DUMP_MEMOIZER_CACHE_ENABLED: Callable[[], bool] = partial(
    _env_var_config,
    _DUMP_MEMOIZER_CACHE_ENV_VAR,
    _DUMP_MEMOIZER_CACHE_DEFAULT,
)
