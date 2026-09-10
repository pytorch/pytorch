import os
from collections.abc import Callable
from functools import cache, partial
from typing import TypeVar

import torch
from torch._environment import is_fbcode


T = TypeVar("T")


def _is_force_disable_caches() -> bool:
    """Check if caching is force disabled via inductor config.

    This defers importing torch._inductor.config to avoid circular imports.

    Returns:
        True if force_disable_caches is set in inductor config, False otherwise.
    """
    from torch._inductor import config as inductor_config

    return inductor_config.force_disable_caches


@cache
def _env_var_val(env_var: str, default: T) -> str | T:
    """Get the value of an environment variable or return the default.

    Args:
        env_var: Environment variable name to check
        default: Default value to return if environment variable is not set

    Returns:
        The value from the environment variable as a string, or the default if not set
    """
    return os.environ.get(env_var, default)


@cache
def _env_var_config(env_var: str, default: bool) -> bool:
    env_val = _env_var_val(env_var, None)
    if env_val is not None:
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


def _is_caching_module_enabled_base() -> bool:
    """Base check for caching module enablement via versioned config."""
    return _versioned_config(
        _CACHING_MODULE_VERSION_JK,
        _CACHING_MODULE_VERSION,
        _CACHING_MODULE_OSS_DEFAULT,
        _CACHING_MODULE_ENV_VAR_OVERRIDE,
    )


def IS_CACHING_MODULE_ENABLED() -> bool:
    """Check if the caching module is enabled.

    Returns False if:
    - The versioned config disables it
    - force_disable_caches is set in inductor config

    Returns:
        True if caching module is enabled, False otherwise.
    """
    if not _is_caching_module_enabled_base():
        return False
    if _is_force_disable_caches():
        return False
    return True


# Controls whether the Memoizer dumps its cache to a JSON file on destruction.
# When enabled, the Memoizer will write its in-memory cache to a JSON file
# (memoizer_cache.json in the cache directory) on destruction. This dump file
# can later be used to pre-populate Memoizers via CACHE_DUMP_FILE_PATH.
#
# This is useful for:
# - Debugging and inspection of cached values
# - Creating cache snapshots that can be reused across runs
# - Pre-warming caches for subsequent executions
#
# Set via environment variable: TORCHINDUCTOR_DUMP_MEMOIZER_CACHE=1
_DUMP_MEMOIZER_CACHE_ENV_VAR: str = "TORCHINDUCTOR_DUMP_MEMOIZER_CACHE"
_DUMP_MEMOIZER_CACHE_DEFAULT: bool = False
IS_DUMP_MEMOIZER_CACHE_ENABLED: Callable[[], bool] = partial(
    _env_var_config,
    _DUMP_MEMOIZER_CACHE_ENV_VAR,
    _DUMP_MEMOIZER_CACHE_DEFAULT,
)


# Path to a cache dump file to pre-populate Memoizers on initialization.
# This should point to a JSON file produced by IS_DUMP_MEMOIZER_CACHE_ENABLED.
#
# When set, Memoizers will load cached entries from this file on initialization,
# allowing cache values to be reused across separate runs without recomputation.
# This is particularly useful for:
# - Pre-warming caches with known-good values
# - Reproducing behavior from a previous run
# - Sharing cached computations across different environments
#
# The dump file format is produced by the Memoizer's _dump_to_disk method when
# IS_DUMP_MEMOIZER_CACHE_ENABLED is set to true.
#
# Set via environment variable: TORCHINDUCTOR_CACHE_DUMP_FILE_PATH=/path/to/dump.json
_CACHE_DUMP_FILE_PATH_ENV_VAR: str = "TORCHINDUCTOR_CACHE_DUMP_FILE_PATH"
_CACHE_DUMP_FILE_PATH_DEFAULT: str | None = None
CACHE_DUMP_FILE_PATH: Callable[[], str | None] = partial(
    _env_var_val,
    _CACHE_DUMP_FILE_PATH_ENV_VAR,
    _CACHE_DUMP_FILE_PATH_DEFAULT,
)
