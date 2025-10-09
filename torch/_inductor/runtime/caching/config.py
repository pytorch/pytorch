import os
from functools import cache, partial
from typing import Callable, Optional

import torch
from torch._environment import is_fbcode


@cache
def _versioned_config(
    jk_name: str,
    this_version: int,
    oss_default: bool,
    env_var_override: Optional[str] = None,
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
        # returns 0 on exception, in this case the safe thing is to disable
        jk_version: int = torch._utils_internal.justknobs_getval_int(jk_name)
        return (this_version >= jk_version) and (jk_version != 0)
    return oss_default


_DETERMINISTIC_CACHING_VERSION: int = 0
_DETERMINISTIC_CACHING_VERSION_JK: str = (
    "pytorch/inductor:deterministic_caching_version"
)
_DETERMINISTIC_CACHING_OSS_DEFAULT: bool = False
_DETERMINISTIC_CACHING_ENV_VAR_OVERRIDE: str = "TORCHINDUCTOR_ENABLE_DETERMINISTIC_CACHING"
IS_DETERMINISTIC_CACHING_ENABLED: Callable[[], bool] = partial(
    _versioned_config,
    _DETERMINISTIC_CACHING_VERSION_JK,
    _DETERMINISTIC_CACHING_VERSION,
    _DETERMINISTIC_CACHING_OSS_DEFAULT,
    _DETERMINISTIC_CACHING_ENV_VAR_OVERRIDE,
)

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
