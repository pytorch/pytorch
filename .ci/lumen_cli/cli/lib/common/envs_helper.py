"""
Environment Variables and Dataclasses Utility helpers for CLI tasks.
"""

import os
from dataclasses import field
from pathlib import Path
from typing import Optional, Union

from cli.lib.common.utils import str2bool


def get_env(name: str, default: str = "") -> str:
    """
    Get an environment variable with a default fallback.
    If the variable is not set or empty str, returns the default.
    """
    val = os.environ.get(name)
    if not val:  # catches None and ""
        val = default
    return val


def env_path_optional(
    name: str,
    default: Optional[Union[str, Path]] = None,
    *,
    resolve: bool = True,
) -> Optional[Path]:
    """
    Get an environment variable as a Path with a default fallback.
    If the variable is not set or empty str, returns the default.
    the default is None.
    If resolve=True, returns the resolved path.
    """
    val = get_env(name)
    if not val:
        val = default

    if not val:
        return None

    if isinstance(val, Path):
        p = val
    else:
        p = Path(val)

    return p.resolve() if resolve else p


def env_path(
    name: str,
    default: Optional[Union[str, Path]] = None,
    *,
    resolve: bool = True,
) -> Path:
    p = env_path_optional(name, default, resolve=resolve)
    if not p:
        raise ValueError(f"Missing path value for {name}: {p}")
    return p


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    val = get_env(name)
    if not val:
        return default
    return str2bool(val)


# ------------------ dataclass fields helper ------------------ #


def env_bool_field(
    name: str,
    default: bool = False,
):
    """
    returns dataclass's factory function for bool field with default

    """
    return field(default_factory=lambda: env_bool(name, default))


def env_path_field(
    name: str,
    default: Union[str, Path] = "",
    *,
    resolve: bool = True,
) -> Path:
    """
    returns dataclass's factory function for Path field with Default value
    If resolve=True, returns the resolved(absolute) path.
    """
    return field(default_factory=lambda: env_path(name, default, resolve=resolve))


def env_str_field(
    name: str,
    default: str = "",
) -> str:
    """
    returns dataclass's factory function for str field with default value
    """
    return field(default_factory=lambda: get_env(name, default))
