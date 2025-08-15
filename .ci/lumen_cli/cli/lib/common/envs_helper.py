"""
Environment Variables and Dataclasses Utility helpers for CLI tasks.
"""

import os
from dataclasses import field, fields, is_dataclass, MISSING
from pathlib import Path
from textwrap import indent
from typing import Optional, Union

from cli.lib.common.utils import str2bool


def get_env(name: str, default: str = "") -> str:
    """Get environment variable with default fallback."""
    return os.environ.get(name) or default


def env_path_optional(
    name: str,
    default: Optional[Union[str, Path]] = None,
    resolve: bool = True,
) -> Optional[Path]:
    """Get environment variable as optional Path."""
    val = get_env(name) or default
    if not val:
        return None

    path = Path(val)
    return path.resolve() if resolve else path


def env_path(
    name: str,
    default: Optional[Union[str, Path]] = None,
    resolve: bool = True,
) -> Path:
    """Get environment variable as Path, raise if missing."""
    path = env_path_optional(name, default, resolve)
    if not path:
        raise ValueError(f"Missing path value for {name}")
    return path


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    val = get_env(name)
    if not val:
        return default
    return str2bool(val)


def env_bool_field(
    name: str,
    default: bool = False,
):
    return field(default_factory=lambda: env_bool(name, default))


def env_path_field(
    name: str,
    default: Union[str, Path] = "",
    *,
    resolve: bool = True,
) -> Path:
    return field(default_factory=lambda: env_path(name, default, resolve=resolve))


def env_str_field(
    name: str,
    default: str = "",
) -> str:
    return field(default_factory=lambda: get_env(name, default))


def generate_dataclass_help(cls) -> str:
    """Auto-generate help text for dataclass fields."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    def get_value(f):
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:
            try:
                return f.default_factory()
            except Exception as e:
                return f"<error: {e}>"
        return "<required>"

    lines = [f"{f.name:<22} = {repr(get_value(f))}" for f in fields(cls)]
    return indent("\n".join(lines), "    ")


def with_params_help(params_cls: type, title: str = "Parameter defaults"):
    """
    Class decorator that appends a help table generated from another dataclass
    (e.g., VllmParameters) to the decorated class's docstring.
    """
    if not is_dataclass(params_cls):
        raise TypeError(f"{params_cls} must be a dataclass")

    def _decorator(cls: type) -> type:
        block = generate_dataclass_help(params_cls)
        cls.__doc__ = (cls.__doc__ or "") + f"\n\n{title}:\n{block}"
        return cls

    return _decorator
