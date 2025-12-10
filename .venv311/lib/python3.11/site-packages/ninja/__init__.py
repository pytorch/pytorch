from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from typing import NoReturn

from ._version import version as __version__
from .ninja_syntax import Writer, escape, expand

__all__ = ["BIN_DIR", "DATA", "Writer", "__version__", "escape", "expand", "ninja"]


def __dir__() -> list[str]:
    return __all__


def _get_ninja_dir() -> str:
    ninja_exe = "ninja" + sysconfig.get_config_var("EXE")

    # Default path
    path = os.path.join(sysconfig.get_path("scripts"), ninja_exe)
    if os.path.isfile(path):
        return os.path.dirname(path)

    # User path
    if sys.version_info >= (3, 10):
        user_scheme = sysconfig.get_preferred_scheme("user")
    elif os.name == "nt":
        user_scheme = "nt_user"
    elif sys.platform.startswith("darwin") and getattr(sys, "_framework", None):
        user_scheme = "osx_framework_user"
    else:
        user_scheme = "posix_user"

    path = sysconfig.get_path("scripts", scheme=user_scheme)

    if os.path.isfile(os.path.join(path, ninja_exe)):
        return path

    # Fallback to python location
    path = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(path, ninja_exe)):
        return path

    return ""


BIN_DIR = _get_ninja_dir()


def _program(name: str, args: Iterable[str]) -> int:
    cmd = os.path.join(BIN_DIR, name)
    return subprocess.call([cmd, *args], close_fds=False)


def ninja() -> NoReturn:
    raise SystemExit(_program('ninja', sys.argv[1:]))
