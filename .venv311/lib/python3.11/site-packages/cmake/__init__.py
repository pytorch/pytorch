from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

from ._version import version as __version__

TYPE_CHECKING = False

if TYPE_CHECKING:
    from typing import Iterable, NoReturn


__all__ = ["CMAKE_BIN_DIR", "CMAKE_DATA", "CMAKE_DOC_DIR", "CMAKE_SHARE_DIR", "__version__", "cmake", "cpack", "ctest"]


def __dir__() -> list[str]:
    return __all__


cmake_executable_path = None
cmake_files = distribution("cmake").files
assert cmake_files is not None, "This is the cmake package so it must be installed and have files"
for script in cmake_files:
    if str(script).startswith("cmake/data/bin/cmake"):
        resolved_script = Path(script.locate()).resolve(strict=True)
        cmake_executable_path = resolved_script.parents[1]
        break
CMAKE_DATA = str(cmake_executable_path) if cmake_executable_path else None

assert CMAKE_DATA is not None
assert os.path.exists(CMAKE_DATA)

CMAKE_BIN_DIR = os.path.join(CMAKE_DATA, 'bin')
CMAKE_DOC_DIR = os.path.join(CMAKE_DATA, 'doc')
CMAKE_SHARE_DIR = os.path.join(CMAKE_DATA, 'share')


def _program(name: str, args: Iterable[str]) -> int:
    return subprocess.call([os.path.join(CMAKE_BIN_DIR, name), *args], close_fds=False)

def _program_exit(name: str, *args: str) -> NoReturn:
    if sys.platform.startswith("win"):
        raise SystemExit(_program(name, args))
    cmake_exe = os.path.join(CMAKE_BIN_DIR, name)
    os.execl(cmake_exe, cmake_exe, *args)


def ccmake() -> NoReturn:
    _program_exit('ccmake', *sys.argv[1:])


def cmake() -> NoReturn:
    _program_exit('cmake', *sys.argv[1:])


def cpack() -> NoReturn:
    _program_exit('cpack', *sys.argv[1:])


def ctest() -> NoReturn:
    _program_exit('ctest', *sys.argv[1:])
