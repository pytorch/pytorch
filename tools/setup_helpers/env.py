import os
import platform
import struct
import sys
from itertools import chain
from typing import cast, Iterable, List, Optional


IS_WINDOWS = platform.system() == "Windows"
IS_DARWIN = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

IS_CONDA = (
    "conda" in sys.version
    or "Continuum" in sys.version
    or any(x.startswith("CONDA") for x in os.environ)
)
CONDA_DIR = os.path.join(os.path.dirname(sys.executable), "..")

IS_64BIT = struct.calcsize("P") == 8

BUILD_DIR = "build"

LIBTORCH_PKG_NAME = "libtorchsplit"


def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def check_negative_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]


def gather_paths(env_vars: Iterable[str]) -> List[str]:
    return list(chain(*(os.getenv(v, "").split(os.pathsep) for v in env_vars)))


def lib_paths_from_base(base_path: str) -> List[str]:
    return [os.path.join(base_path, s) for s in ["lib/x64", "lib", "lib64"]]


# We promised that CXXFLAGS should also be affected by CFLAGS
if "CFLAGS" in os.environ and "CXXFLAGS" not in os.environ:
    os.environ["CXXFLAGS"] = os.environ["CFLAGS"]


class BuildType:
    """Checks build type. The build type will be given in :attr:`cmake_build_type_env`. If :attr:`cmake_build_type_env`
    is ``None``, then the build type will be inferred from ``CMakeCache.txt``. If ``CMakeCache.txt`` does not exist,
    os.environ['CMAKE_BUILD_TYPE'] will be used.

    Args:
      cmake_build_type_env (str): The value of os.environ['CMAKE_BUILD_TYPE']. If None, the actual build type will be
        inferred.

    """

    def __init__(self, cmake_build_type_env: Optional[str] = None) -> None:
        if cmake_build_type_env is not None:
            self.build_type_string = cmake_build_type_env
            return

        cmake_cache_txt = os.path.join(BUILD_DIR, "CMakeCache.txt")
        if os.path.isfile(cmake_cache_txt):
            # Found CMakeCache.txt. Use the build type specified in it.
            from .cmake_utils import get_cmake_cache_variables_from_file

            with open(cmake_cache_txt) as f:
                cmake_cache_vars = get_cmake_cache_variables_from_file(f)
            # Normally it is anti-pattern to determine build type from CMAKE_BUILD_TYPE because it is not used for
            # multi-configuration build tools, such as Visual Studio and XCode. But since we always communicate with
            # CMake using CMAKE_BUILD_TYPE from our Python scripts, this is OK here.
            self.build_type_string = cast(str, cmake_cache_vars["CMAKE_BUILD_TYPE"])
        else:
            self.build_type_string = os.environ.get("CMAKE_BUILD_TYPE", "Release")

    def is_debug(self) -> bool:
        "Checks Debug build."
        return self.build_type_string == "Debug"

    def is_rel_with_deb_info(self) -> bool:
        "Checks RelWithDebInfo build."
        return self.build_type_string == "RelWithDebInfo"

    def is_release(self) -> bool:
        "Checks Release build."
        return self.build_type_string == "Release"


# hotpatch environment variable 'CMAKE_BUILD_TYPE'. 'CMAKE_BUILD_TYPE' always prevails over DEBUG or REL_WITH_DEB_INFO.
if "CMAKE_BUILD_TYPE" not in os.environ:
    if check_env_flag("DEBUG"):
        os.environ["CMAKE_BUILD_TYPE"] = "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        os.environ["CMAKE_BUILD_TYPE"] = "RelWithDebInfo"
    else:
        os.environ["CMAKE_BUILD_TYPE"] = "Release"

build_type = BuildType()
