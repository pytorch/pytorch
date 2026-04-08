"""Manages CMake."""

from __future__ import annotations

import functools
import json
import multiprocessing
import os
import platform
import shutil
import sys
import sysconfig
from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output, DEVNULL

from .cmake_utils import CMakeValue, get_cmake_cache_variables_from_file
from .env import (
    BUILD_DIR,
    check_negative_env_flag,
    CMAKE_MINIMUM_VERSION_STRING,
    IS_64BIT,
    IS_DARWIN,
    IS_WINDOWS,
)


try:
    from packaging.version import Version
except ImportError:
    try:
        from setuptools.dist import Version  # type: ignore[attr-defined,no-redef]
    except ImportError:
        from distutils.version import (  # type: ignore[assignment,no-redef]
            LooseVersion as Version,
        )


def _mkdir_p(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create folder {os.path.abspath(d)}: {e.strerror}"
        ) from e


# Print to stderr
eprint = functools.partial(print, file=sys.stderr, flush=True)


# Ninja
# Use ninja if it is on the PATH. Previous version of PyTorch required the
# ninja python package, but we no longer use it, so we do not have to import it
USE_NINJA = bool(not check_negative_env_flag("USE_NINJA") and shutil.which("ninja"))
if "CMAKE_GENERATOR" in os.environ:
    USE_NINJA = os.environ["CMAKE_GENERATOR"].lower() == "ninja"


CMAKE_MINIMUM_VERSION = Version(CMAKE_MINIMUM_VERSION_STRING)


class CMake:
    "Manages cmake."

    def __init__(self, build_dir: str = BUILD_DIR) -> None:
        self._cmake_command = CMake._get_cmake_command()
        self.build_dir = build_dir

    @property
    def _cmake_cache_file(self) -> str:
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, "CMakeCache.txt")

    @property
    def _ninja_build_file(self) -> str:
        r"""Returns the path to build.ninja.

        Returns:
          string: The path to build.ninja.
        """
        return os.path.join(self.build_dir, "build.ninja")

    @staticmethod
    def _get_cmake_command() -> str:
        """Returns cmake command."""

        if IS_WINDOWS:
            return "cmake"

        cmake_versions: list[str] = []
        valid_cmake_versions: dict[str, Version] = {}
        for cmd in ("cmake", "cmake3"):
            command = shutil.which(cmd)
            ver = CMake._get_version(command)
            if ver is not None:
                eprint(f"Found {cmd} ({command}) version: {ver}", end="")
                cmake_versions.append(f"{cmd}=={ver}")
                # pyrefly: ignore [unsupported-operation]
                if ver >= CMAKE_MINIMUM_VERSION:
                    eprint(f" (>={CMAKE_MINIMUM_VERSION})")
                    valid_cmake_versions[cmd] = ver
                else:
                    eprint(f" (<{CMAKE_MINIMUM_VERSION})")

        if not valid_cmake_versions:
            raise RuntimeError(
                f"no cmake or cmake3 with version >= {CMAKE_MINIMUM_VERSION}, "
                f"found: {cmake_versions}"
            )
        return max(valid_cmake_versions, key=valid_cmake_versions.get)  # type: ignore[arg-type]

    @staticmethod
    def _get_version(cmd: str | None) -> Version | None:
        """Returns cmake version."""

        if cmd is None:
            return None

        try:
            cmake_capabilities = json.loads(
                check_output(
                    [cmd, "-E", "capabilities"],
                    stderr=DEVNULL,
                    text=True,
                ),
            )
        except (OSError, CalledProcessError, json.JSONDecodeError):
            cmake_capabilities = {}
        cmake_version = cmake_capabilities.get("version", {}).get("string")
        if cmake_version is not None:
            return Version(cmake_version)
        raise RuntimeError(f"Failed to get CMake version from command: {cmd}")

    def run(self, args: list[str], env: dict[str, str]) -> None:
        """Executes cmake with arguments and an environment."""

        command = [self._cmake_command] + args
        eprint(" ".join(command))
        try:
            check_call(command, cwd=self.build_dir, env=env)
        except (CalledProcessError, KeyboardInterrupt):
            # This error indicates that there was a problem with cmake, the
            # Python backtrace adds no signal here so skip over it by catching
            # the error and exiting manually
            sys.exit(1)

    @staticmethod
    def defines(args: list[str], **kwargs: CMakeValue) -> None:
        """Adds definitions to a cmake argument list."""
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append(f"-D{key}={value}")

    def get_cmake_cache_variables(self) -> dict[str, CMakeValue]:
        r"""Gets values in CMakeCache.txt into a dictionary.
        Returns:
          dict: A ``dict`` containing the value of cached CMake variables.
        """
        with open(self._cmake_cache_file) as f:
            return get_cmake_cache_variables_from_file(f)

    def generate(
        self,
        version: str | None,
        cmake_python_library: str | None,
        build_python: bool,
        build_test: bool,
        my_env: dict[str, str],
        rerun: bool,
    ) -> None:
        """Runs cmake to generate native build files."""

        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)

        cmake_cache_file_available = os.path.exists(self._cmake_cache_file)
        if cmake_cache_file_available:
            cmake_cache_variables = self.get_cmake_cache_variables()
            make_program: str | None = cmake_cache_variables.get("CMAKE_MAKE_PROGRAM")  # type: ignore[assignment]
            if make_program and not shutil.which(make_program):
                # CMakeCache.txt exists, but the make program (e.g., ninja) does not.
                # See also: https://github.com/astral-sh/uv/issues/14269
                # This can happen if building with PEP-517 build isolation, where `ninja` was
                # installed in the isolated environment of the previous build run, but it has been
                # removed. The `ninja` executable with an old absolute path not available anymore.
                eprint(
                    "!!!WARNING!!!: CMakeCache.txt exists, "
                    f"but CMAKE_MAKE_PROGRAM ({make_program!r}) does not exist. "
                    "Clearing CMake cache."
                )
                self.clear_cache()
                cmake_cache_file_available = False

        if cmake_cache_file_available and (
            not USE_NINJA or os.path.exists(self._ninja_build_file)
        ):
            # Everything's in place. Do not rerun.
            return

        args = []
        if USE_NINJA:
            # Avoid conflicts in '-G' and the `CMAKE_GENERATOR`
            os.environ["CMAKE_GENERATOR"] = "Ninja"
            args.append("-GNinja")
        elif IS_WINDOWS:
            generator = os.getenv("CMAKE_GENERATOR", "Visual Studio 16 2019")
            supported = ["Visual Studio 16 2019", "Visual Studio 17 2022"]
            if generator not in supported:
                eprint("Unsupported `CMAKE_GENERATOR`: " + generator)
                eprint("Please set it to one of the following values: ")
                eprint("\n".join(supported))
                sys.exit(1)
            args.append("-G" + generator)
            toolset_dict = {}
            toolset_version = os.getenv("CMAKE_GENERATOR_TOOLSET_VERSION")
            if toolset_version is not None:
                toolset_dict["version"] = toolset_version
                curr_toolset = os.getenv("VCToolsVersion")
                if curr_toolset is None:
                    eprint(
                        "When you specify `CMAKE_GENERATOR_TOOLSET_VERSION`, you must also "
                        "activate the vs environment of this version. Please read the notes "
                        "in the build steps carefully."
                    )
                    sys.exit(1)
            if IS_64BIT:
                if platform.machine() == "ARM64":
                    args.append("-A ARM64")
                else:
                    args.append("-Ax64")
                    toolset_dict["host"] = "x64"
            if toolset_dict:
                toolset_expr = ",".join([f"{k}={v}" for k, v in toolset_dict.items()])
                args.append("-T" + toolset_expr)

        base_dir = str(Path(__file__).absolute().parents[2])
        install_dir = os.path.join(base_dir, "torch")

        _mkdir_p(install_dir)
        _mkdir_p(self.build_dir)

        # Environment variable forwarding (BUILD_*, USE_*, CMAKE_*, aliases,
        # passthrough vars, CMAKE_PREFIX_PATH, low-priority aliases) is now
        # handled by cmake/EnvVarForwarding.cmake, which is included early in
        # the top-level CMakeLists.txt. Only options that require Python-side
        # detection are passed here.

        build_options: dict[str, CMakeValue] = {
            "CMAKE_INSTALL_PREFIX": install_dir,
            "BUILD_PYTHON": build_python,
            "BUILD_TEST": build_test,
        }

        use_numpy = not check_negative_env_flag("USE_NUMPY")
        build_options["USE_NUMPY"] = use_numpy
        if use_numpy:
            try:
                import numpy

                build_options["Python_NumPy_INCLUDE_DIR"] = numpy.get_include()
            except ImportError:
                pass

        # NVSHMEM detection from Python lib path
        py_lib_path = sysconfig.get_path("purelib")
        nvshmem_py_dir = py_lib_path + "/nvidia/nvshmem"
        if os.path.exists(nvshmem_py_dir):
            build_options["NVSHMEM_PY_DIR"] = nvshmem_py_dir

        CMake.defines(
            args,
            Python_EXECUTABLE=sys.executable,
            TORCH_BUILD_VERSION=version,
            **build_options,
        )

        expected_wrapper = "/usr/local/opt/ccache/libexec"
        if IS_DARWIN and os.path.exists(expected_wrapper):
            if "CMAKE_C_COMPILER" not in build_options and "CC" not in os.environ:
                CMake.defines(args, CMAKE_C_COMPILER=f"{expected_wrapper}/gcc")
            if "CMAKE_CXX_COMPILER" not in build_options and "CXX" not in os.environ:
                CMake.defines(args, CMAKE_CXX_COMPILER=f"{expected_wrapper}/g++")

        for env_var_name in my_env:
            if env_var_name.startswith("gh"):
                # github env vars use utf-8, on windows, non-ascii code may
                # cause problem, so encode first
                try:
                    my_env[env_var_name] = str(my_env[env_var_name].encode("utf-8"))
                except UnicodeDecodeError as e:
                    shex = ":".join(f"{ord(c):02x}" for c in my_env[env_var_name])
                    eprint(f"Invalid ENV[{env_var_name}] = {shex}")
                    eprint(e)
        # According to the CMake manual, we should pass the arguments first,
        # and put the directory as the last element. Otherwise, these flags
        # may not be passed correctly.
        # Reference:
        # 1. https://cmake.org/cmake/help/latest/manual/cmake.1.html#synopsis
        # 2. https://stackoverflow.com/a/27169347
        args.append(base_dir)
        self.run(args, env=my_env)

    def build(self, my_env: dict[str, str]) -> None:
        """Runs cmake to build binaries."""

        from .env import build_type

        build_args = [
            "--build",
            ".",
            "--target",
            "install",
            "--config",
            build_type.build_type_string,
        ]

        # Determine the parallelism according to the following
        # priorities:
        # 1) MAX_JOBS environment variable
        # 2) If using the Ninja build system, delegate decision to it.
        # 3) Otherwise, fall back to the number of processors.

        # Allow the user to set parallelism explicitly. If unset,
        # we'll try to figure it out.
        max_jobs = os.getenv("MAX_JOBS")

        if max_jobs is not None or not USE_NINJA:
            # Ninja is capable of figuring out the parallelism on its
            # own: only specify it explicitly if we are not using
            # Ninja.

            # This lists the number of processors available on the
            # machine. This may be an overestimate of the usable
            # processors if CPU scheduling affinity limits it
            # further. In the future, we should check for that with
            # os.sched_getaffinity(0) on platforms that support it.
            max_jobs = max_jobs or str(multiprocessing.cpu_count())

            # CMake 3.12 provides a '-j' option.
            build_args += ["-j", max_jobs]
        self.run(build_args, my_env)

    def clear_cache(self) -> None:
        """Clears the CMake cache."""
        if os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        if os.path.isfile(self._ninja_build_file):
            os.remove(self._ninja_build_file)
