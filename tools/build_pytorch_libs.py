from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from .setup_helpers.cmake import CMake, USE_NINJA
from .setup_helpers.env import (
    check_env_flag,
    check_negative_env_flag,
    IS_64BIT,
    IS_WINDOWS,
)


repo_root = Path(__file__).absolute().parent.parent
third_party_path = os.path.join(repo_root, "third_party")


def _get_vc_env(vc_arch: str) -> dict[str, str]:
    try:
        from setuptools import distutils  # type: ignore[import]

        return distutils._msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return]
    except AttributeError:
        from setuptools._distutils import _msvccompiler  # type: ignore[import]

        return _msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return]


def _overlay_windows_vcvars(env: dict[str, str]) -> dict[str, str]:
    vc_arch = "x64" if IS_64BIT else "x86"

    if platform.machine() == "ARM64":
        vc_arch = "x64_arm64"

        # First Win11 Windows on Arm build version that supports x64 emulation
        # is 10.0.22000.
        win11_1st_version = (10, 0, 22000)
        current_win_version = tuple(
            int(version_part) for version_part in platform.version().split(".")
        )
        if current_win_version < win11_1st_version:
            vc_arch = "x86_arm64"
            print(
                "Warning: 32-bit toolchain will be used, but 64-bit linker "
                "is recommended to avoid out-of-memory linker error!"
            )
            print(
                "Warning: Please consider upgrading to Win11, where x64 "
                "emulation is enabled!"
            )

    vc_env = _get_vc_env(vc_arch)
    # Keys in `_get_vc_env` are always lowercase.
    # We turn them into uppercase before overlaying vcvars
    # because OS environ keys are always uppercase on Windows.
    # https://stackoverflow.com/a/7797329
    vc_env = {k.upper(): v for k, v in vc_env.items()}
    for k, v in env.items():
        uk = k.upper()
        if uk not in vc_env:
            vc_env[uk] = v
    return vc_env


def _create_build_env() -> dict[str, str]:
    # XXX - our cmake file sometimes looks at the system environment
    # and not cmake flags!
    # you should NEVER add something to this list. It is bad practice to
    # have cmake read the environment
    my_env = os.environ.copy()
    if IS_WINDOWS and USE_NINJA:
        # When using Ninja under Windows, the gcc toolchain will be chosen as
        # default. But it should be set to MSVC as the user's first choice.
        my_env = _overlay_windows_vcvars(my_env)
        my_env.setdefault("CC", "cl")
        my_env.setdefault("CXX", "cl")
    return my_env


def read_nccl_pin() -> str:
    nccl_file = "nccl-cu12.txt"
    if os.getenv("DESIRED_CUDA", "").startswith("11") or os.getenv(
        "CUDA_VERSION", ""
    ).startswith("11"):
        nccl_file = "nccl-cu11.txt"
    nccl_pin_path = os.path.join(
        repo_root, ".ci", "docker", "ci_commit_pins", nccl_file
    )
    with open(nccl_pin_path) as f:
        return f.read().strip()


def checkout_nccl() -> None:
    release_tag = read_nccl_pin()
    print(f"-- Checkout nccl release tag: {release_tag}")
    nccl_basedir = os.path.join(third_party_path, "nccl")
    if not os.path.exists(nccl_basedir):
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                release_tag,
                "https://github.com/NVIDIA/nccl.git",
                "nccl",
            ],
            cwd=third_party_path,
        )


def build_pytorch(
    version: str | None,
    cmake_python_library: str | None,
    build_python: bool,
    rerun_cmake: bool,
    cmake_only: bool,
    cmake: CMake,
) -> None:
    my_env = _create_build_env()
    if (
        not check_negative_env_flag("USE_CUDA")
        and not check_negative_env_flag("USE_NCCL")
        and not check_env_flag("USE_SYSTEM_NCCL")
    ):
        checkout_nccl()
    build_test = not check_negative_env_flag("BUILD_TEST")
    cmake.generate(
        version, cmake_python_library, build_python, build_test, my_env, rerun_cmake
    )
    if cmake_only:
        return
    cmake.build(my_env)
