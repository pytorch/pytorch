"""
Version-adaptive frozen libtorch-agnostic extension for BC testing.

ABI surface is selected by TORCH_TARGET_VERSION (compile define), which can be
set from the environment before invoking setup.py / bdist_wheel, e.g.:

  export TORCH_TARGET_VERSION=0x0209000000000000   # or 2.9 / 2_9

When unset, defaults from the build torch: 2.9.x -> 2.9 target, else if
torch >= 2.13 -> 2.13 target.

  - target < 2.13:  2.9-era API style (manual boxed kernels)
  - target >= 2.13: TORCH_BOX registration + csrc/v213 ops
"""

import distutils.command.clean
import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
    IS_WINDOWS,
)


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"
PACKAGE = "libtorch_agn_frozen"


# Same encoding as the other libtorch_agn_*_extension setup.py files:
#   (major, minor) -> 0xMMNN000000000000
def _version_to_hex(major: int, minor: int) -> str:
    return f"0x{major:02x}{minor:02x}000000000000"


_TARGET_2_9 = _version_to_hex(2, 9)
_TARGET_2_13 = _version_to_hex(2, 13)
_TARGET_2_13_INT = int(_TARGET_2_13, 16)


def _torch_version_tuple():
    # "2.9.0+cpu" / "2.14.0a0+git..." -> (2, 9) / (2, 14)
    base = torch.__version__.split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
    parts = base.split(".")
    return int(parts[0]), int(parts[1])


def _parse_target_version(raw: str) -> str:
    """Return a 0x... hex string suitable for -DTORCH_TARGET_VERSION=."""
    s = raw.strip()
    if s.lower().startswith("0x"):
        int(s, 16)  # validate
        return s.lower()
    # Accept 2.9 / 2_9 / 2.14 / 2_14 / ...
    norm = s.replace("_", ".")
    parts = norm.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid TORCH_TARGET_VERSION={raw!r}; expected 0x... or M.N / M_N"
        )
    return _version_to_hex(int(parts[0]), int(parts[1]))


def _resolve_target_version_hex() -> str:
    env = os.environ.get("TORCH_TARGET_VERSION")
    if env:
        return _parse_target_version(env)
    major, minor = _torch_version_tuple()
    if (major, minor) >= (2, 13):
        return _TARGET_2_13
    return _TARGET_2_9


class clean(distutils.command.clean.clean):
    def run(self):
        distutils.command.clean.clean.run(self)
        for path in (ROOT_DIR / PACKAGE).glob("**/*.so"):
            path.unlink()
        for path in (
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / f"{PACKAGE}.egg-info",
        ):
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    target_hex = _resolve_target_version_hex()
    target_int = int(target_hex, 16)
    use_v213 = target_int >= _TARGET_2_13_INT
    print(
        f"{PACKAGE}: build torch={torch.__version__} "
        f"TORCH_TARGET_VERSION={target_hex} "
        f"(from {'env' if os.environ.get('TORCH_TARGET_VERSION') else 'default'})"
    )

    cxx_flags = [
        "-DTORCH_STABLE_ONLY",
        f"-DTORCH_TARGET_VERSION={target_hex}",
        f"-DSTABLE_LIB_NAME={PACKAGE}",
    ]
    if not IS_WINDOWS:
        cxx_flags.append("-fdiagnostics-color=always")

    extra_compile_args = {"cxx": cxx_flags}

    sources = [CSRC_DIR / "kernel.cpp"]
    if use_v213:
        sources.extend(sorted((CSRC_DIR / "v213").glob("**/*.cpp")))

    extension = CppExtension
    if torch.cuda._is_compiled() and CUDA_HOME is not None:
        extra_compile_args["cxx"].append("-DLAE_USE_CUDA")
        extra_compile_args["nvcc"] = [
            "-O2",
            "-DUSE_CUDA",
            f"-DTORCH_TARGET_VERSION={target_hex}",
            f"-DSTABLE_LIB_NAME={PACKAGE}",
        ]
        extension = CUDAExtension
        sources.extend(sorted(CSRC_DIR.glob("*.cu")))
        if use_v213:
            sources.extend(sorted((CSRC_DIR / "v213").glob("**/*.cu")))

    return [
        extension(
            f"{PACKAGE}._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
        )
    ]


setup(
    name=PACKAGE,
    version="0.0",
    author="PyTorch Core Team",
    description=(
        "Frozen libtorch-agnostic toy extension; ABI surface selected by "
        "TORCH_TARGET_VERSION"
    ),
    packages=find_packages(exclude=("test",)),
    package_data={PACKAGE: ["*.dll", "*.dylib", "*.so"]},
    # Intentionally no install_requires=["torch"]: pip installing this wheel
    # into an env must not pull a PyPI torch over an editable branch install.
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
