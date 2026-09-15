import distutils.command.clean
import shutil
from pathlib import Path

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    IS_WINDOWS,
)


ROOT_DIR = Path(__file__).parent
CSRC_DIR = ROOT_DIR / "csrc"

# Include csrc from previous versions for forward compatibility testing
PREV_CSRC_DIRS = [
    ROOT_DIR.parent / "libtorch_agn_2_9_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_10_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_11_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_12_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_13_extension" / "csrc",
    ROOT_DIR.parent / "libtorch_agn_2_14_extension" / "csrc",
]


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agn_2_15").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agn_2_15.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    common_cxx = ["-DTORCH_TARGET_VERSION=0x020f000000000000"]
    if not IS_WINDOWS:
        common_cxx.append("-fdiagnostics-color=always")

    all_csrc_dirs = [CSRC_DIR, *PREV_CSRC_DIRS]
    mps_available = torch.backends.mps.is_available()

    # Op extension (_C): this version's ops + inherited 2.9-2.14 csrc, minus the
    # interop module.
    op_sources = [
        s
        for d in all_csrc_dirs
        for s in d.glob("**/*.cpp")
        if s.name != "pyobject_interop_module.cpp"
        and (s.parent.name != "mps" or mps_available)
    ]

    op_cxx = common_cxx + ["-DSTABLE_LIB_NAME=libtorch_agn_2_15"]
    op_extra = {"cxx": op_cxx}
    op_extension = CppExtension
    # allow including <cuda_runtime.h>
    if torch.cuda.is_available():
        op_cxx.append("-DLAE_USE_CUDA")
        op_extra["nvcc"] = [
            "-O2",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020f000000000000",
            "-DSTABLE_LIB_NAME=libtorch_agn_2_15",
        ]
        op_extension = CUDAExtension
        op_sources.extend(CSRC_DIR.glob("**/*.cu"))
        for prev_dir in PREV_CSRC_DIRS:
            op_sources.extend(prev_dir.glob("**/*.cu"))

    if mps_available:
        op_cxx.append("-DUSE_MPS")

    return [
        op_extension(
            "libtorch_agn_2_15._C",
            sources=sorted(str(s) for s in op_sources),
            py_limited_api=True,
            extra_compile_args=op_extra,
            extra_link_args=[],
        ),
    ]


setup(
    name="libtorch_agn_2_15",
    version="0.0",
    author="PyTorch Core Team",
    description="Example of libtorch agnostic extension for PyTorch 2.15+",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agn_2_15": ["*.dll", "*.dylib", "*.so"]},
    install_requires=[
        "torch",
    ],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
