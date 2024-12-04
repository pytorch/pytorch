#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
    * Latest XPU
"""

import os
from typing import Dict, List, Optional, Tuple


# NOTE: Also update the CUDA sources in tools/nightly.py when changing this list
CUDA_ARCHES = ["11.8", "12.4", "12.6"]
CUDA_ARCHES_FULL_VERSION = {"11.8": "11.8.0", "12.4": "12.4.1", "12.6": "12.6.2"}
CUDA_ARCHES_CUDNN_VERSION = {"11.8": "9", "12.4": "9", "12.6": "9"}

ROCM_ARCHES = ["6.1", "6.2"]

XPU_ARCHES = ["xpu"]

CPU_CXX11_ABI_ARCH = ["cpu-cxx11-abi"]

CPU_AARCH64_ARCH = ["cpu-aarch64"]

CPU_S390X_ARCH = ["cpu-s390x"]

CUDA_AARCH64_ARCH = ["cuda-aarch64"]


PYTORCH_EXTRA_INSTALL_REQUIREMENTS = {
    "11.8": (
        "nvidia-cuda-nvrtc-cu11==11.8.89; platform_system == 'Linux' and platform_machine == 'x86_64' | "  # noqa: B950
        "nvidia-cuda-runtime-cu11==11.8.89; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-cupti-cu11==11.8.87; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cudnn-cu11==9.1.0.70; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cublas-cu11==11.11.3.6; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufft-cu11==10.9.0.58; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-curand-cu11==10.3.0.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusolver-cu11==11.4.1.48; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparse-cu11==11.7.5.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nccl-cu11==2.21.5; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvtx-cu11==11.8.86; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
    "12.4": (
        "nvidia-cuda-nvrtc-cu12==12.4.127; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-runtime-cu12==12.4.127; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-cupti-cu12==12.4.127; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cudnn-cu12==9.1.0.70; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cublas-cu12==12.4.5.8; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufft-cu12==11.2.1.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-curand-cu12==10.3.5.147; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusolver-cu12==11.6.1.9; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparse-cu12==12.3.1.170; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparselt-cu12==0.6.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nccl-cu12==2.21.5; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvtx-cu12==12.4.127; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvjitlink-cu12==12.4.127; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
    "12.6": (
        "nvidia-cuda-nvrtc-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-runtime-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-cupti-cu12==12.6.80; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cudnn-cu12==9.5.1.17; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cublas-cu12==12.6.4.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufft-cu12==11.3.0.4; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-curand-cu12==10.3.7.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusolver-cu12==11.7.1.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparse-cu12==12.5.4.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparselt-cu12==0.6.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nccl-cu12==2.21.5; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvtx-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvjitlink-cu12==12.6.85; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
    "xpu": (
        "intel-cmplr-lib-rt==2025.0.2 | "
        "intel-cmplr-lib-ur==2025.0.2 | "
        "intel-cmplr-lic-rt==2025.0.2 | "
        "intel-sycl-rt==2025.0.2 | "
        "tcmlib==1.2.0 | "
        "umf==0.9.1 | "
        "intel-pti==0.10.0; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
}


def get_nccl_submodule_version() -> str:
    from pathlib import Path

    nccl_version_mk = (
        Path(__file__).absolute().parent.parent.parent
        / "third_party"
        / "nccl"
        / "nccl"
        / "makefiles"
        / "version.mk"
    )
    if not nccl_version_mk.exists():
        raise RuntimeError(
            "Please make sure that nccl submodule is checked out when importing this script"
        )
    with nccl_version_mk.open("r") as f:
        content = f.read()
    d = {}
    for l in content.split("\n"):
        if not l.startswith("NCCL_"):
            continue
        (k, v) = l.split(":=")
        d[k.strip()] = v.strip()
    return f"{d['NCCL_MAJOR']}.{d['NCCL_MINOR']}.{d['NCCL_PATCH']}"


def get_nccl_wheel_version(arch_version: str) -> str:
    import re

    requirements = map(
        str.strip, re.split("[;|]", PYTORCH_EXTRA_INSTALL_REQUIREMENTS[arch_version])
    )
    return next(x for x in requirements if x.startswith("nvidia-nccl-cu")).split("==")[
        1
    ]


def validate_nccl_dep_consistency(arch_version: str) -> None:
    wheel_ver = get_nccl_wheel_version(arch_version)
    submodule_ver = get_nccl_submodule_version()
    if wheel_ver != submodule_ver:
        raise RuntimeError(
            f"NCCL submodule version {submodule_ver} differs from wheel version {wheel_ver}"
        )


def arch_type(arch_version: str) -> str:
    if arch_version in CUDA_ARCHES:
        return "cuda"
    elif arch_version in ROCM_ARCHES:
        return "rocm"
    elif arch_version in XPU_ARCHES:
        return "xpu"
    elif arch_version in CPU_CXX11_ABI_ARCH:
        return "cpu-cxx11-abi"
    elif arch_version in CPU_AARCH64_ARCH:
        return "cpu-aarch64"
    elif arch_version in CPU_S390X_ARCH:
        return "cpu-s390x"
    elif arch_version in CUDA_AARCH64_ARCH:
        return "cuda-aarch64"
    else:  # arch_version should always be "cpu" in this case
        return "cpu"


# This can be updated to the release version when cutting release branch, i.e. 2.1
DEFAULT_TAG = os.getenv("RELEASE_VERSION_TAG", "main")

WHEEL_CONTAINER_IMAGES = {
    "11.8": f"pytorch/manylinux-builder:cuda11.8-{DEFAULT_TAG}",
    "12.4": f"pytorch/manylinux-builder:cuda12.4-{DEFAULT_TAG}",
    "12.6": f"pytorch/manylinux2_28-builder:cuda12.6-{DEFAULT_TAG}",
    **{
        gpu_arch: f"pytorch/manylinux-builder:rocm{gpu_arch}-{DEFAULT_TAG}"
        for gpu_arch in ROCM_ARCHES
    },
    "xpu": f"pytorch/manylinux2_28-builder:xpu-{DEFAULT_TAG}",
    "cpu": f"pytorch/manylinux-builder:cpu-{DEFAULT_TAG}",
    "cpu-cxx11-abi": f"pytorch/manylinuxcxx11-abi-builder:cpu-cxx11-abi-{DEFAULT_TAG}",
    "cpu-aarch64": f"pytorch/manylinux2_28_aarch64-builder:cpu-aarch64-{DEFAULT_TAG}",
    "cpu-s390x": f"pytorch/manylinuxs390x-builder:cpu-s390x-{DEFAULT_TAG}",
    "cuda-aarch64": f"pytorch/manylinuxaarch64-builder:cuda12.4-{DEFAULT_TAG}",
}


PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"
RELEASE = "release"
DEBUG = "debug"

LIBTORCH_CONTAINER_IMAGES: Dict[Tuple[str, str], str] = {
    **{
        (
            gpu_arch,
            PRE_CXX11_ABI,
        ): f"pytorch/manylinux-builder:cuda{gpu_arch}-{DEFAULT_TAG}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        (
            gpu_arch,
            CXX11_ABI,
        ): f"pytorch/libtorch-cxx11-builder:cuda{gpu_arch}-{DEFAULT_TAG}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        (
            gpu_arch,
            PRE_CXX11_ABI,
        ): f"pytorch/manylinux-builder:rocm{gpu_arch}-{DEFAULT_TAG}"
        for gpu_arch in ROCM_ARCHES
    },
    **{
        (
            gpu_arch,
            CXX11_ABI,
        ): f"pytorch/libtorch-cxx11-builder:rocm{gpu_arch}-{DEFAULT_TAG}"
        for gpu_arch in ROCM_ARCHES
    },
    ("cpu", PRE_CXX11_ABI): f"pytorch/manylinux-builder:cpu-{DEFAULT_TAG}",
    ("cpu", CXX11_ABI): f"pytorch/libtorch-cxx11-builder:cpu-{DEFAULT_TAG}",
}

FULL_PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        "cpu": "cpu",
        "cpu-aarch64": "cpu",
        "cpu-cxx11-abi": "cpu-cxx11-abi",
        "cpu-s390x": "cpu",
        "cuda": f"cu{gpu_arch_version.replace('.', '')}",
        "cuda-aarch64": "cu124",
        "rocm": f"rocm{gpu_arch_version}",
        "xpu": "xpu",
    }.get(gpu_arch_type, gpu_arch_version)


def list_without(in_list: List[str], without: List[str]) -> List[str]:
    return [item for item in in_list if item not in without]


def generate_libtorch_matrix(
    os: str,
    abi_version: str,
    arches: Optional[List[str]] = None,
    libtorch_variants: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    if arches is None:
        arches = ["cpu"]
        if os == "linux":
            arches += CUDA_ARCHES
            arches += ROCM_ARCHES
        elif os == "windows":
            arches += CUDA_ARCHES
    if libtorch_variants is None:
        libtorch_variants = [
            "shared-with-deps",
            "shared-without-deps",
            "static-with-deps",
            "static-without-deps",
        ]

    ret: List[Dict[str, str]] = []
    for arch_version in arches:
        for libtorch_variant in libtorch_variants:
            # one of the values in the following list must be exactly
            # CXX11_ABI, but the precise value of the other one doesn't
            # matter
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            # ROCm builds without-deps failed even in ROCm runners; skip for now
            if gpu_arch_type == "rocm" and "without-deps" in libtorch_variant:
                continue
            ret.append(
                {
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "libtorch_variant": libtorch_variant,
                    "libtorch_config": abi_version if os == "windows" else "",
                    "devtoolset": abi_version if os != "windows" else "",
                    "container_image": (
                        LIBTORCH_CONTAINER_IMAGES[(arch_version, abi_version)]
                        if os != "windows"
                        else ""
                    ),
                    "package_type": "libtorch",
                    "build_name": f"libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{abi_version}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_wheels_matrix(
    os: str,
    arches: Optional[List[str]] = None,
    python_versions: Optional[List[str]] = None,
    use_split_build: bool = False,
) -> List[Dict[str, str]]:
    package_type = "wheel"
    if os == "linux" or os == "linux-aarch64" or os == "linux-s390x":
        # NOTE: We only build manywheel packages for x86_64 and aarch64 and s390x linux
        package_type = "manywheel"

    if python_versions is None:
        python_versions = FULL_PYTHON_VERSIONS + ["3.13", "3.13t"]

    if arches is None:
        # Define default compute archivectures
        arches = ["cpu"]
        if os == "linux":
            arches += CPU_CXX11_ABI_ARCH + CUDA_ARCHES + ROCM_ARCHES + XPU_ARCHES
        elif os == "windows":
            arches += CUDA_ARCHES + XPU_ARCHES
        elif os == "linux-aarch64":
            # Only want the one arch as the CPU type is different and
            # uses different build/test scripts
            arches = ["cpu-aarch64", "cuda-aarch64"]
        elif os == "linux-s390x":
            # Only want the one arch as the CPU type is different and
            # uses different build/test scripts
            arches = ["cpu-s390x"]

    ret: List[Dict[str, str]] = []
    for python_version in python_versions:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = (
                ""
                if arch_version == "cpu"
                or arch_version == "cpu-cxx11-abi"
                or arch_version == "cpu-aarch64"
                or arch_version == "cpu-s390x"
                or arch_version == "cuda-aarch64"
                or arch_version == "xpu"
                else arch_version
            )

            # TODO: Enable python 3.13 on rocm, aarch64, windows
            if (
                gpu_arch_type == "rocm"
                or os
                not in [
                    "linux",
                    "linux-s390x",
                    "linux-aarch64",
                    "macos-arm64",
                    "windows",
                ]
            ) and python_version in ["3.13", "3.13t"]:
                continue

            # TODO: Enable python 3.13t on xpu and cpu-s390x or MacOS or Windows
            if (
                gpu_arch_type in ["xpu", "cpu-s390x"]
                or os == "macos-arm64"
                or os == "linux-aarch64"
                or os == "windows"
            ) and python_version == "3.13t":
                continue

            if use_split_build and (
                arch_version not in ["12.6", "12.4", "11.8", "cpu"] or os != "linux"
            ):
                raise RuntimeError(
                    "Split build is only supported on linux with cuda 12.6, 12.4, 11.8, and cpu.\n"
                    f"Currently attempting to build on arch version {arch_version} and os {os}.\n"
                    "Please modify the matrix generation to exclude this combination."
                )

            # cuda linux wheels require PYTORCH_EXTRA_INSTALL_REQUIREMENTS to install

            if (
                arch_version in ["12.6", "12.4", "11.8"]
                and os == "linux"
                or arch_version == "cuda-aarch64"
            ):
                ret.append(
                    {
                        "python_version": python_version,
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": translate_desired_cuda(
                            gpu_arch_type, gpu_arch_version
                        ),
                        "use_split_build": "True" if use_split_build else "False",
                        "devtoolset": (
                            "cxx11-abi" if arch_version == "cuda-aarch64" else ""
                        ),
                        "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
                        "package_type": package_type,
                        "pytorch_extra_install_requirements": (
                            PYTORCH_EXTRA_INSTALL_REQUIREMENTS[arch_version]
                            if os != "linux-aarch64"
                            else ""
                        ),
                        "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(  # noqa: B950
                            ".", "_"
                        ),
                    }
                )
                # Special build building to use on Colab. Python 3.11 for 12.4 CUDA
                if python_version == "3.11" and arch_version == "12.4":
                    ret.append(
                        {
                            "python_version": python_version,
                            "gpu_arch_type": gpu_arch_type,
                            "gpu_arch_version": gpu_arch_version,
                            "desired_cuda": translate_desired_cuda(
                                gpu_arch_type, gpu_arch_version
                            ),
                            "use_split_build": "True" if use_split_build else "False",
                            "devtoolset": "",
                            "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
                            "package_type": package_type,
                            "pytorch_extra_install_requirements": "",
                            "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}-full".replace(  # noqa: B950
                                ".", "_"
                            ),
                        }
                    )
            else:
                ret.append(
                    {
                        "python_version": python_version,
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": translate_desired_cuda(
                            gpu_arch_type, gpu_arch_version
                        ),
                        "use_split_build": "True" if use_split_build else "False",
                        "devtoolset": (
                            "cxx11-abi"
                            if arch_version in ["cpu-cxx11-abi", "cpu-aarch64"]
                            else ""
                        ),
                        "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
                        "package_type": package_type,
                        "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                            ".", "_"
                        ),
                        "pytorch_extra_install_requirements": (
                            PYTORCH_EXTRA_INSTALL_REQUIREMENTS["xpu"]
                            if gpu_arch_type == "xpu"
                            else PYTORCH_EXTRA_INSTALL_REQUIREMENTS["12.4"]
                            if os != "linux"
                            else ""
                        ),
                    }
                )

    return ret


validate_nccl_dep_consistency("12.6")
validate_nccl_dep_consistency("12.4")
validate_nccl_dep_consistency("11.8")
