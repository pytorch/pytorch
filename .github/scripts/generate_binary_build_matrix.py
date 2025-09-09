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
from typing import Optional


# NOTE: Please also update the CUDA sources in `PIP_SOURCES` in tools/nightly.py when changing this
CUDA_ARCHES = ["12.6", "12.8", "13.0"]
CUDA_STABLE = "12.8"
CUDA_ARCHES_FULL_VERSION = {
    "12.6": "12.6.3",
    "12.8": "12.8.1",
    "13.0": "13.0.0",
}
CUDA_ARCHES_CUDNN_VERSION = {
    "12.6": "9",
    "12.8": "9",
    "13.0": "9",
}

# NOTE: Please also update the ROCm sources in `PIP_SOURCES` in tools/nightly.py when changing this
ROCM_ARCHES = ["6.3", "6.4"]

XPU_ARCHES = ["xpu"]

CPU_AARCH64_ARCH = ["cpu-aarch64"]

CPU_S390X_ARCH = ["cpu-s390x"]

CUDA_AARCH64_ARCHES = ["12.6-aarch64", "12.8-aarch64", "13.0-aarch64"]


PYTORCH_EXTRA_INSTALL_REQUIREMENTS = {
    "12.6": (
        "nvidia-cuda-nvrtc-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-runtime-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-cupti-cu12==12.6.80; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cublas-cu12==12.6.4.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufft-cu12==11.3.0.4; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-curand-cu12==10.3.7.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusolver-cu12==11.7.1.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparse-cu12==12.5.4.2; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvshmem-cu12==3.3.20; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvtx-cu12==12.6.77; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvjitlink-cu12==12.6.85; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufile-cu12==1.11.1.6; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
    "12.8": (
        "nvidia-cuda-nvrtc-cu12==12.8.93; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-runtime-cu12==12.8.90; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-cupti-cu12==12.8.90; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cudnn-cu12==9.10.2.21; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cublas-cu12==12.8.4.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufft-cu12==11.3.3.83; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-curand-cu12==10.3.9.90; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusolver-cu12==11.7.3.90; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparse-cu12==12.5.8.93; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparselt-cu12==0.7.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nccl-cu12==2.27.5; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvshmem-cu12==3.3.20; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvtx-cu12==12.8.90; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvjitlink-cu12==12.8.93; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufile-cu12==1.13.1.3; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
    "13.0": (
        "nvidia-cuda-nvrtc==13.0.48; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-runtime==13.0.48; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cuda-cupti==13.0.48; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cudnn-cu13==9.13.0.50; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cublas==13.0.0.19; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufft==12.0.0.15; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-curand==10.4.0.35; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusolver==12.0.3.29; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparse==12.6.2.49; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cusparselt-cu13==0.8.0; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nccl-cu13==2.27.7; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvshmem-cu13==3.3.24; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvtx==13.0.39; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-nvjitlink==13.0.39; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "nvidia-cufile==1.15.0.42; platform_system == 'Linux' and platform_machine == 'x86_64'"
    ),
    "xpu": (
        "intel-cmplr-lib-rt==2025.2.1 | "
        "intel-cmplr-lib-ur==2025.2.1 | "
        "intel-cmplr-lic-rt==2025.2.1 | "
        "intel-sycl-rt==2025.2.1 | "
        "oneccl-devel==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "oneccl==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "impi-rt==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
        "onemkl-sycl-blas==2025.2.0 | "
        "onemkl-sycl-dft==2025.2.0 | "
        "onemkl-sycl-lapack==2025.2.0 | "
        "onemkl-sycl-rng==2025.2.0 | "
        "onemkl-sycl-sparse==2025.2.0 | "
        "dpcpp-cpp-rt==2025.2.1 | "
        "intel-opencl-rt==2025.2.1 | "
        "mkl==2025.2.0 | "
        "intel-openmp==2025.2.1 | "
        "tbb==2022.2.0 | "
        "tcmlib==1.4.0 | "
        "umf==0.11.0 | "
        "intel-pti==0.13.1"
    ),
}


def get_nccl_wheel_version(arch_version: str) -> str:
    import re

    requirements = map(
        str.strip, re.split("[;|]", PYTORCH_EXTRA_INSTALL_REQUIREMENTS[arch_version])
    )
    return next(x for x in requirements if x.startswith("nvidia-nccl")).split("==")[1]


def read_nccl_pin(arch_version: str) -> str:
    from pathlib import Path

    nccl_pin_path = os.path.join(
        Path(__file__).absolute().parents[2],
        ".ci",
        "docker",
        "ci_commit_pins",
        f"nccl-cu{arch_version[:2]}.txt",
    )
    with open(nccl_pin_path) as f:
        return f.read().strip()


def validate_nccl_dep_consistency(arch_version: str) -> None:
    nccl_release_tag = read_nccl_pin(arch_version)
    wheel_ver = get_nccl_wheel_version(arch_version)
    if not nccl_release_tag.startswith(f"v{wheel_ver}"):
        raise RuntimeError(
            f"{arch_version} NCCL release tag version {nccl_release_tag} does not correspond to wheel version {wheel_ver}"
        )


def arch_type(arch_version: str) -> str:
    if arch_version in CUDA_ARCHES:
        return "cuda"
    elif arch_version in ROCM_ARCHES:
        return "rocm"
    elif arch_version in XPU_ARCHES:
        return "xpu"
    elif arch_version in CPU_AARCH64_ARCH:
        return "cpu-aarch64"
    elif arch_version in CPU_S390X_ARCH:
        return "cpu-s390x"
    elif arch_version in CUDA_AARCH64_ARCHES:
        return "cuda-aarch64"
    else:  # arch_version should always be "cpu" in this case
        return "cpu"


DEFAULT_TAG = os.getenv("RELEASE_VERSION_TAG", "main")

WHEEL_CONTAINER_IMAGES = {
    **{gpu_arch: f"manylinux2_28-builder:cuda{gpu_arch}" for gpu_arch in CUDA_ARCHES},
    **{
        gpu_arch: f"manylinuxaarch64-builder:cuda{gpu_arch.replace('-aarch64', '')}"
        for gpu_arch in CUDA_AARCH64_ARCHES
    },
    **{gpu_arch: f"manylinux2_28-builder:rocm{gpu_arch}" for gpu_arch in ROCM_ARCHES},
    "xpu": "manylinux2_28-builder:xpu",
    "cpu": "manylinux2_28-builder:cpu",
    "cpu-aarch64": "manylinux2_28_aarch64-builder:cpu-aarch64",
    "cpu-s390x": "pytorch/manylinuxs390x-builder:cpu-s390x",
}

RELEASE = "release"
DEBUG = "debug"

LIBTORCH_CONTAINER_IMAGES: dict[str, str] = {
    **{gpu_arch: f"libtorch-cxx11-builder:cuda{gpu_arch}" for gpu_arch in CUDA_ARCHES},
    **{gpu_arch: f"libtorch-cxx11-builder:rocm{gpu_arch}" for gpu_arch in ROCM_ARCHES},
    "cpu": "libtorch-cxx11-builder:cpu",
}

FULL_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.13t", "3.14", "3.14t"]


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        "cpu": "cpu",
        "cpu-aarch64": "cpu",
        "cpu-s390x": "cpu",
        "cuda": f"cu{gpu_arch_version.replace('.', '')}",
        "cuda-aarch64": f"cu{gpu_arch_version.replace('-aarch64', '').replace('.', '')}",
        "rocm": f"rocm{gpu_arch_version}",
        "xpu": "xpu",
    }.get(gpu_arch_type, gpu_arch_version)


def list_without(in_list: list[str], without: list[str]) -> list[str]:
    return [item for item in in_list if item not in without]


def generate_libtorch_matrix(
    os: str,
    release_type: str,
    arches: Optional[list[str]] = None,
    libtorch_variants: Optional[list[str]] = None,
) -> list[dict[str, str]]:
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

    ret: list[dict[str, str]] = []
    for arch_version in arches:
        for libtorch_variant in libtorch_variants:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            # ROCm builds without-deps failed even in ROCm runners; skip for now
            if gpu_arch_type == "rocm" and ("without-deps" in libtorch_variant):
                continue
            ret.append(
                {
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "libtorch_config": release_type,
                    "libtorch_variant": libtorch_variant,
                    "container_image": (
                        LIBTORCH_CONTAINER_IMAGES[arch_version].split(":")[0]
                        if os not in ("windows", "windows-arm64")
                        else ""
                    ),
                    "container_image_tag_prefix": (
                        LIBTORCH_CONTAINER_IMAGES[arch_version].split(":")[1]
                        if os not in ("windows", "windows-arm64")
                        else ""
                    ),
                    "package_type": "libtorch",
                    "build_name": f"libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{release_type}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_wheels_matrix(
    os: str,
    arches: Optional[list[str]] = None,
    python_versions: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    package_type = "wheel"
    if os == "linux" or os == "linux-aarch64" or os == "linux-s390x":
        # NOTE: We only build manywheel packages for x86_64 and aarch64 and s390x linux
        package_type = "manywheel"

    if python_versions is None:
        python_versions = FULL_PYTHON_VERSIONS

    if arches is None:
        # Define default compute archivectures
        arches = ["cpu"]
        if os == "linux":
            arches += CUDA_ARCHES + ROCM_ARCHES + XPU_ARCHES
        elif os == "windows":
            arches += CUDA_ARCHES + XPU_ARCHES
        elif os == "linux-aarch64":
            # Separate new if as the CPU type is different and
            # uses different build/test scripts
            arches = CPU_AARCH64_ARCH + CUDA_AARCH64_ARCHES
        elif os == "linux-s390x":
            # Only want the one arch as the CPU type is different and
            # uses different build/test scripts
            arches = ["cpu-s390x"]

    ret: list[dict[str, str]] = []
    for python_version in python_versions:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = (
                ""
                if arch_version == "cpu"
                or arch_version == "cpu-aarch64"
                or arch_version == "cpu-s390x"
                or arch_version == "xpu"
                else arch_version
            )

            # TODO: Enable python 3.14 for rest
            if os not in [
                "linux",
                "linux-aarch64",
                "linux-s390x",
                "macos-arm64",
                "windows",
            ] and (python_version == "3.14" or python_version == "3.14t"):
                continue

            # cuda linux wheels require PYTORCH_EXTRA_INSTALL_REQUIREMENTS to install

            if (
                arch_version in ["13.0", "12.8", "12.6"]
                and os == "linux"
                or arch_version in CUDA_AARCH64_ARCHES
            ):
                desired_cuda = translate_desired_cuda(gpu_arch_type, gpu_arch_version)
                ret.append(
                    {
                        "python_version": python_version,
                        "gpu_arch_type": gpu_arch_type,
                        "gpu_arch_version": gpu_arch_version,
                        "desired_cuda": desired_cuda,
                        "container_image": WHEEL_CONTAINER_IMAGES[arch_version].split(
                            ":"
                        )[0],
                        "container_image_tag_prefix": WHEEL_CONTAINER_IMAGES[
                            arch_version
                        ].split(":")[1],
                        "package_type": package_type,
                        "pytorch_extra_install_requirements": (
                            PYTORCH_EXTRA_INSTALL_REQUIREMENTS[
                                f"{desired_cuda[2:4]}.{desired_cuda[4:]}"  # for cuda-aarch64: cu126 -> 12.6
                            ]
                            if os == "linux-aarch64"
                            else PYTORCH_EXTRA_INSTALL_REQUIREMENTS[arch_version]
                        ),
                        "build_name": (
                            f"{package_type}-py{python_version}-{gpu_arch_type}"
                            f"{'-' if 'aarch64' in gpu_arch_type else ''}{gpu_arch_version.replace('-aarch64', '')}".replace(
                                ".", "_"
                            )
                        ),  # include special case for aarch64 build, remove the -aarch64 postfix
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
                        "container_image": WHEEL_CONTAINER_IMAGES[arch_version].split(
                            ":"
                        )[0],
                        "container_image_tag_prefix": WHEEL_CONTAINER_IMAGES[
                            arch_version
                        ].split(":")[1],
                        "package_type": package_type,
                        "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                            ".", "_"
                        ),
                        "pytorch_extra_install_requirements": (
                            PYTORCH_EXTRA_INSTALL_REQUIREMENTS["xpu"]
                            if gpu_arch_type == "xpu"
                            else ""
                        ),
                    }
                )

    return ret


validate_nccl_dep_consistency("13.0")
validate_nccl_dep_consistency("12.8")
validate_nccl_dep_consistency("12.6")
