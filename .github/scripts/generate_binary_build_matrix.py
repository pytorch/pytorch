#!/usr/bin/env python3

"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
"""

from typing import Dict, List, Tuple


CUDA_ARCHES = ["10.2", "11.1", "11.3", "11.5"]


ROCM_ARCHES = ["4.3.1", "4.5.2"]


def arch_type(arch_version: str) -> str:
    if arch_version in CUDA_ARCHES:
        return "cuda"
    elif arch_version in ROCM_ARCHES:
        return "rocm"
    else:  # arch_version should always be "cpu" in this case
        return "cpu"


WHEEL_CONTAINER_IMAGES = {
    **{
        gpu_arch: f"pytorch/manylinux-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        gpu_arch: f"pytorch/manylinux-builder:rocm{gpu_arch}"
        for gpu_arch in ROCM_ARCHES
    },
    "cpu": "pytorch/manylinux-builder:cpu",
}

CONDA_CONTAINER_IMAGES = {
    **{gpu_arch: f"pytorch/conda-builder:cuda{gpu_arch}" for gpu_arch in CUDA_ARCHES},
    "cpu": "pytorch/conda-builder:cpu",
}

PRE_CXX11_ABI = "pre-cxx11"
CXX11_ABI = "cxx11-abi"

LIBTORCH_CONTAINER_IMAGES: Dict[Tuple[str, str], str] = {
    **{
        (gpu_arch, PRE_CXX11_ABI): f"pytorch/manylinux-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    **{
        (gpu_arch, CXX11_ABI): f"pytorch/libtorch-cxx11-builder:cuda{gpu_arch}"
        for gpu_arch in CUDA_ARCHES
    },
    ("cpu", PRE_CXX11_ABI): "pytorch/manylinux-builder:cpu",
    ("cpu", CXX11_ABI): "pytorch/libtorch-cxx11-builder:cpu",
}

FULL_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10"]


def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    return {
        "cpu": "cpu",
        "cuda": f"cu{gpu_arch_version.replace('.', '')}",
        "rocm": f"rocm{gpu_arch_version}",
    }.get(gpu_arch_type, gpu_arch_version)


def list_without(in_list: List[str], without: List[str]) -> List[str]:
    return [item for item in in_list if item not in without]


def generate_conda_matrix(os: str) -> List[Dict[str, str]]:
    ret: List[Dict[str, str]] = []
    arches = ["cpu"]
    if os == "linux":
        arches += CUDA_ARCHES
    elif os == "windows":
        # We don't build CUDA 10.2 for window
        arches += list_without(CUDA_ARCHES, ["10.2"])
    for python_version in FULL_PYTHON_VERSIONS:
        # We don't currently build conda packages for rocm
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            ret.append(
                {
                    "python_version": python_version,
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "container_image": CONDA_CONTAINER_IMAGES[arch_version],
                    "package_type": "conda",
                    "build_name": f"conda-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_libtorch_matrix(os: str, abi_version: str) -> List[Dict[str, str]]:
    libtorch_variants = [
        "shared-with-deps",
        "shared-without-deps",
        "static-with-deps",
        "static-without-deps",
    ]
    ret: List[Dict[str, str]] = []
    arches = ["cpu"]
    if os == "linux":
        arches += CUDA_ARCHES
    elif os == "windows":
        # We don't build CUDA 10.2 for window
        arches += list_without(CUDA_ARCHES, ["10.2"])
    for arch_version in ["cpu"] + CUDA_ARCHES:
        for libtorch_variant in libtorch_variants:
            # We don't currently build libtorch for rocm
            # one of the values in the following list must be exactly
            # CXX11_ABI, but the precise value of the other one doesn't
            # matter
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            ret.append(
                {
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "libtorch_variant": libtorch_variant,
                    "devtoolset": abi_version,
                    "container_image": LIBTORCH_CONTAINER_IMAGES[
                        (arch_version, abi_version)
                    ],
                    "package_type": "libtorch",
                    "build_name": f"libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{abi_version}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_wheels_matrix(os: str) -> List[Dict[str, str]]:
    arches = ["cpu"]
    package_type = "wheel"
    if os == "linux":
        arches += CUDA_ARCHES + ROCM_ARCHES
        # NOTE: We only build manywheel packages for linux
        package_type = "manywheel"
    elif os == "windows":
        # We don't build CUDA 10.2 for window
        arches += list_without(CUDA_ARCHES, ["10.2"])
    ret: List[Dict[str, str]] = []
    for python_version in FULL_PYTHON_VERSIONS:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = "" if arch_version == "cpu" else arch_version
            ret.append(
                {
                    "python_version": python_version,
                    "gpu_arch_type": gpu_arch_type,
                    "gpu_arch_version": gpu_arch_version,
                    "desired_cuda": translate_desired_cuda(
                        gpu_arch_type, gpu_arch_version
                    ),
                    "container_image": WHEEL_CONTAINER_IMAGES[arch_version],
                    "package_type": package_type,
                    "build_name": f"{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}".replace(
                        ".", "_"
                    ),
                }
            )
    return ret


def generate_binary_build_matrix(os: str) -> List[Dict[str, str]]:
    return {
        "linux": [
            *generate_conda_matrix(os),
            *generate_libtorch_matrix(os, abi_version=PRE_CXX11_ABI),
            *generate_libtorch_matrix(os, abi_version=CXX11_ABI),
            *generate_wheels_matrix(os),
        ]
    }[os]
